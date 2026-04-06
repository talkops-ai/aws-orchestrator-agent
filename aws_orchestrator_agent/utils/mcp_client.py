"""
MCP (Model Context Protocol) Client for AWS Orchestrator Agent.

Manages sessions to one-or-many MCP servers, exposing their tools, resources,
and prompts as LangChain-compatible objects.

Design principles:
    • **No global monkey-patching** — timeout config is passed per-client.
    • **Single session lifecycle** — ``async with client.connect(): ...``
    • **Tool error resilience** — every wrapped tool has ``handle_tool_error=True``.
    • **Clean shutdown** — ``connect()`` guarantees ``close()`` via context manager.

Usage::

    from aws_orchestrator_agent.utils.mcp_client import MCPClient

    client = MCPClient(config)
    async with client.connect():
        tools = client.get_tools()               # List[BaseTool]
        resources = await client.list_resources() # per-server
        messages  = await client.get_prompt("generate-vpc", server="github_mcp")
"""


import os
import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, AsyncExitStack
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from aws_orchestrator_agent.utils.logger import AgentLogger
from aws_orchestrator_agent.utils.exceptions import MCPClientError

if TYPE_CHECKING:
    from aws_orchestrator_agent.config import Config

logger = AgentLogger("MCPClient")

# Default tool execution timeout (seconds).  Overridable via Config.
_DEFAULT_TOOL_TIMEOUT: float = 300.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_auth_headers(server_def: dict[str, Any]) -> dict[str, str]:
    """Build HTTP headers including Bearer token from env-var if configured."""
    headers: dict[str, str] = dict(server_def.get("headers") or {})
    env_var: str | None = server_def.get("auth_token_env_var")
    if env_var:
        token = os.getenv(env_var)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            logger.warning(
                "Auth token env var not set",
                extra={"env_var": env_var, "server": server_def.get("name")},
            )
    return headers


def _build_server_configs(
    config: "Config",
    server_filter: list[str] | None,
) -> dict[str, dict[str, Any]]:
    """
    Transform ``Config.mcp_config`` into the dict format expected by
    ``MultiServerMCPClient``.

    Returns ``{server_name: connection_kwargs}``.
    """
    mcp = config.get_mcp_config()
    default_transport: str = mcp.get("default_transport", "sse")
    servers: dict[str, dict[str, Any]] = {}

    for sdef in mcp.get("servers", []):
        name = sdef.get("name")
        if not name:
            logger.warning("Skipping MCP server entry without a 'name' key")
            continue
        if server_filter and name not in server_filter:
            continue
        if sdef.get("disabled", False):
            logger.debug("Skipping disabled server", extra={"server": name})
            continue

        transport = sdef.get("transport", default_transport)

        if transport in ("sse", "http"):
            url = sdef.get("url")
            if not url:
                logger.warning(f"{transport} transport requires 'url'", extra={"server": name})
                continue
            entry: dict[str, Any] = {"url": url, "transport": transport}
            headers = _resolve_auth_headers(sdef)
            if headers:
                entry["headers"] = headers
            servers[name] = entry

        elif transport == "stdio":
            servers[name] = {
                "command": sdef.get("command", "python"),
                "args": sdef.get("args", []),
                "transport": "stdio",
            }
        else:
            logger.warning("Unknown transport", extra={"server": name, "transport": transport})

    return servers


def _wrap_tool(
    original: BaseTool,
    server_name: str,
    execute_fn: Callable[..., Any],
) -> BaseTool:
    """
    Wrap an MCP tool so it routes execution through ``MCPClient.execute_tool``.

    Preserves the original schema and description, and sets
    ``handle_tool_error=True`` so LangGraph won't crash on failures.
    """
    tool_name = original.name

    async def _proxy(**kwargs: Any) -> Any:
        return await execute_fn(server_name, tool_name, kwargs)

    tool_kwargs: Dict[str, Any] = {
        "name": tool_name,
        "description": original.description or "",
        "coroutine": _proxy,
        "func": lambda **_: None,  # sync stub required by StructuredTool
        "handle_tool_error": True,  # prevents ToolException from crashing the graph
    }
    if original.args_schema is not None:
        tool_kwargs["args_schema"] = original.args_schema
    wrapped = StructuredTool(**tool_kwargs)  # type: ignore[arg-type]
    return wrapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MCPClient:
    """
    Async MCP client that manages connections to one or more MCP servers.

    Lifecycle::

        client = MCPClient(config)
        async with client.connect():
            tools = client.get_tools()
            # ... use tools in your LangGraph agent ...
        # all sessions are closed here
    """

    __slots__ = (
        "_config",
        "_server_filter",
        "_server_configs",
        "_exit_stack",
        "_client",
        "_sessions",
        "_tools",
        "_tool_map",
        "_tool_timeout",
    )

    def __init__(
        self,
        config: Optional["Config"] = None,
        server_filter: Optional[List[str]] = None,
    ) -> None:
        if config is None:
            from aws_orchestrator_agent.config import Config
            config = Config()

        self._config = config
        self._server_filter = server_filter
        self._server_configs = _build_server_configs(config, server_filter)
        self._tool_timeout: float = float(
            config.get("MCP_TIMEOUT_TOTAL", _DEFAULT_TOOL_TIMEOUT)
        )

        # Session state (populated in connect())
        self._exit_stack: Optional[AsyncExitStack] = None
        self._client: Optional[MultiServerMCPClient] = None
        self._sessions: dict[str, Any] = {}
        self._tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}

        logger.info(
            "MCPClient created",
            extra={"servers": list(self._server_configs.keys())},
        )

    # ── Connection lifecycle ─────────────────────────────────────────────

    @asynccontextmanager
    async def connect(self) -> AsyncIterator["MCPClient"]:
        """
        Open persistent sessions to all configured MCP servers.

        Usage::

            async with client.connect():
                tools = client.get_tools()

        On exit the context manager closes every session and resets state.
        """
        if not self._server_configs:
            logger.warning("No MCP servers configured — nothing to connect to")
            yield self
            return

        self._exit_stack = AsyncExitStack()
        try:
            self._client = MultiServerMCPClient(self._server_configs)  # type: ignore[arg-type]
            all_tools: list[BaseTool] = []

            for name in self._server_configs:
                session = await self._exit_stack.enter_async_context(
                    self._client.session(name)
                )
                self._sessions[name] = session

                server_tools = await load_mcp_tools(session)
                logger.info(
                    "Connected to MCP server",
                    extra={"server": name, "tools": len(server_tools)},
                )
                for t in server_tools:
                    all_tools.append(_wrap_tool(t, name, self.execute_tool))

            self._tools = all_tools
            self._tool_map = {t.name: t for t in all_tools}
            logger.info("All MCP tools loaded", extra={"total": len(all_tools)})

            yield self

        finally:
            await self._close()

    async def _close(self) -> None:
        """Tear down all sessions and reset internal state."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._sessions.clear()
        self._client = None
        self._tools.clear()
        self._tool_map.clear()
        logger.info("MCPClient closed — all sessions terminated")

    # ── Tool access ──────────────────────────────────────────────────────

    def get_tools(self) -> List[BaseTool]:
        """Return all loaded LangChain tools (across all servers)."""
        return list(self._tools)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Look up a single tool by name. Returns ``None`` if not found."""
        return self._tool_map.get(name)

    # ── Tool execution ───────────────────────────────────────────────────

    async def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Execute a tool on the specified MCP server.

        Applies a timeout (from ``MCP_TIMEOUT_TOTAL`` config key) to prevent
        hanging on unresponsive servers.
        """
        session = self._get_session(server_name)

        logger.debug(
            "Executing MCP tool",
            extra={"server": server_name, "tool": tool_name},
        )

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=self._tool_timeout,
            )

            # Unwrap single-text results for convenience
            if (
                result.content
                and len(result.content) == 1
                and hasattr(result.content[0], "text")
            ):
                return result.content[0].text

            return result.content

        except asyncio.TimeoutError:
            logger.error(
                "Tool execution timed out",
                extra={
                    "server": server_name,
                    "tool": tool_name,
                    "timeout_s": self._tool_timeout,
                },
            )
            raise MCPClientError(
                f"Tool '{tool_name}' on '{server_name}' timed out "
                f"after {self._tool_timeout}s"
            )
        except Exception as exc:
            logger.error(
                "Tool execution failed",
                extra={"server": server_name, "tool": tool_name, "error": str(exc)},
            )
            raise MCPClientError(f"Tool execution failed: {exc}") from exc

    # ── Resources ────────────────────────────────────────────────────────

    async def list_resources(
        self,
        server_name: Optional[str] = None,
    ) -> List[Any]:
        """List available resources on a server (defaults to first connected)."""
        session, resolved = self._resolve_session(server_name)
        logger.debug("Listing MCP resources", extra={"server": resolved})
        try:
            result = await session.list_resources()
            return result.resources
        except Exception as exc:
            raise MCPClientError(f"Failed to list resources on '{resolved}': {exc}") from exc

    async def read_resource(
        self,
        uri: str,
        server_name: Optional[str] = None,
    ) -> Any:
        """Read a single resource by URI."""
        session, resolved = self._resolve_session(server_name)
        logger.debug("Reading MCP resource", extra={"server": resolved, "uri": uri})
        try:
            return await session.read_resource(uri)
        except Exception as exc:
            raise MCPClientError(f"Failed to read resource '{uri}': {exc}") from exc

    # ── Prompts ──────────────────────────────────────────────────────────

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        Fetch a prompt template and return it as LangChain messages.
        """
        session, resolved = self._resolve_session(server_name)
        logger.debug(
            "Fetching MCP prompt",
            extra={"server": resolved, "prompt": prompt_name},
        )
        try:
            result = await session.get_prompt(
                prompt_name, arguments=arguments or {}
            )
            return self._convert_messages(result.messages)
        except Exception as exc:
            raise MCPClientError(
                f"Failed to fetch prompt '{prompt_name}': {exc}"
            ) from exc

    # ── Private helpers ──────────────────────────────────────────────────

    def _get_session(self, server_name: str) -> Any:
        """Get a session by exact server name, or raise."""
        session = self._sessions.get(server_name)
        if session is None:
            raise MCPClientError(
                f"MCP server '{server_name}' not connected. "
                f"Available: {list(self._sessions.keys())}"
            )
        return session

    def _resolve_session(
        self, server_name: str | None
    ) -> tuple[Any, str]:
        """
        Resolve a session — explicit name or fallback to first connected.

        Returns ``(session, resolved_server_name)``.
        """
        if server_name:
            return self._get_session(server_name), server_name
        if not self._sessions:
            raise MCPClientError("No MCP servers connected")
        name = next(iter(self._sessions))
        return self._sessions[name], name

    @staticmethod
    def _convert_messages(raw_messages: list[Any]) -> list[BaseMessage]:
        """Convert MCP prompt messages → LangChain BaseMessage list."""
        result: list[BaseMessage] = []
        for msg in raw_messages:
            # Extract text content
            if hasattr(msg.content, "text"):
                text = msg.content.text
            elif isinstance(msg.content, list):
                text = "\n".join(
                    c.text for c in msg.content if hasattr(c, "text")
                )
            else:
                text = str(msg.content)

            # Map role
            if msg.role == "user":
                result.append(HumanMessage(content=text))
            elif msg.role == "assistant":
                result.append(AIMessage(content=text))
            else:
                result.append(SystemMessage(content=text))
        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def create_mcp_client(
    config: Optional["Config"] = None,
    server_filter: Optional[List[str]] = None,
) -> AsyncIterator[MCPClient]:
    """
    One-liner factory::

        async with create_mcp_client(config) as client:
            tools = client.get_tools()
    """
    client = MCPClient(config, server_filter)
    async with client.connect():
        yield client
