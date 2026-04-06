from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from aws_orchestrator_agent.config import Config


# ---------------------------------------------------------------------------
# Agent Response (unchanged)
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """
    Response from an agent during execution.

    This represents a single response item from the agent's stream,
    containing the content, metadata, and control flags.
    """

    model_config = ConfigDict(extra="allow")

    content: Any = Field(..., description="The response content (text or data)")
    response_type: str = Field(
        default="text",
        description=(
            "Type of response: 'token' (LLM token stream), 'text' (complete message), "
            "'data' (structured data), 'interrupt', or 'human_input'"
        ),
    )
    is_task_complete: bool = Field(
        default=False, description="Whether this response indicates task completion"
    )
    require_user_input: bool = Field(
        default=False,
        description="Whether this response requires user input to continue",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )
    root: Optional[Any] = Field(
        default=None, description="Root object for A2A protocol integration"
    )


# ---------------------------------------------------------------------------
# BaseAgent — A2A protocol integration contract
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Base interface for all AWS Orchestrator Agent implementations.

    This abstract base class defines the contract that all agent implementations
    must follow to work with the A2A protocol integration.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the agent."""
        ...

    @abstractmethod
    async def stream(
        self,
        query: Union[str, Command],
        context_id: str,
        task_id: str,
        use_ui: bool = False,
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Stream responses for a given query.

        When the agent encounters a HITL action, it should yield an
        ``AgentResponse`` with ``require_user_input=True`` and
        ``response_type="interrupt"``.
        """
        ...

    async def initialize(self) -> None:
        """Optional async initialisation hook."""

    async def cleanup(self) -> None:
        """Optional async teardown hook."""


# ---------------------------------------------------------------------------
# BaseSubgraphAgent — Send() / Command-based subgraph agent
# ---------------------------------------------------------------------------

class BaseSubgraphAgent(ABC):
    """
    Base abstract class for subgraph agents using LangGraph Send() pattern.

    All specialized agents (Generation, Editor, Validation, etc.) must inherit
    from this class and implement the required abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for Send() routing and identification."""
        ...

    @property
    @abstractmethod
    def state_model(self) -> type[BaseModel]:
        """Pydantic model for agent's state schema."""
        ...

    @property
    def memory(self) -> Any:
        """Memory/checkpointer instance for this agent."""
        return None

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for this agent."""
        ...

    @abstractmethod
    def input_transform(self, send_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Send() payload from supervisor to agent state."""
        ...

    @abstractmethod
    def output_transform(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform agent state back to supervisor state."""
        ...

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tools available to this agent."""
        return []


# ---------------------------------------------------------------------------
# SubAgent — lightweight sub-agent for supervisor delegation
# ---------------------------------------------------------------------------

class SubAgent(ABC):
    """
    Base abstract class for sub-agents (Context Detector, Policy Validator, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for routing."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description — used as prompt template."""
        ...

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Get tools available to this agent."""
        ...

    @abstractmethod
    def build_agent(self) -> CompiledStateGraph:
        """Build the compiled LangGraph agent."""
        ...


# ---------------------------------------------------------------------------
# BaseDeepAgent — abstract coordinator for Deep Agent pattern
# ---------------------------------------------------------------------------

class BaseDeepAgent(ABC):
    """
    Abstract base class for deep agent coordinators.

    Wraps LangChain's ``create_deep_agent`` API with a consistent lifecycle:

        build_agent() → seed_files() → agent.invoke(files=seed, ...) / .astream(...)

    Subclasses define:
        - **What** the agent does  → ``system_prompt``, ``get_subagent_specs()``
        - **How** it runs          → ``get_model()``, ``get_tools()``, ``get_interrupt_config()``
        - **Where** files live     → ``get_skill_paths()``, ``get_memory_paths()``
        - **Where** data persists  → ``make_backend()``, ``build_store()``, ``build_checkpointer()``
        - **What** config looks like → ``context_schema``

    Concept mapping from LangChain Deep Agents docs:
        - **Skills** — file-based SKILL.md directories auto-loaded into sub-agent context
        - **Memory** — persistent markdown files read at session start (e.g. AGENTS.md)
        - **Backend** — ``CompositeBackend`` routing paths to different storage backends
        - **Store** — ``InMemoryStore`` / ``PostgresStore`` for cross-thread long-term memory
        - **Checkpointer** — ``MemorySaver`` for thread-scoped state snapshots (HITL resume)
        - **Subagents** — dict specs with name/description/system_prompt/tools/skills/model
        - **Context Schema** — TypedDict injected via ``config["context"]``
    """

    def __init__(self, config: Optional["Config"] = None) -> None:
        if config is None:
            from aws_orchestrator_agent.config import Config
            config = Config()
        self._config: "Config" = config

    @property
    def config(self) -> "Config":
        return self._config

    # ── Abstract — MUST override ──────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this deep agent coordinator."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        System prompt that defines the coordinator's role, workflows,
        sub-agent delegation rules, and memory conventions.
        """
        ...

    @property
    @abstractmethod
    def context_schema(self) -> type:
        """
        TypedDict class for runtime context injected via
        ``config["context"]`` in LangGraph invocation.
        """
        ...

    @abstractmethod
    def get_model(self) -> Any:
        """
        Return an initialized LLM model for the coordinator.

        Should use ``create_model(config.get_llm_*_config())`` to produce
        a fully configured ``BaseChatModel`` instance.
        """
        ...

    @abstractmethod
    async def get_subagent_specs(self) -> List[Dict[str, Any]]:
        """
        Return sub-agent specifications for ``create_deep_agent(subagents=...)``.

        Each spec is a dict with keys:
            name, description, system_prompt, tools, model, skills (optional)

        May be ``async`` because some sub-agents need MCP tools loaded first.
        """
        ...

    # ── Virtual — CAN override (sensible defaults) ────────────────────────

    async def get_tools(self) -> List[Any]:
        """
        Coordinator-level tools (available to the coordinator LLM directly).

        Override to add tools like ``delete_module``, GitHub MCP tools, etc.
        Default: no tools.
        """
        return []

    def get_skill_paths(self) -> List[str]:
        """
        Virtual filesystem paths for skill directories.

        Skills are auto-loaded into the agent's context and contain
        ``SKILL.md`` (YAML frontmatter + workflow) + ``references/``.
        Default: ``["/skills/"]``.
        """
        return ["/skills/"]

    def get_memory_paths(self) -> List[str]:
        """
        Virtual filesystem paths for memory files read at session start.

        E.g. ``["/memories/AGENTS.md", "/memories/org-standards.md"]``.
        Default: no memory files.
        """
        return []

    def get_interrupt_config(self) -> Dict[str, Any]:
        """
        HITL interrupt configuration for ``create_deep_agent(interrupt_on=...)``.

        Example::

            {
                "delete_module": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
            }

        Default: no interrupts.
        """
        return {}

    def input_transform(self, send_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Send() payload from deep agent to subgraph agent state.

        Override when mounting a ``BaseSubgraphAgent`` as a ``CompiledSubAgent``
        node, to bridge the coordinator's invocation context into the
        subgraph's state schema.

        Default: identity (pass-through).
        """
        return send_payload

    def output_transform(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform subgraph agent state back to deep agent state.

        Override when mounting a ``BaseSubgraphAgent`` as a ``CompiledSubAgent``
        node, to extract results from the subgraph's state schema back into
        the coordinator's state.

        Default: identity (pass-through).
        """
        return agent_state

    # ── Abstract — MUST override (implementation-specific) ────────────────

    @abstractmethod
    def make_backend(self, runtime: Any) -> Any:
        """
        Build a backend for the deep agent's virtual filesystem.

        Typically a ``CompositeBackend`` routing paths to different storage
        backends (e.g. ``/memories/`` → ``StoreBackend``, default → ``StateBackend``).
        """
        ...

    @abstractmethod
    def build_store(self) -> Any:
        """
        Build the LangGraph Store for cross-thread long-term memory.

        E.g. ``InMemoryStore()`` for development, ``PostgresStore`` for production.
        """
        ...

    @abstractmethod
    def build_checkpointer(self) -> Any:
        """
        Build the checkpointer for thread-scoped state persistence.

        E.g. ``MemorySaver()`` for in-memory, ``PostgresSaver`` for production.
        """
        ...

    @abstractmethod
    async def build_agent(self) -> CompiledStateGraph:
        """
        Assemble all components into a ``create_deep_agent()`` call.

        Wire model, prompt, tools, subagents, skills, memory, backend,
        store, checkpointer, HITL config, and context schema.

        Returns:
            A ``CompiledStateGraph`` ready for ``.invoke()`` / ``.astream()``.
        """
        ...

    @abstractmethod
    def seed_files(
        self,
        skills_dir: Optional[Any] = None,
        memory_dir: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load skill files and initial memory into the virtual FS dict.

        Call this to produce the ``files`` dict for::

            agent.invoke({"messages": [...], "files": seed_files()})
        """
        ...

    # ── Concrete — inherited as-is ────────────────────────────────────────

    def build_memory_components(
        self,
    ) -> tuple[Callable[..., Any], Any, Any]:
        """
        Return ``(backend_factory, store, checkpointer)`` for ``create_deep_agent``.

        The backend factory is a callable that receives a LangGraph ``runtime``
        object and returns a backend instance.
        """
        store = self.build_store()
        checkpointer = self.build_checkpointer()
        return self.make_backend, store, checkpointer
