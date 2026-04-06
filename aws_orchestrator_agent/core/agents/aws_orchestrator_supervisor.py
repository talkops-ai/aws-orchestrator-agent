"""
AWS Orchestrator Supervisor Agent.

Uses the CI-supervisor "tool-wrapper" pattern to delegate Terraform module
generation/update requests to the TFCoordinator deep agent.

Architecture::

    A2AExecutor.execute()
      → SupervisorAgent.stream(query, context_id, task_id)
          → compiled_graph.astream(
                input, config,
                stream_mode=["updates","messages"], subgraphs=True, version="v2",
            )
              → yields AgentResponse (working / input_required / completed)

Tool-wrapper delegation::

    SupervisorAgent (create_agent)
      → transfer_to_terraform @tool
          → coordinator.input_transform(state) → deep_agent.ainvoke() → coordinator.output_transform()
          → Command(update={terraform_output: ..., messages: [ToolMessage]})
"""

import json
import os
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Union, cast

from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from aws_orchestrator_agent.core.agents.types import AgentResponse, BaseAgent
from aws_orchestrator_agent.core.state.supervisor_state import (
    SupervisorState,
    SupervisorWorkflowState,
)
from aws_orchestrator_agent.utils import AgentLogger
from aws_orchestrator_agent.utils.llm import create_model

logger = AgentLogger("SupervisorAgent")

# Note: We now process streams dynamically by message type, so no hardcoded _INTERESTING_NODES are required.


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """\
You are a supervisor managing the Terraform Deep Agent for AWS infrastructure generation and updates.

**Capabilities:**
1. **Terraform Module Generation:** Create new Terraform modules (VPC, S3, EC2, RDS, IAM, Lambda, etc.) following AWS best practices.
2. **Terraform Module Update:** Modify existing Terraform modules on GitHub with targeted changes.

**VALID REQUEST EXAMPLES:**
- "Create a VPC module with public and private subnets" (Terraform Generation)
- "Generate a Terraform module for S3 with encryption" (Terraform Generation)
- "Update the VPC module to add NAT gateway" (Terraform Update)
- "Create infrastructure for a 3-tier web app" (Terraform Generation)
- "Add a CloudFront distribution to our static site module" (Terraform Update)
- "Generate Terraform for an EKS cluster" (Terraform Generation)

**OUT-OF-SCOPE REQUEST HANDLING:**
If a request is NOT related to Terraform module generation or updates:
1. **CRITICAL: You MUST use the request_human_input tool** — DO NOT output text directly
2. Create a friendly, contextual message that guides the user to Terraform module tasks
3. **NEVER output conversational text without calling request_human_input tool first**

**HUMAN INPUT TOOL USAGE:**
- request_human_input: Use this tool when you need to:
  * **MANDATORY for greetings** (hello, hi, how are you, etc.)
  * **MANDATORY for out-of-scope requests** — guide user to Terraform tasks
  * Clarify ambiguous requirements
  * Get approval for decisions

Available tools:
- transfer_to_terraform: Delegate Terraform module generation or update tasks to the deep agent. Pass a clear, intent-based task_description that you craft from the user's request.
- request_human_input: Request human feedback, clarification, or respond to greetings/out-of-scope requests

**TASK DESCRIPTION CRAFTING (CRITICAL for transfer_to_terraform):**
When calling `transfer_to_terraform`, you MUST craft a human-readable task_description from the user's intent. Parse the request, extract the intent, and create a clear professional description. Do NOT pass raw user messages verbatim.

**Workflow Logic:**

1. **For Terraform Requests** ("Create a module...", "Generate Terraform...", "Update the module...", etc.):
   - Parse the user's request and craft a clear task_description
   - DIRECTLY call `transfer_to_terraform(task_description=crafted_description)`
   - Do NOT ask for details yourself — the deep agent gathers requirements
   - Do NOT respond with "Sure, I can help" — just call the tool

2. **For Greetings or Out-of-Scope Requests**:
   - DIRECTLY call `request_human_input(question=your_contextual_message)`

**CRITICAL RULES:**
- Always check if the user wants Terraform generation/update or something else
- For Terraform requests, delegate to `transfer_to_terraform` immediately
- Do NOT try to create Terraform code yourself. Use the tools.
- **Always call tools immediately, don't describe what you will do**
- You are a ROUTER, not a CREATOR
""".strip()


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------

class SupervisorAgent(BaseAgent):
    """Wraps the TFCoordinator deep agent as a tool in a create_agent supervisor.

    Lifecycle::

        agent = SupervisorAgent(config)
        await agent.initialize()            # builds deep-agent graph, loads MCP
        async for resp in agent.stream(...): # streams AgentResponse
            ...
        await agent.cleanup()
    """

    def __init__(
        self,
        agents: Optional[List[Any]] = None,
        config: Optional[Any] = None,
        name: str = "aws_orchestrator",
    ) -> None:
        self._name = name
        self._config = config
        self._agents = agents or []
        
        # Extract the TFCoordinator from the injected agents list if present
        from aws_orchestrator_agent.core.agents.tf_operator.tf_cordinator import TFCoordinator
        coord = next((a for a in self._agents if isinstance(a, TFCoordinator)), None)
        
        from aws_orchestrator_agent.config import Config
        cfg = self._config if isinstance(self._config, Config) else Config(self._config or {})
        self.config_instance = cfg
        
        if coord is None:
            logger.info("No TFCoordinator injected via agents list, instantiating a new one")
            coord = TFCoordinator(config=cfg)
            
        self._coordinator = coord

        from aws_orchestrator_agent.core.checkpointer import get_checkpointer
        self.memory = get_checkpointer(
            config=cfg,
            prefer_postgres=True
        )

        # Build supervisor synchronously upon instantiation, same as ci-copilot
        self._graph = self._build_supervisor_graph(
            coordinator=self._coordinator,
            config=cfg,
            memory=self.memory,
        )

        logger.info(
            "Main SupervisorAgent initialized",
            extra={
                "name": self._name,
                "compiled_graph_type": type(self._graph).__name__,
                "has_coordinator": True,
            },
        )

    # ── BaseAgent interface ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name



    async def cleanup(self) -> None:
        """Release resources and drop graph reference."""
        self._coordinator = None
        self._graph = None
        logger.info("SupervisorAgent cleaned up")

    # ── Graph builder (CI-supervisor pattern) ──────────────────────────

    def _build_supervisor_graph(
        self,
        coordinator: Any,
        config: Any,
        memory: Any,
    ) -> CompiledStateGraph:
        """Build supervisor graph using ``create_agent()`` with tool wrappers.

        Mirrors the CI-supervisor pattern:
        - ``transfer_to_terraform`` wraps the deep agent invocation
        - ``request_human_input`` gates HITL
        - All tools return ``Command(update=...)`` for state merging
        """
        agent_tools = self._create_agent_tools(
            coordinator=coordinator,
        )

        # Initialize LLM for the supervisor (Config always has defaults from DefaultConfig)
        model = create_model(config.get_llm_config())

        all_tools = agent_tools + [self._make_request_human_input_tool()]

        logger.info(
            "Creating supervisor with create_agent()",
            extra={"tool_names": [t.name for t in all_tools]},
        )

        return create_agent(
            model=model,
            tools=all_tools,
            system_prompt=SUPERVISOR_PROMPT,
            state_schema=SupervisorState,
            checkpointer=memory,
        )

    def _create_agent_tools(
        self,
        coordinator: Any,
    ) -> List:
        """
        Create tool wrappers for the TFCoordinator deep agent.

        Pattern (matches ``ci-supervisor.py::_create_agent_tools``):
        - Wrap the deep agent graph invocation in a @tool
        - Tool transforms SupervisorState → deep-agent input via coordinator.input_transform
        - Tool invokes deep_agent.ainvoke() (lazily initializing deep_agent the first time)
        - Tool transforms deep-agent output → SupervisorState via coordinator.output_transform
        - Tool returns Command(update=...) to merge results back
        """
        tools: List[Any] = []

        @tool
        async def transfer_to_terraform(
            task_description: str,
            runtime: ToolRuntime[None, SupervisorState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            config: RunnableConfig,
        ) -> Command:
            """
            Delegate Terraform module generation or update to the deep agent.

            task_description: A clear, intent-based summary you craft from the user's request.
            Parse intent, extract what they need, and pass a clear description.
            Do NOT pass the raw user message verbatim.
            """
            logger.info(
                "Delegating to TFCoordinator deep agent",
                extra={
                    "tool_name": "transfer_to_terraform",
                    "task_preview": task_description[:200],
                    "session_id": runtime.state.get("session_id"),
                    "task_id": runtime.state.get("task_id"),
                },
            )

            # Lazy initialize the deep agent graph
            if not getattr(coordinator, "_is_initialized", False):
                logger.info("Building TFCoordinator deep agent graph lazily")
                coordinator._deep_agent_graph = await coordinator.build_agent()
                coordinator._is_initialized = True
            
            deep_agent_graph = coordinator._deep_agent_graph

            # Build input: supervisor state → deep agent input
            send_payload: Dict[str, Any] = dict(runtime.state)
            send_payload["messages"] = [HumanMessage(content=task_description)]
            send_payload["user_query"] = task_description
            child_input = coordinator.input_transform(send_payload)

            # Invoke the deep agent
            # Build config: coordinator owns what context the deep agent needs
            child_config: Dict[str, Any] = {
                k: v for k, v in config.items() if k != "store"
            }
            
            # Extract the actual store bound to the deep agent graph
            child_store = getattr(deep_agent_graph, "store", None)
            if child_store is None and hasattr(deep_agent_graph, "bound"):
                child_store = getattr(deep_agent_graph.bound, "store", None)
            
            configurable = dict(config.get("configurable", {}))
            
            # CRITICAL FIX: The outer config carries `__pregel_runtime` which
            # has its own `store=None` (since supervisor compiles without a store).
            # When bridging config into `ainvoke`, LangGraph Pregel keeps the parent's
            # Runtime object. We MUST override its `store` attribute so the deepagents
            # middleware (like MemoryMiddleware) has access to the InMemoryStore.
            runtime_obj = configurable.get("__pregel_runtime")
            if runtime_obj is not None and hasattr(runtime_obj, "override") and child_store is not None:
                configurable["__pregel_runtime"] = runtime_obj.override(store=child_store)

            child_config["configurable"] = {
                **configurable,
                "thread_id": runtime.state.get("session_id", "default"),
                "context": coordinator.build_context(
                    supervisor_state=dict(runtime.state),
                ),
            }

            child_config["recursion_limit"] = getattr(self.config_instance, "RECURSION_LIMIT", 250)

            final_child_state = await deep_agent_graph.ainvoke(
                child_input,
                config=cast(RunnableConfig, child_config),
            )
            if final_child_state is None:
                raise ValueError("TFCoordinator deep agent returned no state")

            # Transform output: deep agent state → supervisor payload
            if isinstance(final_child_state, dict):
                child_state_dict: Dict[str, Any] = cast(Dict[str, Any], final_child_state)
            elif hasattr(final_child_state, "model_dump"):
                child_state_dict = cast(Dict[str, Any], final_child_state.model_dump())
            else:
                child_state_dict = cast(Dict[str, Any], dict(final_child_state))

            terraform_payload = coordinator.output_transform(child_state_dict)

            # Update workflow state
            wf = self._coerce_workflow_state(dict(runtime.state))
            wf.terraform_complete = True
            wf.last_agent = "terraform_coordinator"
            wf.next_agent = None
            wf.workflow_complete = True

            # Build tool message for the coordinator
            final_msg_content = terraform_payload.get("final_message") or "Terraform deep agent completed."
            tool_msg = ToolMessage(
                content=final_msg_content,
                tool_call_id=tool_call_id,
            )

            logger.info(
                "TFCoordinator deep agent finished",
                extra={
                    "workflow_progress": wf.get_workflow_progress(),
                    "has_terraform_output": bool(terraform_payload.get("terraform_output")),
                },
            )

            return Command(
                update={
                    "workflow_state": wf.model_dump(),
                    "terraform_output": terraform_payload.get("terraform_output", {}),
                    "messages": [tool_msg],
                    "status": "completed",
                    "workflow_complete": True,
                }
            )

        tools.append(transfer_to_terraform)
        return tools

    @staticmethod
    def _make_request_human_input_tool():
        """Create the HITL tool for greetings/out-of-scope/clarification."""

        @tool
        def request_human_input(
            question: str,
            context: Optional[str] = None,
            tool_call_id: Annotated[str, InjectedToolCallId] = "",
        ) -> Command:
            """
            Request human input for greetings, out-of-scope requests, or clarifications.

            Use when:
            - User sends greetings — create friendly, contextual greeting
            - Request is out-of-scope — guide user to Terraform tasks
            - Need clarification on ambiguous requirements

            Args:
                question: Dynamic, contextual message for the human
                context: Optional context about why feedback is needed
            """
            if not tool_call_id:
                tool_call_id = "unknown"

            logger.info(
                "Supervisor requesting human input",
                extra={
                    "question_preview": question[:200],
                    "tool_call_id": tool_call_id,
                },
            )

            # Pause execution — interrupt() returns human response on resume
            payload = {
                "pending_feedback_requests": {
                    "status": "input_required",
                    "question": question,
                    "tool_name": "request_human_input",
                }
            }

            response = interrupt(payload)
            response_str = str(response) if response is not None else ""

            tool_msg = ToolMessage(
                content=f"Human input received: {response_str}",
                tool_call_id=tool_call_id,
            )

            return Command(
                update={
                    "pending_feedback_requests": {},
                    "messages": [tool_msg],
                    "user_query": response_str,
                    "status": "in_progress",
                }
            )

        return request_human_input

    @staticmethod
    def _coerce_workflow_state(state: Dict[str, Any]) -> SupervisorWorkflowState:
        """Coerce workflow_state to a SupervisorWorkflowState instance."""
        existing = state.get("workflow_state")
        if isinstance(existing, SupervisorWorkflowState):
            return existing
        if isinstance(existing, dict):
            return SupervisorWorkflowState(**existing)
        return SupervisorWorkflowState()

    # ── Streaming ─────────────────────────────────────────────────────

    async def stream(
        self,
        query: Union[str, Command],
        context_id: str,
        task_id: str,
        use_ui: bool = False,
    ) -> AsyncGenerator[AgentResponse, None]:
        """Stream graph execution, yielding :class:`AgentResponse` objects.

        Handles both new queries (``str``) and resume-after-interrupt
        (``Command(resume=...)``).
        """
        if not self._graph:
            raise RuntimeError("Supervisor graph not constructed")

        is_resume = False
        stream_input: Any

        # Case 1: Command (resume from HITL)
        if isinstance(query, Command):
            stream_input = query
            is_resume = True
        else:
            # Case 2: String — check for JSON HITL resume payload
            try:
                parsed = json.loads(query)
                if isinstance(parsed, dict) and "decisions" in parsed:
                    stream_input = Command(resume=parsed)
                    is_resume = True
                else:
                    stream_input = self._build_initial_input(query, context_id, task_id)
            except (json.JSONDecodeError, TypeError):
                stream_input = self._build_initial_input(query, context_id, task_id)

        logger.info(
            "Starting supervisor stream",
            extra={
                "task_id": task_id,
                "context_id": context_id,
                "is_resume": is_resume,
                "query_preview": str(query)[:200] if not is_resume else None,
            },
        )

        config = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": context_id,
                    "context": self._coordinator.build_context(
                        supervisor_state={"session_id": context_id, "task_id": task_id}
                    ) if self._coordinator else {},
                },
            },
        )

        async for response in self._run_stream(stream_input, config, context_id, task_id):
            yield response

    # ── Internal v2 streaming (from existing implementation) ──────────

    async def _run_stream(
        self,
        stream_input: Any,
        config: RunnableConfig,
        context_id: str,
        task_id: str,
    ) -> AsyncGenerator[AgentResponse, None]:
        """Core streaming loop — processes v2 chunks from the supervisor graph."""
        assert self._graph is not None

        pending_interrupt: Optional[tuple] = None
        completed = False

        async for chunk in self._graph.astream(
            stream_input,
            config=config,
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2",
        ):
            if not isinstance(chunk, dict):
                continue

            chunk_type = chunk.get("type")
            ns: tuple = chunk.get("ns", ())
            data = chunk.get("data")
            if data is None:
                continue

            # ── updates: detect interrupts, subagent completion, progress ──
            if chunk_type == "updates" and isinstance(data, dict):
                update_data = cast(Dict[str, Any], data)

                interrupt_payload = self._extract_interrupt(update_data)
                if interrupt_payload:
                    pending_interrupt = interrupt_payload
                    continue

                # Check for subagent completion messages
                completion = self._extract_completion(ns, update_data, task_id, context_id)
                if completion:
                    yield completion
                    completed = True
                    continue

                # Intermediate progress
                for progress in self._extract_progress(ns, update_data, context_id, task_id):
                    yield progress

            # ── messages: stream LLM tokens ──
            elif chunk_type == "messages":
                token_response = self._extract_token(ns, data, context_id, task_id)
                if token_response:
                    yield token_response

        # After stream exhausts — handle pending interrupt
        if pending_interrupt:
            yield self._build_interrupt_response(pending_interrupt, context_id, task_id)
            return

        # If no explicit completion was yielded, emit a generic one
        if not completed:
            yield AgentResponse(
                content="Workflow completed.",
                response_type="text",
                is_task_complete=True,
                require_user_input=False,
                metadata={"context_id": context_id, "task_id": task_id, "status": "completed"},
            )

    # ── Chunk processors ──────────────────────────────────────────────

    @staticmethod
    def _extract_interrupt(data: Dict[str, Any]) -> Optional[tuple]:
        """Detect ``__interrupt__`` in an updates chunk."""
        if "__interrupt__" in data:
            return data["__interrupt__"]
        for v in data.values():
            if isinstance(v, dict) and "__interrupt__" in v:
                return v["__interrupt__"]
        return None

    # Tools that handle HITL flows — their ToolMessages are NOT subagent completions.
    _HITL_TOOL_NAMES: frozenset = frozenset({"request_human_input"})

    def _extract_completion(
        self,
        ns: tuple,
        data: Dict[str, Any],
        task_id: str,
        context_id: str,
    ) -> Optional[AgentResponse]:
        """Detect subagent completion in the tools node (result returned to supervisor).

        Guards:
        - Only inspect top-level (non-subgraph) updates (``ns`` must be empty).
        - Skip if ``tools_data["status"] == "in_progress"`` — the workflow
          continues (e.g. after ``request_human_input`` resolves its interrupt
          and the graph loops back to the model).
        - Skip ToolMessages whose ``name`` is a known HITL tool — those are
          interrupt-resume bookkeeping messages, not real subagent results.
        """
        if ns:
            return None  # only check top-level updates

        tools_data = data.get("tools")
        if not tools_data or not isinstance(tools_data, dict):
            return None

        # Explicit in_progress status means the workflow is still running
        # (e.g. request_human_input just resolved; graph loops back to model).
        if tools_data.get("status") == "in_progress":
            logger.debug(
                "tools node status=in_progress — skipping completion check",
                extra={"task_id": task_id},
            )
            return None

        messages = tools_data.get("messages", [])
        for msg in messages:
            if getattr(msg, "type", None) != "tool":
                continue

            name = getattr(msg, "name", "subagent")

            # HITL tools emit a ToolMessage after interrupt() resolves,
            # but the graph is NOT done — it loops back to the model.
            if name in self._HITL_TOOL_NAMES:
                logger.debug(
                    "Skipping HITL tool message — not a subagent completion",
                    extra={"tool_name": name, "task_id": task_id},
                )
                continue

            content = _extract_content_text(getattr(msg, "content", ""))

            logger.info(
                "Subagent completed",
                extra={"subagent": name, "content_preview": content[:200]},
            )

            return AgentResponse(
                content=content,
                response_type="text",
                is_task_complete=True,
                require_user_input=False,
                metadata={
                    "context_id": context_id,
                    "task_id": task_id,
                    "status": "completed",
                    "subagent": name,
                },
            )
        return None

    # ── Formatting constants (shared by _extract_progress) ─────────────
    _TOOL_CALL_FMT = "> ⚙️ **Tool Call** (`{name}`): _{arg}_  "
    _TOOL_CALL_FMT_SHORT = "> ⚙️ **Tool Call** (`{name}`)  "
    _TOOL_RESULT_FMT = "> {icon} **Result** (`{name}`): {snippet}...\n\n"
    _TOOL_RESULT_FMT_DONE = "> {icon} **Result** (`{name}`) completed.\n\n"

    # Keys to surface as the "key arg" for a tool call (in priority order)
    _KEY_ARG_FIELDS = ("task_description", "question", "query", "message", "content")

    @staticmethod
    def _extract_progress(
        ns: tuple,
        data: Dict[str, Any],
        context_id: str,
        task_id: str,
    ) -> List[AgentResponse]:
        """Extract intermediate progress updates from ``updates`` chunks.

        Dispatches by message type:
          • ``ai`` with tool_calls → formatted tool-call blockquotes
          • ``tool`` at subgraph level → formatted result snippet
          • Everything else → skipped (AI text is handled by ``_extract_token``)
        """
        source = _source_label(ns)
        responses: List[AgentResponse] = []

        for node_name, node_data in data.items():
            if not isinstance(node_data, dict):
                continue
            for msg in _iter_messages(node_data):
                msg_type = getattr(msg, "type", None)
                resp = None

                if msg_type == "ai":
                    resp = SupervisorAgent._progress_from_ai(
                        msg, source, node_name, context_id, task_id
                    )
                elif msg_type == "tool" and ns:
                    resp = SupervisorAgent._progress_from_tool(
                        msg, source, node_name, context_id, task_id
                    )

                if resp is not None:
                    responses.append(resp)

        return responses

    @staticmethod
    def _progress_from_ai(
        msg: Any, source: str, node: str, context_id: str, task_id: str
    ) -> Optional[AgentResponse]:
        """Build a tool-call progress response from an AIMessage.

        Shows only the tool name badge — the key argument text is already
        streamed as AI text by ``_extract_token`` (Gemini path) or appeared
        in the ``content`` field (OpenAI/Anthropic path).
        """
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return None

        lines: List[str] = []
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool")
            lines.append(SupervisorAgent._TOOL_CALL_FMT_SHORT.format(name=name))

        return AgentResponse(
            content="\n".join(lines) + "\n\n",
            response_type="token",
            is_task_complete=False,
            require_user_input=False,
            metadata={
                "context_id": context_id, "task_id": task_id,
                "status": "working", "message_type": "tool_call",
                "source": source, "node": node,
            },
        )

    @staticmethod
    def _progress_from_tool(
        msg: Any, source: str, node: str, context_id: str, task_id: str
    ) -> AgentResponse:
        """Build a tool-result progress response from a ToolMessage."""
        tool_name = getattr(msg, "name", "")
        snippet = _extract_content_text(getattr(msg, "content", "")).strip().replace("\n", " ")[:200]

        is_error = getattr(msg, "status", "success") == "error" or any(
            err_term in snippet.lower() for err_term in ("error", "exception", "failed", "could not")
        )
        icon = "❌" if is_error else "✅"

        if snippet:
            display = SupervisorAgent._TOOL_RESULT_FMT.format(icon=icon, name=tool_name, snippet=snippet)
        else:
            display = SupervisorAgent._TOOL_RESULT_FMT_DONE.format(icon=icon, name=tool_name)

        return AgentResponse(
            content=display,
            response_type="token",
            is_task_complete=False,
            require_user_input=False,
            metadata={
                "context_id": context_id, "task_id": task_id,
                "status": "working", "message_type": "tool_result",
                "tool_name": tool_name, "source": source, "node": node,
            },
        )

    # Fields whose values represent the AI's "reasoning" / "intent" text
    # when the model embeds its thinking inside tool-call arguments.
    _AI_TEXT_FIELDS = ("question", "task_description", "description", "message", "query", "content")

    @staticmethod
    def _extract_token(
        ns: tuple,
        data: Any,
        context_id: str,
        task_id: str,
    ) -> Optional[AgentResponse]:
        """Extract streaming AI text tokens from a ``messages`` chunk.

        Two paths:

        1. **Standard (OpenAI / Anthropic)** — ``token.content`` has text.
           Stream it directly as a token.

        2. **Gemini** — ``token.content`` is ``[]`` (empty), but the model's
           reasoning is embedded inside ``tool_call_chunks[*].args`` as JSON.
           Extract the most meaningful argument value and stream it as AI
           text so the user sees *what* the model decided, not just silence.

        ``ToolMessage`` and ``chunk_position='last'`` sentinels are skipped.
        """
        if not isinstance(data, (list, tuple)) or len(data) < 1:
            return None

        token = data[0]
        chunk_meta = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}
        source = _source_label(ns)
        agent_name = (
            chunk_meta.get("lc_agent_name")
            or chunk_meta.get("langgraph_node")
            or ""
        )

        if getattr(token, "type", None) not in ("ai", "AIMessageChunk"):
            return None   # ToolMessage / other — handled via updates path

        # Skip the empty sentinel chunk (chunk_position='last')
        if getattr(token, "chunk_position", None) == "last":
            return None

        # ── Path 1: text in content (OpenAI / Anthropic) ──────────────
        text = _extract_content_text(getattr(token, "content", ""))
        if text:
            return SupervisorAgent._token_response(
                text, context_id, task_id, source, agent_name
            )

        # ── Path 2: Gemini — text buried in tool_call_chunks args ─────
        text = _extract_text_from_tool_chunks(
            getattr(token, "tool_call_chunks", None),
            SupervisorAgent._AI_TEXT_FIELDS,
        )
        if text:
            return SupervisorAgent._token_response(
                text + "\n\n",  # line break before the tool badge that follows
                context_id, task_id, source, agent_name,
                message_type="tool_call",  # Gemini reasoning → inside thinking block
            )

        return None

    @staticmethod
    def _token_response(
        text: str, context_id: str, task_id: str, source: str, agent_name: str,
        message_type: Optional[str] = None,
    ) -> AgentResponse:
        """Build a streaming AI-text token response."""
        meta: Dict[str, Any] = {
            "context_id": context_id,
            "task_id": task_id,
            "status": "working",
            "stream_mode": "messages",
            "source": source,
            "agent_name": agent_name,
        }
        if message_type:
            meta["message_type"] = message_type
        return AgentResponse(
            content=text,
            response_type="token",
            is_task_complete=False,
            require_user_input=False,
            metadata=meta,
        )

    # ── HITL interrupt handling ────────────────────────────────────────

    @staticmethod
    def _build_interrupt_response(
        interrupt_payload: tuple,
        context_id: str,
        task_id: str,
    ) -> AgentResponse:
        """Convert a LangGraph interrupt payload into an ``AgentResponse``.

        Handles three interrupt shapes:

        1. **pending_feedback_requests** — ``request_human_input`` tool pattern.
           Content becomes ``{"type": "human_feedback_request", ...}``.

        2. **Custom interrupt types** — Tools that call ``interrupt()`` directly
           with a ``{"type": "...", ...}`` payload (e.g. ``user_input_request``).
           The original payload is passed through as-is in ``AgentResponse.content``
           so the A2UI component registry can route to the matching component.

        3. **action_requests** — ``HumanInTheLoopMiddleware`` pattern (approve/
           edit/reject).  Content becomes ``{"type": "hitl_approval", ...}``.
        """
        first = interrupt_payload[0] if interrupt_payload else {}
        value = getattr(first, "value", first)

        if not isinstance(value, dict):
            value = {"type": "generic", "data": str(value)}

        # ── Branch 1: pending_feedback_requests (request_human_input) ─────
        feedback_raw = value.get("pending_feedback_requests", {})
        if feedback_raw and isinstance(feedback_raw, dict):
            feedback: Dict[str, Any] = feedback_raw
            return AgentResponse(
                content={
                    "type": "human_feedback_request",
                    "question": feedback.get("question", "Input required"),
                    "status": feedback.get("status", "input_required"),
                },
                response_type="human_input",
                is_task_complete=False,
                require_user_input=True,
                metadata={
                    "context_id": context_id,
                    "task_id": task_id,
                    "interrupt_type": "human_feedback",
                    "pending_feedback_requests": feedback,
                },
            )

        # ── Branch 2: Custom interrupt types (user_input_request, etc.) ──────
        # Tools that call interrupt() directly with {"type": "...", ...}
        # pass through their payload unmodified so the A2UI component
        # registry can match on the type field.
        custom_type = value.get("type", "")
        if custom_type and custom_type not in ("generic", "hitl_approval"):
            logger.info(
                "Custom interrupt type detected — passing through to A2UI",
                extra={
                    "task_id": task_id,
                    "interrupt_type": custom_type,
                },
            )
            return AgentResponse(
                content=value,  # Pass through entire payload as-is
                response_type="human_input",
                is_task_complete=False,
                require_user_input=True,
                metadata={
                    "context_id": context_id,
                    "task_id": task_id,
                    "interrupt_type": custom_type,
                },
            )

        # ── Branch 3: action_requests (HITL middleware — approve/edit/reject)
        action_requests = value.get("action_requests", [])
        summary_parts: List[str] = []
        for req in action_requests:
            if isinstance(req, dict):
                req_name = req.get("name", "unknown")
                args = req.get("args", {})
                summary_parts.append(f"Tool: {req_name}, Args: {json.dumps(args, indent=2)}")

        summary = "\n".join(summary_parts) if summary_parts else json.dumps(value, indent=2, default=str)

        logger.info(
            "HITL interrupt detected",
            extra={
                "task_id": task_id,
                "action_count": len(action_requests),
                "summary_preview": summary[:200],
            },
        )

        return AgentResponse(
            content={
                "type": "hitl_approval",
                "summary": f"Approval required:\n{summary}",
                "question": "Do you approve this action?",
                "action_requests": action_requests,
                "raw_interrupt": value,
            },
            response_type="human_input",
            is_task_complete=False,
            require_user_input=True,
            metadata={
                "context_id": context_id,
                "task_id": task_id,
                "interrupt_type": "hitl_approval",
                "action_requests": action_requests,
            },
        )

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_initial_input(
        query: str,
        context_id: str,
        task_id: str,
    ) -> Dict[str, Any]:
        """Build the initial input dict for a new conversation."""
        return {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "session_id": context_id,
            "task_id": task_id,
            "workflow_state": SupervisorWorkflowState(
                current_phase="terraform_generation",
                terraform_complete=False,
                workflow_complete=False,
                next_agent="terraform_coordinator",
            ).model_dump(),
            "status": "in_progress",
        }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_messages(node_data: Dict[str, Any]):
    """Yield individual messages from a node's update data.

    Handles the quirks of LangGraph state diffs:
      • ``node_data["messages"]`` may be a list, a single object, or wrapped
        in a ``.value`` attribute (LangGraph state channels).
    """
    messages = node_data.get("messages", [])
    if hasattr(messages, "value"):
        messages = messages.value
    if not messages:
        return
    if not isinstance(messages, list):
        messages = [messages]
    yield from messages


def _extract_key_arg(
    args: Any, fields: tuple, max_len: int = 150
) -> str:
    """Extract the most meaningful argument from a tool call's args dict.

    Checks ``fields`` in priority order and returns the first non-empty
    value, cleaned up for inline display.
    """
    if not isinstance(args, dict):
        return ""
    for key in fields:
        val = args.get(key, "")
        if val and str(val).strip():
            return str(val).strip().replace("\n", " ")[:max_len]
    return ""


def _extract_content_text(content: Any) -> str:
    """Extract plain text from an AIMessageChunk's ``content`` field.

    Handles both the string format (OpenAI) and the list-of-blocks format
    (Anthropic / multi-modal).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content) if content else ""


def _extract_text_from_tool_chunks(
    chunks: Any, fields: tuple
) -> str:
    """Extract AI-reasoning text from Gemini-style ``tool_call_chunks``.

    Gemini puts its entire response inside function-call arguments rather
    than in ``content``.  This function parses the JSON ``args`` string and
    returns the first field value that looks like the AI's reasoning text.
    """
    if not chunks:
        return ""
    for tcc in chunks:
        args_str = tcc.get("args", "") if isinstance(tcc, dict) else getattr(tcc, "args", "")
        if not args_str or not isinstance(args_str, str):
            continue
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            continue
        for key in fields:
            val = args.get(key, "")
            if val and str(val).strip():
                return str(val).strip()
    return ""


# Matches UUID-like strings (e.g. tool call IDs used as namespace segments)
import re as _re
_UUID_RE = _re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-")


def _source_label(ns: tuple) -> str:
    """Derive a human-readable label from the v2 namespace tuple.

    LangGraph v2 streaming emits ``ns`` as a tuple of namespace segments.
    For subgraph streaming these look like:
      - ``()`` → top-level coordinator
      - ``('tools:transfer_to_terraform',)`` → first-level subagent
      - ``('tools:a85ae515-308e-...',)`` → UUID tool-call ID (common)

    We extract the deepest meaningful (non-UUID) agent name for display.
    """
    if not ns:
        return "coordinator"
    # Walk segments in reverse to find the deepest named agent
    for seg in reversed(ns):
        if not isinstance(seg, str):
            continue
        raw_name = seg
        # Strip 'tools:' prefix if present
        if raw_name.startswith("tools:"):
            raw_name = raw_name.split(":", 1)[1]
        # Skip UUID-like segments — they're tool call IDs, not agent names
        if _UUID_RE.match(raw_name):
            continue
        # Strip common tool-name prefixes for cleaner display
        for prefix in ("transfer_to_", "call_", "invoke_"):
            if raw_name.startswith(prefix):
                raw_name = raw_name[len(prefix):]
                break
        label = raw_name.replace("_", " ").strip()
        if label:
            return label
    # All segments were UUIDs — we're in a subagent but can't name it
    return "subagent"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_supervisor_agent(
    agents: List[Any],
    config: Optional[Any] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[str] = None,
    name: str = "aws_orchestrator",
) -> SupervisorAgent:
    """
    Create a SupervisorAgent with centralized configuration.

    Convenience factory that mirrors the k8s-autopilot pattern.

    Args:
        agents: List of subgraph agents/coordinators to orchestrate.
        config: ``Config`` instance or raw dict. Defaults to ``Config()``.
        custom_config: Optional overrides merged into the config.
        prompt_template: Optional custom system prompt (replaces default).
        name: Name of the supervisor agent.

    Returns:
        Configured ``SupervisorAgent`` ready for ``initialize()`` and ``stream()``.
    """
    from aws_orchestrator_agent.config import Config as _Config

    if config is None:
        cfg = _Config(custom_config)
    elif isinstance(config, dict):
        merged = {**config, **(custom_config or {})}
        cfg = _Config(merged)
    else:
        cfg = config

    agent = SupervisorAgent(agents=agents, config=cfg, name=name)

    # Allow callers to override the default prompt
    if prompt_template is not None:
        # The prompt is baked into _build_supervisor_graph at initialize() time,
        # so we store it on the instance for later injection during build.
        agent._custom_prompt = prompt_template  # type: ignore[attr-defined]

    return agent
