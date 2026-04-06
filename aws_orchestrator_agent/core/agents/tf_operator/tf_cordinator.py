"""
Terraform Deep Agent Coordinator.

Production-grade implementation of the deep agent pattern for Terraform
module generation and updates. Wires backends, MCP tools, and subagents
via the ``BaseDeepAgent`` abstract class.

Workflows:
    **New Module**:  tf-planner → tf-skill-builder → tf-generator → tf-validator → HITL → github-agent
    **Update Module**: update-planner → tf-updater → tf-validator → HITL → github-agent
"""

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from aws_orchestrator_agent.config import Config
from aws_orchestrator_agent.core.agents.tf_operator.backends.memory import (
    TFOperatorBackendMixin,
    sync_workspace_to_disk,
)
from aws_orchestrator_agent.core.agents.tf_operator.middleware import (
    build_deep_agent_middleware,
)
from aws_orchestrator_agent.core.agents.tf_operator.subagents import (
    get_subagent_specs,
)
from aws_orchestrator_agent.core.agents.types import BaseDeepAgent
from aws_orchestrator_agent.core.state.tf_coordinator_state import (
    TFCoordinatorContext,
)
from aws_orchestrator_agent.utils import AgentLogger
from aws_orchestrator_agent.utils.llm import create_model
from aws_orchestrator_agent.utils.mcp_client import MCPClient

logger = AgentLogger("TFCoordinator")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

COORDINATOR_SYSTEM_PROMPT = """\
You are the Terraform AWS Module Generator coordinator.
You orchestrate creating and updating Terraform modules via specialized sub-agents.

## Sub-Agent Skills
Each sub-agent has its own specialized skills loaded automatically:
- tf-generator: generates .tf files using skill-defined patterns, variables, and outputs
- tf-validator: terraform validation workflow and common error rules
- github-agent: GitHub commit workflow via MCP tools
- tf-skill-builder: creates new skill directories when no skill exists for a service
- tf-updater: fetches and patches existing modules from GitHub
- update-planner: analyses existing modules for targeted change planning
- tf-planner: orchestrates requirements analysis → security/best-practices → execution planning

You do NOT need to tell sub-agents to read skill files — they load automatically.

## Tool: request_user_input
You have a generic HITL tool to pause and ask the user anything.
Call it whenever you need human input.  Arguments:
  - question (str, required): the question to ask
  - title (str): card header
  - context (str): extra context shown below the question
  - options (list of dicts): buttons for the user, each with:
      {"key": "...", "label": "...", "primary": true/false}
  - input_fields (list of dicts): text fields to collect data, each with:
      {"key": "...", "label": "...", "default": "...", "required": true/false}

The user's response comes back as a ToolMessage.  Read it and act accordingly.

## HITL Policy
At session start, read /memories/hitl-policies.md for the complete HITL policy.
That file defines WHEN you MUST call request_user_input (mandatory gates) and
WHEN you MAY call it (optional gates, at your discretion).
The workflow steps below show HOW to call it (which options and fields to use).

## Workflow — New Module
1. Check if skills exist for the requested service:
   - Use read_file to check /skills/{service}-module-generator/SKILL.md
   - If it exists AND metadata.provider-version matches the current version:
     SKIP tf-planner and tf-skill-builder entirely. Go directly to step 3.
   - If it does NOT exist, OR the version is stale:
     task(tf-planner): "Plan Terraform module for: {request}" (runs req-analysis → sec-best-practices → execution-planning → skill-writing)
2. CHECK the planner's output message:
   - IF the planner output contains "Skills written for" → the planner has already created authoritative skill files from MCP data. **SKIP tf-skill-builder entirely.**
   - IF the planner output does NOT mention skills written → task(tf-skill-builder): "Build skill files for: {request}"
3. task(tf-generator): "Generate {service} module."
4. Call sync_workspace tool to materialise virtual files to disk before validation
5. task(tf-validator): "Validate module at service {service} under workspace/terraform_modules/"
6. If INVALID: task(tf-generator): "Fix these errors: {errors}"  → call sync_workspace again → repeat step 5
7. **[Commit Gate]** — (Mandatory per hitl-policies.md §1) call `request_user_input` with:
   - title: "Module '{service}' Validated ✅"
   - question: summary of generated files and local path
   - context: "Would you like to push to GitHub or keep locally?"
   - options:
     - {key: "push_to_github", label: "🚀 Push to GitHub", primary: true}
     - {key: "keep_local", label: "📁 Keep Local"}
   - input_fields:
     - {key: "repository", label: "Repository (owner/repo)"}
     - {key: "branch", label: "Branch", default: "main"}
8. **Handle the response:**
   - If user chose "push_to_github" and provided repository + branch:
     → task(github-agent): "Commit {service} module to {repository} branch {branch}"
   - If user chose "keep_local" or provided no repository:
     → Report local file paths to user. Do NOT call github-agent.
9. Update /memories/module-index.md and /memories/org-standards.md as needed
10. **[Next Steps Gate]** — (Mandatory per hitl-policies.md §2) call `request_user_input` with:
    - title: "What's Next?"
    - question: summary of what was accomplished (module name, files, commit status)
    - context: "I can generate more modules, update existing ones, or we're done."
    - options:
      - {key: "generate_another", label: "➕ Generate Another Module", primary: true}
      - {key: "update_existing", label: "🔄 Update Existing Module"}
      - {key: "done", label: "✅ I'm Done"}
    - input_fields:
      - {key: "details", label: "Provide details for next action"}
    - If user chose "generate_another" → loop back to step 1 with new details
    - If user chose "update_existing" → switch to Update Module workflow with details
    - If user chose "done" → complete the task

## Workflow — Update Module
1. task(update-planner): "Analyse {module_path} on {repo}: {what to change}"
2. task(tf-updater): "Fetch and update {module_path} on {repo}: {what to change}"
3. task(tf-validator): "Validate module at service {service}"
4. If INVALID: task(tf-updater): "Fix: {errors}" → repeat step 3
5. **[Commit Gate]** — same as New Module step 7 using `request_user_input`
6. **[Next Steps Gate]** — same as New Module step 10 using `request_user_input`

## Memory Rules
- At session start: ALWAYS read in this order:
  1. /memories/AGENTS.md (memory index — what files exist)
  2. /memories/hitl-policies.md (HITL gate policies — when to pause for human input)
  3. /memories/org-standards.md (org conventions — if it exists)
- For UPDATE mode: also read /memories/module-index.md
- At session end: update /memories/module-index.md and any other relevant memory files
- If you discover a new situation that should require HITL, update /memories/hitl-policies.md

## Parallel Execution (Phase 2)
When the user requests multiple independent AWS modules in a single message:
- Dispatch tf-generator for all modules in ONE response (parallel)
- After all complete, dispatch tf-validator for all modules in ONE response (parallel)
- Only after all validations pass: one request_user_input covering all modules
- Then dispatch github-agent once to commit everything

## CRITICAL: Step Budget
You have a limited number of steps (~150 total). Be efficient:
- The typical flow is: tf-planner → tf-generator → tf-validator → request_user_input (3 sub-agent calls + 1–2 tool calls).
- If tf-planner writes skills, tf-skill-builder is NEVER needed — still 3 sub-agent calls.
- NEVER call more than 5 sub-agents for a single module request (including github-agent if approved).
- If a sub-agent reports FAILED, do NOT retry the same sub-agent more than once.

## Workspace Sync
Generated .tf files are written to a virtual filesystem under /workspace/.
The coordinator automatically syncs them to the real disk for validation.
Do NOT ask tf-generator to re-write files just because tf-validator says
"directory not found" — the sync happens automatically.

## Rules — Never Violate
- NEVER write .tf files yourself — always delegate to tf-generator or tf-updater
- NEVER run terraform commands yourself — always delegate to tf-validator
- NEVER interact with GitHub yourself — always delegate to github-agent
- NEVER commit to GitHub without the user providing repository and branch
- ALWAYS follow the HITL policies in /memories/hitl-policies.md
- The DEFAULT outcome for the commit gate is always LOCAL — do NOT assume GitHub push
"""


# ---------------------------------------------------------------------------
# TFCoordinator — the deep agent
# ---------------------------------------------------------------------------

class TFCoordinator(BaseDeepAgent):
    """
    Terraform Deep Agent Coordinator.

    Production implementation of the deep agent pattern that:
    - Inherits lifecycle from ``BaseDeepAgent``
    - Uses ``TFOperatorBackendMixin`` for Terraform-specific backend routing
    - Connects to GitHub MCP server for file operations
    - Manages sub-agents (dict specs + CompiledSubAgent) for the full pipeline
    - Supports HITL approval gates before GitHub commits
    - Implements ``input_transform`` / ``output_transform`` for subgraph state bridging
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        mcp_server_filter: Optional[List[str]] = None,
    ) -> None:
        super().__init__(config=config)
        self._mcp_server_filter = mcp_server_filter

        logger.info("TFCoordinator initialized")

    # ── Abstract implementations — Properties ────────────────────────────

    @property
    def name(self) -> str:
        return "terraform-coordinator"

    @property
    def system_prompt(self) -> str:
        return COORDINATOR_SYSTEM_PROMPT

    @property
    def context_schema(self) -> type:
        return TFCoordinatorContext

    # ── Abstract implementations — Model ─────────────────────────────────

    def get_model(self) -> Any:
        """Return an initialized deep-agent tier LLM model.

        Uses ``create_model()`` with the full ``LLM_DEEPAGENT_*`` config
        (provider, model, temperature, max_tokens) from ``Config``.
        """
        return create_model(self._config.get_llm_deepagent_config())

    def _get_validator_model(self) -> Any:
        """Return an initialized standard-tier LLM for the tf-validator.

        Uses ``create_model()`` with the full ``LLM_*`` config
        (provider, model, temperature, max_tokens) from ``Config``.
        """
        return create_model(self._config.get_llm_config())

    # ── Abstract implementations — Sub-agents ────────────────────────────

    async def get_subagent_specs(self) -> List[Any]:
        """
        Build sub-agent specs.

        Returns a mixed list of:
        - Dict specs for simple sub-agents (tf-generator, tf-validator, etc.)
        - `CompiledSubAgent` wrappers for GitHub MCP-dependent agents (JIT nodes)
        - `CompiledSubAgent` for the planner supervisor (compiled LangGraph subgraph)

        **State bridging for PlannerSupervisorAgent:**

        The deep agent framework invokes ``CompiledSubAgent.runnable.invoke(state)``
        where ``state = {parent_state_minus_excluded, messages: [HumanMessage(task_desc)]}``.
        Since ``TFPlannerState`` has a different schema (``user_query``, ``workflow_state``,
        ``active_agent``, etc.), we wrap the compiled planner graph in a
        ``RunnableLambda`` that follows the official LangGraph
        "call a subgraph inside a node" pattern:

            1. ``planner.input_transform(state)`` → bridges deep-agent state → TFPlannerState
            2. ``planner_graph.invoke(transformed)`` → runs the 3-phase pipeline
            3. ``planner.output_transform(result)`` → bridges TFPlannerState → deep-agent state

        The wrapper ensures the output always contains a ``messages`` key, which is
        required by the framework's ``_return_command_with_state_update()``.
        """
        from deepagents.middleware.subagents import CompiledSubAgent

        from aws_orchestrator_agent.core.agents.tf_operator.tf_planner import (
            PlannerSupervisorAgent,
        )

        # Get basic and JIT-compiled subagents from the spec definitions
        specs: List[Any] = get_subagent_specs(
            coordinator_model=self.get_model(),
            validator_model=self._get_validator_model(),
        )

        # Build the planner subgraph
        planner = PlannerSupervisorAgent(config=self._config)
        planner_graph = planner.build_graph()

        # ── RunnableLambda wrapper (official LangGraph pattern) ───────────
        #
        # When parent and subgraph have different state schemas, the official
        # docs recommend wrapping the subgraph invocation in a node function
        # that explicitly transforms state in both directions:
        #
        #   def call_subgraph(state: ParentState):
        #       response = subgraph.invoke({"bar": state["foo"]})
        #       return {"foo": response["bar"]}
        #
        # See: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
        #      #call-a-subgraph-inside-a-node
        #
        # Here, PlannerSupervisorAgent already has input_transform/output_transform
        # that handle the schema bridging.

        async def _planner_wrapper(
            state: Dict[str, Any],
            config: Optional[RunnableConfig] = None,
        ) -> Dict[str, Any]:
            """
            Bridge deep-agent state → TFPlannerState → deep-agent state.

            The deep agent framework calls node functions with (state, config).
            ``config["context"]`` carries the ``TFCoordinatorContext`` that was
            assembled by ``TFCoordinator.build_context()`` at the supervisor
            level.  This is the authoritative source for session-scoped
            identifiers (`session_id`, `task_id`) and user intent
            (`user_query`) that the planner's ``input_transform`` requires but
            that are NOT part of the deep agent's internal graph state schema.

            Data flow:
                supervisor.build_context(runtime_state)
                  → config["context"]  (TFCoordinatorContext)
                    → _planner_wrapper enriches `state` from context
                      → planner.input_transform(enriched_state)
                        → planner_graph.invoke(TFPlannerState)
                          → planner.output_transform(result)
                            → deep-agent state update
            """
            # ── Extract session context injected by supervisor ─────────────
            # config["configurable"]["context"] = TFCoordinatorContext dict built by
            # TFCoordinator.build_context(supervisor_state=dict(runtime.state))
            coordinator_ctx: Dict[str, Any] = {}
            if config and hasattr(config, "get"):
                configurable = config.get("configurable") or {}
                coordinator_ctx = configurable.get("context") or config.get("context") or {}

            # ── Enrich state with context values ──────────────────────────
            # Prefer values already in state (non-empty), fall back to context.
            # This ensures the planner's input_transform always sees:
            #   - session_id: for checkpoint thread identity
            #   - task_id:    for A2A task correlation
            #   - user_query: the human's original request
            enriched_state: Dict[str, Any] = {
                **state,
                "session_id": (
                    state.get("session_id")
                    or coordinator_ctx.get("session_id")
                ),
                "task_id": (
                    state.get("task_id")
                    or coordinator_ctx.get("task_id")
                ),
                "user_query": (
                    state.get("user_query")
                    or coordinator_ctx.get("user_query")
                    # last-resort: pull from the first human message content
                    or next(
                        (
                            getattr(m, "content", None)
                            or (m.get("content") if isinstance(m, dict) else None)
                            for m in reversed(state.get("messages") or [])
                            if (
                                getattr(m, "type", None) == "human"
                                or (isinstance(m, dict) and m.get("role") == "user")
                            )
                        ),
                        None,
                    )
                ),
            }

            logger.info(
                "_planner_wrapper: enriched state from TFCoordinatorContext",
                extra={
                    "session_id": enriched_state.get("session_id"),
                    "task_id": enriched_state.get("task_id"),
                    "has_user_query": bool(enriched_state.get("user_query")),
                    "message_count": len(enriched_state.get("messages") or []),
                },
            )

            # 1. Enriched deep-agent state → TFPlannerState input
            subgraph_input = planner.input_transform(enriched_state)

            # 2. Invoke the compiled planner subgraph
            #    Pass config through so planner nodes can also access context
            subgraph_output = await planner_graph.ainvoke(subgraph_input, config=config)

            # 3. TFPlannerState → CompiledSubAgent return value
            #    planner.output_transform owns the full output contract:
            #      - messages: [AIMessage(summary)]  → coordinator LLM context
            #      - files: {plan/*.json, ...}        → virtual FS for downstream agents
            #    Pass parent_files so the planner merges skills/memories with plan outputs.
            parent_files: Dict[str, Any] = enriched_state.get("files") or {}
            return planner.output_transform(subgraph_output, parent_files=parent_files)

        # Register as CompiledSubAgent with RunnableLambda as the runnable.
        # The framework calls runnable.invoke(state), which triggers our
        # wrapper → input_transform → graph.invoke → output_transform.
        planner_compiled = CompiledSubAgent(
            name="tf-planner",
            description=(
                "Orchestrates Terraform module planning through a 3-phase pipeline: "
                "requirements analysis → security & best practices → execution planning. "
                "Use this when a NEW module generation request arrives to produce a "
                "comprehensive plan before the tf-skill-builder and tf-generator run."
            ),
            runnable=RunnableLambda(_planner_wrapper),
        )
        specs.append(planner_compiled)

        return specs

    # ── Virtual overrides ─────────────────────────────────────────────────

    async def get_tools(self) -> List[Any]:
        """
        Coordinator-level tools.

        - ``sync_workspace``: Materialises virtual /workspace/ files to the
          real filesystem.  The coordinator MUST call this between
          tf-generator and tf-validator so `terraform validate` can find
          the generated .tf files on disk.
        - ``request_user_input``: Generic HITL gate — pause and ask the user
          anything (commit approval, next steps, clarification, etc.).
        """
        from langchain.tools import tool, ToolRuntime

        from aws_orchestrator_agent.core.agents.tf_operator.tools import (
            create_user_input_tool,
        )

        @tool
        def sync_workspace(
            runtime: ToolRuntime,
        ) -> str:
            """Sync virtual /workspace/ files to real disk.

            MUST be called after tf-generator finishes and BEFORE tf-validator runs.
            This materialises the generated .tf files from the virtual filesystem
            to the real project directory so terraform CLI commands can access them.

            Returns a summary of synced files.
            """
            # Read the files dict from the deep agent's current state
            state_files: Dict[str, Any] = {}
            if hasattr(runtime, "state") and isinstance(runtime.state, dict):
                state_files = runtime.state.get("files", {})
            elif hasattr(runtime, "state") and hasattr(runtime.state, "get"):
                state_files = runtime.state.get("files", {})

            if not state_files:
                return (
                    "No /workspace/ files found in state to sync. "
                    "Ensure tf-generator has completed before calling sync_workspace."
                )

            synced = sync_workspace_to_disk(state_files)

            if not synced:
                return (
                    "No /workspace/ files found in state to sync. "
                    "Files may use a different path prefix."
                )

            synced_list = "\n".join(
                f"  - {vpath} → {real}" for vpath, real in synced.items()
            )
            return (
                f"Synced {len(synced)} file(s) to disk:\n{synced_list}\n\n"
                "tf-validator can now run terraform validate against the real filesystem. "
                "Use execute() with relative paths (no leading /) for terraform commands."
            )

        # Build the generic user input HITL tool
        user_input = create_user_input_tool()

        return [sync_workspace, user_input]

    def get_skill_paths(self) -> List[str]:
        return ["/skills/"]

    def get_memory_paths(self) -> List[str]:
        return [
            "/memories/AGENTS.md",
            "/memories/hitl-policies.md",
        ]

    def get_interrupt_config(self) -> Dict[str, Any]:
        """HITL gates: require approval before destructive file operations."""
        return {
            "delete_module": {
                "allowed_decisions": cast(list, ["approve", "edit", "reject"]),
            },
        }

    # ── Abstract implementations — Backend & Storage ─────────────────────

    def make_backend(self, runtime: Any) -> Any:
        """Use terraform-specific backend with LocalShellBackend for CLI."""
        return TFOperatorBackendMixin.make_backend(runtime)

    def build_store(self) -> InMemoryStore:
        """InMemoryStore for cross-thread long-term memory (dev mode).

        Pre-seeds the store with memory files from disk so the deep agent's
        ``MemoryMiddleware`` can read ``/memories/AGENTS.md`` etc. on startup.
        Per the docs: *"you must add the expected skill or memory files to
        the backend before creating the agent"*.
        """
        from deepagents.backends.utils import create_file_data

        store = InMemoryStore()
        memory_dir = Path("./memory")

        if memory_dir.exists():
            # Namespace must match the StoreBackend route in make_backend().
            # The route uses: namespace=lambda ctx: (ctx.runtime.context.get("org_name", "default_org"),)
            namespace = ("default_org",)
            for path in memory_dir.rglob("*"):
                if path.is_file() and path.name != ".gitkeep":
                    # StoreBackend strips the route prefix ("/memories/")
                    # and stores relative to the namespace using the
                    # remaining path as the key.
                    key = path.relative_to(memory_dir).as_posix()
                    store.put(
                        namespace,
                        key,
                        create_file_data(path.read_text(encoding="utf-8")),
                    )
            logger.info(
                "build_store: pre-seeded InMemoryStore with memory files",
                extra={"namespace": namespace, "memory_dir": str(memory_dir)},
            )

        return store

    def build_checkpointer(self) -> MemorySaver:
        """MemorySaver for thread-scoped state persistence (dev mode)."""
        return MemorySaver()

    # ── Abstract implementations — build_agent & seed_files ──────────────

    async def build_agent(self) -> CompiledStateGraph:
        """
        Assemble all components into a ``create_deep_agent()`` call.

        Wires model, prompt, tools, subagents (dict + CompiledSubAgent),
        skills, memory, backend, store, checkpointer, HITL config,
        and context schema into a compiled LangGraph.
        """
        from deepagents import create_deep_agent

        make_backend_fn, store, checkpointer = self.build_memory_components()
        tools = await self.get_tools()
        subagents = await self.get_subagent_specs()
        middleware = build_deep_agent_middleware(config=self._config)

        return create_deep_agent(
            model=self.get_model(),
            name=self.name,
            system_prompt=self.system_prompt,
            tools=tools,
            subagents=subagents,
            skills=self.get_skill_paths(),
            memory=self.get_memory_paths(),
            backend=make_backend_fn,
            store=store,
            checkpointer=checkpointer,
            interrupt_on=self.get_interrupt_config(),
            context_schema=self.context_schema,
            middleware=middleware,
        )

    def seed_files(
        self,
        skills_dir: Optional[Any] = None,
        memory_dir: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load skill files and initial memory into the virtual FS dict.

        Walks ``skills_dir`` and ``memory_dir`` on disk, mapping each file
        to a virtual path (``/skills/...`` and ``/memories/...``).
        """
        from deepagents.backends.utils import create_file_data

        skills_path = Path(skills_dir) if skills_dir else Path("./skills")
        memory_path = Path(memory_dir) if memory_dir else Path("./memory")
        files: Dict[str, Any] = {}

        if skills_path.exists():
            for path in skills_path.rglob("*"):
                if path.is_file():
                    vpath = f"/skills/{path.relative_to(skills_path).as_posix()}"
                    files[vpath] = create_file_data(
                        path.read_text(encoding="utf-8")
                    )

        if memory_path.exists():
            for path in memory_path.rglob("*"):
                if path.is_file() and path.name != ".gitkeep":
                    vpath = f"/memories/{path.relative_to(memory_path).as_posix()}"
                    files[vpath] = create_file_data(
                        path.read_text(encoding="utf-8")
                    )

        return files

    # ── Supervisor-level state transforms ────────────────────────────────
    #
    # These transforms bridge SupervisorState ↔ deep agent invocation state.
    # They are called by the `transfer_to_terraform` @tool in the supervisor
    # when delegating work to the TFCoordinator deep agent.
    #
    # NOTE: These are DIFFERENT from the planner-level transforms in
    # PlannerSupervisorAgent.input_transform/output_transform, which bridge
    # deep agent internal state ↔ TFPlannerState.

    def input_transform(self, send_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform supervisor ``@tool`` payload → deep agent graph input (state).

        Extracts ``messages`` and seeds the virtual filesystem so the deep agent
        starts with skills and memory already loaded.

        Args:
            send_payload: Dict from ``dict(runtime.state)`` in the tool wrapper,
                          with ``messages`` replaced by ``[HumanMessage(task_description)]``.

        Returns:
            Deep agent graph input: ``{messages: [...], files: {...}}``
        """
        messages = send_payload.get("messages", [])
        files = self.seed_files()

        transformed: Dict[str, Any] = {
            "messages": messages,
        }

        # Only include files if there are any to seed
        if files:
            transformed["files"] = files

        logger.info(
            "input_transform: SupervisorState → deep agent input",
            extra={
                "message_count": len(messages),
                "file_count": len(files),
            },
        )

        return transformed

    def build_context(
        self,
        supervisor_state: Optional[Dict[str, Any]] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build the ``TFCoordinatorContext`` dict for ``config["context"]``.

        This is the **second** side of the three-way state bridge between the
        supervisor and the deep agent:

        1. ``input_transform``  → deep agent **graph state** (messages, files)
        2. ``build_context``    → deep agent **runtime config** (TFCoordinatorContext)
        3. ``output_transform`` → supervisor **graph state** (terraform_output, status)

        Merges three sources in ascending priority order so the caller only
        needs to supply what is available — all fields are optional:

        1. **Environment variables** — infra config (GitHub, AWS, Terraform).
        2. **Supervisor runtime state** — live session identifiers
           (``session_id``, ``task_id``, ``context``).
        3. **Caller-injected ``context`` dict** — if the A2A adapter already
           placed a ``TFCoordinatorContext``-shaped dict in ``SupervisorState``,
           those values win over everything else.

        Args:
            supervisor_state: ``dict(runtime.state)`` from the tool wrapper, or
                              ``{"session_id": ..., "task_id": ...}`` from the
                              top-level stream call.
            task_description: The crafted task string from the supervisor LLM.
                              Used to carry intent but *not* stored in context —
                              the coordinator's system prompt handles routing.

        Returns:
            A ``TFCoordinatorContext``-shaped dict ready for
            ``config={"context": ..., ...}`` in ``deep_agent_graph.ainvoke()``.
        """
        state = supervisor_state or {}

        # ── 1. Env-var base ───────────────────────────────────────────────
        ctx: Dict[str, Any] = {
            # GitHub
            "github_repo":          os.getenv("GITHUB_REPO", ""),
            "github_branch":        os.getenv("GITHUB_BRANCH", "main"),
            "github_commit_author": os.getenv("GITHUB_COMMIT_AUTHOR", "TalkOps Bot"),
            # Workspace
            "workspace_path":  os.getenv("TERRAFORM_WORKSPACE", "./workspace/terraform_modules"),
            "workflow_mode":   os.getenv("WORKFLOW_MODE", "new_module"),
            # AWS
            "aws_account_id": os.getenv("AWS_ACCOUNT_ID", ""),
            "aws_region":     os.getenv("AWS_REGION", "us-east-1"),
            "aws_profile":    os.getenv("AWS_PROFILE", ""),
            # Terraform
            "tf_version_constraint": os.getenv("TF_VERSION_CONSTRAINT", ">= 1.9.0, < 2.0.0"),
            "aws_provider_version":  os.getenv("AWS_PROVIDER_VERSION", ">= 5.40.0"),
            "tf_backend_type":       os.getenv("TF_BACKEND_TYPE", "s3"),
            "tf_state_bucket":       os.getenv("TF_STATE_BUCKET", ""),
            # Organization
            "org_name":        os.getenv("ORG_NAME", ""),
            "module_prefix":   os.getenv("MODULE_PREFIX", ""),
            "environment":     os.getenv("ENVIRONMENT", "development"),
            "require_approval": os.getenv("REQUIRE_APPROVAL", "true").lower() != "false",
        }

        # ── 2. Supervisor runtime state ───────────────────────────────────
        if state.get("session_id"):
            ctx["session_id"] = state["session_id"]
        if state.get("task_id"):
            ctx["task_id"] = state["task_id"]
        if "dry_run" in state:
            ctx["dry_run"] = bool(state["dry_run"])

        # ── 3. Caller-injected context (highest priority) ─────────────────
        # The A2A adapter may already have placed a TFCoordinatorContext dict
        # in SupervisorState["context"]. Those values override env vars.
        caller_ctx: Dict[str, Any] = state.get("context") or {}
        if isinstance(caller_ctx, dict):
            ctx.update({k: v for k, v in caller_ctx.items() if v is not None and v != ""})

        # Strip empty-string optional fields so the deep agent doesn't see ""
        for key in ("github_repo", "aws_account_id", "aws_profile",
                    "tf_state_bucket", "org_name", "module_prefix"):
            if ctx.get(key) == "":
                ctx.pop(key, None)

        logger.info(
            "build_context: TFCoordinatorContext assembled",
            extra={
                "fields": sorted(ctx.keys()),
                "session_id": ctx.get("session_id"),
                "task_id": ctx.get("task_id"),
                "github_repo": ctx.get("github_repo"),
                "environment": ctx.get("environment"),
            },
        )

        return ctx


    def output_transform(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform deep agent final state → supervisor-mergeable payload.

        Extracts the final message and any structured outputs from the deep
        agent's completed state, packaging them for the supervisor's
        ``Command(update={terraform_output: ..., ...})``.

        Args:
            agent_state: The dict returned by ``deep_agent.ainvoke()``.

        Returns:
            Payload with ``final_message``, ``status``, and full ``terraform_output``
            dict for the supervisor to merge into ``SupervisorState``.
        """
        # Handle both dict and Pydantic model
        state: Dict[str, Any] = agent_state
        if not isinstance(agent_state, dict) and hasattr(agent_state, "model_dump"):
            state = agent_state.model_dump()  # type: ignore[union-attr]

        # ── Sync virtual /workspace/ files to real disk ───────────────────
        # The tf-generator writes to StateBackend (/workspace/...), which is
        # ephemeral.  Materialise those files to the real filesystem so
        # tf-validator can run `terraform validate` against actual .tf files.
        files: Dict[str, Any] = state.get("files", {})
        synced: Dict[str, Any] = {}
        if files:
            try:
                synced = sync_workspace_to_disk(files)
            except Exception as e:
                logger.error(
                    "output_transform: failed to sync workspace files to disk",
                    extra={"error": str(e)},
                )

        # Extract the final message from the deep agent's conversation
        final_message: Optional[str] = None
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            final_message = getattr(last_msg, "content", None) or (
                last_msg.get("content") if isinstance(last_msg, dict) else None
            )

        # Build a structured output summary
        output: Dict[str, Any] = {
            "final_message": final_message or "Terraform deep agent completed.",
            "status": "completed",
            "terraform_output": {
                "messages": messages,
                "files": files,
                "synced_paths": {k: str(v) for k, v in synced.items()},
                "structured_response": state.get("structured_response"),
            },
        }

        logger.info(
            "output_transform: deep agent state → SupervisorState payload",
            extra={
                "has_final_message": bool(final_message),
                "message_count": len(messages),
                "synced_files": len(synced),
            },
        )

        return output




# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_tf_coordinator(
    config: Optional[Config] = None,
    mcp_server_filter: Optional[List[str]] = None,
) -> TFCoordinator:
    """
    Create a TFCoordinator instance.

    Args:
        config: Configuration object. Defaults to ``Config()`` (env + defaults).
        mcp_server_filter: Optional list of MCP server names to connect to.

    Returns:
        A ready-to-use ``TFCoordinator`` (call ``connect_mcp()`` before
        ``build_agent()`` if you need GitHub tools).

    Example::

        coord = create_tf_coordinator()
        await coord.connect_mcp()
        agent = await coord.build_agent()
        files = coord.seed_files()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Create a VPC module"}], "files": files},
            config={"configurable": {"thread_id": "t1"}, "context": {...}},
        )
    """
    return TFCoordinator(config=config, mcp_server_filter=mcp_server_filter)
