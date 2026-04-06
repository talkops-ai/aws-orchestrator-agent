"""
Planner Supervisor Agent — orchestrates the 3 sub-agents for Terraform
planning via a LangGraph subgraph with Command-based handoffs.

Pipeline:
  ReqAnalyserAgent → SecBestPracticesAgent → ExecutionPlannerAgent → END

Follows the same pattern as ``GenerationSupervisorAgent`` (ci_supervisor.py):
  - Handoff tools return ``Command(goto=..., graph=Command.PARENT)``
  - ``route_initial`` / ``route_after_agent`` use ``workflow_state.next_agent``
  - ``input_transform`` / ``output_transform`` bridge supervisor ↔ subgraph state
"""

import json
from typing import Any, Dict, Literal, Optional, Union, cast

from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from aws_orchestrator_agent.config import Config
from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module import (
    ExecutionPlannerAgent,
    ReqAnalyserAgent,
    SecBestPracticesAgent,
)
from aws_orchestrator_agent.core.agents.types import BaseSubgraphAgent
from aws_orchestrator_agent.core.state import (
    TFPlannerState,
    TFPlannerWorkflowState,
    WorkflowStatus,
)
from aws_orchestrator_agent.utils import (
    AgentLogger,
    initialize_llm_higher,
    initialize_llm_model,
)


logger = AgentLogger("PlannerSupervisorAgent")


# ============================================================================
# Supervisor Agent
# ============================================================================

class PlannerSupervisorAgent(BaseSubgraphAgent):
    """
    Planner Supervisor — subgraph agent that orchestrates the three planning
    sub-agents (requirements analyser → security/best-practices → execution planner)
    via Command-based handoff tools.

    Implements ``BaseSubgraphAgent`` so it can be mounted as a node inside
    a higher-level supervisor graph.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "planner_supervisor_agent",
        memory: Optional[MemorySaver] = None,
    ):
        logger.info("Initializing PlannerSupervisorAgent")

        self.config = config or Config(custom_config or {})
        self.llm_config = self.config.get_llm_config()
        self.llm_higher_config = self.config.get_llm_higher_config()

        # Models used by sub-agents
        self.model = initialize_llm_model(self.llm_config)
        self.model_higher = initialize_llm_higher(self.llm_higher_config)

        self._name = name
        self._memory = memory or MemorySaver()

        # Build sub-agent graphs (with handoff tools injected)
        self._initialize_sub_agents()

        logger.info("PlannerSupervisorAgent initialized successfully")

    # -- BaseSubgraphAgent properties --------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def state_model(self) -> type[TFPlannerState]:
        return TFPlannerState


    @property
    def memory(self) -> MemorySaver:
        return self._memory

    @memory.setter
    def memory(self, value: MemorySaver) -> None:
        self._memory = value

    # -- Helper ------------------------------------------------------------

    @staticmethod
    def _coerce_workflow_state(state: Any) -> TFPlannerWorkflowState:
        existing = state.get("workflow_state")
        if isinstance(existing, TFPlannerWorkflowState):
            return existing
        if isinstance(existing, dict):
            return TFPlannerWorkflowState(**existing)
        return TFPlannerWorkflowState()

    # -- Sub-agent initialisation (handoff tools + build) ------------------

    def _initialize_sub_agents(self) -> None:
        """Create handoff tools and build each sub-agent graph."""

        _coerce = self._coerce_workflow_state

        # ----------------------------------------------------------------
        # Handoff: requirements_analyser → security_best_practices
        # ----------------------------------------------------------------
        @tool
        def transfer_to_security_best_practices(
            runtime: ToolRuntime[None, TFPlannerState],
        ) -> Command:
            """Transfer from requirements analyser to security & best practices agent."""
            wf = _coerce(runtime.state)
            wf.req_analyser_complete = True
            wf.last_agent = "requirements_analyser"
            wf.next_agent = "security_best_practices"
            wf.current_phase = "sec_n_best_practices"

            logger.info(
                "Handoff: requirements_analyser → security_best_practices",
                extra={"workflow_progress": wf.get_workflow_progress()},
            )

            last_ai = next(
                msg for msg in reversed(runtime.state["messages"])
                if isinstance(msg, AIMessage)
            )
            transfer_msg = ToolMessage(
                content="Requirements analysis complete. Transferring to security & best practices agent.",
                tool_call_id=runtime.tool_call_id,
            )
            return Command(
                goto="security_best_practices",
                update={
                    "active_agent": "security_best_practices",
                    "current_step": "sec_n_best_practices",
                    "workflow_state": wf.model_dump(),
                    "status": WorkflowStatus.IN_PROGRESS.value,
                    "messages": [last_ai, transfer_msg],
                    "req_analyser_output": runtime.state.get("req_analyser_output", {}),
                    "sec_n_best_practices_output": runtime.state.get("sec_n_best_practices_output", {}),
                    "execution_planner_output": runtime.state.get("execution_planner_output", {}),
                    "files": runtime.state.get("files", {}),
                },
                graph=Command.PARENT,
            )

        # ----------------------------------------------------------------
        # Handoff: security_best_practices → execution_planner
        # ----------------------------------------------------------------
        @tool
        def transfer_to_execution_planner(
            runtime: ToolRuntime[None, TFPlannerState],
        ) -> Command:
            """Transfer from security & best practices to execution planner agent."""
            wf = _coerce(runtime.state)
            wf.sec_n_best_practices_complete = True
            wf.last_agent = "security_best_practices"
            wf.next_agent = "execution_planner"
            wf.current_phase = "execution_planner"

            logger.info(
                "Handoff: security_best_practices → execution_planner",
                extra={"workflow_progress": wf.get_workflow_progress()},
            )

            last_ai = next(
                msg for msg in reversed(runtime.state["messages"])
                if isinstance(msg, AIMessage)
            )
            transfer_msg = ToolMessage(
                content="Security & best practices analysis complete. Transferring to execution planner.",
                tool_call_id=runtime.tool_call_id,
            )
            return Command(
                goto="execution_planner",
                update={
                    "active_agent": "execution_planner",
                    "current_step": "execution_planner",
                    "workflow_state": wf.model_dump(),
                    "status": WorkflowStatus.IN_PROGRESS.value,
                    "messages": [last_ai, transfer_msg],
                    "req_analyser_output": runtime.state.get("req_analyser_output", {}),
                    "sec_n_best_practices_output": runtime.state.get("sec_n_best_practices_output", {}),
                    "execution_planner_output": runtime.state.get("execution_planner_output", {}),
                    "files": runtime.state.get("files", {}),
                },
                graph=Command.PARENT,
            )

        # ----------------------------------------------------------------
        # Handoff: execution_planner → END (or reroute on incomplete)
        # ----------------------------------------------------------------
        @tool
        def complete_workflow(
            runtime: ToolRuntime[None, TFPlannerState],
        ) -> Command:
            """Complete the planning workflow after execution planning.

            IMPORTANT: Only ends the graph when ALL phases are complete.
            """
            wf = _coerce(runtime.state)
            wf.execution_planner_complete = True
            wf.last_agent = "execution_planner"
            wf.next_agent = None
            wf.workflow_complete = wf.is_complete
            wf.current_phase = cast(  # type: ignore[assignment]
                Literal["req_analyser", "sec_n_best_practices", "execution_planner", "complete"],
                "complete" if wf.is_complete else (wf.next_phase or "execution_planner"),
            )

            logger.info(
                "Attempting workflow completion",
                extra={
                    "workflow_complete": wf.is_complete,
                    "workflow_progress": wf.get_workflow_progress(),
                },
            )

            last_ai = next(
                msg for msg in reversed(runtime.state["messages"])
                if isinstance(msg, AIMessage)
            )

            # If something is missing, reroute to the next incomplete phase
            if not wf.is_complete:
                next_phase = wf.next_phase or "req_analyser"
                phase_to_agent = {
                    "req_analyser": "requirements_analyser",
                    "sec_n_best_practices": "security_best_practices",
                    "execution_planner": "execution_planner",
                }
                next_agent = phase_to_agent.get(next_phase, "requirements_analyser")
                wf.next_agent = next_agent
                wf.current_phase = cast(Any, next_phase)

                logger.warning(
                    "Workflow not complete; rerouting",
                    extra={
                        "missing_phase": next_phase,
                        "to_agent": next_agent,
                        "workflow_progress": wf.get_workflow_progress(),
                    },
                )

                reroute_msg = ToolMessage(
                    content=(
                        f"Execution planning finished, but workflow is not complete. "
                        f"Routing to missing phase: {next_phase}."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
                return Command(
                    goto=next_agent,
                    update={
                        "active_agent": next_agent,
                        "current_step": next_phase,
                        "workflow_state": wf.model_dump(),
                        "status": WorkflowStatus.IN_PROGRESS.value,
                        "messages": [last_ai, reroute_msg],
                        "req_analyser_output": runtime.state.get("req_analyser_output", {}),
                        "sec_n_best_practices_output": runtime.state.get("sec_n_best_practices_output", {}),
                        "execution_planner_output": runtime.state.get("execution_planner_output", {}),
                        "files": runtime.state.get("files", {}),
                    },
                    graph=Command.PARENT,
                )

            # All phases complete — end the graph
            completion_msg = ToolMessage(
                content="Terraform planning workflow complete.",
                tool_call_id=runtime.tool_call_id,
            )
            logger.info(
                "Workflow complete; ending graph",
                extra={
                    "status": str(WorkflowStatus.COMPLETED),
                    "workflow_progress": wf.get_workflow_progress(),
                },
            )
            return Command(
                goto=END,
                update={
                    "workflow_state": wf.model_dump(),
                    "status": WorkflowStatus.COMPLETED.value,
                    "messages": [last_ai, completion_msg],
                    "req_analyser_output": runtime.state.get("req_analyser_output", {}),
                    "sec_n_best_practices_output": runtime.state.get("sec_n_best_practices_output", {}),
                    "execution_planner_output": runtime.state.get("execution_planner_output", {}),
                    "files": runtime.state.get("files", {}),
                },
                graph=Command.PARENT,
            )

        # ----------------------------------------------------------------
        # Build sub-agent graphs with handoff tools injected
        # ----------------------------------------------------------------

        self._req_analyser_agent = ReqAnalyserAgent(
            model=self.model,
            extra_tools=[transfer_to_security_best_practices],
        ).build_agent()

        self._sec_best_practices_agent = SecBestPracticesAgent(
            model=self.model,
            extra_tools=[transfer_to_execution_planner],
        ).build_agent()

        self._execution_planner_agent = ExecutionPlannerAgent(
            model=self.model_higher,
            extra_tools=[complete_workflow],
        ).build_agent()

    # -- State transforms --------------------------------------------------

    def input_transform(self, send_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the supervisor payload into ``TFPlannerState`` input."""
        messages = send_payload.get("messages") or []
        user_query = send_payload.get("user_query")

        if not user_query and messages:
            last = messages[-1]
            user_query = getattr(last, "content", None) or (
                last.get("content") if isinstance(last, dict) else None
            )

        wf_raw = send_payload.get("workflow_state")
        if isinstance(wf_raw, TFPlannerWorkflowState):
            workflow_state = wf_raw
        elif isinstance(wf_raw, dict):
            workflow_state = TFPlannerWorkflowState(**wf_raw)
        else:
            workflow_state = TFPlannerWorkflowState()

        # Normalise to start of pipeline
        workflow_state.current_phase = "req_analyser"
        workflow_state.next_agent = "requirements_analyser"
        workflow_state.last_agent = None
        workflow_state.req_analyser_complete = False
        workflow_state.sec_n_best_practices_complete = False
        workflow_state.execution_planner_complete = False
        workflow_state.workflow_complete = False

        transformed: Dict[str, Any] = {
            "messages": messages,
            "user_query": user_query or "",
            "session_id": send_payload.get("session_id"),
            "task_id": send_payload.get("task_id"),
            # Serialize to plain dict/str — LangGraph's msgpack checkpointer
            # cannot handle custom Pydantic models or Enum instances directly.
            "workflow_state": workflow_state.model_dump(),
            "status": WorkflowStatus.IN_PROGRESS.value,
            "active_agent": "requirements_analyser",
            "current_step": "req_analyser",
            "files": send_payload.get("files", {}),
        }

        logger.info(
            "input_transform complete",
            extra={
                "user_query_preview": transformed["user_query"][:120],
                "session_id": transformed.get("session_id"),
                "task_id": transformed.get("task_id"),
                "workflow_progress": workflow_state.get_workflow_progress(),
            },
        )

        return transformed

    def output_transform(
        self,
        agent_state: Union[Dict[str, Any], Any],
        parent_files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Transform ``TFPlannerState`` output → ``CompiledSubAgent`` return value.

        This is the **third** side of the planner's three-way bridge:

        1. ``input_transform``  — SupervisorState/deep-agent state → TFPlannerState
        2. ``planner_graph.invoke`` — runs the 3-phase planning pipeline
        3. ``output_transform`` — TFPlannerState → deep-agent state update

        ``CompiledSubAgent`` contract
        (https://docs.langchain.com/oss/python/deepagents/subagents#compiledsubagent):

        - ``result["messages"][-1]``  MUST be an ``AIMessage`` — this is what
          the coordinator LLM sees as the subagent’s single reply. We build a
          concise, structured-text summary so the coordinator’s context stays
          lean (the full data lives in the virtual FS).
        - ``result["files"]``  (optional) is merged back into the deep-agent
          virtual FS so downstream subagents (tf-skill-builder, tf-generator)
          can ``read_file`` the plan outputs without the coordinator carrying
          the raw JSON in its message history.

        Args:
            agent_state: Final ``TFPlannerState`` dict (or Pydantic model) from
                         ``planner_graph.invoke()``.
            parent_files: Existing virtual-FS files from the deep-agent’s
                          state, passed in by ``_planner_wrapper`` so they are
                          preserved (skills, memories) alongside the newly-generated
                          skill files.
        """
        if hasattr(agent_state, "model_dump"):
            agent_state = agent_state.model_dump()  # type: ignore[union-attr]

        wf_raw = agent_state.get("workflow_state")
        if isinstance(wf_raw, TFPlannerWorkflowState):
            wf = wf_raw
        elif isinstance(wf_raw, dict):
            wf = TFPlannerWorkflowState(**wf_raw)
        else:
            wf = TFPlannerWorkflowState()

        req_out  = agent_state.get("req_analyser_output") or {}
        sec_out  = agent_state.get("sec_n_best_practices_output") or {}
        exec_out = agent_state.get("execution_planner_output") or {}

        # ── Pass-through virtual FS files ─────────────────────────────────
        # write_service_skills_tool already serialised everything it needed
        # into agent_state["files"] (skill MDs, reference docs, etc.) during
        # graph execution. We do NOT re-serialise or add more files here —
        # that would duplicate work and risk overwriting skill files.
        #
        # We only merge parent_files (coordinator seed files: memories, context)
        # so they are preserved alongside the newly-generated skill files.
        planner_files: Dict[str, Any] = agent_state.get("files") or {}
        merged_files: Dict[str, Any] = {**(parent_files or {}), **planner_files}

        # ── Build concise AIMessage summary for the coordinator ───────────
        # The coordinator LLM sees ONLY this message as the planner's output.
        # Structured but brief — full details are in the VFS skill files.
        summary_lines = ["## tf-planner: Planning complete"]

        if req_out:
            service = req_out.get("service") or req_out.get("aws_service", "")
            summary_lines.append(f"\n**Requirements analysed** — Service: `{service}`")
            tf_files = req_out.get("terraform_files") or []
            if tf_files:
                summary_lines.append(
                    "  File set: " + ", ".join(f"`{f}`" for f in tf_files)
                )

        if sec_out:
            summary_lines.append("\n**Security & best practices** — review complete.")
            guidelines = (
                sec_out.get("guidelines") or sec_out.get("recommendations") or []
            )
            if guidelines:
                summary_lines.append(
                    "  Key rules: " + "; ".join(str(g) for g in guidelines[:3])
                )

        if exec_out:
            summary_lines.append("\n**Execution plan** — ready.")
            steps = exec_out.get("steps") or exec_out.get("phase_order") or []
            if steps:
                summary_lines.append(
                    "  Phases: " + " → ".join(str(s) for s in steps)
                )

        # List skill files the planner wrote — so the coordinator knows
        # which skills are available without carrying their full content.
        skill_keys = sorted(
            k for k in merged_files if k.startswith("/skills") or k.startswith("skills")
        )
        if skill_keys:
            summary_lines.append(
                "\n**Skill files written to virtual FS:**\n"
                + "\n".join(f"  - `{k}`" for k in skill_keys)
            )
            summary_lines.append(
                "\nDownstream subagents (tf-skill-builder, tf-generator) "
                "will load these skills automatically via the deep-agent VFS."
            )

        summary_text = "\n".join(summary_lines)

        logger.info(
            "output_transform: TFPlannerState → deep-agent CompiledSubAgent payload",
            extra={
                "workflow_complete": wf.is_complete,
                "workflow_progress": wf.get_workflow_progress(),
                "vfs_keys": list(merged_files.keys()),
                "skill_count": len(skill_keys),
                "summary_preview": summary_text[:200],
            },
        )

        return {
            "messages": [AIMessage(content=summary_text)],
            "files": merged_files,
        }

    # -- Graph builder -----------------------------------------------------

    def build_graph(self) -> Any:  # Returns CompiledStateGraph
        """Build the LangGraph subgraph with Command-based handoff routing."""
        logger.info("Building PlannerSupervisorAgent graph")

        def route_initial(state: TFPlannerState) -> str:
            existing = state.get("workflow_state")
            wf: Optional[TFPlannerWorkflowState] = None
            if isinstance(existing, dict):
                wf = TFPlannerWorkflowState(**existing)
            elif isinstance(existing, TFPlannerWorkflowState):
                wf = existing

            if wf and wf.next_agent:
                return wf.next_agent

            return cast(str, state.get("active_agent", "requirements_analyser"))

        def route_after_agent(state: TFPlannerState) -> str:
            existing = state.get("workflow_state")
            wf: Optional[TFPlannerWorkflowState] = None
            if isinstance(existing, dict):
                wf = TFPlannerWorkflowState(**existing)
            elif isinstance(existing, TFPlannerWorkflowState):
                wf = existing

            # End when workflow is complete
            if wf and wf.is_complete:
                return "__end__"

            # Safety net (official docs pattern): if the last message is an
            # AIMessage without tool_calls, the agent finished without calling
            # a handoff tool — end the graph to prevent infinite loops.
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    return "__end__"

            # Route to next agent
            if wf and wf.next_agent:
                return wf.next_agent

            return cast(str, state.get("active_agent", "requirements_analyser"))

        # Build the graph
        builder = StateGraph(TFPlannerState)

        builder.add_node("requirements_analyser", self._req_analyser_agent)
        builder.add_node("security_best_practices", self._sec_best_practices_agent)
        builder.add_node("execution_planner", self._execution_planner_agent)

        # Entry routing
        builder.add_conditional_edges(
            START,
            route_initial,
            ["requirements_analyser", "security_best_practices", "execution_planner"],
        )

        # Each agent can route to any other or END
        all_targets = [
            "requirements_analyser",
            "security_best_practices",
            "execution_planner",
            END,
        ]
        for agent_node in ["requirements_analyser", "security_best_practices", "execution_planner"]:
            builder.add_conditional_edges(agent_node, route_after_agent, all_targets)

        return builder.compile(checkpointer=self.memory)


# ============================================================================
# Factory function
# ============================================================================

def create_planner_supervisor_agent(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    name: str = "planner_supervisor_agent",
    memory: Optional[MemorySaver] = None,
) -> PlannerSupervisorAgent:
    """
    Create a planner supervisor agent.

    Args:
        config: Configuration object.
        custom_config: Custom configuration dict.
        name: Agent name for routing.
        memory: MemorySaver checkpointer.

    Returns:
        PlannerSupervisorAgent instance.
    """
    return PlannerSupervisorAgent(
        config=config,
        custom_config=custom_config,
        name=name,
        memory=memory,
    )
