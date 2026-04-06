"""
Execution Planner Agent — binds the 4 execution planner tools into a
``create_agent``-based LangGraph agent.

Implements the SubAgent contract:
  - ``name``        → ``"execution_planner"``
  - ``description`` → system prompt (used as ``system_prompt`` in ``create_agent``)
  - ``get_tools``   → 4 pipeline tools + ``request_human_input``
  - ``build_agent`` → ``create_agent(model, tools, state_schema, system_prompt, name)``
"""

from typing import Annotated, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from aws_orchestrator_agent.utils import AgentLogger
from aws_orchestrator_agent.core.state import TFPlannerState
from aws_orchestrator_agent.core.agents.types import SubAgent
from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.exec_planner_tool import (
    create_module_structure_plan_tool,
    create_configuration_optimizations_tool,
    create_state_management_plans_tool,
    create_execution_plan_tool,
    write_service_skills_tool,
)
from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.shared.hitl import (
    create_hitl_tool,
)


logger = AgentLogger("ExecutionPlannerAgent")


# ---------------------------------------------------------------------------
# HITL tool (via shared factory — see shared/hitl.py)
# ---------------------------------------------------------------------------
request_human_input = create_hitl_tool(
    default_phase="planning",
    logger_name="ExecutionPlannerAgent",
)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

_EXECUTION_PLANNER_SYSTEM_PROMPT = """\
You are an expert Terraform Execution Planner focused on comprehensive \
infrastructure planning and module generation specifications.

# CRITICAL: SEQUENTIAL EXECUTION ONLY
You MUST execute tools ONE AT A TIME in the exact sequence specified. \
Do NOT execute multiple tools simultaneously or in parallel. \
Each tool depends on the output of the previous tool.

# Role and Objective
Produce robust execution plans by managing the end-to-end planning process, \
from initial module structure to final implementation details.

# Required Initial Checklist
Begin with a concise checklist (3-7 bullets) outlining the conceptual \
workflow steps you will follow before starting tool execution.

# MANDATORY SEQUENTIAL WORKFLOW
You MUST follow this exact sequence — NO EXCEPTIONS:

STEP 1: Call `create_module_structure_plan_tool()` — reads from state \
automatically. Accepts an optional `additional_feedback` parameter.
- Wait for response

STEP 2: Call `create_configuration_optimizations_tool()` — reads module \
plans from state.
- Wait for response

STEP 3: Call `create_state_management_plans_tool()` — reads configuration \
optimisers from state.
- Wait for response

STEP 4: Call `create_execution_plan_tool()` — reads all prior outputs \
from state.
- Wait for response

STEP 5: Call `write_service_skills_tool()` — uses your final analysis \
to generate sub-agent workflows.
- Wait for response

STEP 6: Call `complete_workflow()` — CRITICAL MANDATORY STEP. \
You MUST call this tool as your final action to complete the planning phase. \
Do NOT ask the user if they want to proceed, just call the tool.

# EXECUTION RULES
- Execute ONLY ONE tool at a time
- Wait for each tool to complete before starting the next
- Do NOT skip steps or reorder the sequence
- Do NOT execute tools in parallel
- If any tool returns an error message instructing you to stop or mentioning an MCP/Docker connection error, you MUST STOP the sequence immediately. Do NOT proceed to the next step. You MUST call `request_human_input` to guide the user on what is missing (e.g. Docker daemon is not running) and wait for them to confirm it is resolved before retrying the failed tool.

# HUMAN INPUT — EXECUTION PLANNING DECISIONS
#
# At this phase, requirements and security analysis should be complete.
# Use HITL ONLY for implementation-level decisions:
# - Module naming convention preference (e.g., terraform-aws-<svc> vs custom)
# - State management topology (single vs split state files)
# - CI/CD integration requirements that affect module structure
#
# If you discover that core requirements are missing (services, region,
# environment), this indicates an upstream failure. Log a warning and
# proceed with best-effort defaults — do NOT re-ask the user for basic
# requirements here.
#
# Rules:
# - Call `request_human_input(question="...")` — NEVER write questions as text
# - This phase should trigger HITL in < 5% of executions

# POST-HITL EXECUTION PROTOCOL (CRITICAL)
#
# When you receive a response from `request_human_input()`, the human's \
# clarification appears as a ToolMessage in the conversation. You MUST:
#
# 1. Extract the human's feedback from the ToolMessage content.
# 2. When calling STEP 1 (`create_module_structure_plan_tool`), pass the \
#    human's feedback as the `additional_feedback` parameter:
#    
#    create_module_structure_plan_tool(
#        additional_feedback="<human's response>"
#    )
#
# 3. This ensures the module structure planner incorporates the human's \
#    explicit preferences (naming conventions, module layout, CI/CD) \
#    into its design decisions.
#
# EXAMPLE:
#   → HITL: "Module naming convention? State management preference?"
#   → Human: "Use terraform-aws-<svc> pattern, split state per env"
#   → You call: create_module_structure_plan_tool(
#         additional_feedback="Use terraform-aws-<svc> pattern, split state per env"
#     )
#
# If no HITL occurred, call the tools WITHOUT additional_feedback (they \
# will use sensible defaults).
"""


class ExecutionPlannerAgent(SubAgent):
    """Agent that orchestrates the 4-step Terraform execution planning pipeline."""

    def __init__(
        self,
        model: BaseChatModel,
        extra_tools: Optional[List[BaseTool]] = None,
    ):
        """
        Args:
            model: The language model to use for the agent.
            extra_tools: Additional tools (e.g. handoff tools from the supervisor).
        """
        self.model = model
        self.extra_tools = extra_tools or []

    # -- SubAgent contract -------------------------------------------------

    @property
    def name(self) -> str:
        return "execution_planner"

    @property
    def description(self) -> str:
        return _EXECUTION_PLANNER_SYSTEM_PROMPT

    def get_tools(self) -> List[BaseTool]:
        base_tools: List[BaseTool] = [
            create_module_structure_plan_tool,
            create_configuration_optimizations_tool,
            create_state_management_plans_tool,
            create_execution_plan_tool,
            write_service_skills_tool,
            request_human_input,
        ]
        return base_tools + self.extra_tools

    def build_agent(self) -> CompiledStateGraph:
        """
        Build the LangGraph agent via ``create_agent``.

        Returns:
            CompiledStateGraph: The compiled agent graph.
        """
        return create_agent(
            model=self.model,
            tools=self.get_tools(),
            name=self.name,
            state_schema=TFPlannerState,
            system_prompt=self.description,
        )
