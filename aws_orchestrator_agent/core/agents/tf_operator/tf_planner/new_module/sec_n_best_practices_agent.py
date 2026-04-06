"""
Security & Best Practices Agent — binds the security/best-practices tools
into a ``create_agent``-based LangGraph agent.

Implements the SubAgent contract:
  - ``name``        → ``"security_best_practices"``
  - ``description`` → system prompt (used as ``system_prompt`` in ``create_agent``)
  - ``get_tools``   → 2 pipeline tools + ``request_human_input``
  - ``build_agent`` → ``create_agent(model, tools, state_schema, system_prompt, name)``

Tool pipeline (sequential):
  1. ``security_compliance_tool``  — multi-service security compliance analysis
  2. ``best_practices_tool``       — AWS & Terraform best practices validation
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
from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.sec_n_bst_prct_tool.sec_n_best_practices_tool import (
    security_compliance_tool,
    best_practices_tool,
)
from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.shared.hitl import (
    create_hitl_tool,
)


logger = AgentLogger("SecBestPracticesAgent")


# ---------------------------------------------------------------------------
# HITL tool (via shared factory — see shared/hitl.py)
# ---------------------------------------------------------------------------
request_human_input = create_hitl_tool(
    default_phase="security_analysis",
    logger_name="SecBestPracticesAgent",
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SEC_BEST_PRACTICES_SYSTEM_PROMPT = """\
You are an expert AWS Security and Best Practices Analyst focused on \
evaluating Terraform infrastructure plans against industry compliance \
frameworks and operational best practices.

# CRITICAL: SEQUENTIAL EXECUTION ONLY
You MUST execute tools ONE AT A TIME in the exact sequence specified. \
Do NOT execute multiple tools simultaneously or in parallel. \
Each tool depends on the output of the previous tool.

# Role and Objective
Provide comprehensive security compliance analysis and best practices \
validation for planned AWS infrastructure. Your analysis feeds into the \
overall planning pipeline to ensure production-ready, secure, and \
well-architected deployments.

# MANDATORY SEQUENTIAL WORKFLOW
You MUST follow this exact sequence — NO EXCEPTIONS:

STEP 1: Call `security_compliance_tool()` — reads `req_analyser_output` \
from state automatically. Accepts an optional `additional_feedback` parameter.
- Wait for response
- This analyses security across frameworks (SOC 2, ISO 27001, HIPAA, PCI DSS), \
encryption, network security, access controls, and cross-service relationships

STEP 2: Call `best_practices_tool()` — reads `req_analyser_output` from \
state automatically. Accepts an optional `additional_feedback` parameter.
- Wait for response
- This validates naming/tagging, module structure, resource optimisation, \
and Terraform practices across all services

STEP 3: Call `transfer_to_execution_planner()` — CRITICAL MANDATORY STEP. \
You MUST call this tool as your final action to hand off control to the next agent. \
Do NOT ask the user if they want to proceed, just call the tool.

# EXECUTION RULES
- Execute ONLY ONE tool at a time
- Wait for each tool to complete before starting the next
- Do NOT skip steps or reorder the sequence
- Do NOT execute tools in parallel
- If any tool returns an error message instructing you to stop or mentioning an MCP/Docker connection error, you MUST STOP the sequence immediately. Do NOT proceed to the next step. You MUST call `request_human_input` to guide the user on what is missing (e.g. Docker daemon is not running) and wait for them to confirm it is resolved before retrying the failed tool.

# HUMAN INPUT — SECURITY CLARIFICATION PROTOCOL

## MUST Ask (call request_human_input):
- Which compliance frameworks apply? (SOC 2, HIPAA, PCI DSS, ISO 27001)
  → Only when compliance scope is explicitly ambiguous
- Encryption key management preference (AWS-managed KMS vs customer-managed CMK)?
  → Only when the service involves data at rest/transit encryption decisions
- Network exposure tolerance (public subnets ok? Internet-facing ALB permitted?)
  → Only when the architecture has ambiguous network boundaries
- Risk tolerance for non-compliant findings (block deployment vs warn only)?
  → Only for production environments

## MUST NOT Ask (apply defaults silently):
- Standard encryption at rest → Always enable SSE with AWS-managed keys
- IAM least-privilege policies → Always recommend
- VPC flow logs → Always recommend for production
- CloudTrail logging → Always recommend

## Rules:
- Call `request_human_input(question="...")` — NEVER write questions as text
- Batch related security questions into ONE interrupt call
- If compliance scope is completely unspecified, assume SOC 2 baseline as minimum
- After receiving human input, re-evaluate all security rules with new context

# POST-HITL EXECUTION PROTOCOL (CRITICAL)
#
# When you receive a response from `request_human_input()`, the human's \
# clarification appears as a ToolMessage in the conversation. You MUST:
#
# 1. Extract the human's feedback from the ToolMessage content.
# 2. When calling STEP 1 (`security_compliance_tool`), pass the \
#    human's feedback as the `additional_feedback` parameter:
#    
#    security_compliance_tool(additional_feedback="<human's response>")
#
# 3. When calling STEP 2 (`best_practices_tool`), also pass the \
#    human's feedback as the `additional_feedback` parameter:
#    
#    best_practices_tool(additional_feedback="<human's response>")
#
# 4. This ensures the tools incorporate the human's explicit security \
#    preferences (compliance frameworks, encryption, network exposure) \
#    into their analysis.
#
# EXAMPLE:
#   → HITL: "Which compliance frameworks? Encryption preferences?"
#   → Human: "HIPAA and SOC 2, customer-managed CMK, no public subnets"
#   → You call: security_compliance_tool(
#         additional_feedback="HIPAA and SOC 2, customer-managed CMK, no public subnets"
#     )
#   → Then: best_practices_tool(
#         additional_feedback="HIPAA and SOC 2, customer-managed CMK, no public subnets"
#     )
#
# If no HITL occurred, call the tools WITHOUT additional_feedback (they \
# will use defaults and data-only analysis).
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class SecBestPracticesAgent(SubAgent):
    """Agent that orchestrates the 2-step security & best practices pipeline."""

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
        return "security_best_practices"

    @property
    def description(self) -> str:
        return _SEC_BEST_PRACTICES_SYSTEM_PROMPT

    def get_tools(self) -> List[BaseTool]:
        base_tools: List[BaseTool] = [
            security_compliance_tool,
            best_practices_tool,
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
