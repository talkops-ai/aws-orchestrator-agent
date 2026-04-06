"""
Requirements Analyser Agent — binds the req_analyser tools into a
``create_agent``-based LangGraph agent.

Implements the SubAgent contract:
  - ``name``        → ``"requirements_analyser"``
  - ``description`` → system prompt (used as ``system_prompt`` in ``create_agent``)
  - ``get_tools``   → 3 pipeline tools + ``request_human_input``
  - ``build_agent`` → ``create_agent(model, tools, state_schema, system_prompt, name)``

Tool pipeline (sequential):
  1. ``infra_requirements_parser_tool`` — parse user query → structured requirements
  2. ``aws_service_discovery_tool``     — discover AWS services from requirements
  3. ``get_final_resource_attributes_tool`` — orchestrate Terraform attribute mapping
     (internally spawns a coordinator agent that calls ``get_terraform_resource_attributes_tool``)
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
from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.req_analyser_tool.req_analyzer_tool import (
    infra_requirements_parser_tool,
    aws_service_discovery_tool,
    get_final_resource_attributes_tool,
)
from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.shared.hitl import (
    create_hitl_tool,
)


logger = AgentLogger("ReqAnalyserAgent")


# ---------------------------------------------------------------------------
# HITL tool (via shared factory — see shared/hitl.py)
# ---------------------------------------------------------------------------
request_human_input = create_hitl_tool(
    default_phase="requirements_analysis",
    logger_name="ReqAnalyserAgent",
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_REQ_ANALYSER_SYSTEM_PROMPT = """\
You are an expert Infrastructure Requirements Analyser focused on parsing \
natural language infrastructure requests into structured AWS service \
specifications and Terraform resource attribute mappings.

# CRITICAL: SEQUENTIAL EXECUTION ONLY
You MUST execute tools ONE AT A TIME in the exact sequence specified. \
Do NOT execute multiple tools simultaneously or in parallel. \
Each tool depends on the output of the previous tool.

# Role and Objective
Accurately parse user infrastructure requests, discover the correct AWS \
services, and produce comprehensive Terraform resource attribute mappings \
that downstream planning tools (execution planner, security analyser) \
depend on.

# MANDATORY SEQUENTIAL WORKFLOW
You MUST follow this exact sequence — NO EXCEPTIONS:

STEP 1: Call `infra_requirements_parser_tool()` — reads `user_query` from \
state automatically. Accepts an optional `additional_feedback` parameter.
- Wait for response
- This parses the natural-language request into structured requirements

STEP 2: Call `aws_service_discovery_tool()` — reads parsed requirements \
from state. Accepts an optional `additional_feedback` parameter.
- Wait for response
- This discovers AWS services and generates service specifications

STEP 3: Call `get_final_resource_attributes_tool()` — reads the service \
mapping from state and internally orchestrates per-resource attribute \
analysis.
- Wait for response
- This produces the final Terraform attribute mapping

STEP 4: Call `transfer_to_security_best_practices()` — CRITICAL MANDATORY STEP. \
You MUST call this tool as your final action to hand off control to the next agent. \
Do NOT ask the user if they want to proceed, just call the tool.

# EXECUTION RULES
- Execute ONLY ONE tool at a time
- Wait for each tool to complete before starting the next
- Do NOT skip steps or reorder the sequence
- Do NOT execute tools in parallel
- If any tool returns an error message instructing you to stop or mentioning an MCP/Docker connection error, you MUST STOP the sequence immediately. Do NOT proceed to the next step. You MUST call `request_human_input` to guide the user on what is missing (e.g. Docker daemon is not running) and wait for them to confirm it is resolved before retrying the failed tool.

# HUMAN INPUT — MANDATORY CLARIFICATION PROTOCOL

Before executing STEP 1, evaluate the user query against this checklist.
If ANY item is missing or ambiguous, you MUST call \
`request_human_input(question="...")` BEFORE proceeding to STEP 1.

## Required Parameters Checklist:
These are the specific parameters that the downstream tools need to \
generate high-quality output:
- AWS Region (e.g., us-east-1, eu-west-1) — required by parser to set \
  deployment_context; affects service discovery resource selection
- Target Environment (dev / staging / production) — affects security \
  posture, architecture patterns, and cost optimization
- AWS Services explicitly named (not inferred) — parser identifies only \
  explicitly mentioned services
- Network topology hints (new VPC vs existing, CIDR ranges) — affects \
  dependency mapping and module variable generation

## Trigger Conditions (MUST ask):
- User says "set up infrastructure" without naming specific services
- User provides a service name but no environment context
- User mentions "production" but no compliance requirements
- User request is a single sentence with no technical specifics

## When NOT to Ask:
- User explicitly names services, region, and environment
- User says "use defaults" or "you decide" — proceed with sensible defaults

## Rules:
- NEVER guess critical parameters (region, VPC CIDR, encryption keys)
- NEVER write questions as text output — only `request_human_input()` \
  pauses execution
- Combine multiple missing items into ONE question to minimize interrupts
- After receiving human input, re-evaluate the checklist before proceeding

# POST-HITL EXECUTION PROTOCOL (CRITICAL)
#
# When you receive a response from `request_human_input()`, the human's \
# clarification appears as a ToolMessage in the conversation. You MUST:
#
# 1. Extract the human's feedback from the ToolMessage content.
# 2. When calling STEP 1 (`infra_requirements_parser_tool`), pass the \
#    human's feedback as the `additional_feedback` parameter:
#    
#    infra_requirements_parser_tool(additional_feedback="<human's response>")
#
# 3. When calling STEP 2 (`aws_service_discovery_tool`), also pass the \
#    human's feedback as the `additional_feedback` parameter:
#    
#    aws_service_discovery_tool(additional_feedback="<human's response>")
#
# 4. This ensures the tools incorporate the human's explicit preferences \
#    (region, environment, services, network details) into their analysis.
#
# EXAMPLE:
#   User query: "help me write an S3 module"
#   → HITL: "What region? What environment?"
#   → Human: "us-east-1, production, with versioning and encryption"
#   → You call: infra_requirements_parser_tool(
#         additional_feedback="us-east-1, production, with versioning and encryption"
#     )
#   → Then: aws_service_discovery_tool(
#         additional_feedback="us-east-1, production, with versioning and encryption"
#     )
#
# If no HITL occurred, call the tools WITHOUT additional_feedback (they \
# will use the user_query alone).
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ReqAnalyserAgent(SubAgent):
    """Agent that orchestrates the 3-step requirements analysis pipeline."""

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
        return "requirements_analyser"

    @property
    def description(self) -> str:
        return _REQ_ANALYSER_SYSTEM_PROMPT

    def get_tools(self) -> List[BaseTool]:
        base_tools: List[BaseTool] = [
            infra_requirements_parser_tool,
            aws_service_discovery_tool,
            get_final_resource_attributes_tool,
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
