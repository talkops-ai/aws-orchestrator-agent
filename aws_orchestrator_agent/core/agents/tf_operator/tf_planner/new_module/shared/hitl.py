"""
Shared Human-in-the-Loop (HITL) module for the tf_planner pipeline.

Provides:
- HITLInterruptPayload: Pydantic schema for interrupt payloads
- create_hitl_tool: Factory function to create phase-specific HITL tools

Architecture Decision:
  We use a factory pattern rather than LangGraph middleware because:
  1. Sub-agents use create_agent() with internal graphs — middleware at the 
     parent level can't intercept tool calls inside nested agent graphs
  2. Each phase has different trigger conditions 
  3. The existing request_human_input pattern is already established
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt
from aws_orchestrator_agent.core.state import TFPlannerState
from aws_orchestrator_agent.utils import AgentLogger


class HITLInterruptPayload(BaseModel):
    """Standardized schema for all HITL interrupt payloads.
    
    This schema is what the client/frontend receives when the pipeline
    pauses for human input. Every field must be JSON-serializable.
    
    The client uses this to:
    - Render the question to the user
    - Show context about which phase is asking
    - Optionally present structured options
    - Track via session_id/task_id
    """
    status: str = Field(
        default="input_required",
        description="Always 'input_required' when interrupt is active"
    )
    session_id: str = Field(
        description="Session identifier for traceability"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task identifier — MUST be populated from state"
    )
    question: str = Field(
        description="The question to present to the human operator"
    )
    active_phase: str = Field(
        description="Which pipeline phase triggered this interrupt"
    )
    tool_name: str = Field(
        default="request_human_input",
        description="Tool that triggered the interrupt"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context for why input is needed"
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="Optional structured choices for the UI to render"
    )
    required: bool = Field(
        default=True,
        description="Whether the user MUST answer to proceed"
    )


def create_hitl_tool(default_phase: str, logger_name: str):
    """Factory that creates a phase-specific request_human_input tool.
    
    Args:
        default_phase: Fallback phase name if workflow_state.current_phase 
                       is not available (e.g., "requirements_analysis",
                       "security_analysis", "planning")
        logger_name: Name for the AgentLogger instance
        
    Returns:
        A @tool-decorated function that triggers interrupt() with a 
        standardized HITLInterruptPayload.
        
    CRITICAL RULES (from LangGraph docs):
        - Do NOT wrap interrupt() in try/except — it raises GraphInterrupt
        - interrupt() payload MUST be JSON-serializable
        - Side effects before interrupt() MUST be idempotent (node restarts
          from the beginning on resume)
    """
    _logger = AgentLogger(logger_name)

    @tool
    def request_human_input(
        question: str,
        runtime: ToolRuntime[None, TFPlannerState],
        context: Optional[str] = None,
    ) -> Command:
        """Request human input during workflow execution.

        **CRITICAL: This is the ONLY way to request human input.
        DO NOT write questions as text output — only tool calls
        can trigger the human-in-the-loop interrupt.**

        Use when:
        - You need clarification on ambiguous requirements
        - You need approval for a decision
        - You need human input to proceed

        Args:
            question: The question or request for the human
            context: Optional context about why feedback is needed
        """
        session_id = runtime.state.get("session_id", "unknown")
        task_id = runtime.state.get("task_id", "unknown")

        # Safely extract current phase from workflow state
        phase = default_phase
        wf_state = runtime.state.get("workflow_state")
        if wf_state:
            if isinstance(wf_state, dict):
                phase = wf_state.get("current_phase", default_phase)
            elif hasattr(wf_state, "current_phase"):
                phase = wf_state.current_phase

        # Build validated payload using Pydantic schema
        payload = HITLInterruptPayload(
            session_id=session_id,
            task_id=task_id,
            question=question,
            active_phase=phase,
            context=context,
        )

        # Wrap in dict for backward compatibility with existing consumers
        interrupt_payload = {
            "pending_feedback_requests": payload.model_dump()
        }

        _logger.info(
            "Requesting human feedback",
            extra={
                "phase": phase,
                "question_preview": question[:100],
                "context": context,
                "session_id": session_id,
                "task_id": task_id,
            },
        )

        # CRITICAL: Do NOT wrap this in try/except
        # interrupt() raises GraphInterrupt, which must propagate
        human_response = interrupt(interrupt_payload)
        human_response_str = str(human_response) if human_response else ""

        return Command(update={
            "messages": [
                ToolMessage(
                    content=(
                        f"User provided additional requirements: {human_response_str}. "
                        "Check if you have all the information to proceed."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    return request_human_input
