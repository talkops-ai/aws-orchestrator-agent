"""
Supervisor state schema for the AWS Orchestrator Agent.

Minimal state that the supervisor graph uses. The deep-agent coordinator
manages its own internal state — these fields are only for the A2A adapter
layer (session tracking, workflow progress, HITL coordination).
"""

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain.agents import AgentState
from typing_extensions import NotRequired

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Workflow state (mirrors CI-supervisor pattern)
# ---------------------------------------------------------------------------

class SupervisorWorkflowState(BaseModel):
    """Tracks which phases have completed within the supervisor pipeline."""

    current_phase: Optional[str] = None
    """Active phase: 'terraform_generation', 'terraform_update', etc."""

    last_agent: Optional[str] = None
    """Last sub-agent / deep agent that ran."""

    next_agent: Optional[str] = None
    """Next sub-agent to run (set by routing logic)."""

    terraform_complete: bool = False
    """Whether the Terraform deep agent finished its pipeline."""

    workflow_complete: bool = False
    """Whether the entire supervisor workflow is done."""

    def get_workflow_progress(self) -> Dict[str, Any]:
        """Return a snapshot of workflow progress for logging."""
        return {
            "current_phase": self.current_phase,
            "last_agent": self.last_agent,
            "terraform_complete": self.terraform_complete,
            "workflow_complete": self.workflow_complete,
        }


# ---------------------------------------------------------------------------
# Supervisor state
# ---------------------------------------------------------------------------

class SupervisorState(AgentState):
    """State schema for the AWS Orchestrator supervisor agent.

    Uses ``Annotated`` types with LangGraph reducers for proper concurrent
    update handling. ``NotRequired`` fields are populated during execution.
    """

    # ── Required at invocation ─────────────────────────────────────────
    messages: Annotated[List[AnyMessage], add_messages]

    # ── Core identifiers ──────────────────────────────────────────────
    user_query: NotRequired[str]
    session_id: NotRequired[str]
    task_id: NotRequired[str]

    # ── Runtime context (injected into deep-agent config) ─────────────
    context: NotRequired[Dict[str, Any]]

    # ── Workflow tracking ─────────────────────────────────────────────
    status: NotRequired[str]
    current_phase: NotRequired[str]
    workflow_state: NotRequired[Dict[str, Any]]
    workflow_complete: NotRequired[bool]

    # ── Deep agent outputs ────────────────────────────────────────────
    terraform_output: NotRequired[Dict[str, Any]]
    """Structured output from the TFCoordinator deep agent."""

    # ── HITL ──────────────────────────────────────────────────────────
    pending_feedback_requests: NotRequired[Dict[str, Any]]
    """Interrupt payload for human-in-the-loop approval requests."""