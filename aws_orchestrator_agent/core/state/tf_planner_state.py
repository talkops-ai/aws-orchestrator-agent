"""
Terraform Planner State.
This module contains the state management for the Terraform Planner workflow.
"""

from typing import Annotated, Dict, List, Optional, Any, Literal, Union
from operator import add
from enum import Enum
from datetime import datetime, timezone
from typing_extensions import NotRequired

from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain.agents import AgentState



# ============================================================================
# Shared Pydantic Models
# ============================================================================

ApprovalDecision = Literal["pending", "approved", "rejected", "modified"]

class TFPlannerWorkflowState(BaseModel):
    """Workflow state tracking for CI pipeline workflow."""
    current_phase: Literal[
        "req_analyser",
        "sec_n_best_practices",
        "execution_planner",
        "complete",
    ] = Field(default="req_analyser", description="Current workflow phase")
    
    req_analyser_complete: bool = Field(default=False, description="req_analyser phase complete")
    sec_n_best_practices_complete: bool = Field(default=False, description="sec_n_best_practices phase complete")
    execution_planner_complete: bool = Field(default=False, description="execution_planner phase complete")

    # Workflow control
    workflow_complete: bool = Field(default=False, description="Overall workflow complete")
    last_agent: Optional[str] = Field(default=None, description="Last agent that completed")
    next_agent: Optional[str] = Field(default=None, description="Next agent to invoke")
    
    # Workflow metadata
    workflow_id: str = Field(default="", description="Unique workflow identifier")
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc), description="Workflow start time")

    @property
    def is_complete(self) -> bool:
        """Check if all required phases are complete."""
        return all([
            self.req_analyser_complete,
            self.sec_n_best_practices_complete,
            self.execution_planner_complete,
        ])
    
    @property
    def next_phase(self) -> Optional[str]:
        """Determine the next phase based on completion status."""
        if not self.req_analyser_complete:
            return "req_analyser"
        elif not self.sec_n_best_practices_complete:
            return "sec_n_best_practices"
        elif not self.execution_planner_complete:
            return "execution_planner"
        else:
            return "complete"
    
    def set_phase_complete(self, phase: str) -> None:
        """Mark a specific phase as complete and update state."""
        if phase == "req_analyser":
            self.req_analyser_complete = True
            self.current_phase = "req_analyser"
        elif phase == "sec_n_best_practices":
            self.sec_n_best_practices_complete = True
            self.current_phase = "sec_n_best_practices"
        elif phase == "execution_planner":
            self.execution_planner_complete = True
            self.current_phase = "execution_planner"
        else:
            self.workflow_complete = True
            self.current_phase = "complete"
    
        # Check if workflow is complete
        if self.is_complete:
            self.workflow_complete = True
            self.current_phase = "complete"
    

    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get current workflow progress for monitoring."""
        return {
            "current_phase": self.current_phase,
            "req_analyser_complete": self.req_analyser_complete,
            "sec_n_best_practices_complete": self.sec_n_best_practices_complete,
            "execution_planner_complete": self.execution_planner_complete,
        }

class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    HUMAN_APPROVAL = "human_approval"
    INPUT_REQUIRED = "input_required"
    ROLLED_BACK = "rolled_back"
    INTERRUPTED = "interrupted"



class ApprovalStatus(BaseModel):
    """Human approval status"""
    status: ApprovalDecision
    reviewer: Optional[str] = None
    comments: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="ISO format timestamp")




class ErrorContext(BaseModel):
    """Error tracking context"""
    error_type: str
    error_message: str
    failed_agent: str
    retry_count: int = 0
    max_retries: int = 3
    recoverable: bool = True
    stack_trace: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ============================================================================
# Main CI Pipeline State
# ============================================================================

class TFPlannerState(AgentState):
    """
    Complete CI pipeline state for LangGraph workflow.
    
    """
    
    # ==================== Message History ====================
    messages: Annotated[List[AnyMessage], add_messages]
    """
    LLM conversation history (HumanMessage, AIMessage, ToolMessage).
    Reducer: add_messages (LangGraph message deduplication)
    """
    
    # ==================== Workflow Identification ====================
    user_query: NotRequired[str]
    """User query that triggered the workflow"""
    
    session_id: NotRequired[str] 
    task_id: NotRequired[str]
    """Unique task ID for this workflow session"""
    
    # ==================== Handoff State (CRITICAL) ====================
    workflow_state: NotRequired[Union["TFPlannerWorkflowState", Dict[str, Any]]]
    """
    Current workflow step - controls handoff routing.
    Accepts either a TFPlannerWorkflowState Pydantic model or a plain dict
    so that both ``input_transform`` (which writes the model object) and
    handoff tools (which write ``wf.model_dump()`` dicts) are accepted
    without Pydantic serialization warnings.
    Values: "req_analyser" | "sec_n_best_practices" | "execution_planner"
    Reducer: overwrite (lambda x, y: y)
    """

    status: NotRequired[str]
    """
    Current workflow status.
    Values: "pending" | "in_progress" | "completed" | "failed" | "human_approval" | "input_required" | "rolled_back" | "interrupted"
    """
    
    # ==================== Step 1: Request Analysis ====================
    req_analyser_output: NotRequired[Annotated[Dict[str, Any], lambda x, y: {**(x or {}), **(y or {})}]]
    
    # ==================== Step 2: Security & Best Practices ====================
    sec_n_best_practices_output: NotRequired[Annotated[Dict[str, Any], lambda x, y: {**(x or {}), **(y or {})}]]
    
    # ==================== Step 3: Execution Planner ====================
    execution_planner_output: NotRequired[Annotated[Dict[str, Any], lambda x, y: {**(x or {}), **(y or {})}]]
    
    # ==================== Error Tracking ====================
    error_state: NotRequired[ErrorContext]
    """Error context for recovery"""
    
    # ==================== Virtual Filesystem ====================
    files: NotRequired[Dict[str, Any]]
    """Virtual FS entries (create_file_data format) containing generated skills"""

