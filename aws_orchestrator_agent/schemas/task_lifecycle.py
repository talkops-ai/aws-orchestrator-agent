"""
Task Lifecycle schemas for A2A protocol.

This module defines Pydantic models for task lifecycle management according to the A2A protocol
specification, including task states, transitions, history, and artifacts.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict

from a2a.types import TaskState as A2ATaskState


class TaskState(str, Enum):
    """
    Task states according to A2A protocol specification.
    
    These states represent the lifecycle of a task as it moves through
    agent workflows, ensuring consistency and interoperability.
    """
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    
    @classmethod
    def from_a2a_state(cls, a2a_state: A2ATaskState) -> "TaskState":
        """Convert A2A SDK TaskState to our TaskState enum."""
        mapping = {
            A2ATaskState.SUBMITTED: cls.SUBMITTED,
            A2ATaskState.WORKING: cls.WORKING,
            A2ATaskState.INPUT_REQUIRED: cls.INPUT_REQUIRED,
            A2ATaskState.COMPLETED: cls.COMPLETED,
            A2ATaskState.FAILED: cls.FAILED,
            A2ATaskState.CANCELED: cls.CANCELED,
        }
        return mapping.get(a2a_state, cls.SUBMITTED)
    
    def to_a2a_state(self) -> A2ATaskState:
        """Convert our TaskState enum to A2A SDK TaskState."""
        mapping = {
            self.SUBMITTED: A2ATaskState.SUBMITTED,
            self.WORKING: A2ATaskState.WORKING,
            self.INPUT_REQUIRED: A2ATaskState.INPUT_REQUIRED,
            self.COMPLETED: A2ATaskState.COMPLETED,
            self.FAILED: A2ATaskState.FAILED,
            self.CANCELED: A2ATaskState.CANCELED,
        }
        return mapping[self]


class TaskTransition(BaseModel):
    """
    Represents a state transition in a task's lifecycle.
    
    This captures the complete context of a state change including
    the previous state, new state, timestamp, and any associated metadata.
    """
    
    from_state: Optional[TaskState] = Field(default=None)
    to_state: TaskState = Field(..., description="New task state")
    timestamp: Any = Field(default_factory=datetime.utcnow)
    reason: Optional[str] = Field(default=None, description="Reason for the state change")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional transition metadata")
    agent_id: Optional[str] = Field(default=None, description="ID of the agent that triggered the transition")
    user_id: Optional[str] = Field(default=None, description="ID of the user who triggered the transition")
    
    @field_validator('to_state')
    def validate_to_state(cls, v, info):
        """Validate that the state transition is allowed."""
        from_state = info.data.get('from_state')
        # Define allowed transitions
        allowed_transitions = {
            TaskState.SUBMITTED: [TaskState.WORKING, TaskState.CANCELED, TaskState.FAILED],
            TaskState.WORKING: [TaskState.INPUT_REQUIRED, TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED],
            TaskState.INPUT_REQUIRED: [TaskState.WORKING, TaskState.FAILED, TaskState.CANCELED],
            TaskState.COMPLETED: [],  # Terminal state
            TaskState.FAILED: [],     # Terminal state
            TaskState.CANCELED: [],   # Terminal state
        }
        
        if from_state is None:
            # Allow initial transition only to SUBMITTED
            if v != TaskState.SUBMITTED:
                raise ValueError(f"Initial transition can only be to {TaskState.SUBMITTED}, got {v}")
        else:
            if v not in allowed_transitions.get(from_state, []):
                raise ValueError(f"Invalid transition from {from_state} to {v}")
        return v


class TaskStatus(BaseModel):
    """
    Current status of a task with state and context information.
    
    This represents the current state of a task along with any
    associated messages, metadata, and timing information.
    """
    
    state: TaskState = Field(..., description="Current task state")
    message: Optional[str] = Field(default=None, description="Status message or description")
    timestamp: Optional[Any] = Field(default_factory=datetime.utcnow, description="When this status was set")
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Task progress (0.0 to 1.0)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional status metadata")
    error_code: Optional[str] = Field(default=None, description="Error code if task failed")
    error_details: Optional[str] = Field(default=None, description="Detailed error information")


class TaskArtifact(BaseModel):
    """
    Artifact produced by a task.
    
    This represents any output, file, or data produced during
    task execution that should be preserved and accessible.
    """
    
    id: str = Field(..., description="Unique artifact identifier")
    name: str = Field(..., description="Human-readable artifact name")
    type: str = Field(..., description="Type of artifact (file, data, report, etc.)")
    content: Optional[str] = Field(default=None, description="Artifact content or data")
    file_path: Optional[str] = Field(default=None, description="Path to artifact file if stored on disk")
    mime_type: Optional[str] = Field(default=None, description="MIME type of the artifact")
    size_bytes: Optional[int] = Field(default=None, description="Size of the artifact in bytes")
    created_at: Any = Field(default_factory=datetime.utcnow, description="When the artifact was created")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional artifact metadata")


class TaskHistory(BaseModel):
    """
    Complete history of a task including all state transitions and events.
    
    This provides a complete audit trail of a task's lifecycle,
    enabling debugging, compliance, and analysis.
    """
    
    task_id: str = Field(..., description="ID of the task")
    created_at: Any = Field(default_factory=datetime.utcnow, description="When the task was created")
    transitions: List[TaskTransition] = Field(default_factory=list, description="List of state transitions")
    events: List[Any] = Field(default_factory=list, description="Additional events and logs")
    artifacts: List[TaskArtifact] = Field(default_factory=list, description="Artifacts produced by the task")
    
    @property
    def current_state(self) -> Optional[TaskState]:
        """Get the current state from the most recent transition."""
        if self.transitions:
            return self.transitions[-1].to_state
        return None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate the total duration of the task in seconds."""
        if self.transitions:
            start_time = self.transitions[0].timestamp
            end_time = self.transitions[-1].timestamp
            return (end_time - start_time).total_seconds()
        return None
    
    def add_transition(self, transition: TaskTransition):
        """Add a new transition to the history."""
        self.transitions.append(transition)
    
    def add_event(self, event: Any):
        """Add an event to the history."""
        self.events.append(event)
    
    def add_artifact(self, artifact: TaskArtifact):
        """Add an artifact to the history."""
        self.artifacts.append(artifact)


class TaskLifecycleConfig(BaseModel):
    """
    Configuration for task lifecycle management.
    
    This defines various settings and limits for task lifecycle
    management including timeouts, retry policies, and validation rules.
    """
    
    max_task_duration_seconds: int = Field(
        default=3600,  # 1 hour
        description="Maximum allowed duration for a task in seconds"
    )
    input_required_timeout_seconds: int = Field(
        default=300,  # 5 minutes
        description="Timeout for input-required state before auto-canceling"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed tasks"
    )
    retry_delay_seconds: int = Field(
        default=60,
        description="Delay between retries in seconds"
    )
    enable_audit_trail: bool = Field(
        default=True,
        description="Whether to maintain detailed audit trail"
    )
    max_history_entries: int = Field(
        default=1000,
        description="Maximum number of history entries to keep per task"
    )
    auto_cleanup_completed_tasks: bool = Field(
        default=False,
        description="Whether to automatically cleanup completed tasks"
    )
    cleanup_delay_days: int = Field(
        default=30,
        description="Days to wait before cleaning up completed tasks"
    )


class TaskMetrics(BaseModel):
    """
    Performance metrics for task execution.
    
    This captures various metrics about task performance including
    timing, resource usage, and success rates.
    """
    
    task_id: str = Field(..., description="ID of the task")
    total_duration_seconds: float = Field(..., description="Total execution time in seconds")
    processing_duration_seconds: float = Field(..., description="Time spent in working state")
    waiting_duration_seconds: float = Field(..., description="Time spent in input-required state")
    transition_count: int = Field(..., description="Number of state transitions")
    retry_count: int = Field(..., description="Number of retries attempted")
    artifact_count: int = Field(..., description="Number of artifacts produced")
    memory_usage_mb: Optional[float] = Field(default=None, description="Peak memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(default=None, description="Average CPU usage percentage")
    success: bool = Field(..., description="Whether the task completed successfully") 