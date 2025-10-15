"""
Planning State Schema for Planner Sub-Supervisor.

This module defines the state schema for the planner sub-supervisor that:
- Tracks planning workflow state across phases
- Stores data from each sub-agent
- Manages planning context and results
- Integrates with main supervisor state
"""

from typing import List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph.message import add_messages
from aws_orchestrator_agent.utils.logger import AgentLogger
logger = AgentLogger("PLANNER_SUPERVISOR_STATE")

class PlanningWorkflowState(BaseModel):
    """State for tracking planning workflow progress with completion detection and loop prevention."""
    current_phase: str = Field(default="requirements_analysis", description="Current planning phase")
    requirements_complete: bool = Field(default=False, description="Requirements analysis complete")
    security_n_best_practices_evaluator_complete: bool = Field(default=False, description="security_n_best_practices_evaluator complete")
    execution_complete: bool = Field(default=False, description="Execution planning complete")
    planning_complete: bool = Field(default=False, description="Overall planning complete")
    loop_counter: int = Field(default=0, ge=0, le=10, description="Loop counter for infinite loop prevention")
    last_phase_transition: Optional[str] = Field(default=None, description="Timestamp of last phase transition")
    error_occurred: bool = Field(default=False, description="Whether an error occurred")
    error_message: Optional[str] = Field(default=None, description="Error message if any")
    
    @property
    def is_complete(self) -> bool:
        """Check if all phases are complete."""
        return all([
            self.requirements_complete,
            #self.security_n_best_practices_evaluator_complete,
            self.execution_complete
        ])
    
    @property
    def next_phase(self) -> Optional[str]:
        """Determine the next phase based on completion status."""
        if not self.requirements_complete:
            return "requirements_analysis"
        # elif not self.security_n_best_practices_evaluator_complete:
        #     return "security_n_best_practices_evaluator"
        elif not self.execution_complete:
            return "execution_planning"
        else:
            return None  # All phases complete
    
    def increment_loop_counter(self) -> None:
        """Increment loop counter and check for limits."""
        self.loop_counter += 1
        if self.loop_counter > 10:
            self.error_occurred = True
            self.error_message = "Maximum iterations reached (10)"
    
    def set_phase_complete(self, phase: str) -> None:
        """Mark a specific phase as complete."""
        if phase == "requirements_analysis":
            self.requirements_complete = True
        # elif phase == "security_n_best_practices_evaluator":
        #     self.security_n_best_practices_evaluator_complete = True
        elif phase == "execution_planning":
            self.execution_complete = True
        
        # Update current phase and check if planning is complete
        if self.is_complete:
            self.planning_complete = True
            self.current_phase = "complete"
        else:
            self.current_phase = self.next_phase or "complete"

class RequirementsData(BaseModel):
    """Data from Requirements Analyzer agent."""
    analysis_results: Optional[Dict[str, Any]] = Field(default=None, description="Analysis results")
    analysis_complete: bool = Field(default=False, description="Whether requirements analysis is complete")
    aws_service_mapping: Optional[Dict[str, Any]] = Field(default=None, description="AWS service mapping from discovery")
    aws_service_mapping_complete: bool = Field(default=False, description="Whether AWS service mapping is complete")
    terraform_attribute_mapping: Optional[Dict[str, Any]] = Field(default=None, description="Terraform attribute mapping from attribute mapping")
    terraform_attribute_mapping_complete: bool = Field(default=False, description="Whether Terraform attribute mapping is complete")
    timestamp: Optional[str] = Field(default=None, description="Timestamp when analysis was completed")

class Security_N_Best_Practices_Evaluator_Data(BaseModel):
    """Data from security_n_best_practices_evaluator agent."""
    primary_service: str = Field(default="", description="Primary AWS service")
    mandatory_dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Mandatory dependencies")
    optional_dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Optional dependencies")
    dependency_categories: Dict[str, List[str]] = Field(default_factory=dict, description="Dependencies by category")
    setup_prerequisites: List[str] = Field(default_factory=list, description="Setup prerequisites")
    terraform_provider_requirements: List[str] = Field(default_factory=list, description="Terraform provider requirements")
    dependency_explanations: Dict[str, str] = Field(default_factory=dict, description="Dependency explanations")
    follow_up_questions: List[str] = Field(default_factory=list, description="Follow-up questions")
    user_responses: List[Dict[str, str]] = Field(default_factory=list, description="User responses to questions")

class ExecutionData(BaseModel):
    """Data from Execution Planner agent."""
    module_structure_plan: Optional[Dict[str, Any]] = Field(default=None, description="Module structure plan")
    module_structure_plan_complete: bool = Field(default=False, description="Whether module structure plan is complete")
    configuration_optimizer_data: Optional[Dict[str, Any]] = Field(default=None, description="Configuration optimizer data")
    configuration_optimizer_complete: bool = Field(default=False, description="Whether configuration optimizer is complete")
    state_management_data: Optional[Dict[str, Any]] = Field(default=None, description="State management data")
    state_management_complete: bool = Field(default=False, description="Whether state management is complete")
    execution_plan_data: Optional[Dict[str, Any]] = Field(default=None, description="Execution plan data")
    execution_plan_complete: bool = Field(default=False, description="Whether execution plan is complete")
    agent_completion: Optional[Dict[str, Any]] = Field(default=None, description="Agent completion data")
    timestamp: Optional[str] = Field(default=None, description="Timestamp when execution planning was completed")

class PlanningResults(BaseModel):
    """Complete planning results from all phases."""
    requirements_data: RequirementsData = Field(default_factory=RequirementsData, description="Requirements analysis results")
    security_n_best_practices_evaluator_data: Security_N_Best_Practices_Evaluator_Data = Field(default_factory=Security_N_Best_Practices_Evaluator_Data, description="security_n_best_practices_evaluator results")
    execution_data: ExecutionData = Field(default_factory=ExecutionData, description="Execution planning results")
    overall_complexity_score: int = Field(default=1, description="Overall complexity score (1-10)")
    estimated_deployment_time: str = Field(default="", description="Estimated deployment time")
    risk_level: str = Field(default="low", description="Overall risk level")
    summary: str = Field(default="", description="Planning summary")

# class PlannerData(BaseModel):
#     """Data from planner sub-supervisor."""
#     requirements_data: Dict[str, Any] = Field(default_factory=dict, description="Requirements analysis results as serialized dict")
#     execution_data: Dict[str, Any] = Field(default_factory=dict, description="Execution planning results as serialized dict")
#     planning_results: Dict[str, Any] = Field(default_factory=dict, description="Planning results as serialized dict")
#     planning_complete: bool = Field(default=False, description="Whether planning is complete")
#     completion_timestamp: Optional[str] = Field(default=None, description="Timestamp when planning was completed")

class PlannerSupervisorState(BaseModel):
    """State schema for the Planner Sub-Supervisor using langgraph-supervisor."""
    
    # Required by langgraph-supervisor
    messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Message history for the planning workflow"
    )
    remaining_steps: int = Field(
        default=50,
        description="Remaining steps for the agent to complete"
    )
    
    # LLM input messages for langgraph-supervisor
    llm_input_messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Message history for langgraph-supervisor LLM input"
    )
    
    # Planning workflow state - renamed to avoid conflict with supervisor's workflow_state
    planning_workflow_state: PlanningWorkflowState = Field(
        default_factory=PlanningWorkflowState,
        description="Current state of the planning workflow"
    )
    
    # User request and context
    user_request: str = Field(
        default="",
        description="Original user request for planning"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task identifier"
    )
    
    # Agent-specific data
    requirements_data: RequirementsData = Field(
        default_factory=RequirementsData,
        description="Data from requirements analysis"
    )
    security_n_best_practices_evaluator_data: Security_N_Best_Practices_Evaluator_Data = Field(
        default_factory=Security_N_Best_Practices_Evaluator_Data,
        description="Data from security_n_best_practices_evaluator"
    )
    execution_data: ExecutionData = Field(
        default_factory=ExecutionData,
        description="Data from execution planning"
    )
    
    # Planning results
    planning_results: PlanningResults = Field(
        default_factory=PlanningResults,
        description="Final aggregated planning results"
    )
    
    # Supervisor-specific fields
    active_agent: Optional[str] = Field(
        default=None,
        description="Currently active agent in the planning workflow"
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Current task description for the active agent"
    )
    planning_context: Optional[str] = Field(
        default=None,
        description="Context information for planning decisions"
    )
    status: str = Field(
        default="pending",
        description="Current status of the planning workflow"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the planning workflow"
    )
    
    # NEW: Completion signaling flags for preventing infinite loops
    completion_emitted: bool = Field(
        default=False,
        description="Prevents re-entry into completion logic in pre_model_hook"
    )
    completion_lock: bool = Field(
        default=False,
        description="Atomic lock for completion operations to prevent race conditions"
    )

    # planner_data: Optional[PlannerData] = Field(
    #     default=None,
    #     description="Planning data from planner sub-supervisor"
    # )
    
    @model_validator(mode='after')
    def validate_completion_state(self):
        """Validate completion state consistency and loop prevention."""
        # Check loop counter limits
        if self.planning_workflow_state.loop_counter > 10:
            self.error = "Maximum iterations reached (10)"
            self.status = "error"
            return self
        
        # Validate phase consistency
        if self.planning_workflow_state.planning_complete and not self.planning_workflow_state.is_complete:
            self.error = "Planning marked complete but not all phases are done"
            self.status = "error"
            return self
        
        # Validate current phase matches completion status
        expected_phase = self.planning_workflow_state.next_phase
        if expected_phase and self.planning_workflow_state.current_phase != expected_phase:
            # Auto-correct phase if possible
            self.planning_workflow_state.current_phase = expected_phase
        
        return self
    
    def increment_loop_counter(self) -> None:
        """Increment loop counter and check for limits."""
        self.planning_workflow_state.increment_loop_counter()
        if self.planning_workflow_state.error_occurred:
            self.error = self.planning_workflow_state.error_message
            self.status = "error"
    
    def set_phase_complete(self, phase: str) -> None:
        """Mark a specific phase as complete and update state."""
        self.planning_workflow_state.set_phase_complete(phase)
        
        # Update status based on completion
        if self.planning_workflow_state.planning_complete:
            self.status = "completed"
        else:
            self.status = "in_progress"
        
        # Log phase completion
        logger.log_structured(
            level="INFO",
            message=f"Phase {phase} marked complete",
            extra={
                "phase": phase,
                "current_phase": self.planning_workflow_state.current_phase,
                "planning_complete": self.planning_workflow_state.planning_complete,
                "loop_counter": self.planning_workflow_state.loop_counter
            }
        )

def create_initial_planner_state(
    user_request: str,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None
) -> PlannerSupervisorState:
    """
    Create initial planner supervisor state.
    
    Args:
        user_request: The user's infrastructure request
        session_id: Session ID for tracking
        task_id: Task ID for tracking
        
    Returns:
        Initial planner supervisor state
    """
    return PlannerSupervisorState(
        user_request=user_request,
        session_id=session_id,
        task_id=task_id,
        planning_workflow_state=PlanningWorkflowState(current_phase="requirements_analysis"),
        planning_context="Initial planning phase started",
        status="pending"
    )

def update_planning_context(
    state: PlannerSupervisorState,
    phase: str,
    data: Optional[Dict[str, Any]] = None
) -> PlannerSupervisorState:
    """
    Update planning context with new phase and data.
    
    Args:
        state: Current planner state
        phase: New planning phase
        data: Additional data to update
        
    Returns:
        Updated planner state
    """
    # Update planning workflow state
    planning_workflow_state = state.planning_workflow_state.copy()
    planning_workflow_state.current_phase = phase
    
    if phase == "requirements":
        planning_workflow_state.requirements_complete = True
    elif phase == "tf_security_n_best_practices_evaluator":
        planning_workflow_state.security_n_best_practices_evaluator_complete = True
    elif phase == "execution":
        planning_workflow_state.execution_complete = True
    elif phase == "complete":
        planning_workflow_state.planning_complete = True
    
    # Update planning context
    planning_context = state.planning_context.copy() if state.planning_context else {}
    planning_context["planning_phase"] = phase
    planning_context["requirements_complete"] = planning_workflow_state.requirements_complete
    planning_context["security_n_best_practices_evaluator_complete"] = planning_workflow_state.security_n_best_practices_evaluator_complete
    planning_context["execution_complete"] = planning_workflow_state.execution_complete
    planning_context["planning_complete"] = planning_workflow_state.planning_complete
    
    if data:
        planning_context.update(data)
    
    # Create updated state
    updated_state = state.copy()
    updated_state.planning_workflow_state = planning_workflow_state
    updated_state.planning_context = planning_context
    
    return updated_state

def create_planning_results(state: PlannerSupervisorState | Dict[str, Any]) -> PlanningResults:
    """
    Create complete planning results from state data.
    
    Args:
        state: Planner supervisor state (can be PlannerSupervisorState object or dict)
        
    Returns:
        Complete planning results
    """
    # Handle both object and dict inputs
    if isinstance(state, dict):
        execution_data = state.get("execution_data", {})
        requirements_data = state.get("requirements_data", {})
        dependency_data = state.get("dependency_data", {})
    else:
        execution_data = state.execution_data
        requirements_data = state.requirements_data
        dependency_data = state.dependency_data
    
    # Calculate overall complexity score
    complexity_score = 1  # Base score
    
    if isinstance(execution_data, dict):
        if execution_data.get("complexity_score"):
            complexity_score = execution_data["complexity_score"]
    else:
        if execution_data.complexity_score:
            complexity_score = execution_data.complexity_score
    
    # Determine risk level
    risk_level = "low"
    if isinstance(execution_data, dict):
        risk_assessment = execution_data.get("risk_assessment")
        if risk_assessment:
            risk_level = risk_assessment.get("overall_risk_level", "low")
    else:
        if execution_data.risk_assessment:
            risk_level = execution_data.risk_assessment.get("overall_risk_level", "low")
    
    # Create summary
    summary_parts = []
    
    if isinstance(requirements_data, dict):
        business_reqs = requirements_data.get("business_requirements", [])
        if business_reqs:
            summary_parts.append(f"Business requirements: {len(business_reqs)} items")
    else:
        if requirements_data.business_requirements:
            summary_parts.append(f"Business requirements: {len(requirements_data.business_requirements)} items")
    
    if isinstance(dependency_data, dict):
        mandatory_deps = dependency_data.get("mandatory_dependencies", [])
        if mandatory_deps:
            summary_parts.append(f"Dependencies: {len(mandatory_deps)} mandatory")
    else:
        if dependency_data.mandatory_dependencies:
            summary_parts.append(f"Dependencies: {len(dependency_data.mandatory_dependencies)} mandatory")
    
    if isinstance(execution_data, dict):
        execution_steps = execution_data.get("execution_steps", [])
        if execution_steps:
            summary_parts.append(f"Execution: {len(execution_steps)} steps")
    else:
        if execution_data.execution_steps:
            summary_parts.append(f"Execution: {len(execution_data.execution_steps)} steps")
    
    summary = "; ".join(summary_parts) if summary_parts else "Planning completed"
    
    # Get estimated deployment time
    if isinstance(execution_data, dict):
        estimated_time = execution_data.get("total_estimated_time", "")
    else:
        estimated_time = execution_data.total_estimated_time if hasattr(execution_data, 'total_estimated_time') else ""
    
    return PlanningResults(
        requirements_data=requirements_data,
        dependency_data=dependency_data,
        execution_data=execution_data,
        overall_complexity_score=complexity_score,
        estimated_deployment_time=estimated_time,
        risk_level=risk_level,
        summary=summary
    )
