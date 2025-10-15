"""
State schema definitions for the custom supervisor agent system.

This module defines the Pydantic-based state schemas for the Supervisor Agent
and all specialized agent subgraphs, following LangGraph multi-agent best practices
with minimal state overlap and external artifact references.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Union, Annotated, AsyncGenerator, Set
from enum import Enum
from datetime import datetime, timezone
import uuid

# LangGraph imports for proper message handling
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

# Import generator types for StateTransformer
from .generator.generator_state import GeneratorSwarmState, GeneratorAgentStatus
from .writer.writer_react_agent import WriterReactState, WriteStatus



class AgentResponse(BaseModel):
    """
    Response from an agent during execution.
    
    This represents a single response item from the agent's stream,
    containing the content, metadata, and control flags.
    """
    
    model_config = ConfigDict(extra="allow")
    
    content: Any = Field(..., description="The response content (text or data)")
    response_type: str = Field(default="text", description="Type of response: 'text' or 'data'")
    is_task_complete: bool = Field(default=False, description="Whether this response indicates task completion")
    require_user_input: bool = Field(default=False, description="Whether this response requires user input to continue")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")
    root: Optional[Any] = Field(default=None, description="Root object for A2A protocol integration")


class BaseAgent(ABC):
    """
    Base interface for all AWS Orchestrator Agent implementations.
    
    This abstract base class defines the contract that all agent implementations
    must follow to work with the A2A protocol integration.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the agent.
        
        Returns:
            The agent's name
        """
        pass
    
    @abstractmethod
    async def stream(
        self, 
        query: str, 
        context_id: str, 
        task_id: str
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Stream responses for a given query.
        
        This method should implement the core agent logic and yield
        AgentResponse objects as the agent processes the query.
        
        Args:
            query: The user query to process
            context_id: The A2A context ID
            task_id: The A2A task ID
            
        Yields:
            AgentResponse objects representing the agent's progress
        """
        pass
    
    async def initialize(self) -> None:
        """
        Initialize the agent.
        
        This method can be overridden to perform any initialization
        required by the agent implementation.
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by the agent.
        
        This method can be overridden to perform any cleanup
        required by the agent implementation.
        """
        pass


# ============================================================================
# ENUMS
# ============================================================================

class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    HUMAN_APPROVAL = "human_approval"
    ROLLED_BACK = "rolled_back"
    INTERRUPTED = "interrupted"

class AgentType(str, Enum):
    """Agent type enumeration."""
    PLANNER = "planner"
    GENERATION = "generation"
    WRITER = "writer"
    EDITOR = "editor"
    SECURITY = "security"
    COST = "cost"


class SupervisorWorkflowState(BaseModel):
    """
    Workflow state tracking for the main supervisor following best practices.
    
    This schema provides:
    - Type-safe workflow progress tracking
    - Automatic phase transitions
    - Loop prevention and error detection
    - Integration with langgraph-supervisor
    """
    
    # Current workflow phase
    current_phase: str = Field(default="planning", description="Current workflow phase")
    
    # Phase completion tracking
    planning_complete: bool = Field(default=False, description="Planning phase complete")
    generation_complete: bool = Field(default=False, description="Generation phase complete")
    writer_complete: bool = Field(default=False, description="Writer phase complete")
    editing_complete: bool = Field(default=False, description="Editing phase complete")
    
    # Workflow control
    workflow_complete: bool = Field(default=False, description="Overall workflow complete")
    loop_counter: int = Field(default=0, ge=0, le=20, description="Loop counter for infinite loop prevention")
    last_phase_transition: Optional[datetime] = Field(default=None, description="Timestamp of last phase transition")
    error_occurred: bool = Field(default=False, description="Whether an error occurred")
    error_message: Optional[str] = Field(default=None, description="Error message if any")
    
    # Agent handoff tracking
    last_agent: Optional[str] = Field(default=None, description="Last agent that completed")
    next_agent: Optional[str] = Field(default=None, description="Next agent to invoke")
    handoff_reason: Optional[str] = Field(default=None, description="Reason for handoff")
    
    @property
    def is_complete(self) -> bool:
        """Check if all required phases are complete."""
        # For infrastructure requests, we typically need planning + generation
        # Validation and editing are optional based on requirements
        return all([
            self.planning_complete,
            self.generation_complete,
            self.writer_complete
        ])
    
    @property
    def next_phase(self) -> Optional[str]:
        """Determine the next phase based on completion status."""
        if not self.planning_complete:
            return "planning"
        elif not self.generation_complete:
            return "generation"
        elif not self.writer_complete:
            return "writer"
        elif not self.editing_complete:
            return "editing"
        else:
            return None  # All phases complete
    
    def increment_loop_counter(self) -> None:
        """Increment loop counter and check for limits."""
        self.loop_counter += 1
        if self.loop_counter > 20:  # Higher limit for main supervisor
            self.error_occurred = True
            self.error_message = "Maximum iterations reached (20)"
    
    def set_phase_complete(self, phase: str) -> None:
        """Mark a specific phase as complete and update state."""
        if phase == "planning":
            self.planning_complete = True
        elif phase == "generation":
            self.generation_complete = True
        elif phase == "writer":
            self.writer_complete = True
        elif phase == "editing":
            self.editing_complete = True
        
        # Update current phase and check if workflow is complete
        if self.is_complete:
            self.workflow_complete = True
            self.current_phase = "complete"
        else:
            self.current_phase = self.next_phase or "complete"
        
        # Update transition timestamp
        self.last_phase_transition = datetime.now(timezone.utc)
    
    def set_agent_handoff(self, from_agent: str, to_agent: str, reason: str) -> None:
        """Track agent handoff for debugging and monitoring."""
        self.last_agent = from_agent
        self.next_agent = to_agent
        self.handoff_reason = reason
        self.last_phase_transition = datetime.now(timezone.utc)
    
    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get current workflow progress for monitoring."""
        return {
            "current_phase": self.current_phase,
            "planning_complete": self.planning_complete,
            "generation_complete": self.generation_complete,
            "writer_complete": self.writer_complete,
            "editing_complete": self.editing_complete,
            "workflow_complete": self.workflow_complete,
            "loop_counter": self.loop_counter,
            "last_agent": self.last_agent,
            "next_agent": self.next_agent,
            "handoff_reason": self.handoff_reason,
            "error_occurred": self.error_occurred,
            "error_message": self.error_message
        }

class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    WARNING = "warning"

class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# SUPERVISOR STATE SCHEMA (CENTRAL ORCHESTRATOR)
# ============================================================================

class SupervisorState(BaseModel):
    """
    Supervisor state following LangGraph best practices.
    
    Central orchestrator that manages workflow, message history, and agent coordination.
    Uses Annotated[list, add_messages] for proper message handling and
    extends with infrastructure orchestration fields.
    """
    
    # Core LangGraph-style state with proper message handling
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    
    # Required by langgraph-supervisor
    remaining_steps: int = Field(default=50, description="Remaining steps for the agent to complete")

    # ## Generator swarm need this to track active agent
    # active_agent: Optional[str] = None

    llm_input_messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="LLM input messages for langgraph-supervisor"
    )
    
    # Workflow state tracking (PRIMARY METHOD - following best practices)
    workflow_state: SupervisorWorkflowState = Field(
        default_factory=SupervisorWorkflowState,
        description="Workflow progress tracking with type-safe state management"
    )
    
    # Minimal workflow metadata
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_agent: Optional[AgentType] = None
    
    # User context (essential for our use case)
    user_request: str
    question: Optional[str] = None
    
    # Infrastructure artifacts (essential for Terraform orchestration)
    workspace_ref: Optional[str] = None  # URI to workspace
    terraform_context: Optional[Dict[str, Any]] = None  # Terraform-specific context
    generated_module_ref: Optional[str] = None  # URI to generated module
    validation_report_ref: Optional[str] = None  # URI to validation report
    security_report_ref: Optional[str] = None  # URI to security scan
    cost_report_ref: Optional[str] = None  # URI to cost analysis
    
    # Human-in-the-loop (essential for approval workflow)
    human_approval_required: bool = False
    approval_context: Optional[Dict[str, Any]] = None
    approval_timeout: Optional[datetime] = None
    
    # Error handling (simplified)
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Minimal audit trail (for compliance)
    workflow_started_at: Optional[datetime] = None
    workflow_completed_at: Optional[datetime] = None
    
    # Agent-specific state data (for coordination)
    planner_data: Optional[Dict[str, Any]] = None
    generation_data: Optional[Dict[str, Any]] = None
    writer_data: Optional[Dict[str, Any]] = None
    editor_data: Optional[Dict[str, Any]] = None
    security_data: Optional[Dict[str, Any]] = None
    cost_data: Optional[Dict[str, Any]] = None
    
    # # Generator Swarm State (nested approach)
    # generator_state: Optional[GeneratorSwarmState] = None
    
    # Generation-specific context fields (extracted from planner_data for direct access)
    execution_plan: Optional[Dict[str, Any]] = None  # Direct access to execution plan
    resource_configurations: Optional[List[Dict[str, Any]]] = None
    variable_definitions: Optional[List[Dict[str, Any]]] = None
    data_sources: Optional[List[Dict[str, Any]]] = None
    local_values: Optional[List[Dict[str, Any]]] = None
    planning_dependencies: Optional[List[Dict[str, Any]]] = None
    security_considerations: Optional[List[Dict[str, Any]]] = None
    cost_estimates: Optional[List[Dict[str, Any]]] = None
    module_name: Optional[str] = None
    service_name: Optional[str] = None
    target_environment: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# SIMPLIFIED AGENT STATE SCHEMAS
# ============================================================================

# Note: PlannerState has been removed. The planner sub-supervisor now manages
# its own state internally using PlannerSupervisorState. The supervisor only
# receives final results via the planner's output_transform() method.

class GenerationState(BaseModel):
    """
    Simplified state schema for the Generation Agent subgraph.
    
    Contains only agent-specific input, processing, and output data.
    All workflow management handled by SupervisorState.
    """
    
    # Input from supervisor/planner
    requirements: Dict[str, Any] = Field(default_factory=dict)
    provider_versions: Dict[str, str] = Field(default_factory=dict)
    registry_schemas_ref: Optional[str] = None  # URI to registry schemas
    standards_profile: Dict[str, Any] = Field(default_factory=dict)
    
    # Question handling
    question: Optional[str] = None  # Question for user input if needed
    
    # Generation process
    design_rationale: Optional[str] = None
    file_plan: List[Dict[str, Any]] = Field(default_factory=list)
    template_params: Dict[str, Any] = Field(default_factory=dict)
    generated_files_ref: Optional[str] = None  # URI to generated files
    
    # Module metadata
    module_name: str
    module_version: str = "1.0.0"
    module_manifest: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality checks
    warnings: List[str] = Field(default_factory=list)
    best_practices_applied: List[str] = Field(default_factory=list)
    
    # Direct execution plan data (from planner)
    execution_plan: Optional[Dict[str, Any]] = None
    resource_configurations: List[Dict[str, Any]] = Field(default_factory=list)
    variable_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    local_values: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: List[Dict[str, Any]] = Field(default_factory=list)
    security_considerations: List[Dict[str, Any]] = Field(default_factory=list)
    cost_estimates: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


class ValidationState(BaseModel):
    """
    Simplified state schema for the Validation Agent subgraph.
    
    Contains only agent-specific input, processing, and output data.
    All workflow management handled by SupervisorState.
    """
    
    # Input from supervisor
    module_ref: Optional[str] = None  # URI to module to validate
    workspace_ref: Optional[str] = None  # URI to workspace
    policy_sets: List[str] = Field(default_factory=list)
    quota_profile: Dict[str, Any] = Field(default_factory=dict)
    
    # Question handling
    question: Optional[str] = None  # Question for user input if needed
    
    # Validation stages (parallel execution)
    static_analysis: Dict[str, Any] = Field(default_factory=dict)
    terraform_validate: Dict[str, Any] = Field(default_factory=dict)
    terraform_plan: Dict[str, Any] = Field(default_factory=dict)
    security_scans: Dict[str, Any] = Field(default_factory=dict)
    compliance_checks: Dict[str, Any] = Field(default_factory=dict)
    quota_evaluation: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation results
    validation_report_ref: Optional[str] = None  # URI to full report
    blockers: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


class EditorState(BaseModel):
    """
    Simplified state schema for the Editor Agent subgraph.
    
    Contains only agent-specific input, processing, and output data.
    All workflow management handled by SupervisorState.
    """
    
    # Input from supervisor
    target_config_ref: Optional[str] = None  # URI to config to edit
    change_request: Dict[str, Any] = Field(default_factory=dict)
    state_diff_context: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph: Dict[str, Any] = Field(default_factory=dict)
    
    # Question handling
    question: Optional[str] = None  # Question for user input if needed
    
    # Editing process
    ast_notes: Dict[str, Any] = Field(default_factory=dict)
    formatting_profile: Dict[str, Any] = Field(default_factory=dict)
    compatibility_findings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Surgical changes
    surgical_changes: List[Dict[str, Any]] = Field(default_factory=list)
    modified_files: List[str] = Field(default_factory=list)
    patch_ref: Optional[str] = None  # URI to patch/diff
    
    # Migration and risk
    migration_notes: List[str] = Field(default_factory=list)
    apply_risk: RiskLevel = RiskLevel.LOW
    rollback_strategy: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


class SecurityState(BaseModel):
    """
    Simplified state schema for the Security Agent subgraph.
    
    Contains only agent-specific input, processing, and output data.
    All workflow management handled by SupervisorState.
    """
    
    # Input from supervisor
    module_ref: Optional[str] = None  # URI to module to analyze
    security_policies: List[str] = Field(default_factory=list)
    compliance_frameworks: List[str] = Field(default_factory=list)
    
    # Question handling
    question: Optional[str] = None  # Question for user input if needed
    
    # Security analysis
    iam_analysis: Dict[str, Any] = Field(default_factory=dict)
    network_security: Dict[str, Any] = Field(default_factory=dict)
    data_protection: Dict[str, Any] = Field(default_factory=dict)
    vulnerability_scan: Dict[str, Any] = Field(default_factory=dict)
    
    # Compliance checks
    compliance_results: Dict[str, Any] = Field(default_factory=dict)
    policy_violations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Security findings
    security_findings: List[Dict[str, Any]] = Field(default_factory=list)
    severity_counts: Dict[str, int] = Field(default_factory=dict)
    required_actions: List[str] = Field(default_factory=list)
    
    # Output reference
    security_report_ref: Optional[str] = None  # URI to security report
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


class CostState(BaseModel):
    """
    Simplified state schema for the Cost Agent subgraph.
    
    Contains only agent-specific input, processing, and output data.
    All workflow management handled by SupervisorState.
    """
    
    # Input from supervisor
    plan_json_ref: Optional[str] = None  # URI to Terraform plan JSON
    region: str = "us-east-1"
    
    # Question handling
    question: Optional[str] = None  # Question for user input if needed
    pricing_cache_hint: Optional[str] = None
    
    # Cost analysis
    monthly_estimate: Optional[float] = None
    annual_forecast: Optional[float] = None
    cost_breakdown: Dict[str, Any] = Field(default_factory=dict)
    resource_costs: Dict[str, float] = Field(default_factory=dict)
    
    # Optimization
    optimization_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    potential_savings: Optional[float] = None
    budget_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Cost tracking
    cost_history: List[Dict[str, Any]] = Field(default_factory=list)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # Output reference
    cost_report_ref: Optional[str] = None  # URI to cost report
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# STATE TRANSFORMATION FUNCTIONS
# ============================================================================

class StateTransformer:
    """Handles state transformations between supervisor and agents."""
    
    @staticmethod
    def supervisor_to_planner(supervisor_state: SupervisorState):
        """Transform supervisor state to planner supervisor state."""
        from .planner.planner_supervisor_state import PlannerSupervisorState, PlanningWorkflowState
        
        return PlannerSupervisorState(
            # Core fields
            user_request=supervisor_state.user_request,
            session_id=supervisor_state.session_id,
            task_id=supervisor_state.task_id,
            status="in_progress",
            
            # Messages - pass only the last human message for context
            messages=[msg for msg in supervisor_state.messages if hasattr(msg, 'content')][-1:] if supervisor_state.messages else [],
            
            # LLM input messages - required by langgraph-supervisor
            llm_input_messages=[msg for msg in supervisor_state.messages if hasattr(msg, 'content')][-1:] if supervisor_state.messages else [],
            
            # Planner-specific fields
            active_agent="planner_sub_supervisor",
            task_description=supervisor_state.user_request,
            
            # Workspace and context
            workspace_ref=supervisor_state.workspace_ref,
            terraform_context=supervisor_state.terraform_context or {},
            
            # Planning-specific data
            requirements_analysis={},
            execution_plan={},
            provider_versions={},
            standards_profile={},
            
            # Planning workflow state - use distinct field name to avoid conflict with supervisor's workflow_state
            planning_workflow_state=PlanningWorkflowState(),
        )
    
    @staticmethod
    def supervisor_to_generation(supervisor_state: SupervisorState) -> GenerationState:
        """Transform supervisor state to generation state with actual planner output structure."""
        planner_data = supervisor_state.planner_data or {}
        execution_data = planner_data.get("execution_data", {})
        execution_plan_data = execution_data.get("execution_plan_data", {})
        
        # Extract execution plans from the nested structure
        execution_plans = execution_plan_data.get("execution_plans", [])
        primary_execution_plan = execution_plans[0] if execution_plans else {}
        
        return GenerationState(
            requirements=planner_data.get("requirements_data", {}),
            provider_versions=primary_execution_plan.get("required_providers", {}),
            registry_schemas_ref=supervisor_state.workspace_ref,
            standards_profile=planner_data.get("standards_profile", {}),
            module_name=primary_execution_plan.get("module_name", f"module_{supervisor_state.workflow_id[:8]}"),
            
            # Direct execution plan data from actual structure
            execution_plan=primary_execution_plan,
            resource_configurations=primary_execution_plan.get("resource_configurations", []),
            variable_definitions=primary_execution_plan.get("variable_definitions", []),
            data_sources=primary_execution_plan.get("data_sources", []),
            local_values=primary_execution_plan.get("local_values", []),
            dependencies=primary_execution_plan.get("resource_dependencies", []),
            security_considerations=primary_execution_plan.get("security_considerations", []),
            cost_estimates=primary_execution_plan.get("estimated_costs", {})
        )

    @staticmethod
    def supervisor_to_writer_react(supervisor_state: SupervisorState) -> WriterReactState:
        """Transform supervisor state to writer react state."""

        if isinstance(supervisor_state, dict):
            generation_data = supervisor_state.get("generation_data", {})
            planner_data = supervisor_state.get("planner_data", {})
            session_id = supervisor_state.get("session_id", None)
            task_id = supervisor_state.get("task_id", None)
        else:
            generation_data = supervisor_state.generation_data
            planner_data = supervisor_state.planner_data
            session_id = supervisor_state.session_id
            task_id = supervisor_state.task_id

        module_structure_plan = planner_data.get("execution_data", {}).get("module_structure_plan", {}).get("module_structure_plans", [])[0].get("recommended_files", [])
        module_name = planner_data.get("execution_data", {}).get("execution_plan_data", {}).get("execution_plans", [])[0].get("module_name", "terraform-module")

        return WriterReactState(
            module_name=module_name,
            session_id=session_id,
            task_id=task_id,
            generation_data=generation_data,
            module_structure_plan=module_structure_plan,
            files_to_write=[],  # Will be populated by _process_generation_data
            status=WriteStatus.PENDING,
            errors=[],
            warnings=[],
            retry_count=0
        )

    @staticmethod
    def supervisor_to_generator_swarm(supervisor_state: SupervisorState) -> GeneratorSwarmState:
        """Transform supervisor state to generator swarm state with actual planner output structure."""
        # Extract planner data and execution plan from the actual structure
        # messages = supervisor_state.messages
        # llm_input_messages = supervisor_state.llm_input_messages
        
        # Handle both dict and SupervisorState objects
        if isinstance(supervisor_state, dict):
            planner_data = supervisor_state.get("planner_data", {})
            session_id = supervisor_state.get("session_id")
            task_id = supervisor_state.get("task_id")
        else:
            planner_data = supervisor_state.planner_data or {}
            session_id = supervisor_state.session_id
            task_id = supervisor_state.task_id
            
        execution_data = planner_data.get("execution_data", {})
        execution_plan_data = execution_data.get("execution_plan_data", {})
        
        # Extract execution plans from the nested structure
        execution_plans = execution_plan_data.get("execution_plans", [])
        primary_execution_plan = execution_plans[0] if execution_plans else {}
        
        # Extract specific data sections for each agent
        resource_configurations = primary_execution_plan.get("resource_configurations", [])
        variable_definitions = primary_execution_plan.get("variable_definitions", [])
        data_sources = primary_execution_plan.get("data_sources", [])
        local_values = primary_execution_plan.get("local_values", [])
        output_definitions = primary_execution_plan.get("output_definitions", [])
        dependencies = primary_execution_plan.get("resource_dependencies", [])
        security_considerations = primary_execution_plan.get("security_considerations", [])
        cost_estimates = primary_execution_plan.get("estimated_costs", {})
        
        # Create default human message for generator swarm
        default_message = HumanMessage(content="Generate Terraform module from execution plan")
        
        return GeneratorSwarmState(
            active_agent="resource_configuration_agent",
            stage_status="planning_active",
            planning_progress={
                "resource_configuration_agent": 0.0,
                "variable_definition_agent": 0.0,
                "data_source_agent": 0.0,
                "local_values_agent": 0.0,
                "output_definition_agent": 0.0,
                "terraform_backend_generator": 0.0,
                "terraform_readme_generator": 0.0
            },
            messages=[default_message],
            llm_input_messages=[default_message],
            agent_status_matrix={
                "resource_configuration_agent": GeneratorAgentStatus.INACTIVE,
                "variable_definition_agent": GeneratorAgentStatus.INACTIVE,
                "data_source_agent": GeneratorAgentStatus.INACTIVE,
                "local_values_agent": GeneratorAgentStatus.INACTIVE,
                "output_definition_agent": GeneratorAgentStatus.INACTIVE,
                "terraform_backend_generator": GeneratorAgentStatus.INACTIVE,
                "terraform_readme_generator": GeneratorAgentStatus.INACTIVE
            },
            # Required fields with defaults
            pending_dependencies={},
            resolved_dependencies={},
            dependency_graph={},
            agent_workspaces={
                "resource_configuration_agent": {},
                "variable_definition_agent": {},
                "data_source_agent": {},
                "local_values_agent": {},
                "output_definition_agent": {},
                "terraform_backend_generator": {},
                "terraform_readme_generator": {}
            },
            session_id=session_id,
            task_id=task_id,
            handoff_queue=[],
            communication_log=[],
            checkpoint_metadata={},
            recovery_context=None,
            approval_required=False,
            approval_context={},
            pending_human_decisions=[],
            
            # Planner data fields - extracted from nested structure
            execution_plan_data=execution_plan_data,
            state_management_plan_data=execution_data.get("state_management_data"),
            configuration_optimizer_plan_data=execution_data.get("configuration_optimizer_data"),
            
            # Additional context fields for enhanced functionality
            planning_context={
                "dependencies": dependencies,
                "security_considerations": security_considerations,
                "cost_estimates": cost_estimates,
                "module_name": primary_execution_plan.get("module_name", "terraform-module"),
                "service_name": primary_execution_plan.get("service_name", "Unknown Service"),
                "target_environment": primary_execution_plan.get("target_environment", "prod"),
                "execution_plan": primary_execution_plan,
                "planner_data": planner_data,
                # Additional context from actual structure
                "module_structure_plan": execution_data.get("module_structure_plan", {}),
                "configuration_optimizer_data": execution_data.get("configuration_optimizer_data", {}),
                "state_management_data": execution_data.get("state_management_data", {}),
                "requirements_data": planner_data.get("requirements_data", {}),
                "terraform_files": primary_execution_plan.get("terraform_files", []),
                "required_providers": primary_execution_plan.get("required_providers", {}),
                "terraform_version_constraint": primary_execution_plan.get("terraform_version_constraint", ">= 1.0.0")
            },
            
            # Stage progress tracking
            stage_progress={
                "planning": 0.0,
                "enhancement": 0.0,
                "integration": 0.0
            },
            
            # Current stage
            current_stage="planning",
            
            # Generator-specific context
            generation_context={
                "module_name": primary_execution_plan.get("module_name", "terraform-module"),
                "service_name": primary_execution_plan.get("service_name", "Unknown Service"),
                "target_environment": primary_execution_plan.get("target_environment", "prod"),
                "terraform_version": primary_execution_plan.get("terraform_version_constraint", ">= 1.0.0"),
                "provider_versions": primary_execution_plan.get("required_providers", {})
            },
            
            # Completion metrics
            completion_metrics={
                "total_agents": 5,
                "completed_agents": 0,
                "generation_started_at": None,
                "generation_completed_at": None
            }
        )
    
    @staticmethod
    def generator_to_supervisor(generator_state: GeneratorSwarmState) -> Dict[str, Any]:
        """
        Transform generator state back to supervisor updates.
        
        Args:
            generator_state: Final generator state
            
        Returns:
            Dict[str, Any]: Updates to merge into supervisor state
        """
        # Extract generated content from agent workspaces with safe access
        agent_workspaces = generator_state.get("agent_workspaces", {})
        generated_resources_block = agent_workspaces.get("resource_configuration_agent", {}).get("complete_resources_file", "")
        generated_variables_block = agent_workspaces.get("variable_definition_agent", {}).get("complete_variables_file", "")
        generated_data_sources_block = agent_workspaces.get("data_source_agent", {}).get("complete_data_sources_file", "")
        generated_locals_block = agent_workspaces.get("local_values_agent", {}).get("complete_locals_file", "")
        generated_outputs_block = agent_workspaces.get("output_definition_agent", {}).get("complete_outputs_file", "")
        generated_backend_block = agent_workspaces.get("terraform_backend_generator", {}).get("complete_configuration", "")
        generated_readme_block = agent_workspaces.get("terraform_readme_generator", {}).get("readme_content", "")
        generation_completion_msg = HumanMessage(content=f"Module Generation completed for module: {generator_state.get('generation_context', {}).get('module_name', 'Unknown')}")
        # Create generation data for supervisor
        generation_data = {
            "generated_module": {
                "resources": generated_resources_block,
                "variables": generated_variables_block,
                "data_sources": generated_data_sources_block,
                "locals": generated_locals_block,
                "outputs": generated_outputs_block,
                "backend": generated_backend_block,
                "readme": generated_readme_block
            },
            "agent_status_matrix": generator_state.get("agent_status_matrix", {}),
            "status": "completed" if generator_state.get("stage_status") == "planning_complete" else "in_progress"
        }
        
        # Return supervisor updates
        return {
            "generation_data": generation_data,
            "messages": generator_state.get("messages", []),
            "llm_input_messages": [generation_completion_msg]  # Pass messages back to supervisor
        }
    
    @staticmethod
    def writer_to_supervisor(writer_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform writer state back to supervisor updates.

        Args:
            writer_state: Final writer state

        Returns:
            Dict[str, Any]: Updates to merge into supervisor state
        """
        # Extract completion data from writer state
        completion_status = writer_state.get("completion_status", "completed")
        completion_summary = writer_state.get("completion_summary", "")
        completion_files_created = writer_state.get("completion_files_created", [])

        writer_completion_msg = HumanMessage(content=f"Writer agent execution completed successfully and has written the following files: {completion_files_created}")

        # Create writer data for supervisor
        writer_data = {
            "status": completion_status,
            "summary": completion_summary,
            "files_created": completion_files_created
        }

        # Return supervisor updates
        return {
            "writer_data": writer_data,
            "llm_input_messages": [writer_completion_msg]
        }
    
    @staticmethod
    def supervisor_to_validation(supervisor_state: SupervisorState) -> ValidationState:
        """Transform supervisor state to validation state."""
        return ValidationState(
            module_ref=supervisor_state.generated_module_ref,
            workspace_ref=supervisor_state.workspace_ref,
            policy_sets=supervisor_state.terraform_context.get("policy_sets", []) if supervisor_state.terraform_context else [],
            quota_profile=supervisor_state.terraform_context.get("quota_profile", {}) if supervisor_state.terraform_context else {},
        )
    
    @staticmethod
    def supervisor_to_editor(supervisor_state: SupervisorState) -> EditorState:
        """Transform supervisor state to editor state."""
        return EditorState(
            target_config_ref=supervisor_state.generated_module_ref,
            change_request=supervisor_state.terraform_context or {},
        )
    
    @staticmethod
    def supervisor_to_security(supervisor_state: SupervisorState) -> SecurityState:
        """Transform supervisor state to security state."""
        return SecurityState(
            module_ref=supervisor_state.generated_module_ref,
            security_policies=supervisor_state.terraform_context.get("security_policies", []) if supervisor_state.terraform_context else [],
            compliance_frameworks=supervisor_state.terraform_context.get("compliance_frameworks", []) if supervisor_state.terraform_context else [],
        )
    
    @staticmethod
    def supervisor_to_cost(supervisor_state: SupervisorState) -> CostState:
        """Transform supervisor state to cost state."""
        validation_data = supervisor_state.validation_data or {}
        return CostState(
            plan_json_ref=validation_data.get("terraform_plan", {}).get("plan_json_ref"),
            region=supervisor_state.terraform_context.get("region", "us-east-1") if supervisor_state.terraform_context else "us-east-1",
        )
    
    @staticmethod
    def planner_to_supervisor(planner_state) -> Dict[str, Any]:
        """Transform planner supervisor state back to supervisor updates."""
        return {
            "planner_data": {
                "requirements_analysis": planner_state.requirements_analysis,
                "execution_plan": planner_state.execution_plan,
                "provider_versions": planner_state.provider_versions,
                "standards_profile": planner_state.standards_profile,
            },
            "question": planner_state.question,
            "current_agent": AgentType.GENERATION,  # Next step after planning
            "workflow_state": {
                "planning_complete": True,
                "next_agent": "generation_agent",
            }
        }
    
    @staticmethod
    def generation_to_supervisor(generation_state: GenerationState) -> Dict[str, Any]:
        """Transform generation state back to supervisor updates."""
        return {
            "generation_data": {
                "design_rationale": generation_state.design_rationale,
                "file_plan": generation_state.file_plan,
                "template_params": generation_state.template_params,
                "module_manifest": generation_state.module_manifest,
                "warnings": generation_state.warnings,
                "best_practices_applied": generation_state.best_practices_applied,
            },
            "generated_module_ref": generation_state.generated_files_ref,
            "question": generation_state.question,
            "current_agent": AgentType.VALIDATION,
        }
    
    @staticmethod
    def generator_swarm_to_supervisor(generator_swarm_state) -> Dict[str, Any]:
        """Transform generator swarm state back to supervisor updates."""
        # Extract generated artifacts from the swarm state
        agent_workspaces = generator_swarm_state.get("agent_workspaces", {})
        
        # Collect all generated artifacts
        generated_resources = agent_workspaces.get("resource_configuration_agent", {}).get("generated_resources", [])
        generated_variables = agent_workspaces.get("variable_definition_agent", {}).get("generated_variables", [])
        generated_data_sources = agent_workspaces.get("data_source_agent", {}).get("generated_data_sources", [])
        generated_locals = agent_workspaces.get("local_values_agent", {}).get("generated_locals", [])
        generated_outputs = agent_workspaces.get("output_definition_agent", {}).get("generated_outputs", [])
        
        # Extract planning context for metadata
        planning_context = generator_swarm_state.get("planning_context", {})
        
        return {
            "generation_data": {
                "generated_resources": generated_resources,
                "generated_variables": generated_variables,
                "generated_data_sources": generated_data_sources,
                "generated_locals": generated_locals,
                "generated_outputs": generated_outputs,
                "module_name": planning_context.get("module_name", "terraform-module"),
                "service_name": planning_context.get("service_name", "Unknown Service"),
                "target_environment": planning_context.get("target_environment", "prod"),
                "stage_progress": generator_swarm_state.get("stage_progress", {}),
                "agent_status_matrix": generator_swarm_state.get("agent_status_matrix", {}),
            },
            "generated_module_ref": f"terraform-module-{planning_context.get('module_name', 'unknown')}",
            "question": None,  # Generator swarm doesn't typically ask questions
            "current_agent": AgentType.VALIDATION,  # Next step after generation
            "workflow_state": {
                "generation_complete": True,
                "next_agent": "validation_agent",
            }
        }
    
    @staticmethod
    def validation_to_supervisor(validation_state: ValidationState) -> Dict[str, Any]:
        """Transform validation state back to supervisor updates."""
        return {
            "validation_data": {
                "static_analysis": validation_state.static_analysis,
                "terraform_validate": validation_state.terraform_validate,
                "terraform_plan": validation_state.terraform_plan,
                "security_scans": validation_state.security_scans,
                "compliance_checks": validation_state.compliance_checks,
                "quota_evaluation": validation_state.quota_evaluation,
                "blockers": validation_state.blockers,
                "warnings": validation_state.warnings,
            },
            "validation_report_ref": validation_state.validation_report_ref,
            "question": validation_state.question,
            "current_agent": None,  # Supervisor decides next step
        }
    
    @staticmethod
    def editor_to_supervisor(editor_state: EditorState) -> Dict[str, Any]:
        """Transform editor state back to supervisor updates."""
        return {
            "editor_data": {
                "ast_notes": editor_state.ast_notes,
                "formatting_profile": editor_state.formatting_profile,
                "compatibility_findings": editor_state.compatibility_findings,
                "surgical_changes": editor_state.surgical_changes,
                "modified_files": editor_state.modified_files,
                "migration_notes": editor_state.migration_notes,
                "apply_risk": editor_state.apply_risk,
                "rollback_strategy": editor_state.rollback_strategy,
            },
            "generated_module_ref": editor_state.patch_ref,
            "question": editor_state.question,
            "current_agent": None,  # Supervisor decides next step
        }
    
    @staticmethod
    def security_to_supervisor(security_state: SecurityState) -> Dict[str, Any]:
        """Transform security state back to supervisor updates."""
        return {
            "security_data": {
                "iam_analysis": security_state.iam_analysis,
                "network_security": security_state.network_security,
                "data_protection": security_state.data_protection,
                "vulnerability_scan": security_state.vulnerability_scan,
                "compliance_results": security_state.compliance_results,
                "policy_violations": security_state.policy_violations,
                "security_findings": security_state.security_findings,
                "severity_counts": security_state.severity_counts,
                "required_actions": security_state.required_actions,
            },
            "security_report_ref": security_state.security_report_ref,
            "question": security_state.question,
            "current_agent": None,  # Supervisor decides next step
        }
    
    @staticmethod
    def cost_to_supervisor(cost_state: CostState) -> Dict[str, Any]:
        """Transform cost state back to supervisor updates."""
        return {
            "cost_data": {
                "monthly_estimate": cost_state.monthly_estimate,
                "annual_forecast": cost_state.annual_forecast,
                "cost_breakdown": cost_state.cost_breakdown,
                "resource_costs": cost_state.resource_costs,
                "optimization_suggestions": cost_state.optimization_suggestions,
                "potential_savings": cost_state.potential_savings,
                "budget_alerts": cost_state.budget_alerts,
                "cost_history": cost_state.cost_history,
                "trend_analysis": cost_state.trend_analysis,
            },
            "cost_report_ref": cost_state.cost_report_ref,
            "question": cost_state.question,
            "current_agent": None,  # Supervisor decides next step
        }
    
    # ============================================================================
    # SUB-AGENT TRANSFORMATION METHODS
    # ============================================================================
    
    # Note: Planner-related transformation methods have been removed.
    # The planner sub-supervisor now manages its own state internally.


# ============================================================================
# TYPE ALIASES AND UTILITIES
# ============================================================================

# Type aliases for common patterns
AgentState = Union[GenerationState, ValidationState, EditorState, SecurityState, CostState]
StateDict = Dict[str, Any]

# LangGraph-compatible state type
MessagesState = Dict[str, Any]  # Simple alias for LangGraph's MessagesState

# Utility function to get state class by agent type
def get_state_class(agent_type: AgentType) -> type:
    """Get the state class for a given agent type."""
    state_classes = {
        AgentType.GENERATION: GenerationState,
        AgentType.VALIDATION: ValidationState,
        AgentType.EDITOR: EditorState,
        AgentType.SECURITY: SecurityState,
        AgentType.COST: CostState,
    }
    return state_classes.get(agent_type, SupervisorState)

# Utility function to create state reference URI
def create_state_ref(agent_type: AgentType, workflow_id: str) -> str:
    """Create a state reference URI for an agent."""
    return f"{agent_type.value}_state_{workflow_id}"

# LangGraph-style utility functions
def create_supervisor_state(
    user_request: str,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    mcp_context: Optional[Dict[str, Any]] = None
) -> SupervisorState:
    """Create a new supervisor state following LangGraph patterns."""
    return SupervisorState(
        user_request=user_request,
        session_id=session_id,
        task_id=task_id,
        terraform_context=mcp_context or {},
        workflow_started_at=datetime.now(timezone.utc)
    )

def add_message_to_state(state: SupervisorState, role: str, content: str, **kwargs) -> SupervisorState:
    """Add a message to the supervisor state (LangGraph-style)."""
    message = {"role": role, "content": content, **kwargs}
    state.messages.append(message)
    return state

def get_last_message(state: SupervisorState) -> Optional[Dict[str, Any]]:
    """Get the last message from the state."""
    return state.messages[-1] if state.messages else None

def is_human_approval_required(state: SupervisorState) -> bool:
    """Check if human approval is required."""
    return state.human_approval_required

def set_approval_required(state: SupervisorState, context: Dict[str, Any]) -> SupervisorState:
    """Set human approval as required."""
    state.human_approval_required = True
    state.approval_context = context
    state.approval_timeout = datetime.utcnow()
    return state

def clear_approval_required(state: SupervisorState) -> SupervisorState:
    """Clear human approval requirement."""
    state.human_approval_required = False
    state.approval_context = None
    state.approval_timeout = None
    return state

# Export all state classes
__all__ = [
    # Enums
    "WorkflowStatus",
    "AgentType", 
    "ValidationStatus",
    "RiskLevel",
    
    # State schemas
    "SupervisorState",
    "GenerationState", 
    "ValidationState",
    "EditorState",
    "SecurityState",
    "CostState",
    
    # State transformation
    "StateTransformer",
    
    # Type aliases
    "AgentState",
    "StateDict",
    "MessagesState",
    
    # Utility functions
    "get_state_class",
    "create_state_ref",
    "create_supervisor_state",
    "add_message_to_state",
    "get_last_message",
    "is_human_approval_required",
    "set_approval_required",
    "clear_approval_required",
]
