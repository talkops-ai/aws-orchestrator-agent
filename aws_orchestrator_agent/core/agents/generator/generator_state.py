from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class GeneratorAgentStatus(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"  
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"

class DependencyType(Enum):
    RESOURCE_TO_VARIABLE = "resource_to_variable"
    RESOURCE_TO_DATA_SOURCE = "resource_to_data_source"
    RESOURCE_TO_LOCAL_VALUES = "resource_to_local_values"
    RESOURCE_TO_OUTPUT = "resource_to_output"
    VARIABLE_TO_LOCAL = "variable_to_local"
    VARIABLE_TO_RESOURCE = "variable_to_resource"
    VARIABLE_TO_DATA_SOURCE = "variable_to_data_source"
    VARIABLE_TO_LOCAL_VALUES = "variable_to_local_values"
    VARIABLE_TO_OUTPUT = "variable_to_output"
    DATA_SOURCE_TO_LOCAL = "data_source_to_local"
    DATA_SOURCE_TO_VARIABLE = "data_source_to_variable"
    DATA_SOURCE_TO_LOCAL_VALUES = "data_source_to_local_values"
    DATA_SOURCE_TO_RESOURCE = "data_source_to_resource"
    DATA_SOURCE_TO_OUTPUT = "data_source_to_output"
    LOCAL_TO_RESOURCE = "local_to_resource"
    LOCAL_VALUES_TO_VARIABLE = "local_values_to_variable"
    LOCAL_VALUES_TO_RESOURCE = "local_values_to_resource"
    LOCAL_VALUES_TO_DATA_SOURCE = "local_values_to_data_source"
    LOCAL_VALUES_TO_OUTPUT = "local_values_to_output"
    # Output agent dependencies
    OUTPUT_TO_RESOURCE = "output_to_resource"
    OUTPUT_TO_DATA_SOURCE = "output_to_data_source"
    OUTPUT_TO_VARIABLE = "output_to_variable"
    OUTPUT_TO_LOCAL_VALUES = "output_to_local_values"

class GeneratorSwarmState(TypedDict):
    """
    Isolated state schema for Generator Swarm subgraph.
    
    This schema has NO shared fields with SupervisorState to prevent state contamination.
    All communication with the supervisor happens through state transformation functions.
    """
    active_agent: str
    
    # Message handling for LangGraph (fixed TypedDict format)
    llm_input_messages: Annotated[List[AnyMessage], add_messages]
    messages: Annotated[List[AnyMessage], add_messages]  # Renamed from 'messages'
    
    # Planning Stage Management
    stage_status: str = "planning_active"  # planning_active, planning_complete, planning_error
    planning_progress: Dict[str, float] = {}  # agent_name -> completion percentage
    
    # Agent Coordination Matrix
    agent_status_matrix: Dict[str, GeneratorAgentStatus] = {
        "resource_configuration_agent": GeneratorAgentStatus.INACTIVE,
        "variable_definition_agent": GeneratorAgentStatus.INACTIVE,
        "data_source_agent": GeneratorAgentStatus.INACTIVE,
        "local_values_agent": GeneratorAgentStatus.INACTIVE,
        "output_definition_agent": GeneratorAgentStatus.INACTIVE,
        "terraform_backend_generator": GeneratorAgentStatus.INACTIVE,
        "terraform_readme_generator": GeneratorAgentStatus.INACTIVE
    }
    
    # Dynamic Dependency Tracking
    pending_dependencies: Dict[str, List[Dict[str, Any]]] = {}  # agent -> list of dependency requests
    resolved_dependencies: Dict[str, List[Dict[str, Any]]] = {}  # agent -> list of completed dependencies
    dependency_graph: Dict[str, Set[str]] = {}  # agent -> set of dependent agents
    
    # Agent Workspaces (Isolated but Coordinated)
    agent_workspaces: Dict[str, Dict[str, Any]] = {
        "resource_configuration_agent": {},
        "variable_definition_agent": {},
        "data_source_agent": {},
        "local_values_agent": {},
        "output_definition_agent": {},
        "terraform_backend_generator": {},
        "terraform_readme_generator": {}
    }
    execution_plan_data: Optional[Dict[str, Any]] = None
    state_management_plan_data: Optional[Dict[str, Any]] = None
    configuration_optimizer_plan_data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    generation_context: Optional[Dict[str, Any]] = None
    # Handoff Context and Communication Log
    handoff_queue: List[Dict[str, Any]] = []  # pending handoffs
    communication_log: List[Dict[str, Any]] = []  # complete handoff history
    
    # Checkpointing and Recovery
    checkpoint_metadata: Dict[str, Any] = {}
    recovery_context: Optional[Dict[str, Any]] = None
    
    # Human-in-the-Loop Integration Points
    approval_required: bool = False
    approval_context: Dict[str, Any] = {}
    pending_human_decisions: List[Dict[str, Any]] = []
    
    # Additional context fields for enhanced functionality
    planning_context: Optional[Dict[str, Any]] = None  # Additional planning context from input_transform
    stage_progress: Optional[Dict[str, float]] = None  # Stage-level progress tracking
    current_stage: Optional[str] = None  # Current stage identifier
    remaining_steps: Optional[List[str]] = None  # Remaining steps to complete
