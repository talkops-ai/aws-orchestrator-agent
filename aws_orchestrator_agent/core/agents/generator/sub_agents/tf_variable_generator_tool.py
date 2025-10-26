
import json
import uuid
import re
from typing import Dict, List, Any, Optional, Annotated, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger
from ..generator_state import GeneratorSwarmState
from .variable_generator_prompts import VARIABLE_DEFINITION_AGENT_SYSTEM_PROMPT, VARIABLE_DEFINITION_AGENT_USER_PROMPT_TEMPLATE, VARIABLE_DEFINITION_AGENT_USER_PROMPT_TEMPLATE_REFINED
from ..global_state import get_current_state, set_current_state, update_agent_workspace, update_current_state
from ..tf_content_compressor import TerraformDataCompressor

# Create agent logger for variable generator
variable_generator_logger = AgentLogger("VARIABLE_GENERATOR")



class TerraformVariableType(str, Enum):
    """Terraform variable types supported - extensible for dynamic discovery"""
    # Basic types
    STRING = "string"
    NUMBER = "number"
    BOOL = "bool"
    LIST = "list"
    MAP = "map"
    OBJECT = "object"
    TUPLE = "tuple"
    SET = "set"
    ANY = "any"
    
    # Complex types with constraints
    LIST_STRING = "list(string)"
    LIST_NUMBER = "list(number)"
    LIST_BOOL = "list(bool)"
    MAP_STRING = "map(string)"
    MAP_NUMBER = "map(number)"
    MAP_BOOL = "map(bool)"
    
    # Dynamic variable type support
    @classmethod
    def from_string(cls, variable_type: str) -> 'TerraformVariableType':
        """Create TerraformVariableType from string, supporting dynamic types"""
        try:
            return cls(variable_type)
        except ValueError:
            # For dynamic variable types not in enum, create a new instance
            return cls._create_dynamic_type(variable_type)
    
    @classmethod
    def _create_dynamic_type(cls, variable_type: str) -> 'TerraformVariableType':
        """Create a dynamic variable type for unknown Terraform types"""
        # This allows the system to handle any Terraform variable type dynamically
        return variable_type  # Return as string for dynamic types

class VariableComplexity(str, Enum):
    """Complexity levels for variable definitions"""
    SIMPLE = "simple"           # Basic string/number/bool variables
    MODERATE = "moderate"       # Lists and maps with simple structures
    COMPLEX = "complex"         # Objects with multiple attributes
    ADVANCED = "advanced"       # Nested objects, tuples, complex validation

class VariableSensitivity(str, Enum):
    """Sensitivity levels for variables"""
    PUBLIC = "public"           # Non-sensitive data
    INTERNAL = "internal"       # Internal configuration data
    CONFIDENTIAL = "confidential"   # Passwords, keys without encryption
    SECRET = "secret"           # Highly sensitive encrypted data

class GeneratorAgentName(str, Enum):
    RESOURCE_CONFIGURATION = "resource_configuration_agent"
    VARIABLE_DEFINITION = "variable_definition_agent"
    DATA_SOURCE = "data_source_agent"
    LOCAL_VALUES = "local_values_agent"
    OUTPUT_DEFINITION = "output_definition_agent"

class TerraformValidationRule(BaseModel):
    """Individual validation rule for a variable"""
    
    condition: str = Field(..., description="Terraform validation condition expression")
    error_message: str = Field(..., description="Error message if validation fails")
    description: str = Field(default="", description="Description of what this validation checks")
    
    @field_validator('condition')
    @classmethod
    def validate_condition_syntax(cls, v):
        # Very lenient validation - just ensure it's a string
        if v is None:
            return ""
        return str(v).strip()

class TerraformVariableBlock(BaseModel):
    """Individual Terraform variable block specification"""
    
    name: str = Field(..., description="Variable name")
    type_constraint: Union[TerraformVariableType, str] = Field(..., description="Variable type constraint (from planner or dynamic discovery)")
    description: str = Field(..., description="Variable description")
    
    # Value characteristics
    default_value: Optional[Any] = Field(None, description="Default value if any")
    sensitive: bool = Field(default=False, description="Mark as sensitive in output")
    nullable: bool = Field(default=False, description="Allow null values")
    
    # Validation
    validation_rules: List[TerraformValidationRule] = Field(default_factory=list, description="Validation rules")
    
    # Metadata
    complexity_level: VariableComplexity = Field(..., description="Complexity assessment")
    sensitivity_level: VariableSensitivity = Field(..., description="Sensitivity classification")
    
    # Usage context
    usage_purpose: str = Field(..., description="Purpose and usage description")
    example_values: List[Any] = Field(default_factory=list, description="Example valid values")
    related_resources: List[str] = Field(default_factory=list, description="Resources that use this variable")
    
    # Dependencies
    depends_on_variables: List[str] = Field(default_factory=list, description="Other variables this depends on")
    used_by_locals: List[str] = Field(default_factory=list, description="Local values that reference this variable")
    used_by_resources: List[str] = Field(default_factory=list, description="Resources that reference this variable")
    
    # Generated HCL
    hcl_block: str = Field(..., description="Complete HCL variable block")
    
    # Organizational metadata
    category: str = Field(default="general", description="Variable category (networking, compute, security, etc.)")
    tags: List[str] = Field(default_factory=list, description="Organizational tags")
    
    @field_validator('name')
    @classmethod
    def validate_variable_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Variable name must be alphanumeric with underscores/hyphens only')
        if v[0].isdigit():
            raise ValueError('Variable name cannot start with a number')
        if len(v) > 64:
            raise ValueError('Variable name cannot exceed 64 characters')
        return v

class DiscoveredVariableDependency(BaseModel):
    """Dependency discovered that requires handoff to another agent"""
    
    dependency_id: str = Field(..., description="Unique dependency identifier")
    dependency_type: str = Field(..., description="Type of dependency discovered")
    target_agent: GeneratorAgentName = Field(..., description="Agent that should handle this dependency")
    
    # Context for handoff
    source_variable: str = Field(..., description="Variable that triggered this dependency")
    requirement_details: Dict[str, Any] = Field(..., description="Specific details about what's needed")
    priority_level: int = Field(default=3, ge=1, le=5, description="Priority level (1=low, 5=critical)")
    
    # Handoff payload
    handoff_context: Dict[str, Any] = Field(..., description="Context to pass to target agent")
    expected_response: Dict[str, str] = Field(..., description="Expected response format from target agent")
    
    # Variable context
    variable_requirements: Dict[str, Any] = Field(..., description="Specific variable requirements")
    validation_needs: List[str] = Field(default_factory=list, description="Validation requirements")
    
    # Timing and blocking
    is_blocking: bool = Field(default=True, description="Whether variable generation should wait")
    timeout_minutes: int = Field(default=30, description="Maximum time to wait for dependency resolution")

class VariableGenerationMetrics(BaseModel):
    """Metrics and performance data for variable generation"""
    
    total_variables_generated: int = Field(..., description="Number of variables successfully generated")
    generation_duration_seconds: float = Field(..., description="Time taken for generation")
    dependencies_discovered: int = Field(..., description="Total dependencies found")
    handoffs_required: int = Field(..., description="Number of handoffs needed")
    
    # Variable type breakdown
    variable_type_counts: Dict[Union[TerraformVariableType, str], int] = Field(default_factory=dict)
    complexity_distribution: Dict[VariableComplexity, int] = Field(default_factory=dict)
    sensitivity_distribution: Dict[VariableSensitivity, int] = Field(default_factory=dict)
    
    # Validation statistics
    total_validation_rules: int = Field(default=0, description="Total validation rules across all variables")
    variables_with_defaults: int = Field(default=0, description="Variables with default values")
    required_variables: int = Field(default=0, description="Variables without defaults (required)")
    sensitive_variables: int = Field(default=0, description="Variables marked as sensitive")
    
    # Category statistics
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Error tracking
    validation_errors: List[str] = Field(default_factory=list)
    warning_messages: List[str] = Field(default_factory=list)
    optimization_recommendations: List[str] = Field(default_factory=list)

class VariableHandoffRecommendation(BaseModel):
    """Recommendation for agent handoff with variable context"""
    
    target_agent: GeneratorAgentName = Field(..., description="Recommended target agent")
    handoff_reason: str = Field(..., description="Reason for handoff")
    handoff_priority: int = Field(default=3, ge=1, le=5)
    
    # Handoff data
    context_payload: Dict[str, Any] = Field(..., description="Data to pass to target agent")
    expected_deliverables: List[str] = Field(..., description="What the target agent should produce")
    
    # Variable specific context
    variable_context: Dict[str, Any] = Field(..., description="Specific variable context for handoff")
    validation_requirements: Dict[str, Any] = Field(..., description="Validation requirements for handoff")
    
    # Coordination
    should_wait_for_completion: bool = Field(default=True)
    can_continue_parallel: bool = Field(default=False)

class TerraformVariableGenerationResponse(BaseModel):
    """Complete response from generate_terraform_variables tool"""
    
    # Generation results
    generated_variables: List[TerraformVariableBlock] = Field(..., description="Successfully generated variable blocks")
    discovered_dependencies: List[DiscoveredVariableDependency] = Field(default_factory=list, description="Dependencies requiring handoffs")
    
    # Agent coordination
    handoff_recommendations: List[VariableHandoffRecommendation] = Field(default_factory=list, description="Recommended handoffs to other agents")
    completion_status: str = Field(..., description="Current completion status")
    next_recommended_action: str = Field(..., description="Recommended next action")
    
    # Generation metadata
    generation_metadata: VariableGenerationMetrics = Field(..., description="Generation performance metrics")
    generation_timestamp: Optional[datetime] = Field(default=None, description="Timestamp when variables were generated")
    
    # Complete variables file
    complete_variables_file: str = Field(..., description="Complete variables.tf file content")
    
    # State updates
    state_updates: Dict[str, Any] = Field(default_factory=dict, description="Updates to apply to swarm state")
    workspace_updates: Dict[str, Any] = Field(..., description="Updates for agent workspace")
    
    # Error handling
    critical_errors: List[str] = Field(default_factory=list, description="Critical errors that block progress")
    recoverable_warnings: List[str] = Field(default_factory=list, description="Warnings that don't block progress")
    
    # Checkpoint data
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict, description="Data for checkpoint creation")
    
    @field_validator('completion_status')
    @classmethod
    def validate_completion_status(cls, v):
        valid_statuses = [
            'in_progress', 
            'completed', 
            'blocked', 
            'error', 
            'waiting_for_dependencies',
            'no_variables_found',
            'completed_no_variables',
            'completed_with_dependencies',
            'partial_completion',
            'requires_human_review',
            'escalated'
        ]
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v

@tool("generate_terraform_variables")
def generate_terraform_variables(
    state: Annotated[Any, InjectedState] = None,
) -> TerraformVariableGenerationResponse:
    """
    Generate Terraform input variables from execution plan specifications and agent requests.
    
    Args:
        state: GeneratorSwarmState containing all the data (execution_plan_data, agent_workspaces, planning_context)
               
    This tool analyzes parameterization needs, generates variables with proper types and validation,
    identifies dependencies, and provides handoff recommendations to other agents. It supports
    both planner specifications and dynamic agent communication.
    """

    start_time = datetime.now()

    last_3_messages = state.get('messages', [])[-4:]
    
    # Check last 3 messages for ToolMessage types and extract state updates from model_extra
    tool_message_analysis = {}
    for i, msg in enumerate(last_3_messages):
        if isinstance(msg, ToolMessage):
            # Extract model_extra field
            model_extra = None
            if hasattr(msg, 'model_extra') and msg.model_extra:
                model_extra = msg.model_extra
            elif hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                model_extra = msg.additional_kwargs.get('model_extra')
            
            # Extract state updates from model_extra if available
            if (model_extra and 'metadata' in model_extra and 'state_updates' in model_extra['metadata'] 
                and model_extra['metadata'].get('handoff_destination') == 'variable_definition_agent'):
                state_updates = model_extra['metadata']['state_updates']
                tool_message_analysis["agent_status_matrix"] = state_updates.get('agent_status_matrix')
                tool_message_analysis["pending_dependencies"] = state_updates.get('pending_dependencies')
                tool_message_analysis["dependency_graph"] = state_updates.get('dependency_graph')
                tool_message_analysis["agent_workspaces"] = state_updates.get('agent_workspaces')
                tool_message_analysis["handoff_queue"] = state_updates.get('handoff_queue')
    
    variable_generator_logger.log_structured(
        level="INFO",
        message="Last 3 messages ToolMessage state updates analysis",
        extra={
            "total_messages": len(last_3_messages),
            "tool_message_count": len(tool_message_analysis)
        }
    )
    previous_state = get_current_state()
    if isinstance(previous_state, str):
        try:
            previous_state = json.loads(previous_state)
        except json.JSONDecodeError as e:
            previous_state = {}
            variable_generator_logger.log_structured(
                level="ERROR",
                message="Failed to parse previous state JSON",
                extra={"error": str(e)})

    try:
        execution_plan_data = previous_state.get("execution_plan_data", {})
        planning_context = previous_state.get("planning_context", {})
        generated_resources = previous_state.get("agent_workspaces", {}).get("resource_configuration_agent", {}).get("complete_resources_file", "")
        generated_variables = previous_state.get("agent_workspaces", {}).get("variable_definition_agent", {}).get("complete_variables_file", "")
        generated_data_sources = previous_state.get("agent_workspaces", {}).get("data_source_agent", {}).get("complete_data_sources_file", "")
        generated_local_values = previous_state.get("agent_workspaces", {}).get("local_values_agent", {}).get("complete_locals_file", "")
        generated_output_definitions = previous_state.get("agent_workspaces", {}).get("output_definition_agent", {}).get("complete_outputs_file", "")
        if execution_plan_data:
            planning_resource_specifications = execution_plan_data.get('execution_plans', [])[0].get('resource_configurations', [])
            planning_variable_definitions = execution_plan_data.get('execution_plans', [])[0].get('variable_definitions', [])
            planning_local_values = execution_plan_data.get('execution_plans', [])[0].get('local_values', [])
            planning_data_sources = execution_plan_data.get('execution_plans', [])[0].get('data_sources', [])
            planning_output_definitions = execution_plan_data.get('execution_plans', [])[0].get('output_definitions', [])
            planning_terraform_files = execution_plan_data.get('execution_plans', [])[0].get('terraform_files', [])
        else:
            planning_resource_specifications = []
            planning_variable_definitions = []
            planning_local_values = []
            planning_data_sources = []
            planning_output_definitions = []
            planning_terraform_files = []

        
        if tool_message_analysis:
            agent_workspaces = tool_message_analysis.get("agent_workspaces", {}).get("variable_definition_agent", {})
        else:
            agent_workspaces = {}
        
        # Extract the current task and context from the handoff
        if agent_workspaces:
            current_task = agent_workspaces.get("current_task", {})
            handoff_context = agent_workspaces.get("context", {})
            agent_workspace = {
                "current_task": current_task,
                "handoff_context": handoff_context
            }
        else:
            agent_workspace = {
                "current_task": {},
                "handoff_context": {}
            }

        # Extract data from parameters

        generation_context = planning_context
        
        variable_generator_logger.log_structured(
            level="INFO",
            message="Starting Terraform variable generation",
            extra={
                "variable_requirements_count": len(planning_variable_definitions),
                "has_execution_plan_data": bool(execution_plan_data),
                "has_agent_workspace": bool(agent_workspace),
                "has_planning_context": bool(planning_context)
            }
        )
        
        # Extract context for prompt formatting
        exec_plan = generation_context.get('execution_plan', {})
        workspace = agent_workspace

        compressor = TerraformDataCompressor()
        
        # Log original data sizes for comparison
        original_sizes = {
            'resource_specifications': len(json.dumps(planning_resource_specifications)),
            'variable_definitions': len(json.dumps(planning_variable_definitions)),
            'local_values': len(json.dumps(planning_local_values)),
            'data_sources': len(json.dumps(planning_data_sources)),
            'output_definitions': len(json.dumps(planning_output_definitions)),
            'terraform_files': len(json.dumps(planning_terraform_files)),
            'generated_resources': len(generated_resources),
            'generated_variables': len(generated_variables),
            'generated_data_sources': len(generated_data_sources),
            'generated_local_values': len(generated_local_values),
            'generated_outputs': len(generated_output_definitions)
        }
        
        compressed_data = compressor.compress_all_planning_data(
            planning_resource_specifications,
            planning_variable_definitions,
            planning_local_values,
            planning_data_sources,
            planning_output_definitions,
            planning_terraform_files,
            generated_resources,
            generated_variables,
            generated_data_sources,
            generated_local_values,
            generated_output_definitions,
            extract_configuration_optimizer_data(generation_context)
        )
        
        # Log compression results
        compressed_sizes = {key: len(value) for key, value in compressed_data.items()}
        total_original = sum(original_sizes.values())
        total_compressed = sum(compressed_sizes.values())
        compression_ratio = (total_original - total_compressed) / total_original * 100 if total_original > 0 else 0

        variable_generator_logger.log_structured(
            level="INFO",
            message="Data compression completed successfully",
            extra={
                "original_total_chars": total_original,
                "compressed_total_chars": total_compressed,
                "compression_ratio_percent": round(compression_ratio, 2),
                "original_sizes": original_sizes,
                "compressed_sizes": compressed_sizes
            }
        )
        # Format user prompt with actual data, escaping curly braces in JSON
        def escape_json_for_template(json_str):
            """Escape curly braces in JSON strings for template compatibility"""
            return json_str.replace('{', '{{').replace('}', '}}')
        
        formatted_user_prompt = VARIABLE_DEFINITION_AGENT_USER_PROMPT_TEMPLATE_REFINED.format(
            service_name=exec_plan.get('service_name', 'unknown'),
            module_name=exec_plan.get('module_name', 'unknown'),
            target_environment=exec_plan.get('target_environment', 'development'),
            generation_id=agent_workspace.get('generation_id', str(uuid.uuid4())),
            variable_specifications=escape_json_for_template(compressed_data['variable_definitions']),
            planning_resources=escape_json_for_template(compressed_data['resource_specifications']),
            planning_local_values=escape_json_for_template(compressed_data['local_values']),
            planning_data_sources=escape_json_for_template(compressed_data['data_sources']),
            planning_output_definitions=escape_json_for_template(compressed_data['output_definitions']),
            # planning_terraform_files=escape_json_for_template(compressed_data['terraform_files']),
            current_stage=planning_context.get('current_stage', 'planning'),
            active_agent=agent_workspace.get('active_agent', 'variable_definition_agent'),
            workspace_generated_resources=escape_json_for_template(compressed_data['workspace_generated_resources']),
            workspace_generated_variables=escape_json_for_template(compressed_data['workspace_generated_variables']),
            workspace_generated_data_sources=escape_json_for_template(compressed_data['workspace_generated_data_sources']),
            workspace_generated_local_values=escape_json_for_template(compressed_data['workspace_generated_local_values']),
            workspace_generated_outputs=escape_json_for_template(compressed_data['workspace_generated_outputs']),
            # specific_requirements_patterns=extract_specific_requirements(generation_context),
            # configuration_optimizer_actionable=escape_json_for_template(compressed_data['optimizer_data']),
            handoff_context=escape_json_for_template(json.dumps(agent_workspace.get('handoff_context', {}), indent=2))
        )
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=TerraformVariableGenerationResponse)
        
        # Build complete prompt, escaping curly braces in system prompt
        escaped_system_prompt = VARIABLE_DEFINITION_AGENT_SYSTEM_PROMPT.replace('{', '{{').replace('}', '}}')
        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system_prompt),
            ("user", formatted_user_prompt),
            ("user", """CRITICAL: You MUST respond with valid JSON matching the TerraformVariableGenerationResponse schema.

REQUIRED FIELDS (ALL MUST BE PRESENT):
- generated_variables: List of TerraformVariableBlock objects
- discovered_dependencies: List of DiscoveredVariableDependency objects  
- handoff_recommendations: List of VariableHandoffRecommendation objects
- completion_status: String (e.g., "completed", "in_progress", "blocked", "error")
- next_recommended_action: String describing next action
- generation_metadata: VariableGenerationMetrics object with all required sub-fields
- generation_timestamp: ISO timestamp string
- complete_variables_file: String with complete variables.tf content
- state_updates: Dict with swarm state updates
- workspace_updates: Dict with agent workspace updates
- critical_errors: List of error strings
- recoverable_warnings: List of warning strings
- checkpoint_data: Dict with checkpoint information

Format Instructions:
{format_instructions}

RESPOND WITH COMPLETE JSON ONLY - NO OTHER TEXT.""")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # Create and execute chain using centralized LLM
        try:
            # Get LLM configuration from centralized config
            config_instance = Config()
            llm_config = config_instance.get_llm_config()
            
            variable_generator_logger.log_structured(
                level="DEBUG",
                message="Initializing LLM for variable generation",
                extra={
                    "llm_provider": llm_config.get('provider'),
                    "llm_model": llm_config.get('model'),
                    "llm_temperature": llm_config.get('temperature'),
                    "llm_max_tokens": llm_config.get('max_tokens')
                }
            )
            
            model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            
            llm_higher_config = config_instance.get_llm_higher_config()
            model_higher = LLMProvider.create_llm(
                provider=llm_higher_config['provider'],
                model=llm_higher_config['model'],
                temperature=llm_higher_config['temperature'],
                max_tokens=llm_higher_config['max_tokens']
            )

            llm_react_config = config_instance.get_llm_react_config()
            model_react = LLMProvider.create_llm(
                provider=llm_react_config['provider'],
                model=llm_react_config['model'],
                temperature=llm_react_config['temperature'],
                max_tokens=llm_react_config['max_tokens']
            )
            variable_generator_logger.log_structured(
                level="DEBUG",
                message="LLM initialized successfully for variable generation",
                extra={
                    "model_type": type(model).__name__
                }
            )
        except Exception as e:
            variable_generator_logger.log_structured(
                level="ERROR",
                message="Failed to initialize LLM for variable generation",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
        
        chain = prompt | model_higher | parser
        
        variable_generator_logger.log_structured(
            level="DEBUG",
            message="Executing LLM chain for variable generation",
            extra={
                "prompt_length": len(formatted_user_prompt),
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        
        # Execute the chain
        try:
            llm_response = chain.invoke({})
            
            variable_generator_logger.log_structured(
                level="DEBUG",
                message="LLM response received, starting post-processing",
                extra={
                    "generated_variables_count": len(llm_response.generated_variables),
                    "discovered_dependencies_count": len(llm_response.discovered_dependencies),
                    "generation_id": agent_workspace.get('generation_id', 'unknown')
                }
            )
        except Exception as parse_error:
            variable_generator_logger.log_structured(
                level="ERROR",
                message="Failed to parse LLM response as TerraformVariableGenerationResponse",
                extra={
                    "error": str(parse_error),
                    "error_type": type(parse_error).__name__,
                    "generation_id": agent_workspace.get('generation_id', 'unknown')
                }
            )
            # Create a minimal valid response to prevent complete failure
            from .tf_variable_generator_tool import VariableGenerationMetrics
            llm_response = TerraformVariableGenerationResponse(
                generated_variables=[],
                discovered_dependencies=[],
                handoff_recommendations=[],
                completion_status="error",
                next_recommended_action="retry_with_simplified_prompt",
                generation_metadata=VariableGenerationMetrics(
                    total_variables_generated=0,
                    generation_duration_seconds=(datetime.now() - start_time).total_seconds(),
                    dependencies_discovered=0,
                    handoffs_required=0,
                    validation_errors=[f"LLM response parsing failed: {str(parse_error)}"]
                ),
                generation_timestamp=datetime.now(),
                complete_variables_file="",
                state_updates={},
                workspace_updates={
                    "error": str(parse_error),
                    "completion_status": "error",
                    "error_timestamp": datetime.now().isoformat()
                },
                critical_errors=[f"LLM response parsing failed: {str(parse_error)}"],
                recoverable_warnings=[],
                checkpoint_data={
                    "stage": "variable_generation",
                    "agent": "variable_definition_agent",
                    "checkpoint_type": "error",
                    "error": str(parse_error),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Post-process and enhance response
        enhanced_response = post_process_variable_response(
            llm_response, 
            agent_workspace, 
            generation_context, 
            start_time
        )
        
        variable_generator_logger.log_structured(
            level="INFO",
            message="Terraform variable generation completed successfully",
            extra={
                "final_variables_count": len(enhanced_response.generated_variables),
                "final_dependencies_count": len(enhanced_response.discovered_dependencies),
                "generation_duration_seconds": enhanced_response.generation_metadata.generation_duration_seconds,
                "completion_status": enhanced_response.completion_status,
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        update_agent_workspace(
            "variable_definition_agent", {
                "complete_variables_file": enhanced_response.complete_variables_file,
                "handoff_recommendations": enhanced_response.handoff_recommendations,
                **enhanced_response.workspace_updates  # Include all workspace_updates
            }
        )
        resolved_dependencies = previous_state.get("pending_dependencies", {}).get("variable_definition_agent", [])
        # Get current resolved dependencies and append new ones
        current_resolved_deps = get_current_state().get("resolved_dependencies", {})
        updated_resolved_deps = {
            **current_resolved_deps,
            "variable_definition_agent": [
                *current_resolved_deps.get("variable_definition_agent", []),
                *resolved_dependencies
            ]
        }
        update_current_state({
            "resolved_dependencies": updated_resolved_deps
        })
        # Return only state_updates as JSON for LangGraph state management
        return enhanced_response.state_updates
        
    except Exception as e:
        generator_state = agent_workspace.get('generator_state', {})
        variable_generator_logger.log_structured(
            level="ERROR",
            message="Terraform variable generation failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "generation_id": agent_workspace.get('generation_id', 'unknown'),
                "current_stage": planning_context.get('current_stage', 'unknown')
            }
        )
        return create_variable_error_response(e, agent_workspace, datetime.now())


def extract_configuration_optimizer_data(context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration optimizer data from planner data structure"""
    optimizer_data = {}
    
    # Extract from planner data structure
    planner_data = context.get('planner_data', {})
    execution_data = planner_data.get('execution_data', {})
    configuration_optimizer_data = execution_data.get('configuration_optimizer_data', {})
    
    # Extract configuration optimizers
    if 'configuration_optimizers' in configuration_optimizer_data:
        optimizer_data['configuration_optimizers'] = configuration_optimizer_data['configuration_optimizers']
    
    return optimizer_data

def extract_specific_requirements(context: Dict[str, Any]) -> str:
    """Extract specific requirements from context"""
    requirements = []
    
    exec_plan = context.get('execution_plan', {})
    
    if 'parameterization_needs' in exec_plan:
        requirements.append(f"Parameterization: {exec_plan['parameterization_needs']}")
    
    if 'security_requirements' in exec_plan:
        requirements.append(f"Security: {exec_plan['security_requirements']}")
    
    if 'validation_requirements' in exec_plan:
        requirements.append(f"Validation: {exec_plan['validation_requirements']}")
    
    if 'flexibility_requirements' in exec_plan:
        requirements.append(f"Flexibility: {exec_plan['flexibility_requirements']}")
    
    if 'compliance_requirements' in exec_plan:
        requirements.append(f"Compliance: {exec_plan['compliance_requirements']}")
    
    return '\n'.join(requirements) if requirements else "No specific requirements specified"

def create_focused_context(generation_context: Dict[str, Any]) -> Dict[str, Any]:
    """Create a focused context with only the essential data points from test.log"""
    focused_context = {}
    
    # Extract from planner data structure
    planner_data = generation_context.get('planner_data', {})
    
    # Architecture & Requirements Data
    aws_service_mapping = planner_data.get('requirements_data', {}).get('aws_service_mapping', {})
    services = aws_service_mapping.get('services', [])
    if services:
        first_service = services[0]
        focused_context.update({
            'architecture_patterns': first_service.get('architecture_patterns'),
            'well_architected_alignment': first_service.get('well_architected_alignment'),
            'cost_optimization_recommendations': first_service.get('cost_optimization_recommendations')
        })
    
    # Security & Reusability Data
    execution_data = planner_data.get('execution_data', {})
    module_structure_plan = execution_data.get('module_structure_plan', {})
    module_structure_plans = module_structure_plan.get('module_structure_plans', [])
    if module_structure_plans:
        first_plan = module_structure_plans[0]
        focused_context.update({
            'security_considerations': first_plan.get('security_considerations'),
            'reusability_guidance': first_plan.get('reusability_guidance')
        })
    
    # Configuration Optimizer Data
    configuration_optimizer_data = execution_data.get('configuration_optimizer_data', {})
    focused_context['configuration_optimizers'] = configuration_optimizer_data.get('configuration_optimizers')
    
    # Execution Plan Data
    execution_plan_data = execution_data.get('execution_plan_data', {})
    execution_plans = execution_plan_data.get('execution_plans', [])
    if execution_plans:
        first_execution_plan = execution_plans[0]
        focused_context.update({
            'terraform_files': first_execution_plan.get('terraform_files'),
            'variable_definitions': first_execution_plan.get('variable_definitions'),
            'local_values': first_execution_plan.get('local_values'),
            'data_sources': first_execution_plan.get('data_sources'),
            'output_definitions': first_execution_plan.get('output_definitions'),
            'resource_configurations': first_execution_plan.get('resource_configurations')
        })
    
    return focused_context

def post_process_variable_response(
    llm_response: TerraformVariableGenerationResponse,
    agent_workspace: Dict[str, Any], 
    context: Dict[str, Any],
    start_time: datetime
) -> TerraformVariableGenerationResponse:
    """Post-process LLM response with comprehensive validation and enhancements"""
    
    # Create focused context with only essential data points
    focused_context = create_focused_context(context)
    
    # Calculate actual generation duration
    generation_duration = (datetime.now() - start_time).total_seconds()
    llm_response.generation_metadata.generation_duration_seconds = generation_duration
    
    # Fix generation timestamp if it's empty or invalid
    if not llm_response.generation_timestamp or llm_response.generation_timestamp == "":
        llm_response.generation_timestamp = datetime.now()
    
    # Validate generated variables with flexible validation
    validated_variables = []
    validation_errors = []
    validation_warnings = []
    
    for variable in llm_response.generated_variables:
        validation_result = validate_terraform_variable(variable)
        
        # Always include variables unless they have critical errors
        if validation_result['valid']:
            validated_variables.append(variable)
            # Add warnings to recoverable warnings
            if validation_result.get('warnings'):
                validation_warnings.extend(validation_result['warnings'])
        else:
            # Only filter out variables with critical errors
            validation_errors.extend(validation_result['errors'])
            # Attempt to fix critical issues
            fixed_variable = attempt_variable_fix(variable, validation_result['errors'])
            if fixed_variable:
                validated_variables.append(fixed_variable)
                llm_response.recoverable_warnings.append(
                    f"Fixed critical validation issues for variable '{variable.name}'"
                )
            else:
                # If we can't fix critical issues, still include the variable but log the error
                validated_variables.append(variable)
                llm_response.recoverable_warnings.append(
                    f"Variable '{variable.name}' has critical issues but included anyway"
                )
    
    # Update response with validated variables
    llm_response.generated_variables = validated_variables
    llm_response.generation_metadata.validation_errors.extend(validation_errors)
    
    # Add validation warnings to recoverable warnings
    if validation_warnings:
        llm_response.recoverable_warnings.extend(validation_warnings)
    
    # Generate complete variables file
    llm_response.complete_variables_file = generate_complete_variables_file(validated_variables)
    
    # Update metrics with actual counts
    update_generation_metrics(llm_response.generation_metadata, validated_variables)
    
    # Use original discovered dependencies without enhancement
    # The LLM already provides the necessary context for each dependency
    
    # Create comprehensive handoff recommendations
    llm_response.handoff_recommendations = create_variable_handoff_recommendations(
        llm_response.discovered_dependencies,
        validated_variables
    )
    
    # Add comprehensive state updates
    llm_response.state_updates = create_variable_state_updates(
        validated_variables,
        llm_response.discovered_dependencies,
        agent_workspace,
        llm_response.completion_status
    )
    
    # Add workspace updates
    llm_response.workspace_updates = create_variable_workspace_updates(
        validated_variables,
        llm_response.discovered_dependencies,
        llm_response.generation_metadata,
        llm_response.completion_status
    )
    
    # Add checkpoint data
    llm_response.checkpoint_data = create_variable_checkpoint_data(
        validated_variables,
        llm_response.discovered_dependencies,
        llm_response.completion_status
    )
    
    return llm_response

def validate_terraform_variable(variable: TerraformVariableBlock) -> Dict[str, Any]:
    """Validate individual Terraform variable"""
    errors = []
    warnings = []
    
    # Validate variable name - treat as warning instead of error
    if not validate_variable_name_convention(variable.name):
        warnings.append(f"Variable name '{variable.name}' doesn't follow standard conventions")
    
    # Validate type constraint - be more lenient
    type_validation = validate_variable_type_constraint(variable.type_constraint, variable.default_value)
    if not type_validation['valid']:
        # Treat type issues as warnings unless they're critical
        for error in type_validation['errors']:
            if 'compatible' in error.lower() and 'default' in error.lower():
                warnings.append(f"Type compatibility warning: {error}")
            else:
                errors.extend([error])
    if type_validation['warnings']:
        warnings.extend(type_validation['warnings'])
    
    # Validate validation rules - treat most as warnings
    for rule in variable.validation_rules:
        rule_validation = validate_validation_rule(rule, variable)
        if not rule_validation['valid']:
            # Treat most validation rule issues as warnings
            for error in rule_validation['errors']:
                if 'missing condition' in error or 'too short' in error:
                    errors.extend([error])  # Critical issues
                else:
                    warnings.extend([f"Validation rule: {error}"])  # Non-critical issues
    
    # Validate HCL block - treat as warning unless completely invalid
    if not validate_variable_hcl_block(variable.hcl_block):
        warnings.append(f"HCL block for variable '{variable.name}' may have formatting issues")
    
    # Security validation
    security_validation = validate_variable_security(variable)
    if security_validation['warnings']:
        warnings.extend(security_validation['warnings'])
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def validate_variable_name_convention(name: str) -> bool:
    """Validate Terraform variable naming convention - more lenient"""
    # Basic checks only - be more permissive
    if not name or not isinstance(name, str):
        return False
    
    # Check if name is not too long
    if len(name) > 64:
        return False
    
    # Only check for truly problematic reserved words
    critical_reserved_words = ['var', 'variable']
    if name.lower() in critical_reserved_words:
        return False
    
    # Allow more flexible naming - just ensure it's not empty and has reasonable length
    if len(name.strip()) == 0:
        return False
    
    return True

def validate_variable_type_constraint(type_constraint: Union[TerraformVariableType, str], default_value: Any) -> Dict[str, Any]:
    """Validate variable type constraint and default value compatibility with support for dynamic types"""
    errors = []
    warnings = []
    
    # Convert to string for validation
    type_constraint_str = str(type_constraint)
    
    # If default value provided, check compatibility with type
    if default_value is not None:
        type_compatibility = check_type_compatibility(type_constraint_str, default_value)
        if not type_compatibility['compatible']:
            errors.append(f"Default value type doesn't match constraint {type_constraint_str}")
        if type_compatibility['warnings']:
            warnings.extend(type_compatibility['warnings'])
    
    # Check for appropriate type usage
    if type_constraint_str == "any":
        warnings.append("Using 'any' type constraint reduces type safety - consider more specific type")
    
    # Validate dynamic type syntax
    if not validate_dynamic_type_syntax(type_constraint_str):
        errors.append(f"Invalid type constraint syntax: {type_constraint_str}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def check_type_compatibility(type_constraint: str, default_value: Any) -> Dict[str, Any]:
    """Check if default value is compatible with type constraint (supporting dynamic types)"""
    errors = []
    warnings = []
    
    # Handle None default values for required variables
    if default_value is None:
        return {'compatible': True, 'warnings': warnings}
    
    type_checks = {
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)),
        "bool": lambda v: isinstance(v, bool),
        "list": lambda v: isinstance(v, list),
        "map": lambda v: isinstance(v, dict),
        "object": lambda v: isinstance(v, dict),
        "set": lambda v: isinstance(v, (list, set)),
        "tuple": lambda v: isinstance(v, (list, tuple)),
        "any": lambda v: True  # Any type is always compatible
    }
    
    # Handle complex types like list(string), map(number), etc.
    if "(" in type_constraint and ")" in type_constraint:
        # For complex types, we're more lenient with validation
        base_type = type_constraint.split("(")[0]
        if base_type in type_checks:
            if not type_checks[base_type](default_value):
                return {'compatible': False, 'warnings': warnings}
    else:
        # For simple types
        check_func = type_checks.get(type_constraint)
        if check_func and not check_func(default_value):
            return {'compatible': False, 'warnings': warnings}
    
    return {'compatible': True, 'warnings': warnings}

def validate_dynamic_type_syntax(type_constraint: str) -> bool:
    """Validate dynamic type constraint syntax - more lenient"""
    # Basic validation for Terraform type syntax
    if not type_constraint or not isinstance(type_constraint, str):
        return False
    
    # Be more permissive - just check for basic type keywords
    type_lower = type_constraint.lower().strip()
    
    # Allow common types and variations
    basic_types = ['string', 'number', 'bool', 'any', 'object', 'list', 'map', 'set', 'tuple']
    
    # Check if it contains any basic type
    for basic_type in basic_types:
        if basic_type in type_lower:
            return True
    
    # Allow complex types with parentheses
    if '(' in type_constraint and ')' in type_constraint:
        return True
    
    # Allow any non-empty string that looks like a type
    if len(type_constraint.strip()) > 0:
        return True
    
    return False

def validate_validation_rule(rule: TerraformValidationRule, variable: TerraformVariableBlock) -> Dict[str, Any]:
    """Validate individual validation rule with more lenient parsing for complex patterns"""
    errors = []
    
    # Very lenient validation - only check for truly problematic cases
    if not rule.condition or not rule.condition.strip():
        # Only error if completely empty
        if not rule.condition:
            errors.append(f"Validation rule for {variable.name} missing condition")
        return {'valid': len(errors) == 0, 'errors': errors}
    
    # Be more lenient about variable references - just check if it looks like a condition
    condition = rule.condition.strip()
    if len(condition) < 3:  # Very short conditions might be problematic
        errors.append(f"Validation rule condition too short for {variable.name}")
    
    # Be lenient about error messages - just ensure it exists
    if not rule.error_message:
        errors.append("Validation error message is required")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_variable_hcl_block(hcl_block: str) -> bool:
    """Validate HCL block syntax for variable - more lenient"""
    try:
        if not hcl_block or not isinstance(hcl_block, str):
            return False
        
        # Very basic checks - just ensure it looks like a variable block
        hcl_lower = hcl_block.lower().strip()
        
        # Must contain 'variable' keyword
        if 'variable' not in hcl_lower:
            return False
        
        # Must have opening brace
        if '{' not in hcl_block:
            return False
        
        # Check for balanced braces (more lenient)
        open_braces = hcl_block.count('{')
        close_braces = hcl_block.count('}')
        if abs(open_braces - close_braces) > 1:  # Allow minor imbalance
            return False
        
        return True
    except Exception:
        return False

def validate_variable_security(variable: TerraformVariableBlock) -> Dict[str, Any]:
    """Validate variable security considerations"""
    warnings = []
    
    # Check for sensitive data patterns in name or description
    sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential', 'api_key']
    
    name_lower = variable.name.lower()
    desc_lower = variable.description.lower()
    
    for pattern in sensitive_patterns:
        if pattern in name_lower or pattern in desc_lower:
            if not variable.sensitive:
                warnings.append(f"Variable '{variable.name}' appears to contain sensitive data but is not marked as sensitive")
    
    # Check for default values on sensitive variables
    if variable.sensitive and variable.default_value is not None:
        warnings.append(f"Sensitive variable '{variable.name}' should not have a default value")
    
    return {
        'warnings': warnings
    }

def attempt_variable_fix(
    variable: TerraformVariableBlock, 
    errors: List[str]
) -> Optional[TerraformVariableBlock]:
    """Attempt to fix common variable issues"""
    
    fixed_variable = variable.copy(deep=True)
    
    # Try to fix naming issues
    if any("doesn't follow" in error for error in errors):
        fixed_name = fix_variable_name(variable.name)
        if fixed_name != variable.name:
            fixed_variable.name = fixed_name
            # Update HCL block with new name
            fixed_variable.hcl_block = re.sub(
                r'variable\s+"[^"]+"\s*{',
                f'variable "{fixed_name}" {{',
                fixed_variable.hcl_block
            )
    
    # Try to fix type compatibility issues
    if any("Default value type doesn't match" in error for error in errors):
        # Remove incompatible default value
        fixed_variable.default_value = None
        # Update HCL block to remove default
        fixed_variable.hcl_block = re.sub(
            r'\s*default\s*=\s*[^\n]*\n',
            '\n',
            fixed_variable.hcl_block
        )
    
    # Validate the fixed variable
    validation_result = validate_terraform_variable(fixed_variable)
    if validation_result['valid']:
        return fixed_variable
    
    return None

def fix_variable_name(name: str) -> str:
    """Fix variable naming convention issues"""
    
    # Convert to lowercase
    fixed = name.lower()
    
    # Replace invalid characters with underscores
    fixed = re.sub(r'[^a-z0-9_]', '_', fixed)
    
    # Ensure it starts with a letter
    if fixed[0].isdigit():
        fixed = 'v_' + fixed
    
    # Remove multiple consecutive underscores
    fixed = re.sub(r'_+', '_', fixed)
    
    # Remove leading/trailing underscores
    fixed = fixed.strip('_')
    
    # Ensure it's not too long
    if len(fixed) > 64:
        fixed = fixed[:64].rstrip('_')
    
    return fixed

def generate_complete_variables_file(variables: List[TerraformVariableBlock]) -> str:
    """Generate complete variables.tf file from variable blocks"""
    
    if not variables:
        return ""
    
    # Sort variables by category and then by name
    sorted_variables = sorted(variables, key=lambda x: (x.category, x.name))
    
    # Group by category
    categories = {}
    for var in sorted_variables:
        if var.category not in categories:
            categories[var.category] = []
        categories[var.category].append(var)
    
    # Generate the complete file
    lines = []
    lines.append("# Terraform Variables")
    lines.append(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    for category, vars_in_category in categories.items():
        if category != "general":
            lines.append(f"# {category.title()} Variables")
            lines.append("")
        
        for var in vars_in_category:
            lines.append(var.hcl_block)
            lines.append("")
    
    return "\n".join(lines)

def update_generation_metrics(
    metrics: VariableGenerationMetrics,
    variables: List[TerraformVariableBlock]
) -> None:
    """Update metrics with actual generated variable data"""
    
    # Update type counts
    for var in variables:
        metrics.variable_type_counts[var.type_constraint] = (
            metrics.variable_type_counts.get(var.type_constraint, 0) + 1
        )
    
    # Update complexity distribution
    for var in variables:
        metrics.complexity_distribution[var.complexity_level] = (
            metrics.complexity_distribution.get(var.complexity_level, 0) + 1
        )
    
    # Update sensitivity distribution
    for var in variables:
        metrics.sensitivity_distribution[var.sensitivity_level] = (
            metrics.sensitivity_distribution.get(var.sensitivity_level, 0) + 1
        )
    
    # Update category distribution
    for var in variables:
        metrics.category_distribution[var.category] = (
            metrics.category_distribution.get(var.category, 0) + 1
        )
    
    # Update validation statistics
    metrics.total_validation_rules = sum(len(var.validation_rules) for var in variables)
    metrics.variables_with_defaults = sum(1 for var in variables if var.default_value is not None)
    metrics.required_variables = sum(1 for var in variables if var.default_value is None)
    metrics.sensitive_variables = sum(1 for var in variables if var.sensitive)

def create_variable_handoff_recommendations(
    dependencies: List[DiscoveredVariableDependency],
    variables: List[TerraformVariableBlock]
) -> List[VariableHandoffRecommendation]:
    """Create comprehensive handoff recommendations for variables"""
    
    recommendations = []
    
    # Group dependencies by target agent
    agent_groups = {}
    for dep in dependencies:
        if dep.target_agent not in agent_groups:
            agent_groups[dep.target_agent] = []
        agent_groups[dep.target_agent].append(dep)
    
    # Create recommendations for each target agent
    for agent, deps in agent_groups.items():
        # Determine handoff characteristics
        max_priority = max(dep.priority_level for dep in deps)
        has_blocking = any(dep.is_blocking for dep in deps)
        
        # Create expected deliverables based on agent type
        if agent == 'resource_configuration_agent':
            deliverables = ['Resource configurations for variable validation', 'Resource attribute mappings']
        elif agent == 'data_source_agent':
            deliverables = ['Data source configurations for external validation', 'External data lookup specifications']
        elif agent == 'local_values_agent':
            deliverables = ['Local value definitions for complex validations', 'Computed validation expressions']
        else:
            deliverables = ['Required configurations']
        
        recommendation = VariableHandoffRecommendation(
            target_agent=agent,
            handoff_reason=f"Resolve {len(deps)} variable dependencies",
            handoff_priority=max_priority,
            context_payload={
                'dependencies': [dep.dict() for dep in deps],
                'total_count': len(deps),
                'priority_levels': [dep.priority_level for dep in deps],
                'affected_variables': [dep.source_variable for dep in deps]
            },
            expected_deliverables=deliverables,
            variable_context={
                'variable_types': [var.type_constraint for var in variables],
                'complexity_levels': [var.complexity_level for var in variables],
                'categories': list(set(var.category for var in variables))
            },
            validation_requirements={
                'validation_rules': [dep.validation_needs for dep in deps],
                'variable_requirements': [dep.variable_requirements for dep in deps]
            },
            should_wait_for_completion=has_blocking,
            can_continue_parallel=not has_blocking
        )
        
        recommendations.append(recommendation)
    
    return recommendations

def create_variable_state_updates(
    variables: List[TerraformVariableBlock],
    dependencies: List[DiscoveredVariableDependency],
    agent_workspace: Dict[str, Any],
    completion_status: str
) -> Dict[str, Any]:
    """Create comprehensive state updates for the swarm"""
    
    updates = {
        'terraform_variables': [var.dict() for var in variables],
        'pending_dependencies': {
            **agent_workspace.get('pending_dependencies', {}),
            'variable_definition_agent': [dep.dict() for dep in dependencies]
        },
        'agent_status_matrix': {
            **agent_workspace.get('agent_status_matrix', {}),
            'variable_definition_agent': completion_status
        },
        'planning_progress': {
            **agent_workspace.get('planning_progress', {}),
            'variable_definition_agent': 1.0 if completion_status == 'completed' else 0.6
        }
    }
    
    return updates

def create_variable_workspace_updates(
    variables: List[TerraformVariableBlock],
    dependencies: List[DiscoveredVariableDependency],
    metrics: VariableGenerationMetrics,
    completion_status: str
) -> Dict[str, Any]:
    """Create workspace updates for the variable definition agent"""
    
    return {
        'generated_variables': [var.dict() for var in variables],
        'pending_dependencies': [dep.dict() for dep in dependencies],
        'generation_metrics': metrics.dict(),
        'completion_status': completion_status,
        'completion_timestamp': datetime.now().isoformat(),
        'variable_summary': {
            'total_variables': len(variables),
            'variable_types': list(set(var.type_constraint for var in variables)),
            'complexity_levels': list(set(var.complexity_level for var in variables)),
            'categories': list(set(var.category for var in variables)),
            'dependencies_discovered': len(dependencies),
            'sensitive_variables': sum(1 for var in variables if var.sensitive),
            'required_variables': sum(1 for var in variables if var.default_value is None)
        }
    }

def create_variable_checkpoint_data(
    variables: List[TerraformVariableBlock],
    dependencies: List[DiscoveredVariableDependency],
    completion_status: str
) -> Dict[str, Any]:
    """Create checkpoint data for recovery"""
    
    return {
        'stage': 'planning',
        'agent': 'variable_definition_agent',
        'checkpoint_type': 'variable_generation_complete',
        'variables_generated': len(variables),
        'dependencies_discovered': len(dependencies),
        'completion_status': completion_status,
        'timestamp': datetime.now().isoformat(),
        'variable_names': [var.name for var in variables],
        'dependency_types': [dep.dependency_type for dep in dependencies],
        'variable_categories': list(set(var.category for var in variables)),
        'complexity_levels': list(set(var.complexity_level for var in variables))
    }

def create_variable_error_response(
    error: Exception, 
    agent_workspace: Dict[str, Any], 
    start_time: datetime
) -> TerraformVariableGenerationResponse:
    """Create error response when tool execution fails"""
    
    generation_duration = (datetime.now() - start_time).total_seconds()
    
    return TerraformVariableGenerationResponse(
        generated_variables=[],
        discovered_dependencies=[],
        handoff_recommendations=[],
        completion_status='error',
        next_recommended_action='escalate_to_human_or_retry',
        generation_metadata=VariableGenerationMetrics(
            total_variables_generated=0,
            generation_duration_seconds=generation_duration,
            dependencies_discovered=0,
            handoffs_required=0,
            validation_errors=[f"Tool execution failed: {str(error)}"]
        ),
        complete_variables_file="",
        state_updates={
            'agent_status_matrix': {
                **agent_workspace.get('agent_status_matrix', {}),
                'variable_definition_agent': 'error'
            }
        },
        workspace_updates={
            'error': str(error),
            'completion_status': 'error',
            'error_timestamp': datetime.now().isoformat()
        },
        critical_errors=[f"Critical tool failure: {str(error)}"],
        checkpoint_data={
            'stage': 'planning',
            'agent': 'variable_definition_agent',
            'checkpoint_type': 'error',
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
    )