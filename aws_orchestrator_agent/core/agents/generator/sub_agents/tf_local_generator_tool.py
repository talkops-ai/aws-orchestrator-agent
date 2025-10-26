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
from ..global_state import get_current_state, set_current_state, update_agent_workspace, update_current_state
from .local_generator_prompts import LOCAL_VALUES_AGENT_SYSTEM_PROMPT, LOCAL_VALUES_AGENT_USER_PROMPT_TEMPLATE, LOCAL_VALUES_AGENT_USER_PROMPT_TEMPLATE_REFINED
from ..tf_content_compressor import TerraformDataCompressor
# Create agent logger for local values generator
local_generator_logger = AgentLogger("LOCAL_GENERATOR")

class LocalValueType(str, Enum):
    """Types of local value expressions - extensible for dynamic discovery"""
    # Basic expression types
    SIMPLE_EXPRESSION = "simple_expression"
    CONDITIONAL_EXPRESSION = "conditional_expression"
    FOR_EXPRESSION = "for_expression"
    FUNCTION_CALL = "function_call"
    RESOURCE_REFERENCE = "resource_reference"
    DATA_REFERENCE = "data_reference"
    VARIABLE_REFERENCE = "variable_reference"
    COMPUTED_VALUE = "computed_value"
    STRING_INTERPOLATION = "string_interpolation"
    MAP_CONSTRUCTION = "map_construction"
    LIST_CONSTRUCTION = "list_construction"
    
    # Advanced expression types
    NESTED_EXPRESSION = "nested_expression"
    DYNAMIC_BLOCK_EXPRESSION = "dynamic_block_expression"
    COMPLEX_CONDITIONAL = "complex_conditional"
    MULTI_STEP_COMPUTATION = "multi_step_computation"
    
    # Dynamic local value type support
    @classmethod
    def from_string(cls, local_type: str) -> 'LocalValueType':
        """Create LocalValueType from string, supporting dynamic types"""
        try:
            return cls(local_type)
        except ValueError:
            # For dynamic local value types not in enum, create a new instance
            return cls._create_dynamic_type(local_type)
    
    @classmethod
    def _create_dynamic_type(cls, local_type: str) -> 'LocalValueType':
        """Create a dynamic local value type for unknown expression types"""
        # This allows the system to handle any local value expression type dynamically
        return local_type  # Return as string for dynamic types

class LocalValueComplexity(str, Enum):
    """Complexity levels for local values"""
    SIMPLE = "simple"           # Basic assignments and references
    MODERATE = "moderate"       # Single function calls, basic conditionals
    COMPLEX = "complex"         # Multiple functions, complex conditionals
    ADVANCED = "advanced"       # Nested expressions, complex for loops

class GeneratorAgentName(str, Enum):
    RESOURCE_CONFIGURATION = "resource_configuration_agent"
    VARIABLE_DEFINITION = "variable_definition_agent"
    DATA_SOURCE = "data_source_agent"
    LOCAL_VALUES = "local_values_agent"
    OUTPUT_DEFINITION = "output_definition_agent"

class TerraformLocalValue(BaseModel):
    """Individual Terraform local value specification"""
    
    name: str = Field(..., description="Local value name")
    expression: str = Field(..., description="Terraform expression")
    description: str = Field(..., description="Purpose and usage description")
    
    # Expression characteristics
    expression_type: Union[LocalValueType, str] = Field(..., description="Type of expression (from planner or dynamic discovery)")
    complexity_level: LocalValueComplexity = Field(..., description="Complexity assessment")
    
    # Dependencies
    depends_on_variables: List[str] = Field(default_factory=list, description="Input variables referenced")
    depends_on_resources: List[str] = Field(default_factory=list, description="Resources referenced")
    depends_on_data_sources: List[str] = Field(default_factory=list, description="Data sources referenced")
    depends_on_locals: List[str] = Field(default_factory=list, description="Other locals referenced")
    
    # Usage and context
    usage_context: str = Field(default="", description="Context where this local is used")
    used_by_resources: List[str] = Field(default_factory=list, description="Resources that use this local")
    used_by_outputs: List[str] = Field(default_factory=list, description="Outputs that use this local")
    
    # Terraform functions used
    terraform_functions_used: List[str] = Field(default_factory=list, description="Terraform functions in expression")
    
    # Generated HCL
    hcl_declaration: str = Field(..., description="Complete HCL local value declaration")
    
    # Validation and optimization
    validation_rules: List[str] = Field(default_factory=list, description="Applied validation rules")
    optimization_notes: List[str] = Field(default_factory=list, description="Performance optimization notes")
    
    @field_validator('name')
    @classmethod
    def validate_local_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Local value name must be alphanumeric with underscores/hyphens only')
        if v[0].isdigit():
            raise ValueError('Local value name cannot start with a number')
        return v

class DiscoveredLocalDependency(BaseModel):
    """Dependency discovered that requires handoff to another agent"""
    
    dependency_id: str = Field(..., description="Unique dependency identifier")
    dependency_type: str = Field(..., description="Type of dependency discovered")
    target_agent: GeneratorAgentName = Field(..., description="Agent that should handle this dependency")
    
    # Context for handoff
    source_local: str = Field(..., description="Local value that triggered this dependency")
    requirement_details: Dict[str, Any] = Field(..., description="Specific details about what's needed")
    priority_level: int = Field(default=3, ge=1, le=5, description="Priority level (1=low, 5=critical)")
    
    # Handoff payload
    handoff_context: Dict[str, Any] = Field(..., description="Context to pass to target agent")
    expected_response: Dict[str, str] = Field(..., description="Expected response format from target agent")
    
    # Expression context
    expression_fragment: str = Field(..., description="Part of expression requiring dependency")
    suggested_replacement: Optional[str] = Field(None, description="Suggested replacement after dependency resolution")
    
    # Timing and blocking
    is_blocking: bool = Field(default=True, description="Whether local generation should wait")
    timeout_minutes: int = Field(default=30, description="Maximum time to wait for dependency resolution")

class LocalValueGenerationMetrics(BaseModel):
    """Metrics and performance data for local value generation"""
    
    total_locals_generated: int = Field(..., description="Number of locals successfully generated")
    generation_duration_seconds: float = Field(..., description="Time taken for generation")
    dependencies_discovered: int = Field(..., description="Total dependencies found")
    handoffs_required: int = Field(..., description="Number of handoffs needed")
    
    # Local value type breakdown
    local_type_counts: Dict[Union[LocalValueType, str], int] = Field(default_factory=dict)
    complexity_distribution: Dict[LocalValueComplexity, int] = Field(default_factory=dict)
    
    # Expression statistics
    total_functions_used: int = Field(default=0, description="Total Terraform functions across all locals")
    unique_functions_used: List[str] = Field(default_factory=list, description="Unique Terraform functions used")
    
    # Dependency statistics
    variable_dependencies: int = Field(default=0, description="Dependencies on input variables")
    resource_dependencies: int = Field(default=0, description="Dependencies on resources")
    data_source_dependencies: int = Field(default=0, description="Dependencies on data sources")
    circular_dependencies: int = Field(default=0, description="Circular dependencies detected")
    
    # Error tracking
    validation_errors: List[str] = Field(default_factory=list)
    warning_messages: List[str] = Field(default_factory=list)
    optimization_opportunities: List[str] = Field(default_factory=list)

class LocalValueHandoffRecommendation(BaseModel):
    """Recommendation for agent handoff with local value context"""
    
    target_agent: GeneratorAgentName = Field(..., description="Recommended target agent")
    handoff_reason: str = Field(..., description="Reason for handoff")
    handoff_priority: int = Field(default=3, ge=1, le=5)
    
    # Handoff data
    context_payload: Dict[str, Any] = Field(..., description="Data to pass to target agent")
    expected_deliverables: List[str] = Field(..., description="What the target agent should produce")
    
    # Local value specific context
    local_value_context: Dict[str, Any] = Field(..., description="Specific local value context for handoff")
    expression_requirements: Dict[str, Any] = Field(..., description="Expression building requirements")
    
    # Coordination
    should_wait_for_completion: bool = Field(default=True)
    can_continue_parallel: bool = Field(default=False)

class TerraformLocalValueGenerationResponse(BaseModel):
    """Complete response from generate_terraform_locals tool"""
    
    # Generation results
    generated_locals: List[TerraformLocalValue] = Field(..., description="Successfully generated local values")
    complete_locals_file: str = Field(..., description="Complete locals.tf file content")
    discovered_dependencies: List[DiscoveredLocalDependency] = Field(default_factory=list, description="Dependencies requiring handoffs")
    
    # Agent coordination
    handoff_recommendations: List[LocalValueHandoffRecommendation] = Field(default_factory=list, description="Recommended handoffs to other agents")
    completion_status: str = Field(..., description="Current completion status")
    next_recommended_action: str = Field(..., description="Recommended next action")
    
    # Generation metadata
    generation_metadata: LocalValueGenerationMetrics = Field(..., description="Generation performance metrics")
    generation_timestamp: Optional[datetime] = Field(default=None, description="Timestamp when local values were generated")
    
    # # Complete locals block
    # complete_locals_block: str = Field(..., description="Complete HCL locals block with all generated values")
    
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
        valid_statuses = ['in_progress', 'completed', 'blocked', 'error', 'waiting_for_dependencies', 'completed_with_dependencies']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v

@tool("generate_terraform_locals")
def generate_terraform_locals(
    state: Annotated[Any, InjectedState] = None,
) -> TerraformLocalValueGenerationResponse:
    """
    Generate Terraform local values from execution plan specifications and agent requests.
    
    Args:
        state: GeneratorSwarmState containing all the data (execution_plan_data, agent_workspaces, planning_context)
               
    This tool analyzes expression needs, generates efficient local values, 
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
                and model_extra['metadata'].get('handoff_destination') == 'local_values_agent'):
                state_updates = model_extra['metadata']['state_updates']
                tool_message_analysis["agent_status_matrix"] = state_updates.get('agent_status_matrix')
                tool_message_analysis["pending_dependencies"] = state_updates.get('pending_dependencies')
                tool_message_analysis["dependency_graph"] = state_updates.get('dependency_graph')
                tool_message_analysis["agent_workspaces"] = state_updates.get('agent_workspaces')
                tool_message_analysis["handoff_queue"] = state_updates.get('handoff_queue')
    
    local_generator_logger.log_structured(
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
            local_generator_logger.log_structured(
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
            agent_workspaces = tool_message_analysis.get("agent_workspaces", {}).get("local_values_agent", {})
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
        local_value_requirements = planning_local_values
        generation_context = planning_context
        
        local_generator_logger.log_structured(
            level="INFO",
            message="Starting Terraform local values generation",
            extra={
                "local_value_requirements_count": len(local_value_requirements),
                "has_execution_plan_data": bool(execution_plan_data),
                "has_agent_workspace": bool(agent_workspace),
                "has_planning_context": bool(planning_context)
            }
        )
        
        # Extract context for prompt formatting
        exec_plan = generation_context.get('execution_plan', {})
        workspace = agent_workspace

        # Use TerraformDataCompressor for efficient data compression
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

        local_generator_logger.log_structured(
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
        
        formatted_user_prompt = LOCAL_VALUES_AGENT_USER_PROMPT_TEMPLATE_REFINED.format(
            service_name=exec_plan.get('service_name', 'unknown'),
            module_name=exec_plan.get('module_name', 'unknown'),
            target_environment=exec_plan.get('target_environment', 'development'),
            generation_id=agent_workspace.get('generation_id', str(uuid.uuid4())),
            local_value_specifications=escape_json_for_template(compressed_data['local_values']),
            planning_resources=escape_json_for_template(compressed_data['resource_specifications']),
            planning_variable_definitions=escape_json_for_template(compressed_data['variable_definitions']),
            planning_data_sources=escape_json_for_template(compressed_data['data_sources']),
            planning_output_definitions=escape_json_for_template(compressed_data['output_definitions']),
            # planning_terraform_files=escape_json_for_template(compressed_data['terraform_files']),
            current_stage=planning_context.get('current_stage', 'planning'),
            active_agent=agent_workspace.get('active_agent', 'local_values_agent'),
            workspace_generated_resources=escape_json_for_template(compressed_data['workspace_generated_resources']),
            workspace_generated_variables=escape_json_for_template(compressed_data['workspace_generated_variables']),
            workspace_generated_data_sources=escape_json_for_template(compressed_data['workspace_generated_data_sources']),
            workspace_generated_local_values=escape_json_for_template(compressed_data['workspace_generated_local_values']),
            workspace_generated_outputs=escape_json_for_template(compressed_data['workspace_generated_outputs']),
            # specific_requirements=extract_specific_requirements(generation_context),
            handoff_context=escape_json_for_template(json.dumps(agent_workspace.get('handoff_context', {}), indent=2))
        )
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=TerraformLocalValueGenerationResponse)
        
        # Build complete prompt, escaping curly braces in system prompt
        escaped_system_prompt = LOCAL_VALUES_AGENT_SYSTEM_PROMPT.replace('{', '{{').replace('}', '}}')
        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system_prompt),
            ("user", formatted_user_prompt),
            ("user", "Please respond with valid JSON matching the TerraformLocalValueGenerationResponse schema:\n{format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # Create and execute chain using centralized LLM
        try:
            # Get LLM configuration from centralized config
            config_instance = Config()
            llm_config = config_instance.get_llm_config()
            
            local_generator_logger.log_structured(
                level="DEBUG",
                message="Initializing LLM for local values generation",
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

            local_generator_logger.log_structured(
                level="DEBUG",
                message="LLM initialized successfully for local values generation",
                extra={
                    "model_type": type(model_react).__name__
                }
            )
        except Exception as e:
            local_generator_logger.log_structured(
                level="ERROR",
                message="Failed to initialize LLM for local values generation",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
        
        chain = prompt | model_react | parser
        
        local_generator_logger.log_structured(
            level="DEBUG",
            message="Executing LLM chain for local values generation",
            extra={
                "prompt_length": len(formatted_user_prompt),
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        
        # Execute the chain
        llm_response = chain.invoke({})
        
        local_generator_logger.log_structured(
            level="DEBUG",
            message="LLM response received, starting post-processing",
            extra={
                "generated_locals_count": len(llm_response.generated_locals),
                "discovered_dependencies_count": len(llm_response.discovered_dependencies),
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        
        # Post-process and enhance response
        enhanced_response = post_process_local_values_response(
            llm_response, 
            agent_workspace, 
            generation_context, 
            start_time
        )
        
        local_generator_logger.log_structured(
            level="INFO",
            message="Terraform local values generation completed successfully",
            extra={
                "final_locals_count": len(enhanced_response.generated_locals),
                "final_dependencies_count": len(enhanced_response.discovered_dependencies),
                "generation_duration_seconds": enhanced_response.generation_metadata.generation_duration_seconds,
                "completion_status": enhanced_response.completion_status,
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        update_agent_workspace(
            "local_values_agent", {
                "complete_locals_file": enhanced_response.complete_locals_file,
                "handoff_recommendations": enhanced_response.handoff_recommendations,
                **enhanced_response.workspace_updates  # Include all workspace_updates
            }
        )
        resolved_dependencies = previous_state.get("pending_dependencies", {}).get("local_values_agent", [])
        # Get current resolved dependencies and append new ones
        current_resolved_deps = get_current_state().get("resolved_dependencies", {})
        updated_resolved_deps = {
            **current_resolved_deps,
            "local_values_agent": [
                *current_resolved_deps.get("local_values_agent", []),
                *resolved_dependencies
            ]
        }
        update_current_state({
            "resolved_dependencies": updated_resolved_deps
        })
        # Return only state_updates as JSON for LangGraph state management
        return enhanced_response.state_updates
        
    except Exception as e:
        local_generator_logger.log_structured(
            level="ERROR",
            message="Terraform local values generation failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "generation_id": agent_workspace.get('generation_id', 'unknown'),
                "current_stage": planning_context.get('current_stage', 'unknown')
            }
        )
        return create_local_values_error_response(e, agent_workspace, datetime.now())


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
    
    if 'expression_requirements' in exec_plan:
        requirements.append(f"Expression Requirements: {exec_plan['expression_requirements']}")
    
    if 'computation_needs' in exec_plan:
        requirements.append(f"Computation Needs: {exec_plan['computation_needs']}")
    
    if 'naming_standards' in exec_plan:
        requirements.append(f"Naming Standards: {exec_plan['naming_standards']}")
    
    if 'tag_strategies' in exec_plan:
        requirements.append(f"Tag Strategies: {exec_plan['tag_strategies']}")
    
    if 'performance_requirements' in exec_plan:
        requirements.append(f"Performance: {exec_plan['performance_requirements']}")
    
    return '\n'.join(requirements) if requirements else "No specific requirements specified"

def post_process_local_values_response(
    llm_response: TerraformLocalValueGenerationResponse,
    agent_workspace: Dict[str, Any], 
    context: Dict[str, Any],
    start_time: datetime
) -> TerraformLocalValueGenerationResponse:
    """Post-process LLM response with comprehensive validation and enhancements"""
    
    # Calculate actual generation duration
    generation_duration = (datetime.now() - start_time).total_seconds()
    llm_response.generation_metadata.generation_duration_seconds = generation_duration
    
    # Fix generation timestamp if it's empty or invalid
    if not llm_response.generation_timestamp or llm_response.generation_timestamp == "":
        llm_response.generation_timestamp = datetime.now()
    
    # Validate generated local values with flexible validation
    validated_locals = []
    validation_errors = []
    validation_warnings = []
    
    for local_value in llm_response.generated_locals:
        validation_result = validate_terraform_local_value(local_value)
        
        # Always include local values unless they have critical errors
        if validation_result['valid']:
            validated_locals.append(local_value)
            # Add warnings to recoverable warnings
            if validation_result.get('warnings'):
                validation_warnings.extend(validation_result['warnings'])
        else:
            # Only filter out local values with critical errors
            validation_errors.extend(validation_result['errors'])
            # Attempt to fix critical issues
            fixed_local = attempt_local_value_fix(local_value, validation_result['errors'])
            if fixed_local:
                validated_locals.append(fixed_local)
                llm_response.recoverable_warnings.append(
                    f"Fixed critical validation issues for local.{local_value.name}"
                )
            else:
                # If we can't fix critical issues, still include the local value but log the error
                validated_locals.append(local_value)
                llm_response.recoverable_warnings.append(
                    f"Local '{local_value.name}' has critical issues but included anyway"
                )
    
    # Update response with validated locals
    llm_response.generated_locals = validated_locals
    llm_response.generation_metadata.validation_errors.extend(validation_errors)
    
    # Add validation warnings to recoverable warnings
    if validation_warnings:
        llm_response.recoverable_warnings.extend(validation_warnings)
    
    # Generate complete locals file from validated locals (source of truth)
    if validated_locals:
        llm_response.complete_locals_file = generate_complete_locals_file(validated_locals)
    elif llm_response.complete_locals_file:
        # Keep LLM-provided file content if present
        llm_response.complete_locals_file = llm_response.complete_locals_file
    else:
        llm_response.complete_locals_file = ""
    
    # Update metrics with actual counts
    update_generation_metrics(llm_response.generation_metadata, validated_locals)
    
    # Use original discovered dependencies without enhancement
    # The LLM already provides the necessary context for each dependency
    
    # Create comprehensive handoff recommendations
    llm_response.handoff_recommendations = create_local_value_handoff_recommendations(
        llm_response.discovered_dependencies,
        validated_locals
    )
    
    # Add comprehensive state updates
    llm_response.state_updates = create_local_value_state_updates(
        validated_locals,
        llm_response.discovered_dependencies,
        agent_workspace,
        llm_response.completion_status
    )
    
    # Add workspace updates
    llm_response.workspace_updates = create_local_value_workspace_updates(
        validated_locals,
        llm_response.discovered_dependencies,
        llm_response.generation_metadata,
        llm_response.completion_status
    )
    
    # Add checkpoint data
    llm_response.checkpoint_data = create_local_value_checkpoint_data(
        validated_locals,
        llm_response.discovered_dependencies,
        llm_response.completion_status
    )
    
    # Complete locals file already set above with proper fallback handling
    
    return llm_response

def generate_complete_locals_file(locals: List[TerraformLocalValue]) -> str:
    """Generate complete locals.tf file content from local values"""
    if not locals:
        return ""
    
    # Generate file header
    file_content = "# Local Values\n"
    file_content += "# This file contains all computed local values for the infrastructure\n\n"
    
    # Generate local values without adding extra locals {} wrapper
    # The LLM should have already provided the complete locals {} block structure
    for local_value in locals:
        file_content += f"# {local_value.description}\n"
        file_content += f"{local_value.hcl_declaration}\n\n"
    
    return file_content.strip()

def update_generation_metrics(
    metrics: LocalValueGenerationMetrics,
    locals: List[TerraformLocalValue]
) -> None:
    """Update metrics with actual generated local value data"""
    
    # Update type counts
    for local_val in locals:
        # Use expression_type (model field) instead of non-existent local_type
        metrics.local_type_counts[local_val.expression_type] = (
            metrics.local_type_counts.get(local_val.expression_type, 0) + 1
        )
    
    # Update complexity distribution
    for local_val in locals:
        metrics.complexity_distribution[local_val.complexity_level] = (
            metrics.complexity_distribution.get(local_val.complexity_level, 0) + 1
        )
    
    # Update function usage statistics
    all_functions = []
    for local_val in locals:
        if hasattr(local_val, 'functions_used') and local_val.functions_used:
            all_functions.extend(local_val.functions_used)
    
    metrics.total_functions_used = len(all_functions)
    metrics.unique_functions_used = list(set(all_functions))
    
    # Update dependency statistics
    for local_val in locals:
        if hasattr(local_val, 'variable_dependencies') and local_val.variable_dependencies:
            metrics.variable_dependencies += len(local_val.variable_dependencies)
        if hasattr(local_val, 'resource_dependencies') and local_val.resource_dependencies:
            metrics.resource_dependencies += len(local_val.resource_dependencies)
        if hasattr(local_val, 'data_source_dependencies') and local_val.data_source_dependencies:
            metrics.data_source_dependencies += len(local_val.data_source_dependencies)

def validate_terraform_local_value(local_value: TerraformLocalValue) -> Dict[str, Any]:
    """Validate individual Terraform local value"""
    errors = []
    warnings = []
    
    # Validate local name
    if not validate_local_name_convention(local_value.name):
        errors.append(f"Local name {local_value.name} doesn't follow conventions")
    
    # Validate expression syntax
    expression_validation = validate_terraform_expression(local_value.expression)
    if not expression_validation['valid']:
        errors.extend(expression_validation['errors'])
    
    # Validate function usage
    function_validation = validate_terraform_functions(local_value.expression, local_value.terraform_functions_used)
    if function_validation['errors']:
        errors.extend(function_validation['errors'])
    if function_validation['warnings']:
        warnings.extend(function_validation['warnings'])
    
    # Check for circular dependencies
    if local_value.name in local_value.depends_on_locals:
        errors.append(f"Circular dependency detected: local.{local_value.name} references itself")
    
    # Validate HCL declaration
    if not validate_local_hcl_declaration(local_value.hcl_declaration):
        errors.append(f"Invalid HCL declaration for local.{local_value.name}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def validate_local_name_convention(name: str) -> bool:
    """Validate Terraform local value naming convention"""
    # Check if name follows snake_case convention
    if not re.match(r'^[a-z][a-z0-9_]*$', name):
        return False
    
    # Check if name is not too long
    if len(name) > 64:
        return False
    
    # Check for reserved words
    reserved_words = ['local', 'locals', 'var', 'data', 'resource', 'module', 'provider']
    if name in reserved_words:
        return False
    
    return True

def validate_terraform_expression(expression: str) -> Dict[str, Any]:
    """Validate Terraform expression syntax with support for dynamic expressions"""
    errors = []
    
    # Check for balanced parentheses
    if expression.count('(') != expression.count(')'):
        errors.append("Unbalanced parentheses in expression")
    
    # Check for balanced brackets
    if expression.count('[') != expression.count(']'):
        errors.append("Unbalanced brackets in expression")
    
    # Check for balanced braces (for map/object construction)
    if expression.count('{') != expression.count('}'):
        errors.append("Unbalanced braces in expression")
    
    # Check for proper string interpolation
    interpolation_pattern = r'\$\{[^}]*\}'
    interpolations = re.findall(interpolation_pattern, expression)
    for interpolation in interpolations:
        # Check for nested interpolations (not allowed)
        inner_content = interpolation[2:-1]  # Remove ${ and }
        if '${' in inner_content:
            errors.append("Nested string interpolation is not allowed")
    
    # Check for valid reference patterns (including dynamic references)
    reference_patterns = [
        r'var\.[a-zA-Z_][a-zA-Z0-9_]*',
        r'local\.[a-zA-Z_][a-zA-Z0-9_]*',
        r'data\.[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*',
        r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*'
    ]
    
    # Validate dynamic expression syntax
    if not validate_dynamic_expression_syntax(expression):
        errors.append("Invalid dynamic expression syntax")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_dynamic_expression_syntax(expression: str) -> bool:
    """Validate dynamic expression syntax for agent-discovered expressions"""
    # Basic validation for Terraform expression syntax
    if not expression or not isinstance(expression, str):
        return False
    
    # Check for valid Terraform expression patterns
    valid_patterns = [
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+$',  # Simple assignment
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*\{.*\}$',  # Map/object assignment
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*\[.*\]$',  # List assignment
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*".*"$',  # String assignment
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*\d+$',  # Number assignment
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(true|false)$',  # Boolean assignment
    ]
    
    import re
    return any(re.match(pattern, expression.strip()) for pattern in valid_patterns)

def validate_terraform_functions(expression: str, declared_functions: List[str]) -> Dict[str, Any]:
    """Validate Terraform function usage in expression"""
    errors = []
    warnings = []
    
    # Common Terraform functions and their argument patterns
    terraform_functions = {
        'format': r'format\s*\(\s*"[^"]*"(?:\s*,\s*[^)]*)*\s*\)',
        'join': r'join\s*\(\s*"[^"]*"\s*,\s*\[[^\]]*\]\s*\)',
        'split': r'split\s*\(\s*"[^"]*"\s*,\s*[^)]*\s*\)',
        'replace': r'replace\s*\(\s*[^,]*\s*,\s*"[^"]*"\s*,\s*"[^"]*"\s*\)',
        'merge': r'merge\s*\(\s*[^)]*\s*\)',
        'concat': r'concat\s*\(\s*[^)]*\s*\)',
        'max': r'max\s*\(\s*[^)]*\s*\)',
        'min': r'min\s*\(\s*[^)]*\s*\)',
        'length': r'length\s*\(\s*[^)]*\s*\)',
        'keys': r'keys\s*\(\s*[^)]*\s*\)',
        'values': r'values\s*\(\s*[^)]*\s*\)',
        'coalesce': r'coalesce\s*\(\s*[^)]*\s*\)',
        'try': r'try\s*\(\s*[^)]*\s*\)',
        'can': r'can\s*\(\s*[^)]*\s*\)'
    }
    
    # Find all function calls in expression
    found_functions = []
    for func_name, pattern in terraform_functions.items():
        matches = re.findall(pattern, expression)
        if matches:
            found_functions.extend([func_name] * len(matches))
    
    # Check if declared functions match found functions
    declared_set = set(declared_functions)
    found_set = set(found_functions)
    
    missing_declarations = found_set - declared_set
    if missing_declarations:
        warnings.extend([f"Function {func} used but not declared in terraform_functions_used" for func in missing_declarations])
    
    extra_declarations = declared_set - found_set
    if extra_declarations:
        warnings.extend([f"Function {func} declared but not found in expression" for func in extra_declarations])
    
    return {
        'errors': errors,
        'warnings': warnings
    }

def validate_local_hcl_declaration(hcl_declaration: str) -> bool:
    """Validate HCL declaration syntax for local value"""
    try:
        # Check for proper local value declaration structure
        if not re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+$', hcl_declaration.strip()):
            return False
        
        # Check for proper assignment operator
        if '=' not in hcl_declaration:
            return False
        
        # Split on first equals sign
        parts = hcl_declaration.split('=', 1)
        if len(parts) != 2:
            return False
        
        name_part = parts[0].strip()
        value_part = parts[1].strip()
        
        # Validate name part
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name_part):
            return False
        
        # Value part should not be empty
        if not value_part:
            return False
        
        return True
    except Exception:
        return False

def attempt_local_value_fix(
    local_value: TerraformLocalValue, 
    errors: List[str]
) -> Optional[TerraformLocalValue]:
    """Attempt to fix common local value issues"""
    
    fixed_local = local_value.copy(deep=True)
    
    # Try to fix naming issues
    if any("doesn't follow conventions" in error for error in errors):
        fixed_name = fix_local_name(local_value.name)
        if fixed_name != local_value.name:
            fixed_local.name = fixed_name
            # Update HCL declaration with new name
            fixed_local.hcl_declaration = re.sub(
                r'^(\s*)[a-zA-Z_][a-zA-Z0-9_]*(\s*=)',
                rf'\1{fixed_name}\2',
                fixed_local.hcl_declaration
            )
    
    # Try to fix expression syntax
    if any("expression" in error.lower() for error in errors):
        fixed_expression = fix_terraform_expression(local_value.expression)
        if fixed_expression != local_value.expression:
            fixed_local.expression = fixed_expression
            # Update HCL declaration
            name_part, _ = fixed_local.hcl_declaration.split('=', 1)
            fixed_local.hcl_declaration = f"{name_part.strip()} = {fixed_expression}"
    
    # Validate the fixed local value
    validation_result = validate_terraform_local_value(fixed_local)
    if validation_result['valid']:
        return fixed_local
    
    return None

def fix_local_name(name: str) -> str:
    """Fix local value naming convention issues"""
    
    # Convert to lowercase
    fixed = name.lower()
    
    # Replace invalid characters with underscores
    fixed = re.sub(r'[^a-z0-9_]', '_', fixed)
    
    # Ensure it starts with a letter
    if fixed[0].isdigit():
        fixed = 'l_' + fixed
    
    # Remove multiple consecutive underscores
    fixed = re.sub(r'_+', '_', fixed)
    
    # Remove leading/trailing underscores
    fixed = fixed.strip('_')
    
    # Ensure it's not too long
    if len(fixed) > 64:
        fixed = fixed[:64].rstrip('_')
    
    return fixed

def fix_terraform_expression(expression: str) -> str:
    """Attempt to fix common Terraform expression issues"""
    
    fixed = expression
    
    # Fix common spacing issues around operators
    fixed = re.sub(r'\s*=\s*', ' = ', fixed)
    fixed = re.sub(r'\s*\?\s*', ' ? ', fixed)
    fixed = re.sub(r'\s*:\s*', ' : ', fixed)
    
    # Fix string interpolation spacing
    fixed = re.sub(r'\$\{\s*', '${', fixed)
    fixed = re.sub(r'\s*\}', '}', fixed)
    
    # Ensure proper comma spacing in function calls
    fixed = re.sub(r'\s*,\s*', ', ', fixed)
    
    return fixed

def generate_complete_locals_block(locals_list: List[TerraformLocalValue]) -> str:
    """Generate complete HCL locals block from local values"""
    
    if not locals_list:
        return ""
    
    # Sort locals by dependency order (locals with fewer dependencies first)
    sorted_locals = sorted(locals_list, key=lambda x: len(x.depends_on_locals))
    
    # Generate the complete locals block
    lines = ["locals {"]
    
    for local_val in sorted_locals:
        # Add description as comment if available
        if local_val.description:
            lines.append(f"  # {local_val.description}")
        
        # Add the local value declaration with proper indentation
        lines.append(f"  {local_val.hcl_declaration}")
        lines.append("")  # Empty line for readability
    
    # Remove the last empty line and close the block
    if lines[-1] == "":
        lines.pop()
    lines.append("}")
    
    return "\n".join(lines)

def create_local_value_handoff_recommendations(
    dependencies: List[DiscoveredLocalDependency],
    locals_list: List[TerraformLocalValue]
) -> List[LocalValueHandoffRecommendation]:
    """Create comprehensive handoff recommendations for local values"""
    
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
        if agent == 'variable_definition_agent':
            deliverables = ['Variable definitions for expression inputs', 'Variable validation rules']
        elif agent == 'resource_configuration_agent':
            deliverables = ['Resource configurations for attribute references', 'Resource dependency coordination']
        elif agent == 'data_source_agent':
            deliverables = ['Data source configurations for external references', 'Data source attribute mappings']
        else:
            deliverables = ['Required configurations']
        
        recommendation = LocalValueHandoffRecommendation(
            target_agent=agent,
            handoff_reason=f"Resolve {len(deps)} local value dependencies",
            handoff_priority=max_priority,
            context_payload={
                'dependencies': [dep.dict() for dep in deps],
                'total_count': len(deps),
                'priority_levels': [dep.priority_level for dep in deps],
                'affected_locals': [dep.source_local for dep in deps]
            },
            expected_deliverables=deliverables,
            local_value_context={
                'local_expressions': [local.expression for local in locals_list],
                'expression_types': [local.expression_type for local in locals_list],
                'dependency_requirements': [dep.requirement_details for dep in deps]
            },
            expression_requirements={
                'expression_fragments': [dep.expression_fragment for dep in deps],
                'suggested_replacements': [dep.suggested_replacement for dep in deps if dep.suggested_replacement]
            },
            should_wait_for_completion=has_blocking,
            can_continue_parallel=not has_blocking
        )
        
        recommendations.append(recommendation)
    
    return recommendations

def create_local_value_state_updates(
    locals_list: List[TerraformLocalValue],
    dependencies: List[DiscoveredLocalDependency],
    agent_workspace: Dict[str, Any],
    completion_status: str
) -> Dict[str, Any]:
    """Create comprehensive state updates for the swarm"""
    
    updates = {
        'terraform_locals': [local.dict() for local in locals_list],
        'pending_dependencies': {
            **agent_workspace.get('pending_dependencies', {}),
            'local_values_agent': [dep.dict() for dep in dependencies]
        },
        'agent_status_matrix': {
            **agent_workspace.get('agent_status_matrix', {}),
            'local_values_agent': completion_status
        },
        'planning_progress': {
            **agent_workspace.get('planning_progress', {}),
            'local_values_agent': 1.0 if completion_status == 'completed' else 0.6
        }
    }
    
    return updates

def create_local_value_workspace_updates(
    locals_list: List[TerraformLocalValue],
    dependencies: List[DiscoveredLocalDependency],
    metrics: LocalValueGenerationMetrics,
    completion_status: str
) -> Dict[str, Any]:
    """Create workspace updates for the local values agent"""
    
    return {
        'generated_locals': [local.dict() for local in locals_list],
        'pending_dependencies': [dep.dict() for dep in dependencies],
        'generation_metrics': metrics.dict(),
        'completion_status': completion_status,
        'completion_timestamp': datetime.now().isoformat(),
        'local_value_summary': {
            'total_locals': len(locals_list),
            'expression_types': list(set(local.expression_type for local in locals_list)),
            'complexity_levels': list(set(local.complexity_level for local in locals_list)),
            'dependencies_discovered': len(dependencies),
            'functions_used': list(set().union(*[local.terraform_functions_used for local in locals_list])),
            'complexity_score': metrics.complexity_distribution
        }
    }

def create_local_value_checkpoint_data(
    locals_list: List[TerraformLocalValue],
    dependencies: List[DiscoveredLocalDependency],
    completion_status: str
) -> Dict[str, Any]:
    """Create checkpoint data for recovery"""
    
    return {
        'stage': 'planning',
        'agent': 'local_values_agent',
        'checkpoint_type': 'local_values_generation_complete',
        'locals_generated': len(locals_list),
        'dependencies_discovered': len(dependencies),
        'completion_status': completion_status,
        'timestamp': datetime.now().isoformat(),
        'local_names': [local.name for local in locals_list],
        'dependency_types': [dep.dependency_type for dep in dependencies],
        'expression_types': [local.expression_type for local in locals_list],
        'complexity_levels': [local.complexity_level for local in locals_list]
    }

def create_local_values_error_response(
    error: Exception, 
    agent_workspace: Dict[str, Any], 
    start_time: datetime
) -> TerraformLocalValueGenerationResponse:
    """Create error response when tool execution fails"""
    
    generation_duration = (datetime.now() - start_time).total_seconds()
    
    return TerraformLocalValueGenerationResponse(
        generated_locals=[],
        discovered_dependencies=[],
        handoff_recommendations=[],
        completion_status='error',
        next_recommended_action='escalate_to_human_or_retry',
        generation_metadata=LocalValueGenerationMetrics(
            total_locals_generated=0,
            generation_duration_seconds=generation_duration,
            dependencies_discovered=0,
            handoffs_required=0,
            validation_errors=[f"Tool execution failed: {str(error)}"]
        ),
        complete_locals_block="",
        state_updates={
            'agent_status_matrix': {
                **agent_workspace.get('agent_status_matrix', {}),
                'local_values_agent': 'error'
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
            'agent': 'local_values_agent',
            'checkpoint_type': 'error',
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
    )