
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
from .output_generator_prompts import OUTPUT_DEFINITION_AGENT_USER_PROMPT_TEMPLATE_REFINED, OUTPUT_DEFINITION_AGENT_SYSTEM_PROMPT
from ..tf_content_compressor import TerraformDataCompressor
# Create agent logger for output generator
output_generator_logger = AgentLogger("OUTPUT_GENERATOR")

class OutputValueType(str, Enum):
    """Types of output values supported - extensible for dynamic discovery"""
    STRING = "string"
    NUMBER = "number"
    BOOL = "bool"
    LIST = "list"
    MAP = "map"
    OBJECT = "object"
    TUPLE = "tuple"
    SET = "set"
    COMPLEX_STRUCTURE = "complex_structure"
    
    # Dynamic type support
    @classmethod
    def from_string(cls, output_type: str) -> 'OutputValueType':
        """Create OutputValueType from string, supporting dynamic types"""
        try:
            return cls(output_type)
        except ValueError:
            return cls._create_dynamic_type(output_type)
    
    @classmethod
    def _create_dynamic_type(cls, output_type: str) -> 'OutputValueType':
        """Create dynamic output type for types not in enum"""
        return output_type  # Return as string for dynamic types

class OutputComplexity(str, Enum):
    """Complexity levels for output definitions"""
    SIMPLE = "simple"           # Basic resource attributes
    MODERATE = "moderate"       # Simple expressions and computations
    COMPLEX = "complex"         # Multiple resources, conditional logic
    ADVANCED = "advanced"       # Complex expressions, nested structures

class OutputSensitivity(str, Enum):
    """Sensitivity levels for outputs"""
    PUBLIC = "public"           # Non-sensitive data safe for sharing
    INTERNAL = "internal"       # Internal configuration data
    CONFIDENTIAL = "confidential"   # Sensitive but not secret
    SECRET = "secret"           # Highly sensitive encrypted data

class OutputUsageContext(str, Enum):
    """Context for how outputs are intended to be used"""
    EXTERNAL_INTEGRATION = "external_integration"  # For external systems
    MODULE_COMPOSITION = "module_composition"      # For other Terraform modules
    AUTOMATION_SCRIPTS = "automation_scripts"     # For CI/CD and scripts
    DEBUGGING_INFO = "debugging_info"             # For troubleshooting
    CONFIGURATION_REFERENCE = "configuration_reference"  # For configuration

class GeneratorAgentName(str, Enum):
    RESOURCE_CONFIGURATION = "resource_configuration_agent"
    VARIABLE_DEFINITION = "variable_definition_agent"
    DATA_SOURCE = "data_source_agent"
    LOCAL_VALUES = "local_values_agent"
    OUTPUT_DEFINITION = "output_definition_agent"


class TerraformPrecondition(BaseModel):
    """Individual precondition for an output"""
    
    condition: str = Field(..., description="Terraform condition expression")
    error_message: str = Field(..., description="Error message if condition fails")
    description: str = Field(default="", description="Description of what this condition validates")
    
    @field_validator('condition')
    @classmethod
    def validate_condition_syntax(cls, v):
        if not v or not v.strip():
            raise ValueError('Precondition condition cannot be empty')
        return v.strip()

class TerraformOutputBlock(BaseModel):
    """Individual Terraform output block specification"""
    
    name: str = Field(..., description="Output name")
    value_expression: str = Field(..., description="Terraform expression for output value")
    description: str = Field(..., description="Output description")
    
    # Output characteristics
    output_type: Union[OutputValueType, str] = Field(..., description="Type of output value (from planner or dynamic discovery)")
    complexity_level: OutputComplexity = Field(..., description="Complexity assessment")
    sensitivity_level: OutputSensitivity = Field(..., description="Sensitivity classification")
    usage_context: OutputUsageContext = Field(..., description="Intended usage context")
    
    # Output configuration
    sensitive: bool = Field(default=False, description="Mark as sensitive in CLI output")
    ephemeral: bool = Field(default=False, description="Prevent storing in state (Terraform 1.10+)")
    
    # Dependencies and validation
    depends_on: List[str] = Field(default_factory=list, description="Explicit dependencies")
    preconditions: List[TerraformPrecondition] = Field(default_factory=list, description="Preconditions to validate")
    
    # Source mapping
    source_resources: List[str] = Field(default_factory=list, description="Resources this output references")
    source_data_sources: List[str] = Field(default_factory=list, description="Data sources this output references")
    source_locals: List[str] = Field(default_factory=list, description="Local values this output references")
    source_variables: List[str] = Field(default_factory=list, description="Variables this output references")
    
    # Usage metadata
    example_usage: str = Field(default="", description="Example of how to use this output")
    related_outputs: List[str] = Field(default_factory=list, description="Related outputs that work with this one")
    
    # Generated HCL
    hcl_block: str = Field(..., description="Complete HCL output block")
    
    # Organizational metadata
    category: str = Field(default="general", description="Output category (connectivity, security, identity, etc.)")
    tags: List[str] = Field(default_factory=list, description="Organizational tags")
    
    @field_validator('name')
    @classmethod
    def validate_output_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Output name must be alphanumeric with underscores/hyphens only')
        if v[0].isdigit():
            raise ValueError('Output name cannot start with a number')
        if len(v) > 64:
            raise ValueError('Output name cannot exceed 64 characters')
        return v

class DiscoveredOutputDependency(BaseModel):
    """Dependency discovered that requires handoff to another agent"""
    
    dependency_id: str = Field(..., description="Unique dependency identifier")
    dependency_type: str = Field(..., description="Type of dependency discovered")
    target_agent: GeneratorAgentName = Field(..., description="Agent that should handle this dependency")
    
    # Context for handoff
    source_output: str = Field(..., description="Output that triggered this dependency")
    requirement_details: Dict[str, Any] = Field(..., description="Specific details about what's needed")
    priority_level: int = Field(default=3, ge=1, le=5, description="Priority level (1=low, 5=critical)")
    
    # Handoff payload
    handoff_context: Dict[str, Any] = Field(..., description="Context to pass to target agent")
    expected_response: Dict[str, str] = Field(..., description="Expected response format from target agent")
    
    # Output context
    output_requirements: Dict[str, Any] = Field(..., description="Specific output requirements")
    validation_needs: List[str] = Field(default_factory=list, description="Validation requirements")
    
    # Timing and blocking
    is_blocking: bool = Field(default=True, description="Whether output generation should wait")
    timeout_minutes: int = Field(default=30, description="Maximum time to wait for dependency resolution")

class OutputGenerationMetrics(BaseModel):
    """Metrics and performance data for output generation"""
    
    total_outputs_generated: int = Field(..., description="Number of outputs successfully generated")
    generation_duration_seconds: float = Field(..., description="Time taken for generation")
    dependencies_discovered: int = Field(..., description="Total dependencies found")
    handoffs_required: int = Field(..., description="Number of handoffs needed")
    
    # Output type breakdown
    output_type_counts: Dict[Union[OutputValueType, str], int] = Field(default_factory=dict)
    complexity_distribution: Dict[OutputComplexity, int] = Field(default_factory=dict)
    sensitivity_distribution: Dict[OutputSensitivity, int] = Field(default_factory=dict)
    usage_distribution: Dict[OutputUsageContext, int] = Field(default_factory=dict)
    
    # Validation statistics
    total_preconditions: int = Field(default=0, description="Total preconditions across all outputs")
    sensitive_outputs: int = Field(default=0, description="Outputs marked as sensitive")
    ephemeral_outputs: int = Field(default=0, description="Outputs marked as ephemeral")
    outputs_with_dependencies: int = Field(default=0, description="Outputs with explicit dependencies")
    
    # Category statistics
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Source statistics
    resource_references: int = Field(default=0, description="Total resource references across outputs")
    data_source_references: int = Field(default=0, description="Total data source references")
    local_references: int = Field(default=0, description="Total local value references")
    variable_references: int = Field(default=0, description="Total variable references")
    
    # Error tracking
    validation_errors: List[str] = Field(default_factory=list)
    warning_messages: List[str] = Field(default_factory=list)
    optimization_recommendations: List[str] = Field(default_factory=list)

class OutputHandoffRecommendation(BaseModel):
    """Recommendation for agent handoff with output context"""
    
    target_agent: GeneratorAgentName = Field(..., description="Recommended target agent")
    handoff_reason: str = Field(..., description="Reason for handoff")
    handoff_priority: int = Field(default=3, ge=1, le=5)
    
    # Handoff data
    context_payload: Dict[str, Any] = Field(..., description="Data to pass to target agent")
    expected_deliverables: List[str] = Field(..., description="What the target agent should produce")
    
    # Output specific context
    output_context: Dict[str, Any] = Field(..., description="Specific output context for handoff")
    validation_requirements: Dict[str, Any] = Field(..., description="Validation requirements for handoff")
    
    # Coordination
    should_wait_for_completion: bool = Field(default=True)
    can_continue_parallel: bool = Field(default=False)

class TerraformOutputGenerationResponse(BaseModel):
    """Complete response from generate_terraform_outputs tool"""
    
    # Generation results
    generated_outputs: List[TerraformOutputBlock] = Field(..., description="Successfully generated output blocks")
    complete_outputs_file: str = Field(..., description="Complete outputs.tf file content")
    discovered_dependencies: List[DiscoveredOutputDependency] = Field(default_factory=list, description="Dependencies requiring handoffs")
    
    # Agent coordination
    handoff_recommendations: List[OutputHandoffRecommendation] = Field(default_factory=list, description="Recommended handoffs to other agents")
    completion_status: str = Field(..., description="Current completion status")
    next_recommended_action: str = Field(..., description="Recommended next action")
    
    # Generation metadata
    generation_metadata: OutputGenerationMetrics = Field(..., description="Generation performance metrics")
    generation_timestamp: Optional[datetime] = Field(default=None, description="Timestamp when outputs were generated")
    
    # # Complete outputs file
    # complete_outputs_file: str = Field(..., description="Complete outputs.tf file content")
    
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


@tool("generate_terraform_outputs")
def generate_terraform_outputs(
    state: Annotated[Any, InjectedState] = None,
) -> Dict[str, Any]:
    """
    Generate Terraform output values from infrastructure and requirements.

    Args:
        state: GeneratorSwarmState containing all the data (execution_plan_data, agent_workspaces, planning_context)
               
    This tool analyzes generated infrastructure, designs appropriate outputs,
    identifies dependencies, and provides handoff recommendations to other agents.
    Supports both planner specifications and dynamic agent communication.
    
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
                and model_extra['metadata'].get('handoff_destination') == 'output_definition_agent'):
                state_updates = model_extra['metadata']['state_updates']
                tool_message_analysis["agent_status_matrix"] = state_updates.get('agent_status_matrix')
                tool_message_analysis["pending_dependencies"] = state_updates.get('pending_dependencies')
                tool_message_analysis["dependency_graph"] = state_updates.get('dependency_graph')
                tool_message_analysis["agent_workspaces"] = state_updates.get('agent_workspaces')
                tool_message_analysis["handoff_queue"] = state_updates.get('handoff_queue')
    
    output_generator_logger.log_structured(
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
            output_generator_logger.log_structured(
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
            agent_workspaces = tool_message_analysis.get("agent_workspaces", {}).get("output_definition_agent", {})
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

        generation_context = planning_context 
        
        output_generator_logger.log_structured(
            level="INFO",
            message="Starting Terraform output generation",
            extra={
                "output_requirements_count": len(planning_output_definitions),
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

        output_generator_logger.log_structured(
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
        
        formatted_user_prompt = OUTPUT_DEFINITION_AGENT_USER_PROMPT_TEMPLATE_REFINED.format(
            service_name=exec_plan.get('service_name', 'unknown'),
            module_name=exec_plan.get('module_name', 'unknown'),
            target_environment=exec_plan.get('target_environment', 'development'),
            generation_id=agent_workspace.get('generation_id', str(uuid.uuid4())),
            output_specifications=escape_json_for_template(compressed_data['output_definitions']),
            planning_resources=escape_json_for_template(compressed_data['resource_specifications']),
            planning_variables=escape_json_for_template(compressed_data['variable_definitions']),
            planning_data_sources=escape_json_for_template(compressed_data['data_sources']),
            planning_local_values=escape_json_for_template(compressed_data['local_values']),
            # planning_terraform_files=escape_json_for_template(compressed_data['terraform_files']),
            current_stage=planning_context.get('current_stage', 'generation'),
            active_agent=agent_workspace.get('active_agent', 'output_definition_agent'),
            workspace_generated_outputs=escape_json_for_template(compressed_data['workspace_generated_outputs']),
            workspace_generated_variables=escape_json_for_template(compressed_data['workspace_generated_variables']),
            workspace_generated_data_sources=escape_json_for_template(compressed_data['workspace_generated_data_sources']),
            workspace_generated_local_values=escape_json_for_template(compressed_data['workspace_generated_local_values']),
            workspace_generated_resources=escape_json_for_template(compressed_data['workspace_generated_resources']),
            handoff_context=escape_json_for_template(json.dumps(agent_workspace.get('handoff_context', {}), indent=2))
        )
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=TerraformOutputGenerationResponse)
        
        # Build complete prompt, escaping curly braces in system prompt
        escaped_system_prompt = OUTPUT_DEFINITION_AGENT_SYSTEM_PROMPT.replace('{', '{{').replace('}', '}}')
        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system_prompt),
            ("user", formatted_user_prompt),
            ("user", "Please respond with valid JSON matching the TerraformOutputGenerationResponse schema:\n{format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        try:
            config_instance = Config()
            llm_config = config_instance.get_llm_config()
            
            output_generator_logger.log_structured(
                level="DEBUG",
                message="Initializing LLM for output generation",
                extra={
                    "llm_provider": llm_config.get('provider'),
                    "llm_model": llm_config.get('model'),
                    "llm_temperature": llm_config.get('temperature'),
                    "llm_max_tokens": llm_config.get('max_tokens')
                }
            )
            
            # Use higher model for output generation
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

        except Exception as e:
            output_generator_logger.log_structured(
                level="ERROR",
                message="Failed to initialize LLM for output generation",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
        
        chain = prompt | model_higher | parser
        
        output_generator_logger.log_structured(
            level="DEBUG",
            message="Executing LLM chain for output generation",
            extra={
                "prompt_length": len(formatted_user_prompt),
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        
        # Execute the chain
        llm_response = chain.invoke({})
        
        output_generator_logger.log_structured(
            level="DEBUG",
            message="LLM response received, starting post-processing",
            extra={
                "generated_outputs_count": len(llm_response.generated_outputs),
                "discovered_dependencies_count": len(llm_response.discovered_dependencies),
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        
        # Post-process and enhance response
        enhanced_response = post_process_output_response(
            llm_response, 
            agent_workspace, 
            generation_context, 
            start_time
        )
        
        # Update agent workspace with specific fields like variable generator        
        output_generator_logger.log_structured(
            level="INFO",
            message="Terraform output generation completed successfully",
            extra={
                "final_outputs_count": len(enhanced_response.generated_outputs),
                "final_dependencies_count": len(enhanced_response.discovered_dependencies),
                "generation_duration_seconds": enhanced_response.generation_metadata.generation_duration_seconds,
                "completion_status": enhanced_response.completion_status,
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        update_agent_workspace(
            "output_definition_agent", {
                "complete_outputs_file": enhanced_response.complete_outputs_file,
                "handoff_recommendations": enhanced_response.handoff_recommendations,
                **enhanced_response.workspace_updates  # Include all workspace_updates
            }
        )
        resolved_dependencies = previous_state.get("pending_dependencies", {}).get("output_definition_agent", [])
        # Get current resolved dependencies and append new ones
        current_resolved_deps = get_current_state().get("resolved_dependencies", {})
        updated_resolved_deps = {
            **current_resolved_deps,
            "output_definition_agent": [
                *current_resolved_deps.get("output_definition_agent", []),
                *resolved_dependencies
            ]
        }
        update_current_state({
            "resolved_dependencies": updated_resolved_deps
        })
        # Return only state_updates as JSON for LangGraph state management
        return enhanced_response.state_updates
        
    except Exception as e:
        output_generator_logger.log_structured(
            level="ERROR",
            message="Terraform output generation failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "generation_id": agent_workspace.get('generation_id', 'unknown'),
                "current_stage": planning_context.get('current_stage', 'unknown')
            }
        )
        return create_output_error_response(e, agent_workspace, datetime.now())


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

def extract_generated_resources(agent_workspace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract generated resources from the agent workspace."""
    return agent_workspace.get('terraform_resources', [])

def extract_generated_data_sources(agent_workspace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract generated data sources from the agent workspace."""
    return agent_workspace.get('terraform_data_sources', [])

def extract_generated_variables(agent_workspace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract generated variables from the agent workspace."""
    return agent_workspace.get('terraform_variables', [])

def extract_generated_locals(agent_workspace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract generated local values from the agent workspace."""
    return agent_workspace.get('terraform_locals', [])

def extract_specific_requirements(context: Dict[str, Any]) -> str:
    """Extract specific requirements from context"""
    requirements = []
    
    exec_plan = context.get('execution_plan', {})
    
    if 'output_requirements' in exec_plan:
        requirements.append(f"Output Requirements: {exec_plan['output_requirements']}")
    
    if 'exposure_requirements' in exec_plan:
        requirements.append(f"Exposure Requirements: {exec_plan['exposure_requirements']}")
    
    if 'integration_requirements' in exec_plan:
        requirements.append(f"Integration Requirements: {exec_plan['integration_requirements']}")
    
    if 'automation_requirements' in exec_plan:
        requirements.append(f"Automation Requirements: {exec_plan['automation_requirements']}")
    
    if 'security_requirements' in exec_plan:
        requirements.append(f"Security Requirements: {exec_plan['security_requirements']}")
    
    return '\n'.join(requirements) if requirements else "No specific requirements specified"

def post_process_output_response(
    llm_response: TerraformOutputGenerationResponse,
    state: Dict[str, Any], 
    context: Dict[str, Any],
    start_time: datetime
) -> TerraformOutputGenerationResponse:
    """Post-process LLM response with comprehensive validation and enhancements"""
    
    # Calculate actual generation duration
    generation_duration = (datetime.now() - start_time).total_seconds()
    llm_response.generation_metadata.generation_duration_seconds = generation_duration
    
    # Fix generation timestamp if it's empty or invalid
    if not llm_response.generation_timestamp or llm_response.generation_timestamp == "":
        llm_response.generation_timestamp = datetime.now()
    
    # Validate generated outputs with flexible validation
    validated_outputs = []
    validation_errors = []
    validation_warnings = []
    
    for output in llm_response.generated_outputs:
        validation_result = validate_terraform_output(output, state)
        
        # Always include outputs unless they have critical errors
        if validation_result['valid']:
            validated_outputs.append(output)
            # Add warnings to recoverable warnings
            if validation_result.get('warnings'):
                validation_warnings.extend(validation_result['warnings'])
        else:
            # Only filter out outputs with critical errors
            validation_errors.extend(validation_result['errors'])
            # Attempt to fix critical issues
            fixed_output = attempt_output_fix(output, validation_result['errors'], state)
            if fixed_output:
                validated_outputs.append(fixed_output)
                llm_response.recoverable_warnings.append(
                    f"Fixed critical validation issues for output '{output.name}'"
                )
            else:
                # If we can't fix critical issues, still include the output but log the error
                validated_outputs.append(output)
                llm_response.recoverable_warnings.append(
                    f"Output '{output.name}' has critical issues but included anyway"
                )
    
    # Update response with validated outputs
    llm_response.generated_outputs = validated_outputs
    llm_response.generation_metadata.validation_errors.extend(validation_errors)
    
    # Add validation warnings to recoverable warnings
    if validation_warnings:
        llm_response.recoverable_warnings.extend(validation_warnings)
    
    # Generate complete outputs file
    llm_response.complete_outputs_file = generate_complete_outputs_file(validated_outputs)
    
    # Update metrics with actual counts
    update_output_generation_metrics(llm_response.generation_metadata, validated_outputs)
    
    # Use original discovered dependencies without enhancement
    # The LLM already provides the necessary context for each dependency
    
    # Create comprehensive handoff recommendations
    llm_response.handoff_recommendations = create_output_handoff_recommendations(
        llm_response.discovered_dependencies,
        validated_outputs
    )
    
    # Add comprehensive state updates
    llm_response.state_updates = create_output_state_updates(
        validated_outputs,
        llm_response.discovered_dependencies,
        state,
        llm_response.completion_status
    )
    
    # Add workspace updates
    llm_response.workspace_updates = create_output_workspace_updates(
        validated_outputs,
        llm_response.discovered_dependencies,
        llm_response.generation_metadata,
        llm_response.completion_status
    )
    
    # Add checkpoint data
    llm_response.checkpoint_data = create_output_checkpoint_data(
        validated_outputs,
        llm_response.discovered_dependencies,
        llm_response.completion_status
    )
    
    return llm_response

def generate_complete_outputs_file(outputs: List[TerraformOutputBlock]) -> str:
    """Generate complete outputs.tf file content from output blocks"""
    if not outputs:
        return ""
    
    # Generate file header
    file_content = "# Outputs\n"
    file_content += "# This file contains all output values for the infrastructure\n\n"
    
    # Group outputs by category for better organization
    outputs_by_category = {}
    for output in outputs:
        category = output.category or "general"
        if category not in outputs_by_category:
            outputs_by_category[category] = []
        outputs_by_category[category].append(output)
    
    # Generate content grouped by category
    for category, category_outputs in outputs_by_category.items():
        file_content += f"# {category.replace('_', ' ').title()} Outputs\n"
        for output in category_outputs:
            file_content += f"{output.hcl_block}\n"
        file_content += "\n"
    
    return file_content.strip()

def validate_terraform_output(
    output: TerraformOutputBlock, 
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate individual Terraform output"""
    errors = []
    warnings = []
    
    # Validate output name
    if not validate_output_name_convention(output.name):
        errors.append(f"Output name '{output.name}' doesn't follow Terraform conventions")
    
    # Validate value expression
    expression_validation = validate_output_expression(output.value_expression, state)
    if not expression_validation['valid']:
        errors.extend(expression_validation['errors'])
    if expression_validation['warnings']:
        warnings.extend(expression_validation['warnings'])
    
    # Validate preconditions
    for precondition in output.preconditions:
        precondition_validation = validate_output_precondition(precondition, output)
        if not precondition_validation['valid']:
            errors.extend(precondition_validation['errors'])
    
    # Validate HCL block
    if not validate_output_hcl_block(output.hcl_block):
        errors.append(f"Invalid HCL block for output '{output.name}'")
    
    # Security validation
    security_validation = validate_output_security(output)
    if security_validation['warnings']:
        warnings.extend(security_validation['warnings'])
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def validate_output_name_convention(name: str) -> bool:
    """Validate Terraform output naming convention"""
    # Check if name follows snake_case convention
    if not re.match(r'^[a-z][a-z0-9_]*$', name):
        return False
    
    # Check if name is not too long
    if len(name) > 64:
        return False
    
    # Check for reserved words
    reserved_words = ['output', 'var', 'local', 'data', 'resource', 'module', 'provider']
    if name in reserved_words:
        return False
    
    return True

def validate_output_expression(
    expression: str, 
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate output value expression"""
    errors = []
    warnings = []
    
    if not expression or not expression.strip():
        errors.append("Output value expression cannot be empty")
        return {'valid': False, 'errors': errors, 'warnings': warnings}
    
    # Check for basic syntax issues
    if expression.count('(') != expression.count(')'):
        errors.append("Unbalanced parentheses in output expression")
    
    if expression.count('[') != expression.count(']'):
        errors.append("Unbalanced brackets in output expression")
    
    if expression.count('{') != expression.count('}'):
        errors.append("Unbalanced braces in output expression")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def extract_references_from_expression(expression: str) -> Dict[str, List[str]]:
    """Extract resource, data, local, and variable references from expression"""
    references = {
        'resources': [],
        'data_sources': [],
        'locals': [],
        'variables': []
    }
    
    # Find resource references (resource_type.name or aws_instance.web)
    resource_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*'
    resource_matches = re.findall(resource_pattern, expression)
    
    for match in resource_matches:
        if match.startswith('data.'):
            references['data_sources'].append(match)
        elif match.startswith('local.'):
            references['locals'].append(match)
        elif match.startswith('var.'):
            references['variables'].append(match)
        else:
            references['resources'].append(match)
    
    return references

def validate_expression_references(
    references: Dict[str, List[str]], 
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate that references exist in the current state"""
    errors = []
    warnings = []
    
    # Get generated infrastructure from state
    generated_resources = state.get('terraform_resources', [])
    generated_data_sources = state.get('terraform_data_sources', [])
    generated_variables = state.get('terraform_variables', [])
    generated_locals = state.get('terraform_locals', [])
    
    # Create lookup sets for validation
    resource_names = {f"{res.get('resource_type', '')}.{res.get('resource_name', '')}" 
                     for res in generated_resources if res.get('resource_type') and res.get('resource_name')}
    
    data_source_names = {f"data.{ds.get('data_source_type', '')}.{ds.get('data_source_name', '')}" 
                        for ds in generated_data_sources if ds.get('data_source_type') and ds.get('data_source_name')}
    
    variable_names = {f"var.{var.get('name', '')}" 
                     for var in generated_variables if var.get('name')}
    
    local_names = {f"local.{local.get('name', '')}" 
                  for local in generated_locals if local.get('name')}
    
    # Validate resource references
    for ref in references['resources']:
        if ref not in resource_names:
            errors.append(f"Referenced resource '{ref}' not found in generated resources")
    
    # Validate data source references
    for ref in references['data_sources']:
        if ref not in data_source_names:
            errors.append(f"Referenced data source '{ref}' not found in generated data sources")
    
    # Validate variable references
    for ref in references['variables']:
        if ref not in variable_names:
            warnings.append(f"Referenced variable '{ref}' not found in generated variables")
    
    # Validate local references
    for ref in references['locals']:
        if ref not in local_names:
            warnings.append(f"Referenced local value '{ref}' not found in generated locals")
    
    return {
        'errors': errors,
        'warnings': warnings
    }

def validate_output_precondition(
    precondition: TerraformPrecondition, 
    output: TerraformOutputBlock
) -> Dict[str, Any]:
    """Validate individual output precondition"""
    errors = []
    
    # Check if condition is not empty
    if not precondition.condition.strip():
        errors.append("Precondition condition cannot be empty")
    
    # Check for common validation patterns
    validation_functions = ['length(', 'can(', 'contains(', 'regex(', 'startswith(', 'endswith(']
    has_validation_function = any(func in precondition.condition for func in validation_functions)
    has_comparison = any(op in precondition.condition for op in ['==', '!=', '>', '<', '>=', '<='])
    
    if not has_validation_function and not has_comparison:
        errors.append("Precondition should use validation functions or comparison operators")
    
    # Check error message quality
    if len(precondition.error_message) < 10:
        errors.append("Precondition error message should be descriptive (at least 10 characters)")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_output_hcl_block(hcl_block: str) -> bool:
    """Validate HCL block syntax for output"""
    try:
        # Check for proper output block structure
        if not re.match(r'output\s+"[^"]+"\s*{', hcl_block):
            return False
        
        # Check for balanced braces
        if hcl_block.count('{') != hcl_block.count('}'):
            return False
        
        # Check for required components
        required_components = ['value', 'description']
        for component in required_components:
            if component not in hcl_block:
                return False
        
        return True
    except Exception:
        return False

def validate_output_security(output: TerraformOutputBlock) -> Dict[str, Any]:
    """Validate output security considerations"""
    warnings = []
    
    # Check for sensitive data patterns in name or description
    sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential', 'private']
    
    name_lower = output.name.lower()
    desc_lower = output.description.lower()
    expr_lower = output.value_expression.lower()
    
    for pattern in sensitive_patterns:
        if pattern in name_lower or pattern in desc_lower or pattern in expr_lower:
            if not output.sensitive:
                warnings.append(f"Output '{output.name}' appears to contain sensitive data but is not marked as sensitive")
    
    # Check for overly broad information exposure
    if output.usage_context == OutputUsageContext.EXTERNAL_INTEGRATION and output.sensitivity_level == OutputSensitivity.INTERNAL:
        warnings.append(f"Output '{output.name}' is marked for external integration but contains internal data")
    
    return {
        'warnings': warnings
    }

def validate_output_dependencies(
    output: TerraformOutputBlock, 
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate output dependencies and explicit depends_on"""
    warnings = []
    
    # Check if explicit dependencies are necessary
    if output.depends_on:
        for dep in output.depends_on:
            if dep in output.value_expression:
                warnings.append(f"Explicit dependency '{dep}' may be unnecessary due to implicit reference in value expression")
    
    return {
        'warnings': warnings
    }

def attempt_output_fix(
    output: TerraformOutputBlock, 
    errors: List[str],
    state: Dict[str, Any]
) -> Optional[TerraformOutputBlock]:
    """Attempt to fix common output issues"""
    
    fixed_output = output.copy(deep=True)
    
    # Try to fix naming issues
    if any("doesn't follow" in error for error in errors):
        fixed_name = fix_output_name(output.name)
        if fixed_name != output.name:
            fixed_output.name = fixed_name
            # Update HCL block with new name
            fixed_output.hcl_block = re.sub(
                r'output\s+"[^"]+"\s*{',
                f'output "{fixed_name}" {{',
                fixed_output.hcl_block
            )
    
    # Try to fix empty expression
    if any("cannot be empty" in error for error in errors):
        if not fixed_output.value_expression.strip():
            # Try to generate a basic expression based on sources
            if fixed_output.source_resources:
                resource_ref = fixed_output.source_resources[0]
                fixed_output.value_expression = f"{resource_ref}.id"
                # Update HCL block
                fixed_output.hcl_block = re.sub(
                    r'value\s*=\s*[^\n]*',
                    f'value = {fixed_output.value_expression}',
                    fixed_output.hcl_block
                )
    
    # Validate the fixed output
    validation_result = validate_terraform_output(fixed_output, state)
    if validation_result['valid']:
        return fixed_output
    
    return None

def fix_output_name(name: str) -> str:
    """Fix output naming convention issues"""
    
    # Convert to lowercase
    fixed = name.lower()
    
    # Replace invalid characters with underscores
    fixed = re.sub(r'[^a-z0-9_]', '_', fixed)
    
    # Ensure it starts with a letter
    if fixed[0].isdigit():
        fixed = 'o_' + fixed
    
    # Remove multiple consecutive underscores
    fixed = re.sub(r'_+', '_', fixed)
    
    # Remove leading/trailing underscores
    fixed = fixed.strip('_')
    
    # Ensure it's not too long
    if len(fixed) > 64:
        fixed = fixed[:64].rstrip('_')
    
    return fixed

def generate_complete_outputs_file(outputs: List[TerraformOutputBlock]) -> str:
    """Generate complete outputs.tf file from output blocks"""
    
    if not outputs:
        return ""
    
    # Sort outputs by category and then by name
    sorted_outputs = sorted(outputs, key=lambda x: (x.category, x.name))
    
    # Group by category
    categories = {}
    for output in sorted_outputs:
        if output.category not in categories:
            categories[output.category] = []
        categories[output.category].append(output)
    
    # Generate the complete file
    lines = []
    lines.append("# Terraform Outputs")
    lines.append(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    for category, outputs_in_category in categories.items():
        if category != "general":
            lines.append(f"# {category.title()} Outputs")
            lines.append("")
        
        for output in outputs_in_category:
            lines.append(output.hcl_block)
            lines.append("")
    
    return "\n".join(lines)

def update_output_generation_metrics(
    metrics: OutputGenerationMetrics,
    outputs: List[TerraformOutputBlock]
) -> None:
    """Update metrics with actual generated output data"""
    
    # Update type counts
    for output in outputs:
        metrics.output_type_counts[output.output_type] = (
            metrics.output_type_counts.get(output.output_type, 0) + 1
        )
    
    # Update complexity distribution
    for output in outputs:
        metrics.complexity_distribution[output.complexity_level] = (
            metrics.complexity_distribution.get(output.complexity_level, 0) + 1
        )
    
    # Update sensitivity distribution
    for output in outputs:
        metrics.sensitivity_distribution[output.sensitivity_level] = (
            metrics.sensitivity_distribution.get(output.sensitivity_level, 0) + 1
        )
    
    # Update usage distribution
    for output in outputs:
        metrics.usage_distribution[output.usage_context] = (
            metrics.usage_distribution.get(output.usage_context, 0) + 1
        )
    
    # Update category distribution
    for output in outputs:
        metrics.category_distribution[output.category] = (
            metrics.category_distribution.get(output.category, 0) + 1
        )
    
    # Update validation and security statistics
    metrics.total_preconditions = sum(len(output.preconditions) for output in outputs)
    metrics.sensitive_outputs = sum(1 for output in outputs if output.sensitive)
    metrics.ephemeral_outputs = sum(1 for output in outputs if output.ephemeral)
    metrics.outputs_with_dependencies = sum(1 for output in outputs if output.depends_on)
    
    # Update source statistics
    metrics.resource_references = sum(len(output.source_resources) for output in outputs)
    metrics.data_source_references = sum(len(output.source_data_sources) for output in outputs)
    metrics.local_references = sum(len(output.source_locals) for output in outputs)
    metrics.variable_references = sum(len(output.source_variables) for output in outputs)

def create_output_handoff_recommendations(
    dependencies: List[DiscoveredOutputDependency],
    outputs: List[TerraformOutputBlock]
) -> List[OutputHandoffRecommendation]:
    """Create comprehensive handoff recommendations for outputs"""
    
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
            deliverables = ['Resource configurations for output value extraction', 'Resource attribute specifications']
        elif agent == 'data_source_agent':
            deliverables = ['Data source configurations for external data', 'Data source attribute mappings']
        elif agent == 'variable_definition_agent':
            deliverables = ['Variable definitions for output context', 'Variable validation specifications']
        elif agent == 'local_values_agent':
            deliverables = ['Local value definitions for complex expressions', 'Computed value specifications']
        else:
            deliverables = ['Required configurations']
        
        recommendation = OutputHandoffRecommendation(
            target_agent=agent,
            handoff_reason=f"Resolve {len(deps)} output dependencies",
            handoff_priority=max_priority,
            context_payload={
                'dependencies': [dep.dict() for dep in deps],
                'total_count': len(deps),
                'priority_levels': [dep.priority_level for dep in deps],
                'affected_outputs': [dep.source_output for dep in deps]
            },
            expected_deliverables=deliverables,
            output_context={
                'output_types': [output.output_type for output in outputs],
                'complexity_levels': [output.complexity_level for output in outputs],
                'usage_contexts': [output.usage_context for output in outputs],
                'categories': list(set(output.category for output in outputs))
            },
            validation_requirements={
                'validation_needs': [dep.validation_needs for dep in deps],
                'output_requirements': [dep.output_requirements for dep in deps]
            },
            should_wait_for_completion=has_blocking,
            can_continue_parallel=not has_blocking
        )
        
        recommendations.append(recommendation)
    
    return recommendations

def create_output_state_updates(
    outputs: List[TerraformOutputBlock],
    dependencies: List[DiscoveredOutputDependency],
    current_state: Dict[str, Any],
    completion_status: str
) -> Dict[str, Any]:
    """Create comprehensive state updates for the swarm"""
    
    updates = {
        'terraform_outputs': [output.dict() for output in outputs],
        'pending_dependencies': {
            **current_state.get('pending_dependencies', {}),
            'output_definition_agent': [dep.dict() for dep in dependencies]
        },
        'agent_status_matrix': {
            **current_state.get('agent_status_matrix', {}),
            'output_definition_agent': completion_status
        },
        'finalization_progress': {
            **current_state.get('finalization_progress', {}),
            'output_definition_agent': 1.0 if completion_status == 'completed' else 0.6
        }
    }
    
    return updates

def create_output_workspace_updates(
    outputs: List[TerraformOutputBlock],
    dependencies: List[DiscoveredOutputDependency],
    metrics: OutputGenerationMetrics,
    completion_status: str
) -> Dict[str, Any]:
    """Create workspace updates for the output definition agent"""
    
    return {
        'generated_outputs': [output.dict() for output in outputs],
        'pending_dependencies': [dep.dict() for dep in dependencies],
        'generation_metrics': metrics.dict(),
        'completion_status': completion_status,
        'completion_timestamp': datetime.now().isoformat(),
        'output_summary': {
            'total_outputs': len(outputs),
            'output_types': list(set(output.output_type for output in outputs)),
            'complexity_levels': list(set(output.complexity_level for output in outputs)),
            'usage_contexts': list(set(output.usage_context for output in outputs)),
            'categories': list(set(output.category for output in outputs)),
            'dependencies_discovered': len(dependencies),
            'sensitive_outputs': sum(1 for output in outputs if output.sensitive),
            'ephemeral_outputs': sum(1 for output in outputs if output.ephemeral)
        }
    }

def create_output_checkpoint_data(
    outputs: List[TerraformOutputBlock],
    dependencies: List[DiscoveredOutputDependency],
    completion_status: str
) -> Dict[str, Any]:
    """Create checkpoint data for recovery"""
    
    return {
        'stage': 'finalization',
        'agent': 'output_definition_agent',
        'checkpoint_type': 'output_generation_complete',
        'outputs_generated': len(outputs),
        'dependencies_discovered': len(dependencies),
        'completion_status': completion_status,
        'timestamp': datetime.now().isoformat(),
        'output_names': [output.name for output in outputs],
        'dependency_types': [dep.dependency_type for dep in dependencies],
        'output_categories': list(set(output.category for output in outputs)),
        'complexity_levels': list(set(output.complexity_level for output in outputs))
    }

def create_output_error_response(
    error: Exception, 
    state: Dict[str, Any], 
    start_time: datetime
) -> TerraformOutputGenerationResponse:
    """Create error response when tool execution fails"""
    
    generation_duration = (datetime.now() - start_time).total_seconds()
    
    return TerraformOutputGenerationResponse(
        generated_outputs=[],
        discovered_dependencies=[],
        handoff_recommendations=[],
        completion_status='error',
        next_recommended_action='escalate_to_human_or_retry',
        generation_metadata=OutputGenerationMetrics(
            total_outputs_generated=0,
            generation_duration_seconds=generation_duration,
            dependencies_discovered=0,
            handoffs_required=0,
            validation_errors=[f"Tool execution failed: {str(error)}"]
        ),
        complete_outputs_file="",
        state_updates={
            'agent_status_matrix': {
                **state.get('agent_status_matrix', {}),
                'output_definition_agent': 'error'
            }
        },
        workspace_updates={
            'error': str(error),
            'completion_status': 'error',
            'error_timestamp': datetime.now().isoformat()
        },
        critical_errors=[f"Critical tool failure: {str(error)}"],
        checkpoint_data={
            'stage': 'finalization',
            'agent': 'output_definition_agent',
            'checkpoint_type': 'error',
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
    )


def _prepare_output_approval_context(output_req: Dict[str, Any], generation_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare approval context for output creation.
    
    Args:
        output_req: Output requirement specification
        generation_context: Generation context
        
    Returns:
        Approval context dictionary
    """
    return {
        "output_name": output_req.get("name"),
        "output_type": output_req.get("type"),
        "sensitivity_level": _assess_output_sensitivity(output_req),
        "complexity_level": _assess_output_complexity(output_req),
        "security_related": _check_security_related_output(output_req),
        "uses_experimental_functions": _check_experimental_functions(output_req),
        "output_config": output_req,
        "generation_context": generation_context
    }


def _check_output_approval(approval_context: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if output creation requires approval.
    
    Args:
        approval_context: Context for approval decision
        state: Current state
        
    Returns:
        Approval result dictionary
    """
    # Check for sensitive outputs
    if approval_context.get("sensitivity_level") in ["confidential", "secret"]:
        return {
            "status": "approved",  # Would be determined by human approval
            "reason": "Sensitive output approved",
            "trigger_type": "security_critical"
        }
    
    # Check for security-related outputs
    if approval_context.get("security_related", False):
        return {
            "status": "approved",  # Would be determined by human approval
            "reason": "Security-related output approved",
            "trigger_type": "security_critical"
        }
    
    # Check for complex outputs that might expose sensitive information
    if approval_context.get("complexity_level") in ["complex", "advanced"]:
        return {
            "status": "approved",  # Would be determined by human approval
            "reason": "Complex output approved",
            "trigger_type": "high_cost_resources"
        }
    
    # Check for experimental function usage
    if approval_context.get("uses_experimental_functions", False):
        return {
            "status": "approved",  # Would be determined by human approval
            "reason": "Experimental function usage approved",
            "trigger_type": "experimental"
        }
    
    return {"status": "approved", "reason": "No approval required"}


def _assess_output_sensitivity(output_req: Dict[str, Any]) -> str:
    """Assess sensitivity level of output."""
    name = output_req.get("name", "").lower()
    description = output_req.get("description", "").lower()
    
    # Check for sensitive patterns
    sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential', 'private', 'auth']
    
    for pattern in sensitive_patterns:
        if pattern in name or pattern in description:
            return "secret"
    
    # Check for internal patterns
    internal_patterns = ['internal', 'private', 'config', 'setting']
    for pattern in internal_patterns:
        if pattern in name or pattern in description:
            return "confidential"
    
    return "public"


def _assess_output_complexity(output_req: Dict[str, Any]) -> str:
    """Assess complexity level of output."""
    value_expression = output_req.get("value_expression", "")
    
    # Count complexity indicators
    complexity_indicators = 0
    
    # Check for function calls
    if '(' in value_expression and ')' in value_expression:
        complexity_indicators += 1
    
    # Check for multiple references
    if value_expression.count('.') > 2:
        complexity_indicators += 1
    
    # Check for conditional logic
    if any(op in value_expression for op in ['?', ':', 'if', 'else']):
        complexity_indicators += 2
    
    # Check for loops or iterations
    if any(func in value_expression for func in ['for', 'foreach', 'map', 'filter']):
        complexity_indicators += 2
    
    if complexity_indicators >= 3:
        return "advanced"
    elif complexity_indicators >= 2:
        return "complex"
    elif complexity_indicators >= 1:
        return "moderate"
    else:
        return "simple"


def _check_security_related_output(output_req: Dict[str, Any]) -> bool:
    """Check if output is security-related."""
    name = output_req.get("name", "").lower()
    description = output_req.get("description", "").lower()
    
    security_patterns = [
        'security', 'auth', 'permission', 'role', 'policy', 'access',
        'firewall', 'vpc', 'subnet', 'security_group', 'iam'
    ]
    
    return any(pattern in name or pattern in description for pattern in security_patterns)


def _check_experimental_functions(output_req: Dict[str, Any]) -> bool:
    """Check if output uses experimental functions."""
    value_expression = output_req.get("value_expression", "")
    
    experimental_functions = [
        'experimental_', 'beta_', 'alpha_', 'preview_'
    ]
    
    return any(func in value_expression for func in experimental_functions)