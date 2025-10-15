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
from aws_orchestrator_agent.core.agents.generator.generator_state import GeneratorSwarmState
from aws_orchestrator_agent.core.agents.types import SupervisorState
from langgraph.types import Command
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger
from .resource_generator_prompts import RESOURCE_CONFIGURATION_USER_PROMPT_TEMPLATE, RESOURCE_CONFIGURATION_SYSTEM_PROMPT, RESOURCE_CONFIGURATION_USER_PROMPT_TEMPLATE_REFINED
from ..global_state import get_current_state, update_agent_workspace, update_current_state
from ..tf_content_compressor import TerraformDataCompressor
# Create agent logger for resource generator
resource_generator_logger = AgentLogger("RESOURCE_GENERATOR")

class DependencyType(str, Enum):
    """Types of dependencies discovered during resource generation"""
    VARIABLE_REQUIRED = "variable_required"
    DATA_SOURCE_REQUIRED = "data_source_required"
    LOCAL_VALUE_REQUIRED = "local_value_required"
    RESOURCE_DEPENDENCY = "resource_dependency"

class GeneratorAgentName(str, Enum):
    RESOURCE_CONFIGURATION = "resource_configuration_agent"
    VARIABLE_DEFINITION = "variable_definition_agent"
    DATA_SOURCE = "data_source_agent"
    LOCAL_VALUES = "local_values_agent"
    OUTPUT_DEFINITION = "output_definition_agent"

class ResourceBlockType(str, Enum):
    """AWS resource types supported - extensible for dynamic discovery"""
    # Core AWS resource types
    EC2_INSTANCE = "aws_instance"
    VPC = "aws_vpc"
    SUBNET = "aws_subnet"
    SECURITY_GROUP = "aws_security_group"
    RDS_INSTANCE = "aws_db_instance"
    S3_BUCKET = "aws_s3_bucket"
    LOAD_BALANCER = "aws_lb"
    AUTO_SCALING_GROUP = "aws_autoscaling_group"
    LAMBDA_FUNCTION = "aws_lambda_function"
    API_GATEWAY = "aws_api_gateway_rest_api"
    
    # Additional common types
    INTERNET_GATEWAY = "aws_internet_gateway"
    NAT_GATEWAY = "aws_nat_gateway"
    ROUTE_TABLE = "aws_route_table"
    ROUTE = "aws_route"
    EIP = "aws_eip"
    VPC_ENDPOINT = "aws_vpc_endpoint"
    CLOUDWATCH_LOG_GROUP = "aws_cloudwatch_log_group"
    IAM_ROLE = "aws_iam_role"
    IAM_ROLE_POLICY = "aws_iam_role_policy"
    FLOW_LOG = "aws_flow_log"
    
    # Dynamic resource type support
    @classmethod
    def from_string(cls, resource_type: str) -> 'ResourceBlockType':
        """Create ResourceBlockType from string, supporting dynamic types"""
        try:
            return cls(resource_type)
        except ValueError:
            # For dynamic resource types not in enum, create a new instance
            return cls._create_dynamic_type(resource_type)
    
    @classmethod
    def _create_dynamic_type(cls, resource_type: str) -> 'ResourceBlockType':
        """Create a dynamic resource type for unknown AWS resources"""
        # This allows the system to handle any AWS resource type dynamically
        return resource_type  # Return as string for dynamic types

class TerraformResourceBlock(BaseModel):
    """Individual Terraform resource block specification"""
    
    resource_type: Union[ResourceBlockType, str] = Field(..., description="AWS resource type (from planner or dynamic discovery)")
    resource_name: str = Field(..., description="Terraform resource name (unique identifier)")
    logical_name: str = Field(..., description="Human-readable resource name")
    
    # Core configuration
    configuration_attributes: Dict[str, Any] = Field(..., description="Resource-specific configuration attributes")
    required_attributes: List[str] = Field(..., description="List of required attribute names")
    optional_attributes: List[str] = Field(..., description="List of optional attribute names")
    
    # Meta-arguments
    meta_arguments: Dict[str, Any] = Field(default_factory=dict, description="Terraform meta-arguments (count, for_each, depends_on)")
    lifecycle_configuration: Optional[Dict[str, Any]] = Field(None, description="Lifecycle block configuration")
    
    # Dependencies discovered
    implicit_dependencies: List[str] = Field(default_factory=list, description="Resources this depends on implicitly")
    explicit_dependencies: List[str] = Field(default_factory=list, description="Resources requiring explicit depends_on")
    
    # Generated HCL
    hcl_block: str = Field(..., description="Complete HCL resource block")
    
    # Validation and compliance
    validation_rules: List[str] = Field(default_factory=list, description="Applied validation rules")
    compliance_tags: Dict[str, str] = Field(default_factory=dict, description="Compliance and governance tags")
    
    @field_validator('resource_name')
    @classmethod
    def validate_resource_name(cls, v):
        """Ensure resource name follows Terraform naming conventions"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Resource name must be alphanumeric with underscores/hyphens only')
        if v.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
            raise ValueError('Resource name cannot start with a number')
        return v

class DiscoveredDependency(BaseModel):
    """Dependency discovered that requires handoff to another agent"""
    
    dependency_id: str = Field(..., description="Unique dependency identifier")
    dependency_type: DependencyType = Field(..., description="Type of dependency discovered")
    target_agent: GeneratorAgentName = Field(..., description="Agent that should handle this dependency")
    
    # Context for handoff
    source_resource: str = Field(..., description="Resource that triggered this dependency")
    requirement_details: Dict[str, Any] = Field(..., description="Specific details about what's needed")
    priority_level: int = Field(default=3, ge=1, le=5, description="Priority level (1=low, 5=critical)")
    
    # Handoff payload
    handoff_context: Dict[str, Any] = Field(..., description="Context to pass to target agent")
    expected_response: Dict[str, str] = Field(..., description="Expected response format from target agent")
    
    # Timing and blocking
    is_blocking: bool = Field(default=True, description="Whether resource generation should wait for this")
    timeout_minutes: int = Field(default=30, description="Maximum time to wait for dependency resolution")

class ResourceGenerationMetrics(BaseModel):
    """Metrics and performance data for resource generation"""
    
    total_resources_generated: int = Field(..., description="Number of resources successfully generated")
    generation_duration_seconds: float = Field(..., description="Time taken for generation")
    dependencies_discovered: int = Field(..., description="Total dependencies found")
    handoffs_required: int = Field(..., description="Number of handoffs needed")
    
    # Resource type breakdown
    resource_type_counts: Dict[Union[ResourceBlockType, str], int] = Field(default_factory=dict)
    complexity_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Complexity score (0-10)")
    
    # Error tracking
    validation_errors: List[str] = Field(default_factory=list)
    warning_messages: List[str] = Field(default_factory=list)


class HandoffRecommendation(BaseModel):
    """Recommendation for agent handoff with specific context"""
    
    target_agent: GeneratorAgentName = Field(..., description="Recommended target agent")
    handoff_reason: str = Field(..., description="Reason for handoff")
    handoff_priority: int = Field(default=3, ge=1, le=5)
    
    # Handoff data
    context_payload: Dict[str, Any] = Field(..., description="Data to pass to target agent")
    expected_deliverables: List[str] = Field(..., description="What the target agent should produce")
    
    # Coordination
    should_wait_for_completion: bool = Field(default=True)
    can_continue_parallel: bool = Field(default=False)

class TerraformResourceGenerationResponse(BaseModel):
    """Complete response from generate_terraform_resources tool"""
    
    # Generation results
    generated_resources: List[TerraformResourceBlock] = Field(..., description="Successfully generated resource blocks")
    discovered_dependencies: List[DiscoveredDependency] = Field(default_factory=list, description="Dependencies requiring handoffs")
    
    # Agent coordination
    handoff_recommendations: List[HandoffRecommendation] = Field(default_factory=list, description="Recommended handoffs to other agents")
    completion_status: str = Field(..., description="Current completion status")
    next_recommended_action: str = Field(..., description="Recommended next action")
    
    # Generation metadata
    generation_metadata: ResourceGenerationMetrics = Field(..., description="Generation performance metrics")
    generation_timestamp: Optional[datetime] = Field(default=None, description="Timestamp when resources were generated")
    
    # Complete resources file
    complete_resources_file: str = Field(..., description="Complete resources.tf file content")
    
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
            'no_resources_found',
            'completed_no_resources',
            'completed_with_dependencies',
            'partial_completion',
            'requires_human_review',
            'escalated'
        ]
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v



@tool("generate_terraform_resources")
def generate_terraform_resources(
    state: Annotated[Any, InjectedState] = None,
) -> TerraformResourceGenerationResponse:
    """
    Generate Terraform AWS resource blocks from execution plan specifications and agent requests.
    
    Args:
        state: GeneratorSwarmState containing all the data (execution_plan_data, agent_workspaces, planning_context)
    
    This function analyzes resource requirements, generates HCL blocks, identifies dependencies,
    and provides handoff recommendations to other agents in the planning stage. It supports
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
                and model_extra['metadata'].get('handoff_destination') == 'resource_configuration_agent'):
                state_updates = model_extra['metadata']['state_updates']
                tool_message_analysis["agent_status_matrix"] = state_updates.get('agent_status_matrix')
                tool_message_analysis["pending_dependencies"] = state_updates.get('pending_dependencies')
                tool_message_analysis["dependency_graph"] = state_updates.get('dependency_graph')
                tool_message_analysis["agent_workspaces"] = state_updates.get('agent_workspaces')
                tool_message_analysis["handoff_queue"] = state_updates.get('handoff_queue')

    resource_generator_logger.log_structured(
        level="INFO",
        message="Last 3 messages ToolMessage state updates analysis",
        extra={
            "total_messages": len(last_3_messages),
            "tool_message_count": len(tool_message_analysis),
            "state_updates_analysis": tool_message_analysis
        }
    )
    
    previous_state = get_current_state()

    if isinstance(previous_state, str):
        try:
            previous_state = json.loads(previous_state)
        except json.JSONDecodeError as e:
            previous_state = {}
            resource_generator_logger.log_structured(
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
            agent_workspaces = tool_message_analysis.get("agent_workspaces", {}).get("resource_configuration_agent", {})
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
            
    

    # Get generation context from planning context
        generation_context = planning_context
        
        resource_generator_logger.log_structured(
            level="INFO",
            message="Starting Terraform resource generation",
            extra={
                "resource_specifications_count": len(planning_resource_specifications),
                "has_execution_plan_data": bool(execution_plan_data),
                "has_agent_workspace": bool(agent_workspace),
                "has_planning_context": bool(planning_context)
            }
        )
        
        # Pre-approval checks for high-risk resources
        # approval_results = []
        # for resource_spec in resource_specifications:
        #     approval_context = _prepare_approval_context(resource_spec, generation_context)
        #     approval_result = _check_resource_approval(approval_context, agent_workspace)
        #     approval_results.append(approval_result)
            
        #     # If approval is rejected, skip this resource
        #     if approval_result.get("status") == "rejected":
        #         continue
        
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
        
        resource_generator_logger.log_structured(
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
        
        # Format user prompt with compressed data
        def escape_json_for_template(json_str):
            """Escape curly braces in JSON strings for template compatibility"""
            return json_str.replace('{', '{{').replace('}', '}}')
        
        formatted_user_prompt = RESOURCE_CONFIGURATION_USER_PROMPT_TEMPLATE_REFINED.format(
            service_name=exec_plan.get('service_name', 'unknown'),
            module_name=exec_plan.get('module_name', 'unknown'),
            target_environment=exec_plan.get('target_environment', 'development'),
            generation_id=agent_workspace.get('generation_id', str(uuid.uuid4())),
            resource_specifications=escape_json_for_template(compressed_data['resource_specifications']),
            planning_variable_definitions=escape_json_for_template(compressed_data['variable_definitions']),
            planning_local_values=escape_json_for_template(compressed_data['local_values']),
            planning_data_sources=escape_json_for_template(compressed_data['data_sources']),
            planning_output_definitions=escape_json_for_template(compressed_data['output_definitions']),
            # planning_terraform_files=escape_json_for_template(compressed_data['terraform_files']),
            current_stage=planning_context.get('current_stage', 'generation'),
            active_agent=agent_workspace.get('active_agent', 'resource_configuration_agent'),
            workspace_generated_resources=escape_json_for_template(compressed_data['workspace_generated_resources']),
            workspace_generated_variables=escape_json_for_template(compressed_data['workspace_generated_variables']),
            workspace_generated_data_sources=escape_json_for_template(compressed_data['workspace_generated_data_sources']),
            workspace_generated_local_values=escape_json_for_template(compressed_data['workspace_generated_local_values']),
            workspace_generated_outputs=escape_json_for_template(compressed_data['workspace_generated_outputs']),
            specific_requirements_patterns=extract_specific_requirements(generation_context),
            configuration_optimizer_actionable=escape_json_for_template(compressed_data['optimizer_data']),
            handoff_context=escape_json_for_template(json.dumps(agent_workspace.get('handoff_context', {}), indent=2))
        ) 
        
        # Create parser for structured output with lenient parsing
        parser = PydanticOutputParser(pydantic_object=TerraformResourceGenerationResponse)
        
        # Build complete prompt, escaping curly braces in system prompt
        escaped_system_prompt = RESOURCE_CONFIGURATION_SYSTEM_PROMPT.replace('{', '{{').replace('}', '}}')
        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system_prompt),
            ("user", formatted_user_prompt),
            ("user", """Please respond with valid JSON matching the TerraformResourceGenerationResponse schema.

IMPORTANT: 
- Keep the JSON structure simple and valid
- Use empty arrays [] for lists if no items
- Use empty strings "" for optional string fields
- Focus on generating the core resources first
- You can return partial results if needed

{format_instructions}""")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # Create and execute chain using centralized LLM
        try:
            # Get LLM configuration from centralized config
            config_instance = Config()
            llm_config = config_instance.get_llm_config()
            
            resource_generator_logger.log_structured(
                level="DEBUG",
                message="Initializing LLM for resource generation",
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

            resource_generator_logger.log_structured(
                level="DEBUG",
                message="LLM initialized successfully for resource generation",
                extra={
                    "model_type": type(model).__name__
                }
            )
        except Exception as e:
            resource_generator_logger.log_structured(
                level="ERROR",
                message="Failed to initialize LLM for resource generation",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
        
        # chain = prompt | model | parser
        chain = prompt | model_higher | parser
        
        resource_generator_logger.log_structured(
            level="DEBUG",
            message="Executing LLM chain for resource generation",
            extra={
                "prompt_length": len(formatted_user_prompt),
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        
        # Execute the chain with error handling
        try:
            llm_response = chain.invoke({})
        except Exception as e:
            resource_generator_logger.log_structured(
                level="ERROR",
                message="LLM chain execution failed, attempting fallback",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "generation_id": agent_workspace.get('generation_id', 'unknown')
                }
            )
            
            # Create fallback response
            llm_response = TerraformResourceGenerationResponse(
                generated_resources=[],
                discovered_dependencies=[],
                handoff_recommendations=[],
                completion_status="error",
                next_recommended_action="retry_with_simplified_prompt",
                generation_metadata=ResourceGenerationMetrics(
                    total_resources_generated=0,
                    generation_duration_seconds=(datetime.now() - start_time).total_seconds(),
                    dependencies_discovered=0,
                    handoffs_required=0,
                    validation_errors=[f"LLM execution failed: {str(e)}"]
                ),
                complete_resources_file="",
                state_updates={
                    'agent_status_matrix': {
                        **agent_workspace.get('agent_status_matrix', {}),
                        'resource_configuration_agent': 'error'
                    }
                },
                workspace_updates={
                    'error': str(e),
                    'completion_status': 'error',
                    'error_timestamp': datetime.now().isoformat()
                },
                critical_errors=[f"LLM execution failed: {str(e)}"],
                checkpoint_data={
                    'stage': 'planning',
                    'agent': 'resource_configuration_agent',
                    'checkpoint_type': 'llm_execution_error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            )
        
        resource_generator_logger.log_structured(
            level="DEBUG",
            message="LLM response received, starting post-processing",
            extra={
                "generated_resources_count": len(llm_response.generated_resources),
                "discovered_dependencies_count": len(llm_response.discovered_dependencies),
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        
        # Post-process and enhance response
        enhanced_response = post_process_llm_response(
            llm_response, 
            agent_workspace, 
            generation_context, 
            start_time
        )
        
        resource_generator_logger.log_structured(
            level="INFO",
            message="Terraform resource generation completed successfully",
            extra={
                "final_resources_count": len(enhanced_response.generated_resources),
                "final_dependencies_count": len(enhanced_response.discovered_dependencies),
                "generation_duration_seconds": enhanced_response.generation_metadata.generation_duration_seconds,
                "completion_status": enhanced_response.completion_status,
                "generation_id": agent_workspace.get('generation_id', 'unknown')
            }
        )
        
        update_agent_workspace(
            "resource_configuration_agent", {
                "handoff_recommendations": enhanced_response.handoff_recommendations,
                "complete_resources_file": enhanced_response.complete_resources_file,
                **enhanced_response.workspace_updates  # Include all workspace_updates
            }
        )
        resolved_dependencies = previous_state.get("pending_dependencies", {}).get("resource_configuration_agent", [])
        # Get current resolved dependencies and append new ones
        current_resolved_deps = get_current_state().get("resolved_dependencies", {})
        updated_resolved_deps = {
            **current_resolved_deps,
            "resource_configuration_agent": [
                *current_resolved_deps.get("resource_configuration_agent", []),
                *resolved_dependencies
            ]
        }

        update_current_state({
            "resolved_dependencies": updated_resolved_deps
        })
        # Return only state_updates as JSON for LangGraph state management
        return enhanced_response.state_updates
        
    except Exception as e:
        resource_generator_logger.log_structured(
            level="ERROR",
            message="Terraform resource generation failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "generation_id": agent_workspace.get('generation_id', 'unknown'),
                "current_stage": planning_context.get('current_stage', 'unknown')
            }
        )
        return create_error_response(e, agent_workspace, datetime.now())


def extract_specific_requirements(context: Dict[str, Any]) -> str:
    """Extract specific requirements from context including architecture patterns, well-architected alignment, and security considerations"""
    requirements = []
    
    # Extract from planner data structure
    planner_data = context.get('planner_data', {})
    
    # Extract architecture patterns from AWS service mapping
    aws_service_mapping = planner_data.get('requirements_data', {}).get('aws_service_mapping', {})
    services = aws_service_mapping.get('services', [])
    if services:
        first_service = services[0]
        if 'architecture_patterns' in first_service:
            requirements.append(f"Architecture Patterns: {first_service['architecture_patterns']}")
        
        if 'well_architected_alignment' in first_service:
            requirements.append(f"Well-Architected Alignment: {first_service['well_architected_alignment']}")
        
        if 'cost_optimization_recommendations' in first_service:
            requirements.append(f"Cost Optimization: {first_service['cost_optimization_recommendations']}")
    
    # Extract security considerations from module structure plan
    execution_data = planner_data.get('execution_data', {})
    module_structure_plan = execution_data.get('module_structure_plan', {})
    module_structure_plans = module_structure_plan.get('module_structure_plans', [])
    if module_structure_plans:
        first_plan = module_structure_plans[0]
        if 'security_considerations' in first_plan:
            requirements.append(f"Security Considerations: {first_plan['security_considerations']}")
        
        if 'reusability_guidance' in first_plan:
            requirements.append(f"Reusability Guidance: {first_plan['reusability_guidance']}")
    
    requirements_text = '\n'.join(requirements) if requirements else "No specific requirements specified"
    
    # Escape curly braces to prevent template variable interpretation
    return requirements_text.replace('{', '{{').replace('}', '}}')

def extract_planning_individual_results(context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract planning results from individual agents in the generator swarm"""
    planning_results = {}
    
    # Extract execution plan data structure
    exec_plan = context.get('execution_plan', {})
    
    # Extract terraform files structure
    if 'terraform_files' in exec_plan:
        planning_results['terraform_files'] = exec_plan['terraform_files']
    
    # Extract variable definitions
    if 'variable_definitions' in exec_plan:
        planning_results['variable_definitions'] = exec_plan['variable_definitions']
    
    # Extract local values
    if 'local_values' in exec_plan:
        planning_results['local_values'] = exec_plan['local_values']
    
    # Extract data sources
    if 'data_sources' in exec_plan:
        planning_results['data_sources'] = exec_plan['data_sources']
    
    # Extract output definitions
    if 'output_definitions' in exec_plan:
        planning_results['output_definitions'] = exec_plan['output_definitions']
    
    # Extract service and module metadata
    planning_results['service_metadata'] = {
        'service_name': exec_plan.get('service_name', 'unknown'),
        'module_name': exec_plan.get('module_name', 'unknown'),
        'target_environment': exec_plan.get('target_environment', 'development')
    }
    
    # Extract planning context metadata
    planning_results['planning_metadata'] = {
        'dependencies': context.get('dependencies', []),
        'security_considerations': context.get('security_considerations', []),
        'cost_estimates': context.get('cost_estimates', {})
    }
    
    return planning_results

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

def post_process_llm_response(
    llm_response: TerraformResourceGenerationResponse,
    agent_workspace: Dict[str, Any], 
    context: Dict[str, Any],
    start_time: datetime
) -> TerraformResourceGenerationResponse:
    """Post-process LLM response with comprehensive validation and enhancements"""
    
    # Create focused context with only essential data points
    focused_context = create_focused_context(context)
    
    # Calculate actual generation duration
    generation_duration = (datetime.now() - start_time).total_seconds()
    llm_response.generation_metadata.generation_duration_seconds = generation_duration
    
    # Fix generation timestamp if it's empty or invalid
    if not llm_response.generation_timestamp or llm_response.generation_timestamp == "":
        llm_response.generation_timestamp = datetime.now()
    
    # Validate generated resources with flexible validation
    validated_resources = []
    validation_errors = []
    validation_warnings = []
    
    for resource in llm_response.generated_resources:
        validation_result = validate_terraform_resource(resource)
        
        # Always include resources unless they have critical errors
        if validation_result['valid']:
            validated_resources.append(resource)
            # Add warnings to recoverable warnings
            if validation_result.get('warnings'):
                validation_warnings.extend(validation_result['warnings'])
        else:
            # Only filter out resources with critical errors
            validation_errors.extend(validation_result['errors'])
            # Attempt to fix critical issues
            fixed_resource = attempt_resource_fix(resource, validation_result['errors'])
            if fixed_resource:
                validated_resources.append(fixed_resource)
                llm_response.recoverable_warnings.append(
                    f"Fixed critical validation issues for {resource.resource_name}"
                )
            else:
                # If we can't fix critical issues, still include the resource but log the error
                validated_resources.append(resource)
                llm_response.recoverable_warnings.append(
                    f"Resource {resource.resource_name} has critical issues but included anyway"
                )
    
    # Update response with validated resources
    llm_response.generated_resources = validated_resources
    llm_response.generation_metadata.validation_errors.extend(validation_errors)
    
    # Add validation warnings to recoverable warnings
    if validation_warnings:
        llm_response.recoverable_warnings.extend(validation_warnings)
    
    # Generate complete resources file
    llm_response.complete_resources_file = generate_complete_resources_file(validated_resources)
    
    # Use original discovered dependencies without enhancement
    # The LLM already provides the necessary context for each dependency
    
    # Create comprehensive handoff recommendations
    llm_response.handoff_recommendations = create_enhanced_handoff_recommendations(
        llm_response.discovered_dependencies,
        validated_resources
    )
    
    # Add comprehensive state updates
    llm_response.state_updates = create_comprehensive_state_updates(
        validated_resources,
        llm_response.discovered_dependencies,
        agent_workspace,
        llm_response.completion_status
    )
    
    # Add workspace updates
    llm_response.workspace_updates = create_workspace_updates(
        validated_resources,
        llm_response.discovered_dependencies,
        llm_response.generation_metadata,
        llm_response.completion_status
    )
    
    # Add checkpoint data
    llm_response.checkpoint_data = create_checkpoint_data(
        validated_resources,
        llm_response.discovered_dependencies,
        llm_response.completion_status
    )
    
    return llm_response

def validate_terraform_resource(resource: TerraformResourceBlock) -> Dict[str, Any]:
    """Validate individual Terraform resource with flexible validation"""
    errors = []
    warnings = []
    
    # Only validate critical issues that would break Terraform
    # Be more lenient with non-critical issues
    
    # 1. Basic HCL syntax (critical)
    if not validate_hcl_syntax(resource.hcl_block):
        errors.append(f"Invalid HCL syntax in {resource.resource_name}")
    
    # 2. AWS resource type validation (more flexible)
    if not validate_aws_resource_type_flexible(resource.resource_type, resource.configuration_attributes):
        warnings.append(f"Resource type {resource.resource_type} may need review")
    
    # 3. Naming convention (warning only, not blocking)
    if not validate_naming_convention(resource.resource_name):
        warnings.append(f"Resource name {resource.resource_name} doesn't follow conventions")
    
    # 4. Skip required attributes validation - trust the LLM
    # No validation needed for required attributes
    
    # Only fail validation for critical errors, not warnings
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def validate_hcl_syntax(hcl_block: str) -> bool:
    """Basic HCL syntax validation"""
    try:
        # Check for balanced braces
        if hcl_block.count('{') != hcl_block.count('}'):
            return False
        
        # Check for valid resource block structure
        if not re.match(r'resource\s+"[^"]+"\s+"[^"]+"\s*{', hcl_block):
            return False
        
        # Check for proper attribute formatting
        lines = hcl_block.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line in ['{', '}']:
                if '=' not in line and 'resource' not in line and '{' not in line and '}' not in line:
                    return False
        
        return True
    except Exception:
        return False

def validate_aws_resource_type(resource_type: Union[ResourceBlockType, str], attributes: Dict[str, Any]) -> bool:
    """Validate AWS resource type and basic attributes with support for dynamic types"""
    
    # Convert to string for validation
    resource_type_str = str(resource_type)
    
    # Define basic required attributes for common resource types
    required_attrs = {
        'aws_instance': ['instance_type'],
        'aws_vpc': ['cidr_block'],
        'aws_subnet': ['vpc_id', 'cidr_block'],
        'aws_security_group': ['name'],
        'aws_s3_bucket': [],  # bucket name can be auto-generated
        'aws_db_instance': ['instance_class', 'engine'],
        'aws_lambda_function': ['function_name', 'role', 'handler', 'runtime'],
        'aws_internet_gateway': ['vpc_id'],
        'aws_nat_gateway': ['subnet_id', 'allocation_id'],
        'aws_route_table': ['vpc_id'],
        'aws_route': ['route_table_id'],
        'aws_eip': ['domain'],
        'aws_vpc_endpoint': ['vpc_id', 'service_name'],
        'aws_cloudwatch_log_group': ['name'],
        'aws_iam_role': ['name', 'assume_role_policy'],
        'aws_iam_role_policy': ['name', 'policy', 'role'],
        'aws_flow_log': ['resource_id', 'resource_type', 'traffic_type']
    }
    
    if resource_type_str not in required_attrs:
        # For dynamic resource types, perform basic validation
        return validate_dynamic_resource_type(resource_type_str, attributes)
    
    for attr in required_attrs[resource_type_str]:
        if attr not in attributes:
            return False
    
    return True

def validate_aws_resource_type_flexible(resource_type: Union[ResourceBlockType, str], attributes: Dict[str, Any]) -> bool:
    """Flexible AWS resource type validation - more lenient for all AWS resources"""
    
    # Convert to string for validation
    resource_type_str = str(resource_type)
    
    # Only validate that it's an AWS resource type
    if not resource_type_str.startswith('aws_'):
        return False
    
    # For all AWS resources, be flexible with configuration
    # AWS resources can have various configurations and optional attributes
    if not attributes:
        return False
    
    # Allow any AWS resource type with basic attributes
    # The LLM is smart enough to generate valid configurations
    return True

def validate_dynamic_resource_type(resource_type: str, attributes: Dict[str, Any]) -> bool:
    """Validate dynamic resource types not in the predefined list"""
    # Basic validation for unknown AWS resource types
    if not resource_type.startswith('aws_'):
        return False
    
    # Check for basic Terraform resource structure
    if not attributes:
        return False
    
    # For dynamic types, we're more lenient - just ensure it looks like a valid AWS resource
    return True

def validate_naming_convention(resource_name: str) -> bool:
    """Validate Terraform resource naming convention"""
    # Check if name follows snake_case convention
    if not re.match(r'^[a-z][a-z0-9_]*$', resource_name):
        return False
    
    # Check if name is not too long
    if len(resource_name) > 64:
        return False
    
    # Check for reserved words
    reserved_words = ['resource', 'data', 'variable', 'output', 'module', 'provider']
    if resource_name in reserved_words:
        return False
    
    return True

def validate_required_attributes(resource: TerraformResourceBlock) -> List[str]:
    """Check for missing required attributes"""
    missing = []
    
    for attr in resource.required_attributes:
        if attr not in resource.configuration_attributes:
            missing.append(attr)
    
    return missing


def attempt_resource_fix(
    resource: TerraformResourceBlock, 
    errors: List[str]
) -> Optional[TerraformResourceBlock]:
    """Attempt to fix common resource issues with support for dynamic types"""
    
    fixed_resource = resource.copy(deep=True)
    
    # Try to fix HCL syntax issues
    if any("Invalid HCL syntax" in error for error in errors):
        fixed_hcl = fix_hcl_syntax(resource.hcl_block)
        if fixed_hcl != resource.hcl_block:
            fixed_resource.hcl_block = fixed_hcl
    
    # Try to fix naming issues
    if any("doesn't follow conventions" in error for error in errors):
        fixed_name = fix_resource_name(resource.resource_name)
        if fixed_name != resource.resource_name:
            fixed_resource.resource_name = fixed_name
            # Update HCL block with new name
            fixed_resource.hcl_block = re.sub(
                f'resource\\s+"{resource.resource_type}"\\s+"{resource.resource_name}"',
                f'resource "{resource.resource_type}" "{fixed_name}"',
                fixed_resource.hcl_block
            )
    
    # Validate the fixed resource
    validation_result = validate_terraform_resource(fixed_resource)
    if validation_result['valid']:
        return fixed_resource
    
    return None

def fix_hcl_syntax(hcl_block: str) -> str:
    """Attempt to fix common HCL syntax issues"""
    
    # Fix common formatting issues
    fixed = hcl_block
    
    # Ensure proper spacing around equals signs
    fixed = re.sub(r'\s*=\s*', ' = ', fixed)
    
    # Ensure proper indentation (2 spaces)
    lines = fixed.split('\n')
    formatted_lines = []
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append('')
            continue
        
        if stripped == '}':
            indent_level = max(0, indent_level - 1)
        
        formatted_lines.append('  ' * indent_level + stripped)
        
        if stripped.endswith('{'):
            indent_level += 1
    
    return '\n'.join(formatted_lines)

def fix_resource_name(resource_name: str) -> str:
    """Fix resource naming convention issues"""
    
    # Convert to lowercase
    fixed = resource_name.lower()
    
    # Replace invalid characters with underscores
    fixed = re.sub(r'[^a-z0-9_]', '_', fixed)
    
    # Ensure it starts with a letter
    if fixed[0].isdigit():
        fixed = 'r_' + fixed
    
    # Remove multiple consecutive underscores
    fixed = re.sub(r'_+', '_', fixed)
    
    # Remove leading/trailing underscores
    fixed = fixed.strip('_')
    
    # Ensure it's not too long
    if len(fixed) > 64:
        fixed = fixed[:64].rstrip('_')
    
    return fixed

def create_enhanced_handoff_recommendations(
    dependencies: List[DiscoveredDependency],
    resources: List[TerraformResourceBlock]
) -> List[HandoffRecommendation]:
    """Create comprehensive handoff recommendations"""
    
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
            deliverables = ['Variable definitions with validation rules', 'Variable descriptions and examples']
        elif agent == 'data_source_agent':
            deliverables = ['Data source configurations', 'External data lookups']
        elif agent == 'local_values_agent':
            deliverables = ['Local value definitions', 'Computed expressions']
        else:
            deliverables = ['Required configurations']
        
        recommendation = HandoffRecommendation(
            target_agent=agent,
            handoff_reason=f"Resolve {len(deps)} dependencies for resource generation",
            handoff_priority=max_priority,
            context_payload={
                'dependencies': [dep.model_dump() for dep in deps],
                'total_count': len(deps),
                'priority_levels': [dep.priority_level for dep in deps],
                'affected_resources': [dep.source_resource for dep in deps]
            },
            expected_deliverables=deliverables,
            should_wait_for_completion=has_blocking,
            can_continue_parallel=not has_blocking
        )
        
        recommendations.append(recommendation)
    
    return recommendations

def create_comprehensive_state_updates(
    resources: List[TerraformResourceBlock],
    dependencies: List[DiscoveredDependency],
    agent_workspace: Dict[str, Any],
    completion_status: str
) -> Dict[str, Any]:
    """Create comprehensive state updates for the swarm"""
    
    updates = {
        'terraform_resources': [resource.model_dump() for resource in resources],
        'pending_dependencies': {
            **agent_workspace.get('pending_dependencies', {}),
            'resource_configuration_agent': [dep.model_dump() for dep in dependencies]
        },
        'agent_status_matrix': {
            **agent_workspace.get('agent_status_matrix', {}),
            'resource_configuration_agent': completion_status
        },
        'planning_progress': {
            **agent_workspace.get('planning_progress', {}),
            'resource_configuration_agent': 1.0 if completion_status == 'completed' else 0.6
        }
    }
    
    return updates

def create_workspace_updates(
    resources: List[TerraformResourceBlock],
    dependencies: List[DiscoveredDependency],
    metrics: ResourceGenerationMetrics,
    completion_status: str
) -> Dict[str, Any]:
    """Create workspace updates for the agent"""
    
    return {
        'generated_resources': [resource.model_dump() for resource in resources],
        'pending_dependencies': [dep.model_dump() for dep in dependencies],
        'generation_metrics': metrics.model_dump(),
        'completion_status': completion_status,
        'completion_timestamp': datetime.now().isoformat(),
        'resource_summary': {
            'total_resources': len(resources),
            'resource_types': list(set(r.resource_type for r in resources)),
            'dependencies_discovered': len(dependencies),
            'complexity_score': metrics.complexity_score
        }
    }

def create_checkpoint_data(
    resources: List[TerraformResourceBlock],
    dependencies: List[DiscoveredDependency],
    completion_status: str
) -> Dict[str, Any]:
    """Create checkpoint data for recovery"""
    
    return {
        'stage': 'planning',
        'agent': 'resource_configuration_agent',
        'checkpoint_type': 'resource_generation_complete',
        'resources_generated': len(resources),
        'dependencies_discovered': len(dependencies),
        'completion_status': completion_status,
        'timestamp': datetime.now().isoformat(),
        'resource_names': [r.resource_name for r in resources],
        'dependency_types': [d.dependency_type for d in dependencies]
    }

def create_error_response(
    error: Exception, 
    agent_workspace: Dict[str, Any], 
    start_time: datetime
) -> TerraformResourceGenerationResponse:
    """Create error response when tool execution fails"""
    
    generation_duration = (datetime.now() - start_time).total_seconds()
    
    return TerraformResourceGenerationResponse(
        generated_resources=[],
        discovered_dependencies=[],
        handoff_recommendations=[],
        completion_status='error',
        next_recommended_action='escalate_to_human_or_retry',
        generation_metadata=ResourceGenerationMetrics(
            total_resources_generated=0,
            generation_duration_seconds=generation_duration,
            dependencies_discovered=0,
            handoffs_required=0,
            validation_errors=[f"Tool execution failed: {str(error)}"]
        ),
        state_updates={
            'agent_status_matrix': {
                **agent_workspace.get('agent_status_matrix', {}),
                'resource_configuration_agent': 'error'
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
            'agent': 'resource_configuration_agent',
            'checkpoint_type': 'error',
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
    )


def _prepare_approval_context(resource_spec: Dict[str, Any], generation_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare approval context for resource creation.
    
    Args:
        resource_spec: Resource specification
        generation_context: Generation context
        
    Returns:
        Approval context dictionary
    """
    return {
        "type": resource_spec.get("type"),
        "estimated_monthly_cost": _estimate_resource_cost(resource_spec),
        "regions": _extract_regions(resource_spec),
        "uses_experimental_features": _check_experimental_features(resource_spec),
        "security_impact": _assess_security_impact(resource_spec),
        "resource_config": resource_spec,
        "generation_context": generation_context
    }


def _check_resource_approval(approval_context: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if resource creation requires approval.
    
    Args:
        approval_context: Context for approval decision
        state: Current state
        
    Returns:
        Approval result dictionary
    """
    # This would integrate with the ApprovalMiddleware in a real implementation
    # For now, we'll simulate the approval check logic
    
    # Check for high-cost resources
    if approval_context.get("estimated_monthly_cost", 0) > 1000:
        return {
            "status": "approved",  # Would be determined by human approval
            "reason": "High-cost resource approved",
            "trigger_type": "high_cost_resources"
        }
    
    # Check for security-critical resources
    if approval_context.get("security_impact") == "high":
        return {
            "status": "approved",  # Would be determined by human approval
            "reason": "Security-critical resource approved",
            "trigger_type": "security_critical"
        }
    
    # Check for cross-region resources
    regions = approval_context.get("regions", [])
    if len(set(regions)) > 1:
        return {
            "status": "approved",  # Would be determined by human approval
            "reason": "Cross-region resource approved",
            "trigger_type": "cross_region"
        }
    
    # Check for experimental features
    if approval_context.get("uses_experimental_features", False):
        return {
            "status": "approved",  # Would be determined by human approval
            "reason": "Experimental feature approved",
            "trigger_type": "experimental"
        }
    
    return {"status": "approved", "reason": "No approval required"}


def _estimate_resource_cost(resource_spec: Dict[str, Any]) -> float:
    """Estimate monthly cost for a resource."""
    # Simplified cost estimation logic
    resource_type = resource_spec.get("type", "")
    
    # Basic cost estimates (simplified)
    cost_map = {
        "aws_instance": 50.0,
        "aws_rds_instance": 200.0,
        "aws_lb": 25.0,
        "aws_nat_gateway": 45.0,
        "aws_elasticache_cluster": 150.0
    }
    
    base_cost = cost_map.get(resource_type, 10.0)
    
    # Adjust based on instance size or other factors
    instance_type = resource_spec.get("instance_type", "")
    if "large" in instance_type:
        base_cost *= 2
    elif "xlarge" in instance_type:
        base_cost *= 4
    
    return base_cost


def _extract_regions(resource_spec: Dict[str, Any]) -> List[str]:
    """Extract regions from resource specification."""
    regions = []
    
    # Check for explicit region specification
    if "region" in resource_spec:
        regions.append(resource_spec["region"])
    
    # Check for availability zones (infer region)
    if "availability_zone" in resource_spec:
        az = resource_spec["availability_zone"]
        if az:
            # Extract region from AZ (e.g., "us-west-2a" -> "us-west-2")
            region = az[:-1] if len(az) > 1 and az[-1].isalpha() else az
            regions.append(region)
    
    return list(set(regions)) if regions else ["us-east-1"]  # Default region


def _check_experimental_features(resource_spec: Dict[str, Any]) -> bool:
    """Check if resource uses experimental features."""
    # Check for experimental resource types or configurations
    experimental_types = [
        "aws_eks_fargate_profile",
        "aws_eks_node_group",
        "aws_ecs_capacity_provider"
    ]
    
    return resource_spec.get("type") in experimental_types


def _assess_security_impact(resource_spec: Dict[str, Any]) -> str:
    """Assess security impact level of resource."""
    security_critical_types = [
        "aws_iam_role", "aws_iam_policy", "aws_security_group",
        "aws_kms_key", "aws_secretsmanager_secret"
    ]
    
    if resource_spec.get("type") in security_critical_types:
        return "high"
    
    # Check for security-related configurations
    if "security_group_ids" in resource_spec or "kms_key_id" in resource_spec:
        return "medium"
    
    return "low"


def generate_complete_resources_file(resources: List[TerraformResourceBlock]) -> str:
    """Generate complete resources.tf file from resource blocks"""
    
    if not resources:
        return ""
    
    # Generate the complete file - preserve the original order from LLM
    lines = []
    lines.append("# Terraform Resources")
    lines.append(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Add resources in the order they were generated by the LLM
    for resource in resources:
        lines.append(resource.hcl_block)
        lines.append("")
    
    return "\n".join(lines)

