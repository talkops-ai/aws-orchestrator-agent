import json
import uuid
from typing import Dict, List, Any, Optional, Annotated
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.prebuilt import InjectedState
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger
from ..global_state import get_current_state, set_current_state, update_agent_workspace, update_current_state
from .backend_generator_prompts import TERRAFORM_BACKEND_GENERATOR_SYSTEM_PROMPT, TERRAFORM_BACKEND_GENERATOR_USER_PROMPT
# Create agent logger for terraform backend generator
backend_generator_logger = AgentLogger("BACKEND_GENERATOR")


class TerraformBackendConfig(BaseModel):
    """Complete Terraform backend configuration including provider and S3 backend settings"""
    
    # Provider configuration
    provider_name: str = Field(..., description="Provider name (e.g., 'aws')")
    provider_source: str = Field(..., description="Provider source (e.g., 'hashicorp/aws')")
    provider_version: str = Field(..., description="Version constraint (e.g., '>= 5.0')")
    
    # S3 backend configuration
    bucket_name: str = Field(..., description="S3 bucket name for state storage")
    key_pattern: str = Field(..., description="Key pattern for state files")
    region: str = Field(..., description="AWS region for S3 bucket")
    dynamodb_table: Optional[str] = Field(None, description="DynamoDB table for state locking")
    encrypt: bool = Field(True, description="Enable S3 encryption")
    versioning: bool = Field(True, description="Enable S3 versioning")
    kms_key_id: Optional[str] = Field(None, description="KMS key ID for encryption")
    sse_algorithm: str = Field("AES256", description="Server-side encryption algorithm")


class TerraformBackendGenerationResponse(BaseModel):
    """Complete response from generate_terraform_backend tool"""
    
    # Generated configuration blocks
    generated_backend_configs: List[TerraformBackendConfig] = Field(..., description="Successfully generated backend configurations")
    terraform_block_hcl: str = Field(..., description="Complete terraform{} configuration block")
    provider_block_hcl: str = Field(..., description="Complete provider{} configuration block")
    complete_configuration: str = Field(..., description="Combined terraform and provider blocks hcl code")
    
    # # Validation and security
    # validation_status: str = Field(..., description="Validation status: valid/invalid/warnings")
    # security_recommendations: List[str] = Field(default_factory=list, description="Security best practices")
    # implementation_notes: List[str] = Field(default_factory=list, description="Implementation guidance")
    # warnings: List[str] = Field(default_factory=list, description="Configuration warnings")
    
    # Agent coordination
    completion_status: str = Field(..., description="Generation completion status")
    # next_recommended_action: str = Field(..., description="Recommended next action")
    
    # State updates
    state_updates: Dict[str, Any] = Field(default_factory=dict, description="Updates to apply to swarm state")
    workspace_updates: Dict[str, Any] = Field(..., description="Updates for agent workspace")
    
    # Metadata
    generation_timestamp: Optional[str] = Field(default=None, description="When configuration was generated (ISO format)")
    generation_duration_seconds: Optional[float] = Field(None, description="Time taken for generation")
    
    # Error handling
    critical_errors: List[str] = Field(default_factory=list, description="Critical errors blocking progress")
    recoverable_warnings: List[str] = Field(default_factory=list, description="Non-blocking warnings")
    
    # Checkpoint data
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict, description="Data for checkpoint creation")
    
    @field_validator('completion_status')
    @classmethod
    def validate_completion_status(cls, v):
        valid_statuses = ['completed', 'completed_with_warnings', 'failed', 'blocked', 'error', 'waiting_for_dependencies', 'completed_with_dependencies']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v



@tool("generate_terraform_backend")
def generate_terraform_backend(
    state: Annotated[Any, InjectedState] = None,
) -> TerraformBackendGenerationResponse:
    """
    Generate Terraform backend and provider configuration blocks from state management plans.
    
    Args:
        state: GeneratorSwarmState containing state_management_plan_data and generation_context
               
    This tool analyzes state management plans, generates secure terraform{} and provider{} blocks,
    applies enterprise security best practices, and provides implementation guidance.
    """
    
    start_time = datetime.now()
    
    backend_generator_logger.log_structured(
        level="INFO",
        message="Starting Terraform backend configuration generation",
        extra={
            "start_time": start_time.isoformat(),
            "tool_name": "generate_terraform_backend"
        }
    )
    
    # Get current state from global state management
    previous_state = get_current_state()
    if isinstance(previous_state, str):
        try:
            previous_state = json.loads(previous_state)
        except json.JSONDecodeError as e:
            previous_state = {}
            backend_generator_logger.log_structured(
                level="ERROR",
                message="Failed to parse previous state JSON",
                extra={"error": str(e)}
            )
    
    try:
        # Extract data from global state
        state_management_plan_data = previous_state.get("state_management_plan_data", {})
        generation_context = previous_state.get("generation_context", {})
        
        # Fallback mechanism: if generation_context is empty, extract from execution_plan_data
        if not generation_context:
            execution_plan_data = previous_state.get("execution_plan_data", {})
            execution_plans = execution_plan_data.get("execution_plans", [])
            
            if execution_plans:
                first_execution_plan = execution_plans[0]
                
                # Extract required providers and terraform version
                required_providers = first_execution_plan.get("required_providers", {})
                terraform_version = first_execution_plan.get("terraform_version_constraint", ">= 1.0")
                
                # Build provider versions dict from required_providers
                provider_versions = {}
                for provider_name, provider_config in required_providers.items():
                    if isinstance(provider_config, dict):
                        provider_versions[provider_name] = provider_config.get("version", ">= 1.0")
                    else:
                        provider_versions[provider_name] = ">= 1.0"
                
                # Create fallback generation_context
                generation_context = {
                    "service_name": first_execution_plan.get("service_name", "unknown"),
                    "module_name": first_execution_plan.get("module_name", "unknown"),
                    "target_environment": first_execution_plan.get("target_environment", "dev"),
                    "terraform_version": terraform_version,
                    "provider_versions": provider_versions
                }
                
                backend_generator_logger.log_structured(
                    level="INFO",
                    message="Using fallback generation_context from execution_plan_data",
                    extra={
                        "service_name": generation_context["service_name"],
                        "module_name": generation_context["module_name"],
                        "target_environment": generation_context["target_environment"],
                        "terraform_version": generation_context["terraform_version"]
                    }
                )
        
        # Get state management plans
        state_management_plans = state_management_plan_data.get("state_management_plans", [])
        
        if not state_management_plans:
            return TerraformBackendGenerationResponse(
                terraform_block_hcl="# Error: No state management plans found",
                provider_block_hcl="# Error: No state management plans found",
                complete_configuration="# Error: No state management plans found",
                completion_status="error",
                critical_errors=["No state management plans found in global state"],
                workspace_updates={
                    "error": "No state management plans found in global state",
                    "completion_status": "error",
                    "error_timestamp": datetime.now().isoformat()
                },
                state_updates={},
                generation_timestamp=datetime.now().isoformat(),
                generation_duration_seconds=(datetime.now() - start_time).total_seconds(),
                checkpoint_data={
                    "stage": "backend_generation",
                    "agent": "backend_generator",
                    "checkpoint_type": "error",
                    "error": "No state management plans found in global state",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Use the first state management plan
        state_plan = state_management_plans[0]
        state_plan_backend_conf = state_plan.get("backend_configuration", {})
        state_plan_locking_conf = state_plan.get("state_locking_configuration", {})
        state_plan_splitting_strategy = state_plan.get("state_splitting_strategy", {})

        
        backend_generator_logger.log_structured(
            level="INFO",
            message="Extracted state management plan and generation context",
            extra={
                "service_name": state_plan.get("service_name", "unknown"),
                "infrastructure_scale": state_plan.get("infrastructure_scale", "unknown"),
                "has_backend_config": bool(state_plan.get("backend_configuration")),
                "has_locking_config": bool(state_plan.get("state_locking_configuration"))
            }
        )

        # Format user prompt with actual data, escaping curly braces in JSON
        def escape_json_for_template(json_str):
            """Escape curly braces in JSON strings for template compatibility"""
            return json_str.replace('{', '{{').replace('}', '}}')

        # Format user prompt with actual data
        formatted_user_prompt = TERRAFORM_BACKEND_GENERATOR_USER_PROMPT.format(
            state_plan_backend_conf=escape_json_for_template(json.dumps(state_plan_backend_conf, indent=2)),
            state_plan_locking_conf = escape_json_for_template(json.dumps(state_plan_locking_conf, indent=2)),
            state_plan_splitting_strategy = escape_json_for_template(json.dumps(state_plan_splitting_strategy, indent=2)),
            # state_management_plan_data=json.dumps(state_management_plan_data, indent=2),
            generation_context=escape_json_for_template(json.dumps(generation_context, indent=2)),
            service_name=generation_context.get('service_name', 'unknown'),
            module_name=generation_context.get('module_name', 'unknown'),
            target_environment=generation_context.get('target_environment', 'dev'),
            generation_id=str(uuid.uuid4())
        )
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=TerraformBackendGenerationResponse)
        
        # Debug logging for parser setup
        backend_generator_logger.log_structured(
            level="DEBUG",
            message="Parser setup completed",
            extra={
                "parser_type": type(parser).__name__,
                "format_instructions_length": len(parser.get_format_instructions())
            }
        )
        
        # Build complete prompt, escaping curly braces in system prompt
        escaped_system_prompt = TERRAFORM_BACKEND_GENERATOR_SYSTEM_PROMPT.replace('{', '{{').replace('}', '}}')
        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system_prompt),
            ("user", formatted_user_prompt),
            ("user", """Please respond with valid JSON matching the TerraformBackendGenerationResponse schema.

IMPORTANT: 
- Keep the JSON structure simple and valid
- Use empty arrays [] for lists if no items
- Use empty strings "" for optional string fields
- Focus on generating the core terraform and provider blocks first
- You can return partial results if needed

{format_instructions}""")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # Create and execute chain using centralized LLM
        try:
            # Get LLM configuration
            config_instance = Config()
            llm_higher_config = config_instance.get_llm_higher_config()
            
            backend_generator_logger.log_structured(
                level="DEBUG",
                message="Initializing LLM for backend generation",
                extra={
                    "llm_provider": llm_higher_config.get('provider'),
                    "llm_model": llm_higher_config.get('model'),
                    "llm_temperature": llm_higher_config.get('temperature')
                }
            )
            
            model = LLMProvider.create_llm(
                provider=llm_higher_config['provider'],
                model=llm_higher_config['model'],
                temperature=llm_higher_config['temperature'],
                max_tokens=llm_higher_config['max_tokens']
            )
            
        except Exception as e:
            backend_generator_logger.log_structured(
                level="ERROR",
                message="Failed to initialize LLM for backend generation",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
        
        # Execute the chain
        chain = prompt | model | parser
        
        backend_generator_logger.log_structured(
            level="DEBUG",
            message="Executing LLM chain for backend generation",
            extra={
                "prompt_length": len(formatted_user_prompt)
            }
        )
        
        llm_response = chain.invoke({})
        
        # Post-process and enhance response
        end_time = datetime.now()
        generation_duration = (end_time - start_time).total_seconds()
        
        # Update response with timing and metadata
        llm_response.generation_timestamp = start_time.isoformat()
        llm_response.generation_duration_seconds = generation_duration
        
        # Add workspace updates
        llm_response.workspace_updates.update({
            "terraform_backend_generator": {
                "generated_backend_configs": llm_response.generated_backend_configs,
                "terraform_block": llm_response.terraform_block_hcl,
                "provider_block": llm_response.provider_block_hcl,
                "complete_configuration": llm_response.complete_configuration,
                "generation_timestamp": start_time.isoformat(),
                "completion_status": llm_response.completion_status
            }
        })

        # Update agent workspace with specific fields like backend generator
        update_agent_workspace(
            "terraform_backend_generator", {
                "generated_backend_configs": llm_response.generated_backend_configs,
                "terraform_block_hcl": llm_response.terraform_block_hcl,
                "provider_block_hcl": llm_response.provider_block_hcl,
                "complete_configuration": llm_response.complete_configuration
            }
        )
        
        backend_generator_logger.log_structured(
            level="INFO",
            message="Terraform backend configuration generation completed successfully",
            extra={
                "generation_duration_seconds": generation_duration,
                "completion_status": llm_response.completion_status,
                "warnings_count": len(llm_response.recoverable_warnings),
                "errors_count": len(llm_response.critical_errors)
            }
        )
        
        return llm_response.workspace_updates
        
    except Exception as e:
        backend_generator_logger.log_structured(
            level="ERROR",
            message="Error during Terraform backend generation",
            extra={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        
        return TerraformBackendGenerationResponse(
            terraform_block_hcl="# Error during generation",
            provider_block_hcl="# Error during generation", 
            complete_configuration="# Error during generation",
            completion_status="error",
            state_updates={},
            critical_errors=[f"Generation error: {str(e)}"],
            workspace_updates={
                "error": str(e),
                "completion_status": "error",
                "error_timestamp": datetime.now().isoformat()
            },
            generation_timestamp=datetime.now().isoformat(),
            generation_duration_seconds=(datetime.now() - start_time).total_seconds(),
            checkpoint_data={
                "stage": "backend_generation",
                "agent": "backend_generator",
                "checkpoint_type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
