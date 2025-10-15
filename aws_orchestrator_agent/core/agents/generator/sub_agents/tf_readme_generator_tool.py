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
from .readme_generator_prompts import TERRAFORM_README_GENERATOR_SYSTEM_PROMPT, TERRAFORM_README_GENERATOR_USER_PROMPT

# Create agent logger for terraform readme generator
readme_generator_logger = AgentLogger("README_GENERATOR")


class ReadmeComponent(BaseModel):
    """Unified README component structure"""
    component_type: str = Field(..., description="Type of component: usage_example, module_metadata, security_consideration, cost_implication")
    title: str = Field(..., description="Title or name of the component")
    content: str = Field(..., description="Main content/description of the component")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata specific to component type")
    priority: str = Field(default="medium", description="Priority level: high/medium/low")
    category: str = Field(..., description="Category or section this component belongs to")


class TerraformReadmeGenerationResponse(BaseModel):
    """Complete response from generate_terraform_readme tool"""
    
    # Generated README content
    readme_content: str = Field(..., description="Complete README.md content with all sections")
    
    # Structured content components
    generated_readme_components: List[ReadmeComponent] = Field(default_factory=list, description="Generated README components (usage examples, metadata, security, cost)")
    
    # Metadata and structure
    table_of_contents: str = Field(..., description="Generated table of contents")
    sections_count: int = Field(..., description="Number of sections in the README")
    word_count: int = Field(..., description="Total word count of the README")
    
    # Agent coordination
    completion_status: str = Field(..., description="Generation completion status")
    
    # State updates
    state_updates: Dict[str, Any] = Field(default_factory=dict, description="Updates to apply to swarm state")
    workspace_updates: Dict[str, Any] = Field(..., description="Updates for agent workspace")
    
    # Metadata
    generation_timestamp: Optional[str] = Field(default=None, description="When README was generated (ISO format)")
    generation_duration_seconds: Optional[float] = Field(None, description="Time taken for generation")
    
    # Error handling
    critical_errors: List[str] = Field(default_factory=list, description="Critical errors blocking progress")
    recoverable_warnings: List[str] = Field(default_factory=list, description="Non-blocking warnings")
    
    # Checkpoint data
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict, description="Data for checkpoint creation")
    
    @field_validator('completion_status')
    @classmethod
    def validate_completion_status(cls, v):
        valid_statuses = ['completed', 'complete', 'completed_with_warnings', 'failed', 'blocked', 'error', 'waiting_for_dependencies', 'completed_with_dependencies']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v


@tool("generate_terraform_readme")
def generate_terraform_readme(
    state: Annotated[Any, InjectedState] = None,
) -> TerraformReadmeGenerationResponse:
    """
    Generate comprehensive README.md documentation for Terraform modules from execution plan data.
    
    Args:
        state: GeneratorSwarmState containing execution_plan_data and generation_context
               
    This tool analyzes execution plans, usage examples, and existing README content to generate
    enterprise-grade documentation with usage examples, security considerations, cost implications,
    and troubleshooting guides.
    """
    
    start_time = datetime.now()
    
    readme_generator_logger.log_structured(
        level="INFO",
        message="Starting Terraform README generation",
        extra={
            "start_time": start_time.isoformat(),
            "tool_name": "generate_terraform_readme"
        }
    )
    
    # Get current state from global state management
    previous_state = get_current_state()
    if isinstance(previous_state, str):
        try:
            previous_state = json.loads(previous_state)
        except json.JSONDecodeError as e:
            previous_state = {}
            readme_generator_logger.log_structured(
                level="ERROR",
                message="Failed to parse previous state JSON",
                extra={"error": str(e)}
            )
    
    try:
        # Extract data from global state
        execution_plan_data = previous_state.get("execution_plan_data", {})
        generation_context = previous_state.get("generation_context", {})
        generated_resources = previous_state.get("agent_workspaces", {}).get("resource_configuration_agent", {}).get("complete_resources_file", "")
        generated_variables = previous_state.get("agent_workspaces", {}).get("variable_definition_agent", {}).get("complete_variables_file", "")
        generated_data_sources = previous_state.get("agent_workspaces", {}).get("data_source_agent", {}).get("complete_data_sources_file", "")
        generated_local_values = previous_state.get("agent_workspaces", {}).get("local_values_agent", {}).get("complete_locals_file", "")
        generated_output_definitions = previous_state.get("agent_workspaces", {}).get("output_definition_agent", {}).get("complete_outputs_file", "")
        # Fallback mechanism: if generation_context is empty, extract from execution_plan_data
        if not generation_context:
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
                
                readme_generator_logger.log_structured(
                    level="INFO",
                    message="Using fallback generation_context from execution_plan_data",
                    extra={
                        "service_name": generation_context["service_name"],
                        "module_name": generation_context["module_name"],
                        "target_environment": generation_context["target_environment"],
                        "terraform_version": generation_context["terraform_version"]
                    }
                )
        
        # Get execution plans
        execution_plans = execution_plan_data.get("execution_plans", [])
        
        if not execution_plans:
            return TerraformReadmeGenerationResponse(
                readme_content="# Error: No execution plans found",
                table_of_contents="# Error: No execution plans found",
                sections_count=0,
                word_count=0,
                completion_status="error",
                critical_errors=["No execution plans found in global state"],
                workspace_updates={
                    "error": "No execution plans found in global state",
                    "completion_status": "error",
                    "error_timestamp": datetime.now().isoformat()
                },
                state_updates={},
                generation_timestamp=datetime.now().isoformat(),
                generation_duration_seconds=(datetime.now() - start_time).total_seconds(),
                checkpoint_data={
                    "stage": "readme_generation",
                    "agent": "readme_generator",
                    "checkpoint_type": "error",
                    "error": "No execution plans found in global state",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Use the first execution plan
        execution_plan = execution_plans[0]
        usage_examples = execution_plan.get("usage_examples", [])
        existing_readme = execution_plan.get("readme_content", "")
        
        readme_generator_logger.log_structured(
            level="INFO",
            message="Extracted execution plan and generation context",
            extra={
                "service_name": execution_plan.get("service_name", "unknown"),
                "module_name": execution_plan.get("module_name", "unknown"),
                "usage_examples_count": len(usage_examples),
                "has_existing_readme": bool(existing_readme),
                "target_environment": generation_context.get("target_environment", "unknown")
            }
        )

        # Format user prompt with actual data, escaping curly braces in JSON
        def escape_json_for_template(json_str):
            """Escape curly braces in JSON strings for template compatibility"""
            return json_str.replace('{', '{{').replace('}', '}}')

        # Format user prompt with actual data
        formatted_user_prompt = TERRAFORM_README_GENERATOR_USER_PROMPT.format(
            usage_examples=escape_json_for_template(json.dumps(usage_examples, indent=2)),
            existing_readme=escape_json_for_template(existing_readme),
            generation_context=escape_json_for_template(json.dumps(generation_context, indent=2)),
            generated_resources=escape_json_for_template(generated_resources),
            generated_variables=escape_json_for_template(generated_variables),
            generated_data_sources=escape_json_for_template(generated_data_sources),
            generated_local_values=escape_json_for_template(generated_local_values),
            generated_output_definitions=escape_json_for_template(generated_output_definitions),
            service_name=generation_context.get('service_name', 'unknown'),
            module_name=generation_context.get('module_name', 'unknown'),
            target_environment=generation_context.get('target_environment', 'dev'),
            terraform_version=generation_context.get('terraform_version', '>= 1.0'),
            provider_versions=escape_json_for_template(json.dumps(generation_context.get('provider_versions', {}), indent=2)),
            generation_id=str(uuid.uuid4())
        )
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=TerraformReadmeGenerationResponse)
        
        # Build complete prompt, escaping curly braces in system prompt
        escaped_system_prompt = TERRAFORM_README_GENERATOR_SYSTEM_PROMPT.replace('{', '{{').replace('}', '}}')
        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system_prompt),
            ("user", formatted_user_prompt),
            ("user", """Please respond with valid JSON matching the TerraformReadmeGenerationResponse schema.

IMPORTANT: 
- Keep the JSON structure simple and valid
- Use empty arrays [] for lists if no items
- Use empty strings "" for optional string fields
- Focus on generating the core README content first
- You can return partial results if needed

{format_instructions}""")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # Create and execute chain using centralized LLM
        try:
            # Get LLM configuration
            config_instance = Config()
            llm_higher_config = config_instance.get_llm_higher_config()
            
            readme_generator_logger.log_structured(
                level="DEBUG",
                message="Initializing LLM for README generation",
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
            readme_generator_logger.log_structured(
                level="ERROR",
                message="Failed to initialize LLM for README generation",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
        
        # Execute the chain
        chain = prompt | model | parser
        
        readme_generator_logger.log_structured(
            level="DEBUG",
            message="Executing LLM chain for README generation",
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
            "terraform_readme_generator": {
                "generated_readme_components": llm_response.generated_readme_components,
                "readme_content": llm_response.readme_content,
                "table_of_contents": llm_response.table_of_contents,
                "generation_timestamp": start_time.isoformat(),
                "completion_status": llm_response.completion_status,
                "sections_count": llm_response.sections_count,
                "word_count": llm_response.word_count
            }
        })

        # Update agent workspace with specific fields like readme generator
        update_agent_workspace(
            "terraform_readme_generator", {
                "generated_readme_components": llm_response.generated_readme_components,
                "readme_content": llm_response.readme_content,
                "table_of_contents": llm_response.table_of_contents,
                "sections_count": llm_response.sections_count,
                "word_count": llm_response.word_count
            }
        )
        
        readme_generator_logger.log_structured(
            level="INFO",
            message="Terraform README generation completed successfully",
            extra={
                "generation_duration_seconds": generation_duration,
                "completion_status": llm_response.completion_status,
                "sections_count": llm_response.sections_count,
                "word_count": llm_response.word_count,
                "warnings_count": len(llm_response.recoverable_warnings)
            }
        )
        
        return llm_response.workspace_updates
        
    except Exception as e:
        readme_generator_logger.log_structured(
            level="ERROR",
            message="Error during Terraform README generation",
            extra={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        
        return TerraformReadmeGenerationResponse(
            readme_content="# Error during generation",
            table_of_contents="# Error during generation",
            sections_count=0,
            word_count=0,
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
                "stage": "readme_generation",
                "agent": "readme_generator",
                "checkpoint_type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
