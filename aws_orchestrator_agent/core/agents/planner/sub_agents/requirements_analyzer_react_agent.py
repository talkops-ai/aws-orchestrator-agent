"""
Requirements Analyzer React Agent for Planner Sub-Supervisor.

This module implements the Requirements Analyzer as a React agent with tools:
- analyze_requirements_tool: Analyzes user requests for infrastructure requirements
- validate_requirements_tool: Validates extracted requirements for completeness
"""

import json
from typing import Dict, Any, List, Literal, Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger
from enum import Enum
from aws_orchestrator_agent.core.agents.planner.planner_supervisor_state import PlannerSupervisorState
from .requirement_analyser_prompts import (
    AWS_SERVICE_DISCOVERY_SYSTEM_PROMPT,
    AWS_SERVICE_DISCOVERY_HUMAN_PROMPT,
    TERRAFORM_RESOURCE_ATTRIBUTES_SYSTEM_PROMPT,
    TERRAFORM_RESOURCE_ATTRIBUTES_HUMAN_PROMPT,
    TERRAFORM_ATTRIBUTE_MAPPER_COORDINATOR_SYSTEM_PROMPT,
    TERRAFORM_ATTRIBUTE_MAPPER_COORDINATOR_HUMAN_PROMPT
)
from ..planner_utils import create_agent_completion_data

from aws_orchestrator_agent.utils.mcp_client import create_mcp_client

# Create logger
requirements_logger = AgentLogger("REQUIREMENTS_ANALYZER_REACT")

# Global variables for LLM and parsers
_model = None
_model_higher = None  # Higher-tier model for complex reasoning tasks
_model_react = None
_requirements_parser_prompt = None
_infra_requirements_parser = None
_service_discovery_parser = None
_service_discovery_system_prompt = None
_service_discovery_human_prompt = None
_service_discovery_prompt = None
_mcp_client = None
_terraform_attribute_mapping_parser = None
_terraform_resource_attributes_parser = None

# Global variable for shared planner state access
_shared_planner_state: Optional[PlannerSupervisorState] = None

# Define the output schema for structured extraction
class ServiceRequirement(BaseModel):
    """Service-specific requirements and specifications"""
    service_name: str = Field(description="Name of the AWS service")
    aws_service_type: str = Field(description="AWS service identifier (e.g., 's3', 'vpc', 'ec2')")
    business_requirements: Dict[str, str] = Field(description="Business needs specific to this service")
    technical_specifications: Dict[str, str] = Field(description="Technical specifications native to this service")

class InfrastructureRequirements(BaseModel):
    """Structured representation of AWS infrastructure requirements"""
    scope_classification: str = Field(description="One of: 'single_service', 'multi_service', 'full_application_stack'")
    deployment_context: str = Field(description="Context like development, production, compliance requirements")
    services: List[ServiceRequirement] = Field(description="List of service-specific requirements")

# Dependency types for service dependencies
class DependencyType(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    RECOMMENDED = "recommended"

# Service dependency specification
class ServiceDependency(BaseModel):
    """Service dependency as module variable"""
    service: str = Field(..., description="Dependent service name")
    variable: str = Field(..., description="Variable name for dependency")
    type: DependencyType = Field(..., description="Dependency type")

# Architecture pattern for services
class ArchitecturePattern(BaseModel):
    """Architecture pattern for the service"""
    pattern_name: str = Field(..., description="Pattern name")
    description: str = Field(..., description="Pattern description")
    best_practices: List[str] = Field(..., description="Best practices for this pattern")

# Well-Architected Framework alignment
class WellArchitectedAlignment(BaseModel):
    """Well-Architected Framework alignment for the service"""
    operational_excellence: List[str] = Field(default_factory=list)
    security: List[str] = Field(default_factory=list)
    reliability: List[str] = Field(default_factory=list)
    performance_efficiency: List[str] = Field(default_factory=list)
    cost_optimization: List[str] = Field(default_factory=list)
    sustainability: List[str] = Field(default_factory=list)

# Cost optimization recommendation
class CostOptimizationRecommendation(BaseModel):
    """Cost optimization recommendation for the service"""
    category: str = Field(..., description="Cost category")
    recommendation: str = Field(..., description="Specific recommendation")
    potential_savings: Optional[str] = Field(None, description="Potential cost savings")
    implementation_difficulty: Literal["low", "medium", "high"] = Field(..., description="Implementation difficulty")

# Individual service specification
class ServiceSpecification(BaseModel):
    """Complete individual service specification"""
    service_name: str = Field(..., description="AWS service name (e.g., 'S3', 'EKS', 'ElastiCache')")
    aws_service_type: str = Field(..., description="AWS service identifier (e.g., 's3', 'eks', 'elasticache')")
    terraform_resources: List[str] = Field(..., description="List of Terraform resources for production-grade deployment")
    dependencies: List[ServiceDependency] = Field(default_factory=list, description="Service dependencies as variables")
    
    # Architecture and patterns
    architecture_patterns: List[ArchitecturePattern] = Field(..., description="Architecture patterns for this service")
    overall_architecture_pattern: Optional[ArchitecturePattern] = Field(None, description="Overall architecture pattern if this is the primary service")
    
    # Best practices
    well_architected_alignment: WellArchitectedAlignment = Field(..., description="Well-Architected Framework alignment")
    cost_optimization_recommendations: List[CostOptimizationRecommendation] = Field(..., description="Cost optimization recommendations")
    
    # Additional metadata
    description: str = Field(..., description="Service description")
    production_features: List[str] = Field(..., description="Production features included")

# Main AWS service discovery output
class AWSServiceMapping(BaseModel):
    """AWS service discovery output - list of individual services"""
    services: List[ServiceSpecification] = Field(..., description="List of all services with complete specifications")

# Final resource attributes output
class AttributeType(str, Enum):
    """Terraform attribute types"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    COMPUTED = "computed"
    DEPRECATED = "deprecated"



class TerraformAttributeModuleDesign(BaseModel):
    """Module design for a Terraform resource"""
    recommended_arguments: List[str] = Field(..., description="Recommended arguments for the resource")
    recommended_outputs: List[str] = Field(..., description="Recommended outputs for the resource")

class TerraformAttribute(BaseModel):
    """Simplified Terraform attribute specification"""
    name: str = Field(..., description="Attribute name")
    type: str = Field(..., description="Terraform data type (string, number, bool, list, map, object, etc.)")
    required: bool = Field(..., description="Whether this attribute is mandatory")
    description: str = Field(..., description="Detailed attribute description and purpose")
    example_value: Optional[Any] = Field(None, description="Practical example value")
    

class TerraformResourceSpecification(BaseModel):
    """Enhanced Terraform resource specification with comprehensive attributes"""
    resource_name: str = Field(..., description="Terraform resource name (e.g., aws_s3_bucket)")
    provider: str = Field(..., description="Provider name (e.g., aws)")
    description: str = Field(..., description="Resource description and purpose")
    required_attributes: List[TerraformAttribute] = Field(..., description="Mandatory attributes")
    optional_attributes: List[TerraformAttribute] = Field(..., description="Optional attributes")
    computed_attributes: List[TerraformAttribute] = Field(..., description="Computed/read-only attributes")
    deprecated_attributes: List[TerraformAttribute] = Field(default_factory=list, description="Deprecated attributes")
    version_requirements: Optional[str] = Field(None, description="Provider version requirements")
    module_design: TerraformAttributeModuleDesign = Field(..., description="Module design for the attribute")

class TerraformServiceAttributeMapping(BaseModel):
    """Attribute mapping for a single AWS service"""
    service_name: str = Field(..., description="AWS service name")
    aws_service_type: str = Field(..., description="AWS service type identifier")
    description: str = Field(..., description="Service description and purpose")
    terraform_resources: List[TerraformResourceSpecification] = Field(..., description="Complete resource specifications")
    version_requirements: Optional[str] = Field(None, description="Provider version requirements")

class TerraformAttributeMapping(BaseModel):
    """Enhanced attribute mapping supporting multiple services"""
    services: List[TerraformServiceAttributeMapping] = Field(..., description="List of service attribute mappings")


def _initialize_requirements_tools(config: Config):
    """Initialize LLM and parsers for requirements tools."""
    global _model, _model_higher, _model_react, _infra_requirements_parser, _requirements_parser_prompt, _service_discovery_parser, _service_discovery_prompt, _service_discovery_system_prompt, _service_discovery_human_prompt, _mcp_client, _terraform_attribute_mapping_parser, _terraform_resource_attributes_parser
    
    if _model is None:
        # Initialize standard LLM for simple tasks
        llm_config = config.get_llm_config()
        _model = LLMProvider.create_llm(
            provider=llm_config['provider'],
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        # Initialize higher-tier LLM for complex reasoning tasks
        llm_higher_config = config.get_llm_higher_config()
        _model_higher = LLMProvider.create_llm(
            provider=llm_higher_config['provider'],
            model=llm_higher_config['model'],
            temperature=llm_higher_config['temperature'],
            max_tokens=llm_higher_config['max_tokens']
        )
        
        llm_react_config = config.get_llm_react_config()
        _model_react = LLMProvider.create_llm(
            provider=llm_react_config['provider'],
            model=llm_react_config['model'],
            temperature=llm_react_config['temperature'],
            max_tokens=llm_react_config['max_tokens']
        )
        
        _mcp_client = create_mcp_client(host=config.TERRAFORM_MCP_SERVER_HOST, port=config.TERRAFORM_MCP_SERVER_PORT, transport=config.TERRAFORM_MCP_SERVER_TRANSPORT)
        
        _infra_requirements_parser = JsonOutputParser(pydantic_object=InfrastructureRequirements)
        _service_discovery_parser = JsonOutputParser(pydantic_object=AWSServiceMapping)
        _terraform_attribute_mapping_parser = JsonOutputParser(pydantic_object=TerraformAttributeMapping)
        _terraform_resource_attributes_parser = JsonOutputParser(pydantic_object=TerraformResourceSpecification)
        _service_discovery_system_prompt = AWS_SERVICE_DISCOVERY_SYSTEM_PROMPT
        _service_discovery_human_prompt = AWS_SERVICE_DISCOVERY_HUMAN_PROMPT
        _service_discovery_prompt = ChatPromptTemplate.from_messages([
            ("system", _service_discovery_system_prompt),
            ("human", _service_discovery_human_prompt)
        ])
        
    # ChatPromptTemplate for Requirements Parser Tool
        _requirements_parser_prompt = ChatPromptTemplate.from_messages([("system", """
You are an expert AWS Infrastructure Requirements Analyst specialized in parsing natural language queries into structured technical specifications for Terraform module development.

Your primary responsibilities:
1. Extract infrastructure requirements from user queries with precision
2. Identify ONLY the AWS services explicitly mentioned in the user query
3. Determine the scope and complexity of the infrastructure request
4. Map business requirements to service-specific technical specifications
5. Classify the deployment context and constraints

ANALYSIS FRAMEWORK:

**Step 1: Service Identification**
- IDENTIFY ONLY EXPLICITLY MENTIONED SERVICES: Extract AWS services that are directly mentioned in the user query
- DO NOT INFER SECONDARY SERVICES: Do not add supporting services like IAM, KMS, CloudWatch unless explicitly mentioned
- FOCUS ON PRIMARY SERVICES: Only include services the user specifically asked for

**Step 2: Scope Classification**
- SINGLE_SERVICE: One main AWS service (e.g., "S3 bucket module")  
- MULTI_SERVICE: Multiple related services (e.g., "S3 and VPC module")
- FULL_APPLICATION_STACK: Complete application infrastructure (e.g., "3-tier web application")

**Step 3: Service-Specific Requirements Mapping**
- BUSINESS REQUIREMENTS: What the user wants to achieve with each specific service
- TECHNICAL SPECIFICATIONS: Service-native capabilities and configurations (no cross-service dependencies)
- DEPLOYMENT CONTEXT: Environment type, compliance needs, security requirements

**Step 4: Service-Native Focus**
- Focus on capabilities native to each service
- Do not include cross-service integrations or dependencies
- Keep technical specifications within the scope of the specific service

**Output Format:**
Provide your analysis in the structured JSON format specified by the schema.
Be thorough but concise. Focus only on explicitly mentioned services and their native capabilities.

**CRITICAL: Return ONLY the JSON object without any markdown formatting, code blocks, or additional text.**
- DO NOT wrap the response in ```json or ``` blocks
- DO NOT add any explanatory text before or after the JSON
- Return ONLY the raw JSON object that matches the schema
"""),
    
    ("human", """
Analyze the following infrastructure request and extract structured requirements:

USER QUERY: {user_query}

Please provide a comprehensive analysis following the framework above. Consider:
- What AWS services are explicitly mentioned?
- What's the scope and complexity?
- What business goals are implied for each service?
- What service-native technical specifications can be inferred?
- What deployment context clues are present?

Your final output MUST be a JSON object matching this Pydantic model: `InfrastructureRequirements`:

class ServiceRequirement(BaseModel):
    service_name: str = Field(description="Name of the AWS service")
    aws_service_type: str = Field(description="AWS service identifier (e.g., 's3', 'vpc', 'ec2')")
    business_requirements: Dict[str, str] = Field(description="Business needs specific to this service")
    technical_specifications: Dict[str, str] = Field(description="Technical specifications native to this service")

class InfrastructureRequirements(BaseModel):
    scope_classification: str = Field(description="One of: 'single_service', 'multi_service', 'full_application_stack'")
    deployment_context: str = Field(description="Context like development, production, compliance requirements")
    services: List[ServiceRequirement] = Field(description="List of service-specific requirements")

**IMPORTANT GUIDELINES:**
- Only include services explicitly mentioned in the user query
- Do not add secondary or supporting services
- Keep technical specifications native to each service
- Do not include cross-service dependencies or integrations

""")
])

@tool
async def infra_requirements_parser_tool(user_query: str) -> InfrastructureRequirements:
    """
    Extracts infrastructure requirements from natural language queries.
    
    This tool:
    - Identifies primary and secondary AWS services mentioned
    - Determines scope (single service, multi-service, full application stack)  
    - Maps business requirements to technical specifications
    - Provides structured output for downstream processing
    - UPDATES SHARED STATE when analysis completes
    
    Args:
        user_query: Natural language description of infrastructure needs
        
    Returns:
        InfrastructureRequirements: Structured analysis of the request
    """
    try:
        # ACCESS SHARED STATE
        global _shared_planner_state
        if _shared_planner_state is None:
            raise ValueError("Shared planner state not initialized")
        
        requirements_logger.log_structured(
            level="INFO",
            message="Starting async Infra requirements parser tool with state access",
            extra={
                "user_request": user_query[:100] + "..." if len(user_query) > 100 else user_query,
                "current_phase": getattr(_shared_planner_state.planning_workflow_state, 'current_phase', 'unknown'),
                "requirements_complete": getattr(_shared_planner_state.planning_workflow_state, 'requirements_complete', False)
            }
        )
        
        if _model is None:
            raise ValueError("Requirements tools not initialized. Call _initialize_requirements_tools first.")
        
        formatted_prompt = _requirements_parser_prompt.format(user_query=user_query)
        llm_response = await _model.ainvoke(formatted_prompt)
        
        if isinstance(llm_response, AIMessage):
            response = llm_response.content
        else:
            response = llm_response
            
        requirements_logger.log_structured(
            level="DEBUG",
            message="Infra requirements parser tool response received",
            extra={
                "response_type": type(llm_response).__name__,
                "has_content": hasattr(llm_response, 'content'),
                "content_length": len(response) if response else 0
            }
        )
        
        # Clean the response content to remove markdown code blocks if present
        content = response.strip()
        
        parsed_response = _infra_requirements_parser.parse(content)
        
        # Debug the parsed response
        requirements_logger.log_structured(
            level="DEBUG",
            message="Parsed response before state update",
            extra={
                "parsed_response_type": type(parsed_response).__name__,
                "parsed_response_keys": list(parsed_response.keys()) if isinstance(parsed_response, dict) else "not_dict",
                "has_scope_classification": hasattr(parsed_response, 'scope_classification') if not isinstance(parsed_response, dict) else 'is_dict',
                "scope_classification": parsed_response.get('scope_classification', 'not_found') if isinstance(parsed_response, dict) else getattr(parsed_response, 'scope_classification', 'not_found')
            }
        )
        requirements_logger.log_structured(
            level="INFO",
            message="Infra requirements parser tool completed and state updated",
            extra={
                "parsed_services_count": len(parsed_response.services) if hasattr(parsed_response, 'services') else 0,
                "scope_classification": getattr(parsed_response, 'scope_classification', 'unknown'),
                "state_updated": True
            }
        )

        _shared_planner_state.requirements_data.analysis_results = parsed_response
        _shared_planner_state.requirements_data.analysis_complete = True
        return parsed_response

    except Exception as e:
        requirements_logger.log_structured(
            level="ERROR",
            message=f"Failed to parse Infra requirements: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Infra requirements parser tool failed: {str(e)}"})

@tool
async def aws_service_discovery_tool(requirements_analysis: str) -> AWSServiceMapping:
    """
    Service-focused AWS service discovery tool for Terraform module generation.
    
    This tool provides:
    - Individual service specifications with production-grade Terraform resources
    - Service dependencies mapped as module variables (not resources)
    - Architecture patterns and best practices for each service
    - Well-Architected Framework alignment per service
    - Cost optimization recommendations per service
    - Production features and security configurations per service
    
    Args:
        requirements_analysis: Structured requirements from the Requirements Parser Tool
        
    Returns:
        AWSServiceMapping: Service-focused specifications suitable for Terraform module generation
        
    Raises:
        ValidationError: If output doesn't meet production quality standards
    """
    try:
        # ACCESS SHARED STATE
        global _shared_planner_state
        if _shared_planner_state is None:
            raise ValueError("Shared planner state not initialized")
        
        requirements_logger.log_structured(
            level="INFO",
            message="Starting async AWS service discovery with state access",
            extra={
                "requirements_analysis_length": len(requirements_analysis),
                "current_phase": getattr(_shared_planner_state.planning_workflow_state, 'current_phase', 'unknown'),
                "requirements_complete": getattr(_shared_planner_state.planning_workflow_state, 'requirements_complete', False)
            }
        )
        
        if _model is None or _model_higher is None:
            raise ValueError("Requirements tools not initialized. Call _initialize_requirements_tools first.")
        
        # Debug the input and prompt template
        requirements_logger.log_structured(
            level="DEBUG",
            message="Debugging prompt formatting",
            extra={
                "requirements_analysis_type": type(requirements_analysis).__name__,
                "requirements_analysis_preview": requirements_analysis[:200] + "..." if len(requirements_analysis) > 200 else requirements_analysis,
                "prompt_template_type": type(_service_discovery_prompt).__name__
            }
        )
        
        # Use a different approach to avoid format() issues with JSON
        try:
            # Try direct format first
            formatted_prompt = _service_discovery_prompt.format(requirements_input=requirements_analysis)
        except (KeyError, ValueError) as format_error:
            # If format fails, manually construct the prompt
            requirements_logger.log_structured(
                level="WARNING",
                message="Format method failed, using manual prompt construction",
                extra={"format_error": str(format_error)}
            )
            
            # Manually construct the prompt by replacing the placeholder
            system_prompt = _service_discovery_system_prompt
            human_prompt = _service_discovery_human_prompt.replace("{requirements_input}", requirements_analysis)
            
            # Create messages directly instead of using ChatPromptTemplate
            formatted_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
        
        # Format the prompt before invoking the LLM
        if 'formatted_messages' not in locals():
            # This means we used the try/except path and already have formatted_messages
            formatted_messages = formatted_prompt
        llm_response = await _model_higher.ainvoke(formatted_messages)
        
        if isinstance(llm_response, AIMessage):
            response = llm_response.content
        else:
            response = llm_response
            
        # Log the raw response for debugging
        requirements_logger.log_structured(
            level="DEBUG",
            message="Raw LLM response received",
            extra={
                "response_type": type(response).__name__,
                "response_length": len(response) if response else 0,
                "response_preview": response[:200] + "..." if response and len(response) > 200 else response
            }
        )
        
        # Clean the response content to remove markdown code blocks if present
        content = response.strip()
        if content.startswith('```json'):
            content = content[7:]  # Remove ```json
        if content.endswith('```'):
            content = content[:-3]  # Remove ```
        content = content.strip()
        
        # Parse the response using the service discovery parser
        parsed_response = _service_discovery_parser.parse(content)

        # CRITICAL: Log successful completion with detailed information
        requirements_logger.log_structured(
            level="INFO",
            message="AWS service discovery completed successfully with valid JSON mapping",
            extra={
                "completion_ready": True,
                "services_discovered": len(parsed_response.get('primary_services', [])),
                "foundation_services_count": len(parsed_response.get('foundation_services', [])),
                "terraform_resources_count": len(parsed_response.get('terraform_resources', [])),
                "deployment_sequence_count": len(parsed_response.get('deployment_sequence', [])),
                "workflow_phase": "requirements_analysis",
                "has_required_fields": True,
                "json_structure_valid": True
            }
        )
        
        _shared_planner_state.requirements_data.aws_service_mapping = parsed_response
        _shared_planner_state.requirements_data.aws_service_mapping_complete = True
        return parsed_response
        
    except Exception as e:
        requirements_logger.log_structured(
            level="ERROR",
            message=f"Async requirements validation failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Requirements validation failed: {str(e)}"})

@tool
async def get_final_resource_attributes_tool(aws_service_mapping: str) -> TerraformAttributeMapping: 
    """
    Get the final resource attributes for the service via Coordinator React agent.
    
    This tool:
    - Parses AWS service mapping to identify Terraform resources
    - Uses Coordinator React agent to orchestrate individual resource analysis
    - Calls get_terraform_resource_attributes_tool for each individual resource
    - Aggregates individual resource results into comprehensive service-level mappings
    - Provides production-grade attribute recommendations and best practices
    - Returns comprehensive attribute mapping with detailed specifications

    Args:
        aws_service_mapping: Structured AWS Service Mapping from the AWS Service Discovery Tool
        
    Returns:
        TerraformAttributeMapping: Complete attribute specifications for all resources including:
            - services: List of service attribute mappings
            - Each service includes terraform_resources with categorized attributes
    """
    try:
        # Log the input received
        requirements_logger.log_structured(
            level="INFO",
            message="=== GET_FINAL_RESOURCE_ATTRIBUTES_TOOL STARTED (COORDINATOR AGENT) ===",
            extra={
                "input_type": type(aws_service_mapping).__name__,
                "input_length": len(aws_service_mapping) if aws_service_mapping else 0,
                "input_preview": aws_service_mapping[:200] + "..." if aws_service_mapping and len(aws_service_mapping) > 200 else aws_service_mapping,
                "tool_name": "get_final_resource_attributes_tool",
                "approach": "coordinator_with_individual_tool"
            }
        )
        global _model, _terraform_attribute_mapping_parser
        
        if _model is None:
            raise ValueError("Requirements tools not initialized. Call _initialize_requirements_tools first.")
    
        
        # Create React agent WITH the individual resource tool
        # Use the system prompt as the agent's prompt (simple string)
        attribute_agent = create_react_agent(
            model=_model,
            tools=[get_terraform_resource_attributes_tool],  # Add the individual resource tool
            name="attribute_coordinator_agent",
            prompt=TERRAFORM_ATTRIBUTE_MAPPER_COORDINATOR_SYSTEM_PROMPT
        )
        
        requirements_logger.log_structured(
            level="DEBUG",
            message="React agent coordinator created successfully, invoking with AWS service mapping",
            extra={
                "agent_type": type(attribute_agent).__name__,
                "tools_count": 1,
                "has_response_format": True,
                "approach": "coordinator_with_individual_tool"
            }
        )
        
        # Format the human prompt with the AWS service mapping
        formatted_human_prompt = TERRAFORM_ATTRIBUTE_MAPPER_COORDINATOR_HUMAN_PROMPT.format(aws_service_mapping=aws_service_mapping)
        
        # Invoke the React agent with proper message dict
        result = await attribute_agent.ainvoke({
            "messages": [
                {"role": "user", "content": formatted_human_prompt}
            ]
        })
        
        requirements_logger.log_structured(
            level="INFO",
            message="React agent coordinator completed successfully",
            extra={
                "result_type": type(result).__name__,
                "result_has_content": hasattr(result, 'content') if hasattr(result, '__dict__') else False,
                "result_length": len(str(result)) if result else 0,
                "approach": "coordinator_with_individual_tool"
            }
        )
        
        # Extract the result
        response = result['messages'][-1]
        if isinstance(response, AIMessage):
            content = response.content
            content_str = str(content) if not isinstance(content, str) else content
        else:
            content_str = str(response) if not isinstance(response, str) else response
        
        # Parse the result using the TerraformAttributeMapping parser
        parsed_result = _terraform_attribute_mapping_parser.parse(content_str)
        
        # Add unified completion tracking to the parsed result
        completion_data = create_agent_completion_data(
            agent_name="requirements_analyzer",
            task_type="terraform_attribute_mapping",
            data_type="terraform_attribute_mapping",
            status="completed"
        )
        
        if isinstance(parsed_result, dict):
            parsed_result['agent_completion'] = completion_data
        else:
            # If it's a Pydantic model, convert to dict and add flag
            parsed_result_dict = parsed_result.dict() if hasattr(parsed_result, 'dict') else parsed_result
            parsed_result_dict['agent_completion'] = completion_data
            parsed_result = parsed_result_dict
        
        # Update shared state
        _shared_planner_state.requirements_data.terraform_attribute_mapping = parsed_result
        _shared_planner_state.requirements_data.terraform_attribute_mapping_complete = True
        
        # CRITICAL: Mark requirements analysis as complete in the workflow state
        # This is what the supervisor checks to determine if requirements are complete
        _shared_planner_state.planning_workflow_state.requirements_complete = True
        
        # Log successful completion
        requirements_logger.log_structured(
            level="INFO",
            message="Terraform attribute mapping completed successfully (Coordinator Agent)",
            extra={
                "completion_ready": True,
                "services_count": len(parsed_result.get('services', [])),
                "total_resources": sum(len(service.get('terraform_resources', [])) for service in parsed_result.get('services', [])),
                "workflow_phase": "attribute_mapping",
                "has_required_fields": True,
                "json_structure_valid": True,
                "approach": "coordinator_with_individual_tool"
            }
        )
        
        return parsed_result
            
    except Exception as e:
        requirements_logger.log_structured(
            level="ERROR",
            message=f"Terraform attribute mapping failed (Coordinator Agent): {e}",
            extra={"error": str(e), "error_type": type(e).__name__, "approach": "coordinator_with_individual_tool"}
        )
        return json.dumps({"error": f"Terraform attribute mapping failed: {str(e)}"})

@tool
async def get_terraform_resource_attributes_tool(terraform_resource_name: str) -> TerraformResourceSpecification:
    """
    Get comprehensive attribute specifications for a single Terraform resource.
    
    This tool:
    - Takes a single Terraform resource name as input
    - Analyzes the resource in detail to generate comprehensive attribute specifications
    - Categorizes attributes as required/optional/computed/deprecated with argument/reference classification
    - Provides production-grade attribute specifications with detailed descriptions, types, and examples
    - Returns complete TerraformResourceSpecification for the individual resource

    Args:
        terraform_resource_name: Name of the Terraform resource (e.g., "aws_s3_bucket")
        
    Returns:
        TerraformResourceSpecification: Complete attribute specifications for the resource including:
            - resource_name: The Terraform resource name
            - provider: Provider name (e.g., aws)
            - description: Resource description and purpose
            - required_attributes: List of mandatory attributes
            - optional_attributes: List of optional attributes
            - computed_attributes: List of computed/read-only attributes
            - deprecated_attributes: List of deprecated attributes
            - version_requirements: Provider version requirements if any

    """
    try:
        # Log the input received
        requirements_logger.log_structured(
            level="INFO",
            message="=== GET_TERRAFORM_RESOURCE_ATTRIBUTES_TOOL STARTED ===",
            extra={
                "input_type": type(terraform_resource_name).__name__,
                "input_value": terraform_resource_name,
                "tool_name": "get_terraform_resource_attributes_tool",
                "approach": "single_resource_analysis"
            }
        )
        global _model, _terraform_resource_attributes_parser, _model_higher
        
        if _model is None or _model_higher is None:
            raise ValueError("Requirements tools not initialized. Call _initialize_requirements_tools first.")
        
        # Create the prompt template for individual resource analysis
        resource_prompt_template = ChatPromptTemplate.from_messages([
            ("system", TERRAFORM_RESOURCE_ATTRIBUTES_SYSTEM_PROMPT),
            ("human", TERRAFORM_RESOURCE_ATTRIBUTES_HUMAN_PROMPT)
        ])
        
        requirements_logger.log_structured(
            level="DEBUG",
            message="Individual resource prompt template created successfully",
            extra={
                "prompt_type": type(resource_prompt_template).__name__,
                "has_system_message": True,
                "has_human_message": True,
                "resource_name": terraform_resource_name
            }
        )
        
        # Format the prompt with the resource name
        formatted_prompt = resource_prompt_template.format(terraform_resource_name=terraform_resource_name)
        
        # Get LLM response
        llm_response = await _model_higher.ainvoke(formatted_prompt)
        
        if isinstance(llm_response, AIMessage):
            content = llm_response.content
            content_str = str(content) if not isinstance(content, str) else content
        else:
            content_str = str(llm_response) if not isinstance(llm_response, str) else llm_response
        
        requirements_logger.log_structured(
            level="DEBUG",
            message="LLM response received for individual resource",
            extra={
                "response_type": type(llm_response).__name__,
                "content_length": len(content_str) if content_str else 0,
                "resource_name": terraform_resource_name
            }
        )
        
        # Log the raw content for debugging
        # requirements_logger.log_structured(
        #     level="DEBUG",
        #     message="Raw LLM response content",
        #     extra={
        #         "raw_content": content_str[:1000] if content_str else "No content",  # First 1000 chars
        #         "resource_name": terraform_resource_name
        #     }
        # )
        
        # Parse the result using the TerraformResourceSpecification parser
        parsed_result = _terraform_resource_attributes_parser.parse(content_str)
        
        # Log successful completion
        requirements_logger.log_structured(
            level="INFO",
            message="Individual Terraform resource attributes completed successfully",
            extra={
                "completion_ready": True,
                "resource_name": terraform_resource_name,
                "required_attributes_count": len(parsed_result.get('required_attributes', [])),
                "optional_attributes_count": len(parsed_result.get('optional_attributes', [])),
                "computed_attributes_count": len(parsed_result.get('computed_attributes', [])),
                "deprecated_attributes_count": len(parsed_result.get('deprecated_attributes', [])),
                "workflow_phase": "individual_resource_analysis",
                "has_required_fields": True,
                "json_structure_valid": True,
                "approach": "single_resource_analysis"
            }
        )
        
        return parsed_result
            
    except Exception as e:
        requirements_logger.log_structured(
            level="ERROR",
            message=f"Individual Terraform resource attributes failed: {e}",
            extra={
                "error": str(e), 
                "error_type": type(e).__name__, 
                "resource_name": terraform_resource_name,
                "approach": "single_resource_analysis"
            }
        )
        return json.dumps({"error": f"Individual Terraform resource attributes failed: {str(e)}"})

def create_requirements_analyzer_react_agent(state: PlannerSupervisorState, config: Config):
    """
    Create a React agent for requirements analysis.
    
    Args:
        state: Shared PlannerSupervisorState instance
        config: Configuration instance
        
    Returns:
        React agent for requirements analysis
    """
    try:
        requirements_logger.log_structured(
            level="INFO",
            message="=== CREATING REQUIREMENTS ANALYZER REACT AGENT ===",
            extra={"config_type": type(config).__name__}
        )
        
        # STORE STATE PARAMETER GLOBALLY FOR TOOLS TO ACCESS
        global _shared_planner_state
        _shared_planner_state = state
        
        requirements_logger.log_structured(
            level="DEBUG",
            message="State parameter stored globally",
            extra={
                "state_type": type(state).__name__,
                "has_planning_workflow_state": hasattr(state, 'planning_workflow_state'),
                "current_phase": getattr(state.planning_workflow_state, 'current_phase', 'unknown'),
                "user_request": getattr(state, 'user_request', ''),
                "task_description": getattr(state, 'task_description', ''),
                "session_id": getattr(state, 'session_id', None),
                "task_id": getattr(state, 'task_id', None)
            }
        )
        
        # Initialize tools (now with access to shared state)
        requirements_logger.log_structured(
            level="DEBUG",
            message="Initializing requirements tools with state access",
            extra={}
        )
        
        _initialize_requirements_tools(config)
        
        # Get LLM from config
        # Use the already initialized _model instead of creating a new LLM instance
        requirements_logger.log_structured(
            level="DEBUG",
            message="Using initialized LLM for requirements analyzer",
            extra={
                "llm_provider": "reused_from_global",
                "llm_model": "reused_from_global"
            }
        )
        
        llm = _model  # Reuse the globally initialized model
        
        # Create React agent with async tools
        requirements_logger.log_structured(
            level="DEBUG",
            message="Creating React agent with async tools",
            extra={
                "tools_count": 3,
                "tool_names": ["infra_requirements_parser_tool", "aws_service_discovery_tool", "get_final_resource_attributes_tool"]
            }
        )
        
        requirements_analyzer = create_react_agent(
            model=llm,
            tools=[infra_requirements_parser_tool, aws_service_discovery_tool, get_final_resource_attributes_tool],
            name="requirements_analyzer",
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
You are an expert AWS Infrastructure Requirements Analyst specializing in comprehensive infrastructure assessment and service mapping.

## CORE MISSION
Extract comprehensive infrastructure requirements from user requests and generate complete AWS service mappings for infrastructure deployment.

## CRITICAL OUTPUT REQUIREMENT
After completing all three tools successfully, return ONLY the JSON output from get_final_resource_attributes_tool. Do not add any explanations, summaries, or additional text. Return the raw JSON object exactly as it was returned by the tool.

## WORKFLOW (MANDATORY SEQUENCE)
1. **ANALYZE**: Use infra_requirements_parser_tool with the user's request along with the state of the planning workflow
2. **MAP SERVICES**: Use aws_service_discovery_tool with parsed requirements  
3. **GET ATTRIBUTES**: Use get_final_resource_attributes_tool with AWS service mapping to get complete Terraform resource attributes
4. **VALIDATE**: Ensure all three tools return complete, valid outputs
5. **RESPOND**: Return the Terraform attribute mapping JSON directly from get_final_resource_attributes_tool

## TOOL EXECUTION RULES
- infra_requirements_parser_tool: Extracts business/technical requirements from ANY input
- aws_service_discovery_tool: Generates AWS service mapping JSON (services, terraform_resources, dependencies, etc.)
- get_final_resource_attributes_tool: Gets complete Terraform resource attribute specifications via MCP server
- ALL THREE tools are MANDATORY regardless of input quality
- Execute tools sequentially, never skip any of them
- If any tool fails, report the specific failure and STOP

## INPUT HANDLING STRATEGY
**Complete Requests**: Use exact user input with infra_requirements_parser_tool
**Incomplete Requests**: Still use infra_requirements_parser_tool with available information
**Empty/Unclear Requests**: Use "AWS infrastructure deployment" as input
**Always proceed with tool execution first - never ask for clarification before using tools**

## SUCCESS VALIDATION CHECKLIST
✓ infra_requirements_parser_tool completed successfully
✓ aws_service_discovery_tool completed successfully  
✓ get_final_resource_attributes_tool completed successfully
✓ AWS service mapping JSON contains: services, terraform_resources, dependencies, etc.
✓ Terraform attribute mapping JSON contains: terraform_resources, mapping_summary, total_attributes, etc.
✓ No malformed or incomplete JSON output

[HANDLING INCOMPLETE REQUESTS]
- For empty requests: Use infra_requirements_parser_tool with "AWS infrastructure deployment"
- For unclear requests: Use infra_requirements_parser_tool with the available information
- For partial requests: Use infra_requirements_parser_tool to extract what you can
- Always use the tools before making any assumptions

## ERROR HANDLING
- Tool failure → Report specific error and terminate analysis
- Invalid JSON → Report JSON validation failure and terminate  
- Missing required fields → Report incomplete mapping and terminate
- Only proceed to final response when ALL validations pass

## IMPORTANT NOTES
- Make reasonable infrastructure assumptions when information is limited
- Use AWS best practices for service selection and architecture patterns
- Focus on production-ready, scalable solutions
- Maintain security and compliance considerations throughout analysis

## FINAL OUTPUT REQUIREMENT
After successfully completing all three tools, return ONLY the JSON output from get_final_resource_attributes_tool. Do not add any explanations, summaries, or additional text. Return the raw JSON object exactly as it was returned by the tool.

CRITICAL: Return ONLY the JSON object, no additional text or formatting.
        """),
                MessagesPlaceholder(variable_name="messages")
            ])
        )
        
        # Log the actual model configuration being used
        requirements_logger.log_structured(
            level="DEBUG",
            message="Global model configuration details",
            extra={
                "global_model_type": type(_model).__name__,
                "global_model_repr": str(_model)
            }
        )
        
        requirements_logger.log_structured(
            level="INFO",
            message="=== REQUIREMENTS ANALYZER REACT AGENT CREATED SUCCESSFULLY ===",
            extra={
                "agent_type": type(requirements_analyzer).__name__,
                "global_model_type": type(_model).__name__,
                "global_model_repr": str(_model),
                "tools_count": 3,
                "enhanced_prompt": True,
                "json_output_required": True,
                "required_workflow": "infra_requirements_parser_tool -> aws_service_discovery_tool -> get_final_resource_attributes_tool -> JSON output"
            }
        )
        
        return requirements_analyzer
        
    except Exception as e:
        requirements_logger.log_structured(
            level="ERROR",
            message="=== FAILED TO CREATE REQUIREMENTS ANALYZER REACT AGENT ===",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "config_type": type(config).__name__ if config else "None"
            }
        )
        raise
