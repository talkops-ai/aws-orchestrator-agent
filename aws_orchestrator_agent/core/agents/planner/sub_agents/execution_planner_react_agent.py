"""
Execution Planner React Agent for Planner Sub-Supervisor.

This module implements the Execution Planner as a React agent with tools:
- create_execution_plan_tool: Creates execution plans based on requirements and dependencies
- assess_risks_tool: Assesses risks associated with execution plans
- calculate_complexity_score_tool: Calculates complexity scores
- validate_execution_plan_tool: Validates execution plans for feasibility
"""

import json
import json_repair
from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re
from langgraph.prebuilt import create_react_agent

from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger
from aws_orchestrator_agent.core.agents.planner.planner_supervisor_state import PlannerSupervisorState
from .execution_planner_prompt import (
    TF_MODULE_STRUCTURE_PLAN_SYSTEM_PROMPT,
    TF_MODULE_STRUCTURE_PLAN_USER_PROMPT,
    TF_MODULE_REACT_AGENT_SYSTEM_PROMPT,
    TF_MODULE_REACT_AGENT_USER_PROMPT,
    TF_CONFIGURATION_OPTIMIZER_SYSTEM_PROMPT,
    TF_CONFIGURATION_OPTIMIZER_USER_PROMPT,
    TF_STATE_MGMT_SYSTEM_PROMPT,
    TF_STATE_MGMT_USER_PROMPT,
    TF_EXECUTION_PLANNER_SYSTEM_PROMPT,
    TF_EXECUTION_PLANNER_USER_PROMPT
)

from ..planner_utils import create_agent_completion_data

# Create logger
execution_logger = AgentLogger("EXECUTION_PLANNER_REACT")

# Global variables for LLM and parsers
_model = None
_model_higher = None
_module_structure_plan_parser = None
_module_react_agent_parser = None
_configuration_optimizer_parser = None
_state_mgmt_parser = None
_execution_planner_parser = None

# Global variables for execution tracking
_execution_sequence = {
    "module_structure_complete": False,
    "configuration_optimization_complete": False,
    "state_management_complete": False,
    "execution_plan_complete": False
}

_shared_planner_state: Optional[PlannerSupervisorState] = None

# Output Schema for Planning Response
class TerraformFileRecommendation(BaseModel):
    """Recommendation for a specific Terraform file in the module"""
    filename: str = Field(..., description="Name of the Terraform file (e.g., 'main.tf', 'variables.tf')")
    required: bool = Field(..., description="Whether this file is required for the module")
    purpose: str = Field(..., description="Explanation of why this file is needed")
    content_description: str = Field(..., description="What content should be included in this file")

class VariableDefinitionPlan(BaseModel):
    """Planned structure for a variable definition"""
    name: str = Field(..., description="Variable name")
    type: str = Field(..., description="Terraform variable type")
    description: str = Field(..., description="Variable description")
    default_value: Optional[Any] = Field(None, description="Default value if applicable")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules to apply")
    sensitive: bool = Field(False, description="Whether variable is sensitive")
    justification: str = Field(..., description="Why this variable is needed")

class OutputDefinitionPlan(BaseModel):
    """Planned structure for an output definition"""
    name: str = Field(..., description="Output name")
    description: str = Field(..., description="Output description")
    value_expression: str = Field(..., description="Terraform expression for the output value")
    sensitive: bool = Field(False, description="Whether output contains sensitive data")
    justification: str = Field(..., description="Why this output is needed")

class ReusabilityGuidance(BaseModel):
    """Guidance on making the module reusable and composable"""
    naming_conventions: List[str] = Field(default_factory=list, description="Recommended naming patterns")
    tagging_strategy: List[str] = Field(default_factory=list, description="Recommended tagging approaches")
    composability_hints: List[str] = Field(default_factory=list, description="How this module can compose with others")
    best_practices: List[str] = Field(default_factory=list, description="Additional best practices to follow")

class ModuleStructurePlanResponse(BaseModel):
    """Complete planning response for a Terraform module structure"""
    service_name: str = Field(..., description="AWS service this module targets")
    recommended_files: List[TerraformFileRecommendation] = Field(
        ..., 
        description="List of recommended Terraform files for the module"
    )
    variable_definitions: List[VariableDefinitionPlan] = Field(
        default_factory=list, 
        description="Planned variable definitions with validation"
    )
    output_definitions: List[OutputDefinitionPlan] = Field(
        default_factory=list, 
        description="Planned output definitions"
    )
    security_considerations: List[str] = Field(
        default_factory=list, 
        description="Security best practices incorporated into the plan"
    )
    reusability_guidance: ReusabilityGuidance = Field(
        ..., 
        description="Guidance on making the module reusable and composable"
    )
    implementation_notes: List[str] = Field(
        default_factory=list, 
        description="Additional notes for implementation teams"
        )

class ModuleStructurePlanResponseList(BaseModel):
    """Complete planning response for a Terraform module React agent structure"""
    module_structure_plans: List[ModuleStructurePlanResponse] = Field(
        ..., 
        description="List of planning details for the module"
    )

# Output Schema - Optimization Recommendations
class CostOptimization(BaseModel):
    """Cost optimization recommendations"""
    resource_name: str = Field(..., description="Resource being optimized")
    current_configuration: Any = Field(..., description="Current configuration")
    optimized_configuration: Any = Field(..., description="Cost-optimized configuration")
    estimated_savings: Optional[str] = Field(None, description="Estimated monthly savings")
    justification: str = Field(..., description="Why this optimization saves costs")

class PerformanceOptimization(BaseModel):
    """Performance optimization recommendations"""
    resource_name: str = Field(..., description="Resource being optimized")
    current_configuration: Any = Field(..., description="Current configuration")
    optimized_configuration: Any = Field(..., description="Performance-optimized configuration")
    performance_impact: str = Field(..., description="Expected performance improvement")
    justification: str = Field(..., description="Why this optimization improves performance")

class SecurityOptimization(BaseModel):
    """Security optimization recommendations"""
    resource_name: str = Field(..., description="Resource being optimized")
    security_issue: str = Field(..., description="Security concern identified")
    current_configuration: Any = Field(..., description="Current configuration")
    secure_configuration: Any = Field(..., description="Security-hardened configuration")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    justification: str = Field(..., description="Why this change improves security")

class SyntaxValidation(BaseModel):
    """Terraform syntax and structure validation results"""
    file_name: str = Field(..., description="File being validated")
    validation_status: str = Field(..., description="Valid, Invalid, or Warning")
    issues_found: List[str] = Field(default_factory=list, description="Syntax or structural issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations to fix issues")

class NamingConvention(BaseModel):
    """Naming convention recommendations"""
    resource_type: str = Field(..., description="Type of resource (variable, output, resource)")
    current_name: str = Field(..., description="Current name")
    recommended_name: str = Field(..., description="Name following conventions")
    convention_rule: str = Field(..., description="Convention rule applied")

class TaggingStrategy(BaseModel):
    """Tagging strategy recommendations"""
    resource_name: str = Field(..., description="Resource to be tagged")
    required_tags: Dict[str, str] = Field(default_factory=dict, description="Required tags based on organization standards")
    optional_tags: Dict[str, str] = Field(default_factory=dict, description="Recommended optional tags")
    tagging_justification: str = Field(default="", description="Why these tags are recommended")

class ConfigurationOptimizerResponse(BaseModel):
    """Complete optimization response"""
    service_name: str = Field(..., description="AWS service being optimized")
    cost_optimizations: List[CostOptimization] = Field(default_factory=list, description="Cost optimization recommendations")
    performance_optimizations: List[PerformanceOptimization] = Field(default_factory=list, description="Performance optimization recommendations") 
    security_optimizations: List[SecurityOptimization] = Field(default_factory=list, description="Security optimization recommendations")
    syntax_validations: List[SyntaxValidation] = Field(default_factory=list, description="Syntax validation results")
    naming_conventions: List[NamingConvention] = Field(default_factory=list, description="Naming convention recommendations")
    tagging_strategies: List[TaggingStrategy] = Field(default_factory=list, description="Tagging strategy recommendations")
    estimated_monthly_cost: Optional[str] = Field(None, description="Estimated monthly cost after optimizations")
    optimization_summary: str = Field(..., description="Summary of all optimizations applied")
    implementation_priority: List[str] = Field(default_factory=list, description="Priority order for implementing optimizations")

class ConfigurationOptimizationResponseList(BaseModel):
    """Optimization recommendation for the configuration react agent"""
    config_optimizer_recommendations: List[ConfigurationOptimizerResponse] = Field(..., description="Configuration optimizer recommendations")

## Terraform State Management

# Input Schema
class InfrastructureScale(str, Enum):
    SMALL = "small"          # < 50 resources
    MEDIUM = "medium"        # 50-200 resources  
    LARGE = "large"          # 200-500 resources
    ENTERPRISE = "enterprise" # 500+ resources

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    SHARED = "shared"

class TeamStructure(BaseModel):
    """Team structure information for state planning"""
    team_size: int = Field(..., description="Number of team members working on infrastructure")
    teams: List[str] = Field(..., description="List of teams (e.g., ['platform', 'backend', 'frontend'])")
    concurrent_operations: bool = Field(..., description="Whether teams work on infrastructure concurrently")
    ci_cd_integration: bool = Field(..., description="Whether CI/CD pipelines will use Terraform")

class ComplianceRequirements(BaseModel):
    """Compliance and security requirements"""
    encryption_required: bool = Field(True, description="Whether encryption is required for state files")
    audit_logging: bool = Field(False, description="Whether audit logging is required")
    backup_retention_days: Optional[int] = Field(None, description="Backup retention period in days")
    compliance_standards: List[str] = Field(default_factory=list, description="Compliance standards (SOC2, HIPAA, etc.)")

class StateManagementPlannerRequest(BaseModel):
    """Input request for state management planning"""
    service_name: str = Field(..., description="AWS service or infrastructure component")
    infrastructure_scale: InfrastructureScale = Field(..., description="Scale of infrastructure")
    environments: List[Environment] = Field(..., description="Environments to support")
    team_structure: TeamStructure = Field(..., description="Team structure information")
    compliance_requirements: ComplianceRequirements = Field(..., description="Compliance and security requirements")
    aws_region: str = Field(..., description="Primary AWS region")
    multi_region: bool = Field(False, description="Whether infrastructure spans multiple regions")
    existing_state_files: Optional[List[str]] = Field(None, description="Existing state files to consider for migration")

# Output Schema - State Management Plan
class BackendConfiguration(BaseModel):
    """S3 backend configuration recommendation"""
    bucket_name: str = Field(..., description="Recommended S3 bucket name")
    key_pattern: str = Field(..., description="Key pattern for state files")
    region: str = Field(..., description="AWS region for the backend")
    encrypt: bool = Field(..., description="Whether to enable encryption")
    versioning: bool = Field(..., description="Whether to enable versioning")
    kms_key_id: Optional[str] = Field(None, description="KMS key ID for encryption if specified")
    server_side_encryption_configuration: Dict[str, Any] = Field(..., description="S3 encryption configuration")

class StateLockingConfiguration(BaseModel):
    """DynamoDB state locking configuration"""
    table_name: str = Field(..., description="DynamoDB table name for state locking")
    billing_mode: str = Field(..., description="Billing mode (PAY_PER_REQUEST or PROVISIONED)")
    hash_key: str = Field("LockID", description="Primary key for the table")
    region: str = Field(..., description="AWS region for DynamoDB table")
    point_in_time_recovery: bool = Field(..., description="Whether to enable point-in-time recovery")
    tags: Dict[str, str] = Field(..., description="Tags for the DynamoDB table")

class StateSplittingStrategy(BaseModel):
    """Strategy for organizing and splitting state files"""
    splitting_approach: str = Field(..., description="Approach for state splitting")
    state_files: List[Dict[str, str]] = Field(..., description="List of recommended state files with descriptions")
    dependencies: List[Dict[str, Any]] = Field(..., description="Dependencies between state files")
    data_source_usage: List[str] = Field(..., description="How to use terraform_remote_state data sources")

class BackendSecurityRecommendations(BaseModel):
    """Security recommendations for the backend"""
    iam_policies: List[Dict[str, Any]] = Field(default_factory=list, description="IAM policies for backend access")
    bucket_policies: List[Dict[str, Any]] = Field(default_factory=list, description="S3 bucket policies")
    access_controls: List[Any] = Field(default_factory=list, description="Access control recommendations")
    monitoring: List[Any] = Field(default_factory=list, description="Monitoring and alerting recommendations")

class StateManagementPlannerResponse(BaseModel):
    """Complete state management plan"""
    service_name: str = Field(..., description="AWS service being planned")
    infrastructure_scale: str = Field(..., description="Infrastructure scale")
    backend_configuration: BackendConfiguration = Field(..., description="S3 backend configuration")
    state_locking_configuration: StateLockingConfiguration = Field(..., description="DynamoDB locking configuration")
    state_splitting_strategy: StateSplittingStrategy = Field(..., description="State splitting and organization strategy")
    security_recommendations: BackendSecurityRecommendations = Field(..., description="Security recommendations")
    migration_plan: Optional[List[str]] = Field(None, description="Migration plan if existing state files present")
    implementation_steps: List[str] = Field(..., description="Step-by-step implementation guide")
    best_practices: List[str] = Field(..., description="State management best practices")
    monitoring_setup: List[str] = Field(..., description="Monitoring and alerting setup recommendations")
    disaster_recovery: List[str] = Field(..., description="Backup and disaster recovery recommendations")

class StateManagementPlanResponseList(BaseModel):
    """Complete state management plan"""
    state_management_plan_responses: List[StateManagementPlannerResponse] = Field(..., description="State management plan responses")



## Terraform Execution Planner Pydantic Models
# Enhanced schemas for comprehensive module specification
class VariableDefinition(BaseModel):
    """Complete variable definition with all attributes"""
    name: str = Field(..., description="Variable name")
    type: str = Field(..., description="Terraform variable type")
    description: str = Field(..., description="Variable description")
    default: Optional[Any] = Field(None, description="Default value")
    sensitive: bool = Field(False, description="Whether variable is sensitive")
    nullable: bool = Field(False, description="Whether variable can be null")
    validation_rules: List[Any] = Field(default_factory=list, description="Validation blocks")
    example_values: List[Any] = Field(default_factory=list, description="Example values for documentation")
    justification: str = Field(default="", description="Why this variable is needed")

class LocalValue(BaseModel):
    """Local value definition"""
    name: str = Field(..., description="Local value name")
    expression: str = Field(..., description="Terraform expression")
    description: str = Field(..., description="Purpose and usage description")
    depends_on: List[str] = Field(default_factory=list, description="Dependencies on variables or other locals")
    usage_context: str = Field(default="", description="Context where this local value is used")

class DataSource(BaseModel):
    """Data source definition"""
    resource_name: str = Field(..., description="Data source resource name (e.g., 'current_caller_identity')")
    data_source_type: str = Field(..., description="Data source type (e.g., 'aws_caller_identity')")
    configuration: Dict[str, Any] = Field(..., description="Data source configuration")
    description: str = Field(..., description="Purpose of this data source")
    exported_attributes: List[str] = Field(..., description="Attributes that will be referenced")

class OutputDefinition(BaseModel):
    """Complete output definition"""
    name: str = Field(..., description="Output name")
    value: str = Field(..., description="Output value expression")
    description: str = Field(..., description="Output description")
    sensitive: bool = Field(False, description="Whether output is sensitive")
    depends_on: List[str] = Field(default_factory=list, description="Explicit dependencies")
    precondition: Optional[Dict[str, str]] = Field(None, description="Precondition block if needed")
    consumption_notes: str = Field(default="", description="Notes about how this output is consumed")

class IAMPolicyDocument(BaseModel):
    """IAM policy document specification"""
    policy_name: str = Field(..., description="Policy identifier")
    version: str = Field("2012-10-17", description="Policy version")
    statements: List[Dict[str, Any]] = Field(..., description="Policy statements")
    description: str = Field(..., description="Policy purpose")
    resource_references: List[str] = Field(..., description="Resources this policy applies to")


class ResourceConfiguration(BaseModel):
    """Complete resource configuration"""
    model_config = ConfigDict(extra='ignore', validate_assignment=False)
    
    resource_address: Optional[str] = Field(None, description="Full resource address")
    resource_type: Optional[str] = Field(None, description="AWS resource type")
    resource_name: Optional[str] = Field(None, description="Resource instance name")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Complete resource configuration")
    depends_on: List[str] = Field(default_factory=list, description="Resource dependencies")
    lifecycle_rules: Optional[Dict[str, Any]] = Field(None, description="Lifecycle configuration")
    tags_strategy: Optional[str] = Field(default="", description="Tagging strategy")
    parameter_justification: str = Field(default="", description="Justification for parameters")


class TerraformFile(BaseModel):
    """Terraform file specification"""
    file_name: str = Field(..., description="Filename (e.g., main.tf)")
    file_purpose: str = Field(..., description="File purpose")
    resources_included: List[str] = Field(default_factory=list, description="Resources defined in this file")
    dependencies: List[str] = Field(default_factory=list, description="File dependencies")
    organization_rationale: str = Field(default="", description="Rationale for file organization")
    # Keep original fields for backward compatibility
    # filename: str = Field(default="", description="Filename (e.g., main.tf)")
    # purpose: str = Field(default="", description="File purpose")
    # content_sections: List[str] = Field(default_factory=list, description="Ordered list of content sections")
    # includes_resources: List[str] = Field(default_factory=list, description="Resources defined in this file")
    # includes_variables: List[str] = Field(default_factory=list, description="Variables defined in this file")
    # includes_outputs: List[str] = Field(default_factory=list, description="Outputs defined in this file")

class ModuleExample(BaseModel):
    """Module usage example"""
    example_name: str = Field(..., description="Example scenario name")
    configuration: str = Field(default="", description="Complete module block code")
    description: str = Field(..., description="What this example demonstrates")
    expected_outputs: List[str] = Field(default_factory=list, description="Expected outputs from this example")
    use_case: str = Field(default="", description="Use case description")
    # Keep original fields for backward compatibility
    # module_call: str = Field(default="", description="Complete module block code")
    # required_variables: Dict[str, Any] = Field(default_factory=dict, description="Required variable values")

class ComprehensiveExecutionPlanResponse(BaseModel):
    """Complete execution plan with full module specification"""
    service_name: str = Field(..., description="AWS service being deployed")
    module_name: str = Field(..., description="Module name")
    target_environment: str = Field(..., description="Target deployment environment")
    plan_generation_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Complete module specification
    terraform_files: List[TerraformFile] = Field(..., description="All Terraform files to be created")
    variable_definitions: List[VariableDefinition] = Field(..., description="Complete variable specifications")
    local_values: List[LocalValue] = Field(..., description="Local value definitions")
    data_sources: List[DataSource] = Field(..., description="Data source definitions")
    output_definitions: List[OutputDefinition] = Field(..., description="Output specifications")
    resource_configurations: List[ResourceConfiguration] = Field(..., description="Complete resource configurations")
    iam_policies: List[IAMPolicyDocument] = Field(default_factory=list, description="IAM policy documents")
    
    # Module documentation and examples
    module_description: str = Field(..., description="Comprehensive module description")
    usage_examples: List[ModuleExample] = Field(..., description="Usage examples for different scenarios")
    readme_content: str = Field(..., description="Complete README.md content")
    
    # Provider and version requirements
    required_providers: Dict[str, Dict[str, str]] = Field(..., description="Provider requirements")
    terraform_version_constraint: str = Field(..., description="Minimum Terraform version")
    
    # Deployment and operational details
    resource_dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Resource dependency graph")
    deployment_phases: List[str] = Field(default_factory=list, description="Deployment phases")
    estimated_costs: Dict[str, Any] = Field(default_factory=dict, description="Cost estimates by resource type")
    # security_considerations: List[str] = Field(default_factory=list, description="Security considerations and warnings")
    
    # Testing and validation
    validation_and_testing: List[str] = Field(default_factory=list, description="Built-in validation rules and testing approaches")
    # validation_rules: List[str] = Field(default_factory=list, description="Built-in validation rules")
    # testing_strategy: List[str] = Field(default_factory=list, description="Recommended testing approaches")
    # compliance_checks: List[str] = Field(default_factory=list, description="Compliance validations built into module")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if execution plan creation failed")

class ExecutionPlanResponseList(BaseModel):
    """Complete execution plan"""
    execution_plan_responses: List[ComprehensiveExecutionPlanResponse] = Field(..., description="Execution plan responses")

def _initialize_execution_tools(config: Config):
    """Initialize LLM and parsers for execution tools."""
    global _model, _model_higher, _module_structure_plan_parser, _module_react_agent_parser, _configuration_optimizer_parser, _state_mgmt_parser, _execution_planner_parser
    
    if _model is None:
        llm_config = config.get_llm_config()
        _model = LLMProvider.create_llm(
            provider=llm_config['provider'],
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        llm_higher_config = config.get_llm_higher_config()
        _model_higher = LLMProvider.create_llm(
            provider=llm_higher_config['provider'],
            model=llm_higher_config['model'],
            temperature=llm_higher_config['temperature'],
            max_tokens=llm_higher_config['max_tokens']
        )
        _module_structure_plan_parser = JsonOutputParser(pydantic_object=ModuleStructurePlanResponse)
        _configuration_optimizer_parser = JsonOutputParser(pydantic_object=ConfigurationOptimizerResponse)
        _state_mgmt_parser = JsonOutputParser(pydantic_object=StateManagementPlannerResponse)
        _execution_planner_parser = JsonOutputParser(pydantic_object=ComprehensiveExecutionPlanResponse)
        
        

def extract_service_data(aws_service_mapping, service_name):
    """
    Extract service data from aws_service_mapping for a specific aws_service_type.
    
    Args:
        aws_service_mapping: Dictionary containing AWS service mapping data
        service_name: Specific AWS service name to extract (e.g., 'vpc', 's3', 'kms')
        
    Returns:
        Dictionary with extracted service information for the specified service type
    """
    try:
        # Check if aws_service_mapping is a string and parse it
        if isinstance(aws_service_mapping, str):
            aws_service_mapping = json.loads(aws_service_mapping)
        
        # Extract specific service by aws_service_type
        if 'services' in aws_service_mapping:
            for service in aws_service_mapping['services']:
                if service.get('service_name', '') == service_name:
                    service_info = {
                        "service_name": service.get('service_name', ''),
                        "aws_service_type": service.get('aws_service_type', ''),
                        "architecture_patterns": service.get('architecture_patterns', []),
                        "well_architected_alignment": service.get('well_architected_alignment', []),
                        "terraform_resources": service.get('terraform_resources', []),
                        "dependencies": service.get('dependencies', []),
                        "cost_optimization_recommendations": service.get('cost_optimization_recommendations', [])
                    }
                    
                    execution_logger.log_structured(
                        level="INFO",
                        message=f"Service data extraction completed for {service_name}",
                        extra={
                            "requested_service_type": service_name,
                            "found_service": service_info["service_name"],
                            "extraction_successful": True
                        }
                    )
                    
                    return service_info
        
        # Service type not found
        execution_logger.log_structured(
            level="WARNING",
            message=f"Service type '{service_name}' not found in mapping",
            extra={
                "requested_service_type": service_name,
                "available_services": [s.get('aws_service_type', '') for s in aws_service_mapping.get('services', [])],
                "extraction_successful": False
            }
        )
        
        return {"error": f"Service type '{service_name}' not found"}
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Service data extraction failed: {e}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "requested_service_type": service_name,
                "extraction_successful": False
            }
        )
        return {"error": str(e)}

def extract_aws_service_names(aws_service_mapping):
    """
    Extract a list of AWS service names from aws_service_mapping.
    
    Args:
        aws_service_mapping: Dictionary containing AWS service mapping data
        
    Returns:
        List[str]: List of AWS service names available in the mapping
        
    Raises:
        Exception: If aws_service_mapping is invalid or parsing fails
    """
    try:
        # Check if aws_service_mapping is a string and parse it
        if isinstance(aws_service_mapping, str):
            aws_service_mapping = json.loads(aws_service_mapping)
        
        # Extract service names from the mapping
        service_names = []
        if 'services' in aws_service_mapping:
            for service in aws_service_mapping['services']:
                service_name = service.get('service_name', '')
                if service_name:  # Only add non-empty service names
                    service_names.append(service_name)
        
        execution_logger.log_structured(
            level="INFO",
            message=f"AWS service names extraction completed",
            extra={
                "total_services_found": len(service_names),
                "service_names": service_names,
                "extraction_successful": True
            }
        )
        
        return service_names
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"AWS service names extraction failed: {e}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "extraction_successful": False
            }
        )
        return []

def extract_terraform_resource_attributes(terraform_attribute_mapping, service_name):
    """
    Extract terraform resource attributes from terraform_attribute_mapping for a specific aws_service_type.
    
    Args:
        terraform_attribute_mapping: Dictionary containing Terraform attribute mapping data
        service_name: Specific AWS service name to extract (e.g., 'vpc', 's3', 'kms')
        
    Returns:
        Dictionary with extracted Terraform resource attributes for the specified service type
    """
    try:
        # Check if terraform_attribute_mapping is a string and parse it
        if isinstance(terraform_attribute_mapping, str):
            terraform_attribute_mapping = json.loads(terraform_attribute_mapping)
        
        # Extract specific service by aws_service_type
        if 'services' in terraform_attribute_mapping:
            for service in terraform_attribute_mapping['services']:
                if service.get('service_name', '') == service_name:
                    service_info = {
                        "service_name": service.get('service_name', ''),
                        "aws_service_type": service.get('aws_service_type', ''),
                        "description": service.get('description', ''),
                        "terraform_resources": []
                    }
                    
                    # Extract Terraform resources for this service
                    if 'terraform_resources' in service:
                        for resource in service['terraform_resources']:
                            # Simplify required attributes to only include name, type, required, description
                            required_attrs = []
                            for attr in resource.get('required_attributes', []):
                                required_attrs.append({
                                    "name": attr.get('name', ''),
                                    "type": attr.get('type', ''),
                                    "required": attr.get('required', True),
                                    "description": attr.get('description', '')
                                })
                            
                            # Simplify optional attributes to only include name, type, required, description
                            optional_attrs = []
                            for attr in resource.get('optional_attributes', []):
                                optional_attrs.append({
                                    "name": attr.get('name', ''),
                                    "type": attr.get('type', ''),
                                    "required": attr.get('required', False),
                                    "description": attr.get('description', '')
                                })
                            
                            resource_info = {
                                "resource_name": resource.get('resource_name', ''),
                                "provider": resource.get('provider', ''),
                                "description": resource.get('description', ''),
                                "required_attributes": required_attrs,
                                "optional_attributes": optional_attrs
                            }
                            
                            # Extract recommended arguments from module_design if available
                            if resource.get('module_design') and isinstance(resource.get('module_design'), dict):
                                resource_info["recommended_arguments"] = resource['module_design'].get('recommended_arguments', [])
                                resource_info["recommended_outputs"] = resource['module_design'].get('recommended_outputs', [])

                            else:
                                resource_info["recommended_arguments"] = []
                            
                            service_info["terraform_resources"].append(resource_info)
                    
                    execution_logger.log_structured(
                        level="INFO",
                        message=f"Terraform resource attributes extraction completed for {service_name}",
                        extra={
                            "requested_service_type": service_name,
                            "found_service": service_info["service_name"],
                            "resources_count": len(service_info["terraform_resources"]),
                            "extraction_successful": True
                        }
                    )
                    
                    return service_info
        
        # Service type not found
        execution_logger.log_structured(
            level="WARNING",
            message=f"Service type '{service_name}' not found in mapping",
            extra={
                "requested_service_type": service_name,
                "available_services": [s.get('service_name', '') for s in terraform_attribute_mapping.get('services', [])],
                "extraction_successful": False
            }
        )
        
        return {"error": f"Service type '{service_name}' not found"}
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Terraform resource attributes extraction failed: {e}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "requested_service_type": service_name,
                "extraction_successful": False
            }
        )
        return {"error": str(e)}


async def create_module_structure_plan(service_name: str) -> ModuleStructurePlanResponse:
    """
    Creates a comprehensive Terraform module structure plan for a specified AWS service.
    
    This tool analyzes AWS service requirements, Terraform resource attributes, and architectural
    patterns to generate a detailed plan for organizing Terraform module files, variables, outputs,
    and implementation guidance. It leverages AI to design reusable, secure, and composable
    Terraform module structures following AWS and Terraform best practices.
    
    Args:
        service_name (str): The AWS service name to plan for (e.g., 'vpc', 's3', 'rds', 'ec2').
                               Must match a service type available in the planner state.
    
    Returns:
        ModuleStructurePlanResponse: A comprehensive module structure plan containing:
            - service_name: Target AWS service name
            - recommended_files: List of Terraform files with purposes and content descriptions
            - variable_definitions: Planned variables with types, validation, and justifications
            - output_definitions: Planned outputs with expressions and justifications
            - security_considerations: Security best practices for the module
            - reusability_guidance: Naming conventions, tagging strategies, and composability hints
            - implementation_notes: Additional guidance for implementation teams
    
    Raises:
        ValueError: If execution tools are not initialized or planner state is unavailable
        Exception: If module structure plan creation fails due to missing data or processing errors
    """
    try:
        if _model is None:
            raise ValueError("Execution tools not initialized. Call _initialize_execution_tools first.")
        
        # Access requirements_data from global state
        global _shared_planner_state
        if _shared_planner_state is None:
            raise ValueError("Planner state not available. Cannot access requirements_data.")


        tf_attribute_mapping = _shared_planner_state.requirements_data.terraform_attribute_mapping
        aws_service_mapping = _shared_planner_state.requirements_data.aws_service_mapping
        if tf_attribute_mapping is None or aws_service_mapping is None:
            execution_logger.log_structured(
                level="ERROR",
                message="No tf_attribute_mapping or aws_service_mapping available in state",
                extra={
                    "service_name": service_name
                }
            )
            return json.dumps({"error": "No tf_attribute_mapping or aws_service_mapping available in state"})
        
        service_data = extract_service_data(aws_service_mapping, service_name)
        terraform_resource_attributes = extract_terraform_resource_attributes(tf_attribute_mapping, service_name)
        execution_logger.log_structured(
            level="INFO",
            message="Starting async module structure plan creation",
            extra={
                "service_name": service_name
            }
        )
        service_name = service_data["service_name"]
        architecture_patterns = service_data["architecture_patterns"]
        well_architected_alignment = service_data["well_architected_alignment"]
        module_dependencies = service_data["dependencies"]
        terraform_resources = terraform_resource_attributes["terraform_resources"]
        _module_structure_plan_prompt = ChatPromptTemplate.from_messages([
            ("system", TF_MODULE_STRUCTURE_PLAN_SYSTEM_PROMPT),
            ("human", TF_MODULE_STRUCTURE_PLAN_USER_PROMPT)
        ])
        formatted_prompt = _module_structure_plan_prompt.format_messages(service_name=service_name, architecture_patterns=architecture_patterns, well_architected_alignment=well_architected_alignment, terraform_resources=terraform_resources, module_dependencies=module_dependencies)

        response = await _model.ainvoke(formatted_prompt)
        if isinstance(response, AIMessage):
            content = response.content
            content_str = str(content) if not isinstance(content, str) else content
        else:
            content_str = str(response) if not isinstance(response, str) else response

        module_structure_plan = _module_structure_plan_parser.parse(content_str)
        execution_logger.log_structured(
            level="INFO",
            message="Module structure plan created successfully",
            extra={
                "module_structure_plan": module_structure_plan
            }
        )
        
        return module_structure_plan

    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Async module structure plan creation failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Module structure plan creation failed: {str(e)}"})

        
async def create_configuration_optimizer(module_structure_plan: ModuleStructurePlanResponse) -> ConfigurationOptimizerResponse:
    """
    Creates configuration optimization recommendations for a Terraform module structure plan.
    
    This tool analyzes a module structure plan and provides comprehensive optimization
    recommendations covering cost efficiency, performance tuning, security hardening,
    syntax validation, naming conventions, and tagging strategies. It leverages AI to
    generate actionable recommendations based on AWS Well-Architected Framework principles
    and Terraform best practices.
    
    Args:
        module_structure_plan (ModuleStructurePlanResponse): The module structure plan to optimize.
    
    Returns:
        ConfigurationOptimizerResponse: Comprehensive optimization recommendations including:
            - service_name: Target AWS service name
            - cost_optimizations: Cost efficiency recommendations with savings estimates
            - performance_optimizations: Performance tuning recommendations
            - security_optimizations: Security hardening recommendations with severity levels
            - syntax_validations: Terraform syntax and structure validation results
            - naming_conventions: AWS naming convention recommendations
            - tagging_strategies: Tagging strategy recommendations
            - estimated_monthly_cost: Estimated monthly cost after optimizations
            - optimization_summary: Summary of all optimizations applied
            - implementation_priority: Priority order for implementing optimizations
    
    Raises:
        ValueError: If execution tools are not initialized or module structure plan is invalid
        Exception: If configuration optimization fails due to processing errors
    """
    try:        
        if _model is None:
            raise ValueError("Execution tools not initialized. Call _initialize_execution_tools first.")
        
        if _configuration_optimizer_parser is None:
            raise ValueError("Configuration optimizer parser not initialized.")
        
        execution_logger.log_structured(
            level="INFO",
            message="Starting configuration optimization analysis",
            extra={
                "service_name": module_structure_plan.service_name
            }
        )
        
        # Extract data from module structure plan
        service_name = module_structure_plan.service_name
        recommended_files = json.dumps([file.model_dump() for file in module_structure_plan.recommended_files])
        variable_definitions = json.dumps([var.model_dump() for var in module_structure_plan.variable_definitions])
        output_definitions = json.dumps([out.model_dump() for out in module_structure_plan.output_definitions])
        security_considerations = json.dumps(module_structure_plan.security_considerations)
        
        # Get configuration values from config
        config = Config()
        
        environment = config.CONFIG_OPTIMIZER_ENVIRONMENT
        expected_load = config.CONFIG_OPTIMIZER_EXPECTED_LOAD
        budget_constraints = config.CONFIG_OPTIMIZER_BUDGET_CONSTRAINTS
        compliance_requirements = json.dumps(config.CONFIG_OPTIMIZER_COMPLIANCE_REQUIREMENTS)
        optimization_targets = json.dumps(config.CONFIG_OPTIMIZER_OPTIMIZATION_TARGETS)
        organization_standards = json.dumps(config.CONFIG_OPTIMIZER_ORGANIZATION_STANDARDS)
        
        # Create prompt template
        _configuration_optimizer_prompt = ChatPromptTemplate.from_messages([
            ("system", TF_CONFIGURATION_OPTIMIZER_SYSTEM_PROMPT),
            ("human", TF_CONFIGURATION_OPTIMIZER_USER_PROMPT)
        ])
        
        # Format the prompt with extracted data
        formatted_prompt = _configuration_optimizer_prompt.format_messages(
            service_name=service_name,
            recommended_files=recommended_files,
            variable_definitions=variable_definitions,
            output_definitions=output_definitions,
            security_considerations=security_considerations,
            environment=environment,
            expected_load=expected_load,
            budget_constraints=budget_constraints,
            compliance_requirements=compliance_requirements,
            optimization_targets=optimization_targets,
            organization_standards=organization_standards
        )
        
        # Invoke the model
        response = await _model.ainvoke(formatted_prompt)
        if isinstance(response, AIMessage):
            content = response.content
            content_str = str(content) if not isinstance(content, str) else content
        else:
            content_str = str(response) if not isinstance(response, str) else response
        
        # Parse the response
        try:
            configuration_optimizer = _configuration_optimizer_parser.parse(content_str)
        except Exception as parse_error:
            execution_logger.log_structured(
                level="WARNING",
                message="Failed to parse configuration optimizer response, using fallback",
                extra={"error": str(parse_error), "response_content": content_str[:200] + "..." if len(content_str) > 200 else content_str}
            )
            
            # Create fallback response with error
            configuration_optimizer = ConfigurationOptimizerResponse(
                service_name=service_name,
                cost_optimizations=[],
                performance_optimizations=[],
                security_optimizations=[],
                syntax_validations=[],
                naming_conventions=[],
                tagging_strategies=[],
                estimated_monthly_cost=None,
                optimization_summary="Fallback summary due to parsing error",
                implementation_priority=["Review and fix parsing issues"],
                error=f"Failed to parse configuration optimizer response: {str(parse_error)}"
            )
        
        execution_logger.log_structured(
            level="INFO",
            message="Configuration optimization completed successfully",
            extra={
                "service_name": service_name,
                "optimization_summary": configuration_optimizer.get('optimization_summary', 'unknown') if isinstance(configuration_optimizer, dict) else configuration_optimizer.optimization_summary
            }
        )
        
        return configuration_optimizer
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Configuration optimization failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        raise Exception(f"Configuration optimization failed: {str(e)}")


async def create_state_mgmt(configuration_optimizer: ConfigurationOptimizerResponse) -> StateManagementPlannerResponse:
    """
    Creates a comprehensive Terraform state management plan based on configuration optimization data.
    
    This tool analyzes configuration optimization results and generates detailed state management
    plans including S3 backend configuration, DynamoDB state locking, state splitting strategies,
    security recommendations, and implementation guidance. It leverages AI to design enterprise-grade
    state management solutions following AWS and Terraform best practices.
    
    Args:
        configuration_optimizer (ConfigurationOptimizerResponse): Configuration optimization data 
                                                               to inform state management planning.
    
    Returns:
        StateManagementPlannerResponse: A comprehensive state management plan containing:
            - service_name: Target AWS service name
            - infrastructure_scale: Scale of infrastructure being managed
            - backend_configuration: S3 backend configuration recommendations
            - state_locking_configuration: DynamoDB locking configuration
            - state_splitting_strategy: State organization and splitting strategy
            - security_recommendations: Security and access control recommendations
            - migration_plan: Migration strategy if existing state files present
            - implementation_steps: Step-by-step implementation guide
            - best_practices: State management best practices
            - monitoring_setup: Monitoring and alerting recommendations
            - disaster_recovery: Backup and disaster recovery procedures
    
    Raises:
        ValueError: If execution tools are not initialized or configuration optimizer is invalid
        Exception: If state management planning fails due to processing errors
    """
    
    try:
        if _model is None:
            raise ValueError("Execution tools not initialized. Call _initialize_execution_tools first.")
        
        execution_logger.log_structured(
            level="INFO",
            message="Starting state management planning",
            extra={
                "service_name": configuration_optimizer.get('service_name', 'unknown') if isinstance(configuration_optimizer, dict) else configuration_optimizer.service_name
            }
        )
        
        # Extract service name from configuration optimizer
        service_name = configuration_optimizer.get('service_name', 'unknown') if isinstance(configuration_optimizer, dict) else configuration_optimizer.service_name
        
        # Get configuration values from config
        config = Config()
        
        _state_mgmt_prompt = ChatPromptTemplate.from_messages([
            ("system", TF_STATE_MGMT_SYSTEM_PROMPT),
            ("user", TF_STATE_MGMT_USER_PROMPT)
        ])
        # Format the prompt with extracted data and default values
        formatted_prompt = _state_mgmt_prompt.format_messages(
            service_name=service_name,
            infrastructure_scale=config.STATE_MGMT_DEFAULT_INFRASTRUCTURE_SCALE,
            environments=json.dumps(config.STATE_MGMT_DEFAULT_ENVIRONMENTS),
            aws_region=config.STATE_MGMT_DEFAULT_AWS_REGION,
            multi_region=config.STATE_MGMT_DEFAULT_MULTI_REGION,
            team_size=config.STATE_MGMT_DEFAULT_TEAM_SIZE,
            teams=json.dumps(config.STATE_MGMT_DEFAULT_TEAMS),
            concurrent_operations=config.STATE_MGMT_DEFAULT_CONCURRENT_OPERATIONS,
            ci_cd_integration=config.STATE_MGMT_DEFAULT_CI_CD_INTEGRATION,
            encryption_required=config.STATE_MGMT_DEFAULT_ENCRYPTION_REQUIRED,
            audit_logging=config.STATE_MGMT_DEFAULT_AUDIT_LOGGING,
            backup_retention_days=config.STATE_MGMT_DEFAULT_BACKUP_RETENTION_DAYS,
            compliance_standards=json.dumps(config.STATE_MGMT_DEFAULT_COMPLIANCE_STANDARDS),
            existing_state_files="[]"  # Default to empty for new infrastructure
        )
        
        # Invoke the model
        response = await _model.ainvoke(formatted_prompt)
        if isinstance(response, AIMessage):
            content = response.content
            content_str = str(content) if not isinstance(content, str) else content
        else:
            content_str = str(response) if not isinstance(response, str) else response
        
        # Parse the response
        try:
            state_mgmt_plan = _state_mgmt_parser.parse(content_str)
        except Exception as parse_error:
            execution_logger.log_structured(
                level="WARNING",
                message="Failed to parse state management plan response, using fallback",
                extra={"error": str(parse_error), "response_content": content_str[:200] + "..." if len(content_str) > 200 else content_str}
            )
            
            # Create fallback response with error
            state_mgmt_plan = StateManagementPlannerResponse(
                service_name=service_name,
                infrastructure_scale="medium",
                backend_configuration=BackendConfiguration(
                    bucket_name="fallback-bucket",
                    key_pattern="fallback-pattern",
                    region="us-east-1",
                    encrypt=True,
                    versioning=True,
                    kms_key_id=None,
                    server_side_encryption_configuration={}
                ),
                state_locking_configuration=StateLockingConfiguration(
                    table_name="fallback-table",
                    billing_mode="PAY_PER_REQUEST",
                    hash_key="LockID",
                    region="us-east-1",
                    point_in_time_recovery=True,
                    tags={}
                ),
                state_splitting_strategy=StateSplittingStrategy(
                    splitting_approach="fallback",
                    state_files=[],
                    dependencies=[],
                    data_source_usage=[]
                ),
                security_recommendations=BackendSecurityRecommendations(
                    iam_policies=[],
                    bucket_policies=[],
                    access_controls=[],
                    monitoring=[]
                ),
                migration_plan=[],
                implementation_steps=["Review and fix parsing issues"],
                best_practices=["Fallback best practices"],
                monitoring_setup=["Fallback monitoring"],
                disaster_recovery=["Fallback disaster recovery"],
                error=f"Failed to parse state management plan: {str(parse_error)}"
            )
        
        execution_logger.log_structured(
            level="INFO",
            message="State management plan created successfully",
            extra={
                "service_name": state_mgmt_plan.get('service_name', 'unknown') if isinstance(state_mgmt_plan, dict) else state_mgmt_plan.service_name,
                "infrastructure_scale": state_mgmt_plan.get('infrastructure_scale', 'unknown') if isinstance(state_mgmt_plan, dict) else state_mgmt_plan.infrastructure_scale
            }
        )
        
        return state_mgmt_plan
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"State management planning failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        raise Exception(f"State management planning failed: {str(e)}")


async def create_execution_plan(state_mgmt_plan: StateManagementPlannerResponse) -> ComprehensiveExecutionPlanResponse:
    """
    Creates a comprehensive Terraform execution plan based on all planning inputs.
    
    This tool aggregates data from module structure plans, configuration optimizations,
    and state management plans to generate a complete execution specification. It leverages 
    AI to create production-ready Terraform module specifications following AWS and 
    Terraform best practices.
    
    Args:
        state_mgmt_plan (StateManagementPlannerResponse): State management plan from the state management function.
    
    Returns:
        ComprehensiveExecutionPlanResponse: Complete execution plan including:
            - Terraform file specifications with exact content and organization
            - Complete variable definitions with validation and examples
            - Local values with expressions and dependencies
            - Data source specifications with configurations
            - Full resource configurations with all parameters
            - IAM policies with complete statements and permissions
            - Output definitions with sensitivity and dependencies
            - Usage examples for different scenarios
            - Complete documentation and README content
            - Validation and testing procedures
            - Deployment phases and cost estimates
    
    Raises:
        ValueError: If execution tools are not initialized or required planning data is missing
        Exception: If execution planning fails due to processing errors
    """
    try:
        if _model is None:
            raise ValueError("Execution tools not initialized. Call _initialize_execution_tools first.")
        
        # Check if state_mgmt_plan is None
        if state_mgmt_plan is None:
            raise ValueError("State management plan is None. Cannot proceed with execution planning.")
        
        # Access shared planner state
        global _shared_planner_state
        if _shared_planner_state is None:
            raise ValueError("Planner state not available. Cannot access planning data.")
        
        # Extract data from shared planner state
        module_structure_plan = _shared_planner_state.execution_data.module_structure_plan
        configuration_optimizer_data = _shared_planner_state.execution_data.configuration_optimizer_data
        
        # Validate required planning data exists
        if not module_structure_plan or not isinstance(module_structure_plan, dict) or not module_structure_plan.get("module_structure_plans"):
            raise ValueError("Module structure plan not available. Run create_module_structure_plan first.")
        
        if not configuration_optimizer_data or not isinstance(configuration_optimizer_data, dict) or not configuration_optimizer_data.get("configuration_optimizers"):
            raise ValueError("Configuration optimizer data not available. Run create_configuration_optimization first.")
        
        # Extract service name from state management plan
        target_service_name = state_mgmt_plan.service_name if hasattr(state_mgmt_plan, 'service_name') else 'unknown'
        
        execution_logger.log_structured(
            level="INFO",
            message="Starting execution planning",
            extra={
                "service_name": target_service_name,
                "infrastructure_scale": state_mgmt_plan.infrastructure_scale if hasattr(state_mgmt_plan, 'infrastructure_scale') else 'unknown'
            }
        )
        
        # Find matching module structure plan for the service
        matching_module_plan = None
        # Extract the list from the dict structure
        module_plans_list = module_structure_plan.get("module_structure_plans", []) if isinstance(module_structure_plan, dict) else module_structure_plan
        for plan in module_plans_list:
            if isinstance(plan, dict) and plan.get("service_name") == target_service_name:
                matching_module_plan = plan
                break
            elif hasattr(plan, 'service_name') and plan.service_name == target_service_name:
                matching_module_plan = plan.model_dump() if hasattr(plan, 'model_dump') else plan
                break
        
        if not matching_module_plan:
            raise ValueError(f"No module structure plan found for service: {target_service_name}")
        
        # Extract data from matching module structure plan
        service_name = matching_module_plan.get("service_name", "")
        recommended_files = json.dumps(matching_module_plan.get("recommended_files", []))
        variable_definitions = json.dumps(matching_module_plan.get("variable_definitions", []))
        output_definitions = json.dumps(matching_module_plan.get("output_definitions", []))
        security_considerations = json.dumps(matching_module_plan.get("security_considerations", []))
        
        # Find matching configuration optimizer data for the service
        matching_config_optimizer = None
        # Extract the list from the dict structure
        config_optimizer_list = configuration_optimizer_data.get("configuration_optimizers", []) if isinstance(configuration_optimizer_data, dict) else configuration_optimizer_data
        for config in config_optimizer_list:
            if isinstance(config, dict) and config.get("service_name") == target_service_name:
                matching_config_optimizer = config
                break
            elif hasattr(config, 'service_name') and config.service_name == target_service_name:
                matching_config_optimizer = config.model_dump() if hasattr(config, 'model_dump') else config
                break
        
        if not matching_config_optimizer:
            raise ValueError(f"No configuration optimizer data found for service: {target_service_name}")
        
        # Extract data from matching configuration optimizer
        cost_optimizations = json.dumps(matching_config_optimizer.get("cost_optimizations", []))
        performance_optimizations = json.dumps(matching_config_optimizer.get("performance_optimizations", []))
        security_optimizations = json.dumps(matching_config_optimizer.get("security_optimizations", []))
        naming_conventions = json.dumps(matching_config_optimizer.get("naming_conventions", []))
        tagging_strategies = json.dumps(matching_config_optimizer.get("tagging_strategies", []))
        
        # Extract data from state management plan
        backend_configuration = json.dumps(state_mgmt_plan.backend_configuration.model_dump() if hasattr(state_mgmt_plan.backend_configuration, 'model_dump') else state_mgmt_plan.backend_configuration)
        state_locking_configuration = json.dumps(state_mgmt_plan.state_locking_configuration.model_dump() if hasattr(state_mgmt_plan.state_locking_configuration, 'model_dump') else state_mgmt_plan.state_locking_configuration)
        state_splitting_strategy = json.dumps(state_mgmt_plan.state_splitting_strategy.model_dump() if hasattr(state_mgmt_plan.state_splitting_strategy, 'model_dump') else state_mgmt_plan.state_splitting_strategy)
        
        # Set deployment context defaults
        target_environment = "dev"  # Default environment
        ci_cd_integration = "GitHub Actions"  # Default CI/CD
        parallel_execution = "enabled"  # Default parallel execution
        
        # Create prompt template
        _execution_planner_prompt = ChatPromptTemplate.from_messages([
            ("system", TF_EXECUTION_PLANNER_SYSTEM_PROMPT),
            ("human", TF_EXECUTION_PLANNER_USER_PROMPT)
        ])
        
        # Format the prompt with extracted data
        formatted_prompt = _execution_planner_prompt.format_messages(
            service_name=service_name,
            recommended_files=recommended_files,
            variable_definitions=variable_definitions,
            output_definitions=output_definitions,
            security_considerations=security_considerations,
            cost_optimizations=cost_optimizations,
            performance_optimizations=performance_optimizations,
            security_optimizations=security_optimizations,
            naming_conventions=naming_conventions,
            tagging_strategies=tagging_strategies,
            backend_configuration=backend_configuration,
            state_locking_configuration=state_locking_configuration,
            state_splitting_strategy=state_splitting_strategy,
            target_environment=target_environment,
            ci_cd_integration=ci_cd_integration,
            parallel_execution=parallel_execution
        )
        
        # Invoke the model
        response = await _model.ainvoke(formatted_prompt)
        if isinstance(response, AIMessage):
            content = response.content
            content_str = str(content) if not isinstance(content, str) else content
        else:
            content_str = str(response) if not isinstance(response, str) else response
        
        # Parse the response
        try:
            repaired_content = json_repair.repair_json(content_str)
            execution_plan = _execution_planner_parser.parse(repaired_content)
        except Exception as parse_error:
            execution_logger.log_structured(
                level="WARNING",
                message="Failed to parse execution plan response, using fallback",
                extra={"error": str(parse_error), "response_content": content_str[:200] + "..." if len(content_str) > 200 else content_str}
            )
            
            # Create fallback response with error
            execution_plan = ComprehensiveExecutionPlanResponse(
                service_name=service_name,
                module_name=f"{service_name}-module",
                target_environment=target_environment,
                plan_generation_timestamp=datetime.now().isoformat(),
                terraform_files=[],
                variable_definitions=[],
                local_values=[],
                data_sources=[],
                resource_configurations=[],
                iam_policies=[],
                output_definitions=[],
                usage_examples=[],
                module_description="Fallback module description due to parsing error",
                readme_content="Fallback README content due to parsing error",
                required_providers={},
                terraform_version_constraint=">=1.0",
                resource_dependencies=[],
                deployment_phases=[],
                estimated_costs={},
                validation_and_testing=[],
                error=f"Failed to parse execution plan: {str(parse_error)}"
            )
        
        execution_logger.log_structured(
            level="INFO",
            message="Execution plan created successfully",
            extra={
                "service_name": execution_plan.get('service_name', 'unknown') if isinstance(execution_plan, dict) else execution_plan.service_name,
                "module_name": execution_plan.get('module_name', 'unknown') if isinstance(execution_plan, dict) else execution_plan.module_name,
                "target_environment": execution_plan.get('target_environment', 'unknown') if isinstance(execution_plan, dict) else execution_plan.target_environment,
                "has_error": execution_plan.get('error', 'unknown') if isinstance(execution_plan, dict) else execution_plan.error is not None
            }
        )
        
        return execution_plan
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Execution planning failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        raise Exception(f"Execution planning failed: {str(e)}")


@tool
async def create_module_structure_plan_tool() -> ModuleStructurePlanResponseList:
    """
    Create comprehensive Terraform module structure plans for multiple AWS services.
    
    This function processes a list of AWS services and generates detailed module structure plans
    for each service, including file organization, variable definitions, outputs, data sources,
    and security considerations. Each plan follows AWS best practices and Terraform conventions.
    
    The function iterates through each AWS service, creating individual module structure plans
    that include comprehensive file layouts, variable schemas, output specifications, and
    security considerations. All plans are designed to be production-ready and follow
    industry-standard Terraform module patterns.
    
    Returns:
        ModuleStructurePlanResponseList: A list of comprehensive module structure plans,
        one for each AWS service, containing file structure, variables, outputs, and
        implementation guidelines.
    
    Raises:
        ValueError: If service_list is empty or contains invalid service types
        RuntimeError: If module structure planning fails for any service
    
    """
    try:        
        # Declare global variables
        global _shared_planner_state, _execution_sequence
        if _shared_planner_state is None:
            raise ValueError("Planner state not available. Cannot access requirements_data.")

        aws_service_mapping = _shared_planner_state.requirements_data.aws_service_mapping
        service_list = extract_aws_service_names(aws_service_mapping)
        if service_list is None:
            execution_logger.log_structured(
                level="ERROR",
                message="No service list available in state",
                extra={}
            )
            return json.dumps({"error": "No service list available in state"})
        results = []
        for service in service_list:
            try:
                result = await create_module_structure_plan(service)
                if isinstance(result, dict):
                    result = ModuleStructurePlanResponse(**result)
                results.append(result)
            except Exception as e:
                execution_logger.log_structured(
                    level="ERROR",
                    message=f"Failed to create module structure plan for {service}",
                    extra={"service": service, "error": str(e)}
                )
                results.append(ModuleStructurePlanResponse(
                    service_name=service,
                    recommended_files=[],
                    variable_definitions=[],
                    output_definitions=[],
                    security_considerations=[],
                    reusability_guidance=ReusabilityGuidance(
                        naming_conventions=[],
                        tagging_strategy=[],
                        composability_hints=[],
                        best_practices=[]
                    ),
                    implementation_notes=[f"Failed to create plan: {str(e)}"]
                ))
        
    
        # Wrap list in dict to match Pydantic model expectations
        module_structure_plans_list = [result.dict() for result in results]
        _shared_planner_state.execution_data.module_structure_plan = {"module_structure_plans": module_structure_plans_list}
        _shared_planner_state.execution_data.module_structure_plan_complete = True
        
        # Mark module structure planning as complete
        _execution_sequence["module_structure_complete"] = True
        
        execution_logger.log_structured(
            level="INFO",
            message="Multi-service module structure planning completed",
            extra={"total_services": len(service_list), "successful_plans": len(results), "sequence_step_complete": "module_structure"}
        )
        return ModuleStructurePlanResponseList(module_structure_plans=results)

    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Failed to create module structure plan for all services",
            extra={"error": str(e)}
        )
        return ModuleStructurePlanResponseList(module_structure_plans=[ModuleStructurePlanResponse(
            service_name=service,
            recommended_files=[],
            variable_definitions=[],
            output_definitions=[],
            security_considerations=[],
            reusability_guidance=ReusabilityGuidance(
                naming_conventions=[],
                tagging_strategy=[],
                composability_hints=[],
                best_practices=[]
            ),
            implementation_notes=[f"Failed to create plan: {str(e)}"]
        )])
    
@tool
async def create_configuration_optimizations_tool(module_plans: str) -> ConfigurationOptimizationResponseList:
    """
    Generate configuration optimization recommendations for Terraform modules.
    
    This tool analyzes module structure plans and provides comprehensive optimization
    strategies including cost optimization, performance improvements, security enhancements,
    and operational best practices. It leverages AWS Well-Architected Framework principles
    to ensure production-ready configurations.
    
    Args:
        module_plans: Module structure plans to optimize. Can be:
                     - JSON string representation of module plans
    
    Returns:
        ConfigurationOptimizationResponseList: A list of optimization recommendations,
        one for each module, containing cost analysis, security improvements,
        performance optimizations, and implementation guidance.
    
    Raises:
        ValueError: If module_plans is empty or malformed
        TypeError: If module_plans contains invalid data types
        RuntimeError: If optimization analysis fails
    
    Example:
        Input: List of ModuleStructurePlanResponse objects
        Output: List of ConfigurationOptimizationResponseList with optimization strategies
    """
    # Declare global variables
    global _execution_sequence, _shared_planner_state
    
    # Validate execution sequence
    if not _execution_sequence["module_structure_complete"]:
        raise ValueError("Configuration optimization tool called before module structure planning is complete. Execute create_module_structure_plan_tool first.")
    
    execution_logger.log_structured(
        level="INFO",
        message="Starting multi-plan configuration optimization",
        extra={"plan_count": len(module_plans), "sequence_validated": True}
    )
    if _shared_planner_state is None:
        raise ValueError("Planner state not available. Cannot access module_plans.")

    if isinstance(module_plans, str):
        try:
            module_plans = json.loads(module_plans)
        except json.JSONDecodeError as e:
            execution_logger.log_structured(
                level="ERROR",
                message=f"Failed to parse module_plans JSON: {e}",
                extra={"error": str(e), "json_length": len(module_plans)}
            )
            # Try to extract data from shared state as fallback
            if _shared_planner_state and _shared_planner_state.execution_data.module_structure_plan:
                module_structure_data = _shared_planner_state.execution_data.module_structure_plan
                # Extract list from dict structure
                if isinstance(module_structure_data, dict) and "module_structure_plans" in module_structure_data:
                    module_plans = module_structure_data["module_structure_plans"]
                else:
                    module_plans = module_structure_data  # Fallback for old format
                execution_logger.log_structured(
                    level="INFO",
                    message="Using module structure plan from shared state as fallback",
                    extra={"fallback_source": "shared_state"}
                )
            else:
                raise ValueError(f"Failed to parse module_plans JSON and no fallback data available: {e}")
    
    # Extract the actual module plans from the JSON structure
    if isinstance(module_plans, dict) and 'module_structure_plans' in module_plans:
        module_plans = module_plans['module_structure_plans']

    results = []
    for plan in module_plans:
        try:
            if isinstance(plan, dict):
                plan = ModuleStructurePlanResponse(**plan)
            result = await create_configuration_optimizer(plan)
            if isinstance(result, dict):
                result = ConfigurationOptimizerResponse(**result)
            results.append(result)
        except Exception as e:
            execution_logger.log_structured(
                level="ERROR",
                message=f"Failed to create configuration optimization for {plan.get('service_name', 'unknown') if isinstance(plan, dict) else plan.service_name}",
                extra={"service_name": plan.get('service_name', 'unknown') if isinstance(plan, dict) else plan.service_name, "error": str(e)}
            )
            # Create error result
            # Create error result
            service_name = plan.get('service_name', 'unknown') if isinstance(plan, dict) else plan.service_name
            error_result = ConfigurationOptimizerResponse(
                service_name=service_name,
                cost_optimizations=[],
                performance_optimizations=[],
                security_optimizations=[],
                syntax_validations=[],
                naming_conventions=[],
                tagging_strategies=[],
                estimated_monthly_cost=None,
                optimization_summary=f"Failed to create optimization: {str(e)}",
                implementation_priority=["Review and fix errors"],
                error=f"Failed to create configuration optimization: {str(e)}"
            )
            results.append(error_result)
    
    # Update shared planner state
    # Wrap list in dict to match Pydantic model expectations
    configuration_optimizer_list = [result.dict() for result in results]
    _shared_planner_state.execution_data.configuration_optimizer_data = {"configuration_optimizers": configuration_optimizer_list}
    _shared_planner_state.execution_data.configuration_optimizer_complete = True
    
    # Mark configuration optimization as complete
    _execution_sequence["configuration_optimization_complete"] = True
    
    execution_logger.log_structured(
        level="INFO",
        message="Multi-plan configuration optimization completed",
        extra={"total_plans": len(module_plans), "successful_optimizations": len(results), "sequence_step_complete": "configuration_optimization"}
    )
    return ConfigurationOptimizationResponseList(config_optimizer_recommendations=results)

@tool
async def create_state_management_plans_tool(optimizations: str) -> StateManagementPlanResponseList:
    """
    Create comprehensive state management plans for Terraform infrastructure.
    
    This tool processes configuration optimizations and generates detailed state management
    strategies including backend configuration, state locking, security policies, and
    disaster recovery plans. It ensures enterprise-grade state management following
    AWS best practices for production environments.
    
    Args:
        optimizations: JSON string or list of configuration optimizations to process.
                      Each optimization should contain service information and configuration
                      details for state management planning.
    
    Returns:
        StateManagementPlanResponseList: A list of state management plans,
        one for each optimization, containing backend configuration, security policies,
        state splitting strategies, and implementation guidelines.
    
    Raises:
        ValueError: If optimizations is empty or malformed
        RuntimeError: If state management planning fails for any optimization
    
    Example:
        Input: List of ConfigurationOptimizerResponse objects
        Output: List of StateManagementPlanResponseList with state management strategies
    """
    # Declare global variables
    global _execution_sequence, _shared_planner_state
    
    # Validate execution sequence
    if not _execution_sequence["configuration_optimization_complete"]:
        raise ValueError("State management planning tool called before configuration optimization is complete. Execute create_configuration_optimizations_tool first.")
    
    execution_logger.log_structured(
        level="INFO",
        message="Starting multi-optimization state management planning",
        extra={"optimization_count": len(optimizations), "sequence_validated": True}
    )
    if _shared_planner_state is None:
        raise ValueError("Planner state not available. Cannot access optimizations.")

    if isinstance(optimizations, str):
        try:
            optimizations = json.loads(optimizations)
        except json.JSONDecodeError as e:
            execution_logger.log_structured(
                level="ERROR",
                message=f"Failed to parse optimizations JSON: {e}",
                extra={"error": str(e), "json_length": len(optimizations)}
            )
            # Try to extract data from shared state as fallback
            if _shared_planner_state and _shared_planner_state.execution_data.configuration_optimizer_data:
                configuration_data = _shared_planner_state.execution_data.configuration_optimizer_data
                # Extract list from dict structure
                if isinstance(configuration_data, dict) and "configuration_optimizers" in configuration_data:
                    optimizations = configuration_data["configuration_optimizers"]
                else:
                    optimizations = configuration_data  # Fallback for old format
                execution_logger.log_structured(
                    level="INFO",
                    message="Using configuration optimizer data from shared state as fallback",
                    extra={"fallback_source": "shared_state"}
                )
            else:
                raise ValueError(f"Failed to parse optimizations JSON and no fallback data available: {e}")
    
    # Extract the actual optimizations from the JSON structure
    if isinstance(optimizations, dict) and 'config_optimizer_recommendations' in optimizations:
        optimizations = optimizations['config_optimizer_recommendations']
    elif isinstance(optimizations, dict) and 'optimizations' in optimizations:
        optimizations = optimizations['optimizations']
    
    results = []
    for optimization in optimizations:
        try:
            if isinstance(optimization, dict):
                optimization = ConfigurationOptimizerResponse(**optimization)
            result = await create_state_mgmt(optimization)
            if isinstance(result, dict):
                result = StateManagementPlannerResponse(**result)
            results.append(result)
        except Exception as e:
            execution_logger.log_structured(
                level="ERROR",
                message=f"Failed to create state management plan for {optimization.get('service_name', 'unknown') if isinstance(optimization, dict) else optimization.service_name}",
                extra={"service_name": optimization.get('service_name', 'unknown') if isinstance(optimization, dict) else optimization.service_name, "error": str(e)}
            )
            # Create error result with minimal required fields
            error_result = StateManagementPlannerResponse(
                service_name=optimization.get('service_name', 'unknown') if isinstance(optimization, dict) else optimization.service_name,
                infrastructure_scale="unknown",
                backend_configuration=BackendConfiguration(
                    bucket_name="error-bucket",
                    key_pattern="error-pattern",
                    region="us-east-1",
                    encrypt=True,
                    versioning=True,
                    kms_key_id=None,
                    server_side_encryption_configuration={}
                ),
                state_locking_configuration=StateLockingConfiguration(
                    table_name="error-table",
                    billing_mode="PAY_PER_REQUEST",
                    hash_key="LockID",
                    region="us-east-1",
                    point_in_time_recovery=True,
                    tags={}
                ),
                state_splitting_strategy=StateSplittingStrategy(
                    splitting_approach="error",
                    state_files=[],
                    dependencies=[],
                    data_source_usage=[]
                ),
                security_recommendations=BackendSecurityRecommendations(
                    iam_policies=[],
                    bucket_policies=[],
                    access_controls=[],
                    monitoring=[]
                ),
                migration_plan=[],
                implementation_steps=["Review and fix errors"],
                best_practices=["Error recovery best practices"],
                monitoring_setup=["Error monitoring setup"],
                disaster_recovery=["Error disaster recovery"],
                error=f"Failed to create state management plan: {str(e)}"
            )
            results.append(error_result)
    
    # Update shared planner state

    # Wrap list in dict to match Pydantic model expectations
    state_management_list = [result.dict() for result in results]
    _shared_planner_state.execution_data.state_management_data = {"state_management_plans": state_management_list}
    _shared_planner_state.execution_data.state_management_complete = True
    
    # Mark state management planning as complete
    _execution_sequence["state_management_complete"] = True
    
    execution_logger.log_structured(
        level="INFO",
        message="Multi-optimization state management planning completed",
        extra={"total_optimizations": len(optimizations), "successful_plans": len(results), "sequence_step_complete": "state_management"}
    )
   
    return StateManagementPlanResponseList(state_management_plan_responses=results)


@tool
async def create_execution_plan_tool(state_mgmt_plans: str) -> ExecutionPlanResponseList:
    """
    Generate comprehensive execution plans for Terraform module implementation.
    
    This tool processes state management plans and creates detailed execution specifications
    that serve as complete blueprints for Terraform module generation. It aggregates
    all planning inputs (module structure, optimizations, state management) to produce
    production-ready implementation plans with testing strategies and deployment guidance.
    
    Args:
        state_mgmt_plans: JSON string or list of state management plans to process.
                         Each plan should contain complete state management configuration
                         and infrastructure details for execution planning.
    
    Returns:
        ExecutionPlanResponseList: A list of comprehensive execution plans,
        one for each state management plan, containing implementation details, testing
        strategies, deployment plans, and monitoring setup.
    
    Raises:
        ValueError: If state_mgmt_plans is empty or malformed
        RuntimeError: If execution planning fails for any plan
    """
    # Declare global variables
    global _execution_sequence, _shared_planner_state
    
    # Validate execution sequence
    if not _execution_sequence["state_management_complete"]:
        raise ValueError("Execution planning tool called before state management planning is complete. Execute create_state_management_plans_tool first.")
    
    execution_logger.log_structured(
        level="INFO",
        message="Starting multi-state management execution planning",
        extra={"state_mgmt_count": len(state_mgmt_plans) if isinstance(state_mgmt_plans, list) else "unknown", "sequence_validated": True}
    )
    if _shared_planner_state is None:
        raise ValueError("Planner state not available. Cannot access state_mgmt_plans.")
        
    if isinstance(state_mgmt_plans, str):
        try:
            state_mgmt_plans = json.loads(state_mgmt_plans)
        except json.JSONDecodeError as e:
            execution_logger.log_structured(
                level="ERROR",
                message=f"Failed to parse state_mgmt_plans JSON: {e}",
                extra={"error": str(e), "json_length": len(state_mgmt_plans)}
            )
            # Try to extract data from shared state as fallback
            if _shared_planner_state and _shared_planner_state.execution_data.state_management_data:
                state_management_data = _shared_planner_state.execution_data.state_management_data
                # Extract list from dict structure
                if isinstance(state_management_data, dict) and "state_management_plans" in state_management_data:
                    state_mgmt_plans = state_management_data["state_management_plans"]
                else:
                    state_mgmt_plans = state_management_data  # Fallback for old format
                execution_logger.log_structured(
                    level="INFO",
                    message="Using state management data from shared state as fallback",
                    extra={"fallback_source": "shared_state"}
                )
            else:
                raise ValueError(f"Failed to parse state_mgmt_plans JSON and no fallback data available: {e}")
    
    # Extract the actual state management plans from the JSON structure
    if isinstance(state_mgmt_plans, dict) and 'state_management_plan_responses' in state_mgmt_plans:
        state_mgmt_plans = state_mgmt_plans['state_management_plan_responses']
    
    # Validate that we have state management plans to process
    if not state_mgmt_plans:
        execution_logger.log_structured(
            level="ERROR",
            message="No state management plans available for execution planning",
            extra={"state_mgmt_plans": state_mgmt_plans}
        )
        raise ValueError("No state management plans available for execution planning")
    
    results = []
    for state_mgmt_plan in state_mgmt_plans:
        try:
            if isinstance(state_mgmt_plan, dict):
                state_mgmt_plan = StateManagementPlannerResponse(**state_mgmt_plan)
            result = await create_execution_plan(state_mgmt_plan)
            if isinstance(result, dict):
                result = ComprehensiveExecutionPlanResponse(**result)
            results.append(result)
        except Exception as e:
            execution_logger.log_structured(
                level="ERROR",
                message=f"Failed to create execution plan for state management plan",
                extra={"error": str(e)}
            )
            # Create error result with minimal required fields
            error_result = ComprehensiveExecutionPlanResponse(
                service_name=state_mgmt_plan.get("service_name", "unknown") if isinstance(state_mgmt_plan, dict) else state_mgmt_plan.service_name,
                module_structure=ModuleStructurePlanResponse(
                    service_name="error-service",
                    module_file_structure=[],
                    variable_definitions=[],
                    output_definitions=[],
                    data_source_requirements=[],
                    local_value_definitions=[],
                    version_constraints={},
                    provider_configuration={},
                    backend_configuration={},
                    module_dependencies=[],
                    security_considerations=[],
                    testing_strategy=[],
                    documentation_requirements=[]
                ),
                configuration_optimizations=[],
                state_management_plan=state_mgmt_plan.dict() if isinstance(state_mgmt_plan, dict) else state_mgmt_plan.model_dump(),
                implementation_plan=[],
                testing_plan=[],
                deployment_strategy=[],
                monitoring_setup=[],
                error=f"Failed to create execution plan: {str(e)}"
            )
            results.append(error_result)
    
    # Update shared planner state
    completion_data = create_agent_completion_data(
        agent_name="execution_planner",
        task_type="execution_plan",
        data_type="execution_plan",
        status="completed"
    )

    # Wrap list in dict to match Pydantic model expectations
    execution_plans_list = [result.model_dump() for result in results]
    _shared_planner_state.execution_data.execution_plan_data = {"execution_plans": execution_plans_list}
    _shared_planner_state.execution_data.execution_plan_complete = True
    _shared_planner_state.execution_data.agent_completion = completion_data
    
    # CRITICAL: Mark execution planning as complete in the workflow state
    # This is what the supervisor checks to determine if execution is complete
    _shared_planner_state.planning_workflow_state.execution_complete = True
    
    # Mark execution planning as complete
    _execution_sequence["execution_plan_complete"] = True
    
    execution_logger.log_structured(
        level="INFO",
        message="Multi-state management execution planning completed",
        extra={"total_plans": len(state_mgmt_plans), "successful_plans": len(results), "sequence_step_complete": "execution_plan"}
    )
    return ExecutionPlanResponseList(execution_plan_responses=results)


def create_execution_planner_react_agent(state: PlannerSupervisorState, config: Config):    
    """
    Create a comprehensive React agent for Terraform execution planning.
    
    This function initializes and configures an execution planner React agent that orchestrates
    the complete Terraform planning workflow. The agent processes AWS services through a
    four-stage pipeline: module structure planning, configuration optimization, state management
    planning, and comprehensive execution plan generation.
    
    The agent integrates with the shared planner state to maintain workflow continuity and
    provides comprehensive logging for debugging and monitoring purposes.
    
    Args:
        state (PlannerSupervisorState): The shared planner state containing workflow data
                                       and execution context. Used to maintain state across
                                       different planning stages.
        config (Config): Configuration instance containing LLM settings, tool configurations,
                        and other execution parameters. Must include valid LLM provider
                        configuration for agent creation.
    
    Returns:
        React agent: A fully configured React agent instance capable of executing the
                    complete Terraform planning workflow with the following tools:
                    - create_module_structure_plan_tool: Generates module structure plans
                    - create_configuration_optimizations_tool: Creates optimization strategies
                    - create_state_management_plans_tool: Generates state management plans
                    - create_execution_plan_tool: Creates comprehensive execution plans
    
    Raises:
        RuntimeError: If LLM configuration is invalid or unavailable
        ValueError: If required configuration parameters are missing
        Exception: If agent creation or tool initialization fails
    
    Side Effects:
        - Initializes global shared planner state
        - Sets up execution tools and configurations
        - Configures logging and monitoring for the agent
    
    Note:
        This function requires a valid LLM configuration in the config object.
        The agent will log all major operations and errors for debugging purposes.
    """
    try:
        execution_logger.log_structured(
            level="INFO",
            message="=== CREATING EXECUTION PLANNER REACT AGENT ===",
            extra={"config_type": type(config).__name__}
        )
        
        global _shared_planner_state, _execution_sequence
        _shared_planner_state = state
        
        # Reset execution sequence for new agent instance
        _execution_sequence = {
            "module_structure_complete": False,
            "configuration_optimization_complete": False,
            "state_management_complete": False,
            "execution_plan_complete": False
        }
        
        _initialize_execution_tools(config)
        
        llm = _model_higher  # Use the higher-tier model for agent tasks
        # Create React agent with async tools
        execution_logger.log_structured(
            level="DEBUG",
            message="Creating React agent with async tools",
            extra={
                "tools_count": 4,
                "tool_names": ["create_execution_plan_tool", "create_module_structure_plan_tool", "create_configuration_optimizations_tool", "create_state_management_plans_tool"]
            }
        )
    
        execution_planner = create_react_agent(
            model=llm,
            tools=[
                create_module_structure_plan_tool,
                create_configuration_optimizations_tool,
                create_state_management_plans_tool,
                create_execution_plan_tool
            ],
            name="execution_planner",
            checkpointer=state.memory if hasattr(state, 'memory') else None,  # Use shared memory for proper state management
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
You are an expert Terraform Execution Planner focused on comprehensive infrastructure planning and module generation specifications.

# CRITICAL: SEQUENTIAL EXECUTION ONLY
You MUST execute tools ONE AT A TIME in the exact sequence specified. Do NOT execute multiple tools simultaneously or in parallel. Each tool depends on the output of the previous tool.

# Role and Objective
Produce robust execution plans by managing the end-to-end planning process, from initial module structure to final implementation details.

# Required Initial Checklist
Begin with a concise checklist (3-7 bullets) outlining the conceptual workflow steps you will follow before starting tool execution.

# MANDATORY SEQUENTIAL WORKFLOW
You MUST follow this exact sequence - NO EXCEPTIONS:

STEP 1: Call `create_module_structure_plan_tool()` with NO parameters
- Wait for complete response
- Validate the output contains module structure plans
- Store the JSON output for next step

STEP 2: Call `create_configuration_optimizations_tool(module_plans)` 
- Pass the EXACT JSON output from Step 1 as the parameter
- Wait for complete response
- Validate the output contains optimization strategies
- Store the JSON output for next step

STEP 3: Call `create_state_management_plans_tool(optimizations)`
- Pass the EXACT JSON output from Step 2 as the parameter
- Wait for complete response
- Validate the output contains state management plans
- Store the JSON output for next step

STEP 4: Call `create_execution_plan_tool(state_mgmt_plans)`
- Pass the EXACT JSON output from Step 3 as the parameter
- Wait for complete response
- Validate the output contains execution plans

# EXECUTION RULES
- Execute ONLY ONE tool at a time
- Wait for each tool to complete before starting the next
- Do NOT skip steps or reorder the sequence
- Do NOT execute tools in parallel
- Each tool MUST receive the JSON output from the previous tool
- If any tool fails, stop immediately and report the error

# VALIDATION REQUIREMENTS
- After each tool call, validate the output is complete and valid
- Only proceed to the next step if validation passes
- If validation fails, report the specific error and terminate

# FINAL OUTPUT
After all 4 tools complete successfully, respond with ONLY the raw JSON from `create_execution_plan_tool`. No additional text or formatting.

# ERROR HANDLING
- If any tool fails, immediately stop and report the error
- Do not attempt to continue with subsequent tools
- Provide specific error details for debugging

Remember: SEQUENTIAL EXECUTION ONLY - One tool at a time, in the exact order specified.
            """),
            MessagesPlaceholder(variable_name="messages")
        ])
    )
        

        execution_logger.log_structured(
            level="INFO",
            message="=== EXECUTION PLANNER REACT AGENT CREATED SUCCESSFULLY ===",
            extra={
                "agent_type": type(execution_planner).__name__,
                "global_model_type": type(_model).__name__,
                "global_model_repr": str(_model),
                "tools_count": 4,
                "enhanced_prompt": True,
                "json_output_required": True,
                "sequential_execution_enforced": True,
                "required_workflow": "create_module_structure_plan_tool -> create_configuration_optimizations_tool -> create_state_management_plans_tool -> create_execution_plan_tool -> JSON output"
            }
        )
        
        return execution_planner
    
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message="=== FAILED TO CREATE EXECUTION PLANNER REACT AGENT ===",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "config_type": type(config).__name__ if config else "None"
            }
        )
        raise
