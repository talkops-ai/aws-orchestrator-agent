"""
Security and Best Practices React Agent for Planner Sub-Supervisor.

This module implements the Security and Best Practices as a React agent with tools:
- security_compliance_tool: Analyzes security compliance and compliance frameworks for AWS infrastructure
- best_practices_tool: Analyzes AWS and Terraform best practices for infrastructure deployment
"""

import json
from enum import Enum
from typing import Dict, Any, List, Literal, Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, model_validator
from langgraph.prebuilt import create_react_agent

from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger
from aws_orchestrator_agent.core.agents.planner.planner_supervisor_state import PlannerSupervisorState
from .sec_n_best_practices_prompts import SECURITY_COMPLIANCE_SYSTEM_PROMPT, SECURITY_COMPLIANCE_USER_PROMPT, BEST_PRACTICES_SYSTEM_PROMPT, BEST_PRACTICES_USER_PROMPT

# Create logger
security_compliance_logger = AgentLogger("SECURITY_COMPLIANCE_REACT")
best_practices_logger = AgentLogger("BEST_PRACTICES_REACT")
security_n_best_practices_logger = AgentLogger("SECURITY_AND_BEST_PRACTICES_REACT")

class ComplianceDetail(BaseModel):
    status: Literal["compliant", "non_compliant"] = Field(
        ..., description="Compliance status for the control"
    )
    issues: List[str] = Field(
        default_factory=list, description="List of identified issues"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Remediation steps for issues"
    )

class NetworkSecurityDetail(BaseModel):
    security_groups: ComplianceDetail = Field(
        ..., description="Analysis of Security Groups configurations"
    )
    network_acls: ComplianceDetail = Field(
        ..., description="Analysis of Network ACLs configurations"
    )

class AccessControlDetail(BaseModel):
    iam_roles: ComplianceDetail = Field(
        ..., description="Analysis of IAM Role definitions and usage"
    )
    iam_policies: ComplianceDetail = Field(
        ..., description="Analysis of IAM Policy documents"
    )

class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

class BestPracticeFinding(BaseModel):
    id: str = Field(..., description="Unique finding ID")
    status: ValidationStatus = Field(..., description="PASS/WARN/FAIL")
    resource: str = Field(..., description="Resource or config checked")
    check: str = Field(..., description="Description of best practice checked")
    recommendation: str = Field(..., description="Remediation or advice")

# Enhanced schemas for multi-service analysis
class ServiceSecurityAnalysis(BaseModel):
    """Security analysis for a single service."""
    service_name: str = Field(..., description="Name of the AWS service")
    service_type: str = Field(..., description="AWS service type identifier")
    
    # Service-specific security analysis
    encryption_at_rest: ComplianceDetail = Field(..., description="Encryption at rest compliance for this service")
    encryption_in_transit: ComplianceDetail = Field(..., description="Encryption in transit compliance for this service")
    network_security: NetworkSecurityDetail = Field(..., description="Network security compliance for this service")
    access_controls: AccessControlDetail = Field(..., description="Access control compliance for this service")
    
    # Service-specific findings
    service_compliance: Literal["compliant", "non_compliant"] = Field(..., description="Compliance status for this service")
    service_issues: List[str] = Field(default_factory=list, description="Service-specific security issues")
    service_recommendations: List[str] = Field(default_factory=list, description="Service-specific security recommendations")

class CrossServiceSecurityAnalysis(BaseModel):
    """Cross-service security relationships and dependencies."""
    service_dependencies: Dict[str, List[str]] = Field(..., description="Security dependencies between services")
    shared_security_risks: List[str] = Field(default_factory=list, description="Security risks that affect multiple services")
    cross_service_recommendations: List[str] = Field(default_factory=list, description="Recommendations for cross-service security")

class OverallSecuritySummary(BaseModel):
    """Overall security summary across all services."""
    total_services: int = Field(..., description="Total number of services analyzed")
    compliant_services: int = Field(..., description="Number of compliant services")
    non_compliant_services: int = Field(..., description="Number of non-compliant services")
    critical_issues_count: int = Field(..., description="Number of critical security issues")
    high_priority_issues_count: int = Field(..., description="Number of high priority security issues")
    overall_risk_level: Literal["low", "medium", "high", "critical"] = Field(..., description="Overall security risk level")

class EnhancedSecurityAnalysisResult(BaseModel):
    """Enhanced security analysis result supporting multiple services."""
    
    # Individual service analysis
    services: List[ServiceSecurityAnalysis] = Field(..., description="Security analysis for each service")
    
    # Cross-service analysis
    cross_service_analysis: CrossServiceSecurityAnalysis = Field(..., description="Cross-service security relationships")
    
    # Overall summary
    overall_summary: OverallSecuritySummary = Field(..., description="Overall security summary")
    
    # Legacy fields for backward compatibility
    encryption_at_rest: ComplianceDetail = Field(..., description="Overall encryption at rest compliance")
    encryption_in_transit: ComplianceDetail = Field(..., description="Overall encryption in transit compliance")
    network_security: NetworkSecurityDetail = Field(..., description="Overall network security compliance")
    access_controls: AccessControlDetail = Field(..., description="Overall access control compliance")
    overall_compliance: Literal["compliant", "non_compliant"] = Field(..., description="Overall compliance status")
    summary_issues: List[str] = Field(default_factory=list, description="Overall summary of security issues")
    
    @model_validator(mode='after')
    def validate_overall_compliance(self):
        """Validate that overall compliance reflects individual service compliance."""
        compliant_count = sum(1 for service in self.services if service.service_compliance == "compliant")
        total_services = len(self.services)
        
        if total_services > 0:
            compliance_ratio = compliant_count / total_services
            if compliance_ratio == 1.0:
                self.overall_compliance = "compliant"
            else:
                self.overall_compliance = "non_compliant"
        
        return self

class ServiceBestPracticesAnalysis(BaseModel):
    """Best practices analysis for a single service."""
    service_name: str = Field(..., description="Name of the AWS service")
    service_type: str = Field(..., description="AWS service type identifier")
    
    # Service-specific best practices
    naming_and_tagging: List[BestPracticeFinding] = Field(..., description="Naming and tagging checks for this service")
    module_structure: List[BestPracticeFinding] = Field(..., description="Module structure checks for this service")
    resource_optimization: List[BestPracticeFinding] = Field(..., description="Resource optimization for this service")
    terraform_practices: List[BestPracticeFinding] = Field(..., description="Terraform practices for this service")
    
    # Service-specific status
    service_status: ValidationStatus = Field(..., description="Overall status for this service")
    service_score: int = Field(..., ge=0, le=100, description="Best practices score for this service (0-100)")

class CrossServiceBestPracticesAnalysis(BaseModel):
    """Cross-service best practices relationships."""
    shared_patterns: List[str] = Field(default_factory=list, description="Best practices patterns shared across services")
    consistency_issues: List[str] = Field(default_factory=list, description="Inconsistencies between services")
    cross_service_recommendations: List[str] = Field(default_factory=list, description="Recommendations for cross-service consistency")

class OverallBestPracticesSummary(BaseModel):
    """Overall best practices summary across all services."""
    total_services: int = Field(..., description="Total number of services analyzed")
    services_passing: int = Field(..., description="Number of services with PASS status")
    services_warning: int = Field(..., description="Number of services with WARN status")
    services_failing: int = Field(..., description="Number of services with FAIL status")
    average_score: float = Field(..., ge=0, le=100, description="Average best practices score across all services")
    overall_status: ValidationStatus = Field(..., description="Overall best practices status")

class EnhancedBestPracticesResponse(BaseModel):
    """Enhanced best practices response supporting multiple services."""
    
    # Individual service analysis
    services: List[ServiceBestPracticesAnalysis] = Field(..., description="Best practices analysis for each service")
    
    # Cross-service analysis
    cross_service_analysis: CrossServiceBestPracticesAnalysis = Field(..., description="Cross-service best practices relationships")
    
    # Overall summary
    overall_summary: OverallBestPracticesSummary = Field(..., description="Overall best practices summary")
    
    # Legacy fields for backward compatibility
    naming_and_tagging: List[BestPracticeFinding] = Field(..., description="Overall naming and tagging checks")
    module_structure: List[BestPracticeFinding] = Field(..., description="Overall module structure checks")
    resource_optimization: List[BestPracticeFinding] = Field(..., description="Overall resource optimization")
    terraform_practices: List[BestPracticeFinding] = Field(..., description="Overall Terraform practices")
    overall_status: ValidationStatus = Field(..., description="Overall status")
    
    @model_validator(mode='after')
    def validate_overall_status(self):
        """Validate that overall status reflects individual service statuses."""
        pass_count = sum(1 for service in self.services if service.service_status == ValidationStatus.PASS)
        warn_count = sum(1 for service in self.services if service.service_status == ValidationStatus.WARN)
        fail_count = sum(1 for service in self.services if service.service_status == ValidationStatus.FAIL)
        
        if fail_count > 0:
            self.overall_status = ValidationStatus.FAIL
        elif warn_count > 0:
            self.overall_status = ValidationStatus.WARN
        else:
            self.overall_status = ValidationStatus.PASS
        
        return self

class DependencyMapping(BaseModel):
    """Output schema for dependency mapping analysis."""
    primary_service: str = Field(description="The main AWS service being requested")
    mandatory_dependencies: List[Dict[str, Any]] = Field(description="Dependencies that must be set up")
    optional_dependencies: List[Dict[str, Any]] = Field(description="Dependencies that may be needed")
    dependency_categories: Dict[str, List[str]] = Field(description="Dependencies grouped by category")
    setup_prerequisites: List[str] = Field(description="Prerequisites that must be in place")
    terraform_provider_requirements: List[str] = Field(description="Required Terraform providers")
    dependency_explanations: Dict[str, str] = Field(description="Explanations for why each dependency is needed")
    follow_up_questions: List[str] = Field(description="Questions to ask the user for clarification")

# Global variables for LLM and parsers
_model = None
_security_compliance_parser = None
_security_compliance_prompt = None
_best_practices_parser = None
_best_practices_prompt = None
_shared_planner_state: Optional[PlannerSupervisorState] = None

def extract_security_relevant_data(terraform_mapping, aws_service_mapping):
    """
    Extract security-relevant data from both sources for multiple services.
    """
    security_data = {
        "services": [],
        "overall_security_summary": {
            "total_services": 0,
            "security_critical_resources": 0,
            "well_architected_security_alignment": []
        }
    }
    
    # Process each service in aws_service_mapping
    for service in aws_service_mapping.get("services", []):
        service_name = service.get("service_name")
        service_type = service.get("aws_service_type")
        
        # Find corresponding terraform data for this service
        terraform_service_data = None
        for terraform_service in terraform_mapping.get("services", []):
            if terraform_service.get("service_name") == service_name:
                terraform_service_data = terraform_service
                break
        
        service_security_data = {
            "service_info": {
                "service_name": service_name,
                "service_type": service_type,
                "description": service.get("description")
            },
            
            "security_architecture": {
                "well_architected_security": service.get("well_architected_alignment", {}).get("security", []),
                "security_features": service.get("production_features", {}).get("security_features", []),
                "architecture_patterns": [pattern.get("pattern_name") for pattern in service.get("architecture_patterns", [])]
            },
            
            "security_dependencies": {
                "required": [dep.get("name") for dep in service.get("dependencies", {}).get("required", [])],
                "optional": [dep.get("name") for dep in service.get("dependencies", {}).get("optional", [])],
                "recommended": [dep.get("name") for dep in service.get("dependencies", {}).get("recommended", [])]
            },
            
            "security_critical_resources": []
        }
        
        # Extract security-critical Terraform resources for this service
        if terraform_service_data:
            for resource in terraform_service_data.get("terraform_resources", []):
                security_attributes = []
                
                for attr in resource.get("attributes", []):
                    # Focus on security-relevant attributes
                    if any(security_keyword in attr.get("name", "").lower() for security_keyword in [
                        "encrypt", "decrypt", "key", "secret", "password", "token", "auth", "permission", 
                        "policy", "role", "security", "access", "grant", "alias", "arn", "id", "certificate",
                        "ssl", "tls", "vpc", "subnet", "route", "gateway", "firewall", "waf", "shield"
                    ]):
                        security_attributes.append({
                            "name": attr.get("name"),
                            "type": attr.get("type"),
                            "required": attr.get("required"),
                            "description": attr.get("description"),
                            "validation_rules": attr.get("validation_rules"),
                            "category": attr.get("category")
                        })
                
                if security_attributes:
                    service_security_data["security_critical_resources"].append({
                        "resource_name": resource.get("resource_name"),
                        "security_attributes": security_attributes
                    })
        
        security_data["services"].append(service_security_data)
        
        # Update overall summary
        security_data["overall_security_summary"]["total_services"] += 1
        security_data["overall_security_summary"]["security_critical_resources"] += len(service_security_data["security_critical_resources"])
        security_data["overall_security_summary"]["well_architected_security_alignment"].extend(
            service_security_data["security_architecture"]["well_architected_security"]
        )
    
    return security_data

def extract_best_practices_data(terraform_mapping, aws_service_mapping):
    """
    Extract best practices-relevant data from both sources for multiple services.
    """
    best_practices_data = {
        "services": [],
        "overall_best_practices_summary": {
            "total_services": 0,
            "total_resources": 0,
            "total_attributes": 0,
            "well_architected_pillars": {
                "operational_excellence": [],
                "security": [],
                "reliability": [],
                "performance_efficiency": [],
                "cost_optimization": [],
                "sustainability": []
            }
        }
    }
    
    # Process each service in aws_service_mapping
    for service in aws_service_mapping.get("services", []):
        service_name = service.get("service_name")
        service_type = service.get("aws_service_type")
        
        # Find corresponding terraform data for this service
        terraform_service_data = None
        for terraform_service in terraform_mapping.get("services", []):
            if terraform_service.get("service_name") == service_name:
                terraform_service_data = terraform_service
                break
        
        service_best_practices_data = {
            "service_info": {
                "service_name": service_name,
                "service_type": service_type
            },
            
            "well_architected_alignment": service.get("well_architected_alignment", {}),
            
            "cost_optimization": service.get("cost_optimization_recommendations", []),
            
            "resource_structure": {
                "total_resources": 0,
                "attribute_summary": {
                    "total_attributes": 0,
                    "required_attributes": 0,
                    "optional_attributes": 0,
                    "computed_attributes": 0
                }
            },
            
            "key_resources": []
        }
        
        # Extract key resources and attributes for this service
        if terraform_service_data:
            for resource in terraform_service_data.get("terraform_resources", []):
                key_attributes = []
                
                for attr in resource.get("attributes", []):
                    # Focus on optimization-relevant attributes
                    if any(opt_keyword in attr.get("name", "").lower() for opt_keyword in [
                        "name", "tag", "description", "alias", "version", "configuration", 
                        "setting", "parameter", "option", "feature", "enable", "disable",
                        "size", "capacity", "instance", "type", "class", "tier", "level"
                    ]):
                        key_attributes.append({
                            "name": attr.get("name"),
                            "type": attr.get("type"),
                            "required": attr.get("required"),
                            "description": attr.get("description"),
                            "category": attr.get("category"),
                            "is_output": attr.get("is_output"),
                            "is_reference": attr.get("is_reference")
                        })
                
                if key_attributes:
                    service_best_practices_data["key_resources"].append({
                        "resource_name": resource.get("resource_name"),
                        "key_attributes": key_attributes,
                        "attribute_counts": resource.get("attribute_counts", {})
                    })
                
                # Update resource structure counts
                service_best_practices_data["resource_structure"]["total_resources"] += 1
                service_best_practices_data["resource_structure"]["attribute_summary"]["total_attributes"] += len(resource.get("attributes", []))
                service_best_practices_data["resource_structure"]["attribute_summary"]["required_attributes"] += len([attr for attr in resource.get("attributes", []) if attr.get("required")])
                service_best_practices_data["resource_structure"]["attribute_summary"]["optional_attributes"] += len([attr for attr in resource.get("attributes", []) if not attr.get("required")])
                service_best_practices_data["resource_structure"]["attribute_summary"]["computed_attributes"] += len([attr for attr in resource.get("attributes", []) if attr.get("category") == "computed"])
        
        best_practices_data["services"].append(service_best_practices_data)
        
        # Update overall summary
        best_practices_data["overall_best_practices_summary"]["total_services"] += 1
        best_practices_data["overall_best_practices_summary"]["total_resources"] += service_best_practices_data["resource_structure"]["total_resources"]
        best_practices_data["overall_best_practices_summary"]["total_attributes"] += service_best_practices_data["resource_structure"]["attribute_summary"]["total_attributes"]
        
        # Aggregate well-architected alignment
        for pillar, recommendations in service.get("well_architected_alignment", {}).items():
            if pillar in best_practices_data["overall_best_practices_summary"]["well_architected_pillars"]:
                best_practices_data["overall_best_practices_summary"]["well_architected_pillars"][pillar].extend(recommendations)
    
    return best_practices_data

def _initialize_dependency_tools(config: Config):
    """Initialize LLM and parsers for dependency tools."""
    global _model, _security_compliance_parser, _best_practices_parser, _security_compliance_prompt, _best_practices_prompt, _shared_planner_state
    
    if _model is None:
        llm_config = config.get_llm_config()
        _model = LLMProvider.create_llm(
            provider=llm_config['provider'],
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        _security_compliance_parser = JsonOutputParser(pydantic_object=EnhancedSecurityAnalysisResult)
        _best_practices_parser = JsonOutputParser(pydantic_object=EnhancedBestPracticesResponse)
        _security_compliance_prompt = ChatPromptTemplate.from_messages([
            ("system", SECURITY_COMPLIANCE_SYSTEM_PROMPT),
            ("human", SECURITY_COMPLIANCE_USER_PROMPT)
        ])
        _best_practices_prompt = ChatPromptTemplate.from_messages([
            ("system", BEST_PRACTICES_SYSTEM_PROMPT),
            ("human", BEST_PRACTICES_USER_PROMPT)
        ])

@tool
async def security_compliance_tool(user_request: str) -> EnhancedSecurityAnalysisResult:
    """
    Production-grade security compliance analysis tool with comprehensive framework coverage and multi-service support.
    
    This tool provides:
    - Complete security compliance analysis across multiple frameworks (SOC 2, ISO 27001, HIPAA, PCI DSS)
    - Multi-service security analysis with cross-service relationships
    - Encryption analysis for data at rest and in transit with specific AWS service recommendations
    - Network security assessment including Security Groups, Network ACLs, and VPC configurations
    - Access control analysis covering IAM roles, policies, and least privilege principles
    - Compliance gap identification with actionable remediation recommendations
    - Security best practices validation against AWS Well-Architected Framework
    - Risk assessment and mitigation strategies for identified vulnerabilities
    
    Args:
        user_request: Natural language description of the infrastructure request
        
    Returns:
        EnhancedSecurityAnalysisResult: Comprehensive security compliance analysis with multi-service support
        
    Raises:
        ValidationError: If output doesn't meet security analysis quality standards
    """
    try:
        if _model is None:
            raise ValueError("Security and Best Practices tools not initialized. Call _initialize_dependency_tools first.")
        
        # Access requirements_data from global state
        global _shared_planner_state
        if _shared_planner_state is None:
            raise ValueError("Planner state not available. Cannot access requirements_data.")
        
        # Extract both data sources
        terraform_mapping = _shared_planner_state.requirements_data.terraform_attribute_mapping
        aws_service_mapping = _shared_planner_state.requirements_data.aws_service_mapping
        
        if terraform_mapping is None or aws_service_mapping is None:
            security_compliance_logger.log_structured(
                level="WARNING",
                message="No terraform_mapping or aws_service_mapping available in state, proceeding with user_request only",
                extra={"user_request": user_request[:100] + "..." if len(user_request) > 100 else user_request}
            )
            # Fallback to basic analysis with user request
            security_data = {"user_request": user_request, "services": []}
        else:
            # Curate security-relevant data for all services
            security_data = extract_security_relevant_data(terraform_mapping, aws_service_mapping)
        
        # Convert to JSON string for prompt
        security_data_str = json.dumps(security_data, indent=2)
        
        security_compliance_logger.log_structured(
            level="INFO",
            message="Starting async security compliance analysis with multi-service data",
            extra={
                "total_services": security_data.get("overall_security_summary", {}).get("total_services", 0),
                "security_critical_resources": security_data.get("overall_security_summary", {}).get("security_critical_resources", 0),
                "user_request": user_request[:100] + "..." if len(user_request) > 100 else user_request,
                "requirements_analysis_str_length": len(security_data_str)
            }
        )
        
        # Use curated data in prompt
        formatted_prompt = _security_compliance_prompt.format_messages(infrastructure_definition=security_data_str)
        response = await _model.ainvoke(formatted_prompt)
        if isinstance(response, AIMessage):
            response = response.content
        else:
            response = response
        security_compliance_logger.log_structured(
            level="DEBUG",
            message="Security compliance analysis LLM response received",
            extra={
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_length": len(response) if hasattr(response, 'content') else 0
            }
        )
        content = response.strip()
        if content.startswith('```json'):
            content = content[7:]  # Remove ```json
        if content.endswith('```'):
            content = content[:-3]  # Remove ```
        content = content.strip()
        security_response = _security_compliance_parser.parse(content)
        return security_response

    except Exception as e:
        security_compliance_logger.log_structured(
            level="ERROR",
            message=f"Async security compliance analysis failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Security compliance analysis failed: {str(e)}"})
    


@tool
async def best_practices_tool() -> EnhancedBestPracticesResponse:
    """
    Production-grade AWS and Terraform best practices analysis tool with comprehensive validation.
    
    This tool provides:
    - Naming and tagging best practices validation with AWS resource naming conventions
    - Module structure analysis for reusability, maintainability, and organization
    - Resource optimization recommendations for cost, performance, and scalability
    - Terraform best practices validation including syntax, state management, and security
    - AWS Well-Architected Framework alignment assessment
    - Infrastructure as Code (IaC) best practices validation
    - Security best practices for Terraform configurations and AWS resources
    - Performance optimization recommendations for resource configurations
    
    Returns:
        EnhancedBestPracticesResponse: Comprehensive best practices analysis with detailed findings and recommendations
        
    Raises:
        ValidationError: If output doesn't meet best practices analysis quality standards
    """
    try:
        if _model is None:
            raise ValueError("Security and Best Practices tools not initialized. Call _initialize_dependency_tools first.")
        
        # Access requirements_data from global state
        global _shared_planner_state
        if _shared_planner_state is None:
            raise ValueError("Planner state not available. Cannot access requirements_data.")
        
        # Extract both data sources
        terraform_mapping = _shared_planner_state.requirements_data.terraform_attribute_mapping
        aws_service_mapping = _shared_planner_state.requirements_data.aws_service_mapping
        
        if terraform_mapping is None or aws_service_mapping is None:
            best_practices_logger.log_structured(
                level="WARNING",
                message="No terraform_mapping or aws_service_mapping available in state, proceeding with empty infrastructure definition",
                extra={}
            )
            # Fallback to basic analysis
            best_practices_data = {"services": [], "overall_best_practices_summary": {"total_services": 0}}
        else:
            # Curate best practices-relevant data for all services
            best_practices_data = extract_best_practices_data(terraform_mapping, aws_service_mapping)
        
        # Convert to JSON string for prompt
        infrastructure_definition = json.dumps(best_practices_data, indent=2)
        
        best_practices_logger.log_structured(
            level="INFO",
            message="Starting async best practices analysis with multi-service data",
            extra={
                "total_services": best_practices_data.get("overall_best_practices_summary", {}).get("total_services", 0),
                "total_resources": best_practices_data.get("overall_best_practices_summary", {}).get("total_resources", 0),
                "infrastructure_definition_length": len(infrastructure_definition)
            }
        )
        formatted_prompt = _best_practices_prompt.format_messages(infrastructure_definition=infrastructure_definition)
        response = await _model.ainvoke(formatted_prompt)
        if isinstance(response, AIMessage):
            response = response.content
        else:
            response = response
        best_practices_logger.log_structured(
            level="DEBUG",
            message="Best practices analysis LLM response received",
            extra={
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_length": len(response) if hasattr(response, 'content') else 0
            }
        )
        content = response.strip()
        if content.startswith('```json'):
            content = content[7:]  # Remove ```json
        if content.endswith('```'):
            content = content[:-3]  # Remove ```
        content = content.strip()
        return _best_practices_parser.parse(content)
        
    except Exception as e:
        best_practices_logger.log_structured(
            level="ERROR",
            message=f"Async best practices analysis failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Best practices analysis failed: {str(e)}"})


def create_security_n_best_practices_react_agent(state: PlannerSupervisorState, config: Config):
    """
    Create a React agent for comprehensive security and best practices analysis.
    
    This agent provides:
    - Security compliance analysis across multiple frameworks (SOC 2, ISO 27001, HIPAA, PCI DSS)
    - AWS and Terraform best practices validation
    - Risk assessment and mitigation strategies
    - Compliance gap identification with actionable recommendations
    - Infrastructure as Code (IaC) best practices validation
    - Performance and cost optimization recommendations
    
    Args:
        config: Configuration instance containing LLM settings and other parameters
        state: PlannerSupervisorState instance containing user request and planner supervisor state parameters
    Returns:
        React agent for security and best practices analysis with enhanced tool integration
        
    Raises:
        ValueError: If configuration is invalid or tools fail to initialize
        Exception: If agent creation fails due to LLM or tool initialization issues
    """
    try:
        security_n_best_practices_logger.log_structured(
            level="INFO",
            message="=== CREATING SECURITY AND BEST PRACTICES REACT AGENT ===",
            extra={"config_type": type(config).__name__}
        )
        
        # Initialize tools
        security_n_best_practices_logger.log_structured(
            level="DEBUG",
            message="Initializing security and best practices tools",
            extra={}
        )
        
        # STORE STATE PARAMETER GLOBALLY FOR TOOLS TO ACCESS
        global _shared_planner_state
        _shared_planner_state = state
        
        _initialize_dependency_tools(config)
        
        # Get LLM from config
        llm_config = config.get_llm_config()
        
        security_n_best_practices_logger.log_structured(
            level="DEBUG",
            message="Creating LLM for security and best practices analysis",
            extra={
                "llm_provider": llm_config.get('provider'),
                "llm_model": llm_config.get('model'),
                "llm_temperature": llm_config.get('temperature'),
                "llm_max_tokens": llm_config.get('max_tokens')
            }
        )
        
        llm = LLMProvider.create_llm(
            provider=llm_config['provider'],
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        # Create React agent with async tools
        security_n_best_practices_logger.log_structured(
            level="DEBUG",
            message="Creating React agent with async tools",
            extra={
                "tools_count": 2,
                "tool_names": ["security_compliance_tool", "best_practices_tool"]
            }
        )
        
        security_n_best_practices_react_agent = create_react_agent(
            model=llm,
            tools=[security_compliance_tool, best_practices_tool],
            name="security_n_best_practices_evaluator",
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
You are an expert AWS Security and Best Practices Analyst specializing in comprehensive security compliance and infrastructure best practices validation with multi-service support.

## CORE MISSION
Analyze infrastructure requirements for security compliance across multiple frameworks and validate AWS/Terraform best practices for production deployment across multiple services.

## CRITICAL OUTPUT REQUIREMENT
After completing both tools successfully, return ONLY the JSON output from both tools combined. Do not add any explanations, summaries, or additional text. Return the raw JSON objects exactly as they were returned by the tools.

## WORKFLOW (MANDATORY SEQUENCE)
1. **SECURITY ANALYSIS**: Use security_compliance_tool with user request for multi-service security analysis
2. **BEST PRACTICES VALIDATION**: Use best_practices_tool for multi-service best practices validation
3. **VALIDATE**: Ensure both tools return complete, valid outputs with multi-service data
4. **RESPOND**: Return the combined analysis results from both tools

## TOOL EXECUTION RULES
- security_compliance_tool: Analyzes security compliance across frameworks (SOC 2, ISO 27001, HIPAA, PCI DSS) for multiple services
- best_practices_tool: Validates AWS and Terraform best practices for infrastructure deployment across multiple services
- BOTH tools are MANDATORY for comprehensive analysis
- Execute tools sequentially, never skip either one
- If either tool fails, report the specific failure and STOP

## SECURITY COMPLIANCE ANALYSIS COVERAGE (MULTI-SERVICE)
- **Service-Specific Security**: Individual security analysis for each service
- **Cross-Service Security**: Security relationships and dependencies between services
- **Overall Security Summary**: Comprehensive security overview across all services
- **Encryption**: Data at rest and in transit analysis per service
- **Network Security**: Security Groups, Network ACLs, VPC configurations per service
- **Access Control**: IAM roles, policies, least privilege principles per service
- **Compliance Frameworks**: SOC 2, ISO 27001, HIPAA, PCI DSS alignment per service
- **Risk Assessment**: Vulnerability identification and mitigation strategies per service

## BEST PRACTICES VALIDATION COVERAGE (MULTI-SERVICE)
- **Service-Specific Best Practices**: Individual best practices analysis for each service
- **Cross-Service Best Practices**: Consistency and patterns across services
- **Overall Best Practices Summary**: Comprehensive best practices overview
- **Naming & Tagging**: AWS resource naming conventions and tagging strategies per service
- **Module Structure**: Reusability, maintainability, and organization analysis per service
- **Resource Optimization**: Cost, performance, and scalability recommendations per service
- **Terraform Practices**: Syntax, state management, and security validation per service
- **IaC Best Practices**: Infrastructure as Code validation and recommendations per service

## SUCCESS VALIDATION CHECKLIST
✓ security_compliance_tool completed successfully with multi-service data
✓ best_practices_tool completed successfully with multi-service data
✓ Security analysis contains: service-specific analysis, cross-service relationships, overall summary
✓ Best practices analysis contains: service-specific analysis, cross-service consistency, overall summary
✓ No malformed or incomplete JSON output

## ERROR HANDLING
- Tool failure → Report specific error and terminate analysis
- Invalid JSON → Report JSON validation failure and terminate
- Missing required fields → Report incomplete analysis and terminate
- Only proceed to final response when ALL validations pass

## IMPORTANT NOTES
- Focus on production-ready security configurations across all services
- Validate against industry-standard compliance frameworks for each service
- Ensure Terraform configurations follow AWS and HashiCorp best practices for each service
- Provide actionable recommendations for security gaps and best practice violations per service
- Consider cost implications of security and optimization recommendations across services
- Analyze cross-service dependencies and security relationships

## FINAL OUTPUT REQUIREMENT
After successfully completing both tools, return ONLY the JSON output from both tools. Do not add any explanations, summaries, or additional text. Return the raw JSON objects exactly as they were returned by the tools.

CRITICAL: Return ONLY the JSON objects, no additional text or formatting.
        """),
                MessagesPlaceholder(variable_name="messages")
            ])
        )
        
        security_n_best_practices_logger.log_structured(
            level="INFO",
            message="=== SECURITY AND BEST PRACTICES REACT AGENT CREATED SUCCESSFULLY ===",
            extra={
                "agent_type": type(security_n_best_practices_react_agent).__name__,
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "tools_count": 2,
                "enhanced_prompt": True,
                "comprehensive_analysis": True,
                "required_workflow": "security_compliance_tool -> best_practices_tool -> combined analysis"
            }
        )
        
        return security_n_best_practices_react_agent
        
    except Exception as e:
        security_n_best_practices_logger.log_structured(
            level="ERROR",
            message="=== FAILED TO CREATE SECURITY AND BEST PRACTICES REACT AGENT ===",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "config_type": type(config).__name__ if config else "None"
            }
        )
        raise
