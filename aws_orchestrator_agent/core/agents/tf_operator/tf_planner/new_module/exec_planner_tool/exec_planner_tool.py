"""
Execution Planner Tools for Terraform Planner.

Implements the ToolRuntime + Command pattern (LangChain latest standard):
  - State is accessed via ``ToolRuntime[None, TFPlannerState]``
  - Inputs are read from ``runtime.state`` (state-driven, not agent-passed)
  - State updates are returned via ``Command(update={...})`` with ``ToolMessage``
  - LLM instances are created per-call via ``initialize_llm_*`` helpers
  - Structured output is parsed via ``PydanticOutputParser``

Tools (sequential pipeline):
  1. ``create_module_structure_plan_tool`` — module file structure for each service
  2. ``create_configuration_optimizations_tool`` — cost/perf/security optimisations
  3. ``create_state_management_plans_tool`` — S3 backend, locking, splitting
  4. ``create_execution_plan_tool`` — comprehensive execution specification
"""

import json
import re
from enum import Enum
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from aws_orchestrator_agent.utils.mcp_client import MCPClient
from aws_orchestrator_agent.config import Config
from aws_orchestrator_agent.core.state import TFPlannerState
from aws_orchestrator_agent.utils import (
    AgentLogger,
    initialize_llm_model,
    initialize_llm_higher,
)
from .exec_planner_prompts import (
    TF_MODULE_STRUCTURE_PLAN_SYSTEM_PROMPT,
    TF_MODULE_STRUCTURE_PLAN_USER_PROMPT,
    TF_CONFIGURATION_OPTIMIZER_SYSTEM_PROMPT,
    TF_CONFIGURATION_OPTIMIZER_USER_PROMPT,
    TF_STATE_MGMT_SYSTEM_PROMPT,
    TF_STATE_MGMT_USER_PROMPT,
    TF_EXECUTION_PLANNER_SYSTEM_PROMPT,
    TF_EXECUTION_PLANNER_USER_PROMPT,
)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = AgentLogger("ExecPlannerTool")


# ============================================================================
# Pydantic Output Schemas — Module Structure
# ============================================================================

class TerraformFileRecommendation(BaseModel):
    filename: str = Field(..., description="Name of the Terraform file (e.g., 'main.tf')")
    required: bool = Field(..., description="Whether this file is required")
    purpose: str = Field(..., description="Explanation of why this file is needed")
    content_description: str = Field(..., description="What content should be included")


class VariableDefinitionPlan(BaseModel):
    name: str = Field(..., description="Variable name")
    type: str = Field(..., description="Terraform variable type")
    description: str = Field(..., description="Variable description")
    default_value: Optional[Any] = Field(None, description="Default value if applicable")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules")
    sensitive: bool = Field(False, description="Whether variable is sensitive")
    justification: str = Field(..., description="Why this variable is needed")


class OutputDefinitionPlan(BaseModel):
    name: str = Field(..., description="Output name")
    description: str = Field(..., description="Output description")
    value_expression: str = Field(..., description="Terraform expression for the output")
    sensitive: bool = Field(False, description="Whether output is sensitive")
    justification: str = Field(..., description="Why this output is needed")


class ReusabilityGuidance(BaseModel):
    naming_conventions: List[str] = Field(default_factory=list)
    tagging_strategy: List[str] = Field(default_factory=list)
    composability_hints: List[str] = Field(default_factory=list)
    best_practices: List[str] = Field(default_factory=list)


class ModuleStructurePlanResponse(BaseModel):
    service_name: str = Field(..., description="AWS service this module targets")
    recommended_files: List[TerraformFileRecommendation] = Field(...)
    variable_definitions: List[VariableDefinitionPlan] = Field(default_factory=list)
    output_definitions: List[OutputDefinitionPlan] = Field(default_factory=list)
    security_considerations: List[str] = Field(default_factory=list)
    reusability_guidance: ReusabilityGuidance = Field(...)
    implementation_notes: List[str] = Field(default_factory=list)


class ModuleStructurePlanResponseList(BaseModel):
    module_structure_plans: List[ModuleStructurePlanResponse] = Field(...)


# ============================================================================
# Pydantic Output Schemas — Configuration Optimizer
# ============================================================================

class CostOptimization(BaseModel):
    resource_name: str = Field(...)
    current_configuration: Any = Field(...)
    optimized_configuration: Any = Field(...)
    estimated_savings: Optional[str] = None
    justification: str = Field(...)


class PerformanceOptimization(BaseModel):
    resource_name: str = Field(...)
    current_configuration: Any = Field(...)
    optimized_configuration: Any = Field(...)
    performance_impact: str = Field(...)
    justification: str = Field(...)


class SecurityOptimization(BaseModel):
    resource_name: str = Field(...)
    security_issue: str = Field(default="")
    current_configuration: Any = Field(...)
    secure_configuration: Any = Field(...)
    severity: str = Field(...)
    justification: str = Field(...)


class SyntaxValidation(BaseModel):
    file_name: str = Field(...)
    validation_status: str = Field(...)
    issues_found: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class NamingConvention(BaseModel):
    resource_type: str = Field(...)
    current_name: str = Field(...)
    recommended_name: str = Field(...)
    convention_rule: str = Field(...)


class TaggingStrategy(BaseModel):
    resource_name: str = Field(...)
    required_tags: Dict[str, str] = Field(default_factory=dict)
    optional_tags: Dict[str, str] = Field(default_factory=dict)
    tagging_justification: str = Field(default="")


class ConfigurationOptimizerResponse(BaseModel):
    service_name: str = Field(...)
    cost_optimizations: List[CostOptimization] = Field(default_factory=list)
    performance_optimizations: List[PerformanceOptimization] = Field(default_factory=list)
    security_optimizations: List[SecurityOptimization] = Field(default_factory=list)
    syntax_validations: List[SyntaxValidation] = Field(default_factory=list)
    naming_conventions: List[NamingConvention] = Field(default_factory=list)
    tagging_strategies: List[TaggingStrategy] = Field(default_factory=list)
    estimated_monthly_cost: Optional[str] = None
    optimization_summary: str = Field(...)
    implementation_priority: List[str] = Field(default_factory=list)


class ConfigurationOptimizationResponseList(BaseModel):
    config_optimizer_recommendations: List[ConfigurationOptimizerResponse] = Field(...)


# ============================================================================
# Pydantic Output Schemas — State Management
# ============================================================================

class InfrastructureScale(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    SHARED = "shared"


class TeamStructure(BaseModel):
    team_size: int = Field(...)
    teams: List[str] = Field(...)
    concurrent_operations: bool = Field(...)
    ci_cd_integration: bool = Field(...)


class ComplianceRequirements(BaseModel):
    encryption_required: bool = Field(True)
    audit_logging: bool = Field(False)
    backup_retention_days: Optional[int] = None
    compliance_standards: List[str] = Field(default_factory=list)


class StateManagementPlannerRequest(BaseModel):
    service_name: str = Field(...)
    infrastructure_scale: InfrastructureScale = Field(...)
    environments: List[Environment] = Field(...)
    team_structure: TeamStructure = Field(...)
    compliance_requirements: ComplianceRequirements = Field(...)
    aws_region: str = Field(...)
    multi_region: bool = Field(False)
    existing_state_files: Optional[List[str]] = None


class BackendConfiguration(BaseModel):
    bucket_name: str = Field(...)
    key_pattern: str = Field(...)
    region: str = Field(...)
    encrypt: bool = Field(...)
    versioning: bool = Field(...)
    kms_key_id: Optional[str] = None
    server_side_encryption_configuration: Dict[str, Any] = Field(...)


class StateLockingConfiguration(BaseModel):
    table_name: str = Field(...)
    billing_mode: str = Field(...)
    hash_key: str = Field("LockID")
    region: str = Field(...)
    point_in_time_recovery: bool = Field(...)
    tags: Dict[str, str] = Field(...)


class StateSplittingStrategy(BaseModel):
    splitting_approach: str = Field(...)
    state_files: List[Dict[str, str]] = Field(...)
    dependencies: List[Dict[str, Any]] = Field(...)
    data_source_usage: List[str] = Field(...)


class BackendSecurityRecommendations(BaseModel):
    iam_policies: List[Dict[str, Any]] = Field(default_factory=list)
    bucket_policies: List[Dict[str, Any]] = Field(default_factory=list)
    access_controls: List[Any] = Field(default_factory=list)
    monitoring: List[Any] = Field(default_factory=list)


class StateManagementPlannerResponse(BaseModel):
    service_name: str = Field(...)
    infrastructure_scale: str = Field(...)
    backend_configuration: BackendConfiguration = Field(...)
    state_locking_configuration: StateLockingConfiguration = Field(...)
    state_splitting_strategy: StateSplittingStrategy = Field(...)
    security_recommendations: BackendSecurityRecommendations = Field(...)
    migration_plan: Optional[List[str]] = None
    implementation_steps: List[str] = Field(...)
    best_practices: List[str] = Field(...)
    monitoring_setup: List[str] = Field(...)
    disaster_recovery: List[str] = Field(...)


class StateManagementPlanResponseList(BaseModel):
    state_management_plan_responses: List[StateManagementPlannerResponse] = Field(...)


# ============================================================================
# Pydantic Output Schemas — Execution Plan
# ============================================================================

class VariableDefinition(BaseModel):
    name: str = Field(default="")
    type: str = Field(default="string")
    description: str = Field(default="")
    default: Optional[Any] = None
    sensitive: bool = Field(False)
    nullable: bool = Field(False)
    validation_rules: List[Any] = Field(default_factory=list)
    example_values: List[Any] = Field(default_factory=list)
    justification: str = Field(default="")


class LocalValue(BaseModel):
    name: str = Field(default="")
    expression: str = Field(default="")
    description: str = Field(default="")
    depends_on: List[str] = Field(default_factory=list)
    usage_context: str = Field(default="")


class DataSource(BaseModel):
    resource_name: str = Field(...)
    data_source_type: str = Field(...)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    description: str = Field(default="")
    exported_attributes: List[str] = Field(default_factory=list)


class OutputDefinition(BaseModel):
    name: str = Field(...)
    value: str = Field(...)
    description: str = Field(default="")
    sensitive: bool = Field(False)
    depends_on: List[str] = Field(default_factory=list)
    precondition: Optional[Dict[str, str]] = None
    consumption_notes: str = Field(default="")


class IAMPolicyDocument(BaseModel):
    policy_name: str = Field(...)
    version: str = Field("2012-10-17")
    statements: List[Dict[str, Any]] = Field(default_factory=list)
    description: str = Field(default="")
    resource_references: List[str] = Field(default_factory=list)


class ResourceConfiguration(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=False)
    resource_address: Optional[str] = None
    resource_type: Optional[str] = None
    resource_name: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    depends_on: List[str] = Field(default_factory=list)
    lifecycle_rules: Optional[Dict[str, Any]] = None
    tags_strategy: Optional[str] = Field(default="")
    parameter_justification: str = Field(default="")


class TerraformFile(BaseModel):
    file_name: str = Field(...)
    file_purpose: str = Field(default="")
    resources_included: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    organization_rationale: str = Field(default="")


class ModuleExample(BaseModel):
    example_name: str = Field(default="example")
    configuration: str = Field(default="")
    description: str = Field(default="")
    expected_outputs: List[str] = Field(default_factory=list)
    use_case: str = Field(default="")


class ComprehensiveExecutionPlanResponse(BaseModel):
    service_name: str = Field(...)
    module_name: str = Field(...)
    target_environment: str = Field(...)
    plan_generation_timestamp: datetime = Field(default_factory=datetime.now)
    terraform_files: List[TerraformFile] = Field(...)
    variable_definitions: List[VariableDefinition] = Field(...)
    local_values: List[LocalValue] = Field(...)
    data_sources: List[DataSource] = Field(...)
    output_definitions: List[OutputDefinition] = Field(...)
    resource_configurations: List[ResourceConfiguration] = Field(...)
    iam_policies: List[IAMPolicyDocument] = Field(default_factory=list)
    module_description: str = Field(default="")
    usage_examples: List[ModuleExample] = Field(default_factory=list)
    readme_content: str = Field(default="")
    required_providers: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    terraform_version_constraint: str = Field(default="")
    resource_dependencies: List[Dict[str, Any]] = Field(default_factory=list)
    deployment_phases: List[str] = Field(default_factory=list)
    estimated_costs: Dict[str, Any] = Field(default_factory=dict)
    validation_and_testing: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ExecutionPlanResponseList(BaseModel):
    execution_plan_responses: List[ComprehensiveExecutionPlanResponse] = Field(...)


# ============================================================================
# Pure Helper Functions (no state, no LLM)
# ============================================================================

def extract_service_data(
    aws_service_mapping: Dict[str, Any], service_name: str
) -> Dict[str, Any]:
    """Extract service data from aws_service_mapping for a specific service."""
    try:
        if isinstance(aws_service_mapping, str):
            aws_service_mapping = json.loads(aws_service_mapping)
        for service in aws_service_mapping.get("services", []):
            if service.get("service_name") == service_name:
                return {
                    "service_name": service.get("service_name", ""),
                    "aws_service_type": service.get("aws_service_type", ""),
                    "architecture_patterns": service.get("architecture_patterns", []),
                    "well_architected_alignment": service.get("well_architected_alignment", []),
                    "terraform_resources": service.get("terraform_resources", []),
                    "dependencies": service.get("dependencies", []),
                    "cost_optimization_recommendations": service.get("cost_optimization_recommendations", []),
                }
        return {"error": f"Service type '{service_name}' not found"}
    except Exception as e:
        return {"error": str(e)}


def extract_aws_service_names(aws_service_mapping: Any) -> List[str]:
    """Extract a list of AWS service names from aws_service_mapping."""
    try:
        if isinstance(aws_service_mapping, str):
            aws_service_mapping = json.loads(aws_service_mapping)
        return [
            s.get("service_name", "")
            for s in aws_service_mapping.get("services", [])
            if s.get("service_name")
        ]
    except Exception:
        return []


def find_matching_service_name(service_name: str, service_list: List[str]) -> str:
    """Find matching service name using case-insensitive matching."""
    lower = service_name.lower()
    for full in service_list:
        if lower == full.lower():
            return full
    for full in service_list:
        if lower in full.lower():
            return full
    for full in service_list:
        for part in re.findall(r"\b\w+\b", full.lower()):
            if part in lower:
                return full
    return service_name


def extract_terraform_resource_names_and_arguments(
    terraform_attribute_mapping: Any, service_name: str
) -> List[Dict[str, Any]]:
    """Extract terraform resource names and recommended arguments for a service."""
    try:
        if isinstance(terraform_attribute_mapping, str):
            terraform_attribute_mapping = json.loads(terraform_attribute_mapping)
        if "terraform_resources" in terraform_attribute_mapping:
            return [
                {
                    "resource_name": r.get("resource_name", ""),
                    "recommended_arguments": r.get("recommended_arguments", []),
                }
                for r in terraform_attribute_mapping["terraform_resources"]
            ]
        return []
    except Exception:
        return []


def extract_terraform_resource_attributes(
    terraform_attribute_mapping: Any, service_name: str
) -> Dict[str, Any]:
    """Extract terraform resource attributes for a specific service."""
    try:
        if isinstance(terraform_attribute_mapping, str):
            terraform_attribute_mapping = json.loads(terraform_attribute_mapping)
        for service in terraform_attribute_mapping.get("services", []):
            if service.get("service_name") == service_name:
                info: Dict[str, Any] = {
                    "service_name": service.get("service_name", ""),
                    "aws_service_type": service.get("aws_service_type", ""),
                    "description": service.get("description", ""),
                    "terraform_resources": [],
                }
                for resource in service.get("terraform_resources", []):
                    r: Dict[str, Any] = {
                        "resource_name": resource.get("resource_name", ""),
                        "provider": resource.get("provider", ""),
                        "description": resource.get("description", ""),
                        "required_attributes": [
                            {"name": a.get("name", ""), "type": a.get("type", ""),
                             "required": a.get("required", True), "description": a.get("description", "")}
                            for a in resource.get("required_attributes", [])
                        ],
                        "optional_attributes": [
                            {"name": a.get("name", ""), "type": a.get("type", ""),
                             "required": a.get("required", False), "description": a.get("description", "")}
                            for a in resource.get("optional_attributes", [])
                        ],
                    }
                    md = resource.get("module_design")
                    if md and isinstance(md, dict):
                        r["recommended_arguments"] = md.get("recommended_arguments", [])
                        r["recommended_outputs"] = md.get("recommended_outputs", [])
                    else:
                        r["recommended_arguments"] = []
                    info["terraform_resources"].append(r)
                return info
        return {"error": f"Service type '{service_name}' not found"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Upstream Context Extraction — Reuse prior agent outputs (token efficient)
# ============================================================================

def extract_upstream_context(
    req_output: Dict[str, Any],
    sec_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract and normalize deployment context, security, and best practices
    from prior agent outputs. This avoids re-generating data that upstream
    agents (req_analyser, sec_n_best_practices) already computed.

    Returns a structured dict with:
      - deployment: region, environment, business_requirements, technical_specs
      - security: per-service and cross-service findings
      - best_practices: per-service and cross-service findings
      - compliance: frameworks, risk level, overall status
      - cost_recs: upstream cost optimization recommendations
    """
    # ── Deployment context from analysis_results ──────────────────────
    analysis = req_output.get("analysis_results", {})
    deployment_ctx = analysis.get("deployment_context", "")
    # Parse "Region: ap-south-1, Environment: production" format
    region = "us-east-1"  # default
    environment = "production"  # default
    if isinstance(deployment_ctx, str):
        for part in deployment_ctx.split(","):
            part = part.strip()
            if part.lower().startswith("region:"):
                region = part.split(":", 1)[1].strip()
            elif part.lower().startswith("environment:"):
                environment = part.split(":", 1)[1].strip()

    services_reqs = analysis.get("services", [])
    scope = analysis.get("scope_classification", "single_service")

    # ── Security analysis ────────────────────────────────────────────
    security = sec_output.get("security_analysis", {})
    overall_compliance = security.get("overall_compliance", "unknown")
    overall_risk = security.get("overall_summary", {}).get("overall_risk_level", "medium")
    summary_issues = security.get("summary_issues", [])
    cross_sec = security.get("cross_service_analysis", {})

    # ── Best practices analysis ──────────────────────────────────────
    best_practices = sec_output.get("best_practices_analysis", {})
    bp_overall_status = best_practices.get("overall_status", "PASS")
    bp_cross = best_practices.get("cross_service_analysis", {})

    # ── Compliance frameworks detected ───────────────────────────────
    compliance_standards = []
    for svc_sec in security.get("services", []):
        for rec in svc_sec.get("service_recommendations", []):
            for std in ["SOC 2", "ISO 27001", "HIPAA", "PCI DSS", "CIS"]:
                if std.lower() in rec.lower() and std not in compliance_standards:
                    compliance_standards.append(std)
    if not compliance_standards:
        compliance_standards = ["CIS", "SOC2"]

    return {
        "deployment": {
            "region": region,
            "environment": environment,
            "scope": scope,
            "deployment_context": deployment_ctx,
            "services_requirements": services_reqs,
        },
        "security": {
            "overall_compliance": overall_compliance,
            "overall_risk_level": overall_risk,
            "summary_issues": summary_issues,
            "cross_service_analysis": cross_sec,
            "services": security.get("services", []),
            "encryption_at_rest": security.get("encryption_at_rest", {}),
            "encryption_in_transit": security.get("encryption_in_transit", {}),
            "network_security": security.get("network_security", {}),
            "access_controls": security.get("access_controls", {}),
        },
        "best_practices": {
            "overall_status": bp_overall_status,
            "cross_service_analysis": bp_cross,
            "services": best_practices.get("services", []),
            "naming_and_tagging": best_practices.get("naming_and_tagging", []),
            "module_structure": best_practices.get("module_structure", []),
            "resource_optimization": best_practices.get("resource_optimization", []),
            "terraform_practices": best_practices.get("terraform_practices", []),
        },
        "compliance": {
            "standards": compliance_standards,
            "risk_level": overall_risk,
            "overall_status": overall_compliance,
        },
    }


def extract_service_security_context(
    upstream: Dict[str, Any], service_name: str
) -> Dict[str, Any]:
    """Extract per-service security + best practices data from upstream context.
    Returns a compact dict ready for prompt injection."""
    # ── Per-service security ──────────────────────────────────────────
    svc_security: Dict[str, Any] = {}
    for svc in upstream.get("security", {}).get("services", []):
        if svc.get("service_name", "").lower() == service_name.lower():
            svc_security = {
                "encryption_at_rest": svc.get("encryption_at_rest", {}),
                "encryption_in_transit": svc.get("encryption_in_transit", {}),
                "network_security": svc.get("network_security", {}),
                "access_controls": svc.get("access_controls", {}),
                "compliance_status": svc.get("service_compliance", "unknown"),
                "issues": svc.get("service_issues", []),
                "recommendations": svc.get("service_recommendations", []),
            }
            break

    # Fallback to overall security if no per-service match
    if not svc_security:
        sec = upstream.get("security", {})
        svc_security = {
            "encryption_at_rest": sec.get("encryption_at_rest", {}),
            "encryption_in_transit": sec.get("encryption_in_transit", {}),
            "network_security": sec.get("network_security", {}),
            "access_controls": sec.get("access_controls", {}),
            "compliance_status": sec.get("overall_compliance", "unknown"),
            "issues": sec.get("summary_issues", []),
            "recommendations": [],
        }

    # ── Per-service best practices ────────────────────────────────────
    svc_bp: Dict[str, Any] = {}
    for svc in upstream.get("best_practices", {}).get("services", []):
        if svc.get("service_name", "").lower() == service_name.lower():
            svc_bp = {
                "naming_and_tagging": svc.get("naming_and_tagging", []),
                "module_structure": svc.get("module_structure", []),
                "resource_optimization": svc.get("resource_optimization", []),
                "terraform_practices": svc.get("terraform_practices", []),
                "service_score": svc.get("service_score", 0),
            }
            break

    # Fallback to overall best practices if no per-service match
    if not svc_bp:
        bp = upstream.get("best_practices", {})
        svc_bp = {
            "naming_and_tagging": bp.get("naming_and_tagging", []),
            "module_structure": bp.get("module_structure", []),
            "resource_optimization": bp.get("resource_optimization", []),
            "terraform_practices": bp.get("terraform_practices", []),
            "service_score": 0,
        }

    # ── Per-service business requirements ─────────────────────────────
    svc_reqs: Dict[str, Any] = {}
    for svc in upstream.get("deployment", {}).get("services_requirements", []):
        if svc.get("service_name", "").lower() == service_name.lower():
            svc_reqs = {
                "business_requirements": svc.get("business_requirements", {}),
                "technical_specifications": svc.get("technical_specifications", {}),
            }
            break

    return {
        "security": svc_security,
        "best_practices": svc_bp,
        "requirements": svc_reqs,
        "deployment": upstream.get("deployment", {}),
        "compliance": upstream.get("compliance", {}),
    }


def _build_upstream_prompt_section(
    svc_ctx: Dict[str, Any],
    service_data: Dict[str, Any],
) -> str:
    """Build a compact prompt section from upstream context for a service.
    This replaces re-generation of security, compliance, naming, and cost data."""
    sections: List[str] = []

    # ── Deployment Context ────────────────────────────────────────────
    deploy = svc_ctx.get("deployment", {})
    if deploy:
        sections.append(
            "\n### DEPLOYMENT CONTEXT (from Requirements Analysis — DO NOT regenerate)\n"
            f"Region: {deploy.get('region', 'us-east-1')}\n"
            f"Environment: {deploy.get('environment', 'production')}\n"
            f"Scope: {deploy.get('scope', 'single_service')}\n"
        )
        reqs = svc_ctx.get("requirements", {})
        if reqs.get("business_requirements"):
            sections.append(
                f"Business Requirements: {json.dumps(reqs['business_requirements'])}\n"
            )
        if reqs.get("technical_specifications"):
            sections.append(
                f"Technical Specifications: {json.dumps(reqs['technical_specifications'])}\n"
            )

    # ── Security & Compliance Context ─────────────────────────────────
    sec = svc_ctx.get("security", {})
    if sec:
        sections.append(
            "\n### SECURITY & COMPLIANCE FINDINGS (from Security Analysis — reuse, DO NOT regenerate)\n"
            f"Compliance Status: {sec.get('compliance_status', 'unknown')}\n"
        )
        if sec.get("issues"):
            sections.append("Security Issues Identified:\n")
            for issue in sec["issues"][:10]:  # Cap to save tokens
                sections.append(f"  - {issue}\n")
        if sec.get("recommendations"):
            sections.append("Security Recommendations:\n")
            for rec in sec["recommendations"][:10]:
                sections.append(f"  - {rec}\n")

        # Encryption findings
        enc_rest = sec.get("encryption_at_rest", {})
        enc_transit = sec.get("encryption_in_transit", {})
        if enc_rest:
            sections.append(
                f"Encryption at Rest: status={enc_rest.get('status', 'unknown')}, "
                f"recommendations={enc_rest.get('recommendations', [])}\n"
            )
        if enc_transit:
            sections.append(
                f"Encryption in Transit: status={enc_transit.get('status', 'unknown')}, "
                f"recommendations={enc_transit.get('recommendations', [])}\n"
            )

        # IAM / Access control findings
        ac = sec.get("access_controls", {})
        if ac:
            for key in ["iam_roles", "iam_policies"]:
                detail = ac.get(key, {})
                if detail:
                    sections.append(
                        f"{key}: status={detail.get('status', 'unknown')}, "
                        f"recommendations={detail.get('recommendations', [])}\n"
                    )

        # Network security findings
        net = sec.get("network_security", {})
        if net:
            for key in ["security_groups", "network_acls"]:
                detail = net.get(key, {})
                if detail:
                    sections.append(
                        f"{key}: status={detail.get('status', 'unknown')}, "
                        f"recommendations={detail.get('recommendations', [])}\n"
                    )

    # ── Best Practices Context ────────────────────────────────────────
    bp = svc_ctx.get("best_practices", {})
    if bp:
        sections.append(
            "\n### BEST PRACTICES FINDINGS (from Best Practices Analysis — reuse, DO NOT regenerate)\n"
            f"Service Score: {bp.get('service_score', 'N/A')}/100\n"
        )
        for category in ["naming_and_tagging", "module_structure", "resource_optimization", "terraform_practices"]:
            findings = bp.get(category, [])
            if findings:
                sections.append(f"{category.replace('_', ' ').title()}:\n")
                for f in findings[:5]:  # Cap per category
                    if isinstance(f, dict):
                        sections.append(
                            f"  - [{f.get('status', '?')}] {f.get('check', '')} → {f.get('recommendation', '')}\n"
                        )
                    else:
                        sections.append(f"  - {f}\n")

    # ── Cost Optimization (already computed upstream) ─────────────────
    cost_recs = service_data.get("cost_optimization_recommendations", [])
    if cost_recs:
        sections.append(
            "\n### COST OPTIMIZATION (from Service Discovery — reuse, DO NOT regenerate)\n"
        )
        for rec in cost_recs:
            if isinstance(rec, dict):
                sections.append(
                    f"  - [{rec.get('category', '')}] {rec.get('recommendation', '')} "
                    f"(savings: {rec.get('potential_savings', 'N/A')}, "
                    f"difficulty: {rec.get('implementation_difficulty', 'N/A')})\n"
                )
            else:
                sections.append(f"  - {rec}\n")

    # ── Compliance Standards ──────────────────────────────────────────
    compliance = svc_ctx.get("compliance", {})
    if compliance.get("standards"):
        sections.append(
            f"\nCompliance Standards: {', '.join(compliance['standards'])}\n"
            f"Risk Level: {compliance.get('risk_level', 'medium')}\n"
        )

    return "".join(sections)


# ============================================================================
# Inner Async Helpers (per-service processing, called by tools)
# ============================================================================

async def _create_module_structure_plan(
    service_name: str,
    aws_service_mapping: Dict[str, Any],
    tf_attribute_mapping: Dict[str, Any],
    runnable_config: RunnableConfig,
    additional_feedback: Optional[str] = None,
    upstream_context: Optional[Dict[str, Any]] = None,
) -> ModuleStructurePlanResponse:
    """Create module structure plan for a single service.

    Args:
        upstream_context: Normalized upstream data from extract_upstream_context().
            Contains security findings, best practices, and deployment context
            so the LLM does not regenerate them — saving tokens and improving
            consistency.
    """
    from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.utils.mcp_terraform_helpers import (
        _extract_top_module_id,
        _truncate_module_details,
    )
    service_data = extract_service_data(aws_service_mapping, service_name)
    terraform_resource_attributes = extract_terraform_resource_attributes(tf_attribute_mapping, service_name)

    svc_name = service_data["service_name"]
    architecture_patterns = service_data.get("architecture_patterns", [])
    well_architected = service_data.get("well_architected_alignment", [])
    deps = service_data.get("dependencies", [])
    tf_resources = terraform_resource_attributes.get("terraform_resources", [])

    # ── Fetch community module reference via MCP ─────────────────────
    module_reference = ""
    try:
        config = Config()
        mcp = MCPClient(config, server_filter=["terraform"])
        async with mcp.connect():
            # Search for a community module matching this service
            module_search = await mcp.execute_tool(
                server_name="terraform",
                tool_name="search_modules",
                arguments={"module_query": f"aws {svc_name}"},
            )

            # Get the top module's details
            module_id = _extract_top_module_id(str(module_search))
            if module_id:
                module_details = await mcp.execute_tool(
                    server_name="terraform",
                    tool_name="get_module_details",
                    arguments={"module_id": module_id},
                )

                truncated = _truncate_module_details(
                    str(module_details), max_chars=4000
                )

                module_reference = (
                    "\n\n## Reference: Community Module (from Terraform Registry)\n\n"
                    f"**Module**: `{module_id}`\n\n"
                    "The following is a well-maintained community module for "
                    "this service. Use its VARIABLE NAMING CONVENTIONS, "
                    "TYPE DEFINITIONS, DESCRIPTIONS, and DEFAULT VALUES "
                    "as a reference. Your module design should follow similar "
                    "patterns — not copy them exactly:\n\n"
                    f"{truncated}\n\n"
                    "---\n"
                    "When designing your module structure, align your variable "
                    "names and output names with the patterns shown above.\n"
                )

                logger.info(
                    f"Module reference fetched for {svc_name}",
                    extra={"module_id": module_id},
                )
            else:
                logger.debug(
                    f"No community module found for {svc_name}",
                )

    except Exception as mcp_err:
        err_str = str(mcp_err)
        if "no modules found" in err_str.lower() or "unmarshalling" in err_str.lower() or "not found" in err_str.lower():
            logger.warning(
                "Module reference fetch yielded no results",
                extra={"service": svc_name, "error": err_str},
            )
        else:
            logger.error(
                "Module reference fetch failed",
                extra={"service": svc_name, "error": err_str},
            )
            raise ValueError(f"MCP server connection failed: {err_str}")

    formatted_user = TF_MODULE_STRUCTURE_PLAN_USER_PROMPT.format(
        service_name=svc_name,
        architecture_patterns=architecture_patterns,
        well_architected_alignment=well_architected,
        terraform_resources=tf_resources,
        module_dependencies=deps,
    )
    
    # inject module_reference into the user prompt
    formatted_user = module_reference + formatted_user

    # ── Inject upstream context (security + best practices + deployment) ──
    if upstream_context:
        svc_ctx = extract_service_security_context(upstream_context, svc_name)
        upstream_section = _build_upstream_prompt_section(svc_ctx, service_data)
        if upstream_section:
            formatted_user = formatted_user + (
                "\n\n## UPSTREAM ANALYSIS RESULTS (Pre-computed — DO NOT regenerate this data)\n"
                "The following findings were computed by upstream agents (Requirements Analyser, "
                "Security & Best Practices). Use them DIRECTLY in your module design:\n"
                "- Security findings → drive your `security_considerations` array\n"
                "- Best practices → drive your `reusability_guidance` and file structure\n"
                "- Deployment context → drive your variable defaults and naming\n"
                "- Cost optimizations → incorporate into implementation_notes\n"
                + upstream_section
            )

    # inject additional_feedback if provided
    feedback_str = additional_feedback or "None"
    feedback_section = (
        "\n\n### ADDITIONAL FEEDBACK (Human Clarification)\n"
        "---\n"
        f"{feedback_str}\n"
        "---\n"
        "If additional feedback is provided (not 'None'), incorporate "
        "those preferences (naming conventions, module layout, CI/CD "
        "requirements) as authoritative decisions into the module design.\n"
    )
    formatted_user = formatted_user + feedback_section

    formatted_user = formatted_user.replace("{", "{{").replace("}", "}}")
    escaped_system = TF_MODULE_STRUCTURE_PLAN_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")

    config = Config()
    llm = initialize_llm_higher(config.get_llm_higher_config())
    parser = PydanticOutputParser(pydantic_object=ModuleStructurePlanResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", escaped_system),
        ("user", formatted_user),
        ("user", "Respond with valid JSON matching the ModuleStructurePlanResponse schema.\n\n{format_instructions}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return await chain.ainvoke({}, config=runnable_config)


async def _create_configuration_optimizer(
    plan: ModuleStructurePlanResponse,
    runnable_config: RunnableConfig,
    upstream_context: Optional[Dict[str, Any]] = None,
) -> ConfigurationOptimizerResponse:
    """Create configuration optimisation for a single module plan.

    Args:
        upstream_context: Normalized upstream data. Used to populate environment,
            compliance_requirements, and optimization_targets from real analysis
            instead of hardcoded config defaults — saving tokens and improving
            accuracy.
    """
    config = Config()
    llm = initialize_llm_model(config.get_llm_config())
    parser = PydanticOutputParser(pydantic_object=ConfigurationOptimizerResponse)

    # ── Derive real values from upstream context if available ─────────
    if upstream_context:
        deploy = upstream_context.get("deployment", {})
        compliance = upstream_context.get("compliance", {})
        environment = deploy.get("environment", "production")
        compliance_reqs = compliance.get("standards", ["CIS", "SOC2"])
        risk_level = compliance.get("risk_level", "medium")
        # Drive optimization targets from risk level
        if risk_level in ("high", "critical"):
            opt_targets = ["security", "reliability", "compliance"]
        else:
            opt_targets = ["security", "reliability", "cost"]
        # Extract expected load from service requirements
        expected_load = "standard"
        for svc in deploy.get("services_requirements", []):
            if svc.get("service_name", "").lower() == plan.service_name.lower():
                biz = svc.get("business_requirements", {})
                if "traffic_requirement" in biz:
                    expected_load = biz["traffic_requirement"]
                elif "availability" in biz:
                    expected_load = biz["availability"]
                break
    else:
        environment = getattr(config, "CONFIG_OPTIMIZER_ENVIRONMENT", "production")
        compliance_reqs = getattr(config, "CONFIG_OPTIMIZER_COMPLIANCE_REQUIREMENTS", [])
        opt_targets = getattr(config, "CONFIG_OPTIMIZER_OPTIMIZATION_TARGETS", ["security", "reliability"])
        expected_load = getattr(config, "CONFIG_OPTIMIZER_EXPECTED_LOAD", "standard")

    formatted_user = TF_CONFIGURATION_OPTIMIZER_USER_PROMPT.format(
        service_name=plan.service_name,
        recommended_files=json.dumps([f.model_dump(mode="json") for f in plan.recommended_files]),
        variable_definitions=json.dumps([v.model_dump(mode="json") for v in plan.variable_definitions]),
        output_definitions=json.dumps([o.model_dump(mode="json") for o in plan.output_definitions]),
        security_considerations=json.dumps(plan.security_considerations),
        environment=environment,
        expected_load=expected_load,
        budget_constraints=getattr(config, "CONFIG_OPTIMIZER_BUDGET_CONSTRAINTS", "standard"),
        compliance_requirements=json.dumps(compliance_reqs),
        optimization_targets=json.dumps(opt_targets),
        organization_standards=json.dumps(getattr(config, "CONFIG_OPTIMIZER_ORGANIZATION_STANDARDS", [])),
    )

    # ── Inject upstream security + best practices into optimizer prompt ──
    if upstream_context:
        svc_ctx = extract_service_security_context(upstream_context, plan.service_name)
        sec = svc_ctx.get("security", {})
        bp = svc_ctx.get("best_practices", {})

        upstream_section = (
            "\n\n## UPSTREAM ANALYSIS RESULTS (Pre-computed — DO NOT regenerate)\n"
            "Use these findings to ground your optimizations instead of inventing generic advice.\n"
        )

        # Security-driven optimizations
        if sec.get("recommendations"):
            upstream_section += "\n### Security Recommendations (from Security Analysis)\n"
            for rec in sec["recommendations"][:8]:
                upstream_section += f"  - {rec}\n"

        if sec.get("issues"):
            upstream_section += "\n### Security Issues to Address\n"
            for issue in sec["issues"][:8]:
                upstream_section += f"  - {issue}\n"

        # Best practices-driven naming/tagging
        if bp.get("naming_and_tagging"):
            upstream_section += "\n### Naming & Tagging Findings (from Best Practices)\n"
            for finding in bp["naming_and_tagging"][:5]:
                if isinstance(finding, dict):
                    upstream_section += (
                        f"  - [{finding.get('status', '?')}] {finding.get('check', '')} "
                        f"→ {finding.get('recommendation', '')}\n"
                    )

        if bp.get("resource_optimization"):
            upstream_section += "\n### Resource Optimization Findings (from Best Practices)\n"
            for finding in bp["resource_optimization"][:5]:
                if isinstance(finding, dict):
                    upstream_section += (
                        f"  - [{finding.get('status', '?')}] {finding.get('check', '')} "
                        f"→ {finding.get('recommendation', '')}\n"
                    )

        formatted_user = formatted_user + upstream_section

    formatted_user = formatted_user.replace("{", "{{").replace("}", "}}")
    escaped_system = TF_CONFIGURATION_OPTIMIZER_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", escaped_system),
        ("user", formatted_user),
        ("user", "Respond with valid JSON matching the ConfigurationOptimizerResponse schema.\n\n{format_instructions}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return await chain.ainvoke({}, config=runnable_config)


async def _create_state_mgmt(
    optimizer: ConfigurationOptimizerResponse,
    runnable_config: RunnableConfig,
    upstream_context: Optional[Dict[str, Any]] = None,
) -> StateManagementPlannerResponse:
    """Create state management plan for a single service.

    Args:
        upstream_context: Normalized upstream data. Used to populate aws_region,
            environments, compliance_standards, and encryption_required from
            real analysis instead of hardcoded generic defaults.
    """
    config = Config()
    llm = initialize_llm_higher(config.get_llm_higher_config())
    parser = PydanticOutputParser(pydantic_object=StateManagementPlannerResponse)

    service_name = optimizer.service_name if hasattr(optimizer, "service_name") else "unknown"

    # ── Derive real values from upstream context if available ─────────
    if upstream_context:
        deploy = upstream_context.get("deployment", {})
        compliance = upstream_context.get("compliance", {})
        sec = upstream_context.get("security", {})
        aws_region = deploy.get("region", "us-east-1")
        environment = deploy.get("environment", "production")
        environments = ["dev", "staging", environment] if environment not in ["dev", "staging"] else ["dev", "staging", "production"]
        compliance_standards = compliance.get("standards", ["CIS", "SOC2"])
        # Determine encryption requirement from security analysis
        enc_rest = sec.get("encryption_at_rest", {})
        encryption_required = True  # Always true for state files
        if enc_rest.get("status") == "non_compliant":
            encryption_required = True  # Enforce encryption if non-compliant
        # Determine infrastructure scale from scope
        scope = deploy.get("scope", "single_service")
        if scope == "full_application_stack":
            infra_scale = "enterprise"
        elif scope == "multi_service":
            infra_scale = "large"
        else:
            infra_scale = "medium"
    else:
        aws_region = getattr(config, "STATE_MGMT_DEFAULT_AWS_REGION", "us-east-1")
        environments = getattr(config, "STATE_MGMT_DEFAULT_ENVIRONMENTS", ["dev", "prod"])
        compliance_standards = getattr(config, "STATE_MGMT_DEFAULT_COMPLIANCE_STANDARDS", ["CIS", "SOC2"])
        encryption_required = getattr(config, "STATE_MGMT_DEFAULT_ENCRYPTION_REQUIRED", True)
        infra_scale = getattr(config, "STATE_MGMT_DEFAULT_INFRASTRUCTURE_SCALE", "medium")

    formatted_user = TF_STATE_MGMT_USER_PROMPT.format(
        service_name=service_name,
        infrastructure_scale=infra_scale,
        environments=environments,
        aws_region=aws_region,
        multi_region=getattr(config, "STATE_MGMT_DEFAULT_MULTI_REGION", False),
        team_size=getattr(config, "STATE_MGMT_DEFAULT_TEAM_SIZE", "medium"),
        teams=getattr(config, "STATE_MGMT_DEFAULT_TEAMS", ["platform", "security"]),
        concurrent_operations=getattr(config, "STATE_MGMT_DEFAULT_CONCURRENT_OPERATIONS", "medium"),
        ci_cd_integration=getattr(config, "STATE_MGMT_DEFAULT_CI_CD_INTEGRATION", "standard"),
        encryption_required=encryption_required,
        audit_logging=getattr(config, "STATE_MGMT_DEFAULT_AUDIT_LOGGING", True),
        backup_retention_days=getattr(config, "STATE_MGMT_DEFAULT_BACKUP_RETENTION_DAYS", 30),
        compliance_standards=compliance_standards,
        existing_state_files="[]",
    )
    formatted_user = formatted_user.replace("{", "{{").replace("}", "}}")
    escaped_system = TF_STATE_MGMT_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", escaped_system),
        ("user", formatted_user),
        ("user", "Respond with valid JSON matching the StateManagementPlannerResponse schema.\n\n{format_instructions}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return await chain.ainvoke({}, config=runnable_config)


async def _create_execution_plan(
    state_mgmt_plan: StateManagementPlannerResponse,
    exec_output: Dict[str, Any],
    aws_service_mapping: Dict[str, Any],
    tf_attribute_mapping: Dict[str, Any],
    req_output: Dict[str, Any],
    runnable_config: RunnableConfig,
    upstream_context: Optional[Dict[str, Any]] = None,
) -> ComprehensiveExecutionPlanResponse:
    """Create comprehensive execution plan for a single service.

    Args:
        upstream_context: Normalized upstream data. Used to inject real
            security/IAM findings and best practices directly into the
            execution blueprint so the LLM produces grounded, consistent
            plans instead of inventing its own.
    """
    config = Config()
    llm = initialize_llm_higher(config.get_llm_higher_config())
    parser = PydanticOutputParser(pydantic_object=ComprehensiveExecutionPlanResponse)

    target_service_name = state_mgmt_plan.service_name
    service_list = extract_aws_service_names(aws_service_mapping)
    canonical_target = find_matching_service_name(target_service_name, service_list)

    provider_version = req_output.get("provider_version")
    if not provider_version:
        provider_version = "6.0"
        try:
            mcp = MCPClient(config, server_filter=["terraform"])
            async with mcp.connect():
                fetched = await mcp.execute_tool(
                    server_name="terraform",
                    tool_name="get_latest_provider_version",
                    arguments={"namespace": "hashicorp", "name": "aws"},
                )
                if fetched:
                    provider_version = str(fetched).strip()
        except Exception as e:
            logger.error("Failed to fetch provider version", extra={"error": str(e)})
            raise ValueError(f"MCP server connection failed: {str(e)}")

    # Find matching module structure plan using canonical names
    module_plans = exec_output.get("module_structure_plans", [])
    matching_module = None
    for p in module_plans:
        p_name = p.get("service_name") if isinstance(p, dict) else getattr(p, "service_name", "unknown")
        if find_matching_service_name(p_name or "", service_list) == canonical_target:
            matching_module = p
            break

    if not matching_module:
        raise ValueError(f"No module structure plan found for service: {target_service_name} (canonical: {canonical_target})")
    if not isinstance(matching_module, dict):
        matching_module = matching_module.model_dump(mode="json") if hasattr(matching_module, "model_dump") else matching_module

    # Find matching config optimizer using canonical names
    config_optimizers = exec_output.get("configuration_optimizers", [])
    matching_config = None
    for c in config_optimizers:
        c_name = c.get("service_name") if isinstance(c, dict) else getattr(c, "service_name", "unknown")
        if find_matching_service_name(c_name or "", service_list) == canonical_target:
            matching_config = c
            break

    if not matching_config:
        raise ValueError(f"No config optimizer data found for service: {target_service_name} (canonical: {canonical_target})")
    if not isinstance(matching_config, dict):
        matching_config = matching_config.model_dump(mode="json") if hasattr(matching_config, "model_dump") else matching_config

    service_name = matching_module.get("service_name", "")
    matched_service = find_matching_service_name(service_name, service_list)
    tf_attrs = extract_terraform_resource_attributes(tf_attribute_mapping, matched_service)
    tf_res_args = extract_terraform_resource_names_and_arguments(tf_attrs, matched_service)

    # ── Derive real environment from upstream context ─────────────────
    target_env = "prod"
    ci_cd = "GitHub Actions"
    if upstream_context:
        deploy = upstream_context.get("deployment", {})
        target_env = deploy.get("environment", "prod")
        # Keep ci_cd from config if available

    formatted_user = TF_EXECUTION_PLANNER_USER_PROMPT.format(
        service_name=service_name,
        terraform_resource_names_and_arguments=tf_res_args,
        recommended_files=json.dumps(matching_module.get("recommended_files", [])),
        variable_definitions=json.dumps(matching_module.get("variable_definitions", [])),
        output_definitions=json.dumps(matching_module.get("output_definitions", [])),
        security_considerations=json.dumps(matching_module.get("security_considerations", [])),
        cost_optimizations=json.dumps(matching_config.get("cost_optimizations", [])),
        performance_optimizations=json.dumps(matching_config.get("performance_optimizations", [])),
        security_optimizations=json.dumps(matching_config.get("security_optimizations", [])),
        naming_conventions=json.dumps(matching_config.get("naming_conventions", [])),
        tagging_strategies=json.dumps(matching_config.get("tagging_strategies", [])),
        backend_configuration=json.dumps(
            state_mgmt_plan.backend_configuration.model_dump(mode="json")
            if hasattr(state_mgmt_plan.backend_configuration, "model_dump")
            else state_mgmt_plan.backend_configuration
        ),
        state_locking_configuration=json.dumps(
            state_mgmt_plan.state_locking_configuration.model_dump(mode="json")
            if hasattr(state_mgmt_plan.state_locking_configuration, "model_dump")
            else state_mgmt_plan.state_locking_configuration
        ),
        state_splitting_strategy=json.dumps(
            state_mgmt_plan.state_splitting_strategy.model_dump(mode="json")
            if hasattr(state_mgmt_plan.state_splitting_strategy, "model_dump")
            else state_mgmt_plan.state_splitting_strategy
        ),
        target_environment=target_env,
        ci_cd_integration=ci_cd,
        parallel_execution="enabled",
    )
    version_directive = (
        f"\n\nIMPORTANT: Use the following exact provider version in "
        f"required_providers block:\n"
        f'  hashicorp/aws = "~> {provider_version}"\n'
        f"Do NOT use any other version.\n"
    )
    formatted_user = formatted_user + version_directive

    # ── Inject upstream security + best practices for IAM grounding ──
    if upstream_context:
        svc_ctx = extract_service_security_context(upstream_context, service_name)
        sec = svc_ctx.get("security", {})
        bp = svc_ctx.get("best_practices", {})

        upstream_section = (
            "\n\n## UPSTREAM ANALYSIS RESULTS (Pre-computed — use for IAM & security grounding)\n"
            "The following data comes from the upstream Security & Best Practices agents.\n"
            "Use it to ground your `iam_policies`, `destructive_operations`, and `local_values`.\n"
        )

        # IAM/Access Control findings → ground iam_policies
        ac = sec.get("access_controls", {})
        if ac:
            upstream_section += "\n### IAM & Access Control (from Security Analysis)\n"
            for key in ["iam_roles", "iam_policies"]:
                detail = ac.get(key, {})
                if detail:
                    upstream_section += f"  {key}: status={detail.get('status', 'unknown')}\n"
                    for rec in detail.get("recommendations", [])[:5]:
                        upstream_section += f"    - {rec}\n"

        # Encryption findings → ensure resources use proper encryption
        enc_rest = sec.get("encryption_at_rest", {})
        enc_transit = sec.get("encryption_in_transit", {})
        if enc_rest or enc_transit:
            upstream_section += "\n### Encryption Requirements (from Security Analysis)\n"
            if enc_rest:
                upstream_section += f"  At Rest: status={enc_rest.get('status', 'unknown')}\n"
                for rec in enc_rest.get("recommendations", [])[:3]:
                    upstream_section += f"    - {rec}\n"
            if enc_transit:
                upstream_section += f"  In Transit: status={enc_transit.get('status', 'unknown')}\n"
                for rec in enc_transit.get("recommendations", [])[:3]:
                    upstream_section += f"    - {rec}\n"

        # Network security findings
        net = sec.get("network_security", {})
        if net:
            upstream_section += "\n### Network Security (from Security Analysis)\n"
            for key in ["security_groups", "network_acls"]:
                detail = net.get(key, {})
                if detail:
                    upstream_section += f"  {key}: status={detail.get('status', 'unknown')}\n"
                    for rec in detail.get("recommendations", [])[:3]:
                        upstream_section += f"    - {rec}\n"

        # Terraform practices findings → inform execution decisions
        if bp.get("terraform_practices"):
            upstream_section += "\n### Terraform Practice Findings (from Best Practices)\n"
            for finding in bp["terraform_practices"][:5]:
                if isinstance(finding, dict):
                    upstream_section += (
                        f"  - [{finding.get('status', '?')}] {finding.get('check', '')} "
                        f"→ {finding.get('recommendation', '')}\n"
                    )

        formatted_user = formatted_user + upstream_section

    formatted_user = formatted_user.replace("{", "{{").replace("}", "}}")
    escaped_system = TF_EXECUTION_PLANNER_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", escaped_system),
        ("user", formatted_user),
        ("user", """Respond with valid JSON matching the ComprehensiveExecutionPlanResponse schema.

CRITICAL: Use ALL terraform_resource_names_and_arguments, variable_definitions, and optimisations.

{format_instructions}"""),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return await chain.ainvoke({}, config=runnable_config)


# ============================================================================
# Tool 1 — Module Structure Plan
# ============================================================================

@tool
async def create_module_structure_plan_tool(
    runtime: ToolRuntime[None, TFPlannerState],
    runnable_config: RunnableConfig,
    additional_feedback: Optional[str] = None,
) -> Command:
    """
    Create comprehensive Terraform module structure plans for all AWS services.

    Reads ``req_analyser_output`` from state for ``aws_service_mapping`` and
    ``terraform_attribute_mapping``.  Iterates through each service and
    generates detailed module file structure, variables, outputs, and
    implementation guidance.

    Args:
        additional_feedback: Optional human-clarified context from HITL.
            When the agent received feedback via request_human_input(),
            it should pass that feedback here so the module structure
            planner can incorporate explicit naming conventions, module
            layout preferences, or CI/CD integration requirements.
    """
    try:
        # ── Read inputs from state ───────────────────────────────────────
        req_output = runtime.state.get("req_analyser_output", {}) or {}
        sec_output = runtime.state.get("sec_n_best_practices_output", {}) or {}
        aws_service_mapping = req_output.get("aws_service_mapping", {})
        tf_attribute_mapping = req_output.get("terraform_attribute_mapping", {})

        if not aws_service_mapping:
            raise ValueError("aws_service_mapping missing — run req_analyser tools first")

        service_list = extract_aws_service_names(aws_service_mapping)
        if not service_list:
            raise ValueError("No service names found in aws_service_mapping")

        # ── Extract upstream context (reuse instead of regenerate) ───────
        upstream_context = extract_upstream_context(req_output, sec_output)
        has_upstream = bool(sec_output.get("security_analysis") or sec_output.get("best_practices_analysis"))

        logger.info(
            "Starting module structure planning",
            extra={
                "services": service_list,
                "has_additional_feedback": additional_feedback is not None,
                "has_upstream_context": has_upstream,
            },
        )

        # ── Process each service ─────────────────────────────────────────
        results: List[ModuleStructurePlanResponse] = []
        for svc in service_list:
            try:
                result = await _create_module_structure_plan(
                    svc, aws_service_mapping, tf_attribute_mapping,
                    runnable_config, additional_feedback=additional_feedback,
                    upstream_context=upstream_context if has_upstream else None,
                )
                results.append(result)
            except Exception as e:
                if "MCP" in str(e) or "TaskGroup" in str(e) or "Docker" in str(e):
                    raise
                logger.error(f"Module structure plan failed for {svc}", extra={"error": str(e)})
                results.append(ModuleStructurePlanResponse(
                    service_name=svc,
                    recommended_files=[], variable_definitions=[], output_definitions=[],
                    security_considerations=[],
                    reusability_guidance=ReusabilityGuidance(),
                    implementation_notes=[f"Failed: {e}"],
                ))

        # ── State update ─────────────────────────────────────────────────
        plans_list = [r.model_dump(mode="json") for r in results]
        current_output = dict(runtime.state.get("execution_planner_output", {}) or {})
        current_output["module_structure_plans"] = plans_list
        current_output["module_structure_complete"] = True

        logger.info("Module structure planning completed", extra={"total": len(results)})

        return Command(update={
            "execution_planner_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"Module structure planning completed: {len(results)} service(s). "
                        "Proceed to create_configuration_optimizations_tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error("create_module_structure_plan_tool failed", extra={"error": str(exc)})
        return Command(update={
            "messages": [
                ToolMessage(
                    content=(
                        f"Module structure planning failed: {exc}. "
                        "Do NOT proceed to the next step. You MUST call request_human_input to inform the user about this "
                        "connection error (e.g. check if Docker is running) and ask them to fix it. "
                        "Wait for their response, then retry this tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })


# ============================================================================
# Tool 2 — Configuration Optimisations
# ============================================================================

@tool
async def create_configuration_optimizations_tool(
    runtime: ToolRuntime[None, TFPlannerState],
    runnable_config: RunnableConfig,
) -> Command:
    """
    Generate configuration optimisation recommendations for Terraform modules.

    Reads ``execution_planner_output.module_structure_plans`` from state.
    Analyses each plan and generates cost, performance, and security
    optimisation strategies.
    """
    try:
        # ── Read inputs from state ───────────────────────────────────────
        exec_output = runtime.state.get("execution_planner_output", {}) or {}
        req_output = runtime.state.get("req_analyser_output", {}) or {}
        sec_output = runtime.state.get("sec_n_best_practices_output", {}) or {}
        module_plans = exec_output.get("module_structure_plans")
        if not module_plans:
            raise ValueError(
                "module_structure_plans missing — run create_module_structure_plan_tool first"
            )

        # ── Extract upstream context (reuse instead of regenerate) ───────
        upstream_context = extract_upstream_context(req_output, sec_output)
        has_upstream = bool(sec_output.get("security_analysis") or sec_output.get("best_practices_analysis"))

        logger.info(
            "Starting configuration optimisation",
            extra={"plan_count": len(module_plans), "has_upstream_context": has_upstream},
        )

        # ── Process each plan ────────────────────────────────────────────
        results: List[ConfigurationOptimizerResponse] = []
        for plan_data in module_plans:
            try:
                plan = ModuleStructurePlanResponse(**plan_data) if isinstance(plan_data, dict) else plan_data
                result = await _create_configuration_optimizer(
                    plan, runnable_config,
                    upstream_context=upstream_context if has_upstream else None,
                )
                result.service_name = plan.service_name
                results.append(result)
            except Exception as e:
                svc = plan_data.get("service_name", "unknown") if isinstance(plan_data, dict) else plan_data.service_name
                logger.error(f"Config optimisation failed for {svc}", extra={"error": str(e)})
                results.append(ConfigurationOptimizerResponse(
                    service_name=svc,
                    optimization_summary=f"Failed: {e}",
                    implementation_priority=["Review and fix errors"],
                ))

        # ── State update ─────────────────────────────────────────────────
        optimizer_list = [r.model_dump(mode="json") for r in results]
        current_output = dict(runtime.state.get("execution_planner_output", {}) or {})
        current_output["configuration_optimizers"] = optimizer_list
        current_output["configuration_optimization_complete"] = True

        logger.info("Configuration optimisation completed", extra={"total": len(results)})

        return Command(update={
            "execution_planner_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"Configuration optimisation completed: {len(results)} plan(s). "
                        "Proceed to create_state_management_plans_tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error("create_configuration_optimizations_tool failed", extra={"error": str(exc)})
        return Command(update={
            "messages": [
                ToolMessage(content=f"Configuration optimisation failed: {exc}", tool_call_id=runtime.tool_call_id)
            ],
        })


# ============================================================================
# Tool 3 — State Management Plans
# ============================================================================

@tool
async def create_state_management_plans_tool(
    runtime: ToolRuntime[None, TFPlannerState],
    runnable_config: RunnableConfig,
) -> Command:
    """
    Create comprehensive state management plans for Terraform infrastructure.

    Reads ``execution_planner_output.configuration_optimizers`` from state.
    Generates S3 backend configuration, DynamoDB locking, state splitting,
    security, and disaster recovery plans.
    """
    try:
        # ── Read inputs from state ───────────────────────────────────────
        exec_output = runtime.state.get("execution_planner_output", {}) or {}
        req_output = runtime.state.get("req_analyser_output", {}) or {}
        sec_output = runtime.state.get("sec_n_best_practices_output", {}) or {}
        optimizers = exec_output.get("configuration_optimizers")
        if not optimizers:
            raise ValueError(
                "configuration_optimizers missing — run create_configuration_optimizations_tool first"
            )

        # ── Extract upstream context (reuse instead of regenerate) ───────
        upstream_context = extract_upstream_context(req_output, sec_output)
        has_upstream = bool(sec_output.get("security_analysis") or sec_output.get("best_practices_analysis"))

        logger.info(
            "Starting state management planning",
            extra={"optimizer_count": len(optimizers), "has_upstream_context": has_upstream},
        )

        # ── Process each optimiser ───────────────────────────────────────
        results: List[StateManagementPlannerResponse] = []
        for opt_data in optimizers:
            try:
                opt = ConfigurationOptimizerResponse(**opt_data) if isinstance(opt_data, dict) else opt_data
                result = await _create_state_mgmt(
                    opt, runnable_config,
                    upstream_context=upstream_context if has_upstream else None,
                )
                result.service_name = opt.service_name
                results.append(result)
            except Exception as e:
                svc = opt_data.get("service_name", "unknown") if isinstance(opt_data, dict) else opt_data.service_name
                logger.error(f"State mgmt planning failed for {svc}", extra={"error": str(e)})
                results.append(StateManagementPlannerResponse(
                    service_name=svc,
                    infrastructure_scale="unknown",
                    backend_configuration=BackendConfiguration(
                        bucket_name="error", key_pattern="error", region="us-east-1",
                        encrypt=True, versioning=True, server_side_encryption_configuration={},
                    ),
                    state_locking_configuration=StateLockingConfiguration(
                        table_name="error", billing_mode="PAY_PER_REQUEST",
                        hash_key="LockID", region="us-east-1", point_in_time_recovery=True, tags={},
                    ),
                    state_splitting_strategy=StateSplittingStrategy(
                        splitting_approach="error", state_files=[], dependencies=[], data_source_usage=[],
                    ),
                    security_recommendations=BackendSecurityRecommendations(),
                    implementation_steps=[f"Review error: {e}"],
                    best_practices=[], monitoring_setup=[], disaster_recovery=[],
                ))

        # ── State update ─────────────────────────────────────────────────
        plans_list = [r.model_dump(mode="json") for r in results]
        current_output = dict(runtime.state.get("execution_planner_output", {}) or {})
        current_output["state_management_plans"] = plans_list
        current_output["state_management_complete"] = True

        logger.info("State management planning completed", extra={"total": len(results)})

        return Command(update={
            "execution_planner_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"State management planning completed: {len(results)} plan(s). "
                        "Proceed to create_execution_plan_tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error("create_state_management_plans_tool failed", extra={"error": str(exc)})
        return Command(update={
            "messages": [
                ToolMessage(content=f"State management planning failed: {exc}", tool_call_id=runtime.tool_call_id)
            ],
        })


# ============================================================================
# Tool 4 — Comprehensive Execution Plan
# ============================================================================

@tool
async def create_execution_plan_tool(
    runtime: ToolRuntime[None, TFPlannerState],
    runnable_config: RunnableConfig,
) -> Command:
    """
    Generate comprehensive execution plans for Terraform module implementation.

    Reads all prior step outputs from ``execution_planner_output`` in state.
    Aggregates module structure, configuration optimisations, and state
    management data to produce production-ready implementation specifications.
    """
    try:
        # ── Read inputs from state ───────────────────────────────────────
        exec_output = runtime.state.get("execution_planner_output", {}) or {}
        req_output = runtime.state.get("req_analyser_output", {}) or {}
        sec_output = runtime.state.get("sec_n_best_practices_output", {}) or {}

        state_mgmt_plans = exec_output.get("state_management_plans")
        if not state_mgmt_plans:
            raise ValueError("state_management_plans missing — run create_state_management_plans_tool first")

        aws_service_mapping = req_output.get("aws_service_mapping", {})
        tf_attribute_mapping = req_output.get("terraform_attribute_mapping", {})

        # ── Extract upstream context (reuse instead of regenerate) ───────
        upstream_context = extract_upstream_context(req_output, sec_output)
        has_upstream = bool(sec_output.get("security_analysis") or sec_output.get("best_practices_analysis"))

        logger.info(
            "Starting execution planning",
            extra={"plan_count": len(state_mgmt_plans), "has_upstream_context": has_upstream},
        )

        # ── Process each state management plan ───────────────────────────
        results: List[ComprehensiveExecutionPlanResponse] = []
        for plan_data in state_mgmt_plans:
            try:
                plan = StateManagementPlannerResponse(**plan_data) if isinstance(plan_data, dict) else plan_data
                result = await _create_execution_plan(
                    plan, exec_output, aws_service_mapping, tf_attribute_mapping,
                    req_output, runnable_config,
                    upstream_context=upstream_context if has_upstream else None,
                )
                result.service_name = plan.service_name
                results.append(result)
            except Exception as e:
                if "MCP" in str(e) or "TaskGroup" in str(e) or "Docker" in str(e):
                    raise
                svc = plan_data.get("service_name", "unknown") if isinstance(plan_data, dict) else plan_data.service_name
                logger.error(f"Execution planning failed for {svc}", extra={"error": str(e)})
                results.append(ComprehensiveExecutionPlanResponse(
                    service_name=svc, module_name=f"{svc}-module",
                    target_environment="prod",
                    terraform_files=[], variable_definitions=[], local_values=[],
                    data_sources=[], output_definitions=[], resource_configurations=[],
                    module_description="", usage_examples=[], readme_content="",
                    required_providers={}, terraform_version_constraint=">=1.0",
                    error=f"Failed: {e}",
                ))

        # ── State update ─────────────────────────────────────────────────
        plans_list = [r.model_dump(mode="json") for r in results]
        current_output = dict(runtime.state.get("execution_planner_output", {}) or {})
        current_output["execution_plans"] = plans_list
        current_output["execution_plan_complete"] = True

        logger.info(
            "Execution planning completed",
            extra={"total": len(results), "services": [r.service_name for r in results]},
        )

        return Command(update={
            "execution_planner_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"Execution planning completed: {len(results)} plan(s) generated. "
                        "All execution planner steps finished."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error("create_execution_plan_tool failed", extra={"error": str(exc)})
        return Command(update={
            "messages": [
                ToolMessage(
                    content=(
                        f"Execution planning failed: {exc}. "
                        "Do NOT proceed to the next step. You MUST call request_human_input to inform the user about this "
                        "connection error (e.g. check if Docker is running) and ask them to fix it. "
                        "Wait for their response, then retry this tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })