"""
Security and Best Practices Tools for Terraform Planner.

Implements the ToolRuntime + Command pattern (LangChain latest standard):
  - State is accessed via ``ToolRuntime[None, TFPlannerState]``
  - Inputs are read from ``runtime.state`` (state-driven, not agent-passed)
  - State updates are returned via ``Command(update={...})`` with ``ToolMessage``
  - LLM instances are created per-call via ``initialize_llm_*`` helpers
  - Structured output is parsed via ``PydanticOutputParser``
"""

import json
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from aws_orchestrator_agent.utils.mcp_client import MCPClient
from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.utils.mcp_terraform_helpers import _extract_policy_id

from aws_orchestrator_agent.config import Config
from aws_orchestrator_agent.core.state import TFPlannerState
from aws_orchestrator_agent.utils import (
    AgentLogger,
    initialize_llm_model,
    initialize_llm_higher,
)
from .sec_n_best_practices_prompts import (
    SECURITY_COMPLIANCE_SYSTEM_PROMPT,
    SECURITY_COMPLIANCE_USER_PROMPT,
    BEST_PRACTICES_SYSTEM_PROMPT,
    BEST_PRACTICES_USER_PROMPT,
)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = AgentLogger("SECURITY_BEST_PRACTICES")


# ============================================================================
# Pydantic Output Schemas
# ============================================================================

# -- Shared building blocks --------------------------------------------------

class ComplianceDetail(BaseModel):
    status: Literal["compliant", "non_compliant"] = Field(
        ..., description="Compliance status for the control"
    )
    issues: List[str] = Field(default_factory=list, description="List of identified issues")
    recommendations: List[str] = Field(default_factory=list, description="Remediation steps for issues")


class NetworkSecurityDetail(BaseModel):
    security_groups: ComplianceDetail = Field(..., description="Analysis of Security Groups configurations")
    network_acls: ComplianceDetail = Field(..., description="Analysis of Network ACLs configurations")


class AccessControlDetail(BaseModel):
    iam_roles: ComplianceDetail = Field(..., description="Analysis of IAM Role definitions and usage")
    iam_policies: ComplianceDetail = Field(..., description="Analysis of IAM Policy documents")


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


# -- Security compliance schemas (multi-service) ----------------------------

class ServiceSecurityAnalysis(BaseModel):
    """Security analysis for a single service."""
    service_name: str = Field(..., description="Name of the AWS service")
    service_type: str = Field(..., description="AWS service type identifier")
    encryption_at_rest: ComplianceDetail = Field(..., description="Encryption at rest compliance")
    encryption_in_transit: ComplianceDetail = Field(..., description="Encryption in transit compliance")
    network_security: NetworkSecurityDetail = Field(..., description="Network security compliance")
    access_controls: AccessControlDetail = Field(..., description="Access control compliance")
    service_compliance: Literal["compliant", "non_compliant"] = Field(..., description="Compliance status")
    service_issues: List[str] = Field(default_factory=list, description="Service-specific security issues")
    service_recommendations: List[str] = Field(default_factory=list, description="Service-specific recommendations")


class CrossServiceSecurityAnalysis(BaseModel):
    """Cross-service security relationships and dependencies."""
    service_dependencies: Dict[str, List[str]] = Field(..., description="Security dependencies between services")
    shared_security_risks: List[str] = Field(default_factory=list, description="Risks affecting multiple services")
    cross_service_recommendations: List[str] = Field(default_factory=list, description="Cross-service recommendations")


class OverallSecuritySummary(BaseModel):
    """Overall security summary across all services."""
    total_services: int = Field(..., description="Total number of services analyzed")
    compliant_services: int = Field(..., description="Number of compliant services")
    non_compliant_services: int = Field(..., description="Number of non-compliant services")
    critical_issues_count: int = Field(..., description="Number of critical security issues")
    high_priority_issues_count: int = Field(..., description="Number of high priority issues")
    overall_risk_level: Literal["low", "medium", "high", "critical"] = Field(..., description="Overall risk level")


class EnhancedSecurityAnalysisResult(BaseModel):
    """Enhanced security analysis result supporting multiple services."""
    services: List[ServiceSecurityAnalysis] = Field(..., description="Per-service security analysis")
    cross_service_analysis: CrossServiceSecurityAnalysis = Field(..., description="Cross-service relationships")
    overall_summary: OverallSecuritySummary = Field(..., description="Overall security summary")
    encryption_at_rest: ComplianceDetail = Field(..., description="Overall encryption at rest")
    encryption_in_transit: ComplianceDetail = Field(..., description="Overall encryption in transit")
    network_security: NetworkSecurityDetail = Field(..., description="Overall network security")
    access_controls: AccessControlDetail = Field(..., description="Overall access control")
    overall_compliance: Literal["compliant", "non_compliant"] = Field(..., description="Overall compliance")
    summary_issues: List[str] = Field(default_factory=list, description="Overall summary of security issues")

    @model_validator(mode="after")
    def validate_overall_compliance(self):
        if self.services:
            compliant = sum(1 for s in self.services if s.service_compliance == "compliant")
            self.overall_compliance = "compliant" if compliant == len(self.services) else "non_compliant"
        return self


# -- Best practices schemas (multi-service) ---------------------------------

class ServiceBestPracticesAnalysis(BaseModel):
    """Best practices analysis for a single service."""
    service_name: str = Field(..., description="Name of the AWS service")
    service_type: str = Field(..., description="AWS service type identifier")
    naming_and_tagging: List[BestPracticeFinding] = Field(..., description="Naming and tagging checks")
    module_structure: List[BestPracticeFinding] = Field(..., description="Module structure checks")
    resource_optimization: List[BestPracticeFinding] = Field(..., description="Resource optimization")
    terraform_practices: List[BestPracticeFinding] = Field(..., description="Terraform practices")
    service_status: ValidationStatus = Field(..., description="Overall status for this service")
    service_score: int = Field(..., ge=0, le=100, description="Best practices score (0-100)")


class CrossServiceBestPracticesAnalysis(BaseModel):
    """Cross-service best practices relationships."""
    shared_patterns: List[str] = Field(default_factory=list, description="Shared best practice patterns")
    consistency_issues: List[str] = Field(default_factory=list, description="Inconsistencies between services")
    cross_service_recommendations: List[str] = Field(default_factory=list, description="Cross-service recommendations")


class OverallBestPracticesSummary(BaseModel):
    """Overall best practices summary across all services."""
    total_services: int = Field(..., description="Total services analyzed")
    services_passing: int = Field(..., description="Services with PASS status")
    services_warning: int = Field(..., description="Services with WARN status")
    services_failing: int = Field(..., description="Services with FAIL status")
    average_score: float = Field(..., ge=0, le=100, description="Average score across services")
    overall_status: ValidationStatus = Field(..., description="Overall best practices status")


class EnhancedBestPracticesResponse(BaseModel):
    """Enhanced best practices response supporting multiple services."""
    services: List[ServiceBestPracticesAnalysis] = Field(..., description="Per-service analysis")
    cross_service_analysis: CrossServiceBestPracticesAnalysis = Field(..., description="Cross-service relationships")
    overall_summary: OverallBestPracticesSummary = Field(..., description="Overall summary")
    naming_and_tagging: List[BestPracticeFinding] = Field(..., description="Overall naming/tagging checks")
    module_structure: List[BestPracticeFinding] = Field(..., description="Overall module structure checks")
    resource_optimization: List[BestPracticeFinding] = Field(..., description="Overall resource optimization")
    terraform_practices: List[BestPracticeFinding] = Field(..., description="Overall Terraform practices")
    overall_status: ValidationStatus = Field(..., description="Overall status")

    @model_validator(mode="after")
    def validate_overall_status(self):
        fail_count = sum(1 for s in self.services if s.service_status == ValidationStatus.FAIL)
        warn_count = sum(1 for s in self.services if s.service_status == ValidationStatus.WARN)
        if fail_count > 0:
            self.overall_status = ValidationStatus.FAIL
        elif warn_count > 0:
            self.overall_status = ValidationStatus.WARN
        else:
            self.overall_status = ValidationStatus.PASS
        return self


# ============================================================================
# Data Extraction Helpers (pure functions — no state, no LLM)
# ============================================================================

def extract_security_relevant_data(
    terraform_mapping: Dict[str, Any],
    aws_service_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract security-relevant data from both sources for multiple services."""
    security_data: Dict[str, Any] = {
        "services": [],
        "overall_security_summary": {
            "total_services": 0,
            "security_critical_resources": 0,
            "well_architected_security_alignment": [],
        },
    }

    _SECURITY_KEYWORDS = frozenset([
        "encrypt", "decrypt", "key", "secret", "password", "token", "auth",
        "permission", "policy", "role", "security", "access", "grant", "alias",
        "arn", "id", "certificate", "ssl", "tls", "vpc", "subnet", "route",
        "gateway", "firewall", "waf", "shield",
    ])

    for service in aws_service_mapping.get("services", []):
        service_name = service.get("service_name")
        service_type = service.get("aws_service_type")

        # Match terraform data for this service
        terraform_service_data = next(
            (ts for ts in terraform_mapping.get("services", [])
             if ts.get("service_name") == service_name),
            None,
        )

        svc: Dict[str, Any] = {
            "service_info": {
                "service_name": service_name,
                "service_type": service_type,
                "description": service.get("description"),
            },
            "security_architecture": {
                "well_architected_security": service.get("well_architected_alignment", {}).get("security", []),
                "security_features": service.get("production_features", []),
                "architecture_patterns": [p.get("pattern_name") for p in service.get("architecture_patterns", [])],
            },
            "security_dependencies": {
                "required": [d.get("service") for d in service.get("dependencies", []) if getattr(d.get("type"), "value", d.get("type")) == "required"],
                "optional": [d.get("service") for d in service.get("dependencies", []) if getattr(d.get("type"), "value", d.get("type")) == "optional"],
                "recommended": [d.get("service") for d in service.get("dependencies", []) if getattr(d.get("type"), "value", d.get("type")) == "recommended"],
            },
            "security_critical_resources": [],
        }

        if terraform_service_data:
            for resource in terraform_service_data.get("terraform_resources", []):
                sec_attrs = [
                    {
                        "name": a.get("name"), "type": a.get("type"),
                        "required": a.get("required"), "description": a.get("description"),
                        "validation_rules": a.get("validation_rules"), "category": a.get("category"),
                    }
                    for a in resource.get("attributes", [])
                    if any(kw in (a.get("name") or "").lower() for kw in _SECURITY_KEYWORDS)
                ]
                if sec_attrs:
                    svc["security_critical_resources"].append({
                        "resource_name": resource.get("resource_name"),
                        "security_attributes": sec_attrs,
                    })

        security_data["services"].append(svc)
        security_data["overall_security_summary"]["total_services"] += 1
        security_data["overall_security_summary"]["security_critical_resources"] += len(svc["security_critical_resources"])
        security_data["overall_security_summary"]["well_architected_security_alignment"].extend(
            svc["security_architecture"]["well_architected_security"]
        )

    return security_data


def extract_best_practices_data(
    terraform_mapping: Dict[str, Any],
    aws_service_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract best-practices-relevant data from both sources for multiple services."""
    bp_data: Dict[str, Any] = {
        "services": [],
        "overall_best_practices_summary": {
            "total_services": 0,
            "total_resources": 0,
            "total_attributes": 0,
            "well_architected_pillars": {
                "operational_excellence": [], "security": [],
                "reliability": [], "performance_efficiency": [],
                "cost_optimization": [], "sustainability": [],
            },
        },
    }

    _OPT_KEYWORDS = frozenset([
        "name", "tag", "description", "alias", "version", "configuration",
        "setting", "parameter", "option", "feature", "enable", "disable",
        "size", "capacity", "instance", "type", "class", "tier", "level",
    ])

    for service in aws_service_mapping.get("services", []):
        service_name = service.get("service_name")
        service_type = service.get("aws_service_type")

        terraform_service_data = next(
            (ts for ts in terraform_mapping.get("services", [])
             if ts.get("service_name") == service_name),
            None,
        )

        svc: Dict[str, Any] = {
            "service_info": {"service_name": service_name, "service_type": service_type},
            "well_architected_alignment": service.get("well_architected_alignment", {}),
            "cost_optimization": service.get("cost_optimization_recommendations", []),
            "resource_structure": {
                "total_resources": 0,
                "attribute_summary": {
                    "total_attributes": 0, "required_attributes": 0,
                    "optional_attributes": 0, "computed_attributes": 0,
                },
            },
            "key_resources": [],
        }

        if terraform_service_data:
            for resource in terraform_service_data.get("terraform_resources", []):
                attrs = resource.get("attributes", [])
                key_attrs = [
                    {
                        "name": a.get("name"), "type": a.get("type"),
                        "required": a.get("required"), "description": a.get("description"),
                        "category": a.get("category"), "is_output": a.get("is_output"),
                        "is_reference": a.get("is_reference"),
                    }
                    for a in attrs
                    if any(kw in (a.get("name") or "").lower() for kw in _OPT_KEYWORDS)
                ]
                if key_attrs:
                    svc["key_resources"].append({
                        "resource_name": resource.get("resource_name"),
                        "key_attributes": key_attrs,
                        "attribute_counts": resource.get("attribute_counts", {}),
                    })

                svc["resource_structure"]["total_resources"] += 1
                svc["resource_structure"]["attribute_summary"]["total_attributes"] += len(attrs)
                svc["resource_structure"]["attribute_summary"]["required_attributes"] += sum(1 for a in attrs if a.get("required"))
                svc["resource_structure"]["attribute_summary"]["optional_attributes"] += sum(1 for a in attrs if not a.get("required"))
                svc["resource_structure"]["attribute_summary"]["computed_attributes"] += sum(1 for a in attrs if a.get("category") == "computed")

        bp_data["services"].append(svc)
        bp_data["overall_best_practices_summary"]["total_services"] += 1
        bp_data["overall_best_practices_summary"]["total_resources"] += svc["resource_structure"]["total_resources"]
        bp_data["overall_best_practices_summary"]["total_attributes"] += svc["resource_structure"]["attribute_summary"]["total_attributes"]

        for pillar, recs in service.get("well_architected_alignment", {}).items():
            if pillar in bp_data["overall_best_practices_summary"]["well_architected_pillars"]:
                bp_data["overall_best_practices_summary"]["well_architected_pillars"][pillar].extend(recs)

    return bp_data


# ============================================================================
# Helper — extract string content from LLM response
# ============================================================================

def _extract_content(llm_response: Any) -> str:
    """Return the string content from an LLM response."""
    if isinstance(llm_response, AIMessage):
        content = llm_response.content
    else:
        content = llm_response
    text = str(content).strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ============================================================================
# Tool 1 — Security Compliance Analysis
# ============================================================================

@tool
async def security_compliance_tool(
    runtime: ToolRuntime[None, TFPlannerState],
    runnable_config: RunnableConfig,
    additional_feedback: Optional[str] = None,
) -> Command:
    """
    Production-grade security compliance analysis with multi-service support.

    Reads ``req_analyser_output`` from state to extract terraform attribute
    mapping and AWS service mapping.  Analyses security across frameworks
    (SOC 2, ISO 27001, HIPAA, PCI DSS), encryption, network security,
    access controls, and cross-service relationships.

    Args:
        additional_feedback: Optional human-clarified context from HITL.
            When the agent received feedback via request_human_input(),
            it should pass that feedback here so the security analysis can
            incorporate explicit compliance framework choices, encryption
            preferences, and network exposure tolerance.
    """
    try:
        # ── Read inputs from state ───────────────────────────────────
        req_output = runtime.state.get("req_analyser_output", {}) or {}
        terraform_mapping = req_output.get("terraform_attribute_mapping", {})
        aws_service_mapping = req_output.get("aws_service_mapping", {})

        if not terraform_mapping or not aws_service_mapping:
            logger.warning(
                "No terraform_mapping or aws_service_mapping in state, "
                "proceeding with empty security data"
            )
            security_data: Dict[str, Any] = {"user_request": "", "services": []}
        else:
            security_data = extract_security_relevant_data(terraform_mapping, aws_service_mapping)

        security_data_str = json.dumps(security_data, indent=2)
        feedback_str = additional_feedback or "None"

        # ── Fetch real CIS policies via MCP ──────────────────────────
        cis_context = ""
        try:
            mcp = MCPClient(config=Config(), server_filter=["terraform"])
            async with mcp.connect():
                policy_search = await mcp.execute_tool(
                    server_name="terraform",
                    tool_name="search_policies",
                    arguments={"policy_query": "cis"},
                )
                
                policy_id = _extract_policy_id(str(policy_search))
                if policy_id:
                    policy_details = await mcp.execute_tool(
                        server_name="terraform",
                        tool_name="get_policy_details",
                        arguments={"terraform_policy_id": policy_id},
                    )

                    cis_context = (
                        "\n\n## Real CIS AWS Foundations Benchmark Policies "
                        "(from Terraform Registry)\n\n"
                        "The following are actual CIS compliance policies "
                        "with their enforcement rules. Use these REAL policy "
                        "names and descriptions to ground your compliance "
                        "analysis — do NOT invent policy names:\n\n"
                        f"{str(policy_details)[:6000]}\n"
                    )
                    logger.info(
                        "CIS policy data fetched successfully",
                        extra={"policy_id": policy_id},
                    )
                else:
                    logger.debug("No CIS policy found in search results")

        except Exception as mcp_err:
            logger.error(
                "CIS policy fetch failed",
                extra={"error": str(mcp_err)},
            )
            return Command(update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Security compliance MCP validation failed: {str(mcp_err)}. "
                            "Do NOT proceed. You MUST call request_human_input to ask the user to fix the MCP connection "
                            "(e.g., check if Docker daemon is running), then retry this tool."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })

        security_data_str = security_data_str + cis_context

        logger.info(
            "Starting security compliance analysis",
            extra={
                "total_services": security_data.get("overall_security_summary", {}).get("total_services", 0),
                "input_length": len(security_data_str),
                "has_additional_feedback": additional_feedback is not None,
            },
        )

        # ── LLM + Parser ────────────────────────────────────────────────
        config = Config()
        llm = initialize_llm_higher(config.get_llm_higher_config())
        parser = PydanticOutputParser(pydantic_object=EnhancedSecurityAnalysisResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SECURITY_COMPLIANCE_SYSTEM_PROMPT),
            ("human", SECURITY_COMPLIANCE_USER_PROMPT + "\n\n{format_instructions}"),
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm | parser
        parsed: EnhancedSecurityAnalysisResult = await chain.ainvoke(
            {
                "infrastructure_definition": security_data_str,
                "additional_feedback": feedback_str,
            },
            config=runnable_config,
        )

        logger.info(
            "Security compliance analysis completed",
            extra={
                "services_count": len(parsed.services),
                "overall_compliance": parsed.overall_compliance,
            },
        )

        # ── State update via Command + ToolMessage ───────────────────────
        current_output = dict(runtime.state.get("sec_n_best_practices_output", {}) or {})
        current_output["security_analysis"] = parsed.model_dump(mode="json")
        current_output["security_analysis_complete"] = True

        return Command(update={
            "sec_n_best_practices_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"Security compliance analysis completed: "
                        f"{len(parsed.services)} service(s), "
                        f"overall={parsed.overall_compliance}. "
                        "Proceed to best_practices_tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error("security_compliance_tool failed", extra={"error": str(exc)})
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"Security compliance analysis failed: {exc}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })


# ============================================================================
# Tool 2 — Best Practices Analysis
# ============================================================================

@tool
async def best_practices_tool(
    runtime: ToolRuntime[None, TFPlannerState],
    runnable_config: RunnableConfig,
    additional_feedback: Optional[str] = None,
) -> Command:
    """
    Production-grade AWS and Terraform best practices analysis with multi-service
    support.

    Reads ``req_analyser_output`` from state to extract terraform attribute
    mapping and AWS service mapping.  Validates naming/tagging, module
    structure, resource optimisation, and Terraform practices across all
    services.

    Args:
        additional_feedback: Optional human-clarified context from HITL.
            When the agent received feedback via request_human_input(),
            it should pass that feedback here so the best practices
            validator can incorporate explicit naming conventions, tagging
            preferences, or optimization priorities.
    """
    try:
        # ── Read inputs from state ───────────────────────────────────
        req_output = runtime.state.get("req_analyser_output", {}) or {}
        terraform_mapping = req_output.get("terraform_attribute_mapping", {})
        aws_service_mapping = req_output.get("aws_service_mapping", {})

        if not terraform_mapping or not aws_service_mapping:
            logger.warning(
                "No terraform_mapping or aws_service_mapping in state, "
                "proceeding with empty best-practices data"
            )
            bp_data: Dict[str, Any] = {
                "services": [],
                "overall_best_practices_summary": {"total_services": 0},
            }
        else:
            bp_data = extract_best_practices_data(terraform_mapping, aws_service_mapping)

        infrastructure_definition = json.dumps(bp_data, indent=2)
        feedback_str = additional_feedback or "None"

        logger.info(
            "Starting best practices analysis",
            extra={
                "total_services": bp_data.get("overall_best_practices_summary", {}).get("total_services", 0),
                "input_length": len(infrastructure_definition),
                "has_additional_feedback": additional_feedback is not None,
            },
        )

        # ── LLM + Parser ────────────────────────────────────────────────
        config = Config()
        llm = initialize_llm_higher(config.get_llm_higher_config())
        parser = PydanticOutputParser(pydantic_object=EnhancedBestPracticesResponse)

        prompt = ChatPromptTemplate.from_messages([
            ("system", BEST_PRACTICES_SYSTEM_PROMPT),
            ("human", BEST_PRACTICES_USER_PROMPT + "\n\n{format_instructions}"),
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm | parser
        parsed: EnhancedBestPracticesResponse = await chain.ainvoke(
            {
                "infrastructure_definition": infrastructure_definition,
                "additional_feedback": feedback_str,
            },
            config=runnable_config,
        )

        logger.info(
            "Best practices analysis completed",
            extra={
                "services_count": len(parsed.services),
                "overall_status": parsed.overall_status.value,
            },
        )

        # ── State update via Command + ToolMessage ───────────────────────
        current_output = dict(runtime.state.get("sec_n_best_practices_output", {}) or {})
        current_output["best_practices_analysis"] = parsed.model_dump(mode="json")
        current_output["best_practices_analysis_complete"] = True

        return Command(update={
            "sec_n_best_practices_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"Best practices analysis completed: "
                        f"{len(parsed.services)} service(s), "
                        f"overall={parsed.overall_status.value}."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error("best_practices_tool failed", extra={"error": str(exc)})
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"Best practices analysis failed: {exc}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })