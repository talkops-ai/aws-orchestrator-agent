"""
Requirements Analyzer Tools for Terraform Planner.

Implements the ToolRuntime + Command pattern (LangChain latest standard):
  - State is accessed via ``ToolRuntime[None, TFPlannerState]``
  - State updates are returned via ``Command(update={...})`` with ``ToolMessage``
  - LLM instances are created per-call via ``initialize_llm_*`` helpers
  - Structured output is parsed via ``PydanticOutputParser``
"""

import json
from typing import Annotated, Any, Dict, List, Literal, Optional
from enum import Enum

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from aws_orchestrator_agent.utils.mcp_client import MCPClient
from aws_orchestrator_agent.config import Config
from aws_orchestrator_agent.core.state import TFPlannerState
from aws_orchestrator_agent.utils import (
    AgentLogger,
    initialize_llm_model,
    initialize_llm_higher,
)
from .req_analyser_prompts import (
    AWS_SERVICE_DISCOVERY_SYSTEM_PROMPT,
    AWS_SERVICE_DISCOVERY_HUMAN_PROMPT,
    TERRAFORM_RESOURCE_ATTRIBUTES_SYSTEM_PROMPT,
    TERRAFORM_RESOURCE_ATTRIBUTES_HUMAN_PROMPT,
    INFRA_REQUIREMENTS_PARSER_SYSTEM_PROMPT,
    INFRA_REQUIREMENTS_PARSER_HUMAN_PROMPT,
)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = AgentLogger("REQUIREMENTS_ANALYZER")


# ============================================================================
# Pydantic Output Schemas
# ============================================================================

# -- Infrastructure Requirements (Step 1) -----------------------------------

class ServiceRequirement(BaseModel):
    """Service-specific requirements and specifications."""
    service_name: str = Field(description="Name of the AWS service")
    aws_service_type: str = Field(description="AWS service identifier (e.g., 's3', 'vpc', 'ec2')")
    business_requirements: Dict[str, str] = Field(description="Business needs specific to this service")
    technical_specifications: Dict[str, str] = Field(description="Technical specifications native to this service")


class InfrastructureRequirements(BaseModel):
    """Structured representation of AWS infrastructure requirements."""
    scope_classification: str = Field(description="One of: 'single_service', 'multi_service', 'full_application_stack'")
    deployment_context: str = Field(description="Context like development, production, compliance requirements")
    services: List[ServiceRequirement] = Field(description="List of service-specific requirements")


# -- AWS Service Discovery (Step 2) -----------------------------------------

class DependencyType(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    RECOMMENDED = "recommended"


class ServiceDependency(BaseModel):
    """Service dependency as module variable."""
    service: str = Field(..., description="Dependent service name")
    variable: str = Field(..., description="Variable name for dependency")
    type: DependencyType = Field(..., description="Dependency type")


class ArchitecturePattern(BaseModel):
    """Architecture pattern for the service."""
    pattern_name: str = Field(..., description="Pattern name")
    description: str = Field(..., description="Pattern description")
    best_practices: List[str] = Field(..., description="Best practices for this pattern")


class WellArchitectedAlignment(BaseModel):
    """Well-Architected Framework alignment for the service."""
    operational_excellence: List[str] = Field(default_factory=list)
    security: List[str] = Field(default_factory=list)
    reliability: List[str] = Field(default_factory=list)
    performance_efficiency: List[str] = Field(default_factory=list)
    cost_optimization: List[str] = Field(default_factory=list)
    sustainability: List[str] = Field(default_factory=list)


class CostOptimizationRecommendation(BaseModel):
    """Cost optimization recommendation for the service."""
    category: str = Field(..., description="Cost category")
    recommendation: str = Field(..., description="Specific recommendation")
    potential_savings: Optional[str] = Field(None, description="Potential cost savings")
    implementation_difficulty: Literal["low", "medium", "high"] = Field(..., description="Implementation difficulty")


class ServiceSpecification(BaseModel):
    """Complete individual service specification."""
    service_name: str = Field(..., description="AWS service name (e.g., 'S3', 'EKS', 'ElastiCache')")
    aws_service_type: str = Field(..., description="AWS service identifier (e.g., 's3', 'eks', 'elasticache')")
    terraform_resources: List[str] = Field(..., description="List of Terraform resources for production-grade deployment")
    dependencies: List[ServiceDependency] = Field(default_factory=list, description="Service dependencies as variables")
    architecture_patterns: List[ArchitecturePattern] = Field(..., description="Architecture patterns for this service")
    overall_architecture_pattern: Optional[ArchitecturePattern] = Field(None, description="Overall architecture pattern if this is the primary service")
    well_architected_alignment: WellArchitectedAlignment = Field(..., description="Well-Architected Framework alignment")
    cost_optimization_recommendations: List[CostOptimizationRecommendation] = Field(..., description="Cost optimization recommendations")
    description: str = Field(..., description="Service description")
    production_features: List[str] = Field(..., description="Production features included")


class AWSServiceMapping(BaseModel):
    """AWS service discovery output — list of individual services."""
    services: List[ServiceSpecification] = Field(..., description="List of all services with complete specifications")


# -- Terraform Resource Attributes (Step 3) ---------------------------------

class AttributeType(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    COMPUTED = "computed"
    DEPRECATED = "deprecated"


class TerraformAttributeModuleDesign(BaseModel):
    """Module design for a Terraform resource."""
    recommended_arguments: List[str] = Field(..., description="Recommended arguments for the resource")
    recommended_outputs: List[str] = Field(..., description="Recommended outputs for the resource")


class TerraformAttribute(BaseModel):
    """Simplified Terraform attribute specification."""
    name: str = Field(..., description="Attribute name")
    type: str = Field(..., description="Terraform data type (string, number, bool, list, map, object, etc.)")
    required: bool = Field(..., description="Whether this attribute is mandatory")
    description: str = Field(..., description="Detailed attribute description and purpose")
    example_value: Optional[Any] = Field(None, description="Practical example value")


class TerraformResourceSpecification(BaseModel):
    """Enhanced Terraform resource specification with comprehensive attributes."""
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
    """Attribute mapping for a single AWS service."""
    service_name: str = Field(..., description="AWS service name")
    aws_service_type: str = Field(..., description="AWS service type identifier")
    description: str = Field(..., description="Service description and purpose")
    terraform_resources: List[TerraformResourceSpecification] = Field(..., description="Complete resource specifications")
    version_requirements: Optional[str] = Field(None, description="Provider version requirements")


class TerraformAttributeMapping(BaseModel):
    """Enhanced attribute mapping supporting multiple services."""
    services: List[TerraformServiceAttributeMapping] = Field(..., description="List of service attribute mappings")


# ============================================================================
# Prompt template (built from constants in req_analyser_prompts.py)
# ============================================================================

_INFRA_REQ_PARSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INFRA_REQUIREMENTS_PARSER_SYSTEM_PROMPT),
    ("human", INFRA_REQUIREMENTS_PARSER_HUMAN_PROMPT),
])


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
    # Strip markdown code fences if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ============================================================================
# Tool 1 — Infrastructure Requirements Parser
# ============================================================================

@tool
async def infra_requirements_parser_tool(
    runtime: ToolRuntime[None, TFPlannerState],
    runnable_config: RunnableConfig,
    additional_feedback: Optional[str] = None,
) -> Command:
    """
    Extracts infrastructure requirements from natural language queries.

    Reads ``user_query`` directly from graph state. Identifies primary AWS
    services, determines scope, maps business requirements to technical
    specifications, and writes the result back to state.

    Args:
        additional_feedback: Optional human-clarified context from HITL.
            When the agent received feedback via request_human_input(),
            it should pass that feedback here so the parser can incorporate
            explicit user preferences (region, environment, services, etc.)
            into the analysis. If None, the tool proceeds with user_query only.
    """
    try:
        # ── Read input from state ────────────────────────────────────────
        user_query: str = runtime.state.get("user_query", "")
        if not user_query:
            raise ValueError("user_query is missing from state")

        feedback_str = additional_feedback or "None"

        logger.info(
            "Starting infra requirements parser",
            extra={
                "user_query_preview": user_query[:100],
                "has_additional_feedback": additional_feedback is not None,
            },
        )

        # ── LLM + Parser ────────────────────────────────────────────────
        config = Config()
        llm = initialize_llm_model(config.get_llm_config())
        parser = PydanticOutputParser(pydantic_object=InfrastructureRequirements)

        prompt = _INFRA_REQ_PARSER_PROMPT.partial(
            format_instructions=parser.get_format_instructions()
        )
        chain = prompt | llm | parser
        parsed: InfrastructureRequirements = await chain.ainvoke(
            {
                "user_query": user_query,
                "additional_feedback": feedback_str,
            },
            config=runnable_config,
        )

        # ── Log ──────────────────────────────────────────────────────────
        logger.info(
            "Infra requirements parsed successfully",
            extra={
                "scope": parsed.scope_classification,
                "services_count": len(parsed.services),
            },
        )

        # ── State update via Command + ToolMessage ───────────────────────
        current_output = dict(runtime.state.get("req_analyser_output", {}) or {})
        current_output["analysis_results"] = parsed.model_dump(mode="json")
        current_output["analysis_complete"] = True

        return Command(update={
            "req_analyser_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"Infrastructure requirements parsed: {parsed.scope_classification} scope, "
                        f"{len(parsed.services)} service(s) identified. "
                        "Proceed to aws_service_discovery_tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error("infra_requirements_parser_tool failed", extra={"error": str(exc)})
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"Infrastructure requirements parsing failed: {exc}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })


# ============================================================================
# Tool 2 — AWS Service Discovery
# ============================================================================

@tool
async def aws_service_discovery_tool(
    runtime: ToolRuntime[None, TFPlannerState],
    runnable_config: RunnableConfig,
    additional_feedback: Optional[str] = None,
) -> Command:
    """
    Service-focused AWS service discovery for Terraform module generation.

    Reads the parsed requirements from ``req_analyser_output.analysis_results``
    in state.  Generates individual service specifications with production-grade
    Terraform resources, dependency mappings, architecture patterns,
    Well-Architected Framework alignment, and cost optimization recommendations.

    Args:
        additional_feedback: Optional human-clarified context from HITL.
            When the agent received feedback via request_human_input(),
            it should pass that feedback here so service discovery can
            incorporate explicit user preferences (region, environment,
            compliance, network topology) into service selection and
            configuration recommendations.
    """
    try:
        # ── Read input from state ────────────────────────────────────────
        req_output = runtime.state.get("req_analyser_output", {}) or {}
        analysis_results = req_output.get("analysis_results")
        if not analysis_results:
            raise ValueError(
                "analysis_results missing from req_analyser_output — "
                "run infra_requirements_parser_tool first"
            )
        requirements_analysis = json.dumps(analysis_results)
        feedback_str = additional_feedback or "None"

        logger.info(
            "Starting AWS service discovery",
            extra={
                "input_length": len(requirements_analysis),
                "has_additional_feedback": additional_feedback is not None,
            },
        )

        # ── LLM + Parser ────────────────────────────────────────────────
        config = Config()
        llm = initialize_llm_higher(config.get_llm_higher_config())
        parser = PydanticOutputParser(pydantic_object=AWSServiceMapping)

        prompt = ChatPromptTemplate.from_messages([
            ("system", AWS_SERVICE_DISCOVERY_SYSTEM_PROMPT),
            ("human", AWS_SERVICE_DISCOVERY_HUMAN_PROMPT + "\n\n{format_instructions}"),
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm | parser
        parsed: AWSServiceMapping = await chain.ainvoke(
            {
                "requirements_input": requirements_analysis,
                "additional_feedback": feedback_str,
            },
            config=runnable_config,
        )

        logger.info(
            "AWS service discovery completed (LLM pass)",
            extra={"services_count": len(parsed.services)},
        )

        # ── MCP Validation (post-LLM) ───────────────────────────────────
        try:
            config = Config()
            mcp = MCPClient(config, server_filter=["terraform"])
            async with mcp.connect():
                for service in parsed.services:
                    validated_resources = []
                    for res_name in service.terraform_resources:
                        slug = res_name.removeprefix("aws_")
                        search = await mcp.execute_tool(
                            server_name="terraform",
                            tool_name="search_providers",
                            arguments={
                                "provider_name": "aws",
                                "provider_namespace": "hashicorp",
                                "service_slug": slug,
                                "provider_document_type": "resources",
                            },
                        )
                        resource_title = res_name.removeprefix("aws_")
                        if f"Title: {resource_title}" in str(search) or f"title: {resource_title}" in str(search).lower():
                            validated_resources.append(res_name)
                        else:
                            logger.warning(
                                f"LLM-generated resource '{res_name}' not found in registry, removing",
                                extra={"service": service.service_name},
                            )
                    service.terraform_resources = validated_resources or service.terraform_resources
        except Exception as mcp_err:
            logger.error(
                "MCP validation failed",
                extra={"error": str(mcp_err)},
            )
            return Command(update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"AWS service discovery MCP validation failed: {str(mcp_err)}. "
                            "Do NOT proceed. You MUST call request_human_input to ask the user to fix the MCP connection "
                            "(e.g., check if Docker daemon is running), then retry this tool."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })

        # ── State update ─────────────────────────────────────────────────
        current_output = dict(runtime.state.get("req_analyser_output", {}) or {})
        current_output["aws_service_mapping"] = parsed.model_dump(mode="json")
        current_output["aws_service_mapping_complete"] = True
        current_output["terraform_resources_validated"] = True

        return Command(update={
            "req_analyser_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"AWS service discovery completed: {len(parsed.services)} service(s) mapped. "
                        "Proceed to get_final_resource_attributes_tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error("aws_service_discovery_tool failed", extra={"error": str(exc)})
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"AWS service discovery failed: {exc}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })


# ============================================================================
# Tool 3 — Final Resource Attributes (Coordinator Agent)
# ============================================================================

@tool
async def get_final_resource_attributes_tool(
    runtime: ToolRuntime[None, TFPlannerState],
) -> Command:
    """
    Get final resource attributes by fetching authoritative data from
    the Terraform Registry via the MCP server.

    Iterates over all services in `aws_service_mapping`, fetching each
    resource's documentation directly via MCP. No inner agent needed.

    Falls back to LLM per-resource when MCP fails for individual resources.
    """
    try:
        # ── Read input from state ────────────────────────────────────
        req_output = runtime.state.get("req_analyser_output", {}) or {}
        service_mapping = req_output.get("aws_service_mapping")
        if not service_mapping:
            raise ValueError(
                "aws_service_mapping missing — run aws_service_discovery_tool first"
            )

        services = service_mapping.get("services", [])
        logger.info(
            "Starting MCP-based attribute fetching",
            extra={"service_count": len(services)},
        )

        # ── Open MCP session ─────────────────────────────────────────
        config = Config()
        client = MCPClient(config, server_filter=["terraform"])

        service_results: List[TerraformServiceAttributeMapping] = []

        async with client.connect():
            # Get provider version once for all resources
            try:
                provider_version = await client.execute_tool(
                    server_name="terraform",
                    tool_name="get_latest_provider_version",
                    arguments={"namespace": "hashicorp", "name": "aws"},
                )
            except Exception:
                provider_version = "6.0"

            for service in services:
                service_name = service.get("service_name", "unknown")
                aws_type = service.get("aws_service_type", "unknown")
                tf_resources = service.get("terraform_resources", [])

                logger.info(
                    f"Processing service: {service_name}",
                    extra={"resources": len(tf_resources)},
                )

                resource_specs: List[TerraformResourceSpecification] = []

                for resource_name in tf_resources:
                    try:
                        spec = await _fetch_resource_via_mcp(
                            client, resource_name, provider_version
                        )
                        resource_specs.append(spec)
                        logger.debug(
                            f"MCP fetch OK: {resource_name}",
                            extra={
                                "required": len(spec.required_attributes),
                                "optional": len(spec.optional_attributes),
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            f"MCP failed for {resource_name}, using LLM fallback",
                            extra={"error": str(e)},
                        )
                        try:
                            fallback_json = await _llm_fallback_resource_attributes(
                                resource_name
                            )
                            spec = TerraformResourceSpecification.model_validate_json(
                                fallback_json
                            )
                            resource_specs.append(spec)
                        except Exception as fb_err:
                            logger.error(
                                f"Both MCP and LLM failed for {resource_name}",
                                extra={"error": str(fb_err)},
                            )

                service_results.append(TerraformServiceAttributeMapping(
                    service_name=service_name,
                    aws_service_type=aws_type,
                    description=service.get("description", ""),
                    terraform_resources=resource_specs,
                    version_requirements=f"~> {provider_version}",
                ))

        # ── Build output ─────────────────────────────────────────────
        mapping = TerraformAttributeMapping(services=service_results)
        parsed = mapping.model_dump(mode="json")

        total_resources = sum(
            len(svc.terraform_resources) for svc in service_results
        )

        logger.info(
            "MCP-based attribute fetching completed",
            extra={
                "services_count": len(service_results),
                "total_resources": total_resources,
            },
        )

        # ── State update ─────────────────────────────────────────────
        current_output = dict(runtime.state.get("req_analyser_output", {}) or {})
        current_output["terraform_attribute_mapping"] = parsed
        current_output["terraform_attribute_mapping_complete"] = True
        current_output["provider_version"] = provider_version

        return Command(update={
            "req_analyser_output": current_output,
            "messages": [
                ToolMessage(
                    content=(
                        f"Terraform attribute mapping completed via MCP: "
                        f"{len(service_results)} service(s), "
                        f"{total_resources} resource(s)."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    except Exception as exc:
        logger.error(
            "get_final_resource_attributes_tool failed",
            extra={"error": str(exc)},
        )
        return Command(update={
            "messages": [
                ToolMessage(
                    content=(
                        f"Terraform attribute mapping failed: {exc}. "
                        "Do NOT proceed to the next step. You MUST call request_human_input to inform the user about this "
                        "MCP connection error (e.g. check if Docker is running) and ask them to fix it. "
                        "Wait for their response, then retry this tool."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })


async def _fetch_resource_via_mcp(
    client: MCPClient,
    terraform_resource_name: str,
    provider_version: str,
) -> TerraformResourceSpecification:
    """
    Fetch a single resource's attributes via the Terraform MCP server.
    """
    from aws_orchestrator_agent.core.agents.tf_operator.tf_planner.new_module.utils.mcp_terraform_helpers import (
        parse_registry_docs_to_spec,
        _extract_provider_doc_id,
    )

    service_slug = terraform_resource_name.removeprefix("aws_")

    search_result = await client.execute_tool(
        server_name="terraform",
        tool_name="search_providers",
        arguments={
            "provider_name": "aws",
            "provider_namespace": "hashicorp",
            "service_slug": service_slug,
            "provider_document_type": "resources",
        },
    )

    provider_doc_id = _extract_provider_doc_id(search_result, terraform_resource_name)
    if not provider_doc_id:
        raise ValueError(f"Resource '{terraform_resource_name}' not found in registry")

    raw_docs = await client.execute_tool(
        server_name="terraform",
        tool_name="get_provider_details",
        arguments={"provider_doc_id": provider_doc_id},
    )

    return parse_registry_docs_to_spec(str(raw_docs), terraform_resource_name, provider_version)


async def _llm_fallback_resource_attributes(
    terraform_resource_name: str,
) -> str:
    """
    Original LLM-based attribute fetching — used as fallback when MCP
    is unavailable.
    """
    try:
        logger.info(
            "Using LLM fallback for resource attributes",
            extra={"resource": terraform_resource_name},
        )
        config = Config()
        llm = initialize_llm_higher(config.get_llm_higher_config())
        parser = PydanticOutputParser(pydantic_object=TerraformResourceSpecification)

        prompt = ChatPromptTemplate.from_messages([
            ("system", TERRAFORM_RESOURCE_ATTRIBUTES_SYSTEM_PROMPT),
            ("human", TERRAFORM_RESOURCE_ATTRIBUTES_HUMAN_PROMPT + "\n\n{format_instructions}"),
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm | parser
        parsed: TerraformResourceSpecification = await chain.ainvoke(
            {"terraform_resource_name": terraform_resource_name}
        )
        return parsed.model_dump_json()

    except Exception as exc:
        logger.error(
            "LLM fallback also failed",
            extra={"resource": terraform_resource_name, "error": str(exc)},
        )
        return json.dumps({"error": str(exc), "resource": terraform_resource_name})


@tool
async def get_terraform_resource_attributes_tool(
    terraform_resource_name: str,
) -> str:
    """
    Get comprehensive attribute specifications for a single Terraform resource.

    Uses the Terraform MCP Server to fetch authoritative resource documentation
    from the Terraform Registry. Falls back to LLM-based analysis if MCP
    is unavailable.
    """
    try:
        logger.info(
            "Fetching resource attributes via MCP",
            extra={"resource": terraform_resource_name},
        )

        config = Config()
        client = MCPClient(config, server_filter=["terraform"])

        async with client.connect():
            # Get latest version for the spec
            provider_version = await client.execute_tool(
                server_name="terraform",
                tool_name="get_latest_provider_version",
                arguments={"namespace": "hashicorp", "name": "aws"},
            )

            # Route to MCP helper
            spec = await _fetch_resource_via_mcp(
                client, terraform_resource_name, str(provider_version)
            )

        logger.info(
            "MCP resource attributes fetched",
            extra={
                "resource": terraform_resource_name,
                "required_count": len(spec.required_attributes),
                "optional_count": len(spec.optional_attributes),
                "deprecated_count": len(spec.deprecated_attributes),
            },
        )

        return spec.model_dump_json()

    except Exception as exc:
        logger.warning(
            "MCP fetch failed, falling back to LLM",
            extra={"resource": terraform_resource_name, "error": str(exc)},
        )
        return await _llm_fallback_resource_attributes(terraform_resource_name)