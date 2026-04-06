"""
Terraform MCP Server — Helper utilities for parsing registry responses.

Converts raw markdown documentation from the Terraform Registry
(via the Terraform MCP server) into the Pydantic schemas used by the
tf_planner tools.
"""

import re
from typing import Any, Dict, List, Optional

from aws_orchestrator_agent.utils.logger import AgentLogger

# Import the existing Pydantic models
from ..req_analyser_tool.req_analyzer_tool import (
    TerraformResourceSpecification,
    TerraformAttribute,
    TerraformAttributeModuleDesign,
)

logger = AgentLogger("MCP_TF_HELPERS")


# ============================================================================
# Registry Doc Parsers
# ============================================================================

def parse_registry_docs_to_spec(
    raw_docs: str,
    resource_name: str,
    provider_version: str = "6.0",
) -> TerraformResourceSpecification:
    """
    Parse raw Terraform Registry markdown docs into a TerraformResourceSpecification.

    Args:
        raw_docs: Full markdown text from get_provider_details MCP tool.
        resource_name: Terraform resource name (e.g., "aws_s3_bucket").
        provider_version: Provider version string from MCP.

    Returns:
        TerraformResourceSpecification with all attributes categorized.
    """
    # Extract description
    description = _extract_description(raw_docs)

    # Split into sections
    sections = _split_sections(raw_docs)

    # Parse arguments
    arg_ref = sections.get("argument reference", "")
    required_attrs = _parse_arguments(arg_ref, required_only=True)
    optional_attrs = _parse_arguments(arg_ref, optional_only=True)
    deprecated_attrs = _parse_deprecated(arg_ref)

    # Parse computed attributes
    attr_ref = sections.get("attribute reference", "")
    computed_attrs = _parse_computed_attributes(attr_ref)

    # Build module design
    module_design = TerraformAttributeModuleDesign(
        recommended_arguments=[a.name for a in required_attrs]
        + [a.name for a in optional_attrs[:5]],  # Top 5 optional
        recommended_outputs=[a.name for a in computed_attrs],
    )

    return TerraformResourceSpecification(
        resource_name=resource_name,
        provider="aws",
        description=description,
        required_attributes=required_attrs,
        optional_attributes=optional_attrs,
        computed_attributes=computed_attrs,
        deprecated_attributes=deprecated_attrs,
        version_requirements=f"~> {provider_version}",
        module_design=module_design,
    )


def _extract_description(docs: str) -> str:
    """Extract the first paragraph description from registry docs."""
    match = re.search(r"description:\s*\|?\-?\s*\n?\s*(.+?)(?:\n---|\n\n|$)", docs, re.DOTALL)
    if match:
        return match.group(1).strip().split("\n")[0]
    # Fallback: first non-header line
    for line in docs.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "---", ">", "~>")):
            return line
    return ""


def _split_sections(docs: str) -> Dict[str, str]:
    """Split markdown docs into sections keyed by heading name (lowercased)."""
    sections: Dict[str, str] = {}
    current_heading = ""
    current_lines: List[str] = []

    for line in docs.splitlines():
        heading_match = re.match(r"^#{1,3}\s+(.+)", line)
        if heading_match:
            if current_heading:
                sections[current_heading] = "\n".join(current_lines)
            current_heading = heading_match.group(1).strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading:
        sections[current_heading] = "\n".join(current_lines)

    return sections


def _parse_arguments(
    arg_section: str,
    required_only: bool = False,
    optional_only: bool = False,
) -> List[TerraformAttribute]:
    """Parse argument reference section into TerraformAttribute list."""
    attrs: List[TerraformAttribute] = []

    # Pattern: * `name` - (Required|Optional...) Description
    pattern = re.compile(
        r"^\*\s+`(\w+)`\s*-\s*\(((?:Required|Optional)[^)]*)\)\s*(.*?)(?=\n\*\s+`|\n#{1,3}\s|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    for match in pattern.finditer(arg_section):
        name = match.group(1)
        qualifier = match.group(2)
        description = match.group(3).strip()

        is_required = "Required" in qualifier
        is_deprecated = "**Deprecated**" in description

        # Skip deprecated if not requested
        if is_deprecated:
            continue

        if required_only and not is_required:
            continue
        if optional_only and is_required:
            continue

        attr_type = _infer_type(description)

        attrs.append(TerraformAttribute(
            name=name,
            type=attr_type,
            required=is_required,
            description=_clean_description(description),
            example_value=_extract_example(description, attr_type),
        ))

    return attrs


def _parse_deprecated(arg_section: str) -> List[TerraformAttribute]:
    """Extract deprecated attributes from argument reference."""
    attrs: List[TerraformAttribute] = []

    pattern = re.compile(
        r"^\*\s+`(\w+)`\s*-\s*\(([^)]*)\)\s*(.*?)(?=\n\*\s+`|\n#{1,3}\s|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    for match in pattern.finditer(arg_section):
        name = match.group(1)
        qualifier = match.group(2)
        description = match.group(3).strip()

        if "**Deprecated**" in description:
            attrs.append(TerraformAttribute(
                name=name,
                type=_infer_type(description),
                required="Required" in qualifier,
                description=_clean_description(description),
                example_value=None,
            ))

    return attrs


def _parse_computed_attributes(attr_section: str) -> List[TerraformAttribute]:
    """Parse attribute reference (computed/exported attributes)."""
    attrs: List[TerraformAttribute] = []

    # Pattern: * `name` - Description
    pattern = re.compile(r"^\*\s+`(\w+)`\s*-\s*(.+?)(?=\n\*\s+`|\n#{1,3}\s|\Z)", re.DOTALL | re.MULTILINE)

    for match in pattern.finditer(attr_section):
        name = match.group(1)
        description = match.group(2).strip()

        attrs.append(TerraformAttribute(
            name=name,
            type=_infer_type(description),
            required=False,
            description=_clean_description(description),
            example_value=None,
        ))

    return attrs


def _infer_type(description: str) -> str:
    """Infer Terraform type from description text."""
    desc_lower = description.lower()
    if any(w in desc_lower for w in ("boolean", "true or false", "true` or `false")):
        return "bool"
    if any(w in desc_lower for w in ("map of", "map(", "key-value")):
        return "map(string)"
    if any(w in desc_lower for w in ("list of", "list(", "one or more")):
        return "list(string)"
    if any(w in desc_lower for w in ("number", "integer", "seconds", "days", "minutes")):
        return "number"
    return "string"


def _clean_description(desc: str) -> str:
    """Clean up description text — remove markdown links, collapse whitespace."""
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)  # [text](url) → text
    desc = re.sub(r"\s+", " ", desc)
    return desc.strip()[:500]  # Cap at 500 chars


def _extract_example(description: str, attr_type: str) -> Optional[Any]:
    """Extract example value from description if present."""
    # Look for Default: patterns
    default_match = re.search(r"Default[s]?:\s*[:`]*\s*([^`\s,.]+)", description)
    if default_match:
        val = default_match.group(1).strip("`")
        if attr_type == "bool":
            return val.lower() in ("true", "1")
        if attr_type == "number":
            try:
                return int(val)
            except ValueError:
                return None
        return val
    return None


# ============================================================================
# Search Result Parsers
# ============================================================================

def _extract_provider_doc_id(
    search_result: str,
    resource_name: str,
) -> Optional[str]:
    """
    Extract the provider_doc_id for a specific resource from search results.

    Search result format:
      - providerDocID: 11821678
      - Title: s3_bucket
      ---
    """
    # Strip "aws_" prefix for title matching
    target_title = resource_name.removeprefix("aws_")
    lines = str(search_result).splitlines()

    current_doc_id = None
    for line in lines:
        line = line.strip()
        if line.startswith("- providerDocID:"):
            current_doc_id = line.split(":", 1)[1].strip()
        elif line.startswith("- Title:"):
            title = line.split(":", 1)[1].strip()
            if title == target_title:
                return current_doc_id

    return None


def _extract_top_module_id(search_result: str) -> Optional[str]:
    """Extract the first module_id from a search_modules result."""
    for line in str(search_result).splitlines():
        if "module_id:" in line:
            parts = line.split("module_id:", 1)
            if len(parts) > 1:
                val = parts[1].strip()
                if "format:" not in val.lower():  # Skip the header instruction
                    return val
    return None


def _extract_policy_id(search_result: str) -> Optional[str]:
    """Extract the first terraform_policy_id from a search_policies result."""
    for line in str(search_result).splitlines():
        if "terraform_policy_id:" in line:
            parts = line.split("terraform_policy_id:", 1)
            if len(parts) > 1:
                val = parts[1].strip()
                if "unique identifier" not in val.lower():  # Skip the header instruction
                    return val
    return None


def _truncate_module_details(details: str, max_chars: int = 4000) -> str:
    """Truncate module details to fit in a prompt, keeping inputs/outputs."""
    if len(details) <= max_chars:
        return details
    # Prioritize inputs and outputs tables
    lines = details.splitlines()
    kept = []
    in_table = False
    for line in lines:
        if "### Inputs" in line or "### Outputs" in line:
            in_table = True
        if in_table:
            kept.append(line)
            if len("\n".join(kept)) > max_chars:
                break
        elif len("\n".join(kept)) < max_chars // 4:
            kept.append(line)
    return "\n".join(kept)[:max_chars]
