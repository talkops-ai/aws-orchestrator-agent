"""
Skill Writer Orchestrator Tool
------------------------------
This module drives the pipeline's virtual filesystem serialization
for Agent Skills using the full pipeline output.

Consumes data from ALL upstream agents:
  - req_analyser: service discovery, terraform attributes, deployment context
  - sec_n_best_practices: security analysis, best practices analysis
  - execution_planner: module plans, config optimizer, state mgmt, execution plans
"""
import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.utils import create_file_data

from aws_orchestrator_agent.core.state import TFPlannerState
from aws_orchestrator_agent.utils import AgentLogger
from .skill_templates import (
    render_skill_md,
    render_resource_patterns,
    render_variables_schema,
    render_outputs_schema,
    render_security_rules,
    render_execution_blueprint,
    render_state_management,
    render_versions_tf_template,
    render_validation_script,
)

logger = AgentLogger("SKILL_WRITER")

def _check_skill_exists(state: Any, skill_dir: str, current_version: str) -> bool:
    """Check if skill already exists and is up-to-date."""
    import yaml
    
    files = state.get("files", {})
    skill_md_path = f"{skill_dir}/SKILL.md"
    if skill_md_path not in files:
        return False
        
    try:
        skill_data = files[skill_md_path]
        content = skill_data.get("content", "")
        if isinstance(content, list):
            content = "\n".join(content)
            
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            if frontmatter:
                skill_version = str(frontmatter.get("metadata", {}).get("provider-version", ""))
                if skill_version == current_version:
                    return True
    except Exception:
        pass
        
    return False

def _slugify(service_name: str) -> str:
    """Convert a service name to a lowercase-hyphenated slug safe for paths.

    Examples:
        "S3 bucket"     → "s3-bucket"
        "VPC"           → "vpc"
        "AWS Lambda"    → "aws-lambda"
    """
    import re
    slug = service_name.lower().strip()
    # Replace whitespace and underscores with hyphens
    slug = re.sub(r'[\s_]+', '-', slug)
    # Remove any characters that are not alphanumeric or hyphens
    slug = re.sub(r'[^a-z0-9\-]', '', slug)
    # Collapse multiple hyphens
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug


def _match_service(candidate: str, service_name: str) -> bool:
    """Return True if candidate string matches service_name with fuzzy logic.

    Normalises both sides by lowercasing, stripping AWS/Amazon prefix,
    and replacing spaces with hyphens before comparison.
    """
    def _norm(s: str) -> str:
        s = s.lower().replace("aws ", "").replace("amazon ", "").strip()
        # Treat spaces and hyphens as equivalent
        import re
        s = re.sub(r'[\s_-]+', '-', s)
        return s

    cn = _norm(candidate)
    sn = _norm(service_name)
    return cn == sn or cn in sn or sn in cn


def _find_service_attributes(tf_mapping: dict, service_name: str) -> List[Dict[str, Any]]:
    """Extract terraform_resources from tf_mapping for the given service."""
    for svc in tf_mapping.get("services", []):
        if _match_service(svc.get("service_name", ""), service_name):
            return svc.get("terraform_resources", [])
    return []


def _extract_security_for_service(sec_output: dict, service_name: str) -> dict:
    """Extract per-service security data from sec_output.

    Handles both the per-service and overall security analysis formats.
    """
    analysis = sec_output.get("security_analysis", {})
    services = analysis.get("services", [])
    
    if not isinstance(services, list):
        if isinstance(analysis, list):
            services = analysis
        else:
            return {}

    for svc_data in services:
        if _match_service(str(svc_data.get("service_name", "")), service_name):
            return svc_data
            
    return {}


def _extract_best_practices_for_service(sec_output: dict, service_name: str) -> dict:
    """Extract per-service best practices data from sec_output.

    Returns the service-level best practices findings including
    naming_and_tagging, module_structure, resource_optimization,
    and terraform_practices.
    """
    analysis = sec_output.get("best_practices_analysis", {})
    services = analysis.get("services", [])

    if not isinstance(services, list):
        return {}

    for svc_data in services:
        if _match_service(str(svc_data.get("service_name", "")), service_name):
            return svc_data

    # Fallback to overall best practices
    result: Dict[str, Any] = {}
    for key in ["naming_and_tagging", "module_structure", "resource_optimization", "terraform_practices"]:
        val = analysis.get(key, [])
        if val:
            result[key] = val
    return result


def _find_module_plan(exec_output: dict, service_name: str) -> dict:
    """Extract module_structure_plan from execution_planner_output."""
    for svc in exec_output.get("module_structure_plans", []):
        if _match_service(str(svc.get("service_name", "")), service_name):
            return svc
            
    # Fallback: old schema
    for svc in exec_output.get("service_execution_plans", []):
        if _match_service(str(svc.get("service_name", "")), service_name):
            return svc.get("module_structure_plan", {})
            
    return {}


def _find_config_optimizer(exec_output: dict, service_name: str) -> dict:
    """Extract config optimizer data from execution_planner_output."""
    for opt in exec_output.get("configuration_optimizers", []):
        if not isinstance(opt, dict):
            continue
        if _match_service(str(opt.get("service_name", "")), service_name):
            return opt

    return {}


def _find_execution_plan(exec_output: dict, service_name: str) -> dict:
    """Extract comprehensive execution plan from execution_planner_output."""
    for plan in exec_output.get("execution_plans", []):
        if not isinstance(plan, dict):
            continue
        if _match_service(str(plan.get("service_name", "")), service_name):
            return plan

    return {}


def _extract_deployment_context(req_output: dict) -> dict:
    """Extract deployment context (region, env, scope) from analysis_results."""
    analysis = req_output.get("analysis_results", {})
    deployment_ctx = analysis.get("deployment_context", "")

    region = ""
    environment = ""
    if isinstance(deployment_ctx, str):
        for part in deployment_ctx.split(","):
            part = part.strip()
            if part.lower().startswith("region:"):
                region = part.split(":", 1)[1].strip()
            elif part.lower().startswith("environment:"):
                environment = part.split(":", 1)[1].strip()

    return {
        "region": region,
        "environment": environment,
        "scope": analysis.get("scope_classification", ""),
        "deployment_context_raw": deployment_ctx,
    }


def _build_files_dict(skill_dir: str, skill_md: str, refs: dict) -> dict:
    files_dict = {}
    files_dict[f"{skill_dir}/SKILL.md"] = create_file_data(skill_md)
    for filename, content in refs.items():
        files_dict[f"{skill_dir}/{filename}"] = create_file_data(content)
    return files_dict


def _sync_skills_to_disk(skill_dir: str, skill_md: str, refs: dict) -> None:
    """Write skill files to local disk for persistence and inspection.

    Syncs generated skills to ./skills/ so that:
      1. Developers can inspect generated skills locally
      2. seed_files() picks them up on next session (avoids re-generation)
      3. Skills survive process restarts without re-running the planner

    Controlled by env var SKILL_WRITER_SYNC_DISK (default: "true").
    The disk path mirrors the VFS path: /skills/vpc-module-generator/ → ./skills/vpc-module-generator/
    """
    if os.getenv("SKILL_WRITER_SYNC_DISK", "true").lower() == "false":
        return

    # skill_dir = "/skills/vpc-module-generator" → local = "./skills/vpc-module-generator"
    local_root = Path(os.getenv("AGENT_PROJECT_ROOT", ".")).resolve()
    local_skill_dir = local_root / skill_dir.lstrip("/")

    try:
        # Write SKILL.md
        local_skill_dir.mkdir(parents=True, exist_ok=True)
        (local_skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

        # Write reference files (references/*.md, assets/*.tmpl, scripts/*.sh)
        for filename, content in refs.items():
            file_path = local_skill_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        logger.info(
            f"Synced skill to disk: {local_skill_dir}",
            extra={"file_count": 1 + len(refs), "disk_path": str(local_skill_dir)},
        )
    except OSError as e:
        logger.warning(f"Failed to sync skill to disk: {e}")

@tool
async def write_service_skills_tool(
    runtime: ToolRuntime[None, TFPlannerState],
) -> Command:
    """
    Write Agent Skills directories for each AWS service identified
    by the planner pipeline.

    Reads req_analyser_output, sec_n_best_practices_output, and
    execution_planner_output from state. For each service, generates:
      /skills/{service}-module-generator/SKILL.md
      /skills/{service}-module-generator/references/resource-patterns.md
      /skills/{service}-module-generator/references/variables-schema.md
      /skills/{service}-module-generator/references/outputs-schema.md
      /skills/{service}-module-generator/references/security-rules.md
      /skills/{service}-module-generator/references/execution-blueprint.md
      /skills/{service}-module-generator/assets/versions.tf.tmpl
      /skills/{service}-module-generator/scripts/validate.sh

    Consumes the FULL pipeline output:
      - Security analysis → security-rules.md, SKILL.md constraints
      - Best practices → security-rules.md, SKILL.md warnings
      - Config optimizer → naming/tagging, cost/perf optimizations
      - Execution plan → execution-blueprint.md, resource patterns, vars, outputs
      - Deployment context → SKILL.md header, variable defaults

    Skips generation if a matching skill directory already exists
    (preserves manually-authored skills like vpc-module-generator).
    """
    state_dict = runtime.state.copy()
    req_output = state_dict.get("req_analyser_output", {}) or {}
    sec_output = state_dict.get("sec_n_best_practices_output", {}) or {}
    exec_output = state_dict.get("execution_planner_output", {}) or {}

    aws_service_mapping = req_output.get("aws_service_mapping", {})
    if hasattr(aws_service_mapping, "model_dump"):
        aws_service_mapping = aws_service_mapping.model_dump(mode="json")
        
    services = aws_service_mapping.get("services", [])
    if not services:
        return Command(
            update={
                "messages": [ToolMessage(
                    content="No services mapped. 0 skills written.",
                    tool_call_id=runtime.tool_call_id,
                )]
            }
        )

    tf_mapping = req_output.get("terraform_attribute_mapping", {})
    if hasattr(tf_mapping, "model_dump"):
        tf_mapping = tf_mapping.model_dump(mode="json")

    provider_version = req_output.get("provider_version", "6.0")

    # ── Extract deployment context (shared across all services) ───────
    deployment_context = _extract_deployment_context(req_output)

    skills_written_manifest = []
    files_dict = {}

    for service in services:
        service_name = service.get("service") or service.get("service_name", "unknown")
        # Slugify the service name: "S3 bucket" → "s3-bucket", "VPC" → "vpc"
        # This ensures consistent VFS paths (no spaces/capitals) that match
        # what seed_files() loads from disk via Path.as_posix().
        service_slug = _slugify(service_name)
        skill_dir = f"/skills/{service_slug}-module-generator"

        if _check_skill_exists(state_dict, skill_dir, str(provider_version)):
            logger.info(f"Skill {skill_dir} already exists and is up-to-date, skipping generation to preserve overrides.")
            continue

        # ── Extract ALL upstream data for this service ────────────────
        service_attrs = _find_service_attributes(tf_mapping, service_name)
        security_data = _extract_security_for_service(sec_output, service_name)
        best_practices_data = _extract_best_practices_for_service(sec_output, service_name)
        module_plan = _find_module_plan(exec_output, service_name)
        config_optimizer = _find_config_optimizer(exec_output, service_name)
        execution_plan = _find_execution_plan(exec_output, service_name)

        # ── Extract state_management_plans for this service ───────────
        # State management is stored as a list of per-service plans.
        state_management = None
        sm_plans = exec_output.get("state_management_plans", [])
        if isinstance(sm_plans, list):
            for sm in sm_plans:
                if isinstance(sm, dict) and _match_service(
                    sm.get("service_name", sm.get("module_name", "")), service_name
                ):
                    state_management = sm
                    break

        has_exec_plan = bool(execution_plan)
        has_security = bool(security_data)
        has_best_practices = bool(best_practices_data)
        has_config_opt = bool(config_optimizer)
        has_state_mgmt = bool(state_management)

        logger.info(
            f"Rendering skill for {service_name}",
            extra={
                "has_execution_plan": has_exec_plan,
                "has_security": has_security,
                "has_best_practices": has_best_practices,
                "has_config_optimizer": has_config_opt,
                "has_state_management": has_state_mgmt,
            },
        )

        try:
            # ── Render SKILL.md (main skill file) ─────────────────────
            skill_md = render_skill_md(
                service_name, service, service_attrs, provider_version,
                execution_plan=execution_plan if has_exec_plan else None,
                config_optimizer=config_optimizer if has_config_opt else None,
                security_data=security_data if has_security else None,
                best_practices_data=best_practices_data if has_best_practices else None,
                deployment_context=deployment_context,
                service_slug=service_slug,
                state_management=state_management if has_state_mgmt else None,
                module_structure_plan=module_plan if module_plan else None,
            )

            # ── Render reference files ────────────────────────────────
            refs = {
                "references/resource-patterns.md": render_resource_patterns(
                    service_name, service_attrs,
                    execution_plan=execution_plan if has_exec_plan else None,
                ),
                "references/variables-schema.md": render_variables_schema(
                    service_name, module_plan, service_attrs,
                    execution_plan=execution_plan if has_exec_plan else None,
                ),
                "references/outputs-schema.md": render_outputs_schema(
                    service_name, service_attrs,
                    execution_plan=execution_plan if has_exec_plan else None,
                ),
                "references/security-rules.md": render_security_rules(
                    service_name, security_data,
                    best_practices_data=best_practices_data if has_best_practices else None,
                    execution_plan=execution_plan if has_exec_plan else None,
                    config_optimizer=config_optimizer if has_config_opt else None,
                    module_structure_plan=module_plan if module_plan else None,
                ),
                "assets/versions.tf.tmpl": render_versions_tf_template(provider_version),
                "scripts/validate.sh": render_validation_script(service_name),
            }

            # ── Render execution blueprint ────────────────────────────
            if has_exec_plan:
                refs["references/execution-blueprint.md"] = render_execution_blueprint(
                    service_name, execution_plan,
                    config_optimizer=config_optimizer if has_config_opt else None,
                )

            # ── Render state management reference (new) ──────────────
            if has_state_mgmt:
                refs["references/state-management.md"] = render_state_management(
                    service_name, state_management,
                )
            rendered_files = _build_files_dict(skill_dir, skill_md, refs)
            files_dict.update(rendered_files)

            # Sync to local disk for persistence and inspection
            _sync_skills_to_disk(skill_dir, skill_md, refs)

            skills_written_manifest.append({
                "service": service_name,
                "path": skill_dir,
                "files": list(rendered_files.keys()),
                "data_sources_used": {
                    "execution_plan": has_exec_plan,
                    "security_analysis": has_security,
                    "best_practices": has_best_practices,
                    "config_optimizer": has_config_opt,
                    "state_management": has_state_mgmt,
                },
            })
            
            logger.info(
                f"Successfully rendered skill files for {service_name}.",
                extra={"file_count": len(rendered_files)},
            )
        except Exception as e:
            logger.warning(f"Failed to render templates for {service_name}: {e}")

    # Append to existing files in state
    existing_files = state_dict.get("files", {}).copy()
    existing_files.update(files_dict)

    service_list = [m["service"] for m in skills_written_manifest]
    current_exec = exec_output.copy()
    current_exec["skills_written"] = skills_written_manifest

    return Command(
        update={
            "files": existing_files,
            "execution_planner_output": current_exec,
            "messages": [ToolMessage(
                content=f"Skills written for {len(skills_written_manifest)} service(s): {', '.join(service_list)}",
                tool_call_id=runtime.tool_call_id,
            )],
        }
    )
