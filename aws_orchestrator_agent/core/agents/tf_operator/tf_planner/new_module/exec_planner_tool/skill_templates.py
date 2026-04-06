"""
Skill Templates Factory
-----------------------
Production-grade template rendering engine that transforms the full
tf_planner pipeline output into agentskills.io compliant skill directories.

Consumes data from ALL upstream agents:
  - req_analyser: service discovery, terraform attributes, deployment context
  - sec_n_best_practices: security analysis, best practices analysis
  - execution_planner: module plans, config optimizer, state mgmt, execution plans
"""
import json
from typing import Dict, Any, List, Optional


# ============================================================================
# Dynamic Synonym Generation (replaces hardcoded _SERVICE_SYNONYMS)
# ============================================================================

def _build_service_synonyms(
    service_name: str,
    service: Dict[str, Any],
) -> List[str]:
    """Generate natural synonyms from upstream service data.

    Extracts keywords from production_features, description,
    architecture_patterns, and aws_service_type for trigger coverage.
    """
    synonyms: List[str] = []

    # From production features (e.g., ["Multi-AZ Subnets", "VPC Flow Logs"])
    for feat in service.get("production_features", []):
        if isinstance(feat, str) and feat not in synonyms:
            synonyms.append(feat)

    # From architecture patterns
    for pat in service.get("architecture_patterns", []):
        if isinstance(pat, dict):
            name = pat.get("pattern_name", "")
            if name and name not in synonyms:
                synonyms.append(name)

    # From aws_service_type
    svc_type = service.get("aws_service_type", "")
    if svc_type and svc_type != service_name and svc_type not in synonyms:
        synonyms.append(svc_type)

    # From description (extract key phrases)
    desc = service.get("description", "")
    if desc:
        # Take first meaningful chunk
        short = desc[:80].strip()
        if short and short not in synonyms:
            synonyms.append(short)

    # Fallback: service name itself
    if not synonyms:
        synonyms = [service_name]

    return synonyms[:6]  # Cap at 6 for prompt size


# ============================================================================
# SKILL.md — Production-Grade Main Skill File
# ============================================================================

def render_skill_md(
    service_name: str,
    service: Dict[str, Any],
    service_attrs: List[Dict[str, Any]],
    provider_version: str,
    execution_plan: Optional[Dict[str, Any]] = None,
    config_optimizer: Optional[Dict[str, Any]] = None,
    security_data: Optional[Dict[str, Any]] = None,
    best_practices_data: Optional[Dict[str, Any]] = None,
    deployment_context: Optional[Dict[str, Any]] = None,
    service_slug: Optional[str] = None,
    state_management: Optional[Dict[str, Any]] = None,
    module_structure_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the main SKILL.md with concrete, pipeline-grounded instructions.

    Unlike the generic version, this embeds:
      - Exact file list from execution plan (not hardcoded 4 files)
      - Architecture pattern and deployment context
      - Security requirements as hard constraints
      - Naming/tagging conventions from config optimizer
      - Cost optimization recommendations

    Args:
        service_slug: URL/path-safe slug for the skill name and directory
                      (e.g. "s3-bucket"). Defaults to service_name if omitted.
    """
    # ── Resource list for description ──────────────────────────────────
    # Derive the slug used in the YAML name field. Must be path-safe (no spaces).
    if not service_slug:
        import re
        _slug = service_name.lower().strip()
        _slug = re.sub(r'[\s_]+', '-', _slug)
        _slug = re.sub(r'[^a-z0-9\-]', '', _slug)
        _slug = re.sub(r'-+', '-', _slug).strip('-')
        service_slug = _slug

    resource_names = [r.get("resource_name", "") for r in service_attrs]
    if len(resource_names) > 5:
        resource_list = ", ".join(resource_names[:5]) + f", and {len(resource_names) - 5} more"
    else:
        resource_list = ", ".join(resource_names) if resource_names else service_name

    # ── Dynamic synonyms ──────────────────────────────────────────────
    synonyms = _build_service_synonyms(service_name, service)
    synonym_list = ", ".join(f'"{s}"' for s in synonyms[:5])

    # ── Feature toggles from terraform attributes ─────────────────────
    feature_toggles = []
    for attr_spec in service_attrs:
        if not isinstance(attr_spec, dict):
            continue
        design = attr_spec.get("module_design", {})
        if design and isinstance(design, dict):
            feature_toggles.extend(design.get("feature_toggles", []))

    seen = set()
    unique_toggles = []
    for f in feature_toggles:
        name_val = f.get("name") if isinstance(f, dict) else str(f)
        if not name_val:
            continue
        name = str(name_val)
        if name not in seen:
            seen.add(name)
            if isinstance(f, str):
                unique_toggles.append({"name": name, "description": "Auto-identified feature option"})
            else:
                unique_toggles.append(f)

    default_features = [f for f in unique_toggles if isinstance(f, dict) and f.get("default", True)]
    optional_features = [f for f in unique_toggles if isinstance(f, dict) and not f.get("default", True)]

    default_text = ""
    for f in default_features:
        default_text += f"- `{f.get('name')}`: {f.get('description', '')}\n"
    if not default_text:
        default_text = "- Standard resource creation\n"

    optional_text = ""
    for f in optional_features:
        optional_text += f"- `{f.get('name')}`: {f.get('description', '')}\n"
    if not optional_text:
        optional_text = "- No optional features identified.\n"

    feature_list = ", ".join([f.get("name", "") for f in unique_toggles[:3]])
    if not feature_list:
        feature_list = "essential resources"

    # ── Deprecated warnings ───────────────────────────────────────────
    deprecated = []
    for attr_spec in service_attrs:
        if not isinstance(attr_spec, dict):
            continue
        for dep in attr_spec.get("deprecated_attributes", []):
            if isinstance(dep, str):
                deprecated.append(f"- ~~`{dep}`~~")
            elif isinstance(dep, dict):
                desc = dep.get('description', '')[:100]
                if desc:
                    desc = "— " + desc
                deprecated.append(f"- ~~`{dep.get('name')}`~~ {desc}")

    deprecated_text = "\n".join(deprecated) if deprecated else "No known deprecated attributes or gotchas."

    provider_version_major = ".".join(provider_version.split(".")[:1])

    # ── Architecture pattern (from upstream) ──────────────────────────
    arch_pattern = ""
    overall_arch = service.get("overall_architecture_pattern", {})
    if overall_arch and isinstance(overall_arch, dict):
        arch_pattern = (
            f"\n**Architecture Pattern**: {overall_arch.get('pattern_name', 'Standard')}\n"
            f"> {overall_arch.get('description', '')}\n"
        )
    elif service.get("architecture_patterns"):
        pat = service["architecture_patterns"][0]
        if isinstance(pat, dict):
            arch_pattern = (
                f"\n**Architecture Pattern**: {pat.get('pattern_name', 'Standard')}\n"
                f"> {pat.get('description', '')}\n"
            )

    # ── Deployment context (from upstream) ────────────────────────────
    deploy_section = ""
    if deployment_context:
        region = deployment_context.get("region", "")
        env = deployment_context.get("environment", "")
        scope = deployment_context.get("scope", "")
        if region or env:
            deploy_section = f"\n**Deployment Target**: Region `{region}`, Environment `{env}`, Scope: {scope}\n"

    # ── File plan (from execution plan OR fallback) ───────────────────
    file_plan_section = ""
    if execution_plan and execution_plan.get("terraform_files"):
        file_plan_section = "\nWrite files to `./workspace/terraform_modules/{service_name}/` in this order:\n\n".format(
            service_name=service_name
        )
        for i, tf_file in enumerate(execution_plan["terraform_files"], 1):
            if isinstance(tf_file, dict):
                fname = tf_file.get("file_name", "")
                purpose = tf_file.get("file_purpose", "")
                resources = tf_file.get("resources_included", [])
                res_str = f" (resources: {', '.join(resources[:3])})" if resources else ""
                file_plan_section += f"{i}. `{fname}` — {purpose}{res_str}\n"
            else:
                file_plan_section += f"{i}. `{tf_file}`\n"
    else:
        file_plan_section = f"""
Write files to `./workspace/terraform_modules/{service_name}/` in this order:

1. `versions.tf` — Copy the template from [assets/versions.tf.tmpl](assets/versions.tf.tmpl) and fill in the provider version.
2. `main.tf` — Declare the resources.
3. `variables.tf` — Input definitions.
4. `outputs.tf` — Output values.
"""

    # ── Security hard constraints (from upstream) ─────────────────────
    security_constraints = ""
    if security_data:
        constraints = []
        # Encryption requirements
        enc_rest = security_data.get("encryption_at_rest", {})
        enc_transit = security_data.get("encryption_in_transit", {})
        if isinstance(enc_rest, dict) and enc_rest.get("status") == "non_compliant":
            constraints.append("⚠️ **Encryption at rest** is non-compliant. MUST add KMS encryption.")
        if isinstance(enc_transit, dict) and enc_transit.get("status") == "non_compliant":
            constraints.append("⚠️ **Encryption in transit** is non-compliant. MUST enforce TLS.")
        # Network security
        net = security_data.get("network_security", {})
        if isinstance(net, dict):
            sgs = net.get("security_groups", {})
            nacls = net.get("network_acls", {})
            if isinstance(sgs, dict) and sgs.get("status") == "non_compliant":
                constraints.append("⚠️ **Security Groups** are missing. MUST define explicit SG rules.")
            if isinstance(nacls, dict) and nacls.get("status") == "non_compliant":
                constraints.append("⚠️ **Network ACLs** are missing. MUST define explicit NACL rules.")
        # IAM
        ac = security_data.get("access_controls", {})
        if isinstance(ac, dict):
            roles = ac.get("iam_roles", {})
            policies = ac.get("iam_policies", {})
            if isinstance(roles, dict) and roles.get("status") == "non_compliant":
                constraints.append("⚠️ **IAM roles** are missing. MUST define least-privilege roles.")
            if isinstance(policies, dict) and policies.get("status") == "non_compliant":
                constraints.append("⚠️ **IAM policies** are missing. MUST define explicit policies.")

        if constraints:
            security_constraints = "\n### Security Hard Constraints (from Security Analysis — MANDATORY)\n\n"
            security_constraints += "\n".join(constraints) + "\n"
            security_constraints += (
                "\nThese were identified by the security analysis agent. "
                "Failure to address them will produce a non-compliant module.\n"
            )

    # ── Naming & tagging conventions (from config optimizer) ──────────
    naming_section = ""
    if config_optimizer:
        naming_convs = config_optimizer.get("naming_conventions", [])
        tagging_strats = config_optimizer.get("tagging_strategies", [])
        if naming_convs or tagging_strats:
            naming_section = "\n### Naming & Tagging Standards (from Configuration Optimizer — MANDATORY)\n\n"
            for nc in naming_convs[:5]:
                if isinstance(nc, dict):
                    naming_section += (
                        f"- `{nc.get('resource_type', '')}`: "
                        f"use `{nc.get('recommended_name', '')}` "
                        f"({nc.get('convention_rule', '')})\n"
                    )
            for ts in tagging_strats[:3]:
                if isinstance(ts, dict):
                    req_tags = ts.get("required_tags", {})
                    if req_tags:
                        tag_str = ", ".join(f"`{k}={v}`" for k, v in list(req_tags.items())[:5])
                        naming_section += f"- `{ts.get('resource_name', '')}` required tags: {tag_str}\n"

    # ── Cost optimizations (from upstream) ────────────────────────────
    cost_section = ""
    cost_recs = service.get("cost_optimization_recommendations", [])
    if cost_recs:
        cost_section = "\n### Cost Optimization (from Requirements Analysis — SHOULD implement)\n\n"
        for rec in cost_recs[:4]:
            if isinstance(rec, dict):
                cost_section += (
                    f"- **{rec.get('category', '')}**: {rec.get('recommendation', '')} "
                    f"(savings: {rec.get('potential_savings', 'N/A')}, "
                    f"difficulty: {rec.get('implementation_difficulty', 'N/A')})\n"
                )
            elif isinstance(rec, str):
                cost_section += f"- {rec}\n"

    # ── Best practices warnings (from upstream) ───────────────────────
    bp_section = ""
    if best_practices_data:
        warnings = []
        for category in ["naming_and_tagging", "terraform_practices"]:
            findings = best_practices_data.get(category, [])
            for f in findings:
                if isinstance(f, dict) and f.get("status") == "WARN":
                    warnings.append(
                        f"- **[{f.get('id', '?')}]** {f.get('check', '')} → {f.get('recommendation', '')}"
                    )
        if warnings:
            bp_section = "\n### Best Practice Warnings (from Best Practices Analysis)\n\n"
            bp_section += "\n".join(warnings[:6]) + "\n"

    # ── Well-Architected Alignment (from service_discover_output) ─────
    wa_section = ""
    wa = service.get("well_architected_alignment", {})
    if wa and isinstance(wa, dict):
        wa_section = "\n### Well-Architected Framework Alignment\n\n"
        wa_section += "| Pillar | Key Decisions |\n|--------|---------------|\n"
        pillar_labels = {
            "operational_excellence": "Operational Excellence",
            "security": "Security",
            "reliability": "Reliability",
            "performance_efficiency": "Performance Efficiency",
            "cost_optimization": "Cost Optimization",
            "sustainability": "Sustainability",
        }
        for key, label in pillar_labels.items():
            value = wa.get(key, "")
            if value:
                # value is a plain string — emit it directly
                if isinstance(value, list):
                    value = " • ".join(str(v) for v in value[:3])
                wa_section += f"| **{label}** | {value} |\n"

    # ── Cross-service dependencies (from service_discover_output) ─────
    deps_section = ""
    svc_deps = service.get("cross_service_dependencies", service.get("dependencies", []))
    if svc_deps:
        deps_section = "\n### Cross-Service Dependencies\n\n"
        deps_section += "| Dependency | Variable | Description |\n|-----------|----------|-------------|\n"
        for dep in svc_deps:
            if isinstance(dep, dict):
                deps_section += (
                    f"| `{dep.get('service', '')}` "
                    f"| `{dep.get('required_variable', dep.get('variable', ''))}` "
                    f"| {dep.get('description', dep.get('type', ''))} |\n"
                )

    # ── Architecture pattern best practices ────────────────────────────
    arch_bp_section = ""
    overall_arch = service.get("overall_architecture_pattern", {})
    if not overall_arch and service.get("architecture_patterns"):
        overall_arch = service["architecture_patterns"][0] if service["architecture_patterns"] else {}
    arch_bps = overall_arch.get("best_practices", []) if isinstance(overall_arch, dict) else []
    if arch_bps:
        arch_bp_section = "\n### Architecture Best Practices\n\n"
        for bp in arch_bps:
            arch_bp_section += f"- {bp}\n"

    # ── Implementation priority (from config_optimizer) ───────────────
    priority_section = ""
    if config_optimizer:
        priorities = config_optimizer.get("implementation_priority", [])
        opt_summary = config_optimizer.get("optimization_summary", "")
        if priorities or opt_summary:
            priority_section = "\n### Implementation Priority (from Configuration Optimizer)\n\n"
            if opt_summary:
                priority_section += f"> {opt_summary}\n\n"
            for i, item in enumerate(priorities, 1):
                priority_section += f"{i}. {item}\n"

    # ── State management reference ─────────────────────────────────────
    state_section = ""
    if state_management:
        backend = state_management.get("backend_configuration", {})
        locking = state_management.get("state_locking_configuration", {})
        splitting = state_management.get("state_splitting_strategy", {})
        state_section = "\n### State Management\n\n"
        state_section += (
            f"> ⚠️ This module uses a **dedicated remote backend**. "
            f"Full backend configuration is in [references/state-management.md](references/state-management.md).\n\n"
        )
        if backend:
            state_section += f"- **S3 Backend**: `{backend.get('bucket_name', 'see state-management.md')}` "
            state_section += f"(key: `{backend.get('key_pattern', '')}`, KMS: `{backend.get('kms_key_id', '')}`)\n"
        if locking:
            state_section += f"- **DynamoDB Lock Table**: `{locking.get('table_name', '')}` "
            state_section += f"(region: `{locking.get('region', '')}`, PITR: `{locking.get('point_in_time_recovery', False)}`)\n"
        if splitting and splitting.get("state_files"):
            sf_names = [sf.get('name', '') for sf in splitting['state_files']]
            state_section += f"- **State Split Strategy**: {splitting.get('splitting_approach', '')}\n"
            state_section += f"  - State files: {', '.join(f'`{n}`' for n in sf_names)}\n"

    # ── Composability hints (from module_structure_plan) ───────────────
    composability_section = ""
    if module_structure_plan:
        reuse = module_structure_plan.get("reusability_guidance", {})
        impl_notes = module_structure_plan.get("implementation_notes", [])
        if isinstance(reuse, dict):
            hints = reuse.get("composability_hints", [])
            best = reuse.get("best_practices", [])
            if hints or best or impl_notes:
                composability_section = "\n### Composability & Reusability\n\n"
                for hint in hints:
                    composability_section += f"- {hint}\n"
                for bp in best:
                    composability_section += f"- {bp}\n"
                for note in impl_notes[:3]:
                    composability_section += f"- 📝 {note}\n"

    # ── Execution blueprint reference ─────────────────────────────────
    blueprint_ref = ""
    if execution_plan:
        blueprint_ref = """
### Execution Blueprint

A comprehensive execution blueprint is available at [references/execution-blueprint.md](references/execution-blueprint.md).
This contains the complete resource configurations, IAM policies, local values, data sources,
and deployment phases pre-planned by the execution planner. **Use it as the authoritative
reference** for resource configuration details, lifecycle rules, and dependency chains.
"""

    # ── Assemble template ─────────────────────────────────────────────
    # Use service_slug for the YAML name: field (must be path-safe).
    # Use service_name.upper() for the human-readable H1 heading.
    template = f"""---
name: {service_slug}-module-generator
description: >
  Generates production-grade AWS {service_name.upper()} Terraform modules with {feature_list}.
  Use when asked to create, configure, or deploy {service_name} infrastructure,
  Terraform modules involving {resource_list}, or when the user mentions
  {synonym_list}. Also use when modifying existing {service_name} modules
  or adding {service_name}-related resources to a project.
compatibility: >
  Requires hashicorp/aws provider >= {provider_version_major}.
  Terraform >= 1.5. Generated HCL uses features from AWS provider {provider_version}.
allowed-tools: write_file read_file edit_file ls grep
metadata:
  author: tf-planner
  version: "1.0"
  generated-by: planner-skill-writer
  provider-version: "{provider_version}"
---

# {service_name.upper()} Module Generator
{arch_pattern}{deploy_section}{wa_section}{arch_bp_section}{deps_section}
## Workflow

Progress:
- [ ] Step 1: Determine required features from user request
- [ ] Step 2: Read relevant reference files (load only what's needed)
- [ ] Step 3: Create module files using the execution blueprint
- [ ] Step 4: Apply security hard constraints
- [ ] Step 5: Configure state backend
- [ ] Step 6: Validate the module (run scripts/validate.sh)
- [ ] Step 7: Review gotchas and verify no deprecated attributes used
- [ ] Step 8: Return summary of generated files

### 1. Determine Required Features

By default, include these features (disable only if the user explicitly opts out):
{default_text}
Include these features only if the user requests them:
{optional_text}
### 2. Read Reference Files

Load **only** the references you need for the user's request:

- **Read first**: [execution-blueprint.md](references/execution-blueprint.md) — Pre-planned resource configurations, variable definitions, output definitions, IAM policies, and deployment phases. This is your **primary reference**.
- **Read for HCL patterns**: [resource-patterns.md](references/resource-patterns.md) — HCL block templates for all {service_name} resources with lifecycle rules and dependency chains.
- **Read when writing variables.tf**: [variables-schema.md](references/variables-schema.md) — Input variable definitions with types, defaults, validation rules, and example values.
- **Read when writing outputs.tf**: [outputs-schema.md](references/outputs-schema.md) — Output definitions with exact value expressions, preconditions, and consumption notes.
- **Read when encryption or access control is involved**: [security-rules.md](references/security-rules.md) — CIS policies, encryption requirements, IAM policy documents, network security rules.
- **Read when configuring the backend**: [state-management.md](references/state-management.md) — S3 backend, DynamoDB lock table, state splitting strategy, IAM policies, DR procedures.

> **Context budget**: Read execution-blueprint.md first for the overall plan, then reference specific files only as needed for each .tf file you generate.

### 3. Create Module Files
{file_plan_section}
### 4. Apply Key Patterns

These patterns are critical — here's what they do and why they matter:

- **Conditional creation**: Use `count` with local boolean guards, because users need to toggle resources on/off without destroying the entire module.
- **Tagging**: Always use `merge({{"Name" = "..."}}, var.tags, var.{service_name}_tags)`, because AWS resources without Name tags are invisible in the console, and the merge pattern lets users override per-resource tags without losing global defaults.
- **Safe outputs**: Use `try(aws_some_resource.this[0].id, null)` for conditional resources, because referencing `.this[0]` on a resource with `count = 0` causes a Terraform error — `try()` returns null gracefully.
- **No hardcoded values**: Region, account ID, credentials must be variables, because hardcoded values make the module non-portable and fail in multi-account/multi-region deployments.
{security_constraints}{naming_section}{cost_section}{bp_section}{priority_section}{blueprint_ref}
### 5. Configure State Backend
{state_section}
> See [references/state-management.md](references/state-management.md) for the full backend HCL, DynamoDB lock table, state splitting strategy, and DR procedure.

### 6. Validate

Run the bundled validation script:
```bash
bash scripts/validate.sh ./workspace/terraform_modules/{service_name}/
```
If validation fails, review the error, fix the issue in the relevant .tf file, and re-run.

### 7. Gotchas & Composability
{composability_section}
{deprecated_text}

### 8. Return Summary

Return: "Generated N files: [list]. Key design decisions: [summary]."
"""
    return template


# ============================================================================
# Resource Patterns — HCL blocks with execution plan enrichment
# ============================================================================

def render_resource_patterns(
    service_name: str,
    service_attrs: List[Dict[str, Any]],
    execution_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """Render resource-patterns.md with HCL blocks enriched by execution plan.

    When execution_plan is available, resource blocks include:
      - lifecycle rules (prevent_destroy, create_before_destroy)
      - depends_on chains from the planned dependency graph
      - configuration hints from resource_configurations
    """
    def _resource_short_name(resource_name: str) -> str:
        return resource_name.removeprefix("aws_")

    # Build lookup from execution plan resource_configurations
    exec_configs: Dict[str, Dict[str, Any]] = {}
    exec_locals: List[Dict[str, Any]] = []
    exec_data_sources: List[Dict[str, Any]] = []
    if execution_plan:
        for rc in execution_plan.get("resource_configurations", []):
            if isinstance(rc, dict):
                key = rc.get("resource_type", rc.get("resource_name", ""))
                if key:
                    exec_configs[key] = rc
        exec_locals = execution_plan.get("local_values", [])
        exec_data_sources = execution_plan.get("data_sources", [])

    toc = []
    blocks = []

    for resource in service_attrs:
        res_name = resource.get("resource_name", "")
        if not res_name:
            continue

        short = _resource_short_name(res_name)
        toc.append(f"- [{res_name}](#{res_name.replace('_', '-')})")

        req_attrs = resource.get("required_attributes", [])
        opt_attrs = resource.get("optional_attributes", [])
        dep_attrs = resource.get("deprecated_attributes", [])

        req_lines = ""
        for attr in req_attrs:
            name = attr.get('name') if isinstance(attr, dict) else str(attr)
            req_lines += f"  {name} = var.{name}\n"

        opt_lines = ""
        for attr in opt_attrs[:5]:
            if isinstance(attr, str):
                opt_lines += f"  {attr} = var.{attr}\n"
            elif isinstance(attr, dict):
                name = attr.get('name')
                desc = attr.get('description', '')[:50]
                atype = attr.get('type', 'any')
                opt_lines += f"  {name} = var.{name}  # {atype} — {desc}\n"

        dep_lines = ""
        for attr in dep_attrs:
            if isinstance(attr, str):
                dep_lines += f"- ~~`{attr}`~~\n"
            elif isinstance(attr, dict):
                dep_lines += f"- ~~`{attr.get('name')}`~~ — **Deprecated**: {attr.get('description', '')[:80]}\n"

        if not dep_lines:
            dep_lines = "No known deprecated attributes."

        # ── Execution plan enrichment for this resource ───────────────
        lifecycle_block = ""
        depends_on_block = ""
        config_hints = ""
        exec_rc = exec_configs.get(res_name, {})
        if exec_rc:
            # Lifecycle rules
            lc = exec_rc.get("lifecycle_rules", {})
            if lc and isinstance(lc, dict):
                lc_lines = []
                if lc.get("prevent_destroy"):
                    lc_lines.append("    prevent_destroy = true")
                if lc.get("create_before_destroy"):
                    lc_lines.append("    create_before_destroy = true")
                if lc.get("ignore_changes"):
                    ign = lc["ignore_changes"]
                    if isinstance(ign, list):
                        ign_str = ", ".join(str(i) for i in ign)
                        lc_lines.append(f"    ignore_changes = [{ign_str}]")
                if lc_lines:
                    lifecycle_block = "\n  lifecycle {\n" + "\n".join(lc_lines) + "\n  }\n"

            # Depends on
            deps = exec_rc.get("depends_on", [])
            if deps:
                deps_str = ", ".join(str(d) for d in deps[:5])
                depends_on_block = f"\n  depends_on = [{deps_str}]\n"

            # Configuration hints
            config = exec_rc.get("configuration", {})
            if config and isinstance(config, dict):
                hints = []
                for k, v in list(config.items())[:5]:
                    hints.append(f"  # Planned: {k} = {json.dumps(v) if not isinstance(v, str) else v}")
                if hints:
                    config_hints = "\n  # ── Execution Plan Configuration Hints ──\n" + "\n".join(hints) + "\n"

            # Parameter justification
            justification = exec_rc.get("parameter_justification", "")
            if justification:
                config_hints += f"\n  # Justification: {justification}\n"

        block = f"""## {res_name}

```hcl
resource "{res_name}" "this" {{
  count = var.create_{short} ? 1 : 0

  # Required attributes
{req_lines}
  # Optional attributes (include when relevant)
{opt_lines}{config_hints}
  tags = merge(
    {{ "Name" = var.name }},
    var.tags,
    var.{short}_tags,
  )
{lifecycle_block}{depends_on_block}}}
```

### Deprecated Attributes — DO NOT USE

{dep_lines}
"""
        blocks.append(block)

    # ── Local values section (from execution plan) ────────────────────
    locals_section = ""
    if exec_locals:
        locals_section = "\n---\n\n## Local Values (Pre-planned)\n\n```hcl\nlocals {\n"
        for lv in exec_locals[:10]:
            if isinstance(lv, dict):
                name = lv.get("name", "")
                expr = lv.get("expression", "")
                desc = lv.get("description", "")
                locals_section += f"  # {desc}\n  {name} = {expr}\n\n"
        locals_section += "}\n```\n"

    # ── Data sources section (from execution plan) ────────────────────
    data_section = ""
    if exec_data_sources:
        data_section = "\n---\n\n## Data Sources (Pre-planned)\n\n"
        for ds in exec_data_sources[:5]:
            if isinstance(ds, dict):
                ds_type = ds.get("data_source_type", "")
                ds_name = ds.get("resource_name", "")
                ds_desc = ds.get("description", "")
                ds_config = ds.get("configuration", {})
                data_section += f"### {ds_type}.{ds_name}\n\n{ds_desc}\n\n```hcl\n"
                data_section += f'data "{ds_type}" "{ds_name}" {{\n'
                for k, v in ds_config.items():
                    data_section += f"  {k} = {json.dumps(v) if not isinstance(v, str) else v}\n"
                data_section += "}\n```\n\n"

    toc_text = "\n".join(toc)
    blocks_text = "\n---\n\n".join(blocks)

    return f"""# {service_name.upper()} Resource Patterns

## Table of Contents
{toc_text}

---

{blocks_text}
{locals_section}{data_section}"""


# ============================================================================
# Variables Schema — enriched with execution plan definitions
# ============================================================================

def render_variables_schema(
    service_name: str,
    module_plan: Dict[str, Any],
    service_attrs: List[Dict[str, Any]],
    execution_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """Render variables-schema.md with execution plan enrichment.

    When execution_plan is available, variables include:
      - validation_rules — exact Terraform validation blocks
      - example_values — sample values for documentation
      - justification — why this variable exists
    """
    var_rows = ""

    # Prefer execution plan variable definitions (richest data)
    source_vars = []
    if execution_plan and execution_plan.get("variable_definitions"):
        source_vars = execution_plan["variable_definitions"]
    elif module_plan and module_plan.get("variable_definitions"):
        source_vars = module_plan["variable_definitions"]

    # Detailed variable blocks (for rich data)
    detailed_blocks = []

    if source_vars:
        for var in source_vars:
            if isinstance(var, str):
                if var in ["name", "tags", f"create_{service_name}"]:
                    continue
                var_rows += f"| `{var}` | `any` | — | | |\n"
            elif isinstance(var, dict):
                name_val = var.get("name")
                if not name_val:
                    continue
                name = str(name_val)
                if name in ["name", "tags", f"create_{service_name}"]:
                    continue
                vtype = var.get("type", "any")
                default = str(var.get("default_value", var.get("default", "—")))
                desc = var.get("description", "")[:100]
                req = "✓" if var.get("required", False) or var.get("sensitive", False) else ""
                var_rows += f"| `{name}` | `{vtype}` | `{default}` | {req} | {desc} |\n"

                # Build detailed block if rich data exists
                validation_rules = var.get("validation_rules", [])
                example_values = var.get("example_values", [])
                justification = var.get("justification", "")
                nullable = var.get("nullable", False)
                sensitive = var.get("sensitive", False)

                if validation_rules or example_values or justification:
                    detail = f"\n### `{name}`\n\n"
                    if justification:
                        detail += f"**Why**: {justification}\n\n"
                    if sensitive:
                        detail += "**⚠️ Sensitive**: This variable is marked sensitive.\n\n"
                    if nullable:
                        detail += "**Nullable**: `true`\n\n"
                    if validation_rules:
                        detail += "**Validation Rules**:\n"
                        for vr in validation_rules[:3]:
                            if isinstance(vr, dict):
                                detail += f"- condition: `{vr.get('condition', '')}`\n"
                                detail += f"  error: `{vr.get('error_message', '')}`\n"
                            else:
                                detail += f"- `{vr}`\n"
                    if example_values:
                        detail += f"\n**Example Values**: `{json.dumps(example_values[:3])}`\n"
                    detailed_blocks.append(detail)
    else:
        # Fallback to service attrs
        seen = set(["name", "tags", f"create_{service_name}"])
        for r in service_attrs:
            if not isinstance(r, dict):
                continue
            for attr in r.get("required_attributes", []):
                name_val = attr.get("name") if isinstance(attr, dict) else str(attr)
                if not name_val:
                    continue
                name = str(name_val)
                if name not in seen:
                    desc = attr.get('description', '')[:100] if isinstance(attr, dict) else ""
                    atype = attr.get('type', 'any') if isinstance(attr, dict) else "any"
                    var_rows += f"| `{name}` | `{atype}` | — | ✓ | {desc} |\n"
                    seen.add(name)
            for attr in r.get("optional_attributes", []):
                name_val = attr.get("name") if isinstance(attr, dict) else str(attr)
                if not name_val:
                    continue
                name = str(name_val)
                if name not in seen:
                    desc = attr.get('description', '')[:100] if isinstance(attr, dict) else ""
                    atype = attr.get('type', 'any') if isinstance(attr, dict) else "any"
                    var_rows += f"| `{name}` | `{atype}` | `null` | | {desc} |\n"
                    seen.add(name)

    detailed_text = "\n".join(detailed_blocks) if detailed_blocks else ""
    detail_header = "\n## Variable Details\n" if detailed_text else ""

    return f"""# {service_name.upper()} Variables Schema

## Core

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `name` | `string` | — | ✓ | Name tag for all resources |
| `create_{service_name}` | `bool` | `true` | | Master toggle |
{var_rows}
## Tags

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `tags` | `map(string)` | `{{}}` | Tags applied to all resources |
{detail_header}{detailed_text}"""


# ============================================================================
# Outputs Schema — enriched with execution plan definitions
# ============================================================================

def render_outputs_schema(
    service_name: str,
    service_attrs: List[Dict[str, Any]],
    execution_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """Render outputs-schema.md with execution plan enrichment.

    When execution_plan is available, outputs include:
      - exact value expressions
      - depends_on chains
      - precondition checks
      - consumption_notes for downstream module usage
    """
    outputs = []
    seen = set()

    # Prefer execution plan outputs (richest data)
    if execution_plan and execution_plan.get("output_definitions"):
        for out in execution_plan["output_definitions"]:
            if not isinstance(out, dict):
                continue
            name = out.get("name", "")
            if not name or name in seen:
                continue
            seen.add(name)

            value = out.get("value", f"null")
            desc = out.get("description", name)[:120]
            sensitive = out.get("sensitive", False)
            depends = out.get("depends_on", [])
            precondition = out.get("precondition", {})
            consumption = out.get("consumption_notes", "")

            block = f'output "{name}" {{\n'
            block += f'  description = "{desc}"\n'
            block += f'  value       = {value}\n'
            if sensitive:
                block += "  sensitive   = true\n"
            if depends:
                deps_str = ", ".join(str(d) for d in depends[:3])
                block += f"  depends_on  = [{deps_str}]\n"
            if precondition and isinstance(precondition, dict):
                cond = precondition.get("condition", "")
                err_msg = precondition.get("error_message", "")
                if cond:
                    block += f"""
  precondition {{
    condition     = {cond}
    error_message = "{err_msg}"
  }}
"""
            block += "}"

            # Add consumption note as comment
            if consumption:
                block = f"# Usage: {consumption}\n{block}"

            outputs.append(block)
    else:
        # Fallback to service attrs
        for r in service_attrs:
            res_name = r.get("resource_name", "")
            if not res_name:
                continue

            short = res_name.removeprefix("aws_")
            attrs = r.get("computed_attributes", []) + r.get("optional_attributes", [])
            rec = r.get("module_design", {}).get("recommended_outputs", []) if isinstance(r.get("module_design"), dict) else []
            for out in rec:
                name_val = out.get("name") if isinstance(out, dict) else str(out)
                if not name_val:
                    continue
                name = str(name_val)
                if name not in seen:
                    desc = out.get("description", name) if isinstance(out, dict) else name
                    outputs.append(f"""output "{service_name}_{name}" {{
  description = "{desc}"
  value       = try({res_name}.this[0].{name}, null)
}}""")
                    seen.add(name)

            for attr in attrs[:5]:
                name_val = attr.get("name") if isinstance(attr, dict) else str(attr)
                if not name_val:
                    continue
                name = str(name_val)
                if name not in seen and name in ["id", "arn", "name"]:
                    desc = attr.get("description", name)[:100] if isinstance(attr, dict) else name[:100]
                    outputs.append(f"""output "{service_name}_{name}" {{
  description = "{desc}"
  value       = try({res_name}.this[0].{name}, null)
}}""")
                    seen.add(name)

    out_text = "\n\n".join(outputs)
    return f"""# {service_name.upper()} Outputs Schema

All outputs use `try()` to safely handle conditional resources.

## Core

```hcl
{out_text}
```
"""


# ============================================================================
# Security Rules — fixed keys, network security, best practices, IAM policies
# ============================================================================

def render_security_rules(
    service_name: str,
    security_data: Dict[str, Any],
    best_practices_data: Optional[Dict[str, Any]] = None,
    execution_plan: Optional[Dict[str, Any]] = None,
    config_optimizer: Optional[Dict[str, Any]] = None,
    module_structure_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """Render security-rules.md consuming BOTH security and best practices analysis.

    Fixes from the old version:
      - Correct key names matching actual security_analysis schema
      - Network security (SGs, NACLs) with CIS policy recommendations
      - Encryption in transit (was previously ignored)
      - Best practices terraform_practices findings
      - IAM policy documents from execution plan
      - Overall risk summary badge
      - Cross-service analysis and shared risks
      - Security optimizations from config_optimizer (current vs secure config)
      - security_considerations from module_structure_plan (CIS checklist)
    """
    if not security_data and not best_practices_data:
        return f"""# {service_name.upper()} Security & Compliance Rules

No explicit security rules provided. Follow general AWS best practices.
"""

    sec = security_data or {}

    # ── Encryption at Rest ────────────────────────────────────────────
    enc_rest_section = ""
    enc_rest = sec.get("encryption_at_rest", {})
    if isinstance(enc_rest, dict):
        status = enc_rest.get("status", "unknown")
        enc_rest_section = f"**Status**: `{status}`\n\n"
        for issue in enc_rest.get("issues", []):
            enc_rest_section += f"- ⚠️ {issue}\n"
        for rec in enc_rest.get("recommendations", []):
            enc_rest_section += f"- ✅ {rec}\n"
    if not enc_rest_section:
        enc_rest_section = "- Enable standard encryption at rest where applicable.\n"

    # ── Encryption in Transit ─────────────────────────────────────────
    enc_transit_section = ""
    enc_transit = sec.get("encryption_in_transit", {})
    if isinstance(enc_transit, dict):
        status = enc_transit.get("status", "unknown")
        enc_transit_section = f"**Status**: `{status}`\n\n"
        for issue in enc_transit.get("issues", []):
            enc_transit_section += f"- ⚠️ {issue}\n"
        for rec in enc_transit.get("recommendations", []):
            enc_transit_section += f"- ✅ {rec}\n"
    if not enc_transit_section:
        enc_transit_section = "- Enforce TLS for all data in transit.\n"

    # ── Network Security ──────────────────────────────────────────────
    network_section = ""
    net = sec.get("network_security", {})
    if isinstance(net, dict):
        for subsystem in ["security_groups", "network_acls"]:
            sub = net.get(subsystem, {})
            if isinstance(sub, dict):
                status = sub.get("status", "unknown")
                label = subsystem.replace("_", " ").title()
                network_section += f"\n### {label}\n\n**Status**: `{status}`\n\n"
                for issue in sub.get("issues", []):
                    network_section += f"- ⚠️ {issue}\n"
                for rec in sub.get("recommendations", []):
                    network_section += f"- ✅ {rec}\n"
    if not network_section:
        network_section = "- Define network security groups and NACLs per CIS benchmarks.\n"

    # ── Access Controls (IAM) ─────────────────────────────────────────
    access_section = ""
    ac = sec.get("access_controls", {})
    if isinstance(ac, dict):
        for subsystem in ["iam_roles", "iam_policies"]:
            sub = ac.get(subsystem, {})
            if isinstance(sub, dict):
                status = sub.get("status", "unknown")
                label = subsystem.replace("_", " ").title()
                access_section += f"\n### {label}\n\n**Status**: `{status}`\n\n"
                for issue in sub.get("issues", []):
                    access_section += f"- ⚠️ {issue}\n"
                for rec in sub.get("recommendations", []):
                    access_section += f"- ✅ {rec}\n"
    if not access_section:
        access_section = "- Restrict access to least privilege.\n"

    # ── Overall risk summary badge ─────────────────────────────────────
    risk_summary_section = ""
    overall_summary = sec.get("overall_summary", {})
    if isinstance(overall_summary, dict) and overall_summary:
        risk_level = overall_summary.get("overall_risk_level", "unknown")
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(risk_level, "⚪")
        total = overall_summary.get("total_services", 0)
        compliant = overall_summary.get("compliant_services", 0)
        non_compliant = overall_summary.get("non_compliant_services", 0)
        high_priority = overall_summary.get("high_priority_issues_count", 0)
        risk_summary_section = (
            f"\n## Risk Summary\n\n"
            f"{risk_emoji} **Overall Risk Level**: `{risk_level.upper()}`  "
            f"| Services: {total} total, {compliant} compliant, {non_compliant} non-compliant  "
            f"| High Priority Issues: {high_priority}\n"
        )

    # ── Security considerations from module_structure_plan ────────────
    module_sec_section = ""
    if module_structure_plan:
        mod_sec = module_structure_plan.get("security_considerations", [])
        if mod_sec:
            module_sec_section = "\n## Module Security Checklist (from Module Structure Plan)\n\n"
            for item in mod_sec:
                module_sec_section += f"- [ ] {item}\n"

    # ── Security optimizations from config_optimizer ──────────────────
    config_sec_section = ""
    if config_optimizer:
        sec_opts = config_optimizer.get("security_optimizations", [])
        if sec_opts:
            config_sec_section = "\n## Security Optimizations Required (from Configuration Optimizer)\n\n"
            config_sec_section += "| Resource | Severity | Current State | Required Configuration |\n"
            config_sec_section += "|----------|----------|---------------|------------------------|\n"
            for opt in sec_opts:
                if isinstance(opt, dict):
                    sev = opt.get("severity", "")
                    sev_emoji = {"Critical": "🚨", "High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(sev, "")
                    config_sec_section += (
                        f"| `{opt.get('resource_name', '')}` "
                        f"| {sev_emoji} {sev} "
                        f"| {opt.get('current_configuration', '—')} "
                        f"| {opt.get('secure_configuration', opt.get('justification', '—'))} |\n"
                    )

    # ── Cross-service analysis ────────────────────────────────────────
    cross_service_section = ""
    cross = sec.get("cross_service_analysis", {})
    if isinstance(cross, dict) and cross:
        cross_service_section = "\n## Cross-Service Security Analysis\n\n"
        svc_deps = cross.get("service_dependencies", {})
        if svc_deps:
            cross_service_section += "**Service Dependencies**:\n"
            for svc, deps in svc_deps.items():
                cross_service_section += f"- `{svc}` depends on: {', '.join(f'`{d}`' for d in deps)}\n"
        shared_risks = cross.get("shared_security_risks", [])
        if shared_risks:
            cross_service_section += "\n**Shared Security Risks**:\n"
            for risk in shared_risks:
                cross_service_section += f"- ⚠️ {risk}\n"
        cross_recs = cross.get("cross_service_recommendations", [])
        if cross_recs:
            cross_service_section += "\n**Cross-Service Recommendations**:\n"
            for rec in cross_recs:
                cross_service_section += f"- ✅ {rec}\n"

    # ── IAM Policy Documents (from execution plan) ────────────────────
    iam_section = ""
    if execution_plan and execution_plan.get("iam_policies"):
        iam_section = "\n## IAM Policy Documents (Pre-planned)\n\n"
        for iam in execution_plan["iam_policies"][:5]:
            if isinstance(iam, dict):
                policy_name = iam.get("policy_name", "unnamed")
                desc = iam.get("description", "")
                statements = iam.get("statements", [])
                refs = iam.get("resource_references", [])

                iam_section += f"### {policy_name}\n\n"
                if desc:
                    iam_section += f"{desc}\n\n"
                if refs:
                    iam_section += f"Referenced resources: {', '.join(refs)}\n\n"

                iam_section += "```json\n"
                iam_section += json.dumps({
                    "Version": iam.get("version", "2012-10-17"),
                    "Statement": statements,
                }, indent=2)
                iam_section += "\n```\n\n"

    # ── Service-level Issues & Recommendations ────────────────────────
    svc_issues_section = ""
    svc_issues = sec.get("service_issues", [])
    svc_recs = sec.get("service_recommendations", [])
    compliance = sec.get("service_compliance", sec.get("overall_compliance", "unknown"))
    if svc_issues or svc_recs:
        svc_issues_section = f"\n## Service Compliance: `{compliance}`\n\n"
        for issue in svc_issues:
            svc_issues_section += f"- ⚠️ {issue}\n"
        for rec in svc_recs:
            svc_issues_section += f"- ✅ {rec}\n"



    # ── Best Practices Findings ───────────────────────────────────────
    bp_section = ""
    if best_practices_data:
        bp_findings = []
        for category in ["naming_and_tagging", "module_structure", "resource_optimization", "terraform_practices"]:
            findings = best_practices_data.get(category, [])
            for f in findings:
                if isinstance(f, dict):
                    status_icon = "✅" if f.get("status") == "PASS" else "⚠️" if f.get("status") == "WARN" else "❌"
                    bp_findings.append(
                        f"- {status_icon} **[{f.get('id', '?')}]** {f.get('check', '')} → {f.get('recommendation', '')}"
                    )
        if bp_findings:
            bp_section = "\n## Best Practices Findings\n\n" + "\n".join(bp_findings) + "\n"

    return f"""# {service_name.upper()} Security & Compliance Rules
{risk_summary_section}
## Encryption at Rest
{enc_rest_section}
## Encryption in Transit
{enc_transit_section}
## Network Security
{network_section}
## Access Controls
{access_section}
{svc_issues_section}{module_sec_section}{config_sec_section}{cross_service_section}{iam_section}{bp_section}"""


# ============================================================================
# Execution Blueprint — full execution plan for downstream agent grounding
# ============================================================================

def render_execution_blueprint(
    service_name: str,
    execution_plan: Dict[str, Any],
    config_optimizer: Optional[Dict[str, Any]] = None,
) -> str:
    """Render execution-blueprint.md — the authoritative reference for the
    downstream code-generation agent.

    Contains the full execution plan: resource configs, variable defs,
    local values, data sources, IAM policies, deployment phases, and
    configuration optimizations.
    """
    ep = execution_plan

    # ── Module overview ───────────────────────────────────────────────
    overview = f"""# {service_name.upper()} Execution Blueprint

> **This file is the authoritative reference for generating Terraform code.**
> Use it to determine exact resource configurations, variable definitions,
> output definitions, IAM policies, local values, and deployment order.

## Module Overview

| Field | Value |
|-------|-------|
| **Service** | {ep.get('service_name', service_name)} |
| **Module Name** | {ep.get('module_name', f'{service_name}-module')} |
| **Target Environment** | {ep.get('target_environment', 'production')} |
| **Terraform Version** | {ep.get('terraform_version_constraint', '>= 1.5')} |
"""

    # Required providers
    req_providers = ep.get("required_providers", {})
    if req_providers:
        overview += "\n### Required Providers\n\n```hcl\n"
        for provider, config in req_providers.items():
            if isinstance(config, dict):
                overview += f'  {provider} = {{\n'
                for k, v in config.items():
                    overview += f'    {k} = "{v}"\n'
                overview += "  }\n"
            else:
                overview += f'  {provider} = "{config}"\n'
        overview += "```\n"

    # ── Module description ────────────────────────────────────────────
    desc = ep.get("module_description", "")
    if desc:
        overview += f"\n## Description\n\n{desc}\n"

    # ── File structure ────────────────────────────────────────────────
    files_section = ""
    tf_files = ep.get("terraform_files", [])
    if tf_files:
        files_section = "\n## File Structure\n\n"
        for tf in tf_files:
            if isinstance(tf, dict):
                fname = tf.get("file_name", "")
                purpose = tf.get("file_purpose", "")
                resources = tf.get("resources_included", [])
                deps = tf.get("dependencies", [])
                rationale = tf.get("organization_rationale", "")
                files_section += f"### `{fname}`\n\n"
                files_section += f"**Purpose**: {purpose}\n\n"
                if resources:
                    files_section += f"**Resources**: {', '.join(resources)}\n\n"
                if deps:
                    files_section += f"**Dependencies**: {', '.join(deps)}\n\n"
                if rationale:
                    files_section += f"**Rationale**: {rationale}\n\n"

    # ── Resource configurations ───────────────────────────────────────
    resource_section = ""
    rcs = ep.get("resource_configurations", [])
    if rcs:
        resource_section = "\n## Resource Configurations\n\n"
        for rc in rcs:
            if isinstance(rc, dict):
                rtype = rc.get("resource_type", rc.get("resource_name", "unknown"))
                rname = rc.get("resource_name", "this")
                config = rc.get("configuration", {})
                lifecycle = rc.get("lifecycle_rules", {})
                tags = rc.get("tags_strategy", "")
                deps = rc.get("depends_on", [])
                just = rc.get("parameter_justification", "")

                resource_section += f"### `{rtype}.{rname}`\n\n"
                if just:
                    resource_section += f"> {just}\n\n"
                if config:
                    resource_section += "**Configuration**:\n```json\n"
                    resource_section += json.dumps(config, indent=2)
                    resource_section += "\n```\n\n"
                if lifecycle:
                    resource_section += f"**Lifecycle**: `{json.dumps(lifecycle)}`\n\n"
                if tags:
                    resource_section += f"**Tags Strategy**: {tags}\n\n"
                if deps:
                    resource_section += f"**Depends On**: {', '.join(str(d) for d in deps)}\n\n"

    # ── IAM policies ──────────────────────────────────────────────────
    iam_section = ""
    iam_policies = ep.get("iam_policies", [])
    if iam_policies:
        iam_section = "\n## IAM Policies\n\n"
        for iam in iam_policies[:5]:
            if isinstance(iam, dict):
                policy_name = iam.get("policy_name", "unnamed")
                desc = iam.get("description", "")
                statements = iam.get("statements", [])
                refs = iam.get("resource_references", [])

                iam_section += f"### `{policy_name}`\n\n"
                if desc:
                    iam_section += f"{desc}\n\n"
                if refs:
                    iam_section += f"**Referenced resources**: {', '.join(refs)}\n\n"

                iam_section += "```json\n"
                iam_section += json.dumps({
                    "Version": iam.get("version", "2012-10-17"),
                    "Statement": statements,
                }, indent=2)
                iam_section += "\n```\n\n"

    # ── Deployment phases ─────────────────────────────────────────────
    phases_section = ""
    phases = ep.get("deployment_phases", [])
    if phases:
        phases_section = "\n## Deployment Phases\n\n"
        for i, phase in enumerate(phases, 1):
            phases_section += f"{i}. {phase}\n"

    # ── Usage examples ────────────────────────────────────────────────
    examples_section = ""
    examples = ep.get("usage_examples", [])
    if examples:
        examples_section = "\n## Usage Examples\n\n"
        for ex in examples:
            if isinstance(ex, dict):
                examples_section += f"### {ex.get('example_name', 'Example')}\n\n"
                examples_section += f"{ex.get('description', '')}\n\n"
                config = ex.get("configuration", "")
                if config:
                    examples_section += f"```hcl\n{config}\n```\n\n"
                use_case = ex.get("use_case", "")
                if use_case:
                    examples_section += f"**Use case**: {use_case}\n\n"

    # ── Config optimizer data ─────────────────────────────────────────
    optimizer_section = ""
    if config_optimizer:
        optimizer_section = "\n## Configuration Optimizations\n\n"

        cost_opts = config_optimizer.get("cost_optimizations", [])
        if cost_opts:
            optimizer_section += "### Cost Optimizations\n\n"
            for opt in cost_opts[:5]:
                if isinstance(opt, dict):
                    optimizer_section += (
                        f"- **{opt.get('resource_name', '')}**: "
                        f"{opt.get('justification', '')} "
                        f"(savings: {opt.get('estimated_savings', 'N/A')})\n"
                    )

        perf_opts = config_optimizer.get("performance_optimizations", [])
        if perf_opts:
            optimizer_section += "\n### Performance Optimizations\n\n"
            for opt in perf_opts[:5]:
                if isinstance(opt, dict):
                    optimizer_section += (
                        f"- **{opt.get('resource_name', '')}**: "
                        f"{opt.get('justification', '')} "
                        f"(impact: {opt.get('performance_impact', 'N/A')})\n"
                    )

        sec_opts = config_optimizer.get("security_optimizations", [])
        if sec_opts:
            optimizer_section += "\n### Security Optimizations\n\n"
            for opt in sec_opts[:5]:
                if isinstance(opt, dict):
                    optimizer_section += (
                        f"- **{opt.get('resource_name', '')}** [{opt.get('severity', '')}]: "
                        f"{opt.get('justification', '')}\n"
                    )

    # ── Estimated costs ───────────────────────────────────────────────
    costs_section = ""
    costs = ep.get("estimated_costs", {})
    if costs:
        costs_section = "\n## Estimated Costs\n\n```json\n"
        costs_section += json.dumps(costs, indent=2)
        costs_section += "\n```\n"

    # ── Validation & testing ──────────────────────────────────────────
    validation_section = ""
    validations = ep.get("validation_and_testing", [])
    if validations:
        validation_section = "\n## Validation & Testing\n\n"
        for v in validations:
            validation_section += f"- {v}\n"

    # ── Resource dependency graph ───────────────────────────────────────
    deps_graph_section = ""
    resource_deps = ep.get("resource_dependencies", [])
    if resource_deps:
        deps_graph_section = "\n## Resource Dependency Graph\n\n"
        deps_graph_section += "| Source | Dependent | Reason |\n|--------|-----------|--------|\n"
        for dep in resource_deps:
            if isinstance(dep, dict):
                deps_graph_section += (
                    f"| `{dep.get('source', '')}` "
                    f"| `{dep.get('dependent', '')}` "
                    f"| {dep.get('reason', '')} |\n"
                )

    return (
        overview
        + files_section
        + resource_section
        + iam_section
        + phases_section
        + examples_section
        + deps_graph_section
        + optimizer_section
        + costs_section
        + validation_section
    )


# ============================================================================
# State Management Reference
# ============================================================================

def render_state_management(
    service_name: str,
    state_management: Optional[Dict[str, Any]],
) -> str:
    """Render references/state-management.md from state_management.json data.

    Provides the downstream code-generation agent with complete guidance on:
      - S3 backend HCL configuration
      - DynamoDB lock table HCL configuration
      - State splitting strategy (which .tfstate files, team ownership)
      - terraform_remote_state data source usage
      - IAM and bucket policies for state access
      - Implementation steps (ordered)
      - State best practices
      - Monitoring and alerting setup
      - Disaster recovery procedures
    """
    if not state_management:
        return f"# {service_name.upper()} State Management\n\nNo state management plan available.\n"
    sm = state_management

    # ── S3 Backend configuration ────────────────────────────────────────
    backend_section = ""
    backend = sm.get("backend_configuration", {})
    if isinstance(backend, dict) and backend:
        backend_type = backend.get("backend_type", "s3")
        bucket = backend.get("bucket_name", "")
        key_pattern = backend.get("key_pattern", "")
        region = backend.get("region", "")
        kms = backend.get("kms_key_id", "")
        encrypt = backend.get("encrypt", True)
        versioning = backend.get("versioning", True)
        dynamodb_table = backend.get("dynamodb_table", "")
        backend_section = f"""
## S3 Backend Configuration

```hcl
terraform {{
  backend "{backend_type}" {{
    bucket         = "{bucket}"
    key            = "{key_pattern}"
    region         = "{region}"
    encrypt        = {str(encrypt).lower()}
    kms_key_id     = "{kms}"
    dynamodb_table = "{dynamodb_table}"
  }}
}}
```

> ⚠️ **Required**: S3 versioning: `{versioning}`. Never commit `.terraform/` or `*.tfstate` to VCS.
"""

    # ── DynamoDB lock table ───────────────────────────────────────────────
    locking_section = ""
    locking = sm.get("state_locking_configuration", {})
    if isinstance(locking, dict) and locking:
        table_name = locking.get("table_name", "")
        lk_region = locking.get("region", "")
        billing = locking.get("billing_mode", "PAY_PER_REQUEST")
        hash_key = locking.get("hash_key", "LockID")
        pitr = locking.get("point_in_time_recovery", False)
        sse = locking.get("server_side_encryption", False)
        locking_section = f"""
## DynamoDB Lock Table

```hcl
resource "aws_dynamodb_table" "terraform_locks" {{
  name         = "{table_name}"
  billing_mode = "{billing}"
  hash_key     = "{hash_key}"

  attribute {{
    name = "{hash_key}"
    type = "S"
  }}

  point_in_time_recovery {{
    enabled = {str(pitr).lower()}
  }}

  server_side_encryption {{
    enabled = {str(sse).lower()}
  }}

  tags = local.common_tags
}}
```

Region: `{lk_region}` | PITR: `{pitr}` | SSE: `{sse}`
"""

    # ── State splitting strategy ──────────────────────────────────────────
    splitting_section = ""
    splitting = sm.get("state_splitting_strategy", {})
    if isinstance(splitting, dict) and splitting:
        approach = splitting.get("splitting_approach", "")
        rationale = splitting.get("rationale", "")
        state_files = splitting.get("state_files", [])
        remote_state_usage = splitting.get("remote_state_data_source_usage", [])

        splitting_section = f"\n## State Splitting Strategy\n\n"
        splitting_section += f"> **Approach**: {approach}\n> **Rationale**: {rationale}\n\n"

        if state_files:
            splitting_section += "### State Files\n\n"
            splitting_section += "| State File | Team | Dependencies |\n|-----------|------|--------------|\n"
            for sf in state_files:
                if isinstance(sf, dict):
                    team = sf.get("team_ownership", sf.get("owner", "—"))
                    depends = sf.get("dependencies", [])
                    dep_str = ", ".join(f"`{d}`" for d in depends) if depends else "—"
                    splitting_section += f"| `{sf.get('name', '')}` | {team} | {dep_str} |\n"

        if remote_state_usage:
            splitting_section += "\n### Remote State Data Source Usage\n\n"
            for rs in remote_state_usage[:3]:
                if isinstance(rs, dict):
                    source_state = rs.get("source_state", "")
                    outputs = rs.get("outputs_consumed", [])
                    splitting_section += f"\n**From `{source_state}`**:\n"
                    splitting_section += "```hcl\n"
                    splitting_section += f'data "terraform_remote_state" "{source_state.replace("/", "_").replace("-", "_")}" {{\n'
                    splitting_section += f'  backend = "s3"\n'
                    splitting_section += f'  config = {{\n'
                    splitting_section += f'    bucket = var.state_bucket\n'
                    splitting_section += f'    key    = "{source_state}"\n'
                    splitting_section += f'    region = var.aws_region\n'
                    splitting_section += f'  }}\n'
                    splitting_section += f'}}\n'
                    splitting_section += "```\n"
                    if outputs:
                        splitting_section += f"Outputs consumed: {', '.join(f'`{o}`' for o in outputs)}\n\n"

    # ── Security recommendations for state ──────────────────────────────
    state_sec_section = ""
    state_sec = sm.get("security_recommendations", {})
    if isinstance(state_sec, dict) and state_sec:
        state_sec_section = "\n## State Access Security\n\n"
        iam_roles = state_sec.get("iam_roles", [])
        if iam_roles:
            state_sec_section += "### IAM Roles\n\n"
            for role in iam_roles:
                if isinstance(role, dict):
                    state_sec_section += f"- **`{role.get('role_name', '')}`** ({role.get('purpose', '')}): "
                    perms = role.get("permissions", [])
                    state_sec_section += ", ".join(f"`{p}`" for p in perms[:5]) + "\n"
        bucket_policies = state_sec.get("bucket_policies", [])
        if bucket_policies:
            state_sec_section += "\n### Bucket Policies\n\n"
            for policy in bucket_policies[:2]:
                if isinstance(policy, dict):
                    state_sec_section += f"**{policy.get('policy_name', '')}**: {policy.get('description', '')}\n"
                    if policy.get("conditions"):
                        conditions = policy["conditions"]
                        state_sec_section += "```json\n"
                        state_sec_section += json.dumps(conditions, indent=2)
                        state_sec_section += "\n```\n\n"
        access_controls = state_sec.get("access_controls", [])
        for ctrl in access_controls:
            state_sec_section += f"- {ctrl}\n"

    # ── Implementation steps ──────────────────────────────────────────────
    steps_section = ""
    impl_steps = sm.get("implementation_steps", [])
    if impl_steps:
        steps_section = "\n## Implementation Steps\n\n"
        for step in impl_steps:
            if isinstance(step, dict):
                steps_section += f"{step.get('step', '')}. **{step.get('action', '')}**"
                if step.get("command"):
                    steps_section += f"\n   ```bash\n   {step['command']}\n   ```"
                elif step.get("description"):
                    steps_section += f": {step['description']}"
                steps_section += "\n"
            elif isinstance(step, str):
                steps_section += f"- {step}\n"

    # ── Best practices ────────────────────────────────────────────────────
    bps_section = ""
    best_practices = sm.get("best_practices", [])
    if best_practices:
        bps_section = "\n## State Best Practices\n\n"
        for i, bp in enumerate(best_practices, 1):
            bps_section += f"{i}. {bp}\n"

    # ── Monitoring setup ────────────────────────────────────────────────
    monitoring_section = ""
    monitoring = sm.get("monitoring_setup", {})
    if isinstance(monitoring, dict) and monitoring:
        monitoring_section = "\n## Monitoring & Alerting\n\n"
        for tool, config in monitoring.items():
            if isinstance(config, dict):
                monitoring_section += f"### {tool.replace('_', ' ').title()}\n\n"
                for k, v in config.items():
                    if isinstance(v, list):
                        monitoring_section += f"- **{k}**: {', '.join(str(i) for i in v[:5])}\n"
                    elif v:
                        monitoring_section += f"- **{k}**: {v}\n"
            elif isinstance(config, list):
                monitoring_section += f"### {tool.replace('_', ' ').title()}\n\n"
                for item in config[:5]:
                    monitoring_section += f"- {item}\n"

    # ── Disaster recovery ────────────────────────────────────────────────
    dr_section = ""
    dr = sm.get("disaster_recovery", {})
    if isinstance(dr, dict) and dr:
        dr_section = "\n## Disaster Recovery\n\n"
        restore_proc = dr.get("restore_procedure", [])
        if restore_proc:
            dr_section += "### Restore Procedure\n\n"
            for i, step in enumerate(restore_proc, 1):
                dr_section += f"{i}. {step}\n"
        backup_retention = dr.get("backup_retention", "")
        rpo = dr.get("rpo", "")
        rto = dr.get("rto", "")
        if backup_retention or rpo or rto:
            dr_section += f"\n| Parameter | Value |\n|-----------|-------|\n"
            if backup_retention:
                dr_section += f"| Backup Retention | {backup_retention} |\n"
            if rpo:
                dr_section += f"| RPO | {rpo} |\n"
            if rto:
                dr_section += f"| RTO | {rto} |\n"

    return (
        f"# {service_name.upper()} State Management\n\n"
        f"> **Reference**: This file configures Terraform state for the {service_name} module.\n"
        f"> Generated from `state_management_plans` output. Use this to configure `backend.tf`.\n"
        + backend_section
        + locking_section
        + splitting_section
        + state_sec_section
        + steps_section
        + bps_section
        + monitoring_section
        + dr_section
    )


# ============================================================================
# Versions Template & Validation Script (unchanged)
# ============================================================================

def render_versions_tf_template(provider_version: str) -> str:
    provider_version_major = ".".join(provider_version.split(".")[:1])
    return f"""# assets/versions.tf.tmpl
# Copy this file to versions.tf and verify the provider version.
terraform {{
  required_version = ">= 1.5"

  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> {provider_version_major}"
    }}
  }}
}}
"""


def render_validation_script(service_name: str) -> str:
    return f"""#!/usr/bin/env bash
# scripts/validate.sh — Validate the generated Terraform module
# Usage: bash scripts/validate.sh <module-directory>
#
# Example:
#   bash scripts/validate.sh ./workspace/terraform_modules/{service_name}/
#
# Exit codes:
#   0 — Validation passed
#   1 — terraform init or validate failed

set -euo pipefail

MODULE_DIR="${{1:?Error: module directory required. Usage: $0 <module-directory>}}"

if [ ! -d "$MODULE_DIR" ]; then
  echo "Error: directory '$MODULE_DIR' does not exist" >&2
  exit 1
fi

echo "Validating module at: $MODULE_DIR"

cd "$MODULE_DIR"

echo "--- terraform init ---"
terraform init -backend=false -input=false 2>&1

echo "--- terraform validate ---"
terraform validate 2>&1

echo "✅ Module validation passed"
"""
