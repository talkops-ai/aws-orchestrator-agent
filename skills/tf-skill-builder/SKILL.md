---
name: tf-skill-builder
description: >
  Generates per-service Agent Skills directories that guide the tf-generator
  subagent for any AWS service. Creates SKILL.md with YAML frontmatter,
  step-by-step workflow instructions, and references/ with HCL resource
  patterns, variable schemas, and output schemas. Use when no skill directory
  exists for the requested AWS service, or when existing skills have a stale
  provider-version and need regeneration.
compatibility: >
  Requires read_file and write_file tools for virtual filesystem operations.
  Designed for LangChain Deep Agents with StateBackend or StoreBackend.
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: read_file write_file ls
---

# Terraform Skill Builder

## Overview

You generate Agent Skills directories that teach the tf-generator subagent how
to produce production-grade Terraform modules for a specific AWS service. Each
skill directory you create becomes a reusable knowledge artifact that the
tf-generator loads automatically via progressive disclosure.

## Workflow

Progress:
- [ ] Step 1: Load organizational standards
- [ ] Step 2: Check for existing skills
- [ ] Step 3: Determine the file set
- [ ] Step 4: Create SKILL.md with frontmatter
- [ ] Step 5: Create reference files
- [ ] Step 6: Return summary

### 1. Load Organizational Standards

Read persistent conventions the organization follows:

```
read_file /memories/org-standards.md
```

If the file does not exist, proceed with industry-standard defaults:
- Provider locking with `>=` constraint
- `merge()` tagging pattern
- `count` with local boolean guards for conditional creation

Also check for existing reference patterns:

```
read_file /memories/examples/
```

Use any reference patterns found as templates for new skills.

### 2. Check for Existing Skills

Before creating a new skill, check if one already exists:

```
ls /skills/
```

If `/skills/{service}-module-generator/SKILL.md` exists:
- Read it and check `metadata.provider-version` in the YAML frontmatter
- If the version matches the current AWS provider version, **STOP** — skill is up to date
- If the version is stale, proceed with regeneration (overwrite)

### 3. Determine the File Set

Every module skill MUST declare which `.tf` files the generator should create.

**Always include** (mandatory for every module):
- `main.tf` — primary resource declarations
- `variables.tf` — input variable definitions
- `outputs.tf` — output value definitions
- `versions.tf` — terraform and provider version constraints
- `locals.tf` — local value computations and feature toggles

**Include when the service requires it:**
- `policies.tf` — IAM policies, SCPs, permission boundaries (services with IAM-heavy workloads)
- `templates.tf` — user-data scripts, Helm values, rendered configs (services with bootstrap scripts)
- `data.tf` — data sources for AMI lookups, SSM params, existing resources (services needing external data)
- `iam.tf` — extensive IAM roles and instance profiles (services with dedicated execution roles)
- `security_groups.tf` — large security group rule sets (services with complex network access)

### 4. Create SKILL.md

Write the main skill file to `/skills/{service}-module-generator/SKILL.md`.

The file MUST follow this exact structure:

```yaml
---
name: {service}-module-generator
description: >
  Generates production-grade AWS {Service} Terraform modules with [key features].
  Use when asked to create a new {service} module, configure {service} infrastructure,
  or when the user mentions [service-specific trigger keywords].
compatibility: >
  Requires hashicorp/aws provider >= {major_version}.
  Terraform >= 1.5.
metadata:
  author: tf-planner
  version: "1.0"
  provider-version: "{full_provider_version}"
allowed-tools: write_file read_file edit_file ls grep
---

# {SERVICE} Module Generator

## Workflow
[Step-by-step instructions for generating the module]

## Key Patterns
[Tagging, conditional creation, safe outputs]

## Gotchas
[Deprecated attributes, common mistakes]
```

> **Critical**: The `description` field must be ≤ 1024 characters and include
> specific trigger keywords so the agent correctly matches this skill to
> relevant user prompts.

### 5. Create Reference Files

Create the following reference files under `/skills/{service}-module-generator/references/`:

**resource-patterns.md** — HCL block templates for every Terraform resource the
service uses. Each block must include:
- `count` with a local boolean guard
- Required and optional attributes
- `tags = merge(...)` pattern
- Deprecated attribute warnings

**variables-schema.md** — Input variable definitions table with columns:
Variable | Type | Default | Required | Description

**outputs-schema.md** — Output definitions using `try()` for conditional
resources.

### 6. Return Summary

Return exactly:
```
Skill written at /skills/{service}-module-generator/. Declared file set: [list of .tf files]
```

## Gotchas

- The `name` field in frontmatter MUST match the parent directory name exactly
- The `name` field MUST be lowercase with hyphens only (no underscores, no uppercase)
- Never include consecutive hyphens (`--`) in the name
- Keep the SKILL.md body under 500 lines — use references/ for detailed content
- Never hardcode region, account ID, or CIDR ranges in resource patterns
