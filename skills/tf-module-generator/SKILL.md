---
name: tf-module-generator
description: "Generates complete, production-ready Terraform .tf files for any AWS module. Follows loaded service-specific skills for HCL patterns, variable schemas, and output schemas. Use when asked to create a new Terraform module, generate infrastructure code, write HCL for any AWS service, or produce .tf files. Works with any AWS service that has hashicorp/aws Terraform provider support."
compatibility: "Requires write_file, read_file, edit_file, ls, and grep tools. Designed for LangChain Deep Agents with virtual filesystem backend. Terraform >= 1.5 and hashicorp/aws provider >= 5.0."
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: write_file read_file edit_file ls grep
---

# Terraform Module Generator

Generates complete, production-ready Terraform .tf files for AWS modules. Uses service-specific skills for exact resource definitions and [references/module-conventions.md](references/module-conventions.md) for mandatory patterns.

## Workflow

Progress:
- [ ] Step 1: Read your service-specific skill
- [ ] Step 2: Read reference files as directed
- [ ] Step 3: Generate all .tf files
- [ ] Step 4: Generate README.md
- [ ] Step 5: Configure State Backend (if applicable)
- [ ] Step 6: Apply mandatory patterns
- [ ] Step 7: Return summary

### 1. Read Your Service-Specific Skill

Your service-specific skill is NOT loaded automatically. Use `ls /skills/` to locate it and `read_file` to load it.

It contains:
- **SKILL.md**: Which files to create, in what order, with which resources
- **references/**: Detailed HCL patterns, variable schemas, output schemas

Follow the service-specific SKILL.md workflow exactly. If no service-specific skill exists, fall back to [references/module-conventions.md](references/module-conventions.md).

### 2. Read Reference Files

Load **only** what you need for the current file being generated:

- `main.tf` → `references/resource-patterns.md`
- `variables.tf` → `references/variables-schema.md`
- `outputs.tf` → `references/outputs-schema.md`
- Security resources → `references/security-rules.md`
- Full execution context → `references/execution-blueprint.md`

> **Context budget**: Do NOT read all references upfront. Read each only when writing the corresponding file.

### 3. Generate All .tf Files

Write every file to `./workspace/terraform_modules/{service}/{filename}`.

Minimum file set:
1. `versions.tf` — Terraform and provider version constraints
2. `locals.tf` — Feature toggles and computed values
3. `main.tf` — Primary resource declarations
4. `variables.tf` — Input variable definitions
5. `outputs.tf` — Output value definitions

Generate additional files as declared by the service-specific skill (`iam.tf`, `security_groups.tf`, `data.tf`, `policies.tf`).

### 4. Generate README.md

Always generate `./workspace/terraform_modules/{service}/README.md` with: module description, usage example with `module` block, inputs table, outputs table, and requirements (Terraform/provider versions).

### 5. Configure State Backend (if applicable)

If the service-specific SKILL.md specifies a state backend (e.g., in `references/state-management.md`), create `backend.tf` with the exact configuration specified.

### 6. Apply Mandatory Patterns

Non-negotiable patterns — see [references/module-conventions.md](references/module-conventions.md) for the full decision table.

**Conditional creation**: `count` for singleton (0-or-1) resources, `for_each` for keyed instances:
```hcl
resource "aws_{resource}" "this" {
  count = local.create_{resource} ? 1 : 0
}
```

**Tagging**: Always merge:
```hcl
tags = merge(
  { "Name" = var.name },
  var.tags,
  var.{resource}_tags,
)
```

**Safe outputs**: `try()` for conditional resources:
```hcl
output "{service}_id" {
  value = try(aws_{resource}.this[0].id, null)
}
```

**No hardcoded values**: Region, account ID, credentials, CIDR ranges must be variables.

**Provider locking**: Always use `>=` constraint in `versions.tf`.

### 7. Return Summary

Return exactly:
```
Generated {N} files: [list]. Key design decisions: [brief summary].
```

## Gotchas

- `execute` uses REAL paths (no leading `/`); `read_file`/`write_file` use VIRTUAL paths (with `/`). Never mix them.
- Every variable must have a non-empty `description`.
- Every resource must have a `Name` tag.
- With `count`, access attributes via `[0]` index: `aws_{resource}.this[0].id`.
- Use `try()` in outputs to handle `count = 0` cases.
- Never use `for_each` with a value that depends on resource attributes — causes "cannot be determined until apply" errors.
