---
name: tf-module-generator
description: >
  Generates complete, production-ready Terraform .tf files for any AWS module.
  Follows loaded service-specific skills for HCL patterns, variable schemas,
  and output schemas. Use when asked to create a new Terraform module, generate
  infrastructure code, write HCL for any AWS service, or produce .tf files.
  Works with any AWS service that has hashicorp/aws Terraform provider support.
compatibility: >
  Requires write_file, read_file, edit_file, ls, and grep tools.
  Designed for LangChain Deep Agents with virtual filesystem backend.
  Terraform >= 1.5 and hashicorp/aws provider >= 5.0.
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: write_file read_file edit_file ls grep
---

# Terraform Module Generator

## Overview

You write complete, production-ready Terraform files for AWS modules. Your
service-specific skill (e.g., `{service}-module-generator/SKILL.md`) is loaded
automatically and contains the exact resources, variables, and outputs to
generate. This skill provides your general workflow and mandatory conventions.

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

Your service-specific skill is NOT loaded into your context automatically.
You MUST use `ls /skills/` to locate it, and `read_file` to load it. 

It contains:
- **SKILL.md**: Which files to create, in what order, with which resources
- **references/**: Detailed HCL patterns, variable schemas, output schemas

Follow the workflow in your service-specific SKILL.md exactly. 
If no service-specific skill exists (e.g. `ls /skills/` is empty or only contains base skills), fall back to the conventions in
[references/module-conventions.md](references/module-conventions.md).

### 2. Read Reference Files

Load **only** what you need for the current file you are generating:

- When writing `main.tf`: read `references/resource-patterns.md`
- When writing `variables.tf`: read `references/variables-schema.md`
- When writing `outputs.tf`: read `references/outputs-schema.md`
- When security resources are involved: read `references/security-rules.md`
- For full execution context: read `references/execution-blueprint.md`

> **Context budget**: Do NOT read all references upfront. Read each one only
> when you are about to write the corresponding .tf file.

### 3. Generate All .tf Files

Write EVERY file to `./workspace/terraform_modules/{service}/{filename}`.

Always generate these files (minimum set):
1. `versions.tf` — Terraform and provider version constraints
2. `locals.tf` — Feature toggles and computed values
3. `main.tf` — Primary resource declarations
4. `variables.tf` — Input variable definitions
5. `outputs.tf` — Output value definitions

Generate additional files as declared by the service-specific skill (e.g.,
`iam.tf`, `security_groups.tf`, `data.tf`, `policies.tf`).

### 4. Generate README.md

ALWAYS generate a `README.md` at `./workspace/terraform_modules/{service}/README.md`
with:
- Module description and purpose
- Usage example with `module` block
- Inputs table (from variables.tf)
- Outputs table (from outputs.tf)
- Requirements (Terraform version, provider version)

### 5. Configure State Backend (if applicable)

If your service-specific SKILL.md dictates a state backend configuration (e.g., in `references/state-management.md`), you MUST create a `backend.tf` file.
Implement the configuration exactly as specified in the reference file, particularly the `terraform { backend "s3" {} }` block. Do NOT skip this if instructed.

### 6. Apply Mandatory Patterns

These patterns are non-negotiable. See
[references/module-conventions.md](references/module-conventions.md) for details.

**Conditional creation**: Use `count` for singleton (0-or-1) resources, `for_each`
for multiple keyed instances. Choose based on the resource's cardinality — see
[references/module-conventions.md](references/module-conventions.md) for the full
decision table.
```hcl
# count — single toggle
resource "aws_{resource}" "this" {
  count = local.create_{resource} ? 1 : 0
}

# for_each — multiple keyed instances
resource "aws_{resource}" "this" {
  for_each = local.{resource}_map
}
```

**Tagging**: Always use the merge pattern.
```hcl
tags = merge(
  { "Name" = var.name },
  var.tags,
  var.{resource}_tags,
)
```

**Safe outputs**: Use `try()` for conditional resources.
```hcl
output "{service}_id" {
  value = try(aws_{resource}.this[0].id, null)
}
```

**No hardcoded values**: Region, account ID, credentials, and CIDR ranges must
be variables. Never hardcode these.

**Provider locking**: Always lock providers in `versions.tf` with `>=` constraint.

### 7. Return Summary

Return exactly:
```
Generated {N} files: [list]. Key design decisions: [brief summary].
```

## Gotchas

- The `execute` tool uses REAL paths (no leading `/`), but `read_file` and
  `write_file` use VIRTUAL paths (with leading `/`). Never mix them.
- Variable `description` fields must be non-empty strings — Terraform warns on
  empty descriptions.
- Every resource MUST have a `Name` tag. AWS resources without Name tags are
  invisible in the console.
- When using `count`, ALWAYS access attributes with `[0]` index:
  `aws_{resource}.this[0].id`, not `aws_{resource}.this.id`.
- Use `try()` in outputs to safely handle `count = 0` cases.
- Never use `for_each` with a computed value that depends on resource
  attributes — it causes the "value depends on resource attributes that cannot
  be determined until apply" error.
