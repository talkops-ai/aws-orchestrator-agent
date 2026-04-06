# Terraform Module Conventions

## Provider Locking

Always lock both the Terraform version and provider version in `versions.tf`:

```hcl
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40.0"
    }
  }
}
```

Use `>=` constraints (not `~>`) so consumers can pin their own upper bound.

## Conditional Creation & Iteration Patterns

Every top-level resource should support conditional creation. The choice
between `count` and `for_each` depends on the module's architecture and
the resource's cardinality.

### When to use `count`

Use `count` for **singleton resources** (0-or-1) controlled by a boolean toggle.
This is the simplest pattern for enabling/disabling an entire resource.

```hcl
# locals.tf
locals {
  create_{resource} = var.create_{resource}
}

# main.tf
resource "aws_{resource}" "this" {
  count = local.create_{resource} ? 1 : 0
  # ...
}
```

Access with index: `aws_{resource}.this[0].id`

### When to use `for_each`

Use `for_each` for resources with **multiple instances** keyed by a stable
identifier (name, AZ, map key). This is preferred when:
- The number of instances is variable (subnets per AZ, rules per CIDR)
- Instances need stable identity (adding/removing one doesn't shift all indexes)
- The resource has a natural key (name, identifier, map key)

```hcl
# locals.tf
locals {
  {resource}_map = var.create_{resource} ? var.{resource}_config : {}
}

# main.tf
resource "aws_{resource}" "this" {
  for_each = local.{resource}_map

  name = each.key
  # ... each.value.attribute
}
```

Access with key: `aws_{resource}.this["key-name"].id`

### Choosing the right pattern

| Scenario | Use | Reason |
|----------|-----|--------|
| Single on/off toggle (one instance) | `count` | Simple boolean guard |
| Multiple instances with known keys | `for_each` with map | Stable identity, no index shift |
| Multiple instances from a list | `for_each` with `toset()` | Stable identity |
| Dynamic count from a number variable | `count` with `var.N` | Numeric cardinality |

> **Important**: Never use `for_each` with a value that depends on resource
> attributes unknown until apply — Terraform cannot determine the set at plan
> time. Use `count` or pre-computed values instead.

### Composing conditions with locals

Locals allow you to compose dependent conditions regardless of whether the
parent uses `count` or `for_each`:

```hcl
locals {
  create_{child_resource} = local.create_{resource} && var.create_{child_resource}
}
```

## Tagging Convention

ALL resources must use the merge tagging pattern:

```hcl
tags = merge(
  { "Name" = "${var.name}-{resource}" },
  var.tags,
  var.{resource}_tags,
)
```

This pattern gives consumers three levels of tag control:
1. **Automatic** — `Name` tag is always set
2. **Global** — `var.tags` applies to all resources in the module
3. **Per-resource** — `var.{resource}_tags` overrides for a specific resource

## Safe Output Pattern

Outputs that reference conditional resources MUST use `try()`:

```hcl
output "{resource}_id" {
  description = "The ID of the {resource}"
  value       = try(aws_{resource}.this[0].id, null)
}
```

Without `try()`, accessing `.this[0]` on a resource with `count = 0` causes a
Terraform error. `try()` gracefully returns `null`.

## Variable Conventions

### Naming
- Use snake_case for all variables
- Prefix boolean toggles with `create_` or `enable_`
- Suffix tag maps with `_tags`

### Required Fields
Every variable MUST have a non-empty `description`:

```hcl
variable "name" {
  description = "Name prefix for all resources in this module"
  type        = string
}
```

### Validation Rules
Add validation blocks for constrained inputs:

```hcl
variable "{resource}_cidr" {
  description = "CIDR block for the {resource}"
  type        = string

  validation {
    condition     = can(cidrhost(var.{resource}_cidr, 0))
    error_message = "Must be a valid CIDR block (e.g., 10.0.0.0/16)."
  }
}
```

## File Organization

| File | Purpose | When to Create |
|------|---------|----------------|
| `versions.tf` | Terraform + provider constraints | Always |
| `locals.tf` | Feature toggles, computed values | Always |
| `main.tf` | Primary resources | Always |
| `variables.tf` | Input definitions | Always |
| `outputs.tf` | Output values | Always |
| `README.md` | Documentation | Always |
| `data.tf` | Data sources (AMI, SSM, existing resource lookups) | When external data is needed |
| `iam.tf` | IAM roles, policies, instance profiles | When IAM is extensive |
| `security_groups.tf` | Security group rules | When SG rules are numerous |
| `policies.tf` | IAM policy documents, SCPs | When policies are complex |
| `templates.tf` | User-data, Helm values, rendered configs | When templates are needed |

## Anti-Patterns

- **Never hardcode** region, account ID, credentials, or CIDR ranges
- **Never use `terraform` blocks in non-root modules** — only `required_version`
  and `required_providers`
- **Never use `for_each` with computed values** that depend on resource
  attributes unknown until apply
- **Never omit the `Name` tag** — AWS resources without it are invisible in
  the console
- **Never use deprecated attributes** even if they still work — they will be
  removed in future provider versions
