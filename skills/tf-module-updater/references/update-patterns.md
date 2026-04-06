# Terraform Module Update Patterns

## Common Update Scenarios

### Adding a New Resource

When adding a resource to an existing module, follow the existing module's
iteration pattern (`count` or `for_each`). Do NOT mix iteration strategies
within the same module unless there is a strong architectural reason.

#### Using `count` (single or numeric instances)

1. **Add the resource block** to `main.tf`:
   ```hcl
   resource "aws_{child_resource}" "{child_name}" {
     count             = local.create_{child_name} ? var.{child_name}_count : 0
     {parent_resource}_id = aws_{parent_resource}.this[0].id
     availability_zone = data.aws_availability_zones.available.names[count.index]

     tags = merge(
       { "Name" = "${var.name}-{child_name}-${count.index}" },
       var.tags,
       var.{child_name}_tags,
     )
   }
   ```

2. **Add the feature toggle** to `locals.tf`:
   ```hcl
   create_{child_name} = local.create_{parent_resource} && var.create_{child_name}
   ```

3. **Add outputs** using splat:
   ```hcl
   output "{child_name}_ids" {
     description = "List of {child_name} IDs"
     value       = aws_{child_resource}.{child_name}[*].id
   }
   ```

> **Note**: Splat `[*]` works with `count` because the result is a list.

#### Using `for_each` (keyed instances)

1. **Add the resource block** to `main.tf`:
   ```hcl
   resource "aws_{child_resource}" "{child_name}" {
     for_each = local.create_{parent_resource} ? var.{child_name}_config : {}

     name = each.key
     # ... each.value.attribute

     tags = merge(
       { "Name" = "${var.name}-${each.key}" },
       var.tags,
       var.{child_name}_tags,
     )
   }
   ```

2. **Add outputs** — `for_each` produces a MAP, not a list:
   ```hcl
   # Option A: Return the full map of IDs
   output "{child_name}_ids" {
     description = "Map of {child_name} IDs by key"
     value       = { for k, v in aws_{child_resource}.{child_name} : k => v.id }
   }

   # Option B: Return as a list (loses key association)
   output "{child_name}_id_list" {
     description = "List of {child_name} IDs"
     value       = values(aws_{child_resource}.{child_name})[*].id
   }
   ```

> **Important**: Splat `[*]` does NOT work directly on `for_each` resources.
> You must use `values()` first to convert the map to a list, or use a
> `for` expression.

### Adding Variables

Always add variables in the same file section as related existing variables.
Every variable MUST have:

```hcl
variable "create_{child_name}" {
  description = "Whether to create {child_name} resources"
  type        = bool
  default     = true
}

variable "{child_name}_config" {
  description = "Configuration map for {child_name} instances"
  type        = map(object({
    attribute = string
  }))
  default     = {}
}

variable "{child_name}_tags" {
  description = "Additional tags for {child_name} resources"
  type        = map(string)
  default     = {}
}
```

### Modifying an Existing Variable

When changing a variable's default, type, or validation:

```hcl
# Before
variable "instance_type" {
  description = "Compute instance type"
  type        = string
  default     = "t3.micro"
}

# After — added validation
variable "instance_type" {
  description = "Compute instance type"
  type        = string
  default     = "t3.micro"

  validation {
    condition     = can(regex("^[a-z][a-z0-9]*\\.[a-z0-9]+$", var.instance_type))
    error_message = "Must be a valid instance type (e.g., t3.micro, m5.large)."
  }
}
```

> **Warning**: Changing a variable's `type` is a breaking change.

### Provider Version Bump

When upgrading the AWS provider version:

1. Update `versions.tf`:
   ```hcl
   required_providers {
     aws = {
       source  = "hashicorp/aws"
       version = ">= 5.60.0"  # was >= 5.40.0
     }
   }
   ```

2. Check for deprecated attributes in the changelog
3. Replace any deprecated attributes with their replacements
4. Run `terraform validate` to catch issues

> **Best practice**: Upgrade one major version at a time. If moving from
> v4 → v6, do v4 → v5 first, resolve all deprecations, then v5 → v6.

### Adding an Output

For `count`-based resources:
```hcl
output "new_output_name" {
  description = "Description of the output"
  value       = try(aws_{resource}.this[0].attribute, null)
}
```

For `for_each`-based resources:
```hcl
output "new_output_map" {
  description = "Map of resource attributes by key"
  value       = { for k, v in aws_{resource}.this : k => v.attribute }
}
```

Use `try()` only for conditional resources (where `count` may be 0). Do not
use `try()` to hide legitimate configuration errors.

## Breaking Change Detection

A change is considered BREAKING if it:

| Change | Impact | Mitigation |
|--------|--------|------------|
| Remove a variable | Consumers get "unsupported argument" error | Deprecate first, remove in next major version |
| Remove an output | Consumers get "unsupported attribute" error | Mark as deprecated first |
| Change variable type | Consumers get type mismatch error | Add a migration note |
| Rename a resource | Terraform destroys + recreates (data loss risk) | Use `moved` blocks |
| Switch count ↔ for_each | Terraform destroys + recreates all instances | Use `moved` blocks (per-instance mapping) |
| Change for_each keys | Terraform destroys old + creates new keyed instances | Use `moved` blocks to map old key → new key |

### Migrating count → for_each safely

This is the most common breaking refactor. Use `moved` blocks to prevent
destruction:

```hcl
# Map each count index to the new for_each key
moved {
  from = aws_{resource}.this[0]
  to   = aws_{resource}.this["primary"]
}

moved {
  from = aws_{resource}.this[1]
  to   = aws_{resource}.this["secondary"]
}
```

After adding `moved` blocks, run `terraform plan` to verify. The plan should
show `0 to add, 0 to change, 0 to destroy` for the affected resources.

> **Important**: Retain `moved` blocks in the configuration permanently.
> Removing them risks that other consumers on older branches lose the mapping.

## Diff Tracking

When reporting changes, use this format:

```
Files modified:
  - variables.tf: Added variable "{child_name}_config" (lines 45-52)
  - main.tf: Added resource "aws_{child_resource}.{child_name}" (lines 78-95)
  - outputs.tf: Added output "{child_name}_ids" (line 32)
  - README.md: Updated inputs table, outputs table

Breaking changes: None
Dependencies affected: None
Iteration pattern used: for_each (consistent with existing module)
```
