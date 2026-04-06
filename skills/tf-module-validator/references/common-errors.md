# Common Terraform Validation Errors

## Init Errors

### Provider not found

```
Error: Failed to install provider

Could not find required providers. hashicorp/aws: no available releases match
the given constraints >= 99.0.0
```

**Cause**: The `version` constraint in `versions.tf` is too restrictive or the
provider source is wrong.

**Fix**: Check `versions.tf` has a valid source and version:
```hcl
required_providers {
  aws = {
    source  = "hashicorp/aws"
    version = ">= 5.40.0"
  }
}
```

### Backend initialization error

```
Error: Backend initialization required, please run "terraform init"
```

**Cause**: Running validate before init, or init was run with different backend
settings.

**Fix**: Run `terraform init -backend=false -reconfigure`.

---

## Format Errors

### Incorrect indentation

```
main.tf
--- old
+++ new
@@ -1,3 +1,3 @@
 resource "aws_vpc" "this" {
-    cidr_block = var.vpc_cidr
+  cidr_block = var.vpc_cidr
 }
```

**Cause**: Using 4 spaces instead of 2, or mixed tabs and spaces.

**Fix**: Run `terraform fmt -recursive` to auto-fix.

---

## Validate Errors

### Missing required argument

```
Error: Missing required argument

  on main.tf line 5, in resource "aws_vpc" "this":
   5: resource "aws_vpc" "this" {

The argument "cidr_block" is required, but no definition was found.
```

**Cause**: A required attribute is missing from the resource block.

**Fix**: Add the missing attribute.

### Unsupported argument

```
Error: Unsupported argument

  on main.tf line 8, in resource "aws_s3_bucket" "this":
   8:   acl = "private"

An argument named "acl" is not expected here.
```

**Cause**: The attribute was deprecated and removed in a newer provider version.
In AWS provider >= 4.x, `acl` was moved to a separate `aws_s3_bucket_acl`
resource.

**Fix**: Remove the deprecated attribute and use the replacement resource.

### Unsupported block type

```
Error: Unsupported block type

  on main.tf line 10, in resource "aws_s3_bucket" "this":
  10:   versioning {

Blocks of type "versioning" are not expected here.
```

**Cause**: The block was moved to a separate resource in a newer provider
version (e.g., `aws_s3_bucket_versioning`).

**Fix**: Use the separate resource instead.

### Reference to undeclared resource

```
Error: Reference to undeclared resource

  on outputs.tf line 3, in output "vpc_id":
   3:   value = aws_vpc.main.id

A managed resource "aws_vpc" "main" has not been declared in the root module.
```

**Cause**: The output or reference uses a resource name that doesn't exist.
Often caused by a typo or rename.

**Fix**: Check the resource name in `main.tf` and update the reference.

### Invalid reference in count

```
Error: Invalid count argument

  on main.tf line 5, in resource "aws_subnet" "private":
   5:   count = aws_vpc.this[0].id != "" ? 1 : 0

The "count" value depends on resource attributes that cannot be determined
until apply.
```

**Cause**: Using a resource attribute in a `count` expression. Terraform needs
to know `count` at plan time, but resource attributes are only known after
apply.

**Fix**: Use a variable or local value instead:
```hcl
count = local.create_private_subnets ? 1 : 0
```

### Duplicate resource name

```
Error: Duplicate resource "aws_security_group" configuration

  on security_groups.tf line 15:
  15: resource "aws_security_group" "this" {

A aws_security_group resource named "this" was already declared at main.tf:42,10-36.
```

**Cause**: Two resources of the same type have the same name.

**Fix**: Give each resource a unique name within its type.

---

## Output-Specific Errors

### Invalid output value

```
Error: Unsupported attribute

  on outputs.tf line 3, in output "vpc_id":
   3:   value = aws_vpc.this.id

This object has no argument, nested block, or exported attribute named "id".
```

**Cause**: When using `count`, you must access the resource with an index:
`aws_vpc.this[0].id`, not `aws_vpc.this.id`.

**Fix**: Use index access and wrap in `try()`:
```hcl
value = try(aws_vpc.this[0].id, null)
```

---

## Variable-Specific Errors

### Empty description

```
Warning: Variable "name" has no description

  on variables.tf line 1, in variable "name":
   1: variable "name" {
```

**Cause**: Variable is missing a `description` field.

**Fix**: Add a meaningful description:
```hcl
variable "name" {
  description = "Name prefix for all resources in this module"
  type        = string
}
```

### Type constraint mismatch

```
Error: Invalid default value for variable

  on variables.tf line 5, in variable "tags":
   5:   default = "none"

This default value is not compatible with the variable's type constraint:
map of string required.
```

**Cause**: The `default` value doesn't match the declared `type`.

**Fix**: Set a default that matches the type:
```hcl
variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
```
