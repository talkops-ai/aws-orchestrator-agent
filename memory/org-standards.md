# Organization Terraform Standards

## Tagging Conventions
All resources must utilize the standardized global tagging module (`terraform-provider-default-tags`).
Mandatory tags include:
- `CostCenter`
- `Owner`
- `Project`
- `Environment` (must be one of: dev, staging, prod, sandbox)
