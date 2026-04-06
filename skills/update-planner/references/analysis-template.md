# Update Analysis Template

## Structured Output Format

Use this template when producing update plans. The coordinator and downstream
agents parse this structure.

```
Update Plan:
  Target: {owner}/{repo} → {module_path}
  Branch: {branch_name}
  Requested Change: {description of what the user wants}

  Files to modify:
    - {filename}: {what changes and why}
    - {filename}: {what changes and why}

  New files to create:
    - {filename}: {purpose}

  Files to delete:
    - {filename}: {reason}

  Detailed Changes:
    ## {filename}
    - Line {N}: {description of change}
    - Add block at line {N}: {resource/variable/output description}
    - Remove lines {N}-{M}: {what is being removed and why}

  Breaking Changes:
    - {description} — Impact: {who/what is affected}
    OR
    - None

  Dependency Analysis:
    - {resource_a} → {resource_b}: {nature of dependency}
    - {variable_name}: used in {file1}, {file2}

  State Impact:
    - {resource}: will be {created|modified|destroyed|recreated}
    - Risk: {description of data loss or downtime risk}

  Risk Assessment: {low|medium|high}
    Justification: {reason for risk level}

  Recommendations:
    1. {actionable recommendation}
    2. {actionable recommendation}
```

## Risk Level Guidelines

### Low Risk
- Adding new resources with no dependencies on existing ones
- Adding new variables with defaults
- Adding new outputs
- Formatting changes
- Documentation updates

### Medium Risk
- Modifying existing resource configurations
- Adding validation rules to existing variables
- Changing default values
- Adding `depends_on` relationships
- Provider version bumps (minor version)

### High Risk
- Removing or renaming resources (state impact)
- Removing variables without defaults (breaks callers)
- Removing outputs (breaks consumers)
- Changing variable types (breaks callers)
- Switching `count` ↔ `for_each` (destroys/recreates all instances)
- Provider version bumps (major version)
- Changes affecting encryption, IAM, or network security

## Dependency Check Patterns

When analyzing dependencies, check these cross-file references:

| In this file... | Check references to... |
|-----------------|----------------------|
| `main.tf` | Variables from `variables.tf`, locals from `locals.tf`, data from `data.tf` |
| `outputs.tf` | Resources from `main.tf`, `iam.tf`, `security_groups.tf` |
| `locals.tf` | Variables from `variables.tf`, data from `data.tf` |
| `iam.tf` | Resources from `main.tf` (ARNs, IDs) |
| `security_groups.tf` | Resources from `main.tf` (resource IDs, subnet IDs) |

## Migration Guidance

When a change requires state manipulation, include `moved` block guidance:

```hcl
# If renaming a resource from "old_name" to "new_name":
moved {
  from = aws_{resource}.old_name
  to   = aws_{resource}.new_name
}
```

This prevents Terraform from destroying and recreating the resource.
