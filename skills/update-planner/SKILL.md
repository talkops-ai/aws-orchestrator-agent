---
name: update-planner
description: "Analyses existing Terraform modules on GitHub to plan targeted modifications. Fetches current module structure, understands resource naming and variable conventions, identifies dependencies, and produces a structured update plan with impact assessment and breaking change detection. Use when asked to analyse, plan, or prepare changes to an existing Terraform module before applying them. Does NOT modify files — only reads and produces a plan for the tf-updater to execute."
compatibility: "Requires GitHub MCP server tools (list_directory_contents, get_file_contents) for fetching module files from the repository. Read-only operation."
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: read_file ls
---

# Terraform Update Planner

Analyses existing Terraform modules to plan targeted modifications. Read-only — produces a structured plan for the tf-updater to execute.

## Workflow

Progress:
- [ ] Step 1: Fetch the current module from GitHub
- [ ] Step 2: Analyse the existing code structure
- [ ] Step 3: Map the requested change to specific file modifications
- [ ] Step 4: Assess impact and detect breaking changes
- [ ] Step 5: Validate plan consistency
- [ ] Step 6: Produce the structured update plan

### 1. Fetch the Current Module

Use GitHub MCP tools to fetch the module structure:

```
list_directory_contents(repo, module_path)
```

Fetch the core files — `main.tf`, `variables.tf`, `outputs.tf`, `versions.tf` — plus any files relevant to the requested change (`locals.tf` for feature toggles, `iam.tf` for IAM changes, `security_groups.tf` for network changes):

```
get_file_contents(repo, "{module_path}/main.tf")
get_file_contents(repo, "{module_path}/variables.tf")
get_file_contents(repo, "{module_path}/outputs.tf")
get_file_contents(repo, "{module_path}/versions.tf")
```

If `get_file_contents` returns a 404, note the file as absent — do not fail. If the module path itself does not exist, STOP and report:
```
Cannot plan: module not found at {module_path}. Verify the path with the coordinator.
```

### 2. Analyse the Existing Code

Identify the module's conventions by examining:

- **Iteration pattern**: `count` vs `for_each` — the plan must follow whichever the module uses
- **Naming conventions**: resource names, variable prefixes (`create_*`, `enable_*`)
- **Dependency chains**: `depends_on`, implicit references across files
- **Output safety**: whether `try()` is used for conditional resources

Example analysis output for a VPC module:
```
Pattern: count-based iteration (all resources use count with local.create_*)
Naming: snake_case, resources named "this", toggles use "create_" prefix
Dependencies: subnets → vpc, route_tables → subnets, nat_gateway → eip
Outputs: all use try() with [0] index for count safety
Provider: aws >= 5.40.0, Terraform >= 1.5
```

### 3. Map Changes to Files

For each requested change, determine:

1. **Which files need modification** — list specific files and line ranges
2. **What specific changes are needed** — concrete additions, modifications, or removals:
   - Add resource: `main.tf` + `variables.tf` + `outputs.tf` + `locals.tf` (toggle)
   - Modify resource: only the affected file(s)
   - Remove resource: `main.tf` + `outputs.tf` + possibly `variables.tf`
3. **Cross-file dependencies** — trace every reference to the changed resource/variable across all files

### 4. Assess Impact

Evaluate each change using the risk levels and breaking change criteria in [references/analysis-template.md](references/analysis-template.md):

- **Breaking changes**: variable removal, type changes, resource renames, `count` ↔ `for_each` switches
- **State impact**: will Terraform destroy/recreate resources? Include `moved` block guidance if so
- **Security impact**: changes to IAM, encryption, or network security
- **Cost impact**: added or removed billable resources

### 5. Validate Plan Consistency

Before producing the final plan, verify:
- Every new variable referenced in `main.tf` changes has a corresponding entry in `variables.tf` changes
- Every new resource has corresponding output entries
- Cross-file references are valid (no dangling references to removed resources)
- The iteration pattern is consistent with the existing module

### 6. Produce the Update Plan

Return a structured plan following the format in [references/analysis-template.md](references/analysis-template.md). At minimum:

```
Update Plan:
  Target: {owner}/{repo} → {module_path}
  Files to modify: [list with line ranges]
  Changes:
    - {file}:{line}: {description of change}
  Breaking changes: [none | list with impact descriptions]
  Dependencies: [affected downstream resources]
  State impact: [resources that will be created/modified/destroyed]
  Risk assessment: [low | medium | high] — {justification}
```

## Gotchas

- NEVER modify files — only read and produce the plan
- Check variable dependencies ACROSS files — a variable in `main.tf` may also appear in `locals.tf` and `outputs.tf`
- When a resource is renamed, check ALL outputs and cross-references
- When removing a variable without a default, ALL callers will break — always flag this
