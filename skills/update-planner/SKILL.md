---
name: update-planner
description: >
  Analyses existing Terraform modules on GitHub to plan targeted modifications.
  Fetches current module structure, understands resource naming and variable
  conventions, identifies dependencies, and produces a structured update plan
  with impact assessment and breaking change detection. Use when asked to
  analyse, plan, or prepare changes to an existing Terraform module before
  applying them. Does NOT modify files — only reads and produces a plan for
  the tf-updater to execute.
compatibility: >
  Requires GitHub MCP server tools (list_directory_contents, get_file_contents)
  for fetching module files from the repository. Read-only operation.
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: read_file ls
---

# Terraform Update Planner

## Overview

You analyse existing Terraform modules to plan targeted modifications. You
fetch the current module from GitHub, understand its structure and conventions,
and produce a structured update plan that the tf-updater will execute. You
NEVER modify files yourself — you only read and plan.

## Workflow

Progress:
- [ ] Step 1: Fetch the current module from GitHub
- [ ] Step 2: Analyse the existing code structure
- [ ] Step 3: Map the requested change to specific file modifications
- [ ] Step 4: Assess impact and detect breaking changes
- [ ] Step 5: Produce the structured update plan

### 1. Fetch the Current Module

Use GitHub MCP tools to fetch the module structure:

1. List all files in the module directory:
   ```
   list_directory_contents(repo, module_path)
   ```

2. Fetch key files for analysis:
   ```
   get_file_contents(repo, "{module_path}/main.tf")
   get_file_contents(repo, "{module_path}/variables.tf")
   get_file_contents(repo, "{module_path}/outputs.tf")
   get_file_contents(repo, "{module_path}/versions.tf")
   ```

3. Optionally fetch additional files if the change may affect them:
   - `locals.tf` — if feature toggles are involved
   - `iam.tf` — if IAM changes are requested
   - `security_groups.tf` — if network changes are requested

### 2. Analyse the Existing Code

Understand the current module by examining:

**Resource structure:**
- What resources are defined and how they're named
- Which resources use `count` vs. `for_each`
- Dependency chains (`depends_on`, implicit references)

**Variable patterns:**
- Naming convention (snake_case, prefix patterns)
- Type constraints used
- Default values and validation rules
- Boolean toggles (`create_*`, `enable_*`)

**Output patterns:**
- Which attributes are exposed
- Whether `try()` is used for conditional resources
- Sensitive outputs

**Provider constraints:**
- Required Terraform version
- Required provider version
- Backend configuration (if any)

### 3. Map Changes to Files

For each requested change, determine:

1. **Which files need modification** — list specific files
2. **What specific changes are needed** in each file:
   - Add resource: main.tf (resource block), variables.tf (new vars), outputs.tf (new outputs), locals.tf (toggle)
   - Modify resource: only the affected file(s)
   - Remove resource: main.tf, outputs.tf, possibly variables.tf
3. **Dependency impacts** — what other resources reference the changed ones

### 4. Assess Impact

Evaluate each change for:

- **Breaking changes**: See [references/analysis-template.md](references/analysis-template.md)
- **State impact**: Will Terraform need to destroy/recreate resources?
- **Security impact**: Does the change affect IAM, encryption, or network security?
- **Cost impact**: Does the change add or remove billable resources?

### 5. Produce the Update Plan

Return a structured plan using this exact format:

```
Update Plan:
  Target: {module_path}
  Files to modify: [list]
  Changes:
    - {file}: {description of change}
    - {file}: {description of change}
  Breaking changes: [none | list with descriptions]
  Dependencies: [affected downstream resources]
  Risk assessment: [low | medium | high] — {reason}
```

## Gotchas

- NEVER modify files yourself — only produce the analysis and plan
- Be specific about line-level changes when possible
- Always flag changes that could break existing infrastructure
- Check for variable dependencies ACROSS files — a variable used in main.tf
  may also be referenced in locals.tf and outputs.tf
- When a resource is renamed, check ALL outputs and cross-references
- When removing a variable, check if it has a default — if not, ALL callers
  will break
