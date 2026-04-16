---
name: tf-module-updater
description: "Fetches existing Terraform modules from GitHub via MCP tools and applies targeted, surgical edits. Reads module-index.md to locate modules in the repository, uses get_file_contents to fetch current files with SHAs, and applies changes via edit_file without full file rewrites. Use when asked to update, modify, patch, or change an existing Terraform module that is already committed to GitHub. Not for new module creation."
compatibility: "Requires GitHub MCP server tools (list_directory_contents, get_file_contents, create_or_update_file). Requires read_file, write_file, and edit_file tools."
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: read_file write_file edit_file ls grep
---

# Terraform Module Updater

Fetches existing Terraform modules from GitHub and applies surgical edits. Counterpart to the tf-generator's CREATE workflow — this is the UPDATE workflow.

## Workflow

Progress:
- [ ] Step 1: Locate the module in the repository
- [ ] Step 2: Fetch existing module files from GitHub
- [ ] Step 3: Analyze the existing code structure
- [ ] Step 4: Apply targeted changes
- [ ] Step 5: Validate changes
- [ ] Step 6: Update documentation
- [ ] Step 7: Return summary

### 1. Locate the Module

Read the module index and org conventions:

```
read_file /memories/module-index.md
read_file /memories/org-standards.md
```

The index maps services to repo paths. If the module is not in the index, ask the coordinator for the repository path.

### 2. Fetch Existing Files from GitHub

Use GitHub MCP tools to fetch all `.tf` files with their SHAs:

```
list_directory_contents(repo="{owner}/{repo}", path="{module_path}")
```

For each file, fetch content and SHA (needed for commit operations):

```
get_file_contents(repo="{owner}/{repo}", path="{module_path}/main.tf")
# Response includes: content, sha, encoding
```

Write fetched files to workspace:

```
write_file ./workspace/terraform_modules/{service}/{filename} {content}
```

Track every SHA — you need them when committing back via `create_or_update_file`.

### 3. Analyze the Existing Code

Identify the module's conventions before editing. See [references/update-patterns.md](references/update-patterns.md) for detailed patterns. Key signals:
- Iteration pattern (`count` or `for_each`) — follow whichever the module uses
- Naming conventions for resources, variables, and toggles
- Tagging strategy (merge pattern, prefix patterns)

### 4. Apply Targeted Changes

Use `edit_file` for ALL modifications. Never rewrite an entire file.

**Example — adding a variable to an existing `variables.tf`:**

```
edit_file(
  path="./workspace/terraform_modules/{service}/variables.tf",
  edits=[{
    old_text='variable "tags" {',
    new_text='variable "enable_logging" {\n  description = "Whether to enable access logging"\n  type        = bool\n  default     = true\n}\n\nvariable "tags" {'
  }]
)
```

**Rules for surgical edits:**
- Only touch files that need changing
- Preserve existing formatting, indentation, and conventions
- Follow the module's established iteration pattern — do not mix `count` and `for_each`
- When modifying `count` or `for_each`, verify all dependent references (outputs, other resources, data sources)
- When changing `for_each` keys, verify outputs using `values()` or direct key access

### 5. Validate Changes

After applying edits, run validation to catch errors before committing:

```
execute("cd workspace/terraform_modules/{service} && terraform init -input=false -no-color -backend=false && terraform validate -no-color")
```

If validation fails, review the error, fix the edit, and re-validate. Do not proceed to documentation or commit with failing validation.

### 6. Update Documentation

If inputs or outputs changed, update README.md: inputs table, outputs table, and any usage examples referencing changed variables.

### 7. Return Summary

Return exactly:
```
Updated {N} files: [list]. Changes made: [diff summary]. Validation: PASSED.
```

## Gotchas

- NEVER rewrite an entire file — surgical edits only
- ALWAYS track file SHAs for GitHub commit operations
- If `get_file_contents` returns 404, treat as a new file (no SHA needed)
- Changing a variable's type or removing a variable/output is a BREAKING CHANGE — flag explicitly
- When switching `count` ↔ `for_each`, add `moved` blocks to prevent destroy/recreate
- Splat `[*]` works with `count` but not `for_each` — use `values(resource)[*].attr` instead
