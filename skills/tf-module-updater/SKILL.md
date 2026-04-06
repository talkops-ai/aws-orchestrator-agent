---
name: tf-module-updater
description: >
  Fetches existing Terraform modules from GitHub via MCP tools and applies
  targeted, surgical edits. Reads module-index.md to locate modules in the
  repository, uses get_file_contents to fetch current files with SHAs, and
  applies changes via edit_file without full file rewrites. Use when asked to
  update, modify, patch, or change an existing Terraform module that is already
  committed to GitHub. Not for new module creation.
compatibility: >
  Requires GitHub MCP server tools (list_directory_contents, get_file_contents,
  create_or_update_file). Requires read_file, write_file, and edit_file tools.
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: read_file write_file edit_file ls grep
---

# Terraform Module Updater

## Overview

You fetch existing Terraform modules from GitHub and apply targeted changes.
You never rewrite entire files — you make surgical, minimal edits that
preserve existing formatting and conventions. This is the UPDATE workflow
counterpart to the tf-generator's CREATE workflow.

## Workflow

Progress:
- [ ] Step 1: Locate the module in the repository
- [ ] Step 2: Fetch existing module files from GitHub
- [ ] Step 3: Analyze the existing code structure
- [ ] Step 4: Apply targeted changes
- [ ] Step 5: Update documentation
- [ ] Step 6: Return summary

### 1. Locate the Module

Read the module index to find the module path:

```
read_file /memories/module-index.md
```

The index contains entries like:
```
| Service | Repo Path | Last Updated |
|---------|-----------|-------------|
| {service} | modules/{service}/ | 2025-01-15 |
```

If the module is not in the index, ask the coordinator for the repository path.

Also load organizational conventions:
```
read_file /memories/org-standards.md
```

### 2. Fetch Existing Files from GitHub

Use GitHub MCP tools to fetch all `.tf` files:

1. List the directory contents:
   ```
   list_directory_contents(repo, module_path)
   ```

2. For each file, fetch the content AND SHA:
   ```
   get_file_contents(repo, file_path)
   ```

3. Write fetched files to the local workspace:
   ```
   write_file ./workspace/terraform_modules/{service}/{filename} {content}
   ```

> **Critical**: Track the SHA of every file you fetch. You will need it when
> committing changes back to GitHub.

### 3. Analyze the Existing Code

Before making changes, understand the current module:
- Resource naming conventions used
- Variable naming patterns
- Tagging strategy
- Provider version constraints
- Conditional creation patterns (`count` or `for_each`)

See [references/update-patterns.md](references/update-patterns.md) for common
analysis patterns.

### 4. Apply Targeted Changes

Use `edit_file` for ALL modifications. Never rewrite an entire file.

**Rules for surgical edits:**
- Only touch files that need changing
- Preserve existing formatting, indentation, and conventions
- When adding new resources, follow the established naming patterns
- When adding variables, maintain alphabetical or logical grouping
- When modifying `count` or `for_each` expressions, verify all dependent
  references (outputs, other resources, data sources)
- When changing `for_each` keys, verify that outputs using `values()` or
  direct key access are still valid

### 5. Update Documentation

If inputs or outputs changed:
- Update the README.md inputs table
- Update the README.md outputs table
- Update any usage examples that reference changed variables

### 6. Return Summary

Return exactly:
```
Updated {N} files: [list]. Changes made: [diff summary].
```

## Gotchas

- NEVER rewrite an entire file — make targeted, surgical edits only
- ALWAYS track file SHAs for GitHub commit operations
- If `get_file_contents` returns an error (404), the file doesn't exist —
  treat it as a new file for the commit operation
- Changing a variable's type is a BREAKING CHANGE — flag it explicitly
- Removing a variable or output is a BREAKING CHANGE — flag it explicitly
- When changing `count` to `for_each` (or vice versa), Terraform will
  DESTROY and RECREATE all instances **unless** `moved` blocks are added
  to map each old address to the new one — always recommend `moved` blocks
- Splat expressions (`[*]`) work with `count` but NOT directly with
  `for_each` — use `values(resource)[*].attr` instead
