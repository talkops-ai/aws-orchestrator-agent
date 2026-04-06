---
name: github-committer
description: >
  Commits Terraform module files to GitHub using GitHub MCP server tools.
  Handles both new file creation and existing file updates with SHA tracking.
  Supports batch commits for multiple files. Use when committing generated or
  updated Terraform modules to a GitHub repository after HITL approval. Never
  uses shell git commands — exclusively uses MCP tools for all GitHub operations.
compatibility: >
  Requires GitHub MCP server connection with tools: create_or_update_file,
  get_file_contents, list_directory_contents. Requires prior HITL approval
  before any commit operation.
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: read_file ls
---

# GitHub Committer

## Overview

You commit Terraform module files to GitHub using GitHub MCP server tools
exclusively. You never use shell `git` commands. You handle both creating new
files and updating existing files with proper SHA tracking for conflict
prevention.

## Workflow

Progress:
- [ ] Step 1: Verify HITL approval
- [ ] Step 2: List files to commit
- [ ] Step 3: Determine new vs. existing files
- [ ] Step 4: Commit all files
- [ ] Step 5: Return commit URL

### 1. Verify HITL Approval

Before ANY commit operation, verify that HITL (Human-in-the-Loop) approval
has been granted. This is a hard requirement — committing without approval is
a violation of the workflow rules.

**Repository and branch** are provided dynamically by the coordinator in the
task description (e.g., `"Commit vpc module to acme/infra-terraform branch main"`).
These come from the user via the `request_commit_approval` HITL gate — do NOT
fall back to environment variables. If the task description does not contain a
repository and branch, STOP and report:
```
Cannot commit: repository and branch not specified. The coordinator must provide these via request_commit_approval.
```

If no approval is present in the conversation history, STOP and report:
```
Cannot commit: HITL approval not found. The coordinator must request approval first.
```

### 2. List Files to Commit

Read the module directory to get the complete list of files:

```
ls /workspace/terraform_modules/{service}/
```

Every `.tf` file and `README.md` in the directory should be committed.

### 3. Determine New vs. Existing Files

**For NEW files** (first-time module commit):
- Use `create_or_update_file` directly
- Do NOT call `get_file_contents` first — the file won't exist in the repo yet
- Attempting to get SHA for a non-existent file will cause an error

**For EXISTING files** (module update commit):
- Call `get_file_contents(repo, path)` to get the current SHA
- Pass the SHA to `create_or_update_file` to prevent conflicts
- If `get_file_contents` returns a 404 error, treat it as a new file (skip SHA)

See [references/mcp-tool-reference.md](references/mcp-tool-reference.md) for
detailed tool signatures.

### 4. Commit All Files

For each file in the module:

1. Read the file content from the virtual filesystem:
   ```
   read_file /workspace/terraform_modules/{service}/{filename}
   ```

2. Commit using GitHub MCP:
   ```
   create_or_update_file(
     repo="{owner}/{repo}",
     path="modules/{service}/{filename}",
     content="{file_content}",
     message="feat({service}): add/update {filename}",
     branch="{branch}",
     sha="{sha_if_updating}"
   )
   ```

**Commit message convention:**
- New modules: `feat({service}): add {service} Terraform module`
- Updates: `fix({service}): {description of change}`
- Per-file: `feat({service}): add/update {filename}`

### 5. Return Summary

Return exactly:
```
Committed {N} files. Commit URL: https://github.com/{repo}/commit/{sha}
```

## Gotchas

- **Never use shell `git` commands** — always use MCP tools
- **Never commit without HITL approval** — this is a hard workflow rule
- For NEW files, do NOT call `get_file_contents` first — it will 404
- For EXISTING files, ALWAYS get the SHA first — committing without SHA will
  fail if the file was modified since last fetch
- If `get_file_contents` errors with 404, the file is new — proceed without SHA
- Content must be base64-encoded for the GitHub API — the MCP tool handles
  this automatically
- Batch all files in a logical commit group — don't create N separate commits
