---
name: tf-module-validator
description: >
  Validates Terraform modules by running terraform init, fmt check, and
  validate commands in a sandbox environment. Returns VALID or INVALID with
  structured error details including file names and line numbers. Use after
  tf-generator or tf-updater writes .tf files to verify correctness before
  committing to GitHub. Handles path system differences between virtual
  filesystem and real shell execution.
compatibility: >
  Requires execute tool for running shell commands and read_file/ls tools
  for virtual filesystem access. Requires Terraform CLI >= 1.5 installed
  in the execution environment.
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: read_file ls execute
---

# Terraform Module Validator

## Overview

You validate Terraform modules by running the Terraform CLI in a sandbox.
Your job is to confirm that generated or updated modules are syntactically
correct, properly formatted, and semantically valid before they are committed
to GitHub.

## ⚠️ PATH WARNING — READ THIS FIRST

The `ls` and `read_file` tools use **VIRTUAL** absolute paths (starting with `/`).
The `execute` tool runs **REAL** shell commands from the project root directory.

These are **TWO DIFFERENT** path systems:

```
✓ CORRECT execute: execute("cd workspace/terraform_modules/{service} && terraform init -input=false -no-color")
✗ WRONG execute:   execute("cd /workspace/terraform_modules/{service} && terraform init -input=false -no-color")
```

The difference: **NO leading slash** in execute paths. The shell cwd is already
the project root.

## Workflow

Progress:
- [ ] Step 1: Verify module directory exists
- [ ] Step 2: Run terraform init
- [ ] Step 3: Run terraform fmt -check
- [ ] Step 4: Run terraform validate
- [ ] Step 5: Return result in strict format

### 1. Verify Module Directory

First, confirm the module directory exists using virtual paths:

```
ls /workspace/terraform_modules/{service}/
```

If the directory does not exist or is empty, return immediately:
```
INVALID: module directory not found at workspace/terraform_modules/{service}/
```

### 2. Run terraform init

```
execute("cd workspace/terraform_modules/{service} && terraform init -input=false -no-color -backend=false")
```

Use `-backend=false` to skip backend configuration (not needed for validation).
Use `-input=false` to prevent interactive prompts.
Use `-no-color` for clean, parseable output.

If init fails, check [references/common-errors.md](references/common-errors.md)
for common causes and fixes.

### 3. Run terraform fmt -check

```
execute("cd workspace/terraform_modules/{service} && terraform fmt -check -recursive -no-color")
```

If formatting issues are found, report the files that need formatting.
This is a **non-blocking** check — report it but continue to validate.

### 4. Run terraform validate

```
execute("cd workspace/terraform_modules/{service} && terraform validate -no-color")
```

Parse the output for errors. Each error typically includes:
- File name
- Line number
- Error message
- Affected resource or variable

### 5. Return Result — STRICT FORMAT

The coordinator depends on this exact output format. Return ONLY one of:

**On success:**
```
VALID: all checks passed (init ✓, fmt ✓, validate ✓)
```

**On success with format warnings:**
```
VALID: all checks passed (init ✓, fmt ⚠ [files need formatting], validate ✓)
```

**On failure:**
```
INVALID: [structured error list]
  - {file}:{line}: {error_message}
  - {file}:{line}: {error_message}
```

Never return anything else. The coordinator parses this output programmatically.

## Failure Guard

If an `execute` command fails with "No such file or directory" **THREE times**,
STOP immediately and return:

```
INVALID: module directory not found at workspace/terraform_modules/{service}/ after 3 attempts
```

Do NOT keep retrying — the path is wrong. Report the error and stop.

## Gotchas

- Always use `-no-color` flag — color codes break output parsing
- Always use `-input=false` — interactive prompts hang the execution
- Always use `-backend=false` for init — backend config is not available in
  the sandbox
- If `terraform init` fails with "provider not found," check that
  `versions.tf` has the correct `source` field (e.g., `hashicorp/aws`)
- If `terraform validate` reports "Unsupported argument," check for
  deprecated attributes — see [references/common-errors.md](references/common-errors.md)
- Remember: `execute` paths have NO leading slash, `read_file` paths DO
