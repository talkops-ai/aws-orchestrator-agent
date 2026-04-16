---
name: tf-module-validator
description: "Validates Terraform modules by running terraform init, fmt check, and validate commands in a sandbox environment. Returns VALID or INVALID with structured error details including file names and line numbers. Use after tf-generator or tf-updater writes .tf files to verify correctness before committing to GitHub. Handles path system differences between virtual filesystem and real shell execution."
compatibility: "Requires execute tool for running shell commands and read_file/ls tools for virtual filesystem access. Requires Terraform CLI >= 1.5 installed in the execution environment."
metadata:
  author: talkops-ai
  version: "1.0"
  generated-by: manual-authoring
allowed-tools: read_file ls execute
---

# Terraform Module Validator

Validates Terraform modules via `terraform init`, `fmt -check`, and `validate` in a sandbox. Returns structured VALID/INVALID results for the coordinator.

## Path Systems

`ls`/`read_file` use **VIRTUAL** paths (leading `/`). `execute` uses **REAL** paths (no leading `/`):

```
✓ CORRECT: execute("cd workspace/terraform_modules/{service} && terraform init -input=false -no-color")
✗ WRONG:   execute("cd /workspace/terraform_modules/{service} && terraform init -input=false -no-color")
```

## Workflow

Progress:
- [ ] Step 1: Verify module directory exists
- [ ] Step 2: Run terraform init
- [ ] Step 3: Run terraform fmt -check
- [ ] Step 4: Run terraform validate
- [ ] Step 5: Return result in strict format

### 1. Verify Module Directory

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

If init fails, check [references/common-errors.md](references/common-errors.md) for common causes.

### 3. Run terraform fmt -check

```
execute("cd workspace/terraform_modules/{service} && terraform fmt -check -recursive -no-color")
```

This is a **non-blocking** check — report files needing formatting but continue to validate.

### 4. Run terraform validate

```
execute("cd workspace/terraform_modules/{service} && terraform validate -no-color")
```

Parse errors for file name, line number, error message, and affected resource/variable.

### 5. Return Result — STRICT FORMAT

The coordinator parses this output programmatically. Return ONLY one of:

**Success:**
```
VALID: all checks passed (init ✓, fmt ✓, validate ✓)
```

**Success with format warnings:**
```
VALID: all checks passed (init ✓, fmt ⚠ [files need formatting], validate ✓)
```

**Failure:**
```
INVALID: [structured error list]
  - {file}:{line}: {error_message}
  - {file}:{line}: {error_message}
```

## Failure Guard

If `execute` fails with "No such file or directory" **three times**, STOP:

```
INVALID: module directory not found at workspace/terraform_modules/{service}/ after 3 attempts
```

## Gotchas

- Always use flags: `-no-color -input=false -backend=false` (color breaks parsing, prompts hang, backend unavailable in sandbox)
- "Provider not found" on init → check `versions.tf` has correct `source` (e.g., `hashicorp/aws`)
- "Unsupported argument" on validate → deprecated attributes. See [references/common-errors.md](references/common-errors.md)
- `execute` paths: NO leading slash. `read_file` paths: leading slash required
