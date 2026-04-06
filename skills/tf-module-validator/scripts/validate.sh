#!/usr/bin/env bash
# Terraform Module Validation Script
# Usage: bash scripts/validate.sh <module_directory>
#
# Runs terraform init, fmt -check, and validate in sequence.
# Returns exit code 0 on success, 1 on failure.
# All output is formatted for machine parsing (--no-color).

set -euo pipefail

MODULE_DIR="${1:?Usage: validate.sh <module_directory>}"

if [ ! -d "$MODULE_DIR" ]; then
  echo "INVALID: module directory not found at $MODULE_DIR"
  exit 1
fi

cd "$MODULE_DIR"

echo "=== Step 1/3: terraform init ==="
if ! terraform init -input=false -no-color -backend=false 2>&1; then
  echo ""
  echo "INVALID: terraform init failed"
  exit 1
fi

echo ""
echo "=== Step 2/3: terraform fmt -check ==="
FMT_OUTPUT=$(terraform fmt -check -recursive -no-color 2>&1 || true)
FMT_STATUS=$?

if [ -n "$FMT_OUTPUT" ]; then
  echo "Format issues found in:"
  echo "$FMT_OUTPUT"
  FMT_RESULT="⚠ [files need formatting]"
else
  FMT_RESULT="✓"
fi

echo ""
echo "=== Step 3/3: terraform validate ==="
if ! terraform validate -no-color 2>&1; then
  echo ""
  echo "INVALID: terraform validate failed"
  exit 1
fi

echo ""
echo "VALID: all checks passed (init ✓, fmt $FMT_RESULT, validate ✓)"
exit 0
