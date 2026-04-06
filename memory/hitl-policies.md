# Human-in-the-Loop Policies

These policies define WHEN the coordinator MUST call `request_user_input`
to pause and ask the human for input. The coordinator reads this file at
session start and follows these policies strictly.

## Mandatory HITL Gates

### 1. Post-Validation Commit Gate
- **When:** After tf-validator confirms VALID
- **Purpose:** Ask whether to push to GitHub or keep local
- **Default:** Keep local (never assume GitHub push)
- **Collect:** repository (owner/repo), branch name
- **Skip if:** User explicitly said "don't push" earlier in the conversation

### 2. Task Completion / Next Steps Gate
- **When:** After the current task is fully complete (commit done OR kept local)
- **Purpose:** Ask if user wants to continue with another action
- **Options:** Generate another module, update existing module, done
- **Collect:** Details for the next action (service name, requirements)
- **Skip if:** User explicitly said "just this one module" earlier

### 3. Destructive Operations
- **When:** Before deleting modules, overwriting existing files, or force-pushing
- **Purpose:** Explicit human approval for irreversible actions
- **Never skip:** Always require approval for destructive operations

## Optional HITL Gates (Agent Discretion)

### 4. Ambiguous Requirements
- **When:** User request is vague or has multiple valid interpretations
- **Purpose:** Clarify before spending compute on the wrong path
- **Examples:**
  - "Create a VPC module" — which region? How many AZs? Public/private?
  - "Update the networking" — which module? What specifically?

### 5. Cost-Sensitive Architecture Decisions
- **When:** Architecture choice significantly affects cloud costs
- **Purpose:** Human sign-off before generating expensive infrastructure
- **Examples:**
  - NAT Gateway per AZ vs shared
  - Dedicated vs shared tenancy
  - Multi-region replication

## Policy Updates
The coordinator SHOULD update this file when:
- A new mandatory gate is identified from user feedback
- An optional gate is promoted to mandatory based on repeated usage
- A gate is found to be unnecessary and should be removed
