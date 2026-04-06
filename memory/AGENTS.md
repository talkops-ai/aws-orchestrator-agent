# Agent Memory Index

This directory holds seed content for `/memories/` in the Deep Agent virtual filesystem.
The coordinator reads this file at session start to know what memory files exist.

## Memory Files

| File | Purpose | Update Frequency |
|------|---------|-----------------|
| **hitl-policies.md** | HITL gate policies — when to pause and ask the human | Updated by agent when new policies are learned |
| **org-standards.md** | Organization-wide Terraform conventions (tags, naming, providers) | Rarely — admin-managed |
| **module-index.md** | Where modules live in the GitHub repo (for update flows) | After every successful commit |
| **user-preferences.md** | Per-user or per-team preferences | As discovered from conversations |
| **failure-log.md** | Validation or deployment failures to avoid repeating | After failures |
| **learned-patterns.md** | Patterns the agent should reuse across sessions | As discovered |

## Reading Rules
- **Always read:** hitl-policies.md (operational policies for every session)
- **Read at start:** org-standards.md (conventions apply to all tasks)
- **Read for updates:** module-index.md (only when updating existing modules)
- **Read on demand:** Others as needed

## Writing Rules
- **hitl-policies.md:** Update when a new mandatory gate is identified from user feedback
- **module-index.md:** Update after every successful GitHub commit with module path + repo
- **org-standards.md:** Update only when user explicitly shares new conventions
- **failure-log.md:** Append after validation or deployment failures
