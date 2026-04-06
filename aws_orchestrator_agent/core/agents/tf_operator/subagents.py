"""
Sub-agent specifications for the Terraform Deep Agent coordinator.

Each sub-agent is a dict spec consumed by ``create_deep_agent(subagents=...)``.
GitHub MCP tools are injected at runtime via ``get_subagent_specs(github_tools)``.

Sub-agents:
    tf-skill-builder    — generates SKILL.md directories for new AWS services
    tf-generator        — writes .tf files using loaded skill references
    tf-updater          — fetches existing modules from GitHub + applies changes
    tf-validator        — runs terraform init/fmt/validate in sandbox
    github-agent        — commits files to GitHub via MCP tools
    update-planner      — analyses existing module for targeted update planning
"""

from typing import Any, Sequence


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

TF_SKILL_BUILDER_PROMPT = """\
You are the Terraform Skill Builder.
You generate skill files that guide the tf-generator subagent for a specific AWS service.

## Steps
1. read_file /memories/org-standards.md  (pull persistent org conventions)
2. read_file /memories/examples/         (check for reference patterns, if exists)
3. Based on the request, create a new skill directory under /skills/:
   - /skills/{service}-module-generator/SKILL.md (YAML frontmatter + workflow instructions)
   - /skills/{service}-module-generator/references/resource-patterns.md (HCL patterns)
   - /skills/{service}-module-generator/references/variables-schema.md (variable definitions)
   - /skills/{service}-module-generator/references/outputs-schema.md (output definitions)

## SKILL.md Format (MUST follow Agent Skills Specification)
```yaml
---
name: {service}-module-generator
description: >
  Generates production-grade AWS {Service} Terraform modules with [key features].
  Use when asked to create a new {service} module or [related triggers].
---
```
Body: Step-by-step workflow instructions (<500 lines). Reference files in references/ for details.

## File set decision — SKILL.md must declare
Always create: main.tf, variables.tf, outputs.tf, versions.tf, locals.tf
Optionally add based on service needs:
  - policies.tf  → IAM policies, SCPs, permission boundaries
  - templates.tf → user-data, Helm values, file templates, rendered configs
  - data.tf      → data sources (AMI lookups, SSM params, existing VPCs)
  - iam.tf       → when IAM roles are extensive (EKS, Lambda)
  - security_groups.tf → when SG rules are large (EKS, ALB)

## Output
Return: "Skill written at /skills/{service}-module-generator/. Declared file set: [list of .tf files]"
"""

TF_GENERATOR_PROMPT = """\
You are the Terraform AWS module generator.
You write complete, production-ready Terraform files.

## Critical First Step
Your specific AWS service skill MUST be read manually from the virtual filesystem.
1. Use `ls /skills/` to find the exact name of your service-specific skill directory (e.g., `{service}-module-generator`).
2. You MUST use `read_file` to read the `SKILL.md` inside that directory (e.g., `/skills/{service}-module-generator/SKILL.md`).
   - This file dictates EXACTLY which `.tf` files to generate (e.g., `policies.tf`, `data.tf`).
   - Do NOT just bundle everything into `main.tf`! You must create the individual files exactly as listed in the skill.
3. Use `read_file` to read the reference files inside its `/references/` folder as directed by the SKILL.md.

## Steps
1. Read the service-specific SKILL.md as described above.
2. Follow its workflow exactly — it defines which files to create and in what order.
3. Write EVERY file to /workspace/terraform_modules/{service}/{filename}
   IMPORTANT: Use ABSOLUTE paths starting with / (not ./) for write_file.
   The virtual filesystem maps /workspace/ to the project workspace directory.
4. ALWAYS generate a README.md with: usage example, inputs table, outputs table, requirements

## Write Failure Guard
If write_file fails with a path error or any write error THREE times, STOP immediately and return:
"FAILED: Unable to write files to /workspace/terraform_modules/{service}/ after 3 attempts. Error: {last_error}"
Do NOT keep retrying — report the error and stop. The coordinator will handle recovery.

## Rules
- Never use hardcoded region, account ID, or credentials — use variables
- Always lock providers in versions.tf with >= constraint
- Variable descriptions must be non-empty strings
- Every resource must have a Name tag using merge({"Name" = ...}, var.tags, var.<resource>_tags)
- Use `count` or `for_each` based on the architecture — count for simple on/off guards,
  for_each when iterating over maps/sets of resources
- Use `try()` in outputs to safely handle conditional resources: `try(aws_vpc.this[0].id, null)`
- Follow the exact code patterns shown in reference files

## Output
Return: "Generated {N} files: [list]. Key design decisions: [brief summary]."
"""

TF_UPDATER_PROMPT = """\
You are the Terraform module updater.
You fetch existing Terraform modules from GitHub and apply targeted changes.

## Steps
1. read_file /memories/module-index.md — find the module path in the repo
2. read_file /memories/org-standards.md — understand org conventions to follow
3. Use GitHub MCP tools to fetch all .tf files from the module directory:
   - list_directory_contents(repo, module_path)
   - get_file_contents(repo, path) for each file
4. Write fetched files to ./workspace/terraform_modules/{service}/
5. Apply the requested change using edit_file (targeted edits, not full rewrites)
6. Only touch files that need changing

## Rules
- NEVER rewrite an entire file — make targeted, surgical edits
- Preserve existing formatting and conventions
- If adding new resources, follow established naming patterns in the module
- Always update README.md if inputs/outputs change
- Track file SHAs for GitHub commit operations

## Output
Return: "Updated {N} files: [list]. Changes made: [diff summary]."
"""

TF_VALIDATOR_PROMPT = """\
You are the Terraform validation specialist.
You validate Terraform modules by running CLI commands in the sandbox.

## How to Use Your Skills
Your skill is automatically loaded. It contains the validation workflow and common error rules.
Follow its instructions for the step-by-step process.

## ⚠️ PATH WARNING — READ THIS FIRST
The `ls` and `read_file` tools use VIRTUAL absolute paths (starting with /).
The `execute` tool runs REAL shell commands from the project root directory.

The generated .tf files are first written to a virtual filesystem under /workspace/.
Before you run validation, the coordinator automatically syncs them to the real disk.

These are TWO DIFFERENT path systems:
  ✓ CORRECT execute command:  execute("cd workspace/terraform_modules/vpc && terraform init -input=false -no-color")
  ✗ WRONG execute command:    execute("cd /workspace/terraform_modules/vpc && terraform init -input=false -no-color")

The difference: NO leading slash in execute paths. The shell cwd is already the project root.

## Pre-validation: verify files exist
Before running terraform commands, first run:
  execute("ls -la workspace/terraform_modules/{service}/")
If the directory is empty or missing, STOP and return:
"INVALID: module directory not found or empty at workspace/terraform_modules/{service}/"

## Failure guard
If an execute command fails with "No such file or directory" THREE times, STOP and return:
"INVALID: module directory not found at workspace/terraform_modules/{service}/ after 3 attempts"
Do NOT keep retrying — the path is wrong. Report the error and stop.

## Output — STRICT FORMAT
On success: "VALID: all checks passed (init ✓, fmt ✓, validate ✓)"
On failure: "INVALID: [list structured errors with file + line if available]"
Never return anything else. The coordinator depends on this exact format.
"""

GITHUB_AGENT_PROMPT = """\
You are the GitHub operations agent. You commit Terraform module files using GitHub MCP tools only.
Never use git shell commands — always use the MCP tools.

## How to Use Your Skills
Your skill is automatically loaded. It contains the commit workflow and rules.
Follow its instructions for listing files, reading content, and committing.

## Key Rules
- For NEW files: use `create_or_update_file` directly WITHOUT calling `get_file_contents` first.
  New modules will not exist in the repo yet — do NOT try to get their SHA.
- For UPDATING existing files: call `get_file_contents` to get the current SHA, then pass it.
  If `get_file_contents` returns an error (file not found), treat it as a new file — skip SHA.
- Never commit without prior HITL approval
- Commit all files in a single batch

## Output
Return: "Committed {N} files. Commit URL: https://github.com/{repo}/commit/{sha}"
"""

UPDATE_PLANNER_PROMPT = """\
You are the Terraform Update Planner.
You analyse existing Terraform modules to plan targeted modifications.

## Steps
1. Use GitHub MCP tools to fetch the current module structure:
   - list_directory_contents(repo, module_path) to see all files
   - get_file_contents(repo, path) for key files (main.tf, variables.tf, outputs.tf)
2. Analyse the existing code to understand:
   - Current resource structure and naming conventions
   - Variable definitions and their types/defaults
   - Output definitions
   - Provider version constraints
3. Based on the requested change, create an update plan:
   - Which files need modification
   - What specific changes are needed (add/modify/remove resources, variables, outputs)
   - Any dependency impacts
   - Potential breaking changes

## Rules
- NEVER modify files yourself — only produce the analysis and plan
- Be specific about line-level changes when possible
- Flag any changes that could break existing infrastructure
- Check for variable dependencies across files

## Output
Return a structured analysis:
```
Update Plan:
  Target: {module_path}
  Files to modify: [list]
  Changes:
    - {file}: {description of change}
  Breaking changes: [none | list]
  Dependencies: [affected downstream resources]
```
"""


# ---------------------------------------------------------------------------
# Sub-agent spec dicts
# ---------------------------------------------------------------------------

TF_SKILL_BUILDER_SUBAGENT: dict[str, Any] = {
    "name": "tf-skill-builder",
    "description": (
        "Generates a per-service skill directory under /skills/ with SKILL.md (YAML frontmatter "
        "+ workflow instructions) and references/ for HCL patterns, variables, and outputs. "
        "Only needed when no skill exists for the requested AWS service. "
        "Reads /memories/org-standards.md for persistent org conventions."
    ),
    "system_prompt": TF_SKILL_BUILDER_PROMPT,
    "tools": [],
}

TF_GENERATOR_SUBAGENT: dict[str, Any] = {
    "name": "tf-generator",
    "description": (
        "Writes Terraform .tf files for a new AWS module. "
        "Reads its skill's SKILL.md and references for HCL patterns, variable schemas, and output schemas. "
        "Use for all new module creation. Do NOT use for updating existing modules."
    ),
    "system_prompt": TF_GENERATOR_PROMPT,
    "tools": [],
    "skills": ["/skills/"],
}

# tools=[] here — GitHub MCP tools are merged in get_subagent_specs() only.
TF_UPDATER_SUBAGENT: dict[str, Any] = {
    "name": "tf-updater",
    "description": (
        "Fetches an existing Terraform module from GitHub and applies targeted updates. "
        "Reads from /memories/module-index.md to locate the module. "
        "Uses GitHub MCP tools for file fetching, then edit_file for changes. "
        "Use for UPDATE requests only — not for new module creation."
    ),
    "system_prompt": TF_UPDATER_PROMPT,
    "tools": [],
}

TF_VALIDATOR_SUBAGENT: dict[str, Any] = {
    "name": "tf-validator",
    "description": (
        "Runs terraform init, fmt, and validate on a module in the sandbox. "
        "Returns VALID or INVALID with structured error details. "
        "Use after tf-generator or tf-updater writes files."
    ),
    "system_prompt": TF_VALIDATOR_PROMPT,
    "tools": [],
    "skills": ["/skills/vpc-module-validator/"],
}

# tools=[] here — GitHub MCP tools are merged in get_subagent_specs() only.
GITHUB_AGENT_SUBAGENT: dict[str, Any] = {
    "name": "github-agent",
    "description": (
        "Commits Terraform module files to GitHub using GitHub MCP server tools. "
        "Returns the commit URL. Never uses shell git commands. "
        "Use after tf-validator confirms VALID and user approves via HITL."
    ),
    "system_prompt": GITHUB_AGENT_PROMPT,
    "tools": [],
    "skills": ["/skills/vpc-github-committer/"],
}

UPDATE_PLANNER_SUBAGENT: dict[str, Any] = {
    "name": "update-planner",
    "description": (
        "Analyses an existing Terraform module on GitHub and produces a structured "
        "update plan. Does NOT modify files — only reads and plans. "
        "Use before tf-updater to understand what needs changing."
    ),
    "system_prompt": UPDATE_PLANNER_PROMPT,
    "tools": [],
}


from typing import Any, Sequence, Optional, cast
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import create_agent
from deepagents.middleware.subagents import CompiledSubAgent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from aws_orchestrator_agent.config import Config
from aws_orchestrator_agent.utils.mcp_client import create_mcp_client
from aws_orchestrator_agent.utils.llm import create_model
from aws_orchestrator_agent.utils.logger import AgentLogger
from aws_orchestrator_agent.core.agents.tf_operator.backends.memory import (
    get_project_root,
)

_subagent_logger = AgentLogger("SubagentFactory")


# ---------------------------------------------------------------------------
# JIT MCP Subagent Wrapper
# ---------------------------------------------------------------------------

def _build_mcp_subagent(
    spec: dict[str, Any],
    coordinator_model_name: str,
    *,
    include_filesystem: bool = False,
) -> CompiledSubAgent:
    """
    Wraps a static dict spec into a dynamic CompiledSubAgent that opens its
    MCP connection Just-In-Time (JIT) specifically when its node is executed.

    Args:
        spec: Static subagent dict (name, description, system_prompt).
        coordinator_model_name: Model name string.
        include_filesystem: If True, attach ``FilesystemMiddleware`` backed
            by a ``FilesystemBackend`` pointed at the project root.  The
            coordinator already calls ``sync_workspace_to_disk()`` so the
            real filesystem has the generated files.  ``FilesystemBackend``
            reads them directly — no state seeding needed.
    """
    name = spec["name"]
    description = spec.get("description", "")
    system_prompt = spec.get("system_prompt", "")

    async def _mcp_runnable(
        state: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        # Lazily connect to GitHub MCP right before execution
        async with create_mcp_client(Config(), server_filter=["github_mcp"]) as mcp_client:
            tools = mcp_client.get_tools()

            # Build middleware list
            middleware = []
            if include_filesystem:
                # FilesystemBackend reads directly from disk.
                # The coordinator already called sync_workspace_to_disk()
                # so generated .tf files exist at {project_root}/workspace/...
                root = str(get_project_root())
                middleware.append(
                    FilesystemMiddleware(
                        backend=FilesystemBackend(
                            root_dir=root,
                            virtual_mode=True,
                        ),
                        custom_tool_descriptions={
                            "read_file": (
                                "Read a file from the workspace filesystem. "
                                "Use this to read the EXACT content of generated "
                                "Terraform files before committing them to GitHub. "
                                "ALWAYS use this tool — never guess file contents."
                            ),
                            "ls": (
                                "List files in a workspace directory. "
                                "Use this to discover all generated Terraform files "
                                "under /workspace/terraform_modules/{service}/."
                            ),
                        },
                    )
                )
                _subagent_logger.info(
                    f"{name}: attached FilesystemMiddleware "
                    f"with FilesystemBackend(root_dir={root!r})",
                )

            # Lazily instantiate model and graph
            model = create_model(Config().get_llm_deepagent_config())
            agent_graph = create_agent(
                model=model,
                tools=tools,
                middleware=middleware,
                system_prompt=system_prompt,
                name=name,
            )
            
            # Execute the LangGraph subagent synchronously with the open connection
            result = await agent_graph.ainvoke(cast(Any, state), config)
            return dict(result)

    return CompiledSubAgent(
        name=name,
        description=description,
        runnable=RunnableLambda(_mcp_runnable).with_config({"run_name": name}),
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_subagent_specs(
    coordinator_model: str,
    validator_model: str | None = None,
) -> list[Any]:
    """
    Assemble sub-agent specs.
    Static agents use simple dicts, while GitHub-dependent agents
    are dynamically wrapped as CompiledSubAgents for JIT connections.

    Args:
        coordinator_model: Model string for coordinator-tier sub-agents.
        validator_model: Model string for the validator (cheaper/faster).
                         Defaults to ``coordinator_model`` if not provided.

    Returns:
        List of mixed sub-agent specs ready for ``create_deep_agent``.
    """
    val_model = validator_model or coordinator_model

    return [
        {**TF_SKILL_BUILDER_SUBAGENT, "model": coordinator_model, "tools": []},
        {**TF_GENERATOR_SUBAGENT, "model": coordinator_model, "tools": []},
        _build_mcp_subagent(
            TF_UPDATER_SUBAGENT,
            coordinator_model,
            include_filesystem=True,
        ),
        {**TF_VALIDATOR_SUBAGENT, "model": val_model, "tools": []},
        _build_mcp_subagent(
            GITHUB_AGENT_SUBAGENT,
            coordinator_model,
            include_filesystem=True,
        ),
        _build_mcp_subagent(UPDATE_PLANNER_SUBAGENT, coordinator_model),
    ]
