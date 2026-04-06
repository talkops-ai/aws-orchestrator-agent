"""
TF Coordinator Context Schema.

This is the **runtime context** for the Terraform Deep Agent coordinator —
values injected via ``config["context"]`` at invocation time.

In the Deep Agents framework, ``context_schema`` serves a specific purpose:
carrying per-session, static configuration that:
  1. Tools can read via ``config.get("context", {}).get("field_name")``.
  2. Avoids bloating the conversation (messages) with boilerplate metadata.
  3. Is propagated automatically to sub-agent contexts.

What does NOT belong here:
  - Workflow phase tracking (use state reducers in TFPlannerState).
  - File contents / artifacts (use the virtual FS — ``files`` state).
  - Long-term memory (use ``/memories/*.md`` via StoreBackend).
  - Conversation history (managed by add_messages reducer).

Reference: https://docs.langchain.com/oss/python/deepagents/context-engineering#runtime-context

Usage::

    agent = await coord.build_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Create a VPC module"}],
         "files": coord.seed_files()},
        config={
            "configurable": {"thread_id": "t-abc123"},
            "context": {
                # Required for GitHub operations
                "github_repo": "acme/infra-terraform",
                "github_branch": "main",

                # Required for workspace routing
                "workspace_path": "./workspace/terraform_modules",
                "service": "vpc",
                "workflow_mode": "new_module",

                # AWS target
                "aws_account_id": "123456789012",
                "aws_region": "us-east-1",

                # Organization conventions
                "org_name": "acme",
                "module_prefix": "acme-aws",
                "environment": "production",

                # Optional
                "dry_run": False,
                "session_id": "sess-xyz",
            },
        },
    )
"""

from typing import Optional
from typing_extensions import NotRequired, TypedDict


# ---------------------------------------------------------------------------
# TFCoordinatorContext
# ---------------------------------------------------------------------------

class TFCoordinatorContext(TypedDict, total=False):
    """
    Runtime context for the Terraform Deep Agent coordinator.

    Passed as ``config["context"]`` on every invocation.  All fields are
    ``NotRequired`` — only supply what is relevant for the current session.
    The deep agent framework propagates this context to every sub-agent call.

    Fields are grouped into four logical sections:
        1. GitHub — repository and commit identity
        2. Workspace — local paths and session intent
        3. AWS — target cloud account and region
        4. Terraform — version and backend configuration
        5. Organization — naming conventions and approval policy
        6. Session — identifiers and feature flags
    """

    # ── 1. GitHub ──────────────────────────────────────────────────────────

    github_repo: NotRequired[str]
    """
    GitHub repository in ``owner/repo`` format.
    Used by: github-agent, tf-updater, update-planner (GitHub MCP tools).
    Example: ``"acme-corp/infra-terraform"``
    """

    github_branch: NotRequired[str]
    """
    Target branch for commits.
    Used by: github-agent (``create_or_update_file``).
    Example: ``"main"``, ``"feature/add-vpc-module"``
    """

    github_commit_author: NotRequired[str]
    """
    Git commit author name embedded in commit messages.
    Used by: github-agent.
    Example: ``"TalkOps Bot"``
    """

    # ── 2. Workspace ───────────────────────────────────────────────────────

    workspace_path: NotRequired[str]
    """
    Root path for generated Terraform modules in the virtual FS.
    Used by: tf-generator, tf-updater, tf-validator.
    Example: ``"./workspace/terraform_modules"``
    Convention: tf files land at ``{workspace_path}/{service}/``
    """

    service: NotRequired[str]
    """
    Primary AWS service being generated or updated in this session.
    Used by coordinator prompt routing (skill check, subagent task strings).
    Example: ``"vpc"``, ``"eks"``, ``"rds-postgres"``
    """

    workflow_mode: NotRequired[str]
    """
    Session intent — controls which workflow branch the coordinator follows.
    Values: ``"new_module"`` | ``"update_module"``
    Default assumed by coordinator: ``"new_module"`` if not set.
    """

    # ── 3. AWS ─────────────────────────────────────────────────────────────

    aws_account_id: NotRequired[str]
    """
    Target AWS account ID (12-digit string, no hyphens).
    Used by: tf-generator (provider block, IAM ARN construction).
    Example: ``"123456789012"``
    """

    aws_region: NotRequired[str]
    """
    Primary AWS region for the module.
    Used by: tf-generator (provider ``region`` variable defaults).
    Example: ``"us-east-1"``, ``"eu-west-1"``
    """

    aws_profile: NotRequired[str]
    """
    Named AWS CLI profile for local sandbox execution.
    Used by: tf-validator (``terraform init`` credential resolution).
    Example: ``"infra-prod"``.  Omit when using IAM role / CI credentials.
    """

    # ── 4. Terraform ───────────────────────────────────────────────────────

    tf_version_constraint: NotRequired[str]
    """
    Terraform CLI version constraint written into ``versions.tf``.
    Used by: tf-generator (``required_version`` in ``terraform`` block).
    Example: ``">= 1.9.0, < 2.0.0"``
    """

    aws_provider_version: NotRequired[str]
    """
    AWS provider version constraint written into ``versions.tf``.
    Used by: tf-generator (``required_providers`` → ``aws.version``).
    Example: ``">= 5.40.0"``
    """

    tf_backend_type: NotRequired[str]
    """
    Remote state backend type embedded in the generated backend stub.
    Values: ``"s3"`` | ``"gcs"`` | ``"azurerm"`` | ``"local"``
    Used by: tf-generator (optional backend block in ``versions.tf``).
    """

    tf_state_bucket: NotRequired[str]
    """
    S3 bucket (or equivalent) for the Terraform remote state backend.
    Used by: tf-generator (backend config stub).
    Example: ``"acme-corp-tf-state-us-east-1"``
    """

    # ── 5. Organization ────────────────────────────────────────────────────

    org_name: NotRequired[str]
    """
    Organization/team name used in resource naming and skill discovery.
    Used by: tf-skill-builder (SKILL.md frontmatter), tf-generator (name tags).
    Example: ``"acme-corp"``
    """

    module_prefix: NotRequired[str]
    """
    Naming prefix applied to every generated Terraform module name.
    Used by: tf-generator (``locals.tf`` name construction).
    Example: ``"acme-aws"`` → module named ``"acme-aws-vpc"``
    """

    environment: NotRequired[str]
    """
    Deployment environment — controls defaults and guard-rails.
    Values: ``"development"`` | ``"staging"`` | ``"production"``
    Used by: coordinator (HITL gate strictness), tf-generator (env variable).
    """

    require_approval: NotRequired[bool]
    """
    Whether HITL approval is mandatory before GitHub commits.
    Default: ``True`` (always enforce the approval gate).
    Set ``False`` only in automated CI/CD pipelines with trusted input.
    """

    mandatory_tags: NotRequired[dict]
    """
    Key-value pairs that MUST appear on every tagged resource.
    Used by: tf-generator (merged into every ``tags`` argument).
    Example: ``{"Owner": "platform-team", "CostCenter": "12345"}``
    """

    # ── 6. Session ─────────────────────────────────────────────────────────

    session_id: NotRequired[str]
    """
    Opaque session identifier for correlation with external systems.
    Used by: coordinator logging, LangSmith trace metadata.
    Example: ``"sess-2026-03-31-abc123"``
    """

    task_id: NotRequired[str]
    """
    Parent A2A task ID.  Threads the deep-agent session back to the
    originating ``TaskStatusUpdateEvent`` for the upstream supervisor.
    Example: ``"task-uuid-1234"``
    """

    dry_run: NotRequired[bool]
    """
    When ``True``, the coordinator skips the github-agent commit step
    and reports what *would* have been committed.
    Useful for preview / audit flows without touching the repo.
    Default: ``False``
    """
