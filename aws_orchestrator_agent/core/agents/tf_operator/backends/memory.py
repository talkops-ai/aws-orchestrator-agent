"""
TF Operator — Backend factory and file seeding for Terraform deep agent.

Ports the POC ``backends/memory.py`` pattern into the production codebase.

Routing:
    /memories/ → StoreBackend  (cross-thread persistence via LangGraph Store)
    /skills/   → StoreBackend  (cross-thread persistence via LangGraph Store)
    /workspace/ → StateBackend  (ephemeral virtual FS — generated .tf files)
    default    → LocalShellBackend (real filesystem + shell for terraform CLI)

The ``/workspace/`` route is intentionally virtual (StateBackend) so that
``write_file`` calls from tf-generator always succeed without needing real
directory creation.  Before tf-validator runs ``execute("terraform validate")``,
the coordinator must call ``sync_workspace_to_disk()`` to materialise the
virtual files onto the real filesystem.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from deepagents.backends import (
    CompositeBackend,
    LocalShellBackend,
    StateBackend,
    StoreBackend,
)

from aws_orchestrator_agent.utils import AgentLogger

logger = AgentLogger("TFBackend")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Filesystem root for virtual paths (``/workspace/...``, etc.)."""
    raw = os.getenv("AGENT_PROJECT_ROOT", "").strip()
    return Path(raw).resolve() if raw else Path.cwd().resolve()


def _terraform_base_dir() -> Path:
    """Physical directory for generated modules (virtual ``/workspace/terraform_modules/...``)."""
    root = get_project_root()
    rel = os.getenv("TERRAFORM_WORKSPACE", "workspace/terraform_modules")
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


# ---------------------------------------------------------------------------
# Workspace sync: virtual → real filesystem
# ---------------------------------------------------------------------------

def sync_workspace_to_disk(
    files: Dict[str, Any],
    *,
    prefix: str = "/workspace/",
    project_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """Materialise virtual ``/workspace/`` files from state onto real disk.

    Scans the ``files`` dict (deep agent state) for keys starting with
    ``prefix`` and writes their content to the corresponding path under
    ``project_root``.

    Args:
        files: The deep agent's ``state["files"]`` dictionary. Each value
               is expected to have a ``"content"`` key (or be directly a str).
        prefix: Virtual path prefix to match (default ``"/workspace/"``).
        project_root: Physical directory to resolve paths against.
                      Defaults to ``get_project_root()``.

    Returns:
        Mapping of virtual path → real ``Path`` for every file written.

    Example::

        synced = sync_workspace_to_disk(state["files"])
        # synced == {
        #     "/workspace/terraform_modules/vpc/main.tf":
        #         Path("/abs/path/workspace/terraform_modules/vpc/main.tf"),
        # }
    """
    root = project_root or get_project_root()
    written: Dict[str, Path] = {}

    for vpath, file_data in files.items():
        if not vpath.startswith(prefix):
            continue

        # Extract content from file_data (could be dict or str)
        if isinstance(file_data, dict):
            content = file_data.get("content", "")
        elif isinstance(file_data, str):
            content = file_data
        else:
            # Attempt common attribute access
            content = getattr(file_data, "content", str(file_data))

        # Map virtual path → real path
        # /workspace/terraform_modules/vpc/main.tf → {root}/workspace/terraform_modules/vpc/main.tf
        rel = vpath.lstrip("/")
        real_path = root / rel

        # Create parent directories
        real_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        real_path.write_text(content, encoding="utf-8")
        written[vpath] = real_path

    if written:
        logger.info(
            "sync_workspace_to_disk: materialised virtual files",
            extra={
                "file_count": len(written),
                "paths": list(written.keys())[:10],  # cap log size
            },
        )
    else:
        logger.warning(
            "sync_workspace_to_disk: no /workspace/ files found in state",
            extra={"total_files_in_state": len(files)},
        )

    return written


# ---------------------------------------------------------------------------
# Backend factory mixin
# ---------------------------------------------------------------------------

class TFOperatorBackendMixin:
    """
    Mixin that supplies ``make_backend()`` and ``seed_files()`` for
    Terraform-focused deep agents.

    Routing:
        ``/memories/`` → ``StoreBackend`` (cross-thread persistence)
        ``/skills/``   → ``StoreBackend`` (cross-thread persistence)
        ``/workspace/`` → ``StateBackend`` (ephemeral — generated .tf files)
        default        → ``LocalShellBackend`` (virtual FS under project root
                         + ``execute`` for terraform CLI commands)
    """

    # ── Backend factory ───────────────────────────────────────────────────

    @staticmethod
    def make_backend(runtime: Any) -> CompositeBackend:
        """
        Build a ``CompositeBackend`` with Terraform-specific shell env.

        The ``LocalShellBackend`` sets ``TF_INPUT=false`` and ``-no-color``
        flags so terraform commands run non-interactively.

        The ``/workspace/`` route uses ``StateBackend`` (virtual) so that
        ``write_file`` calls always succeed.  Before ``tf-validator`` runs
        ``execute("terraform validate")``, the coordinator calls
        ``sync_workspace_to_disk()`` to write files to the real filesystem.
        """
        root = get_project_root()
        tf_base = _terraform_base_dir()
        tf_base.mkdir(parents=True, exist_ok=True)

        shell_env = {
            "TF_INPUT": "false",
            "TF_CLI_ARGS_init": "-no-color",
            "TF_CLI_ARGS_validate": "-no-color",
        }

        default = LocalShellBackend(
            root_dir=str(root),
            virtual_mode=True,
            env=shell_env,
            inherit_env=True,
        )

        return CompositeBackend(
            default=default,
            routes={
                "/memories/": StoreBackend(
                    runtime,
                    namespace=lambda ctx: (
                        ctx.runtime.context.get("org_name", "default_org")
                        if isinstance(ctx.runtime.context, dict)
                        else getattr(ctx.runtime.context, "org_name", "default_org"),
                    ),
                ),
                "/skills/": StateBackend(runtime),
            },
        )

    # ── File seeding ──────────────────────────────────────────────────────

    @staticmethod
    def seed_files(
        skills_dir: Optional[Path] = None,
        memory_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Load skill files and initial memory into the virtual FS dict for
        ``agent.invoke(files=...)``.

        Walks ``skills_dir`` and ``memory_dir`` on disk, mapping each file
        to a virtual path (``/skills/...`` and ``/memories/...``).
        """
        from deepagents.backends.utils import create_file_data

        skills_dir = skills_dir or Path("./skills")
        memory_dir = memory_dir or Path("./memory")
        files: Dict[str, Any] = {}

        if skills_dir.exists():
            for path in skills_dir.rglob("*"):
                if path.is_file():
                    vpath = f"/skills/{path.relative_to(skills_dir).as_posix()}"
                    files[vpath] = create_file_data(
                        path.read_text(encoding="utf-8")
                    )

        if memory_dir.exists():
            for path in memory_dir.rglob("*"):
                if path.is_file() and path.name != ".gitkeep":
                    vpath = f"/memories/{path.relative_to(memory_dir).as_posix()}"
                    files[vpath] = create_file_data(
                        path.read_text(encoding="utf-8")
                    )

        return files
