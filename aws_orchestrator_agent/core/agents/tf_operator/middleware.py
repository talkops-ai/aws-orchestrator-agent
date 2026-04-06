"""
TF Operator — Modular middleware factory for deep agent safety nets.

Provides a configurable factory that assembles middleware stacks for the
TFCoordinator deep agent.  Each middleware is independently toggleable via
``Config`` or environment variables, keeping the coordinator clean.

Middleware catalogue:
    ToolCallLimitMiddleware   — caps runaway tool loops (per-tool or global)
    ModelCallLimitMiddleware  — caps excessive LLM calls
    ToolRetryMiddleware       — auto-retries transient tool failures
    SummarizationMiddleware   — compresses context when window fills

Usage::

    from aws_orchestrator_agent.core.agents.tf_operator.middleware import (
        build_deep_agent_middleware,
    )

    middleware = build_deep_agent_middleware(config)
    agent = create_deep_agent(
        ...,
        middleware=middleware,
    )
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from aws_orchestrator_agent.utils import AgentLogger

if TYPE_CHECKING:
    from aws_orchestrator_agent.config import Config

logger = AgentLogger("TFMiddleware")


# ---------------------------------------------------------------------------
# Default limits (overridable via env vars or Config)
# ---------------------------------------------------------------------------

# Per-tool write_file limit — prevents infinite write-retry loops
_WRITE_FILE_RUN_LIMIT = int(os.getenv("TF_WRITE_FILE_RUN_LIMIT", "20"))

# Global tool call limit per single invocation
_GLOBAL_TOOL_RUN_LIMIT = int(os.getenv("TF_GLOBAL_TOOL_RUN_LIMIT", "60"))

# Model call limit per invocation
_MODEL_CALL_RUN_LIMIT = int(os.getenv("TF_MODEL_CALL_RUN_LIMIT", "40"))

# Whether to enable tool retry middleware (disabled by default as it swallows GraphInterrupt)
_ENABLE_TOOL_RETRY = os.getenv("TF_ENABLE_TOOL_RETRY", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_deep_agent_middleware(
    config: Optional["Config"] = None,
    *,
    write_file_limit: Optional[int] = None,
    global_tool_limit: Optional[int] = None,
    model_call_limit: Optional[int] = None,
    enable_tool_retry: Optional[bool] = None,
    extra_middleware: Optional[List[Any]] = None,
) -> List[Any]:
    """Assemble the middleware stack for the TFCoordinator deep agent.

    Each guard is independently configurable.  Pass ``None`` to use the
    default (env var → compiled default).

    Args:
        config: Application config (currently unused, reserved for future
                per-agent overrides from ``Config``).
        write_file_limit: Max ``write_file`` calls per run.
        global_tool_limit: Max total tool calls per run (all tools).
        model_call_limit: Max LLM calls per run.
        enable_tool_retry: Whether to auto-retry transient tool failures.
        extra_middleware: Additional middleware instances to append.

    Returns:
        A list of middleware instances suitable for
        ``create_deep_agent(middleware=...)``.
    """
    from langchain.agents.middleware import (
        ToolCallLimitMiddleware,
        ModelCallLimitMiddleware,
        ToolRetryMiddleware,
    )

    middleware: List[Any] = []

    # ── 1. Per-tool write_file guard ──────────────────────────────────────
    wf_limit = write_file_limit or _WRITE_FILE_RUN_LIMIT
    middleware.append(
        ToolCallLimitMiddleware(
            tool_name="write_file",
            run_limit=wf_limit,
            exit_behavior="continue",  # Let LLM handle the error message
        )
    )
    logger.info(
        "Middleware: write_file limit",
        extra={"run_limit": wf_limit},
    )

    # ── 2. Global tool call guard ─────────────────────────────────────────
    gt_limit = global_tool_limit or _GLOBAL_TOOL_RUN_LIMIT
    middleware.append(
        ToolCallLimitMiddleware(
            run_limit=gt_limit,
            exit_behavior="continue",
        )
    )
    logger.info(
        "Middleware: global tool limit",
        extra={"run_limit": gt_limit},
    )

    # ── 3. Model call guard ───────────────────────────────────────────────
    mc_limit = model_call_limit or _MODEL_CALL_RUN_LIMIT
    middleware.append(
        ModelCallLimitMiddleware(
            run_limit=mc_limit,
            exit_behavior="end",  # Graceful stop instead of exception
        )
    )
    logger.info(
        "Middleware: model call limit",
        extra={"run_limit": mc_limit},
    )

    # ── 4. Tool retry (transient failures) ────────────────────────────────
    should_retry = enable_tool_retry if enable_tool_retry is not None else _ENABLE_TOOL_RETRY
    if should_retry:
        middleware.append(
            ToolRetryMiddleware(
                max_retries=2,
                backoff_factor=1.5,
                initial_delay=0.5,
                max_delay=10.0,
                on_failure="continue",  # Let LLM see the error
            )
        )
        logger.info("Middleware: tool retry enabled")

    # ── 5. Extra (caller-provided) ────────────────────────────────────────
    if extra_middleware:
        middleware.extend(extra_middleware)
        logger.info(
            "Middleware: extra appended",
            extra={"count": len(extra_middleware)},
        )

    logger.info(
        "Middleware stack assembled",
        extra={
            "total_middleware": len(middleware),
            "types": [type(m).__name__ for m in middleware],
        },
    )

    return middleware
