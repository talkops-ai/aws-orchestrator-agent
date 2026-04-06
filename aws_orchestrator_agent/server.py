"""
AWS Orchestrator Agent — A2A Server.

Initializes the A2A application with the deep-agent ``SupervisorAgent``,
wires the ``A2AExecutor``, and starts the Starlette server.

Usage::

    python -m aws_orchestrator_agent.server
    python -m aws_orchestrator_agent.server --host 0.0.0.0 --port 10103
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCard
from starlette.middleware.cors import CORSMiddleware

from aws_orchestrator_agent.config import Config
from aws_orchestrator_agent.core.a2a_executor import A2AExecutor
from aws_orchestrator_agent.core.agents import (
    SupervisorAgent,
    create_supervisor_agent,
    create_tf_coordinator,
)
from aws_orchestrator_agent.utils.logger import AgentLogger

logger = AgentLogger("OrchestratorServer")


# ---------------------------------------------------------------------------
# Agent card loader
# ---------------------------------------------------------------------------


def load_agent_card(agent_card_path: str, host: str, port: int) -> AgentCard:
    """Load the A2A agent card from disk, injecting the runtime URL.

    Args:
        agent_card_path: Path to the agent card JSON file.
        host: Server host to embed in the card URL.
        port: Server port to embed in the card URL.

    Returns:
        Populated ``AgentCard`` instance.
    """
    path = Path(agent_card_path)
    with path.open() as fh:
        data = json.load(fh)

    if host and port:
        data["url"] = f"http://{host}:{port}"

    return AgentCard(**data)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


@click.command()
@click.option("--host", default=None, help="Server host (default: from config)")
@click.option("--port", type=int, default=None, help="Server port (default: from config)")
@click.option(
    "--agent-card",
    default=None,
    help="Path to agent card JSON (default: from config)",
)
def main(
    host: Optional[str],
    port: Optional[int],
    agent_card: Optional[str],
) -> None:
    """Start the AWS Orchestrator A2A server."""
    try:
        # ── Configuration ─────────────────────────────────────────────
        config = Config()
        resolved_host: str = host or str(config.get("A2A_HOST", "localhost"))
        resolved_port: int = port or int(config.get("A2A_PORT", 10103))
        agent_card_path: str = agent_card or str(
            config.get(
                "A2A_AGENT_CARD",
                "aws_orchestrator_agent/card/aws_orchestrator_agent.json",
            )
        )

        logger.info(
            "Initializing AWS Orchestrator server",
            extra={"host": resolved_host, "port": resolved_port, "agent_card": agent_card_path},
        )

        # ── Agent card ────────────────────────────────────────────────
        agent_card_obj = load_agent_card(agent_card_path, resolved_host, resolved_port)
        logger.info("Agent card loaded", extra={"path": agent_card_path})

        # ── TF Coordinator Deep Agent ─────────────────────────────────
        # Instantiate the TF Coordinator sub-agent.
        # (Initialization and MCP connection happen lazily in SupervisorAgent.initialize)
        tf_coordinator_agent = create_tf_coordinator(config=config)

        # ── Supervisor agent ──────────────────────────────────────────
        supervisor = create_supervisor_agent(
            agents=[tf_coordinator_agent],
            config=config,
            name="aws_orchestrator"
        )

        logger.info(
            "SupervisorAgent created",
            extra={"name": supervisor.name, "agents_count": 1},
        )

        # ── A2A executor ──────────────────────────────────────────────
        executor = A2AExecutor(agent=supervisor)

        # ── A2A request handler ───────────────────────────────────────
        client: httpx.AsyncClient = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=client,
            config_store=push_config_store,
        )

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )

        # ── Starlette app ─────────────────────────────────────────────
        server: A2AStarletteApplication = A2AStarletteApplication(
            agent_card=agent_card_obj,
            http_handler=request_handler,
        )

        app = server.build()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        logger.info(
            f"Starting AWS Orchestrator Server on {resolved_host}:{resolved_port}",
            extra={"host": resolved_host, "port": resolved_port},
        )

        uvicorn.run(
            app,
            host=resolved_host,
            port=resolved_port,
            log_level=config.get("LOG_LEVEL", "INFO").lower(),
        )

    except FileNotFoundError as exc:
        logger.error(f"File not found: {exc}", extra={"error": str(exc)})
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON: {exc}", extra={"error": str(exc)})
        sys.exit(1)
    except Exception as exc:
        logger.error(
            f"Server startup failed: {exc}",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
