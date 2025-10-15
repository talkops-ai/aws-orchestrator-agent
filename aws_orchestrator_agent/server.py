"""
AWS Orchestrator Agent - A2A Server Entry Point

This is the main entry point for the AWS Orchestrator Agent service.
It initializes the A2A server with Custom Supervisor Agent and proper configuration.
"""

import json
import logging
import sys
from pathlib import Path

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCard

from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.core import (
    GenericAgentExecutor,
)
from aws_orchestrator_agent.core.agents.supervisor_agent import create_supervisor_agent
from aws_orchestrator_agent.core.agents.planner import create_planner_sub_supervisor_agent
from aws_orchestrator_agent.core.agents.generator.generator_swarm import create_generator_swarm_agent
from aws_orchestrator_agent.core.agents.writer.writer_react_agent import create_writer_react_agent
from aws_orchestrator_agent.core.task_lifecycle import TaskLifecycleManager
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync

# Create agent logger for server
server_logger = AgentLogger("AWS_ORCHESTRATOR_SERVER")


@click.command()
@click.option('--host', 'host', help='Server host')
@click.option('--port', 'port', type=int, help='Server port')
@click.option('--agent-card', 'agent_card', help='Path to agent card JSON file')
@click.option('--config-file', 'config_file', help='Path to configuration file')
@log_sync
def main(host: str, port: int, agent_card: str, config_file: str) -> None:
    """
    Main entry point for the AWS Orchestrator Agent server.
    """
    try:
        # Load configuration
        config = Config()
        if config_file:
            custom_config = Config.load_config(config_file)
            config = Config(custom_config)

        # Use config values for host and port if not provided
        host = host or config.a2a_server_host
        port = port or config.a2a_server_port

        # Load agent card
        if not agent_card:
            raise ValueError('Agent card is required')

        with Path(agent_card).open() as file:
            data = json.load(file)
        agent_card_obj: AgentCard = AgentCard(**data)

        server_logger.log_structured(
            level="INFO",
            message=f"Initializing AWS Orchestrator Agent with model: {config.llm_model}",
            extra={"llm_model": config.llm_model, "host": host, "port": port}
        )

        # Create specialized agents for the custom supervisor
        server_logger.log_structured(
            level="INFO",
            message="Creating specialized agents for custom supervisor",
            extra={"agent_types": ["planner_sub_supervisor", "generator_swarm"]}
        )
        
        # Create Planner Sub-Supervisor Agent
        planner_sub_supervisor = create_planner_sub_supervisor_agent(config=config)
        
        # Create Generator Swarm Agent
        generator_swarm = create_generator_swarm_agent(config=config)

        # Create Writer React Agent
        writer_react = create_writer_react_agent(config=config)
        
        # Create Custom Supervisor Agent with agents
        supervisor_agent = create_supervisor_agent(
            agents=[planner_sub_supervisor, generator_swarm, writer_react],
            config=config,
            name="aws-orchestrator-supervisor"
        )

        # Verify supervisor is ready
        if not supervisor_agent.is_ready():
            raise RuntimeError("Supervisor agent failed to initialize properly")
        
        server_logger.log_structured(
            level="INFO",
            message="Custom Supervisor Agent initialized successfully",
            extra={
                "supervisor_name": supervisor_agent.name,
                "available_agents": supervisor_agent.list_agents(),
                "supervisor_ready": supervisor_agent.is_ready()
            }
        )

        # Create Task Lifecycle Manager
        task_lifecycle_manager = TaskLifecycleManager()

        # Create A2A Executor with our custom supervisor agent
        executor = GenericAgentExecutor(
            agent=supervisor_agent
        )

        # Create HTTP client
        client: httpx.AsyncClient = httpx.AsyncClient()

        # Create request handler
        request_handler: DefaultRequestHandler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(client),
        )

        # Create A2A server
        server: A2AStarletteApplication = A2AStarletteApplication(
            agent_card=agent_card_obj,
            http_handler=request_handler,
        )

        server_logger.log_structured(
            level="INFO",
            message=f"Starting AWS Orchestrator Agent server on {host}:{port}",
            extra={
                "host": host, 
                "port": port, 
                "log_level": config.log_level,
                "supervisor_agents": supervisor_agent.list_agents()
            }
        )
        uvicorn.run(
            server.build(),  # Use the build() method to get the ASGI application
            host=host,
            port=port,
            log_level=config.log_level.lower()
        )

    except FileNotFoundError as e:
        server_logger.log_structured(
            level="ERROR",
            message=f"File not found: {e}",
            extra={"error": str(e)}
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        server_logger.log_structured(
            level="ERROR",
            message=f"Invalid JSON in configuration file: {e}",
            extra={"error": str(e)}
        )
        sys.exit(1)
    except Exception as e:
        server_logger.log_structured(
            level="ERROR",
            message=f"An error occurred during server startup: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
