"""
Core module for AWS Orchestrator Agent.

This module contains the core components for A2A protocol integration,
agent management, Agent Card service discovery, and Supervisor Agent orchestration.
"""

from aws_orchestrator_agent.core.a2a_executor import GenericAgentExecutor, ExecutorValidationMixin
from aws_orchestrator_agent.core.task_lifecycle import TaskLifecycleManager
from aws_orchestrator_agent.core.agents.supervisor_agent import (
    create_supervisor_agent
)

__all__ = [
    # A2A Executor
    "GenericAgentExecutor",
    "ExecutorValidationMixin",
    # Task Lifecycle Management
    "TaskLifecycleManager",
    # Supervisor Agent
    "create_supervisor_agent",
]
