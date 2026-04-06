"""
TF Operator — Coordinator-level tools.

Tools available directly to the TFCoordinator deep agent LLM (not subagents).
These are registered via ``TFCoordinator.get_tools()``.

Tools:
    sync_workspace      — materialise virtual /workspace/ files to disk
    request_user_input  — generic HITL gate: pause and ask the user anything
"""

from aws_orchestrator_agent.core.agents.tf_operator.tools.user_input_tool import (
    create_user_input_tool,
)

__all__ = [
    "create_user_input_tool",
]
