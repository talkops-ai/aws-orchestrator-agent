"""
Standard A2UI Components — auto-registered on import.

These components handle the common response patterns that every A2A agent
needs: working status, task completion, errors, HITL approval, etc.

Import this package to register all standard components::

    import aws_orchestrator_agent.core.a2ui.components  # registers all
"""

# Import all standard component modules so their @register_component
# decorators fire and populate the global registry.
from aws_orchestrator_agent.core.a2ui.components.working_status import WorkingStatusComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.completion import CompletionComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.error import ErrorComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.info_message import InfoMessageComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.hitl_approval import HITLApprovalComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.interrupt_approval import InterruptApprovalComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.values_confirmation import ValuesConfirmationComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.streaming_indicator import StreamingIndicatorComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.user_input import UserInputComponent  # noqa: F401
from aws_orchestrator_agent.core.a2ui.components.tool_result import ToolResultComponent  # noqa: F401
