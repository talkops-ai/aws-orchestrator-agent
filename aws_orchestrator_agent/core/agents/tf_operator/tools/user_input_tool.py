"""
Generic HITL user input tool for the TF Coordinator deep agent.

A single reusable tool that the coordinator can call for ANY purpose:

    - Commit approval: "Push to GitHub or keep local? Provide repo/branch."
    - Next steps: "Module done. Generate another? Update existing? Done?"
    - Clarification: "Which VPC CIDR range do you want?"
    - Any future need: Tool is fully generic — no hardcoded decisions.

Design:
    - The TOOL is dumb — it just passes the payload to ``interrupt()`` and
      returns the raw response in a ``ToolMessage``.
    - The COORDINATOR LLM is the intelligent agent — it decides WHEN to call
      this tool, WHAT to ask, and HOW to interpret the response.
    - The A2UI COMPONENT is dynamic — it renders whatever options and
      input_fields the tool sends.  No hardcoded field names.

Follows the established HITL pattern from:
    ``tf_planner/new_module/shared/hitl.py``

    - Pydantic schema for the interrupt payload
    - ``interrupt()`` with validated payload
    - Returns ``Command(update={"messages": [ToolMessage(...)]})``
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from aws_orchestrator_agent.utils import AgentLogger

logger = AgentLogger("UserInputTool")


# ---------------------------------------------------------------------------
# Pydantic schema — generic, no hardcoded fields
# ---------------------------------------------------------------------------

class UserInputOption(BaseModel):
    """A single option button for the user to choose from."""

    key: str = Field(description="Machine-readable key (e.g. 'push_to_github', 'generate_another')")
    label: str = Field(description="Human-readable button label (e.g. '🚀 Push to GitHub')")
    description: str = Field(default="", description="Optional tooltip or subtitle")
    primary: bool = Field(default=False, description="Whether this is the primary/highlighted option")


class UserInputField(BaseModel):
    """A text input field for collecting user data."""

    key: str = Field(description="Machine-readable key (e.g. 'repository', 'branch')")
    label: str = Field(description="Human-readable field label (e.g. 'Repository (owner/repo)')")
    default: str = Field(default="", description="Default value to pre-fill")
    required: bool = Field(default=False, description="Whether this field is required")


class UserInputPayload(BaseModel):
    """Schema for the generic HITL interrupt payload.

    Rendered by ``UserInputComponent`` (A2UI).  The ``type`` field
    is used by the component registry for routing.

    The coordinator LLM populates this dynamically — the schema
    imposes no domain-specific constraints.
    """

    type: str = Field(
        default="user_input_request",
        description="Interrupt type — used by A2UI for routing (do not change)",
    )
    status: str = Field(
        default="input_required",
        description="Always 'input_required' when interrupt is active",
    )
    title: str = Field(
        default="",
        description="Card header title (e.g. 'Module Validated', 'Task Complete')",
    )
    question: str = Field(
        description="The question or message to present to the user",
    )
    context: str = Field(
        default="",
        description="Additional context (rendered as body text below the question)",
    )
    options: List[UserInputOption] = Field(
        default_factory=list,
        description="Buttons for the user to choose from",
    )
    input_fields: List[UserInputField] = Field(
        default_factory=list,
        description="Text fields for collecting user data",
    )


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def create_user_input_tool():
    """Factory that returns the generic ``request_user_input`` tool.

    CRITICAL RULES (from LangGraph docs):
        - Do NOT wrap interrupt() in try/except — it raises GraphInterrupt
        - interrupt() payload MUST be JSON-serializable
        - Side effects before interrupt() MUST be idempotent
    """

    @tool
    def request_user_input(
        question: str,
        runtime: ToolRuntime,
        title: str = "",
        context: str = "",
        options: Optional[List[Dict[str, Any]]] = None,
        input_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> Command:
        """Pause execution and ask the user a question.

        This is the ONLY way to request user input during the coordinator workflow.
        DO NOT write questions as text output — only this tool triggers the
        human-in-the-loop interrupt.

        Use this for:
        - Commit approval (push to GitHub or keep local)
        - Asking what the user wants to do next
        - Collecting additional information (repo, branch, CIDR, etc.)
        - Any decision that requires human input

        Args:
            question: The question to present to the user.
            title:    Card header (e.g. "Module Validated", "Task Complete").
            context:  Additional context shown below the question.
            options:  List of option dicts, each with keys:
                      - key: machine-readable id (e.g. "push_to_github")
                      - label: button text (e.g. "🚀 Push to GitHub")
                      - description: optional tooltip (default "")
                      - primary: highlight this button (default false)
            input_fields: List of field dicts, each with keys:
                      - key: machine-readable id (e.g. "repository")
                      - label: field label (e.g. "Repository (owner/repo)")
                      - default: pre-filled value (default "")
                      - required: whether field is mandatory (default false)

        Returns:
            Command with ToolMessage containing the user's response.
        """
        # Build validated options
        opt_models = [
            UserInputOption(**o) for o in (options or [])
        ]
        field_models = [
            UserInputField(**f) for f in (input_fields or [])
        ]

        # Build validated payload
        payload = UserInputPayload(
            title=title,
            question=question,
            context=context,
            options=opt_models,
            input_fields=field_models,
        )

        logger.info(
            "Requesting user input",
            extra={
                "title": title,
                "question_preview": question[:100],
                "option_count": len(opt_models),
                "field_count": len(field_models),
            },
        )

        # CRITICAL: Do NOT wrap this in try/except
        # interrupt() raises GraphInterrupt, which must propagate
        human_response = interrupt(payload.model_dump())

        human_response_str = (
            json.dumps(human_response, indent=2)
            if isinstance(human_response, dict)
            else str(human_response)
        )

        logger.info(
            "User responded",
            extra={
                "response_type": type(human_response).__name__,
                "response_preview": str(human_response)[:200],
            },
        )

        return Command(update={
            "messages": [
                ToolMessage(
                    content=(
                        f"User response to: \"{question}\"\n\n"
                        f"Response: {human_response_str}\n\n"
                        f"Act on the user's response according to your workflow instructions."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    return request_user_input
