"""HITL (Human-in-the-Loop) approval component with Approve/Reject buttons."""


import json
from typing import Any, Dict, List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


def _is_approval_request(content: Any, metadata: Dict[str, Any]) -> bool:
    """Determine if the content constitutes an approval request."""
    approval_keywords = [
        "approve", "approval", "confirm", "review", "accept",
        "reject", "permission", "authorize", "proceed",
    ]

    text = ""
    if isinstance(content, str):
        text = content.lower()
    elif isinstance(content, dict):
        text = str(content.get("question", content.get("message", ""))).lower()

    if any(kw in text for kw in approval_keywords):
        return True

    # Check metadata signals
    if metadata.get("interrupt_type") in (
        "planning_review",
        "generation_review",
        "tool_call_approval",
    ):
        return True

    return False


@register_component(priority=6)
class HITLApprovalComponent(BaseComponent):
    """
    Renders an approval card with Approve / Reject buttons.

    Matches when:
    - User input is required AND
    - The content looks like an approval request (keywords or metadata)
    """

    component_type = "hitl_approval"

    def can_handle(self, ctx: RenderContext) -> bool:
        if not ctx.require_user_input:
            return False
        # Don't match values_confirmation — that has its own component
        if ctx.phase == "values_confirmation":
            return False
        # Don't match interrupt_approval — that has action_requests
        if isinstance(ctx.content, dict) and "action_requests" in ctx.content:
            return False
        return _is_approval_request(ctx.content, ctx.metadata)

    def build(self, ctx: RenderContext) -> List[dict]:
        question = ctx.content_str
        phase = ctx.phase
        context_str = ""
        if isinstance(ctx.content, dict):
            context_str = str(ctx.content.get("context", ""))
        if not context_str:
            context_str = "No additional context provided"

        return [
            {
                "beginRendering": {
                    "surfaceId": "hitl-approval",
                    "root": "hitl-root",
                    "styles": {"primaryColor": "#F59E0B", "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "hitl-approval",
                    "components": [
                        {
                            "id": "hitl-root",
                            "component": {"Card": {"child": "hitl-content"}},
                        },
                        {
                            "id": "hitl-content",
                            "component": {
                                "Column": {
                                    "children": {
                                        "explicitList": [
                                            "hitl-header",
                                            "divider-top",
                                            "hitl-question",
                                            "hitl-context",
                                            "divider-bottom",
                                            "hitl-buttons",
                                        ]
                                    }
                                }
                            },
                        },
                        {
                            "id": "hitl-header",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": ["hitl-icon", "hitl-title"]
                                    },
                                    "alignment": "center",
                                }
                            },
                        },
                        {
                            "id": "hitl-icon",
                            "component": {
                                "Text": {
                                    "text": {"literalString": "⚠️"},
                                    "usageHint": "h3",
                                }
                            },
                        },
                        {
                            "id": "hitl-title",
                            "component": {
                                "Text": {
                                    "text": {"literalString": "Approval Required"},
                                    "usageHint": "h3",
                                }
                            },
                        },
                        {"id": "divider-top", "component": {"Divider": {}}},
                        {
                            "id": "hitl-question",
                            "component": {
                                "Text": {
                                    "text": {"path": "question"},
                                    "usageHint": "body",
                                }
                            },
                        },
                        {
                            "id": "hitl-context",
                            "component": {
                                "Text": {
                                    "text": {"path": "context"},
                                    "usageHint": "caption",
                                }
                            },
                        },
                        {"id": "divider-bottom", "component": {"Divider": {}}},
                        {
                            "id": "hitl-buttons",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": ["reject-btn", "approve-btn"]
                                    },
                                    "distribution": "spaceEvenly",
                                }
                            },
                        },
                        {
                            "id": "reject-btn",
                            "component": {
                                "Button": {
                                    "child": "reject-text",
                                    "primary": False,
                                    "action": {
                                        "name": "hitl_response",
                                        "context": [
                                            {
                                                "key": "decision",
                                                "value": {"literalString": "reject"},
                                            },
                                            {
                                                "key": "phase",
                                                "value": {"literalString": phase},
                                            },
                                        ],
                                    },
                                }
                            },
                        },
                        {
                            "id": "reject-text",
                            "component": {
                                "Text": {"text": {"literalString": "❌ Reject"}}
                            },
                        },
                        {
                            "id": "approve-btn",
                            "component": {
                                "Button": {
                                    "child": "approve-text",
                                    "primary": True,
                                    "action": {
                                        "name": "hitl_response",
                                        "context": [
                                            {
                                                "key": "decision",
                                                "value": {"literalString": "approve"},
                                            },
                                            {
                                                "key": "phase",
                                                "value": {"literalString": phase},
                                            },
                                        ],
                                    },
                                }
                            },
                        },
                        {
                            "id": "approve-text",
                            "component": {
                                "Text": {"text": {"literalString": "✅ Approve"}}
                            },
                        },
                    ],
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "hitl-approval",
                    "path": "/",
                    "contents": [
                        {"key": "question", "valueString": question},
                        {"key": "context", "valueString": context_str},
                        {"key": "phase", "valueString": phase},
                    ],
                }
            },
        ]
