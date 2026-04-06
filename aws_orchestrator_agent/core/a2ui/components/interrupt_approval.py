"""Interrupt approval component — tool-level HITL with Approve/Edit/Reject."""


import json
from typing import Any, Dict, List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=7)
class InterruptApprovalComponent(BaseComponent):
    """
    Renders tool-call approval UI with Approve / Edit / Reject buttons.

    Matches when the content contains ``action_requests`` (from
    HumanInTheLoopMiddleware in LangGraph).
    """

    component_type = "interrupt_approval"

    def can_handle(self, ctx: RenderContext) -> bool:
        return (
            ctx.require_user_input
            and isinstance(ctx.content, dict)
            and "action_requests" in ctx.content
        )

    def build(self, ctx: RenderContext) -> List[dict]:
        content = ctx.content
        interrupt_id = content.get("interrupt_id", "")
        action_requests: List[Dict[str, Any]] = content.get("action_requests", [])
        description = content.get(
            "description", "Tool execution pending approval"
        )

        # Build per-action sub-components
        action_components: List[dict] = []
        action_data: List[dict] = []

        for idx, action in enumerate(action_requests):
            aid = f"action-{idx}"
            tool_name = action.get("action", action.get("name", "Unknown Tool"))
            args = action.get("args", {})
            args_display = json.dumps(args, indent=2) if args else "No arguments"

            action_components.extend(
                [
                    {
                        "id": aid,
                        "component": {"Card": {"child": f"{aid}-content"}},
                    },
                    {
                        "id": f"{aid}-content",
                        "component": {
                            "Column": {
                                "children": {
                                    "explicitList": [
                                        f"{aid}-header",
                                        f"{aid}-args",
                                    ]
                                }
                            }
                        },
                    },
                    {
                        "id": f"{aid}-header",
                        "component": {
                            "Row": {
                                "children": {
                                    "explicitList": [f"{aid}-icon", f"{aid}-title"]
                                },
                                "alignment": "center",
                            }
                        },
                    },
                    {
                        "id": f"{aid}-icon",
                        "component": {
                            "Icon": {"name": {"literalString": "build"}}
                        },
                    },
                    {
                        "id": f"{aid}-title",
                        "component": {
                            "Text": {
                                "usageHint": "h4",
                                "text": {"path": f"tool_{idx}"},
                            }
                        },
                    },
                    {
                        "id": f"{aid}-args",
                        "component": {
                            "Text": {
                                "usageHint": "code",
                                "text": {"path": f"args_{idx}"},
                            }
                        },
                    },
                ]
            )
            action_data.extend(
                [
                    {"key": f"tool_{idx}", "valueString": f"🔧 {tool_name}"},
                    {"key": f"args_{idx}", "valueString": args_display},
                ]
            )

        action_ids = [f"action-{i}" for i in range(len(action_requests))]

        return [
            {
                "beginRendering": {
                    "surfaceId": "interrupt-approval",
                    "root": "approval-root",
                    "styles": {"primaryColor": "#F59E0B", "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "interrupt-approval",
                    "components": [
                        {
                            "id": "approval-root",
                            "component": {"Card": {"child": "approval-content"}},
                        },
                        {
                            "id": "approval-content",
                            "component": {
                                "Column": {
                                    "children": {
                                        "explicitList": [
                                            "approval-header",
                                            "divider1",
                                            "description-text",
                                            "actions-list",
                                            "divider-edit",
                                            "edit-instructions",
                                            "edited-args",
                                            "divider2",
                                            "button-row",
                                        ]
                                    }
                                }
                            },
                        },
                        {
                            "id": "approval-header",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": [
                                            "header-icon",
                                            "header-title",
                                        ]
                                    },
                                    "alignment": "center",
                                }
                            },
                        },
                        {
                            "id": "header-icon",
                            "component": {
                                "Icon": {"name": {"literalString": "warning"}}
                            },
                        },
                        {
                            "id": "header-title",
                            "component": {
                                "Text": {
                                    "usageHint": "h3",
                                    "text": {
                                        "literalString": "⚠️ Action Approval Required"
                                    },
                                }
                            },
                        },
                        {"id": "divider1", "component": {"Divider": {}}},
                        {
                            "id": "description-text",
                            "component": {
                                "Text": {
                                    "usageHint": "body",
                                    "text": {"path": "description"},
                                }
                            },
                        },
                        {
                            "id": "actions-list",
                            "component": {
                                "Column": {
                                    "children": {"explicitList": action_ids}
                                }
                            },
                        },
                        *action_components,
                        {"id": "divider-edit", "component": {"Divider": {}}},
                        {
                            "id": "edit-instructions",
                            "component": {
                                "Text": {
                                    "usageHint": "caption",
                                    "text": {
                                        "literalString": (
                                            "Optional: paste an edited_action JSON for an Edit decision. "
                                            "If you leave it blank, Edit will run without argument changes."
                                        )
                                    },
                                }
                            },
                        },
                        {
                            "id": "edited-args",
                            "component": {
                                "TextField": {
                                    "name": "editedArgsJson",
                                    "label": {
                                        "literalString": "edited_action JSON (optional)"
                                    },
                                    "text": {"path": "/editedArgsJson"},
                                    "textFieldType": "longText",
                                }
                            },
                        },
                        {"id": "divider2", "component": {"Divider": {}}},
                        {
                            "id": "button-row",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": [
                                            "reject-btn",
                                            "edit-btn",
                                            "approve-btn",
                                        ]
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
                                                "value": {
                                                    "literalString": "reject"
                                                },
                                            },
                                            {
                                                "key": "interrupt_id",
                                                "value": {
                                                    "literalString": interrupt_id
                                                },
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
                            "id": "edit-btn",
                            "component": {
                                "Button": {
                                    "child": "edit-text",
                                    "primary": False,
                                    "action": {
                                        "name": "hitl_response",
                                        "context": [
                                            {
                                                "key": "decision",
                                                "value": {
                                                    "literalString": "edit"
                                                },
                                            },
                                            {
                                                "key": "interrupt_id",
                                                "value": {
                                                    "literalString": interrupt_id
                                                },
                                            },
                                            {
                                                "key": "edited_args_json",
                                                "value": {
                                                    "path": "/editedArgsJson"
                                                },
                                            },
                                        ],
                                    },
                                }
                            },
                        },
                        {
                            "id": "edit-text",
                            "component": {
                                "Text": {"text": {"literalString": "✏️ Edit"}}
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
                                                "value": {
                                                    "literalString": "approve"
                                                },
                                            },
                                            {
                                                "key": "interrupt_id",
                                                "value": {
                                                    "literalString": interrupt_id
                                                },
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
                    "surfaceId": "interrupt-approval",
                    "path": "/",
                    "contents": [
                        {"key": "description", "valueString": description},
                        {"key": "interruptId", "valueString": interrupt_id},
                        {"key": "editedArgsJson", "valueString": ""},
                        *action_data,
                    ],
                }
            },
        ]
