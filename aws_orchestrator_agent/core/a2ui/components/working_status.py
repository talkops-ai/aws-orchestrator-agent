"""Working / processing status component."""


from typing import Any, Dict, List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=0)
class WorkingStatusComponent(BaseComponent):
    """
    Displays a status card while the agent is processing.

    This is the **lowest-priority** catch-all — it renders whenever no
    higher-priority component matches and the task is still in progress.
    """

    component_type = "working_status"

    def can_handle(self, ctx: RenderContext) -> bool:
        return (
            not ctx.is_task_complete
            and not ctx.require_user_input
            and ctx.response_type not in ("error", "a2ui")
        )

    def build(self, ctx: RenderContext) -> List[dict]:
        message_type = ctx.metadata.get("message_type", "")

        # Distinguish tool_call activity from generic working status
        if message_type == "tool_call":
            icon = "⚙️"
            color = "#6366F1"   # indigo — distinct from plain working
            title = "Calling tool"
        elif message_type == "ai_text":
            icon = "💬"
            color = "#3B82F6"   # blue
            title = "Agent response"
        else:
            icon_map = {
                "working": "⏳",
                "success": "✅",
                "error": "❌",
                "warning": "⚠️",
                "info": "ℹ️",
            }
            color_map = {
                "working": "#6366F1",
                "success": "#22C55E",
                "error": "#EF4444",
                "warning": "#F59E0B",
                "info": "#3B82F6",
            }
            status = ctx.status
            icon = icon_map.get(status, "⏳")
            color = color_map.get(status, "#6366F1")
            title = "Processing..."

        surface_id = ctx.metadata.get("surface_id", "task-status")

        return [
            {
                "beginRendering": {
                    "surfaceId": surface_id,
                    "root": "task-root",
                    "styles": {"primaryColor": color, "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": surface_id,
                    "components": [
                        {
                            "id": "task-root",
                            "component": {
                                "Text": {
                                    "usageHint": "body",
                                    "text": {"path": "content"},
                                }
                            },
                        }
                    ],
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": surface_id,
                    "path": "/",
                    "contents": [
                        {"key": "content", "valueString": ctx.content_str},
                    ],
                }
            },
        ]
