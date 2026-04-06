"""Tool execution result component."""


import json
from typing import Any, List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=2)
class ToolResultComponent(BaseComponent):
    """
    Renders tool execution results — success or failure.

    Matches when metadata specifies ``component=tool_result``.
    """

    component_type = "tool_result"

    def can_handle(self, ctx: RenderContext) -> bool:
        return (
            ctx.metadata.get("component") == "tool_result"
            or ctx.metadata.get("message_type") == "tool_result"
        )

    def build(self, ctx: RenderContext) -> List[dict]:
        tool_name = (
            ctx.metadata.get("tool_name")
            or ctx.metadata.get("node", "Tool")
        )
        success = ctx.metadata.get("success", True)

        # Format result
        result = ctx.content
        if isinstance(result, dict):
            result_display = json.dumps(result, indent=2)
        elif isinstance(result, str):
            result_display = result
        else:
            result_display = str(result)

        color = "#22C55E" if success else "#EF4444"
        icon = "check_circle" if success else "error"
        status_label = "Success" if success else "Failed"

        return [
            {
                "beginRendering": {
                    "surfaceId": "tool-result",
                    "root": "result-root",
                    "styles": {"primaryColor": color, "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "tool-result",
                    "components": [
                        {
                            "id": "result-root",
                            "component": {"Card": {"child": "result-content"}},
                        },
                        {
                            "id": "result-content",
                            "component": {
                                "Column": {
                                    "children": {
                                        "explicitList": [
                                            "result-header",
                                            "divider",
                                            "result-text",
                                        ]
                                    }
                                }
                            },
                        },
                        {
                            "id": "result-header",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": [
                                            "result-icon",
                                            "result-title",
                                        ]
                                    },
                                    "alignment": "center",
                                }
                            },
                        },
                        {
                            "id": "result-icon",
                            "component": {
                                "Icon": {"name": {"path": "icon"}}
                            },
                        },
                        {
                            "id": "result-title",
                            "component": {
                                "Text": {
                                    "usageHint": "h4",
                                    "text": {"path": "title"},
                                }
                            },
                        },
                        {"id": "divider", "component": {"Divider": {}}},
                        {
                            "id": "result-text",
                            "component": {
                                "Text": {
                                    "usageHint": "code",
                                    "text": {"path": "result"},
                                }
                            },
                        },
                    ],
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "tool-result",
                    "path": "/",
                    "contents": [
                        {
                            "key": "title",
                            "valueString": f"{tool_name} - {status_label}",
                        },
                        {"key": "icon", "valueString": icon},
                        {"key": "result", "valueString": result_display},
                    ],
                }
            },
        ]
