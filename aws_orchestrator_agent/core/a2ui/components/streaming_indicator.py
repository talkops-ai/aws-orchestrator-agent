"""Streaming / processing indicator component."""


from typing import List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=1)
class StreamingIndicatorComponent(BaseComponent):
    """
    Shows an animated processing indicator.

    Only matches when metadata explicitly requests ``component=streaming``.
    """

    component_type = "streaming_indicator"

    def can_handle(self, ctx: RenderContext) -> bool:
        return ctx.metadata.get("component") == "streaming"

    def build(self, ctx: RenderContext) -> List[dict]:
        message = ctx.content_str
        activity = ctx.metadata.get("activity", "working")

        icon_map = {
            "working": "hourglass_empty",
            "thinking": "psychology",
            "fetching": "cloud_download",
            "calling": "phone_in_talk",
            "processing": "settings",
        }
        icon = icon_map.get(activity, "hourglass_empty")

        return [
            {
                "beginRendering": {
                    "surfaceId": "streaming",
                    "root": "streaming-root",
                    "styles": {"primaryColor": "#6366F1", "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "streaming",
                    "components": [
                        {
                            "id": "streaming-root",
                            "component": {"Card": {"child": "streaming-content"}},
                        },
                        {
                            "id": "streaming-content",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": [
                                            "streaming-icon",
                                            "streaming-text",
                                        ]
                                    },
                                    "alignment": "center",
                                }
                            },
                        },
                        {
                            "id": "streaming-icon",
                            "component": {
                                "Icon": {"name": {"path": "icon"}}
                            },
                        },
                        {
                            "id": "streaming-text",
                            "component": {
                                "Text": {
                                    "usageHint": "body",
                                    "text": {"path": "message"},
                                }
                            },
                        },
                    ],
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "streaming",
                    "path": "/",
                    "contents": [
                        {"key": "message", "valueString": message},
                        {"key": "icon", "valueString": icon},
                    ],
                }
            },
        ]
