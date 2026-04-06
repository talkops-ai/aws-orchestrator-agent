"""Error display component."""


from typing import List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=8)
class ErrorComponent(BaseComponent):
    """Renders an error card with red styling."""

    component_type = "error"

    def can_handle(self, ctx: RenderContext) -> bool:
        return ctx.response_type == "error" or ctx.status == "error"

    def build(self, ctx: RenderContext) -> List[dict]:
        return [
            {
                "beginRendering": {
                    "surfaceId": "error-display",
                    "root": "error-root",
                    "styles": {"primaryColor": "#EF4444", "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "error-display",
                    "components": [
                        {
                            "id": "error-root",
                            "component": {"Card": {"child": "error-content"}},
                        },
                        {
                            "id": "error-content",
                            "component": {
                                "Column": {
                                    "children": {
                                        "explicitList": [
                                            "error-header",
                                            "divider",
                                            "error-msg",
                                        ]
                                    }
                                }
                            },
                        },
                        {
                            "id": "error-header",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": ["error-icon", "error-title"]
                                    },
                                    "alignment": "center",
                                }
                            },
                        },
                        {
                            "id": "error-icon",
                            "component": {
                                "Text": {
                                    "text": {"literalString": "❌"},
                                    "usageHint": "h3",
                                }
                            },
                        },
                        {
                            "id": "error-title",
                            "component": {
                                "Text": {
                                    "text": {"literalString": "Error"},
                                    "usageHint": "h3",
                                }
                            },
                        },
                        {"id": "divider", "component": {"Divider": {}}},
                        {
                            "id": "error-msg",
                            "component": {
                                "Text": {
                                    "text": {"path": "errorMessage"},
                                    "usageHint": "body",
                                }
                            },
                        },
                    ],
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "error-display",
                    "path": "/",
                    "contents": [
                        {"key": "errorMessage", "valueString": ctx.content_str},
                    ],
                }
            },
        ]
