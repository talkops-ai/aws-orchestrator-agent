"""Values confirmation component — Accept Defaults style approval."""


from typing import List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=7)
class ValuesConfirmationComponent(BaseComponent):
    """
    Renders a values/defaults confirmation card.

    Matches when the workflow phase is ``values_confirmation``.
    """

    component_type = "values_confirmation"

    def can_handle(self, ctx: RenderContext) -> bool:
        return ctx.require_user_input and ctx.phase == "values_confirmation"

    def build(self, ctx: RenderContext) -> List[dict]:
        question = ctx.content_str
        context_str = ""
        if isinstance(ctx.content, dict):
            context_str = str(ctx.content.get("context", ""))

        return [
            {
                "beginRendering": {
                    "surfaceId": "values-confirmation",
                    "root": "vc-root",
                    "styles": {"primaryColor": "#8B5CF6", "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "values-confirmation",
                    "components": [
                        {
                            "id": "vc-root",
                            "component": {"Card": {"child": "vc-content"}},
                        },
                        {
                            "id": "vc-content",
                            "component": {
                                "Column": {
                                    "children": {
                                        "explicitList": [
                                            "vc-header",
                                            "vc-divider-top",
                                            "vc-question",
                                            "vc-context",
                                            "vc-divider-bottom",
                                            "vc-buttons",
                                        ]
                                    }
                                }
                            },
                        },
                        {
                            "id": "vc-header",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": ["vc-icon", "vc-title"]
                                    },
                                    "alignment": "center",
                                }
                            },
                        },
                        {
                            "id": "vc-icon",
                            "component": {
                                "Text": {
                                    "text": {"literalString": "📋"},
                                    "usageHint": "h3",
                                }
                            },
                        },
                        {
                            "id": "vc-title",
                            "component": {
                                "Text": {
                                    "text": {
                                        "literalString": "Confirm Values"
                                    },
                                    "usageHint": "h3",
                                }
                            },
                        },
                        {"id": "vc-divider-top", "component": {"Divider": {}}},
                        {
                            "id": "vc-question",
                            "component": {
                                "Text": {
                                    "text": {"path": "question"},
                                    "usageHint": "body",
                                }
                            },
                        },
                        {
                            "id": "vc-context",
                            "component": {
                                "Text": {
                                    "text": {"path": "context"},
                                    "usageHint": "code",
                                }
                            },
                        },
                        {"id": "vc-divider-bottom", "component": {"Divider": {}}},
                        {
                            "id": "vc-buttons",
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": [
                                            "vc-reject-btn",
                                            "vc-accept-btn",
                                        ]
                                    },
                                    "distribution": "spaceEvenly",
                                }
                            },
                        },
                        {
                            "id": "vc-reject-btn",
                            "component": {
                                "Button": {
                                    "child": "vc-reject-text",
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
                                        ],
                                    },
                                }
                            },
                        },
                        {
                            "id": "vc-reject-text",
                            "component": {
                                "Text": {
                                    "text": {"literalString": "✏️ Customize"}
                                }
                            },
                        },
                        {
                            "id": "vc-accept-btn",
                            "component": {
                                "Button": {
                                    "child": "vc-accept-text",
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
                                        ],
                                    },
                                }
                            },
                        },
                        {
                            "id": "vc-accept-text",
                            "component": {
                                "Text": {
                                    "text": {"literalString": "✅ Accept Defaults"}
                                }
                            },
                        },
                    ],
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "values-confirmation",
                    "path": "/",
                    "contents": [
                        {"key": "question", "valueString": question},
                        {"key": "context", "valueString": context_str},
                    ],
                }
            },
        ]
