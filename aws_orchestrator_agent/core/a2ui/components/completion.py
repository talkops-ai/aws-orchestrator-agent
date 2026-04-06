"""Task completion component."""


from typing import Any, Dict, List, Optional

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=5)
class CompletionComponent(BaseComponent):
    """
    Renders a success card when the agent marks a task as complete.
    """

    component_type = "completion"

    def can_handle(self, ctx: RenderContext) -> bool:
        return ctx.is_task_complete and ctx.response_type != "error"

    def build(self, ctx: RenderContext) -> List[dict]:
        message = ctx.content_str
        metrics: Optional[Dict[str, Any]] = (
            ctx.content if isinstance(ctx.content, dict) else None
        )

        # Build optional metrics rows
        metric_components: List[dict] = []
        metric_data: List[dict] = []
        metric_ids: List[str] = []

        if metrics:
            for idx, (key, value) in enumerate(metrics.items()):
                if key in ("message", "summary", "question"):
                    continue
                mid = f"metric-{idx}"
                metric_ids.append(mid)
                metric_components.extend(
                    [
                        {
                            "id": mid,
                            "component": {
                                "Row": {
                                    "children": {
                                        "explicitList": [f"{mid}-key", f"{mid}-val"]
                                    },
                                    "distribution": "spaceBetween",
                                }
                            },
                        },
                        {
                            "id": f"{mid}-key",
                            "component": {
                                "Text": {
                                    "text": {"path": f"mk_{idx}"},
                                    "usageHint": "caption",
                                }
                            },
                        },
                        {
                            "id": f"{mid}-val",
                            "component": {
                                "Text": {
                                    "text": {"path": f"mv_{idx}"},
                                    "usageHint": "body",
                                }
                            },
                        },
                    ]
                )
                metric_data.extend(
                    [
                        {"key": f"mk_{idx}", "valueString": str(key)},
                        {"key": f"mv_{idx}", "valueString": str(value)},
                    ]
                )

        children = ["completion-header", "divider", "completion-msg"]
        if metric_ids:
            children.append("metrics-divider")
            children.extend(metric_ids)

        all_components = [
            {
                "id": "completion-root",
                "component": {
                    "Column": {"children": {"explicitList": children}}
                },
            },
            {
                "id": "completion-header",
                "component": {
                    "Row": {
                        "children": {
                            "explicitList": ["completion-icon", "completion-title"]
                        },
                        "alignment": "center",
                    }
                },
            },
            {
                "id": "completion-icon",
                "component": {"Text": {"text": {"literalString": "✅"}, "usageHint": "h3"}},
            },
            {
                "id": "completion-title",
                "component": {
                    "Text": {
                        "text": {"literalString": "Task Completed"},
                        "usageHint": "h3",
                    }
                },
            },
            {"id": "divider", "component": {"Divider": {}}},
            {
                "id": "completion-msg",
                "component": {
                    "Text": {"text": {"path": "message"}, "usageHint": "body"}
                },
            },
        ]

        if metric_ids:
            all_components.append(
                {"id": "metrics-divider", "component": {"Divider": {}}}
            )
            all_components.extend(metric_components)

        return [
            {
                "beginRendering": {
                    "surfaceId": "task-completion",
                    "root": "completion-root",
                    "styles": {"primaryColor": "#22C55E", "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "task-completion",
                    "components": all_components,
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "task-completion",
                    "path": "/",
                    "contents": [
                        {"key": "message", "valueString": message},
                        *metric_data,
                    ],
                }
            },
        ]
