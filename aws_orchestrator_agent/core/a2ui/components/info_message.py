"""Informational message component (no action buttons)."""


from typing import List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=3)
class InfoMessageComponent(BaseComponent):
    """
    Renders a plain informational card.

    Used when the agent needs user input but the message is not an
    approval request (no approve/reject buttons).
    """

    component_type = "info_message"

    def can_handle(self, ctx: RenderContext) -> bool:
        # Catch-all for require_user_input that isn't an approval
        # Higher-priority approval components will match first
        return ctx.require_user_input

    def build(self, ctx: RenderContext) -> List[dict]:
        title = ctx.metadata.get("title", "Clarification Needed")
        message = ctx.content_str

        # Render as clean markdown — no card/oval shape, no icon, no divider.
        # Each line of the question is rendered as readable markdown text so
        # the client displays it inline with the conversation.
        lines = message.strip().split("\n") if message.strip() else []
        formatted_lines = "\n".join(f"- {line.strip()}" for line in lines if line.strip())
        if not formatted_lines:
            formatted_lines = message.strip()

        markdown_text = f"### {title}\n\n{formatted_lines}\n"

        return [
            {
                "beginRendering": {
                    "surfaceId": "info-message",
                    "root": "info-root",
                    "styles": {
                        "primaryColor": "#3B82F6",
                        "foregroundColor": "#E2E8F0",
                        "font": "Inter",
                    },
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "info-message",
                    "components": [
                        {
                            "id": "info-root",
                            "component": {
                                "Text": {
                                    "text": {"path": "markdown"},
                                    "usageHint": "body",
                                }
                            },
                        },
                    ],
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "info-message",
                    "path": "/",
                    "contents": [
                        {"key": "markdown", "valueString": markdown_text},
                    ],
                }
            },
        ]
