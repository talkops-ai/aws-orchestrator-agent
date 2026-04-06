"""
Generic User Input A2UI component.

Dynamically renders whatever the ``request_user_input`` tool sends:
    - Title header
    - Question text
    - Optional context body
    - Dynamic text input fields (from ``input_fields``)
    - Dynamic option buttons (from ``options``)

No hardcoded field names, no domain-specific logic.
Everything is driven by the payload from the tool.

Matches when:
    - ``require_user_input`` is True AND
    - ``content`` is a dict with ``type == "user_input_request"``
"""

from typing import Any, Dict, List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)


@register_component(priority=8)
class UserInputComponent(BaseComponent):
    """Renders a dynamic user input card from the generic HITL tool.

    Priority 8 — above ``InterruptApprovalComponent`` (7) so this
    matches first for ``user_input_request`` type payloads.
    """

    component_type = "user_input_request"

    def can_handle(self, ctx: RenderContext) -> bool:
        return (
            ctx.require_user_input
            and isinstance(ctx.content, dict)
            and ctx.content.get("type") == "user_input_request"
        )

    def build(self, ctx: RenderContext) -> List[dict]:
        content: Dict[str, Any] = ctx.content  # type: ignore[assignment]

        title = content.get("title", "Input Required")
        question = content.get("question", "")
        context_text = content.get("context", "")
        options: List[Dict[str, Any]] = content.get("options", [])
        input_fields: List[Dict[str, Any]] = content.get("input_fields", [])

        # ── Build dynamic child IDs ──────────────────────────────────
        children_ids = ["input-header", "divider-top", "input-question"]

        if context_text:
            children_ids.append("input-context")

        if input_fields:
            children_ids.append("divider-fields")
            for idx, _ in enumerate(input_fields):
                children_ids.append(f"field-{idx}")

        if options:
            children_ids.append("divider-buttons")
            children_ids.append("button-row")

        # ── Build components list ────────────────────────────────────
        components: List[dict] = [
            # Root Column (removed Card wrapper for clean markdown-like look)
            {
                "id": "input-root",
                "component": {
                    "Column": {
                        "children": {"explicitList": children_ids}
                    }
                },
            },
            # Header
            {
                "id": "input-header",
                "component": {
                    "Text": {
                        "text": {"path": "title"},
                        "usageHint": "h3",
                    }
                },
            },
            {"id": "divider-top", "component": {"Divider": {}}},
            # Question
            {
                "id": "input-question",
                "component": {
                    "Text": {
                        "text": {"path": "question"},
                        "usageHint": "body",
                    }
                },
            },
        ]

        # ── Optional context ─────────────────────────────────────────
        if context_text:
            components.append({
                "id": "input-context",
                "component": {
                    "Text": {
                        "text": {"path": "context"},
                        "usageHint": "caption",
                    }
                },
            })

        # ── Dynamic text fields ──────────────────────────────────────
        if input_fields:
            components.append(
                {"id": "divider-fields", "component": {"Divider": {}}}
            )
            for idx, field in enumerate(input_fields):
                field_key = field.get("key", f"field_{idx}")
                field_label = field.get("label", f"Field {idx + 1}")
                components.append({
                    "id": f"field-{idx}",
                    "component": {
                        "TextField": {
                            "name": field_key,
                            "label": {"literalString": field_label},
                            "text": {"path": f"/{field_key}"},
                            "textFieldType": "shortText",
                        }
                    },
                })

        # ── Dynamic option buttons ───────────────────────────────────
        button_ids: List[str] = []
        if options:
            components.append(
                {"id": "divider-buttons", "component": {"Divider": {}}}
            )

            for idx, option in enumerate(options):
                btn_id = f"btn-{idx}"
                btn_text_id = f"btn-text-{idx}"
                button_ids.append(btn_id)

                opt_key = option.get("key", f"option_{idx}")
                opt_label = option.get("label", f"Option {idx + 1}")
                is_primary = option.get("primary", False)

                # Build action context: always include the decision key,
                # plus any input_field values so they're sent back too
                action_context = [
                    {
                        "key": "decision",
                        "value": {"literalString": opt_key},
                    },
                ]
                # Attach all input field values to the action
                for field in input_fields:
                    field_key = field.get("key", "")
                    if field_key:
                        action_context.append({
                            "key": field_key,
                            "value": {"path": f"/{field_key}"},
                        })

                components.append({
                    "id": btn_id,
                    "component": {
                        "Button": {
                            "child": btn_text_id,
                            "primary": is_primary,
                            "action": {
                                "name": "hitl_response",
                                "context": action_context,
                            },
                        }
                    },
                })
                components.append({
                    "id": btn_text_id,
                    "component": {
                        "Text": {
                            "text": {"literalString": opt_label}
                        }
                    },
                })

            components.append({
                "id": "button-row",
                "component": {
                    "Row": {
                        "children": {"explicitList": button_ids},
                        "distribution": "spaceEvenly",
                    }
                },
            })

        # ── Data model ───────────────────────────────────────────────
        data_entries = [
            {"key": "title", "valueString": title},
            {"key": "question", "valueString": question},
        ]
        if context_text:
            data_entries.append(
                {"key": "context", "valueString": context_text}
            )
        # Pre-fill input field defaults
        for field in input_fields:
            field_key = field.get("key", "")
            field_default = field.get("default", "")
            if field_key:
                data_entries.append(
                    {"key": field_key, "valueString": field_default}
                )

        return [
            {
                "beginRendering": {
                    "surfaceId": "user-input",
                    "root": "input-root",
                    "styles": {
                        "primaryColor": "#6366F1",
                        "foregroundColor": "#E2E8F0",
                        "surfaceColor": "#1E293B",
                        "font": "Inter",
                    },
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "user-input",
                    "components": components,
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "user-input",
                    "path": "/",
                    "contents": data_entries,
                }
            },
        ]
