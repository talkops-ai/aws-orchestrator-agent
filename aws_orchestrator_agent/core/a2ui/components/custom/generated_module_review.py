"""
Generated Module Review Component

Renders a rich, interactive review card when the AWS Orchestrator agent has
generated a module (e.g. a Terraform VPC module).  Shows:

  • Module summary (name, description, file count)
  • Per-file collapsible/scrollable cards with syntax-highlighted content
  • Approve / Request Changes buttons

# How to trigger this component from your agent

Yield an ``AgentResponse`` with:

```python
AgentResponse(
    content={
        "module_name": "vpc-module",
        "description": "VPC with public/private subnets",
        "files": [
            {"path": "main.tf",      "content": "resource \\"aws_vpc\\" ...", "language": "hcl"},
            {"path": "variables.tf", "content": "variable \\"vpc_cidr\\" ...", "language": "hcl"},
            {"path": "outputs.tf",   "content": "output \\"vpc_id\\" ...",    "language": "hcl"},
        ],
    },
    is_task_complete=False,
    require_user_input=True,
    response_type="data",
    metadata={"component": "generated_module_review"},
)
```
"""


import json
from typing import Any, Dict, List

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    RenderContext,
    register_component,
)

# Custom catalog URI — could also be None if only standard widgets are used
AWS_ORCHESTRATOR_CATALOG_ID = (
    "https://github.com/talkops-ai/aws-orchestrator-agent/specification/"
    "aws_orchestrator_catalog_definition.json"
)


@register_component(priority=25)
class GeneratedModuleReviewComponent(BaseComponent):
    """
    Renders a generated-module review card.

    The card contains:

    1. **Module header** — name, description, file count.
    2. **File cards** — one collapsible card per generated file showing the
       file path and its content in a scrollable code block.
    3. **Action row** — Approve (applies the module) / Request Changes.

    The component uses only standard A2UI widgets (Card, Column, Row, Text,
    Button, Divider) so it works without a custom catalog.  The rich
    rendering (collapsible / scrollable) is achieved by wrapping each
    file's content in a Card → Column → Text(code) pattern that clients
    can render with native scroll behaviour.
    """

    component_type = "generated_module_review"
    catalog_id = None  # Uses standard widgets only

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def can_handle(self, ctx: RenderContext) -> bool:
        return (
            ctx.metadata.get("component") == "generated_module_review"
            and isinstance(ctx.content, dict)
            and "files" in ctx.content
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, ctx: RenderContext) -> List[dict]:
        content: Dict[str, Any] = ctx.content  # type: ignore[assignment]
        module_name = content.get("module_name", "Generated Module")
        description = content.get("description", "")
        files: List[Dict[str, Any]] = content.get("files", [])

        # ── Component tree ──────────────────────────────────────────
        # root-card
        #  └─ root-column
        #       ├─ header-row  (icon + title + badge)
        #       ├─ description-text
        #       ├─ divider-header
        #       ├─ files-column
        #       │    ├─ file-0-card  ─┐
        #       │    │   └─ file-0-col │  For each file
        #       │    │       ├─ path   │
        #       │    │       ├─ div    │
        #       │    │       └─ code   │
        #       │    ├─ file-1-card  ─┘
        #       │    └─ …
        #       ├─ divider-footer
        #       └─ buttons-row  (reject + approve)

        # --- Build file sub-components & data entries -----------------
        file_components: List[dict] = []
        file_data: List[dict] = []
        file_card_ids: List[str] = []

        for idx, f in enumerate(files):
            fid = f"file-{idx}"
            file_card_ids.append(f"{fid}-card")

            file_path = f.get("path", f"file_{idx}")
            file_content = f.get("content", "")
            language = f.get("language", "")

            file_components.extend(
                [
                    # Card wrapper (renders as collapsible on supported clients)
                    {
                        "id": f"{fid}-card",
                        "component": {"Card": {"child": f"{fid}-col"}},
                    },
                    {
                        "id": f"{fid}-col",
                        "component": {
                            "Column": {
                                "children": {
                                    "explicitList": [
                                        f"{fid}-path-row",
                                        f"{fid}-div",
                                        f"{fid}-code",
                                    ]
                                }
                            }
                        },
                    },
                    # Path header with file icon
                    {
                        "id": f"{fid}-path-row",
                        "component": {
                            "Row": {
                                "children": {
                                    "explicitList": [
                                        f"{fid}-icon",
                                        f"{fid}-path",
                                        f"{fid}-lang",
                                    ]
                                },
                                "alignment": "center",
                            }
                        },
                    },
                    {
                        "id": f"{fid}-icon",
                        "component": {
                            "Icon": {"name": {"literalString": "description"}}
                        },
                    },
                    {
                        "id": f"{fid}-path",
                        "component": {
                            "Text": {
                                "text": {"path": f"fp_{idx}"},
                                "usageHint": "h4",
                            }
                        },
                    },
                    {
                        "id": f"{fid}-lang",
                        "component": {
                            "Text": {
                                "text": {"path": f"fl_{idx}"},
                                "usageHint": "caption",
                            }
                        },
                    },
                    {
                        "id": f"{fid}-div",
                        "component": {"Divider": {}},
                    },
                    # Code block — usageHint "code" triggers monospace / scrollable
                    {
                        "id": f"{fid}-code",
                        "component": {
                            "Text": {
                                "text": {"path": f"fc_{idx}"},
                                "usageHint": "code",
                            }
                        },
                    },
                ]
            )

            file_data.extend(
                [
                    {"key": f"fp_{idx}", "valueString": f"📄 {file_path}"},
                    {"key": f"fl_{idx}", "valueString": language.upper() if language else ""},
                    {"key": f"fc_{idx}", "valueString": file_content},
                ]
            )

        # --- Assemble top-level component tree -------------------------
        all_components: List[dict] = [
            {
                "id": "root-card",
                "component": {"Card": {"child": "root-column"}},
            },
            {
                "id": "root-column",
                "component": {
                    "Column": {
                        "children": {
                            "explicitList": [
                                "header-row",
                                "description-text",
                                "divider-header",
                                "files-column",
                                "divider-footer",
                                "buttons-row",
                            ]
                        }
                    }
                },
            },
            # ── Header ─────────────────────────────────────────────
            {
                "id": "header-row",
                "component": {
                    "Row": {
                        "children": {
                            "explicitList": [
                                "header-icon",
                                "module-name",
                                "file-count-badge",
                            ]
                        },
                        "alignment": "center",
                        "distribution": "spaceBetween",
                    }
                },
            },
            {
                "id": "header-icon",
                "component": {
                    "Text": {
                        "text": {"literalString": "📦"},
                        "usageHint": "h3",
                    }
                },
            },
            {
                "id": "module-name",
                "component": {
                    "Text": {
                        "text": {"path": "moduleName"},
                        "usageHint": "h3",
                    }
                },
            },
            {
                "id": "file-count-badge",
                "component": {
                    "Text": {
                        "text": {"path": "fileCountBadge"},
                        "usageHint": "caption",
                    }
                },
            },
            # ── Description ────────────────────────────────────────
            {
                "id": "description-text",
                "component": {
                    "Text": {
                        "text": {"path": "description"},
                        "usageHint": "body",
                    }
                },
            },
            {"id": "divider-header", "component": {"Divider": {}}},
            # ── File cards ─────────────────────────────────────────
            {
                "id": "files-column",
                "component": {
                    "Column": {
                        "children": {"explicitList": file_card_ids}
                    }
                },
            },
            *file_components,
            # ── Footer buttons ─────────────────────────────────────
            {"id": "divider-footer", "component": {"Divider": {}}},
            {
                "id": "buttons-row",
                "component": {
                    "Row": {
                        "children": {
                            "explicitList": [
                                "changes-btn",
                                "approve-btn",
                            ]
                        },
                        "distribution": "spaceEvenly",
                    }
                },
            },
            {
                "id": "changes-btn",
                "component": {
                    "Button": {
                        "child": "changes-text",
                        "primary": False,
                        "action": {
                            "name": "hitl_response",
                            "context": [
                                {
                                    "key": "decision",
                                    "value": {"literalString": "request_changes"},
                                },
                                {
                                    "key": "module_name",
                                    "value": {"path": "moduleName"},
                                },
                            ],
                        },
                    }
                },
            },
            {
                "id": "changes-text",
                "component": {
                    "Text": {"text": {"literalString": "✏️ Request Changes"}}
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
                                    "key": "module_name",
                                    "value": {"path": "moduleName"},
                                },
                            ],
                        },
                    }
                },
            },
            {
                "id": "approve-text",
                "component": {
                    "Text": {"text": {"literalString": "✅ Approve Module"}}
                },
            },
        ]

        # --- Data model ------------------------------------------------
        data_contents = [
            {"key": "moduleName", "valueString": module_name},
            {
                "key": "description",
                "valueString": description or "Generated module ready for review",
            },
            {
                "key": "fileCountBadge",
                "valueString": f"📁 {len(files)} file{'s' if len(files) != 1 else ''}",
            },
            *file_data,
        ]

        return [
            {
                "beginRendering": {
                    "surfaceId": "module-review",
                    "root": "root-card",
                    "styles": {"primaryColor": "#8B5CF6", "font": "Inter"},
                }
            },
            {
                "surfaceUpdate": {
                    "surfaceId": "module-review",
                    "components": all_components,
                }
            },
            {
                "dataModelUpdate": {
                    "surfaceId": "module-review",
                    "path": "/",
                    "contents": data_contents,
                }
            },
        ]
