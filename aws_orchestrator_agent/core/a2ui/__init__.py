"""
A2UI Component Registry for the AWS Orchestrator Agent.

Public API::

    from aws_orchestrator_agent.core.a2ui import (
        # Registry
        get_registry,          # Global registry singleton
        BaseComponent,         # ABC for custom components
        RenderContext,         # Input to every component
        register_component,    # Decorator for registration
        ComponentRegistry,     # The registry class itself

        # Catalog
        get_catalog_manager,   # Global catalog manager singleton
        CatalogManager,        # Catalog negotiation class
        CatalogEntry,          # Data class for registered catalogs
        STANDARD_CATALOG_ID,   # Well-known standard catalog URI
        A2UI_CLIENT_CAPABILITIES_KEY,
        SUPPORTED_CATALOG_IDS_KEY,
        INLINE_CATALOGS_KEY,
    )

Standard & custom components are auto-registered on first access via
``get_registry()``.
"""

from aws_orchestrator_agent.core.a2ui.registry import (
    BaseComponent,
    ComponentRegistry,
    RenderContext,
    get_registry,
    register_component,
)

from aws_orchestrator_agent.core.a2ui.catalog_manager import (
    CatalogEntry,
    CatalogManager,
    get_catalog_manager,
    STANDARD_CATALOG_ID,
    A2UI_CLIENT_CAPABILITIES_KEY,
    SUPPORTED_CATALOG_IDS_KEY,
    INLINE_CATALOGS_KEY,
)

__all__ = [
    # Registry
    "BaseComponent",
    "ComponentRegistry",
    "RenderContext",
    "get_registry",
    "register_component",
    # Catalog
    "CatalogEntry",
    "CatalogManager",
    "get_catalog_manager",
    "STANDARD_CATALOG_ID",
    "A2UI_CLIENT_CAPABILITIES_KEY",
    "SUPPORTED_CATALOG_IDS_KEY",
    "INLINE_CATALOGS_KEY",
]

