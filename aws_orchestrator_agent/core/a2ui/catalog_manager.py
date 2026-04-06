"""
A2UI Catalog Manager

Generic, extensible catalog manager for A2UI schema negotiation.
Replaces the hardcoded ``ComponentCatalogBuilder`` with a registry-aware
approach where catalogs are registered dynamically alongside components.

Usage::

    from aws_orchestrator_agent.core.a2ui import get_catalog_manager

    mgr = get_catalog_manager()

    # Register a custom catalog (components auto-register via @register_component)
    mgr.register_catalog(
        catalog_id="https://example.com/my_catalog.json",
        local_path=Path(__file__).parent / "my_catalog.json",
        aliases=["my-agent:catalog"],
        priority=10,
    )

    # Negotiate with a client
    catalog_uri, schema = mgr.select_catalog(client_capabilities)
"""


import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import constants from the SDK to avoid duplication (DRY)
try:
    from a2ui.core.schema.constants import (
        A2UI_CLIENT_CAPABILITIES_KEY,
        SUPPORTED_CATALOG_IDS_KEY,
        INLINE_CATALOGS_KEY,
    )
except ImportError:
    # Fallback if SDK not on path
    A2UI_CLIENT_CAPABILITIES_KEY = "a2uiClientCapabilities"
    SUPPORTED_CATALOG_IDS_KEY = "supportedCatalogIds"
    INLINE_CATALOGS_KEY = "inlineCatalogs"

from aws_orchestrator_agent.utils.logger import AgentLogger

logger = AgentLogger("A2UICatalogManager")

# ── Well-known constants ──────────────────────────────────────────────
STANDARD_CATALOG_ID = (
    "https://github.com/google/A2UI/blob/main/specification/"
    "v0_8/json/standard_catalog_definition.json"
)


# ── Data class for registered catalogs ────────────────────────────────
@dataclass
class CatalogEntry:
    """A registered catalog with its metadata."""

    catalog_id: str
    """Primary URI identifier (GitHub URL or other canonical ID)."""

    local_path: Optional[Path] = None
    """Path to the local JSON file. ``None`` for the built-in standard catalog."""

    aliases: list[str] = field(default_factory=list)
    """Alternative IDs that also resolve to this catalog (e.g. local shortcuts)."""

    priority: int = 0
    """Higher priority = preferred. Custom > Standard."""

    description: str = ""
    """Human-readable description."""


class CatalogManager:
    """
    Generic, extensible A2UI catalog manager.

    Features:
    - **Dynamic registration**: any agent can register custom catalogs.
    - **Priority-based selection**: clients advertise supported catalog IDs;
      the manager picks the highest-priority match.
    - **Schema merging**: standard + custom catalog definitions are merged
      into a single base schema for validation.
    - **Caching**: loaded catalogs and merged schemas are cached.
    """

    def __init__(self, catalogs_dir: Optional[Path] = None) -> None:
        self._catalogs_dir = catalogs_dir or Path(__file__).parent
        self._entries: Dict[str, CatalogEntry] = {}  # id → entry
        self._alias_map: Dict[str, str] = {}  # alias → canonical id
        self._file_cache: Dict[str, Dict[str, Any]] = {}  # path → parsed json
        self._base_schema: Optional[Dict[str, Any]] = None
        self._merged_cache: Dict[str, Tuple[str, Dict[str, Any]]] = {}  # id → (id, merged)

        # Always register the standard catalog at lowest priority
        self.register_catalog(
            CatalogEntry(
                catalog_id=STANDARD_CATALOG_ID,
                local_path=self._catalogs_dir / "standard_catalog_definition.json",
                aliases=["standard_catalog_definition.json"],
                priority=-1,
                description="A2UI standard catalog (built-in widgets)",
            )
        )

    # ── Registration ──────────────────────────────────────────────────

    def register_catalog(self, entry: CatalogEntry) -> None:
        """
        Register a catalog entry.

        Args:
            entry: The catalog entry to register.
        """
        self._entries[entry.catalog_id] = entry
        for alias in entry.aliases:
            self._alias_map[alias] = entry.catalog_id
        # Invalidate merge cache when a catalog changes
        self._merged_cache.clear()
        logger.debug(f"Registered catalog {entry.catalog_id} (priority={entry.priority})")

    def unregister_catalog(self, catalog_id: str) -> None:
        """Remove a catalog entry by its ID."""
        entry = self._entries.pop(catalog_id, None)
        if entry:
            for alias in entry.aliases:
                self._alias_map.pop(alias, None)
            self._merged_cache.clear()

    # ── Querying ──────────────────────────────────────────────────────

    def get_supported_catalog_ids(self) -> List[str]:
        """Return all catalog IDs + aliases this agent advertises."""
        ids: List[str] = []
        for entry in self._entries.values():
            ids.append(entry.catalog_id)
            ids.extend(entry.aliases)
        return ids

    def list_catalogs(self) -> List[Dict[str, Any]]:
        """Return registered catalogs as dicts (for debugging / introspection)."""
        return [
            {
                "catalog_id": e.catalog_id,
                "aliases": e.aliases,
                "priority": e.priority,
                "local_path": str(e.local_path) if e.local_path else None,
                "description": e.description,
            }
            for e in sorted(self._entries.values(), key=lambda e: -e.priority)
        ]

    def resolve_id(self, raw_id: str) -> Optional[str]:
        """Resolve an alias or canonical ID to its canonical ID."""
        if raw_id in self._entries:
            return raw_id
        return self._alias_map.get(raw_id)

    # ── Schema loading & merging ──────────────────────────────────────

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load and cache a JSON file."""
        cache_key = str(path)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]
        if not path.exists():
            logger.warning(f"Catalog file not found: {path}")
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._file_cache[cache_key] = data
            return data
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def _load_base_schema(self) -> Dict[str, Any]:
        """Load the base A2UI server-to-client schema."""
        if self._base_schema:
            return self._base_schema

        schema_path = self._catalogs_dir / "server_to_client.json"
        loaded = self._load_json(schema_path)
        if loaded:
            self._base_schema = loaded
            return self._base_schema

        # Minimal fallback
        self._base_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "A2UI Server to Client Messages",
            "type": "object",
            "oneOf": [
                {"$ref": "#/definitions/beginRendering"},
                {"$ref": "#/definitions/surfaceUpdate"},
                {"$ref": "#/definitions/dataModelUpdate"},
                {"$ref": "#/definitions/deleteSurface"},
            ],
            "definitions": {},
        }
        return self._base_schema

    def _merge_catalog_into_schema(
        self,
        base: Dict[str, Any],
        catalog: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge catalog definitions/components into a base schema."""
        merged = copy.deepcopy(base)

        sources: List[Dict[str, Any]] = []
        if "definitions" in catalog:
            sources.append(catalog["definitions"])
        if "components" in catalog:
            sources.append(catalog["components"])

        if not sources:
            return merged

        merged.setdefault("definitions", {})
        for source in sources:
            for key, value in source.items():
                merged["definitions"][key] = value

        # Inject into surfaceUpdate component wrapper for valid component types
        try:
            comp_wrapper = (
                merged["properties"]["surfaceUpdate"]["properties"]
                ["components"]["items"]["properties"]["component"]
            )
            comp_wrapper.setdefault("properties", {})
            for key, value in merged["definitions"].items():
                comp_wrapper["properties"][key] = value
        except KeyError:
            pass  # Schema structure doesn't have this path — that's fine

        return merged

    # ── Client negotiation ────────────────────────────────────────────

    def select_catalog(
        self,
        client_capabilities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Negotiate the best catalog based on client capabilities.

        Selection logic (priority order):
        1. If client lists ``supportedCatalogIds``, pick the highest-priority
           registered catalog that the client supports.
        2. If client provides ``inlineCatalogs``, merge them into the base schema.
        3. Fall back to the highest-priority registered catalog.

        Returns:
            ``(catalog_id, merged_schema)``
        """
        if not client_capabilities:
            return self._load_and_merge_best()

        supported_ids = client_capabilities.get(SUPPORTED_CATALOG_IDS_KEY, [])
        inline_catalogs = client_capabilities.get(INLINE_CATALOGS_KEY)

        # Rule: supportedCatalogIds takes precedence over inlineCatalogs
        if supported_ids and inline_catalogs:
            logger.warning(
                "Both supportedCatalogIds and inlineCatalogs provided; "
                "using supportedCatalogIds"
            )
            inline_catalogs = None

        # Match client-supported IDs against registered catalogs by priority
        if supported_ids:
            best = self._best_match(supported_ids)
            if best:
                return self._load_and_merge(best)

        # Inline catalogs
        if inline_catalogs:
            try:
                inline = (
                    json.loads(inline_catalogs)
                    if isinstance(inline_catalogs, str)
                    else inline_catalogs
                )
                base = self._load_base_schema()
                merged = self._merge_catalog_into_schema(base, inline)
                return ("inline", merged)
            except Exception as e:
                logger.error(f"Failed to parse inline catalog: {e}")

        # Fallback
        return self._load_and_merge_best()

    def _best_match(self, client_ids: List[str]) -> Optional[str]:
        """
        Find the highest-priority registered catalog that the client supports.
        """
        matches: List[Tuple[int, str]] = []
        for cid in client_ids:
            canonical = self.resolve_id(cid)
            if canonical and canonical in self._entries:
                matches.append((self._entries[canonical].priority, canonical))

        if not matches:
            return None
        # Highest priority wins
        matches.sort(key=lambda x: -x[0])
        return matches[0][1]

    def _load_and_merge_best(self) -> Tuple[str, Dict[str, Any]]:
        """Load and merge the highest-priority custom catalog."""
        # Pick highest priority non-standard catalog, or standard if none
        candidates = sorted(
            self._entries.values(), key=lambda e: -e.priority
        )
        for c in candidates:
            if c.catalog_id != STANDARD_CATALOG_ID:
                return self._load_and_merge(c.catalog_id)
        return self._load_and_merge(STANDARD_CATALOG_ID)

    def _load_and_merge(self, catalog_id: str) -> Tuple[str, Dict[str, Any]]:
        """Load a catalog by ID and merge with base + standard schemas."""
        if catalog_id in self._merged_cache:
            return self._merged_cache[catalog_id]

        base = copy.deepcopy(self._load_base_schema())

        # Always merge standard catalog first
        std_entry = self._entries.get(STANDARD_CATALOG_ID)
        if std_entry and std_entry.local_path:
            std_data = self._load_json(std_entry.local_path)
            if std_data:
                base = self._merge_catalog_into_schema(base, std_data)

        # If requesting the standard catalog itself, we're done
        if catalog_id == STANDARD_CATALOG_ID:
            result = (catalog_id, base)
            self._merged_cache[catalog_id] = result
            return result

        # Load + merge the requested custom catalog
        canonical = self.resolve_id(catalog_id) or catalog_id
        entry = self._entries.get(canonical)
        if entry and entry.local_path:
            data = self._load_json(entry.local_path)
            if data:
                merged = self._merge_catalog_into_schema(base, data)
                result = (catalog_id, merged)
                self._merged_cache[catalog_id] = result
                return result

        # Fallback to base
        logger.warning(f"Could not load catalog {catalog_id}, using base schema")
        result = (catalog_id, base)
        self._merged_cache[catalog_id] = result
        return result

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def validate_a2ui_message(
        message: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that an A2UI message has the expected structure.

        This is a lightweight structural check, not full JSON-schema
        validation.
        """
        valid_types = {
            "beginRendering", "surfaceUpdate", "dataModelUpdate", "deleteSurface"
        }
        if not any(k in message for k in valid_types):
            return False, f"Message must contain one of: {sorted(valid_types)}"

        if "surfaceUpdate" in message:
            su = message["surfaceUpdate"]
            for comp in su.get("components", []):
                if "component" not in comp:
                    return False, (
                        f"Component {comp.get('id', '?')} missing 'component'"
                    )
                if not isinstance(comp["component"], dict) or len(comp["component"]) != 1:
                    return False, (
                        f"Component {comp.get('id', '?')} must have exactly "
                        "one component type"
                    )

        return True, None

    # ── SDK integration ───────────────────────────────────────────────

    def get_agent_extension(self, version: str = "0.8"):
        """
        Build an ``AgentExtension`` for the agent card using the SDK.

        Returns:
            An ``AgentExtension`` object, or a dict fallback if the SDK
            is not available.
        """
        try:
            from a2ui.a2a import get_a2ui_agent_extension
            return get_a2ui_agent_extension(
                version=version,
                accepts_inline_catalogs=True,
                supported_catalog_ids=[
                    e.catalog_id
                    for e in self._entries.values()
                ],
            )
        except ImportError:
            # Fallback for environments where SDK is not on the path
            return {
                "uri": f"https://a2ui.org/a2a-extension/a2ui/v{version}",
                "description": "Provides agent driven UI using the A2UI JSON format.",
                "params": {
                    "acceptsInlineCatalogs": True,
                    "supportedCatalogIds": [
                        e.catalog_id for e in self._entries.values()
                    ],
                },
            }


# ── Global singleton ──────────────────────────────────────────────────

_catalog_manager: Optional[CatalogManager] = None


def get_catalog_manager() -> CatalogManager:
    """Get or create the global ``CatalogManager`` instance."""
    global _catalog_manager
    if _catalog_manager is None:
        _catalog_manager = CatalogManager()
    return _catalog_manager
