"""
AWS Orchestrator Agent — Configuration Engine.

This module is the INTERNAL implementation. Users should only edit ``default.py``
to change default values. Overrides are applied automatically via environment
variables and runtime ``config`` dicts.

Precedence (highest → lowest):
    1. Runtime overrides  (``Config({"LLM_PROVIDER": "anthropic"})``)
    2. Environment variables / ``.env`` file
    3. Defaults from ``DefaultConfig`` in ``default.py``
"""


import json
import os
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from dotenv import load_dotenv

from aws_orchestrator_agent.config.default import DefaultConfig
from aws_orchestrator_agent.utils.exceptions import ConfigError

# Load environment variables once at module level
load_dotenv()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convert_env_value(key: str, env_value: str, type_hint: Type) -> Any:
    """Coerce a raw env-var string to the type declared in ``DefaultConfig``."""
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle Optional[X] / Union[X, None]
    if origin is Union:
        for arg in args:
            if arg is type(None):
                if env_value.lower() in ("none", "null", ""):
                    return None
            else:
                try:
                    return _convert_env_value(key, env_value, arg)
                except Exception:
                    continue
        raise ConfigError(f"Cannot convert env var '{key}'={env_value!r} to {args}")

    if type_hint is bool:
        return env_value.lower() in ("true", "1", "yes", "on")
    if type_hint is int:
        return int(env_value)
    if type_hint is float:
        return float(env_value)
    if type_hint in (str, Any):
        return env_value
    if origin is list or type_hint is list:
        return json.loads(env_value)
    if origin is dict or type_hint is dict:
        return json.loads(env_value)

    raise ConfigError(f"Unsupported type {type_hint} for config key '{key}'")


def _collect_defaults() -> tuple[dict[str, Any], dict[str, Type]]:
    """
    Read all user-declared defaults + type annotations from ``DefaultConfig``.

    Returns ``(defaults_dict, annotations_dict)``.
    """
    annotations: dict[str, Type] = {}
    # Walk the MRO so subclasses of DefaultConfig also work
    for cls in reversed(DefaultConfig.__mro__):
        annotations.update(getattr(cls, "__annotations__", {}))

    defaults = {
        key: getattr(DefaultConfig, key)
        for key in annotations
        if hasattr(DefaultConfig, key)
    }
    return defaults, annotations


def _build_llm_kwargs(
    store: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """
    Build a ``langchain.chat_models.init_chat_model()``-compatible kwargs dict.

    ``prefix`` is one of ``"LLM_"``, ``"LLM_HIGHER_"``, ``"LLM_DEEPAGENT_"`` etc.
    Eliminates the 3× copy-paste that existed before.
    """
    provider: str = store.get(f"{prefix}PROVIDER", "openai")
    model: str = store.get(f"{prefix}MODEL", "gpt-4o-mini")

    kwargs: dict[str, Any] = {
        "temperature": store.get(f"{prefix}TEMPERATURE", 0.0),
        "max_tokens": store.get(f"{prefix}MAX_TOKENS", 15000),
    }

    # Provider-specific model string for init_chat_model
    if provider == "azure_openai":
        kwargs["model"] = f"azure_openai:{model}"
        deployment = store.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        if deployment:
            kwargs["azure_deployment"] = deployment
    elif provider in ("google_genai", "gemini"):
        kwargs["model"] = f"google_genai:{model}"
    elif provider in ("bedrock", "aws_bedrock"):
        kwargs["model"] = model
        kwargs["model_provider"] = "bedrock_converse"
    else:
        kwargs["model"] = model
        if provider:
            kwargs["model_provider"] = provider

    # Backward-compat key (safe to remove once all callers use the property)
    kwargs["provider"] = provider
    return kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Config:
    """
    Resolved configuration for the AWS Orchestrator Agent.

    **Users should never edit this file.**  Change defaults in ``default.py``,
    override at runtime via env-vars or the ``config`` dict.

    Precedence (highest → lowest):
        1. ``config`` dict passed to ``__init__``
        2. Environment variables (``.env`` or system)
        3. ``DefaultConfig`` values in ``default.py``

    Access style::

        cfg = Config()
        cfg.LLM_PROVIDER        # → "openai"  (canonical UPPER key)
        cfg["LLM_PROVIDER"]     # → same, dict-style
    """

    # ── construction ──────────────────────────────────────────────────────

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        overrides = config or {}
        defaults, annotations = _collect_defaults()

        store: dict[str, Any] = {}

        # Layer 1 – defaults  →  Layer 2 – env-vars
        for key, default_value in defaults.items():
            env_value = os.getenv(key)
            if env_value is not None and env_value.strip():
                type_hint = annotations.get(key, type(default_value))
                try:
                    store[key] = _convert_env_value(key, env_value, type_hint)
                except Exception as exc:
                    raise ConfigError(
                        f"Failed to convert env var '{key}'={env_value!r} "
                        f"to {type_hint}: {exc}"
                    ) from exc
            else:
                store[key] = default_value

        # Layer 3 – runtime overrides (highest priority)
        store.update(overrides)

        # Freeze the internal dict
        self._store: dict[str, Any] = store

    # ── attribute access (single path, no divergence) ─────────────────────

    def __getattr__(self, name: str) -> Any:
        # Allow both UPPER and lower lookups:  cfg.LLM_PROVIDER  /  cfg.llm_provider
        store = self.__dict__.get("_store")
        if store is None:
            raise AttributeError(name)
        if name in store:
            return store[name]
        upper = name.upper()
        if upper in store:
            return store[upper]
        raise AttributeError(f"Config has no key '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __contains__(self, key: str) -> bool:
        return key in self._store or key.upper() in self._store

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``.get()`` with fallback."""
        try:
            return self[key]
        except KeyError:
            return default

    # ── LLM config properties (DRY — single builder) ─────────────────────

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Standard LLM config kwargs for ``init_chat_model()``."""
        return _build_llm_kwargs(self._store, "LLM_")

    @property
    def llm_higher_config(self) -> Dict[str, Any]:
        """Higher-tier LLM config kwargs for ``init_chat_model()``."""
        return _build_llm_kwargs(self._store, "LLM_HIGHER_")

    @property
    def llm_deepagent_config(self) -> Dict[str, Any]:
        """DeepAgent LLM config kwargs for ``init_chat_model()``."""
        return _build_llm_kwargs(self._store, "LLM_DEEPAGENT_")

    # Convenience aliases (some call-sites use method style)
    def get_llm_config(self) -> Dict[str, Any]:
        return self.llm_config

    def get_llm_higher_config(self) -> Dict[str, Any]:
        return self.llm_higher_config

    def get_llm_deepagent_config(self) -> Dict[str, Any]:
        return self.llm_deepagent_config

    # ── MCP config ────────────────────────────────────────────────────────

    @property
    def mcp_config(self) -> Dict[str, Any]:
        """Return MCP server configuration for ``MCPClient``."""
        raw = self._store.get("MCP_SERVERS", [])

        if isinstance(raw, str):
            try:
                servers = json.loads(raw) if raw else []
            except (json.JSONDecodeError, TypeError):
                servers = []
        else:
            servers = list(raw)  # defensive copy

        return {
            "servers": servers,
            "timeout": {
                "total": self._store.get("MCP_TIMEOUT_TOTAL", 600.0),
                "connect": self._store.get("MCP_TIMEOUT_CONNECT", 300.0),
            },
            "default_host": self._store.get("MCP_DEFAULT_HOST", "localhost"),
            "default_transport": self._store.get("MCP_DEFAULT_TRANSPORT", "sse"),
        }

    def get_mcp_config(self) -> Dict[str, Any]:
        return self.mcp_config

    # ── mutators (update store directly — no stale attrs) ─────────────────

    def set(self, key: str, value: Any) -> None:
        """Set a config key at runtime (highest priority)."""
        self._store[key] = value

    def set_llm_config(self, values: Dict[str, Any]) -> None:
        """Convenience: set standard LLM fields from a dict."""
        _KEY_MAP = {
            "provider": "LLM_PROVIDER",
            "model": "LLM_MODEL",
            "temperature": "LLM_TEMPERATURE",
            "max_tokens": "LLM_MAX_TOKENS",
        }
        for k, v in values.items():
            store_key = _KEY_MAP.get(k)
            if store_key:
                self._store[store_key] = v

    def set_llm_higher_config(self, values: Dict[str, Any]) -> None:
        """Convenience: set higher-tier LLM fields from a dict."""
        _KEY_MAP = {
            "provider": "LLM_HIGHER_PROVIDER",
            "model": "LLM_HIGHER_MODEL",
            "temperature": "LLM_HIGHER_TEMPERATURE",
            "max_tokens": "LLM_HIGHER_MAX_TOKENS",
        }
        for k, v in values.items():
            store_key = _KEY_MAP.get(k)
            if store_key:
                self._store[store_key] = v

    def set_mcp_servers(self, servers: List[Dict[str, Any]]) -> None:
        """Replace MCP server list."""
        self._store["MCP_SERVERS"] = servers

    def add_mcp_server(
        self,
        name: str,
        host: str,
        port: int,
        transport: str = "sse",
        disabled: bool = False,
    ) -> None:
        """Append an MCP server definition."""
        current = self.mcp_config["servers"]
        current.append(
            {
                "name": name,
                "host": host,
                "port": port,
                "transport": transport,
                "disabled": disabled,
            }
        )
        self.set_mcp_servers(current)

    # ── serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of all resolved config values."""
        return dict(self._store)

    @classmethod
    def load_config(cls, config_path: str) -> "Config":
        """
        Load configuration from a JSON file, merged with defaults.

        Returns a ``Config`` instance (not a raw dict) so precedence
        rules are always enforced.
        """
        if not os.path.exists(config_path):
            raise ConfigError(
                f"Configuration file not found: '{config_path}'"
            )

        with open(config_path, "r") as fh:
            custom = json.load(fh)

        return cls(config=custom)

    def __repr__(self) -> str:
        keys = sorted(self._store)
        return f"Config({', '.join(f'{k}=...' for k in keys[:5])}, ...)"