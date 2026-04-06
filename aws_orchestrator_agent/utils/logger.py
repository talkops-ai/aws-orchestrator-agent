"""
Logging module for AWS Orchestrator Agent.

Provides color-coded, multi-sink (console + file + websocket) logging with
both structured (JSON) and human-readable output modes.

Configuration is driven by ``Config`` (from ``default.py``).  Every key is
resolved lazily so the logger never triggers a circular import.

Usage::

    from aws_orchestrator_agent.utils.logger import AgentLogger

    log = AgentLogger("tf-generator")
    log.info("Module created", extra={"module": "vpc", "files": 5})
    log.warning("Drift detected", task_id="deploy-42")
    log.error("Validation failed", extra={"exit_code": 1})
"""


import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional

from colorama import Fore, Style


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

class _LevelColor(Enum):
    """Traffic-light colors for log levels."""
    DEBUG = Fore.LIGHTBLACK_EX
    INFO = Fore.BLUE
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    CRITICAL = Fore.LIGHTRED_EX


# ---------------------------------------------------------------------------
# Singleton config accessor (lazy — avoids circular imports)
# ---------------------------------------------------------------------------

_config_cache: Optional[Any] = ...  # sentinel: `...` means "not loaded yet"


def _get_config() -> Any:
    """Lazy-load Config once.  Returns the Config instance or None."""
    global _config_cache
    if _config_cache is ...:
        try:
            from aws_orchestrator_agent.config import Config
            _config_cache = Config()
        except Exception:
            _config_cache = None
    return _config_cache


def _cfg(key: str, fallback: Any) -> Any:
    """Read a single config value, falling back if Config isn't available."""
    cfg = _get_config()
    if cfg is None:
        return fallback
    return getattr(cfg, key, fallback)


# ---------------------------------------------------------------------------
# Colored console formatter (for StreamHandler)
# ---------------------------------------------------------------------------

class _ColorFormatter(logging.Formatter):
    """
    Applies per-level color to log records for console output.

    Reads ``LOG_DATE_FORMAT`` from config for timestamp formatting.
    Keeps ANSI codes OUT of the ``logging.FileHandler`` path automatically
    because only the ``StreamHandler`` uses this formatter.
    """

    def __init__(self) -> None:
        super().__init__()
        self._date_fmt: str = _cfg("LOG_DATE_FORMAT", "%Y-%m-%dT%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        agent = getattr(record, "agent_name", "BASE")

        # Only level gets color — agent name stays neutral
        try:
            lc = _LevelColor[level].value
        except KeyError:
            lc = Fore.WHITE

        ts = datetime.now(timezone.utc).strftime(self._date_fmt)
        msg = record.getMessage()

        extras = getattr(record, "_structured_extra", None)
        if extras:
            extra_pieces = [f"{k}={v}" for k, v in extras.items()]
            extra_str = " | " + " ".join(extra_pieces)
        else:
            extra_str = ""

        return (
            f"{lc}[{level}]{Style.RESET_ALL} "
            f"{agent}: "
            f"[{ts}] {msg}{extra_str}"
        )


# ---------------------------------------------------------------------------
# Structured (JSON) formatter (for both file and console when enabled)
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emits each log record as a single JSON line (structured logging)."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "agent": getattr(record, "agent_name", None),
            "message": record.getMessage(),
        }
        # Merge structured extras (task_id, context_id, custom fields)
        extras = getattr(record, "_structured_extra", None)
        if extras:
            entry.update(extras)
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# File formatter (plain text, no color codes)
# ---------------------------------------------------------------------------

class _PlainFormatter(logging.Formatter):
    """
    Plain-text format for file output — no ANSI escape codes.

    Respects ``LOG_FORMAT`` and ``LOG_DATE_FORMAT`` from config.
    """

    def __init__(self) -> None:
        log_fmt: str = _cfg(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        date_fmt: str = _cfg("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")
        super().__init__(fmt=log_fmt, datefmt=date_fmt)

    def format(self, record: logging.LogRecord) -> str:
        # Inject agent_name into the record so %(name)s shows it
        record.name = getattr(record, "agent_name", record.name)
        msg = super().format(record)
        
        extras = getattr(record, "_structured_extra", None)
        if extras:
            extra_pieces = [f"{k}={v}" for k, v in extras.items()]
            msg += " | " + " ".join(extra_pieces)
            
        return msg


# ---------------------------------------------------------------------------
# Handler factory (creates handlers once per logger name)
# ---------------------------------------------------------------------------

_initialised_loggers: set[str] = set()


def _ensure_handlers(py_logger: logging.Logger, agent_name: str) -> None:
    """
    Attach console + file handlers exactly once per logger name.

    Prevents the duplicate-handler bug that occurred when multiple
    ``AgentLogger("X")`` instances were created.
    """
    if py_logger.name in _initialised_loggers:
        return
    _initialised_loggers.add(py_logger.name)

    py_logger.handlers.clear()
    py_logger.propagate = False

    level_name: str = _cfg("LOG_LEVEL", "INFO")
    py_logger.setLevel(getattr(logging, level_name, logging.INFO))

    structured: bool = _cfg("LOG_STRUCTURED_JSON", False)

    # ── Console handler ──
    if _cfg("LOG_TO_CONSOLE", True):
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(py_logger.level)
        console.setFormatter(_JsonFormatter() if structured else _ColorFormatter())
        py_logger.addHandler(console)

    # ── File handler ──
    if _cfg("LOG_TO_FILE", True):
        log_file: str = _cfg("LOG_FILE", "aws_orchestrator.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(py_logger.level)
        fh.setFormatter(_JsonFormatter() if structured else _PlainFormatter())
        py_logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AgentLogger:
    """
    Per-agent logger with color-coded console output, structured JSON support,
    file logging, and optional websocket streaming.

    Usage::

        log = AgentLogger("tf-validator")
        log.info("Starting validation")
        log.error("terraform validate failed", extra={"exit_code": 1})
        log.warning("Drift detected", task_id="deploy-42", context_id="sess-abc")
    """

    __slots__ = ("agent_name", "_logger", "_websocket", "_stream_fn")

    def __init__(self, agent_name: str = "BASE") -> None:
        self.agent_name = agent_name
        self._logger = logging.getLogger(f"agent.{agent_name}")
        self._websocket: Any = None
        self._stream_fn: Optional[Callable[..., Any]] = None

        _ensure_handlers(self._logger, agent_name)

    # ── Convenience level methods (sync — suitable for most call-sites) ──

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, msg, **kwargs)

    # ── Core emit (unified path for ALL log output) ──────────────────────

    def _emit(
        self,
        level: int,
        message: str,
        *,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Single code-path that feeds console, file, and websocket."""
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        # Attach agent identity + structured extras to the record
        record.agent_name = self.agent_name  # type: ignore[attr-defined]
        structured_extra: dict[str, Any] = {}
        if task_id:
            structured_extra["task_id"] = task_id
        if context_id:
            structured_extra["context_id"] = context_id
        if extra:
            structured_extra.update(extra)
        record._structured_extra = structured_extra  # type: ignore[attr-defined]

        self._logger.handle(record)

    # ── Backward-compat: log_structured() ────────────────────────────────

    def log_structured(
        self,
        level: str = "INFO",
        message: str = "",
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Backward-compatible structured log call.

        Prefer ``log.info(...)``, ``log.error(...)`` etc. for new code.
        """
        py_level = getattr(logging, level.upper(), logging.INFO)
        self._emit(py_level, message, task_id=task_id, context_id=context_id, extra=extra)

    # ── Async log (for websocket streaming) ──────────────────────────────

    async def alog(
        self,
        message: str,
        level: str = "INFO",
        **kwargs: Any,
    ) -> None:
        """Async variant that also pushes to websocket if configured."""
        py_level = getattr(logging, level.upper(), logging.INFO)
        self._emit(py_level, message, **kwargs)

        # Websocket push (fire-and-forget)
        if self._websocket and self._stream_fn:
            try:
                await self._stream_fn("logs", self.agent_name, message, self._websocket)
            except Exception:
                pass  # never let WS failure crash the agent

    def set_websocket(self, websocket: Any, stream_fn: Callable[..., Any]) -> None:
        """Attach a websocket sink for live streaming."""
        self._websocket = websocket
        self._stream_fn = stream_fn

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def create(cls, agent_name: str) -> "AgentLogger":
        """Factory method (alias for constructor)."""
        return cls(agent_name)

    # Keep old name for backward compat
    create_logger = create

    def __repr__(self) -> str:
        return f"AgentLogger({self.agent_name!r})"