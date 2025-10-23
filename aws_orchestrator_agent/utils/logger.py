# Copyright (C) 2025 StructBinary
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from colorama import Fore, Style
from enum import Enum
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from functools import wraps
import functools
import inspect
import json

# Try to import config, fallback to defaults if not available
try:
    from aws_orchestrator_agent.config.config import Config
    config = Config()
except Exception:
    config = None

# Configure logging
logger = logging.getLogger(__name__)

class AgentColor(Enum):
    # Planner/Orchestration Components (Green Family)
    PLANNER_SUPERVISOR = Fore.GREEN
    PLANNER_SUPERVISOR_STATE = Fore.LIGHTGREEN_EX
    PLANNER_HANDOFF_TOOLS = Fore.LIGHTGREEN_EX
    EXECUTION_PLANNER_REACT = Fore.LIGHTGREEN_EX
    REQUIREMENTS_ANALYZER_REACT = Fore.LIGHTGREEN_EX
    PLANNER_AGENT = Fore.GREEN
    MODIFICATION_PLANNER = Fore.LIGHTGREEN_EX
    
    # Generator/Processing Components (Yellow/Magenta Family)
    GENERATOR_SWARM = Fore.YELLOW
    GENERATOR_STAGE_HANDOFF_MANAGER = Fore.LIGHTYELLOW_EX
    GENERATOR_HANDOFF = Fore.MAGENTA
    BACKEND_GENERATOR = Fore.CYAN
    DATA_GENERATOR = Fore.LIGHTMAGENTA_EX
    LOCAL_GENERATOR = Fore.LIGHTYELLOW_EX
    OUTPUT_GENERATOR = Fore.LIGHTMAGENTA_EX
    README_GENERATOR = Fore.LIGHTMAGENTA_EX
    VARIABLE_GENERATOR = Fore.LIGHTMAGENTA_EX
    RESOURCE_GENERATOR = Fore.LIGHTMAGENTA_EX
    
    # Infrastructure/Server Components (Blue Family)
    AWS_ORCHESTRATOR_SERVER = Fore.LIGHTBLUE_EX
    A2A_EXECUTOR = Fore.BLUE
    SUPERVISOR = Fore.LIGHTBLUE_EX
    SUPERVISOR_HANDOFF = Fore.LIGHTBLUE_EX
    # Writer/Output Components (Magenta Family)
    WRITER_REACT_AGENT = Fore.MAGENTA
    SUPERVISOR_ADAPTER = Fore.LIGHTMAGENTA_EX
    GENERIC_AGENT_EXECUTOR = Fore.LIGHTMAGENTA_EX
    # Base/Default
    BASE = Fore.WHITE

class LogLevelColor(Enum):
    """Enterprise-standard log level colors following traffic light semantics."""
    DEBUG = Fore.LIGHTBLACK_EX      # Gray - Low priority
    INFO = Fore.BLUE                # Blue - Normal operation
    WARNING = Fore.YELLOW           # Yellow - Potential issues
    ERROR = Fore.RED               # Red - Operation failed
    CRITICAL = Fore.LIGHTRED_EX    # Bright Red - System failure

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AgentLogger:
    """Logger class for agent output with color encoding and multiple output methods."""
    
    def __init__(self, agent_name: str = "BASE", log_to_console: Optional[bool] = None, log_to_file: Optional[bool] = None, log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
        self.agent_name = agent_name
        self.websocket = None
        self.stream_output = None
        # Determine config values
        if config:
            self.log_to_console = log_to_console if log_to_console is not None else getattr(config, 'LOG_TO_CONSOLE', True)
            self.log_to_file = log_to_file if log_to_file is not None else getattr(config, 'LOG_TO_FILE', True)
            self.log_level = log_level if log_level is not None else getattr(config, 'LOG_LEVEL', 'INFO')
            self.log_file = log_file if log_file is not None else getattr(config, 'LOG_FILE', 'planner_agent.log')
        else:
            self.log_to_console = log_to_console if log_to_console is not None else True
            self.log_to_file = log_to_file if log_to_file is not None else True
            self.log_level = log_level if log_level is not None else 'INFO'
            self.log_file = log_file if log_file is not None else 'planner_agent.log'
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        self.logger.setLevel(self.log_level)
        # Remove all handlers to avoid duplicate logs
        self.logger.handlers = []
        # Always use only the message, no preamble
        formatter = logging.Formatter('%(message)s')
        if self.log_to_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        # Always add a stream handler for console, but only emit if log_to_console is True
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.log_level)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
    def set_websocket(self, websocket: Any, stream_output: Callable) -> None:
        """Set websocket and stream output function."""
        self.websocket = websocket
        self.stream_output = stream_output
        
    def _get_color(self, agent: str) -> str:
        """Get color for agent."""
        try:
            return AgentColor[agent].value
        except KeyError:
            return AgentColor.BASE.value
    
    def _get_level_color(self, level: str) -> str:
        """Get color for log level."""
        try:
            return LogLevelColor[level].value
        except KeyError:
            return Fore.WHITE
            
    def _format_log_entry(self, message: str, level: str = "INFO") -> Dict[str, Any]:
        """Format log entry."""
        return {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "message": message,
            "level": level
        }
        
    def _log_to_console(self, message: str, level: str = "INFO") -> None:
        """Log to console with color, if enabled."""
        if self.log_to_console:
            agent_color = self._get_color(self.agent_name)
            level_color = self._get_level_color(level)
            # Format: [LEVEL] AGENT: [timestamp] message (clean format with timestamp)
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"
            formatted_message = f"{level_color}[{level}]{Style.RESET_ALL} {agent_color}{self.agent_name}{Style.RESET_ALL}: [{timestamp}] {message}"
            print(formatted_message)
        
    def _log_to_file(self, message: str, level: str = "INFO") -> None:
        """Log to file, if enabled."""
        if self.log_to_file:
            log_method = getattr(self.logger, level.lower())
            log_method(message)
        
    async def _log_to_websocket(self, message: str) -> None:
        """Log to websocket."""
        if self.websocket and self.stream_output:
            await self.stream_output("logs", self.agent_name, message, self.websocket)
            
    async def log(self, message: str, level: str = "INFO") -> None:
        """Log message to all configured outputs."""
        # Log to console
        self._log_to_console(message, level)
        
        # Log to file
        self._log_to_file(message, level)
        
        # Log to websocket if available
        await self._log_to_websocket(message)
        
    @staticmethod
    def format_log_message(message: str, level: str = "INFO") -> str:
        """Format a log message."""
        return f"[{level}] {message}"
        
    @classmethod
    def create_logger(cls, agent_name: str) -> 'AgentLogger':
        """Create a new logger instance."""
        return cls(agent_name)

    def log_structured(
        self,
        level: str = "INFO",
        message: str = "",
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Log a structured message with context fields.
        If LOG_STRUCTURED_JSON is True, outputs JSON; otherwise, outputs a formatted string.
        All output is routed through the logger's console and file handlers, respecting config.
        Args:
            level: Log level (e.g., "INFO", "ERROR")
            message: Log message
            task_id: Optional task ID
            context_id: Optional context/session ID
            extra: Optional dict of extra fields (e.g., agent_name, etc.)
        """
        structured = False
        if config:
            structured = getattr(config, 'LOG_STRUCTURED_JSON', False)
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_name": getattr(self, 'agent_name', None),
            "log_type": level,
            "message": message,
            "task_id": task_id,
            "context_id": context_id
        }
        if extra:
            log_entry.update(extra)
        if structured:
            msg = json.dumps(log_entry)
            # Only log to console, avoid file logging to prevent duplication
            self._log_to_console(msg, level=level)
        else:
            # Compose a readable string with all fields
            parts = [
                message,
                f"task_id={task_id}" if task_id else "",
                f"context_id={context_id}" if context_id else ""
            ]
            # Add any extra fields
            if extra:
                for k, v in extra.items():
                    if k not in ("agent_name",):
                        parts.append(f"{k}={v}")
            std_msg = " ".join([p for p in parts if p])
            # Only log to console, avoid file logging to prevent duplication
            self._log_to_console(std_msg, level=level)

# Global AgentLogger for decorator logs (must be after class definition)
_decorator_logger = AgentLogger("DECORATOR")

def log_sync(func: Callable) -> Callable:
    """No-op decorator - structured logging is used instead."""
    return func

def log_async(func: Callable) -> Callable:
    """No-op decorator - structured logging is used instead."""
    return func