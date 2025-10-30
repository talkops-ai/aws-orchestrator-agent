"""Custom exceptions for the AWS Orchestrator Agent."""

from typing import Optional, Any


class AWSOrchestratorAgentError(Exception):
    """Base exception for all AWS Orchestrator Agent errors."""
    pass

class ValidationError(AWSOrchestratorAgentError):
    """Raised for validation errors in messages or data."""
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value

class MessageFormatError(ValidationError):
    """Raised for invalid message format or structure."""
    def __init__(self, message: str, expected_format: str, received_format: str) -> None:
        super().__init__(message)
        self.expected_format = expected_format
        self.received_format = received_format

class JsonRpcValidationError(ValidationError):
    """Raised for JSON-RPC 2.0 specification violations."""
    def __init__(self, message: str, rpc_error_code: int, field: Optional[str] = None) -> None:
        super().__init__(message, field)
        self.rpc_error_code = rpc_error_code

class A2AProtocolError(ValidationError):
    """Raised for A2A protocol-specific violations."""
    def __init__(self, message: str, protocol_version: str, agent_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.protocol_version = protocol_version
        self.agent_id = agent_id

class LLMConfigurationError(AWSOrchestratorAgentError):
    """Raised when LLM configuration is invalid."""
    pass

class UnsupportedProviderError(AWSOrchestratorAgentError):
    """Raised when an unsupported LLM provider is requested."""
    pass

class ConfigError(AWSOrchestratorAgentError):
    """Raised for configuration-related errors."""
    pass