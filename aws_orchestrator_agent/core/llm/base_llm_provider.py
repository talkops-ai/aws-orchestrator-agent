from abc import ABC, abstractmethod
from typing import Any, Optional
from langchain_core.runnables import Runnable

class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Defines the interface for creating LangChain-compatible LLM instances.
    """
    @abstractmethod
    def create_llm(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs: Any
    ) -> Runnable:
        """
        Create a LangChain LLM instance that implements the Runnable interface.
        Args:
            model: Model name (e.g., 'gpt-4')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
        Returns:
            Configured LangChain LLM instance (Runnable)
        """
        pass 