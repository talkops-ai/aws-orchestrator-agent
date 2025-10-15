import os
from typing import Optional, Dict, Any, Callable
from langchain_core.runnables import Runnable
from aws_orchestrator_agent.utils.exceptions import UnsupportedProviderError, LLMConfigurationError
from .base_llm_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    Concrete LLM provider for OpenAI models.
    Implements the BaseLLMProvider interface.
    """
    def create_llm(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs: Any
    ) -> Runnable:
        LLMProvider._check_package("langchain_openai", "OpenAI")
        from langchain_openai import ChatOpenAI
        api_key = kwargs.pop('api_key', None) or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise LLMConfigurationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        config = {
            "model": model,
            "temperature": temperature,
            "openai_api_key": api_key,
        }
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if kwargs.get('base_url'):
            config["openai_api_base"] = kwargs.pop('base_url')
        if kwargs.get('organization'):
            config["openai_organization"] = kwargs.pop('organization')
        config.update(kwargs)
        return ChatOpenAI(**config)


class AnthropicProvider(BaseLLMProvider):
    """
    Concrete LLM provider for Anthropic models.
    Implements the BaseLLMProvider interface.
    """
    def create_llm(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs: Any
    ) -> Runnable:
        LLMProvider._check_package("langchain_anthropic", "Anthropic")
        from langchain_anthropic import ChatAnthropic  # type: ignore
        api_key = kwargs.pop('api_key', None) or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise LLMConfigurationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        config = {
            "model": model,
            "temperature": temperature,
            "anthropic_api_key": api_key,
        }
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        config.update(kwargs)
        return ChatAnthropic(**config)


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Concrete LLM provider for Azure OpenAI models.
    Implements the BaseLLMProvider interface.
    """
    def create_llm(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs: Any
    ) -> Runnable:
        LLMProvider._check_package("langchain_openai", "Azure OpenAI")
        from langchain_openai import ChatOpenAI
        api_key = kwargs.pop('api_key', None) or os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = kwargs.pop('endpoint', None) or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not api_key or not endpoint:
            raise LLMConfigurationError(
                "Azure OpenAI API key or endpoint not found. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                "environment variables or pass api_key and endpoint parameters."
            )
        config = {
            "model": model,
            "temperature": temperature,
            "openai_api_key": api_key,
            "openai_api_base": endpoint,
        }
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if kwargs.get('api_version'):
            config["openai_api_version"] = kwargs.pop('api_version')
        if kwargs.get('deployment_name'):
            config["azure_deployment"] = kwargs.pop('deployment_name')
        config.update(kwargs)
        return ChatOpenAI(**config)


class LLMProvider:
    """Factory for creating LangChain-compatible LLM instances that always return Runnables."""
    
    # Supported providers registry
    _SUPPORTED_PROVIDERS = {
        "openai", 
        "anthropic", 
        "azure_openai"
    }
    
    @staticmethod
    def create_llm(
        provider: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs: Any
    ) -> Runnable:
        """
        Create a LangChain LLM instance that implements the Runnable interface.
        
        Args:
            provider: LLM provider name ('openai', 'anthropic', 'azure_openai')
            model: Model name (e.g., 'gpt-4', 'claude-3-sonnet-20240229')
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Configured LangChain LLM instance (guaranteed to be a Runnable)
            
        Raises:
            UnsupportedProviderError: If provider is not supported
            LLMConfigurationError: If configuration is invalid
        """
        provider = provider.lower().strip()
        
        if provider not in LLMProvider._SUPPORTED_PROVIDERS:
            supported = ", ".join(LLMProvider._SUPPORTED_PROVIDERS)
            raise UnsupportedProviderError(
                f"Unsupported provider: '{provider}'. "
                f"Supported providers: {supported}"
            )
        
        try:
            return LLMProvider._create_provider_instance(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
        except LLMConfigurationError as e:
            raise e
        except Exception as e:
            raise LLMConfigurationError(
                f"Failed to create LLM for provider '{provider}': {e}"
            )
    
    @staticmethod
    def _check_package(package_name: str, provider_name: str) -> None:
        """Check if required package is installed."""
        try:
            __import__(package_name)
        except ImportError:
            raise LLMConfigurationError(
                f"{package_name} package is required for {provider_name} provider. "
                f"Install with: pip install {package_name}"
            )
    
    @staticmethod
    def _create_provider_instance(
        provider: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: int,
        **kwargs: Any
    ) -> Runnable:
        """
        Create the actual provider instance using the factory pattern.
        All LangChain chat models are Runnables by default.
        """
        if provider == "openai":
            return OpenAIProvider().create_llm(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
        elif provider == "anthropic":
            return AnthropicProvider().create_llm(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
        elif provider == "azure_openai":
            return AzureOpenAIProvider().create_llm(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
        else:
            # This should never happen due to validation above
            raise UnsupportedProviderError(f"Provider '{provider}' not implemented")
    
    @staticmethod
    def get_supported_providers() -> Dict[str, Dict[str, str]]:
        """Get information about supported providers and their requirements."""
        return {
            "openai": {
                "description": "OpenAI GPT models",
                "required_env": "OPENAI_API_KEY",
                "package": "langchain-openai",
                "example_models": "gpt-4, gpt-4-turbo, gpt-3.5-turbo"
            },
            "anthropic": {
                "description": "Anthropic Claude models", 
                "required_env": "ANTHROPIC_API_KEY",
                "package": "langchain-anthropic",
                "example_models": "claude-3-opus-20240229, claude-3-sonnet-20240229"
            },
            "azure_openai": {
                "description": "Azure OpenAI Service",
                "required_env": "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT",
                "package": "langchain-openai", 
                "example_models": "gpt-4, gpt-35-turbo (deployment names)"
            }
        }
    
    @staticmethod
    def validate_environment(provider: str) -> Dict[str, bool]:
        """
        Validate that required environment variables are set for a provider.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            Dictionary with validation results
        """
        provider = provider.lower().strip()
        validation = {"valid": True, "missing": []}
        
        if provider == "openai":
            if not os.getenv('OPENAI_API_KEY'):
                validation["valid"] = False
                validation["missing"].append("OPENAI_API_KEY")
        
        elif provider == "anthropic":
            if not os.getenv('ANTHROPIC_API_KEY'):
                validation["valid"] = False
                validation["missing"].append("ANTHROPIC_API_KEY")
        
        elif provider == "azure_openai":
            if not os.getenv('AZURE_OPENAI_API_KEY'):
                validation["valid"] = False
                validation["missing"].append("AZURE_OPENAI_API_KEY")
            if not os.getenv('AZURE_OPENAI_ENDPOINT'):
                validation["valid"] = False
                validation["missing"].append("AZURE_OPENAI_ENDPOINT")
        
        else:
            validation["valid"] = False
            validation["missing"] = [f"Unsupported provider: {provider}"]
        
        return validation


# Convenience function for quick LLM creation
def create_llm_from_env(
    provider: str = "openai", 
    model: str = "gpt-4",
    **kwargs: Any
) -> Runnable:
    """
    Convenience function to create an LLM using environment variables.
    
    Args:
        provider: LLM provider name (default: 'openai')
        model: Model name (default: 'gpt-4')
        **kwargs: Additional parameters for LLM configuration
        
    Returns:
        Configured LangChain LLM instance (guaranteed to be a Runnable)
    """
    return LLMProvider.create_llm(provider=provider, model=model, **kwargs) 