# Onboarding Guide: Adding a New LLM Provider

This guide explains how to add ("onboard") a new LLM provider to the `aws-orchestrator-agent` project using the established abstract base class pattern.

## Overview

The project uses a modular, extensible LLM provider architecture. Each provider is implemented as a subclass of `BaseLLMProvider` and registered in the `LLMProvider` factory. This ensures all LLMs are created via a consistent interface and are easily swappable.

---

## Steps to Add a New LLM Provider

### 1. Implement the Provider Class
- **Location:** `aws_orchestrator_agent/core/llm/`
- **File:** Add or update in `llm_provider.py` (or create a new file if preferred)
- **Base Class:** Inherit from `BaseLLMProvider` (see `base_llm_provider.py`)

**Example Skeleton:**
```python
from .base_llm_provider import BaseLLMProvider
from langchain_core.runnables import Runnable

class MyNewProvider(BaseLLMProvider):
    """
    Concrete LLM provider for MyNewProvider models.
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
        # Import your provider's SDK or LangChain integration here
        # from my_provider_sdk import MyProviderLLM
        # Build config dict as needed
        config = {
            "model": model,
            "temperature": temperature,
            # ... other params ...
        }
        config.update(kwargs)
        # Return a LangChain-compatible Runnable
        return MyProviderLLM(**config)
```

### 2. Register the Provider in the Factory
- **Location:** `planner_agent/core/llm/llm_provider.py`
- **Action:** Update the `_create_provider_instance` method in `LLMProvider` to add a new `elif` branch for your provider:

```python
elif provider == "mynewprovider":
    return MyNewProvider().create_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs
    )
```
- Also add your provider name to `_SUPPORTED_PROVIDERS` set at the top of the class.

### 3. Add Environment Variable Support (Optional)
- If your provider requires API keys or endpoints, document the required environment variables in this guide and handle them in your provider class (see OpenAI/Anthropic examples).

### 4. Update Documentation
- Add your provider to this onboarding guide and to any user-facing documentation (e.g., README, config docs).

### 5. Test Your Provider
- Ensure your provider works by running the agent with your provider selected in config or environment variables.
- Add or update tests if possible.

---

## Best Practices
- Follow the structure and docstring style of existing providers.
- Use type hints and clear error messages.
- Keep all provider-specific logic encapsulated in your provider class.
- Do not modify agent or planner code—only the provider and factory.

---

## Example: Adding "MyNewProvider"
1. Implement `MyNewProvider` in `llm_provider.py` (see skeleton above).
2. Register in `LLMProvider._create_provider_instance` and `_SUPPORTED_PROVIDERS`.
3. Handle API keys/endpoints as needed.
4. Test and document.

---

## References
- [base_llm_provider.py](../aws_orchestrator_agent/core/llm/base_llm_provider.py)
- [llm_provider.py](../aws_orchestrator_agent/core/llm/llm_provider.py)

---
