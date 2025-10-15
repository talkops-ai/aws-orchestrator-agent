import json
import os
import warnings
from typing import Dict, Any, List, Union, Type, get_origin, get_args, Optional, Tuple
from aws_orchestrator_agent.config.default import DefaultConfig
from aws_orchestrator_agent.utils.exceptions import ConfigError
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

class Config:
    """
    Configuration class for Planner Agent.

    Precedence order for config values:
    1. Defaults from DefaultConfig
    2. Environment variables (including .env)
    3. Runtime/programmatic overrides (via config dict)

    All config keys are available as attributes and in the internal _config dict.
    """

    def __init__(self, config: Dict[str, Any] = {}) -> None:
        """
        Initialize the configuration.
        Args:
            config: Optional configuration dictionary to override defaults (highest precedence)
        """
        # Start with default configuration
        default_config = {key: getattr(DefaultConfig, key) for key in dir(DefaultConfig) if not key.startswith('_')}
        
        # Merge with provided config
        self._config = default_config.copy()
        self._config.update(config)
        
        # Set attributes from configuration and environment variables
        self._set_attributes(self._config)
    

    def _set_attributes(self, config: Dict[str, Any]) -> None:
        """
        Set attributes from configuration and environment variables.
        Environment variables take precedence over defaults and runtime config.
        Updates both attributes and the internal _config dict.
        Args:
            config: Configuration dictionary
        """
        for key, value in config.items():
            env_value = os.getenv(key)
            if env_value is not None:
                value = self.convert_env_value(key, env_value, DefaultConfig.__annotations__[key])
            setattr(self, key.lower(), value)
            self._config[key] = value  # Ensure internal dict reflects env override

    def __getattr__(self, item: str) -> Any:
        """
        Allow attribute-style access to config keys.
        Raises AttributeError if the key is missing.
        """
        if item in self._config:
            return self._config[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get the standard LLM configuration."""
        return {
            'provider': self._config.get('LLM_PROVIDER') or os.getenv('LLM_PROVIDER') or 'openai',
            'model': self._config.get('LLM_MODEL') or os.getenv('LLM_MODEL') or 'gpt-4o-mini',
            'temperature': self._config.get('LLM_TEMPERATURE') or float(os.getenv('LLM_TEMPERATURE', '0.0')),
            'max_tokens': self._config.get('LLM_MAX_TOKENS') or int(os.getenv('LLM_MAX_TOKENS', '1000'))
        }

    @property
    def llm_higher_config(self) -> Dict[str, Any]:
        """Get the higher-tier LLM configuration."""
        return {
            'provider': self._config.get('LLM_HIGHER_PROVIDER') or os.getenv('LLM_HIGHER_PROVIDER') or 'openai',
            'model': self._config.get('LLM_HIGHER_MODEL') or os.getenv('LLM_HIGHER_MODEL') or 'gpt-5',
            'temperature': self._config.get('LLM_HIGHER_TEMPERATURE') or float(os.getenv('LLM_HIGHER_TEMPERATURE', '1.0')),
            'max_tokens': self._config.get('LLM_HIGHER_MAX_TOKENS') or int(os.getenv('LLM_HIGHER_MAX_TOKENS', '12000'))
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """Get standard LLM configuration.
        
        Returns:
            Standard LLM configuration dictionary
        """
        return self.llm_config

    def get_llm_higher_config(self) -> Dict[str, Any]:
        """Get higher-tier LLM configuration.
        
        Returns:
            Higher-tier LLM configuration dictionary
        """
        return self.llm_higher_config


    def set_llm_config(self, config: Dict[str, Any]) -> None:
        """Set the standard LLM configuration.
        
        Args:
            config: Standard LLM configuration dictionary
        """
        for key, value in config.items():
            if key == 'provider':
                self._config['LLM_PROVIDER'] = value
            elif key == 'model':
                self._config['LLM_MODEL'] = value
            elif key == 'temperature':
                self._config['LLM_TEMPERATURE'] = value
            elif key == 'max_tokens':
                self._config['LLM_MAX_TOKENS'] = value

    def set_llm_higher_config(self, config: Dict[str, Any]) -> None:
        """Set the higher-tier LLM configuration.
        
        Args:
            config: Higher-tier LLM configuration dictionary
        """
        for key, value in config.items():
            if key == 'provider':
                self._config['LLM_HIGHER_PROVIDER'] = value
            elif key == 'model':
                self._config['LLM_HIGHER_MODEL'] = value
            elif key == 'temperature':
                self._config['LLM_HIGHER_TEMPERATURE'] = value
            elif key == 'max_tokens':
                self._config['LLM_HIGHER_MAX_TOKENS'] = value

    @staticmethod
    def convert_env_value(key: str, env_value: str, type_hint: Type) -> Any:
        """Convert environment variable to the appropriate type.
        
        Args:
            key: Configuration key
            env_value: Environment variable value
            type_hint: Type hint for the value
            
        Returns:
            Converted value
        """
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            for arg in args:
                if arg is type(None):
                    if env_value.lower() in ("none", "null", ""):
                        return None
                else:
                    try:
                        return Config.convert_env_value(key, env_value, arg)
                    except Exception:
                        continue
            raise ConfigError(f"Cannot convert {env_value} to any of {args}")

        if type_hint is bool:
            return env_value.lower() in ("true", "1", "yes", "on")
        elif type_hint is int:
            return int(env_value)
        elif type_hint is float:
            return float(env_value)
        elif type_hint in (str, Any):
            return env_value
        elif origin is list or origin is List:
            return json.loads(env_value)
        else:
            raise ConfigError(f"Unsupported type {type_hint} for key {key}")

    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """

        if not os.path.exists(config_path):
            print(f"Warning: Configuration not found at '{config_path}'. Using default configuration.")

        with open(config_path, "r") as f:
            custom_config = json.load(f)

        # Merge with default config
        merged_config = DefaultConfig.__dict__.copy()
        merged_config.update(custom_config)
        return merged_config