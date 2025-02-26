"""Factory for creating and managing LLM providers."""
from typing import Dict, Type, Callable, Any
import logging
from .protocol import LLMProviderProtocol

# Set up logging
logger = logging.getLogger(__name__)

# Provider registry - maps provider names to factory functions
_PROVIDER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], LLMProviderProtocol]] = {}

def register_provider(name: str):
    """
    Decorator to register a provider factory.
    
    Args:
        name: Name to register the provider under
        
    Returns:
        Decorator function
    """
    def decorator(provider_class):
        _PROVIDER_REGISTRY[name] = lambda config: provider_class(config)
        logger.info(f"Registered provider: {name}")
        return provider_class
    return decorator

def create_provider(provider_name: str, config: Dict[str, Any]) -> LLMProviderProtocol:
    """
    Create a provider instance by name.
    
    Args:
        provider_name: Name of the provider to create
        config: Configuration for the provider
        
    Returns:
        Provider instance
        
    Raises:
        ValueError: If provider is not registered
    """
    if provider_name not in _PROVIDER_REGISTRY:
        available = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {available}")
    
    # Create the provider with config
    return _PROVIDER_REGISTRY[provider_name](config)

def list_providers() -> list[str]:
    """
    List all registered providers.
    
    Returns:
        List of provider names
    """
    return list(_PROVIDER_REGISTRY.keys())
