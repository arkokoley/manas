"""LLM Provider implementations."""
from typing import Dict, Type
from .base import BaseLLMProvider
from .openai import OpenAIProvider
from .huggingface import HuggingFaceProvider
from .ollama import OllamaProvider

PROVIDERS = {
    "openai": OpenAIProvider,
    "huggingface": HuggingFaceProvider,
    "ollama": OllamaProvider,
}

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider", 
    "HuggingFaceProvider",
    "OllamaProvider",
    "PROVIDERS"
]