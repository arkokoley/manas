"""Base implementation for LLM providers."""
from typing import Dict, Any, Optional, Union, AsyncIterator, List
from abc import ABC, abstractmethod
import logging
from contextlib import asynccontextmanager
from .protocol import LLMProviderProtocol, ProviderError

# Set up logging
logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers with common functionality."""
    
    provider_name = None
    supports_streaming = False
    supports_embeddings = False
    default_embedding_dimension = 384
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with provider configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._initialized = False
    
    async def initialize(self):
        """Initialize the provider (load models, etc)."""
        self._initialized = True
    
    async def cleanup(self):
        """Cleanup provider resources."""
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    @asynccontextmanager
    async def session(self):
        """Create a managed provider session."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()
    
    async def _ensure_initialized(self):
        """Ensure the provider is initialized before use."""
        if not self._initialized:
            await self.initialize()
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for this provider."""
        return self.default_embedding_dimension