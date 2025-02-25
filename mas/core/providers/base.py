"""Base interface for LLM providers."""
from typing import Any, Dict, Optional, Union, AsyncIterator, List, Tuple, ClassVar
from abc import ABC, abstractmethod
import logging
from contextlib import asynccontextmanager

# Set up logging
logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base error class for provider-related errors."""
    pass


class InitializationError(ProviderError):
    """Error raised when provider initialization fails."""
    pass


class GenerationError(ProviderError):
    """Error raised when text generation fails."""
    pass


class EmbeddingError(ProviderError):
    """Error raised when embedding creation fails."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    # Class variables for provider registration
    provider_name: ClassVar[str] = None
    supports_streaming: ClassVar[bool] = False
    supports_embeddings: ClassVar[bool] = False
    default_embedding_dimension: ClassVar[int] = 384
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with provider configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize the provider (load models, etc)."""
        self._initialized = True
    
    @abstractmethod
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
    
    @abstractmethod
    async def generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate completion from the LLM.
        
        Args:
            prompt: Text prompt or structured input
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that should stop generation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text completion
            
        Raises:
            GenerationError: If generation fails
        """
        await self._ensure_initialized()
    
    @abstractmethod
    async def stream_generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream completion from the LLM.
        
        Args:
            prompt: Text prompt or structured input
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that should stop generation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            AsyncIterator yielding generated text chunks
            
        Raises:
            GenerationError: If streaming fails
        """
        await self._ensure_initialized()
        if not self.supports_streaming:
            raise NotImplementedError(f"Provider {self.__class__.__name__} does not support streaming")
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Get embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            EmbeddingError: If embedding fails
        """
        await self._ensure_initialized()
        if not self.supports_embeddings:
            raise NotImplementedError(f"Provider {self.__class__.__name__} does not support embeddings")
    
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Default implementation calls embed() for each text, but providers
        should override this for efficient batched implementation.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings, one per input text
            
        Raises:
            EmbeddingError: If batch embedding fails
        """
        await self._ensure_initialized()
        if not self.supports_embeddings:
            raise NotImplementedError(f"Provider {self.__class__.__name__} does not support embeddings")
            
        results = []
        for text in texts:
            embedding = await self.embed(text)
            results.append(embedding)
        return results
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for this provider."""
        return self.default_embedding_dimension


# Dictionary to register provider implementations
PROVIDERS = {}


def register_provider(cls):
    """
    Decorator to register a provider implementation.
    
    Usage:
        @register_provider
        class MyProvider(BaseLLMProvider):
            provider_name = "my_provider"
            # ...
    """
    if not cls.provider_name:
        raise ValueError(f"Provider {cls.__name__} must define a provider_name")
    
    PROVIDERS[cls.provider_name] = cls
    logger.info(f"Registered provider: {cls.provider_name}")
    return cls


# Export all necessary elements
__all__ = [
    'BaseLLMProvider', 
    'ProviderError',
    'InitializationError', 
    'GenerationError', 
    'EmbeddingError',
    'register_provider', 
    'PROVIDERS'
]