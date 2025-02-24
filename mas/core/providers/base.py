"""Base interface for LLM providers."""
from typing import Any, Dict, Optional, Union, AsyncIterator
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the provider (load models, etc)."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup provider resources."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    @abstractmethod
    async def generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        """Generate completion from the LLM."""
        pass
    
    @abstractmethod
    async def stream_generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion from the LLM."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Get embeddings for text."""
        pass