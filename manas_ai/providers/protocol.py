"""Protocol definition for LLM providers."""
from typing import Protocol, Dict, Any, AsyncIterator, Optional, List, ClassVar, runtime_checkable

@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Protocol defining the interface for LLM providers."""
    
    provider_name: ClassVar[str]
    supports_streaming: ClassVar[bool]
    supports_embeddings: ClassVar[bool]
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        ...
    
    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        ...
    
    async def generate(self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text completion."""
        ...
    
    async def stream_generate(self, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream text completion."""
        ...
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        ...

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
