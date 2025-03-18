"""Mock LLM provider for testing purposes."""
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider
from .factory import register_provider

@register_provider("mock")
class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock provider with configuration."""
        self.responses = config.get("responses", {
            "default": "This is a mock response"
        })
        self.calls = []
    
    async def initialize(self):
        """Initialize the provider."""
        pass
    
    async def cleanup(self):
        """Clean up provider resources."""
        pass
    
    async def generate(self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        """Generate text response from prompt."""
        self.calls.append({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop_sequences": stop_sequences,
            "kwargs": kwargs
        })
        return self.responses.get(prompt, self.responses["default"])
    
    async def stream_generate(self, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ):
        """Generate text response from prompt as a stream."""
        response = self.responses.get(prompt, self.responses["default"])
        for chunk in response.split():
            yield chunk + " "
    
    async def embed(self, text: str) -> List[float]:
        """Return mock embeddings."""
        return [0.1] * 384  # Common embedding dimension

    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Return mock batch embeddings."""
        return [[0.1] * 384 for _ in texts]

    @property
    def embedding_dimension(self) -> int:
        """Return mock embedding dimension."""
        return 384
