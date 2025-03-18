"""Ollama provider implementation using OpenAI client."""
from typing import Any, Dict, Union, Optional, AsyncIterator, List
from .factory import register_provider
from .base import BaseLLMProvider

@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):
    """Ollama API provider implementation."""
    provider_name = "ollama"
    supports_streaming = True
    supports_embeddings = False
    EMBEDDING_DIMENSIONS = {
        "deepseekr1": 3584,
        "llama3.2": 4096,
        "default": 384
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration explicitly
        self.model = config.get("model", "llama3.2")
        self.base_url = config.get("base_url", "http://localhost:11434/v1")
        
        # Import AsyncOpenAI here to avoid circular imports
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="ollama"  # Ollama doesn't require an API key
        )

    async def initialize(self):
        """Initialize the Ollama client."""
        await super().initialize()  # Set _initialized flag

    async def cleanup(self):
        """Cleanup Ollama client resources."""
        await super().cleanup()  # Clear _initialized flag

    async def generate(self,
        prompt: Union[str, Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        await self._ensure_initialized()
        
        messages = self._prepare_messages(prompt)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            **kwargs
        )
        return response.choices[0].message.content

    async def stream_generate(self,
        prompt: Union[str, Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        await self._ensure_initialized()
        
        messages = self._prepare_messages(prompt)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            stream=True,
            **kwargs
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, text: str) -> List[float]:
        """Get embeddings using the Ollama model."""
        await self._ensure_initialized()
        
        # Use the same model for embeddings
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        # Ensure we return the expected dimension by padding or truncating if needed
        embedding = response.data[0].embedding
        if len(embedding) > self.embedding_dimension:
            return embedding[:self.embedding_dimension]
        elif len(embedding) < self.embedding_dimension:
            return embedding + [0.0] * (self.embedding_dimension - len(embedding))
        return embedding

    def _prepare_messages(self, prompt: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            return [prompt]
        else:
            raise ValueError("Prompt must be string or message dict")