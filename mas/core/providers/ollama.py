"""Ollama provider implementation using OpenAI client."""
from typing import Any, Dict, Optional, Union, AsyncIterator
import asyncio
from openai import AsyncOpenAI

from .base import BaseLLMProvider
from ..models import ModelProviderConfig

class OllamaConfig(ModelProviderConfig):
    """Configuration for Ollama provider."""
    def __init__(self, provider_config: Dict[str, Any], **kwargs):
        # Always set provider_name to "ollama" for this config
        kwargs["provider_name"] = "ollama"
        super().__init__(**kwargs)
        self.provider_config = provider_config

class OllamaProvider(BaseLLMProvider):
    """Ollama API provider implementation."""
    
    EMBEDDING_DIMENSIONS = {
        "deepseek-r1": 3584,
        "llama2": 4096,
        "default": 384
    }
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        model_name = config.provider_config.get("model", "").lower().split(":")[0]
        self.embedding_dimension = self.EMBEDDING_DIMENSIONS.get(model_name, self.EMBEDDING_DIMENSIONS["default"])
        self.client = AsyncOpenAI(
            base_url=config.provider_config["base_url"],
            api_key="ollama"  # Ollama doesn't require an API key
        )
    
    async def initialize(self):
        """Initialize the Ollama client."""
        pass  # Client is already initialized in __init__
    
    async def cleanup(self):
        """Cleanup Ollama client resources."""
        pass  # No special cleanup needed
    
    async def generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        messages = self._prepare_messages(prompt)
        response = await self.client.chat.completions.create(
            model=self.config.provider_config["model"],
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
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        messages = self._prepare_messages(prompt)
        stream = await self.client.chat.completions.create(
            model=self.config.provider_config["model"],
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
    
    async def embed(self, text: str) -> list[float]:
        """Get embeddings using the Ollama model."""
        # Use the same model for embeddings
        model = self.config.provider_config["model"]
        response = await self.client.embeddings.create(
            model=model,
            input=text
        )
        # Ensure we return the expected dimension by padding or truncating if needed
        embedding = response.data[0].embedding
        if len(embedding) > self.embedding_dimension:
            return embedding[:self.embedding_dimension]
        elif len(embedding) < self.embedding_dimension:
            return embedding + [0.0] * (self.embedding_dimension - len(embedding))
        return embedding
    
    def _prepare_messages(self, prompt: Union[str, Dict[str, Any]]) -> list[dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            return [prompt]
        else:
            raise ValueError("Prompt must be string or message dict")