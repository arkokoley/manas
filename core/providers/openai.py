"""OpenAI provider implementation."""
from typing import Any, Dict, Optional, Union, AsyncIterator
import asyncio
from openai import AsyncOpenAI
from pydantic import Field

from core.providers.base import BaseLLMProvider
from core.models import ModelProviderConfig

class OpenAIConfig(ModelProviderConfig):
    """Configuration for OpenAI provider."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field("gpt-3.5-turbo", description="Model identifier")
    organization: Optional[str] = Field(None, description="OpenAI organization ID")
    embedding_model: str = Field("text-embedding-ada-002", description="Model for embeddings")
    
    model_config = dict(protected_namespaces=())
    provider: str = "openai"

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            organization=config.organization
        )
    
    async def initialize(self):
        """Initialize the OpenAI client."""
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization
        )
    
    async def cleanup(self):
        """Cleanup OpenAI client resources."""
        # OpenAI client doesn't require special cleanup
        pass
    
    async def generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        messages = self._prepare_messages(prompt)
        response = await self.client.chat.completions.create(
            model=self.config.model,
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
            model=self.config.model,
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
        response = await self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _prepare_messages(self, prompt: Union[str, Dict[str, Any]]) -> list[dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            return [prompt]
        else:
            raise ValueError("Prompt must be string or message dict")