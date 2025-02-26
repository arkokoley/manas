"""Anthropic Claude provider implementation."""
from typing import Any, Dict, Optional, Union, AsyncIterator, List
import anthropic
from anthropic import Anthropic, AsyncAnthropic

from core.providers.base import BaseLLMProvider, register_provider
from core.models import ModelProviderConfig

class AnthropicConfig(ModelProviderConfig):
    """Configuration for Anthropic provider."""
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(provider_name="anthropic", **kwargs)
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

@register_provider
class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider implementation."""

    provider_name = "anthropic"
    supports_streaming = True
    supports_embeddings = False

    def __init__(self, config: AnthropicConfig):
        super().__init__(config)
        self.config = config
        self.client = None

    async def initialize(self):
        """Initialize the Anthropic client."""
        self.client = AsyncAnthropic(api_key=self.config.api_key)
        self._initialized = True

    async def cleanup(self):
        """Cleanup Anthropic client resources."""
        self.client = None
        self._initialized = False

    async def generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate completion using Claude."""
        await self._ensure_initialized()
        
        if isinstance(prompt, dict):
            messages = [{"role": prompt["role"], "content": prompt["content"]}]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = await self.client.messages.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stop_sequences=stop_sequences,
            **kwargs
        )
        return response.content[0].text

    async def stream_generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion using Claude."""
        await self._ensure_initialized()
        
        if isinstance(prompt, dict):
            messages = [{"role": prompt["role"], "content": prompt["content"]}]
        else:
            messages = [{"role": "user", "content": prompt}]

        stream = await self.client.messages.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stop_sequences=stop_sequences,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.content and chunk.content[0].text:
                yield chunk.content[0].text