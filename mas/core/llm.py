"""Core LLM components with provider integration."""
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from uuid import UUID

from .base import Node
from .providers.base import BaseLLMProvider
from .providers import PROVIDERS

class LLMConfig:
    """Configuration for LLM operations."""
    def __init__(self, provider_name: str, provider_config: Dict[str, Any],
                 max_retries_on_timeout: int = 3, context_window: Optional[int] = None,
                 truncate_input: bool = True, stop_sequences: List[str] = None,
                 temperature: float = 0.7, streaming: bool = False,
                 max_tokens: Optional[int] = None, embedding_dimension: Optional[int] = None):
        self.provider_name = provider_name
        self.provider_config = provider_config
        self.max_retries_on_timeout = max_retries_on_timeout
        self.context_window = context_window
        self.truncate_input = truncate_input
        self.stop_sequences = stop_sequences or []
        self.temperature = temperature
        self.streaming = streaming
        self.max_tokens = max_tokens
        self.embedding_dimension = embedding_dimension  # Now optional

class LLMNode(Node):
    """Node for LLM operations using configured provider."""
    def __init__(self, name: str, config: LLMConfig):
        super().__init__(name=name)
        self.config = config
        provider_cls = PROVIDERS.get(config.provider_name)
        if not provider_cls:
            raise ValueError(f"Unknown provider: {config.provider_name}")
        self._provider = provider_cls(config)
        self._initialized = False
        # Get embedding dimension from provider if not specified in config
        if self.config.embedding_dimension is None:
            self.config.embedding_dimension = getattr(self._provider, 'embedding_dimension', 384)
    
    async def initialize(self):
        """Initialize the LLM provider."""
        if not self._initialized:
            await self._provider.initialize()
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup provider resources."""
        if self._initialized:
            await self._provider.cleanup()
            self._initialized = False
    
    async def call_llm(self, prompt: Union[str, Dict[str, Any]]) -> Union[str, AsyncIterator[str]]:
        """Make an LLM API call using configured provider."""
        if self.config.streaming:
            return await self._provider.stream_generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop_sequences=self.config.stop_sequences
            )
        else:
            return await self._provider.generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop_sequences=self.config.stop_sequences
            )
    
    async def get_embedding(self, text: str) -> list[float]:
        """Get embeddings for text using provider."""
        return await self._provider.embed(text)
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs using the LLM with automatic lifecycle management."""
        if not self._initialized:
            await self.initialize()
            
        try:
            if "prompt" not in inputs:
                raise ValueError("Input must contain 'prompt'")
            
            if not self.config.streaming:
                response = await self.call_llm(inputs["prompt"])
                return {"response": response}
            else:
                response = []
                async for chunk in await self.call_llm(inputs["prompt"]):
                    response.append(chunk)
                return {"response": "".join(response)}
        except Exception as e:
            # Ensure cleanup on error
            await self.cleanup()
            raise e

class PromptTemplate:
    """Template for structured prompts."""
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)

class ChainNode(Node):
    """Node that chains multiple LLM calls together."""
    def __init__(self, name: str, nodes: List[LLMNode], prompt_templates: List[PromptTemplate]):
        super().__init__(name=name)
        self.nodes = nodes
        self.prompt_templates = prompt_templates
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs through the chain of LLM calls."""
        current_context = inputs.copy()
        
        for node, template in zip(self.nodes, self.prompt_templates):
            prompt = template.format(**current_context)
            result = await node.process({"prompt": prompt})
            current_context.update(result)
            
        return current_context