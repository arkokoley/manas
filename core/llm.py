"""Core LLM components with provider integration."""
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from uuid import UUID
import logging
import asyncio
from core.base import Node
from core.providers.factory import create_provider, list_providers
from core.providers.protocol import GenerationError

# Set up logging
logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration for LLM operations."""
    def __init__(self, provider_name: str, provider_config: Dict[str, Any],
                 max_retries_on_timeout: int = 3, context_window: Optional[int] = None,
                 truncate_input: bool = True, stop_sequences: List[str] = None,
                 temperature: float = 0.7, streaming: bool = False,
                 max_tokens: Optional[int] = None, embedding_dimension: Optional[int] = None):
        
        # Validate provider exists
        available_providers = list_providers()
        if provider_name not in available_providers:
            raise ValueError(f"Unknown provider: {provider_name}. Available providers: {available_providers}")
            
        self.provider_name = provider_name
        self.provider_config = provider_config
        self.max_retries_on_timeout = max_retries_on_timeout
        self.context_window = context_window
        self.truncate_input = truncate_input
        self.stop_sequences = stop_sequences or []
        
        # Validate temperature
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        self.temperature = temperature
        self.streaming = streaming
        self.max_tokens = max_tokens
        self.embedding_dimension = embedding_dimension


class LLMNode(Node):
    """Node for LLM operations using configured provider."""
    def __init__(self, name: str, config: LLMConfig):
        super().__init__(name=name)
        self.config = config
        
        # Create provider with the provider configuration dict
        self._provider = create_provider(
            config.provider_name, 
            config.provider_config
        )
        self._initialized = False
        
        # Get embedding dimension from provider if not specified in config
        if self.config.embedding_dimension is None:
            self.config.embedding_dimension = getattr(self._provider, 'embedding_dimension', None)
    
    async def initialize(self):
        """Initialize the LLM provider."""
        if not self._initialized:
            try:
                logger.debug(f"Initializing LLM provider: {self.config.provider_name}")
                await self._provider.initialize()
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize LLM provider: {e}")
                raise
    
    async def cleanup(self):
        """Cleanup provider resources."""
        if self._initialized:
            try:
                logger.debug(f"Cleaning up LLM provider: {self.config.provider_name}")
                await self._provider.cleanup()
            except Exception as e:
                logger.warning(f"Error during provider cleanup: {e}")
            finally:
                self._initialized = False
    
    async def call_llm(self, prompt: Union[str, Dict[str, Any]]) -> Union[str, AsyncIterator[str]]:
        """Make an LLM API call using configured provider with retry logic."""
        attempts = 0
        last_error = None
        
        while attempts <= self.config.max_retries_on_timeout:
            try:
                if not self._initialized:
                    await self.initialize()
                    
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
            except asyncio.TimeoutError as e:
                attempts += 1
                last_error = e
                if attempts <= self.config.max_retries_on_timeout:
                    wait_time = 2 ** attempts  # Exponential backoff
                    logger.warning(f"LLM call timed out, retrying in {wait_time}s ({attempts}/{self.config.max_retries_on_timeout})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"LLM call failed after {attempts} attempts")
                    raise GenerationError(f"Maximum retries exceeded: {str(e)}") from e
            except Exception as e:
                logger.error(f"LLM call error: {str(e)}")
                raise GenerationError(f"Error during LLM generation: {str(e)}") from e
        
        if last_error:
            raise GenerationError(f"Maximum retries exceeded: {str(last_error)}") from last_error
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text using provider."""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self._provider.embed(text)
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        if not self._initialized:
            await self.initialize()
            
        try:
            return await self._provider.batch_embed(texts)
        except Exception as e:
            logger.error(f"Batch embedding error: {str(e)}")
            raise
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs using the LLM with automatic lifecycle management."""
        try:
            if not self._initialized:
                await self.initialize()
                
            if "prompt" not in inputs:
                raise ValueError("Input must contain 'prompt'")
            
            if not self.config.streaming:
                response = await self.call_llm(inputs["prompt"])
                return {"response": response}
            else:
                response = []
                async for chunk in await self.call_llm(inputs["prompt"]):
                    response.append(chunk)
                    # Allow passing a callback for streaming if provided
                    if "stream_callback" in inputs and callable(inputs["stream_callback"]):
                        await inputs["stream_callback"](chunk)
                return {"response": "".join(response)}
        except Exception as e:
            logger.error(f"Error in LLMNode.process: {str(e)}")
            # Don't cleanup here automatically to allow for retry
            raise


class PromptTemplate:
    """Template for structured prompts."""
    def __init__(self, template: str, input_variables: List[str], validator=None):
        self.template = template
        self.input_variables = input_variables
        self.validator = validator
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
            
        # Apply validator if provided
        if self.validator and callable(self.validator):
            for var, value in kwargs.items():
                if var in self.input_variables:
                    kwargs[var] = self.validator(var, value)
                    
        return self.template.format(**kwargs)


class ChainNode(Node):
    """Node that chains multiple LLM calls together."""
    def __init__(self, name: str, nodes: List[LLMNode], prompt_templates: List[PromptTemplate]):
        super().__init__(name=name)
        
        if len(nodes) != len(prompt_templates):
            raise ValueError("Number of nodes must match number of prompt templates")
            
        self.nodes = nodes
        self.prompt_templates = prompt_templates
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs through the chain of LLM calls."""
        current_context = inputs.copy()
        chain_results = []
        
        for i, (node, template) in enumerate(zip(self.nodes, self.prompt_templates)):
            try:
                prompt = template.format(**current_context)
                result = await node.process({"prompt": prompt})
                current_context.update(result)
                chain_results.append(result)
            except Exception as e:
                logger.error(f"Error in chain step {i}: {str(e)}")
                raise
                
        # Add chain_results to context for debugging/analysis
        current_context["chain_results"] = chain_results
            
        return current_context


# Export classes
__all__ = [
    "LLMConfig",
    "LLMNode",
    "PromptTemplate",
    "ChainNode"
]