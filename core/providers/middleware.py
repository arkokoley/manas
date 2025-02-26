"""Enhanced base provider with middleware support."""
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from abc import ABC, abstractmethod
import logging
from core.chat import Message, ChatSession, Middleware, FunctionDefinition
from core.providers.base import BaseLLMProvider

# Set up logging
logger = logging.getLogger(__name__)


class MiddlewareProvider(BaseLLMProvider):
    """Base provider with middleware support."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with provider configuration.
        
        Args:
            config: Provider-specific configuration
        """
        super().__init__(config)
        self.middleware: List[Middleware] = []
        self._session = ChatSession()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the provider and middleware components."""
        await super().initialize()
        for middleware in self.middleware:
            if hasattr(middleware, 'initialize'):
                await middleware.initialize()
        self._initialized = True
    
    async def cleanup(self):
        """Cleanup provider and middleware resources."""
        # Clean up middleware first
        for middleware in self.middleware:
            if hasattr(middleware, 'cleanup'):
                try:
                    await middleware.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up middleware {middleware.__class__.__name__}: {e}")
        
        # Then clean up provider
        await super().cleanup()
        self._initialized = False
    
    def add_middleware(self, middleware: Middleware):
        """
        Add middleware to the provider.
        
        Args:
            middleware: Middleware instance to add
        """
        self.middleware.append(middleware)
        logger.debug(f"Added middleware: {middleware.__class__.__name__}")
    
    def set_functions(self, functions: List[FunctionDefinition]):
        """
        Set available functions for the session.
        
        Args:
            functions: List of function definitions
        """
        self._session.functions = functions
        logger.debug(f"Set {len(functions)} available functions")
    
    def get_session(self) -> ChatSession:
        """
        Get current chat session.
        
        Returns:
            The current chat session
        """
        return self._session
    
    def clear_session(self):
        """Clear the current session history."""
        self._session.clear_history()
        logger.debug("Session history cleared")
    
    async def _process_middleware(self, message: Message, pre: bool = True) -> Message:
        """
        Process message through middleware chain.
        
        Args:
            message: Message to process
            pre: Whether this is pre-processing (True) or post-processing (False)
            
        Returns:
            Processed message
        """
        current_message = message
        middleware_type = "pre-processing" if pre else "post-processing"
        
        for m in self.middleware:
            try:
                if pre:
                    current_message = await m.pre_process(self._session, current_message)
                else:
                    current_message = await m.post_process(self._session, current_message)
            except Exception as e:
                logger.error(f"Error in middleware {m.__class__.__name__} {middleware_type}: {e}")
                # Continue with current message if middleware fails
        
        return current_message
    
    async def generate(self, 
        prompt: Union[str, Dict[str, Any], Message], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate completion with middleware processing.
        
        Args:
            prompt: Prompt as string, dict or Message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that should stop generation
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        # Convert prompt to Message if needed
        if isinstance(prompt, str):
            message = Message(role="user", content=prompt)
        elif isinstance(prompt, dict):
            message = Message(**prompt)
        else:
            message = prompt
            
        # Pre-process through middleware
        message = await self._process_middleware(message, pre=True)
        logger.debug(f"Pre-processed message: {message.role} - {message.content[:50]}...")
        
        # Add to session history
        self._session.messages.append(message)
        
        # Get raw completion
        response = await self._raw_generate(
            message,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs
        )
        
        # Convert response to Message if needed
        if isinstance(response, str):
            response = Message(role="assistant", content=response)
            
        # Post-process through middleware
        response = await self._process_middleware(response, pre=False)
        logger.debug(f"Post-processed response: {response.role} - {response.content[:50]}...")
        
        # Add to session history
        self._session.messages.append(response)
        
        return response.content if response.content else ""
    
    async def stream_generate(self,
        prompt: Union[str, Dict[str, Any], Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream completion with middleware processing.
        
        Args:
            prompt: Prompt as string, dict or Message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that should stop generation
            **kwargs: Additional parameters
            
        Yields:
            Generated text chunks
        """
        # Convert prompt to Message
        if isinstance(prompt, str):
            message = Message(role="user", content=prompt)
        elif isinstance(prompt, dict):
            message = Message(**prompt)
        else:
            message = prompt
            
        # Pre-process through middleware
        message = await self._process_middleware(message, pre=True)
        
        # Add to session history
        self._session.messages.append(message)
        
        # Stream raw completion
        response_content = []
        async for chunk in self._raw_stream_generate(
            message,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs
        ):
            response_content.append(chunk)
            yield chunk
            
        # Create full response message
        response = Message(
            role="assistant",
            content="".join(response_content)
        )
        
        # Post-process through middleware
        response = await self._process_middleware(response, pre=False)
        
        # Add to session history
        self._session.messages.append(response)
    
    @abstractmethod
    async def _raw_generate(self,
        message: Message,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Union[str, Message]:
        """
        Raw generation implementation to be provided by concrete classes.
        
        Args:
            message: Input message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that should stop generation
            **kwargs: Additional parameters
            
        Returns:
            Generated text or Message
        """
        pass
    
    @abstractmethod
    async def _raw_stream_generate(self,
        message: Message,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Raw streaming implementation to be provided by concrete classes.
        
        Args:
            message: Input message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that should stop generation
            **kwargs: Additional parameters
            
        Yields:
            Generated text chunks
        """
        pass


# Export classes
__all__ = ["MiddlewareProvider"]