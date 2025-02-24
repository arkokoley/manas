"""Enhanced base provider with middleware support."""
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from ..chat import Message, ChatSession, Middleware, FunctionDefinition

class MiddlewareProvider(BaseLLMProvider):
    """Base provider with middleware support."""
    
    def __init__(self):
        self.middleware: List[Middleware] = []
        self._session = ChatSession()
    
    def add_middleware(self, middleware: Middleware):
        """Add middleware to the provider."""
        self.middleware.append(middleware)
    
    def set_functions(self, functions: List[FunctionDefinition]):
        """Set available functions for the session."""
        self._session.functions = functions
    
    async def _process_middleware(self, message: Message, pre: bool = True) -> Message:
        """Process message through middleware chain."""
        for m in self.middleware:
            if pre:
                message = await m.pre_process(self._session, message)
            else:
                message = await m.post_process(self._session, message)
        return message
    
    async def generate(self, 
        prompt: Union[str, Dict[str, Any], Message], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        """Generate completion with middleware processing."""
        # Convert prompt to Message if needed
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
        
        # Add to session history
        self._session.messages.append(response)
        
        return response.content if response.content else ""
    
    async def stream_generate(self,
        prompt: Union[str, Dict[str, Any], Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion with middleware processing."""
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
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> Union[str, Message]:
        """Raw generation implementation to be provided by concrete classes."""
        pass
    
    @abstractmethod
    async def _raw_stream_generate(self,
        message: Message,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Raw streaming implementation to be provided by concrete classes."""
        pass