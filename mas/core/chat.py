"""Message format definitions and middleware components."""
from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from uuid import UUID, uuid4

class FunctionCall:
    """Represents a function call in chat messages."""
    def __init__(self, name: str, arguments: Dict[str, Any]):
        self.name = name
        self.arguments = arguments

class FunctionDefinition:
    """Defines a function that can be called by the LLM."""
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

class Message:
    """Base message class for chat interactions."""
    def __init__(self, role: str = "user", content: Optional[str] = None,
                 name: Optional[str] = None, function_call: Optional[FunctionCall] = None):
        self.id = uuid4()
        self.role = role
        self.content = content
        self.name = name
        self.function_call = function_call
        self.timestamp = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}

class ChatSession:
    """Represents a chat session with history."""
    def __init__(self):
        self.id = uuid4()
        self.messages: List[Message] = []
        self.functions: List[FunctionDefinition] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, content: str, role: str = "user",
                   name: Optional[str] = None,
                   function_call: Optional[FunctionCall] = None) -> Message:
        """Add a message to the chat history."""
        message = Message(
            role=role,
            content=content,
            name=name,
            function_call=function_call
        )
        self.messages.append(message)
        return message
    
    def get_context_window(self, max_messages: Optional[int] = None) -> List[Message]:
        """Get recent message history, optionally limited to max_messages."""
        if max_messages is None:
            return self.messages
        return self.messages[-max_messages:]
    
    def clear_history(self):
        """Clear chat history while preserving functions and metadata."""
        self.messages = []

class Memory:
    """Base class for memory implementations."""
    async def add(self, key: str, value: Any):
        """Add an item to memory."""
        raise NotImplementedError
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from memory."""
        raise NotImplementedError
    
    async def remove(self, key: str):
        """Remove an item from memory."""
        raise NotImplementedError
    
    async def clear(self):
        """Clear all items from memory."""
        raise NotImplementedError

class SimpleMemory(Memory):
    """Simple in-memory implementation."""
    def __init__(self):
        self._store: Dict[str, Any] = {}
    
    async def add(self, key: str, value: Any):
        self._store[key] = value
    
    async def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)
    
    async def remove(self, key: str):
        self._store.pop(key, None)
    
    async def clear(self):
        self._store.clear()

class Middleware:
    """Base class for chat middleware."""
    async def pre_process(self, session: ChatSession, message: Message) -> Message:
        """Process message before sending to LLM."""
        return message
    
    async def post_process(self, session: ChatSession, message: Message) -> Message:
        """Process message after receiving from LLM."""
        return message

class MemoryMiddleware(Middleware):
    """Middleware that handles memory operations."""
    def __init__(self, memory: Memory):
        self.memory = memory
    
    async def pre_process(self, session: ChatSession, message: Message) -> Message:
        """Add memory context to message if needed."""
        if "requires_memory" in message.metadata:
            keys = message.metadata["requires_memory"]
            memory_context = {}
            for key in keys:
                value = await self.memory.get(key)
                if value is not None:
                    memory_context[key] = value
            
            if memory_context:
                memory_prefix = "Previous context:\n" + "\n".join(
                    f"{k}: {v}" for k, v in memory_context.items()
                )
                message.content = f"{memory_prefix}\n\n{message.content}"
        
        return message
    
    async def post_process(self, session: ChatSession, message: Message) -> Message:
        """Store relevant information in memory."""
        if "store_in_memory" in message.metadata:
            key_mappings = message.metadata["store_in_memory"]
            for memory_key, content_key in key_mappings.items():
                if isinstance(content_key, str):
                    await self.memory.add(memory_key, message.content)
                elif isinstance(content_key, dict):
                    if message.function_call and message.function_call.name == content_key["function"]:
                        value = message.function_call.arguments.get(content_key["arg"])
                        if value is not None:
                            await self.memory.add(memory_key, value)
        
        return message