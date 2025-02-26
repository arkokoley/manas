# MAS Framework Specification

## Core Design Principles

1. **Modular Architecture**
   - Each component is independently usable and testable
   - Clear interfaces between components via protocols
   - No tight coupling between modules
   - Factory-based extension system

2. **Clean Code Practices**
   - Follow DRY (Don't Repeat Yourself) principle
   - Single Responsibility Principle
   - Dependency injection over global state
   - Comprehensive error handling

3. **Asynchronous First**
   - All operations are async by default
   - Proper resource management with context managers
   - Structured concurrency patterns
   - Stream processing support

4. **Type Safety**
   - Extensive use of type hints
   - Generic types for flexibility
   - Runtime type validation without Pydantic
   - Clear interface contracts

## Component Architecture

### Base Components

1. **Protocol System**
   - Clear protocol definitions using Python's Protocol type
   - Runtime protocol validation
   - Separation of interface from implementation
   - Type-safe interactions between components

2. **Factory System**
   - Centralized component creation
   - Runtime component registration
   - Explicit dependency injection
   - Configurable implementations

3. **Provider System**
   - Protocol-based provider interface
   - Decorator-based provider registration
   - Common base functionality
   - Configurable behavior

4. **Vector Store System**
   - Protocol-based vector store interface 
   - Factory-based vector store creation
   - Consistent embedding integration
   - Pluggable backends

5. **Node System**
   - Abstract Node base class
   - Standardized input/output validation
   - Resource lifecycle management
   - Error propagation

6. **Flow System**
   - Directed acyclic graph (DAG) processing
   - Multiple execution modes
   - Event-based progress tracking
   - Error recovery

7. **Error Management**
   - Centralized error handling
   - Contextual error information
   - Severity levels
   - Error recovery strategies

### Extension Points

1. **Custom Providers**
```python
from mas.core.providers.factory import register_provider
from mas.core.providers.base import BaseLLMProvider

@register_provider("custom")
class CustomProvider(BaseLLMProvider):
    provider_name = "custom"
    supports_streaming = True
    
    async def initialize(self):
        # Setup
        await super().initialize()
        
    async def generate(self, prompt: str, **kwargs) -> str:
        await self._ensure_initialized()
        # Implementation
        pass
```

2. **Custom Vector Stores**
```python
from mas.core.vectorstores.factory import register_vectorstore
from mas.core.vectorstores.base import VectorStoreProvider

@register_vectorstore("custom_store")
class CustomVectorStore(VectorStoreProvider):
    async def add_documents(self, documents: List[Document]) -> int:
        await self._ensure_initialized()
        # Implementation
        return len(documents)
    
    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        await self._ensure_initialized()
        # Implementation
        pass
```

3. **Custom Nodes**
```python
class CustomNode(Node):
    async def _process_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Processing logic
        pass
```

4. **Custom Middleware**
```python
class CustomMiddleware(BaseMiddleware):
    async def pre_process(self, msg: Message) -> Message:
        # Pre-processing
        pass

    async def post_process(self, msg: Message) -> Message:
        # Post-processing
        pass
```

## Module Dependency Graph
The following diagram represents the dependencies between core modules:

```
                                      +--------------+
                                      |    errors    |
                                      +--------------+
                                            ^
                                            |
                                            v
+--------------+      +--------------+      +--------------+      +--------------+
|    models    |<---->|     base     |<---->|     llm      |<---->|    agent     |
+--------------+      +--------------+      +--------------+      +--------------+
      ^                     ^                     ^                     ^
      |                     |                     |                     |
      v                     v                     v                     v
+--------------+      +--------------+      +--------------+      +--------------+
|  vectorstores |     |     flow     |      |   providers  |      |     rag      |
+--------------+      +--------------+      +--------------+      +--------------+
                                            |      |
                               +------------+      +------------+
                               |                                |
                               v                                v
                       +--------------+                  +--------------+
                       |  middleware  |                  |  provider    |
                       +--------------+                  | implementations|
                                                         +--------------+
```

### Module Responsibilities
1. **errors**: Base error classes and error management system
2. **models**: Data models and configuration classes
3. **base**: Core Node base class and abstractions
4. **llm**: LLM node implementation and LLM integration
5. **providers**: Provider registration and factory functions
6. **provider**: Base provider classes and interfaces
7. **agent**: Agent system implementation
8. **flow**: Workflow engine for connecting nodes
9. **vectorstores**: Vector database integrations
10. **rag**: Retrieval augmented generation
11. **middleware**: Provider middleware system

### Circular Import Resolution
The framework uses several techniques to resolve circular import issues:

1. **Protocol-Based Architecture**: Using Protocol classes to define interfaces separate from implementations.

2. **Factory Pattern**: Centralizing component creation in factory modules that import specific implementations.

3. **Runtime Imports**: Deferring imports to runtime when necessary.

4. **Registration System**: Using decorator-based registration to decouple component definition from use.

Implementation details:
- `LLMProviderProtocol` in `providers/protocol.py` defines provider interface
- `register_provider` decorator in `providers/factory.py` handles registration
- `create_provider` function in `providers/factory.py` creates instances
- `BaseLLMProvider` in `providers/base.py` provides common functionality
- Specific providers import from `protocol.py` and `factory.py`, not vice versa

This pattern is also applied to vector stores and other extensible components.

## Validation & Configuration

1. **Configuration**
   - Use dataclasses for configuration
   - Runtime validation
   - Type checking
   - Default values

2. **Input Validation**
   - Required vs optional inputs
   - Type validation
   - Value constraints
   - Custom validators

3. **Error Handling**
   - Specific error types
   - Error context
   - Recovery strategies
   - Logging

## Anti-Patterns to Avoid

1. **No Global State**
   ❌ Avoid global variables
   ✅ Use dependency injection
   ✅ Pass configuration explicitly

2. **No Direct Framework Dependencies**
   ❌ No Pydantic dependencies
   ❌ No direct OpenAI/other provider imports
   ✅ Use abstract interfaces
   ✅ Plugin system for integrations

3. **No Mixed Responsibilities**
   ❌ Don't mix processing and I/O
   ❌ Don't mix configuration and logic
   ✅ Single responsibility principle
   ✅ Clear separation of concerns

4. **No Direct Framework Coupling**
   ❌ No hardcoded dependencies
   ❌ No framework-specific code
   ✅ Abstract interfaces
   ✅ Plugin architecture

5. **No Circular Imports**
   ❌ Avoid importing types from modules that import from you
   ✅ Use deferred imports inside methods where possible
   ✅ Refactor modules to break circular dependencies
   ✅ Use Abstract Base Classes and Protocol classes to define interfaces

## Best Practices

1. **Resource Management**
   - Use async context managers
   - Proper cleanup in __aexit__
   - Resource pooling
   - Connection management

2. **Error Handling**
   - Always use specific error types
   - Include context information
   - Proper error propagation
   - Recovery strategies

3. **Testing**
   - Unit tests for components
   - Integration tests for flows
   - Mock external dependencies
   - Comprehensive coverage

4. **Performance**
   - Async I/O
   - Resource pooling
   - Batch operations
   - Caching strategies

5. **Clean Provider Implementation**
   - Inherit from `BaseLLMProvider`
   - Register with `@register_provider("name")`
   - Call parent class methods with `await super().method()`
   - Always use `await self._ensure_initialized()` before operations
   - Handle errors gracefully

6. **Clean Vector Store Implementation**
   - Inherit from `VectorStoreProvider`
   - Register with `@register_vectorstore("name")`
   - Use explicit initialization and cleanup
   - Call parent class methods when extending behavior
   - Properly handle async operations

## Extension Guidelines

1. **Provider Extensions**
   - Implement required protocol methods
   - Register using decorator pattern
   - Validate configuration early
   - Support both sync and async contexts
   - Clear error handling

2. **Node Extensions**
   - Clear input/output contract
   - Resource management
   - Error handling
   - State management

3. **Flow Extensions**
   - DAG validation
   - Execution modes
   - Progress tracking
   - Error recovery

4. **Middleware Extensions**
   - Pre/post processing
   - Resource management
   - Error handling
   - Chain composition