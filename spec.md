# MAS Framework Specification

## Core Design Principles

1. **Modular Architecture**
   - Each component is independently usable and testable
   - Clear interfaces between components via abstract base classes
   - No tight coupling between modules
   - Plugin-based extension system

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

1. **Node System**
   - Abstract Node base class
   - Standardized input/output validation
   - Resource lifecycle management
   - Error propagation

2. **Provider System**
   - Abstract BaseLLMProvider
   - Plugin-based registration
   - Middleware support
   - Configurable behavior

3. **Flow System**
   - Directed acyclic graph (DAG) processing
   - Multiple execution modes
   - Event-based progress tracking
   - Error recovery

4. **Error Management**
   - Centralized error handling
   - Contextual error information
   - Severity levels
   - Error recovery strategies

### Extension Points

1. **Custom Providers**
```python
class CustomProvider(BaseLLMProvider):
    provider_name = "custom"
    supports_streaming = True
    
    async def initialize(self):
        # Setup
        pass

    async def generate(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass
```

2. **Custom Nodes**
```python
class CustomNode(Node):
    async def _process_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Processing logic
        pass
```

3. **Custom Middleware**
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
The framework has several circular import issues that need to be resolved:

1. **llm.py ↔ providers/base.py**: The main circular dependency is between LLMNode needing 
   provider registration and providers needing BaseLLMProvider
   
2. **providers/base.py ↔ providers/provider.py**: Separating provider interface from registration

To resolve these issues:
1. Extract BaseLLMProvider into providers/provider.py
2. Keep provider registration in providers/base.py
3. Have specific providers import from provider.py, not base.py
4. Import PROVIDERS registry at runtime in LLMNode initialization

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

## Extension Guidelines

1. **Provider Extensions**
   - Implement required methods
   - Handle rate limits
   - Support streaming
   - Error handling

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