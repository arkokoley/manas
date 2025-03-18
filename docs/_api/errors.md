---
title: Error Handling
description: Error types and handling in Manas
layout: reference
parent: API Reference
api_metadata:
  since: "0.1.0"
  status: "stable"
  type: "Core"
---

# Error Handling

This guide explains error handling in Manas and documents all available exceptions.

## Base Exceptions

### ManasError

Base exception for all Manas errors:

```python
from manas_ai.errors import ManasError

class CustomError(ManasError):
    """Custom error implementation."""
    
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause
```

## Component Errors

### NodeError

Errors related to nodes:

```python
from manas_ai.errors import NodeError

# Node initialization error
raise NodeError("Failed to initialize node: invalid configuration")

# Node processing error
raise NodeError("Error processing input", cause=original_error)
```

### FlowError

Flow-related errors:

```python
from manas_ai.errors import FlowError

# Invalid flow structure
raise FlowError("Cycle detected in flow graph")

# Flow execution error
raise FlowError("Node execution failed", node_id="processor")
```

### ProviderError

Provider-related errors:

```python
from manas_ai.errors import ProviderError

# API authentication error
raise ProviderError("Invalid API key")

# Rate limit exceeded
raise ProviderError("Rate limit exceeded", retry_after=60)
```

### RAGError 

RAG system errors:

```python
from manas_ai.errors import RAGError

# Embedding generation failed
raise RAGError("Failed to generate embeddings")

# Vector store error
raise RAGError("Vector similarity search failed")
```

## Operation Errors

### ValidationError

Input validation errors:

```python
from manas_ai.errors import ValidationError

# Invalid parameter
raise ValidationError("temperature must be between 0 and 1")

# Missing required field
raise ValidationError("system_prompt is required")
```

### ConfigurationError

Configuration-related errors:

```python
from manas_ai.errors import ConfigurationError

# Missing configuration
raise ConfigurationError("API key not found in environment")

# Invalid settings
raise ConfigurationError("Invalid vector store configuration")
```

### ResourceError

Resource management errors:

```python
from manas_ai.errors import ResourceError

# Resource not found
raise ResourceError("Document not found: doc1.pdf")

# Resource exhausted
raise ResourceError("Memory limit exceeded")
```

### TimeoutError

Timeout-related errors:

```python
from manas_ai.errors import TimeoutError

# Operation timeout
raise TimeoutError("Node execution timed out after 30s")

# Provider timeout
raise TimeoutError("API request timed out")
```

## Error Handling

### Try-Except Patterns

```python
from manas_ai.errors import ManasError, NodeError, TimeoutError

async def process_with_retries(node, input_data):
    """Process with error handling and retries."""
    max_retries = 3
    attempt = 0
    
    while attempt < max_retries:
        try:
            return await node.process(input_data)
            
        except TimeoutError as e:
            attempt += 1
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 ** attempt)
            
        except NodeError as e:
            logger.error(f"Node error: {e}")
            raise
            
        except ManasError as e:
            logger.error(f"Unexpected error: {e}")
            raise
```

### Error Recovery

```python
from manas_ai.errors import FlowError
from typing import Dict, Any

class RecoverableFlow:
    """Flow with error recovery capabilities."""
    
    async def process_with_recovery(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process with recovery strategies."""
        try:
            return await self.process(input_data)
            
        except FlowError as e:
            if self.can_recover(e):
                return await self.recover_flow(e)
            raise
            
    async def recover_flow(self, error: FlowError) -> Dict[str, Any]:
        """Implement recovery strategy."""
        if error.node_id:
            # Try alternate node
            alternate = self.get_alternate_node(error.node_id)
            if alternate:
                return await self.retry_with_node(alternate)
                
        # Fall back to default
        return await self.fallback_processing()
```

### Custom Error Handlers

```python
from manas_ai.errors import ErrorHandler
from typing import Optional

class CustomErrorHandler(ErrorHandler):
    """Custom error handling logic."""
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle specific error types."""
        if isinstance(error, TimeoutError):
            return await self.handle_timeout(error, context)
            
        if isinstance(error, ValidationError):
            return await self.handle_validation(error, context)
            
        # Log unexpected errors
        logger.error(f"Unhandled error: {error}", exc_info=True)
        return None
        
    async def handle_timeout(
        self,
        error: TimeoutError,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle timeout errors."""
        # Implement timeout recovery
        return {
            "status": "timeout",
            "message": str(error),
            "retry_after": 30
        }
```

## Error Prevention

### Input Validation

```python
from manas_ai.errors import ValidationError
from typing import Dict, Any

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values."""
    if "temperature" in config:
        temp = config["temperature"]
        if not (0 <= temp <= 1):
            raise ValidationError(
                "temperature must be between 0 and 1"
            )
            
    if "max_tokens" in config:
        tokens = config["max_tokens"]
        if not isinstance(tokens, int) or tokens < 1:
            raise ValidationError(
                "max_tokens must be a positive integer"
            )
```

### Resource Management

```python
from manas_ai.errors import ResourceError
from contextlib import asynccontextmanager

class ResourceManager:
    """Manage system resources."""
    
    @asynccontextmanager
    async def managed_resource(self):
        """Resource context manager."""
        try:
            # Acquire resource
            resource = await self.acquire()
            yield resource
            
        except Exception as e:
            # Handle cleanup on error
            await self.cleanup(resource)
            raise ResourceError(
                "Resource error",
                cause=e
            )
            
        finally:
            # Always release
            await self.release(resource)
```

## Best Practices

1. **Error Hierarchy**
   - Use appropriate base class
   - Create specific error types
   - Include error context
   - Maintain error chain

2. **Error Handling**
   - Handle expected errors
   - Log unexpected errors
   - Implement recovery
   - Clean up resources

3. **Error Prevention**
   - Validate inputs
   - Check preconditions
   - Use context managers
   - Handle edge cases

4. **Error Reporting**
   - Include context
   - Provide solutions
   - Log details
   - Enable debugging

5. **Testing**
   - Test error cases
   - Verify recovery
   - Check cleanup
   - Validate handlers

## Notes

- Use appropriate error types
- Include error context
- Enable proper recovery
- Clean up resources
- Log error details
- Test error cases