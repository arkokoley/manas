---
title: Node
description: Base node class for all Manas nodes
parent: API Reference
has_children: true
---

# Node

The `Node` class is the foundational building block for all specialized nodes in Manas. It provides core functionality for input processing, state management, and resource handling.

## Import

```python
from core.base import Node
```

## Constructor

```python
def __init__(self, name: str, description: Optional[str] = None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Unique identifier for the node |
| description | Optional[str] | Optional description of node's purpose |

## Core Methods

### initialize

```python
async def initialize(self) -> None
```

Initialize node resources. Must be called before first use.

### cleanup

```python
async def cleanup(self) -> None
```

Clean up node resources. Should be called when node is no longer needed.

### process

```python
async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]
```

Process node inputs. All nodes must implement this method.

| Parameter | Type | Description |
|-----------|------|-------------|
| inputs | Dict[str, Any] | Input data for processing |

Returns a dictionary containing processed results.

### _validate_inputs

```python
def _validate_inputs(self, inputs: Dict[str, Any]) -> None
```

Validate input data. Override to add custom validation.

### _process_impl

```python
async def _process_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]
```

Internal processing implementation. Override this instead of `process()`.

## Properties

### state

```python
@property
def state(self) -> Dict[str, Any]
```

Get current node state (read-only).

### is_initialized

```python
@property
def is_initialized(self) -> bool
```

Check if node is initialized.

## State Management

### update_state

```python
def update_state(self, updates: Dict[str, Any]) -> None
```

Update node state with new values.

### clear_state

```python
def clear_state(self) -> None
```

Clear node state.

## Example Usage

### Basic Node Implementation

```python
class CustomNode(Node):
    """A custom node implementation."""
    
    async def _process_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Custom processing logic
        result = await self.process_data(inputs["data"])
        return {"result": result}
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        if "data" not in inputs:
            raise ValueError("Missing required 'data' input")

# Usage
node = CustomNode(name="custom_processor")
await node.initialize()

try:
    result = await node.process({"data": "example"})
    print(result)
finally:
    await node.cleanup()
```

### State Management

```python
class StatefulNode(Node):
    """A node that maintains state."""
    
    async def _process_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Update counter in state
        current_count = self.state.get("count", 0)
        self.update_state({"count": current_count + 1})
        
        return {
            "current_count": self.state["count"],
            "result": self.process_with_state(inputs["data"])
        }
```

## Flow Integration

```python
from core import Flow

# Create flow
flow = Flow()

# Add nodes
node1 = CustomNode(name="processor1")
node2 = CustomNode(name="processor2")

flow.add_node(node1)
flow.add_node(node2)

# Connect nodes
flow.add_edge(node1, node2)
```

## Best Practices

1. **Resource Management**
   - Always call `initialize()` before use
   - Use `cleanup()` in finally blocks
   - Implement proper async cleanup

2. **State Handling**
   - Use state for persistent data
   - Clear state when appropriate
   - Don't modify state directly

3. **Error Handling**
   - Validate inputs properly
   - Handle async errors
   - Clean up on errors

4. **Implementation**
   - Override `_process_impl()` not `process()`
   - Add custom validation in `_validate_inputs()`
   - Keep nodes focused and single-purpose

## Notes

- The Node class is abstract; don't use it directly
- All node methods are coroutines
- State is maintained between process calls
- Nodes must be properly initialized and cleaned up