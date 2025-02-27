"""Factory for creating specialized nodes."""
from typing import Dict, Callable, Any, Type, Optional
import logging
from core.base import Node

# Set up logging
logger = logging.getLogger(__name__)

# Node registry - maps node types to factory functions
_NODE_REGISTRY: Dict[str, Callable[..., Node]] = {}

def register_node(name: str):
    """
    Decorator to register a node factory.
    
    Args:
        name: Name to register the node type under
        
    Returns:
        Decorator function
    """
    def decorator(node_class):
        _NODE_REGISTRY[name] = node_class
        logger.info(f"Registered node type: {name}")
        return node_class
    return decorator

def create_node(
    node_type: str, 
    name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Node:
    """
    Create a node instance by type.
    
    Args:
        node_type: Type of node to create
        name: Name for the node instance
        config: Optional configuration for the node
        **kwargs: Additional node-specific parameters
        
    Returns:
        Node instance
        
    Raises:
        ValueError: If node type is not registered
    """
    if node_type not in _NODE_REGISTRY:
        available = ", ".join(_NODE_REGISTRY.keys())
        raise ValueError(f"Unknown node type: {node_type}. Available types: {available}")
    
    node_class = _NODE_REGISTRY[node_type]
    
    # Create the node
    if config is not None:
        return node_class(name=name, config=config, **kwargs)
    else:
        return node_class(name=name, **kwargs)

def list_node_types() -> list[str]:
    """
    List all registered node types.
    
    Returns:
        List of node type names
    """
    return list(_NODE_REGISTRY.keys())
