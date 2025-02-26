"""Factory for creating and managing vector stores."""
from typing import Dict, Callable, Any, Type
import logging
from core.llm import LLMNode
from .protocol import VectorStoreProtocol

# Set up logging
logger = logging.getLogger(__name__)

# Vector store registry - maps vector store types to factory functions
_VECTORSTORE_REGISTRY: Dict[str, Callable[[Dict[str, Any], LLMNode], VectorStoreProtocol]] = {}

def register_vectorstore(name: str):
    """
    Decorator to register a vector store factory.
    
    Args:
        name: Name to register the vector store under
        
    Returns:
        Decorator function
    """
    def decorator(vectorstore_class):
        _VECTORSTORE_REGISTRY[name] = lambda config, embedding_node: vectorstore_class(config, embedding_node)
        logger.info(f"Registered vector store: {name}")
        return vectorstore_class
    return decorator

def create_vectorstore(
    store_type: str, 
    config: Dict[str, Any],
    embedding_node: LLMNode
) -> VectorStoreProtocol:
    """
    Create a vector store instance by type.
    
    Args:
        store_type: Type of vector store to create
        config: Configuration for the vector store
        embedding_node: LLM node for embeddings
        
    Returns:
        Vector store instance
        
    Raises:
        ValueError: If store type is not registered
    """
    if store_type not in _VECTORSTORE_REGISTRY:
        available = ", ".join(_VECTORSTORE_REGISTRY.keys())
        raise ValueError(f"Unknown vector store type: {store_type}. Available types: {available}")
    
    # Create the vector store with config and embedding node
    return _VECTORSTORE_REGISTRY[store_type](config, embedding_node)

def list_vectorstores() -> list[str]:
    """
    List all registered vector store types.
    
    Returns:
        List of vector store type names
    """
    return list(_VECTORSTORE_REGISTRY.keys())
