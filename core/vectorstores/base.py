"""Base implementation for vector store providers."""
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging
from contextlib import asynccontextmanager

from core.models import Document
from core.llm import LLMNode
from .protocol import VectorStoreError

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers with common functionality."""
    
    def __init__(self, config: Dict[str, Any], embedding_node: LLMNode):
        """
        Initialize vector store provider.
        
        Args:
            config: Configuration dictionary
            embedding_node: Node for generating embeddings
        """
        self.config = config
        self.embedding_node = embedding_node
        self._initialized = False
    
    async def initialize(self):
        """Initialize the vector store."""
        self._initialized = True
    
    async def cleanup(self):
        """Cleanup vector store resources."""
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    @asynccontextmanager
    async def session(self):
        """Create a managed vector store session."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()
    
    async def _ensure_initialized(self):
        """Ensure the vector store is initialized before use."""
        if not self._initialized:
            await self.initialize()
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def similarity_search(self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete(self, filter: Dict[str, Any]):
        """Delete documents matching the filter."""
        pass