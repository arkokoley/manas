"""Base interface for vector store providers."""
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from ..models import Document

class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup vector store resources."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
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