"""Protocol definition for vector store providers."""
from typing import Protocol, Dict, Any, List, Optional, runtime_checkable
from uuid import UUID
from core.models import Document

@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol defining the interface for vector stores."""
    
    async def initialize(self) -> None:
        """Initialize the vector store."""
        ...
    
    async def cleanup(self) -> None:
        """Cleanup vector store resources."""
        ...
    
    async def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Number of documents added
        """
        ...
    
    async def similarity_search(self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query string or embedding
            k: Number of documents to return
            filter: Optional filter to apply
            
        Returns:
            List of similar documents
        """
        ...
    
    async def delete(self, ids_or_filter: Any) -> int:
        """
        Delete documents from the store.
        
        Args:
            ids_or_filter: Document IDs or filter specification
            
        Returns:
            Number of documents deleted
        """
        ...

class VectorStoreError(Exception):
    """Base error class for vector store operations."""
    pass

class InitializationError(VectorStoreError):
    """Error during vector store initialization."""
    pass

class SearchError(VectorStoreError):
    """Error during similarity search."""
    pass

class DocumentError(VectorStoreError):
    """Error handling documents."""
    pass
