"""Pinecone vector store implementation."""
from typing import Any, Dict, List, Optional
import pinecone
from pinecone import Pinecone, Config

from .base import VectorStoreProvider
from ..models import Document

class PineconeVectorStore(VectorStoreProvider):
    """Pinecone vector store implementation."""
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        namespace: Optional[str] = None,
        dimension: int = 384,
        metric: str = "cosine"
    ):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.pc = None
        self.index = None

    async def initialize(self):
        """Initialize Pinecone client and create index if needed."""
        self.pc = Pinecone(
            api_key=self.api_key,
            config=Config(
                environment=self.environment
            )
        )
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric
            )
            
        self.index = self.pc.Index(self.index_name)

    async def cleanup(self):
        """Cleanup Pinecone resources."""
        if self.index:
            self.index = None
        if self.pc:
            self.pc = None

    async def add_documents(self, documents: List[Document]):
        """
        Add documents to Pinecone.
        
        Args:
            documents: List of documents with embeddings
        """
        vectors = []
        for i, doc in enumerate(documents):
            if not doc.embedding:
                raise ValueError(f"Document {i} has no embedding")
                
            metadata = doc.metadata or {}
            metadata["text"] = doc.page_content
            
            vectors.append({
                "id": str(doc.id),
                "values": doc.embedding,
                "metadata": metadata
            })
            
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents in Pinecone.
        
        Args:
            query: Query embedding values
            k: Number of results to return
            filter: Metadata filter
            
        Returns:
            List of similar documents
        """
        results = self.index.query(
            vector=query,
            top_k=k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True
        )
        
        documents = []
        for match in results.matches:
            doc = Document(
                page_content=match.metadata.pop("text"),
                metadata=match.metadata
            )
            documents.append(doc)
            
        return documents

    async def delete(self, filter: Dict[str, Any]):
        """
        Delete vectors matching the filter.
        
        Args:
            filter: Metadata filter for vectors to delete
        """
        self.index.delete(
            filter=filter,
            namespace=self.namespace
        )