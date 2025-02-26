"""Pinecone vector store implementation."""
from typing import Any, Dict, List, Optional, Union
import pinecone
from pinecone import Pinecone, Config
import uuid

from core.vectorstores.base import VectorStoreProvider
from core.models import Document
from core.llm import LLMNode
from .factory import register_vectorstore

@register_vectorstore("pinecone")
class PineconeVectorStore(VectorStoreProvider):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: Dict[str, Any], embedding_node: LLMNode):
        super().__init__(config, embedding_node)
        self.api_key = config.get("api_key")
        self.environment = config.get("environment")
        self.index_name = config.get("index_name")
        self.namespace = config.get("namespace")
        self.dimension = config.get("dimension", embedding_node.config.embedding_dimension or 384)
        self.metric = config.get("metric", "cosine")
        self.pc = None
        self.index = None
    
    async def initialize(self):
        """Initialize Pinecone client and create index if needed."""
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        if not self.environment:
            raise ValueError("Pinecone environment is required")
        if not self.index_name:
            raise ValueError("Pinecone index name is required")
            
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
        
        await super().initialize()
    
    async def cleanup(self):
        """Cleanup Pinecone resources."""
        if self.index:
            self.index = None
        if self.pc:
            self.pc = None
        
        await super().cleanup()
    
    async def add_documents(self, documents: List[Document]) -> int:
        """Add documents to Pinecone."""
        await self._ensure_initialized()
        
        vectors = []
        for doc in documents:
            if not doc.embedding:
                doc.embedding = await self.embedding_node.get_embedding(doc.content)
                
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata["text"] = doc.content
            
            vectors.append({
                "id": str(doc.id),
                "values": doc.embedding,
                "metadata": metadata
            })
            
        if vectors:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            
        return len(documents)
    
    async def similarity_search(
        self,
        query: Union[str, List[float]],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents in Pinecone."""
        await self._ensure_initialized()
        
        # Get query embedding if it's a string
        query_embedding = query
        if isinstance(query, str):
            query_embedding = await self.embedding_node.get_embedding(query)
            
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True
        )
        
        documents = []
        for match in results.matches:
            text = match.metadata.pop("text", "")
            doc = Document(
                content=text,
                metadata=match.metadata,
                embedding=match.values if hasattr(match, 'values') else None,
                id=uuid.UUID(match.id) if match.id else uuid.uuid4()
            )
            documents.append(doc)
            
        return documents
    
    async def delete(self, ids_or_filter: Any) -> int:
        """Delete vectors matching IDs or filter."""
        await self._ensure_initialized()
        
        if isinstance(ids_or_filter, dict):
            # Delete by filter
            self.index.delete(
                filter=ids_or_filter,
                namespace=self.namespace
            )
            # Note: Pinecone doesn't return count of deleted items
            return 0  # Can't determine count
        else:
            # Delete by IDs
            ids = [str(id) for id in ids_or_filter]
            self.index.delete(
                ids=ids,
                namespace=self.namespace
            )
            return len(ids)