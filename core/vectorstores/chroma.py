"""Chroma vector store implementation."""
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

from core.vectorstores.base import VectorStoreProvider
from core.models import Document
from core.llm import LLMNode
from .factory import register_vectorstore

@register_vectorstore("chroma")
class ChromaVectorStore(VectorStoreProvider):
    """Chroma-based vector store implementation."""
    
    def __init__(self, config: Dict[str, Any], embedding_node: LLMNode):
        super().__init__(config, embedding_node)
        self._executor = ThreadPoolExecutor()
        self.collection_name = config.get("collection_name", "default")
        self.persist_directory = config.get("persist_directory")
        self.client_settings = config.get("client_settings", {})
        
    async def initialize(self):
        """Initialize Chroma client and collection."""
        settings = Settings(
            persist_directory=self.persist_directory,
            **self.client_settings
        )
        
        loop = asyncio.get_event_loop()
        self.client = await loop.run_in_executor(
            self._executor,
            lambda: chromadb.Client(settings)
        )
        
        # Get or create collection
        self.collection = await loop.run_in_executor(
            self._executor,
            lambda: self.client.get_or_create_collection(self.collection_name)
        )
        
        await super().initialize()
    
    async def cleanup(self):
        """Cleanup Chroma resources."""
        if self.persist_directory:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: self.client.persist()
            )
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        await super().cleanup()
    
    async def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the Chroma collection."""
        await self._ensure_initialized()
        
        embeddings = []
        ids = []
        metadatas = []
        texts = []
        
        # Prepare batch
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = await self.embedding_node.get_embedding(doc.content)
            embeddings.append(doc.embedding)
            ids.append(str(doc.id))  # Chroma requires string IDs
            metadatas.append(doc.metadata)
            texts.append(doc.content)
        
        # Add to Chroma
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
        )
        
        return len(documents)
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        await self._ensure_initialized()
        
        # Get query embedding
        query_embedding = await self.embedding_node.get_embedding(query)
        
        # Search in Chroma
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            lambda: self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter
            )
        )
        
        # Convert to Documents
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                embedding=results['embeddings'][0][i],
                id=uuid.UUID(results['ids'][0][i]) if results['ids'][0][i] else uuid.uuid4()
            )
            documents.append(doc)
            
        return documents
    
    async def delete(self, ids_or_filter: Any) -> int:
        """Delete documents matching the filter or IDs."""
        await self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        
        if isinstance(ids_or_filter, dict):
            # Delete by filter
            deleted = await loop.run_in_executor(
                self._executor,
                lambda: self.collection.delete(
                    where=ids_or_filter
                )
            )
            return len(deleted) if deleted else 0
        else:
            # Delete by IDs
            ids_to_remove = [str(id) for id in ids_or_filter]
            deleted = await loop.run_in_executor(
                self._executor,
                lambda: self.collection.delete(
                    ids=ids_to_remove
                )
            )
            return len(deleted) if deleted else 0