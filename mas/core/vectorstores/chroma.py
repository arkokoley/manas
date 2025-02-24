"""Chroma vector store implementation."""
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base import VectorStoreProvider
from ..models import Document
from ..llm import LLMNode

class ChromaConfig(BaseModel):
    """Configuration for Chroma vector store."""
    collection_name: str
    persist_directory: Optional[str] = None
    client_settings: Optional[Dict[str, Any]] = None

class ChromaVectorStore(VectorStoreProvider):
    """Chroma-based vector store implementation."""
    
    def __init__(self, config: ChromaConfig, embedding_node: LLMNode):
        self.config = config
        self.embedding_node = embedding_node
        self._executor = ThreadPoolExecutor()
        
    async def initialize(self):
        """Initialize Chroma client and collection."""
        settings = Settings(
            persist_directory=self.config.persist_directory,
            **(self.config.client_settings or {})
        )
        
        loop = asyncio.get_event_loop()
        self.client = await loop.run_in_executor(
            self._executor,
            chromadb.Client,
            settings
        )
        
        # Get or create collection
        self.collection = await loop.run_in_executor(
            self._executor,
            self.client.get_or_create_collection,
            self.config.collection_name
        )
    
    async def cleanup(self):
        """Cleanup Chroma resources."""
        if self.config.persist_directory:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self.client.persist
            )
        self._executor.shutdown()
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to the Chroma collection."""
        embeddings = []
        ids = []
        metadatas = []
        texts = []
        
        # Prepare batch
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                doc.embedding = await self.embedding_node.get_embedding(doc.content)
            embeddings.append(doc.embedding)
            ids.append(str(i))  # Chroma requires string IDs
            metadatas.append(doc.metadata)
            texts.append(doc.content)
        
        # Add to Chroma
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self.collection.add,
            ids,
            embeddings,
            metadatas,
            texts
        )
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        # Get query embedding
        query_embedding = await self.embedding_node.get_embedding(query)
        
        # Search in Chroma
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter
        )
        
        # Convert to Documents
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                embedding=results['embeddings'][0][i]
            )
            documents.append(doc)
            
        return documents
    
    async def delete(self, filter: Dict[str, Any]):
        """Delete documents matching the filter."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self.collection.delete,
            where=filter
        )