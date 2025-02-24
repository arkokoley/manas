"""FAISS vector store implementation."""
from typing import Any, Dict, List, Optional
import os
import numpy as np
import faiss
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base import VectorStoreProvider
from ..models import Document
from ..llm import LLMNode

class FAISSVectorStore(VectorStoreProvider):
    """FAISS-based vector store implementation."""
    
    def __init__(self, config: Dict[str, Any], embedding_node: LLMNode):
        self.config = config  # Now accepts raw dictionary
        self.embedding_node = embedding_node
        self._executor = ThreadPoolExecutor()
        self._documents = []  # Store documents to maintain mapping
        
        # Get dimension from config or from embedding node
        self.dimension = config.get("dimension") or embedding_node.config.embedding_dimension
        if not self.dimension:
            raise ValueError("dimension must be specified in config or available from embedding_node")
    
    async def initialize(self):
        """Initialize FAISS index."""
        index_type = self.config.get("index_type", "L2")
        device = self.config.get("device", "cpu")
        store_path = self.config.get("store_path")
        
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "Cosine":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        if device == "gpu" and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
            
        if store_path and os.path.exists(store_path):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                faiss.read_index,
                store_path
            )
    
    async def cleanup(self):
        """Cleanup FAISS resources."""
        store_path = self.config.get("store_path")
        if store_path:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                faiss.write_index,
                self.index,
                store_path
            )
        self._executor.shutdown()
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to the FAISS index."""
        # Get or compute embeddings
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = await self.embedding_node.get_embedding(doc.content)
            if len(doc.embedding) != self.dimension:
                raise ValueError(f"Document embedding dimension {len(doc.embedding)} does not match index dimension {self.dimension}")
            embeddings.append(doc.embedding)
            
        # Add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        if self.config.get("index_type") == "Cosine":
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self.index.add,
            embeddings_array
        )
        
        # Store documents
        self._documents.extend(documents)
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        # Get query embedding
        query_embedding = await self.embedding_node.get_embedding(query)
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding dimension {len(query_embedding)} does not match index dimension {self.dimension}")
            
        query_array = np.array([query_embedding]).astype('float32')
        
        if self.config.get("index_type") == "Cosine":
            faiss.normalize_L2(query_array)
        
        # Search
        loop = asyncio.get_event_loop()
        D, I = await loop.run_in_executor(
            self._executor,
            self.index.search,
            query_array,
            k
        )
        
        # Get documents and filter if needed
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self._documents):  # Handle invalid indices
                continue
            doc = self._documents[idx]
            if filter is None or all(
                doc.metadata.get(k) == v for k, v in filter.items()
            ):
                results.append(doc)
                
        return results
    
    async def delete(self, filter: Dict[str, Any]):
        """Delete documents matching the filter."""
        # FAISS doesn't support direct deletion, so we rebuild the index
        kept_docs = []
        kept_embeddings = []
        
        for doc in self._documents:
            if not all(doc.metadata.get(k) == v for k, v in filter.items()):
                kept_docs.append(doc)
                kept_embeddings.append(doc.embedding)
                
        # Reset index
        await self.initialize()
        
        if kept_docs:
            # Add remaining documents
            embeddings_array = np.array(kept_embeddings).astype('float32')
            if self.config.get("index_type") == "Cosine":
                faiss.normalize_L2(embeddings_array)
                
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self.index.add,
                embeddings_array
            )
            
        self._documents = kept_docs