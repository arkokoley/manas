"""RAG (Retrieval Augmented Generation) components."""
from typing import Any, Dict, List, Optional, Type
from .base import Node
from .llm import LLMNode
from .vectorstores import VECTORSTORES, VectorStoreProvider
from .models import Document
import asyncio

class RAGConfig:
    """Configuration for RAG components."""
    def __init__(self, vectorstore_type: str, vectorstore_config: Dict[str, Any],
                 num_results: int = 4, rerank_results: bool = False,
                 filter: Optional[Dict[str, Any]] = None):
        self.vectorstore_type = vectorstore_type
        self.vectorstore_config = vectorstore_config
        self.num_results = num_results
        self.rerank_results = rerank_results
        self.filter = filter

class RAGNode(Node):
    """Node that implements retrieval-augmented generation."""
    
    def __init__(self, 
        name: str,
        config: RAGConfig,
        embedding_node: LLMNode,
        llm_node: Optional[LLMNode] = None
    ):
        super().__init__(name=name)
        self.config = config
        self.embedding_node = embedding_node
        self.llm_node = llm_node
        
        vectorstore_cls = VECTORSTORES.get(config.vectorstore_type)
        if not vectorstore_cls:
            raise ValueError(f"Unknown vector store type: {config.vectorstore_type}")
            
        self.vector_store = vectorstore_cls(
            config=config.vectorstore_config,
            embedding_node=embedding_node
        )
        self._initialized = False
        self._has_documents = False
    
    async def initialize(self):
        """Initialize vector store."""
        if not self._initialized:
            await self.vector_store.initialize()
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._initialized:
            await self.vector_store.cleanup()
            self._initialized = False
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs using RAG:
        1. Initialize if needed
        2. Retrieve relevant documents if available
        3. Optionally rerank results
        4. Generate response if LLM node is provided
        """
        if not self._initialized:
            await self.initialize()
        
        query = inputs.get("query")
        if not query:
            raise ValueError("Query is required for RAG processing")
            
        try:
            result = {
                "query": query,
                "context": "",
                "retrieved_docs": []
            }
            
            if self._has_documents:
                # Only try to retrieve if we have documents
                docs = await self.vector_store.similarity_search(
                    query,
                    k=self.config.num_results,
                    filter=self.config.filter
                )
                
                # Rerank if enabled and we have more results than needed
                if self.config.rerank_results and len(docs) > self.config.num_results:
                    docs = await self._rerank_documents(query, docs)
                
                # Augment input context
                result["context"] = "\n".join([doc.content for doc in docs])
                result["retrieved_docs"] = docs
            
            # Generate response if LLM node is provided and requested
            if self.llm_node and inputs.get("generate_response", True):
                prompt = self._create_prompt(query, result["context"])
                llm_result = await self.llm_node.process({"prompt": prompt})
                result["response"] = llm_result["response"]
                
            return result
            
        except Exception as e:
            await self.cleanup()
            raise e
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not self._initialized:
            await self.initialize()
        await self.vector_store.add_documents(documents)
        self._has_documents = True
    
    async def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents using cross-attention scores if LLM supports it."""
        if not self.llm_node:
            return docs[:self.config.num_results]
            
        # This is a simple implementation - could be enhanced with
        # proper cross-attention scoring or a dedicated reranker
        scores = []
        for doc in docs:
            prompt = f"Rate the relevance of this document to the query on a scale of 0-10:\nQuery: {query}\nDocument: {doc.content}"
            result = await self.llm_node.process({"prompt": prompt})
            try:
                score = float(result["response"].strip())
                scores.append((score, doc))
            except ValueError:
                scores.append((0, doc))
                
        scores.sort(reverse=True)
        return [doc for _, doc in scores[:self.config.num_results]]
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM using retrieved context."""
        return f"""Use the following context to answer the question. If you cannot answer the question based on the context alone, say so.

Context:
{context}

Question: {query}

Answer:"""

class DocumentLoader(Node):
    """Node for loading and preprocessing documents."""
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load and preprocess documents from various sources."""
        documents = inputs.get("documents", [])
        if not documents:
            raise ValueError("No documents provided for loading")
            
        processed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                processed_docs.append(Document(content=doc))
            elif isinstance(doc, dict):
                processed_docs.append(Document(**doc))
            elif isinstance(doc, Document):
                processed_docs.append(doc)
                
        return {"documents": processed_docs}