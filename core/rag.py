"""RAG (Retrieval Augmented Generation) components."""
from typing import Any, Dict, List, Optional, Type, Union, Callable
from uuid import UUID
import logging
from core.base import Node
from core.llm import LLMNode
from core.vectorstores import VECTORSTORES
from core.models import Document

# Set up logging
logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass


class RAGConfig:
    """Configuration for RAG components."""
    def __init__(self, 
                vectorstore_type: str, 
                vectorstore_config: Dict[str, Any],
                num_results: int = 4, 
                rerank_results: bool = False,
                filter: Optional[Dict[str, Any]] = None,
                prompt_template: Optional[str] = None):
        """
        Initialize RAG configuration.
        
        Args:
            vectorstore_type: Type of vector store to use
            vectorstore_config: Vector store configuration
            num_results: Number of results to return from search
            rerank_results: Whether to rerank results
            filter: Optional filter for vector store search
            prompt_template: Optional custom prompt template
        """
        # Validate vectorstore type
        if vectorstore_type not in VECTORSTORES:
            raise ValueError(f"Unknown vector store type: {vectorstore_type}. Available types: {list(VECTORSTORES.keys())}")
        
        self.vectorstore_type = vectorstore_type
        self.vectorstore_config = vectorstore_config
        
        # Validate num_results
        if num_results <= 0:
            raise ValueError("num_results must be positive")
        self.num_results = num_results
        
        self.rerank_results = rerank_results
        self.filter = filter
        
        # Default prompt template if none provided
        self.prompt_template = prompt_template or """
Use the following context to answer the question. If you cannot answer the question based on the context alone, say so.

Context:
{context}

Question: {query}

Answer:
"""


class RAGNode(Node):
    """
    Node that implements retrieval-augmented generation.
    
    This node provides the following capabilities:
    1. Document storage and retrieval from vector stores
    2. Semantic search based on query embeddings
    3. Context augmentation for LLM prompts
    4. Optional reranking of search results
    5. Optional response generation
    """
    
    def __init__(self, 
        name: str,
        config: RAGConfig,
        embedding_node: LLMNode,
        llm_node: Optional[LLMNode] = None
    ):
        """
        Initialize RAG node.
        
        Args:
            name: Node name
            config: RAG configuration
            embedding_node: LLM node used for generating embeddings
            llm_node: Optional LLM node for generating responses
        """
        super().__init__(name=name)
        self.config = config
        self.embedding_node = embedding_node
        self.llm_node = llm_node
        
        vectorstore_cls = VECTORSTORES.get(config.vectorstore_type)
        self.vector_store = vectorstore_cls(
            config=config.vectorstore_config,
            embedding_node=embedding_node
        )
        self._initialized = False
        self._has_documents = False
    
    async def initialize(self):
        """Initialize vector store."""
        if not self._initialized:
            logger.info(f"Initializing RAG node: {self.name}")
            await self.vector_store.initialize()
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._initialized:
            logger.info(f"Cleaning up RAG node: {self.name}")
            try:
                await self.vector_store.cleanup()
            except Exception as e:
                logger.warning(f"Error during vector store cleanup: {str(e)}")
            finally:
                self._initialized = False
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs using RAG:
        1. Initialize if needed
        2. Retrieve relevant documents if available
        3. Optionally rerank results
        4. Generate response if LLM node is provided
        
        Args:
            inputs: Dictionary containing at least 'query'
            
        Returns:
            Dictionary with query, context, retrieved documents and optionally response
            
        Raises:
            RAGError: If processing fails
            ValueError: If required inputs are missing
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
                
                if not docs:
                    logger.warning(f"No documents found for query: {query[:50]}...")
                else:
                    logger.info(f"Retrieved {len(docs)} documents for query")
                
                # Rerank if enabled and we have more results than needed
                if self.config.rerank_results and len(docs) > self.config.num_results:
                    docs = await self._rerank_documents(query, docs)
                
                # Augment input context
                context_parts = []
                for i, doc in enumerate(docs):
                    # Include document index for better referencing
                    context_parts.append(f"[{i+1}] {doc.content}")
                
                result["context"] = "\n\n".join(context_parts)
                result["retrieved_docs"] = docs
            else:
                logger.warning("No documents available in the vector store")
            
            # Generate response if LLM node is provided and requested
            if self.llm_node and inputs.get("generate_response", True):
                prompt = self._create_prompt(query, result["context"])
                llm_result = await self.llm_node.process({"prompt": prompt})
                result["response"] = llm_result["response"]
                
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            raise RAGError(f"RAG processing failed: {str(e)}") from e
    
    async def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Number of documents added
            
        Raises:
            RAGError: If adding documents fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            count = await self.vector_store.add_documents(documents)
            self._has_documents = True
            logger.info(f"Added {count} documents to vector store")
            return count
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise RAGError(f"Failed to add documents: {str(e)}") from e
    
    async def delete_documents(self, ids: List[Union[str, UUID]]) -> int:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
            
        Raises:
            RAGError: If deleting documents fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            count = await self.vector_store.delete(ids)
            logger.info(f"Deleted {count} documents from vector store")
            return count
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise RAGError(f"Failed to delete documents: {str(e)}") from e
    
    async def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Rerank documents using cross-attention scores if LLM supports it.
        
        Args:
            query: Search query
            docs: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not self.llm_node:
            return docs[:self.config.num_results]
            
        logger.info(f"Reranking {len(docs)} documents")
        
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
        """
        Create a prompt for the LLM using retrieved context and template.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        return self.config.prompt_template.format(context=context, query=query)

    def set_prompt_template(self, template: str):
        """
        Set a custom prompt template.
        
        Args:
            template: New prompt template (must contain {context} and {query} placeholders)
        """
        if "{context}" not in template or "{query}" not in template:
            raise ValueError("Prompt template must contain {context} and {query} placeholders")
        self.config.prompt_template = template


class DocumentLoader(Node):
    """
    Node for loading and preprocessing documents.
    
    This node handles converting various input formats into Document objects.
    """
    
    def __init__(self, name: str, preprocessors: List[Callable[[Document], Document]] = None):
        """
        Initialize document loader.
        
        Args:
            name: Node name
            preprocessors: Optional list of document preprocessor functions
        """
        super().__init__(name=name)
        self.preprocessors = preprocessors or []
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and preprocess documents from various sources.
        
        Args:
            inputs: Dictionary containing 'documents' list
            
        Returns:
            Dictionary with processed documents
            
        Raises:
            ValueError: If no documents are provided
        """
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
        
        # Apply preprocessors in sequence
        for preprocessor in self.preprocessors:
            processed_docs = [preprocessor(doc) for doc in processed_docs]
                
        logger.info(f"Processed {len(processed_docs)} documents")
        return {"documents": processed_docs}
    
    def add_preprocessor(self, preprocessor: Callable[[Document], Document]):
        """
        Add a document preprocessor function.
        
        Args:
            preprocessor: Function that takes and returns a Document
        """
        self.preprocessors.append(preprocessor)


# Common preprocessors that can be used with DocumentLoader
def chunk_document(max_length: int, overlap: int = 0) -> Callable[[Document], List[Document]]:
    """
    Create a preprocessor that chunks documents by size.
    
    Args:
        max_length: Maximum chunk length
        overlap: Overlap between chunks
        
    Returns:
        Preprocessor function
    """
    def preprocessor(doc: Document) -> List[Document]:
        if len(doc.content) <= max_length:
            return [doc]
            
        chunks = []
        content = doc.content
        i = 0
        
        while i < len(content):
            chunk_end = min(i + max_length, len(content))
            chunks.append(Document(
                content=content[i:chunk_end],
                metadata=doc.metadata.copy()
            ))
            i += max_length - overlap
            
        return chunks
    
    return preprocessor


# Export classes and functions
__all__ = [
    "RAGConfig", 
    "RAGNode", 
    "DocumentLoader", 
    "RAGError", 
    "chunk_document"
]