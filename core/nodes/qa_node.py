"""Node implementation for question answering."""
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass, field
from core.base import Node
from core.llm import LLMNode, LLMConfig
from core.rag import RAGNode, RAGConfig
from .factory import register_node
from .protocol import QANodeProtocol

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class QAConfig:
    """Configuration for QA nodes."""
    model: str = "llama3"
    temperature: float = 0.7
    use_rag: bool = False
    rag_config: Optional[Dict[str, Any]] = None
    prompt_template: str = (
        "Answer the following question based on the provided context.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    follow_up_template: str = (
        "Based on our previous conversation, answer the following question:\n\n"
        "Previous answer: {previous_answer}\n\n"
        "New question: {question}\n\n"
        "Answer:"
    )
    # Advanced options
    options: Dict[str, Any] = field(default_factory=lambda: {
        "summarize_answer": True,
        "include_sources": True,
        "max_sources": 3,
        "confidence_threshold": 0.7
    })

@register_node("qa")
class QANode(Node):
    """Node for question answering with optional RAG support."""
    
    def __init__(self, name: str, config: QAConfig, llm_node: Optional[LLMNode] = None, rag_node: Optional[RAGNode] = None):
        super().__init__(name=name)
        self.config = config
        
        # Use provided LLM node or create a new one
        self.llm_node = llm_node
        
        # Use provided RAG node or create a new one if enabled
        self.rag_node = rag_node
        
        # Keep track of conversation history for follow-up questions
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize QA node resources."""
        if not self._initialized:
            # Create LLM node if not provided
            if self.llm_node is None:
                self.llm_node = LLMNode(
                    name=f"{self.name}_llm",
                    config=LLMConfig(
                        provider_name="ollama" if self.config.model.startswith("llama") else "openai",
                        provider_config={"model": self.config.model},
                        temperature=self.config.temperature
                    )
                )
                await self.llm_node.initialize()
            
            # Create RAG node if enabled and not provided
            if self.config.use_rag and self.rag_node is None and self.config.rag_config:
                from core.vectorstores.factory import create_vectorstore
                
                # Create embedding node if needed
                embedding_node = LLMNode(
                    name=f"{self.name}_embeddings",
                    config=LLMConfig(
                        provider_name="openai",
                        provider_config={"model": "text-embedding-ada-002"},
                    )
                )
                await embedding_node.initialize()
                
                # Create RAG node
                self.rag_node = RAGNode(
                    name=f"{self.name}_rag",
                    config=RAGConfig(**self.config.rag_config),
                    embedding_node=embedding_node,
                    llm_node=self.llm_node
                )
                await self.rag_node.initialize()
            
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._initialized:
            if self.llm_node:
                await self.llm_node.cleanup()
            if self.rag_node:
                await self.rag_node.cleanup()
            self._initialized = False
    
    def _get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if session_id not in self.history:
            self.history[session_id] = []
        return self.history[session_id]
    
    def _add_to_history(self, session_id: str, question: str, answer: str, metadata: Optional[Dict[str, Any]] = None):
        """Add Q&A pair to conversation history."""
        history = self._get_session_history(session_id)
        history.append({
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        })
    
    async def answer(self, 
        question: str, 
        context: Optional[str] = None,
        session_id: Optional[str] = None,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question with optional context and conversation history.
        
        Args:
            question: The question to answer
            context: Optional context for the question
            session_id: Optional session ID for conversation history
            include_history: Whether to include conversation history
            
        Returns:
            Dictionary containing answer and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        # Get conversation history if available
        history = []
        previous_answer = None
        if session_id and include_history:
            history = self._get_session_history(session_id)
            if history:
                previous_answer = history[-1]["answer"]
        
        # Try RAG if enabled and no context provided
        retrieved_docs = []
        if self.config.use_rag and not context and self.rag_node:
            try:
                rag_result = await self.rag_node.process({"query": question})
                context = rag_result.get("context", "")
                retrieved_docs = rag_result.get("retrieved_docs", [])
            except Exception as e:
                logger.error(f"RAG processing error: {str(e)}")
                # Continue without RAG context
        
        # Prepare prompt based on available information
        if previous_answer and not context:
            # Follow-up question with previous answer
            prompt = self.config.follow_up_template.format(
                previous_answer=previous_answer,
                question=question
            )
        else:
            # Regular question with context if available
            prompt = self.config.prompt_template.format(
                context=context or "No additional context available.",
                question=question
            )
        
        # Get answer from LLM
        llm_result = await self.llm_node.process({"prompt": prompt})
        answer = llm_result.get("response", "")
        
        # Add to history if session provided
        if session_id:
            metadata = {
                "context_used": bool(context),
                "rag_used": self.config.use_rag and bool(retrieved_docs),
                "sources": [doc.metadata for doc in retrieved_docs] if retrieved_docs else []
            }
            self._add_to_history(session_id, question, answer, metadata)
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "confidence": 1.0,  # Default confidence
        }
        
        # Add sources if requested
        if retrieved_docs and self.config.options.get("include_sources", True):
            max_sources = self.config.options.get("max_sources", 3)
            result["sources"] = [
                {
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs[:max_sources]
            ]
        
        return result
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs by answering a question.
        
        Args:
            inputs: Dictionary containing at least 'question'
            
        Returns:
            Dictionary containing answer and metadata
            
        Raises:
            ValueError: If required inputs are missing or invalid
        """
        question = inputs.get("question")
        if not question:
            raise ValueError("Input must contain 'question'")
        
        # Extract additional parameters
        context = inputs.get("context")
        session_id = inputs.get("session_id")
        include_history = inputs.get("include_history", True)
        
        # Answer the question
        answer = await self.answer(
            question=question,
            context=context,
            session_id=session_id,
            include_history=include_history
        )
        
        return answer
