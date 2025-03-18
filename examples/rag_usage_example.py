"""
Examples of using the RAG system with different vector stores and advanced features.
Demonstrates:
1. Basic RAG setup
2. Custom document processing
3. Hybrid search implementation
"""

import os
import asyncio
from typing import List, Tuple, Dict, Any
from pathlib import Path

from core import RAG, LLM
from manas_ai.models import Document, RAGConfig
from manas_ai.vectorstores import (
    FaissVectorStore,
    ChromaStore, 
    PineconeStore
)

# Basic RAG Setup
async def basic_rag_example():
    """Demonstrate basic RAG usage with FAISS."""
    print("\n=== Basic RAG Example ===")
    
    # Initialize LLM
    model = LLM.from_provider(
        "openai",
        model_name="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create vector store
    store = FaissVectorStore(dimension=1536)  # OpenAI embedding dimension
    
    # Initialize RAG
    rag = RAG(
        llm=model,
        vector_store=store,
        config=RAGConfig(
            chunk_size=500,
            chunk_overlap=50
        )
    )
    
    # Add some documents
    documents = [
        Document(
            content="Quantum computing uses quantum bits or qubits...",
            metadata={"source": "quantum_basics.txt"}
        ),
        Document(
            content="Modern cryptography relies on mathematical problems...",
            metadata={"source": "cryptography.txt"}
        ),
        Document(
            content="Shor's algorithm can break RSA encryption...",
            metadata={"source": "quantum_algorithms.txt"}
        )
    ]
    
    await rag.add_documents(documents)
    
    # Query the system
    result = await rag.query(
        "How does quantum computing affect encryption?",
        max_sources=3
    )
    
    print("\nQuery Result:")
    print(result["answer"])
    print("\nSources Used:")
    for source in result["sources"]:
        print(f"- {source['metadata']['source']}")

# Custom Document Processing
class CustomRAG(RAG):
    """RAG with custom document processing."""
    
    async def process_document(self, doc: Document) -> List[Document]:
        """Custom document processing logic."""
        # Split into sections
        sections = self._split_into_sections(doc.content)
        
        # Process each section
        chunks = []
        for section in sections:
            chunk = Document(
                content=section["text"],
                metadata={
                    "source": doc.metadata["source"],
                    "section": section["title"],
                    "importance": self._calculate_importance(section["text"])
                }
            )
            chunks.append(chunk)
            
        return chunks
        
    def _split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Split text into titled sections."""
        # Simple section splitting on double newlines
        sections = []
        current = {"title": "Introduction", "text": ""}
        
        for line in text.split("\n"):
            if line.strip() and line == line.strip():
                # Possible title
                if current["text"]:
                    sections.append(current)
                    current = {"title": line, "text": ""}
            else:
                current["text"] += line + "\n"
                
        if current["text"]:
            sections.append(current)
            
        return sections
        
    def _calculate_importance(self, text: str) -> float:
        """Calculate section importance (0-1)."""
        # Simple importance based on length and keyword presence
        importance_keywords = [
            "key", "important", "significant", "crucial", "essential",
            "breakthrough", "discovery", "advance", "novel", "innovative"
        ]
        
        # Length score
        length_score = min(len(text.split()) / 500, 1.0)
        
        # Keyword score
        text_lower = text.lower()
        keyword_matches = sum(
            1 for word in importance_keywords 
            if word in text_lower
        )
        keyword_score = min(keyword_matches / 5, 1.0)
        
        # Combined score
        return (length_score + keyword_score) / 2
        
    def get_query_context(self, docs: List[Document]) -> str:
        """Build context with section information."""
        # Sort by importance
        docs.sort(key=lambda d: d.metadata["importance"], reverse=True)
        
        # Build context
        context = []
        for doc in docs:
            context.append(f"[{doc.metadata['section']}]")
            context.append(doc.content)
            
        return "\n\n".join(context)

# Hybrid Search Implementation
class HybridRAG(RAG):
    """RAG with hybrid search capabilities."""
    
    async def retrieve_context(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Implement hybrid semantic + keyword search."""
        # Get semantic search results
        semantic_results = await self.vector_store.similarity_search(
            query,
            k=k
        )
        
        # Get keyword search results
        keyword_results = await self._keyword_search(query, k=k)
        
        # Combine results with weights
        combined = self._merge_results(
            semantic_results,
            keyword_results,
            weights=[0.7, 0.3]  # Favor semantic results
        )
        
        return combined[:k]
        
    async def _keyword_search(
        self,
        query: str,
        k: int
    ) -> List[Tuple[Document, float]]:
        """Simple keyword-based search."""
        results = []
        query_terms = set(query.lower().split())
        
        # Get all documents
        docs = await self.vector_store.get_documents()
        
        # Score each document
        for doc in docs:
            score = self._keyword_score(doc.content, query_terms)
            if score > 0:
                results.append((doc, score))
                
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
        
    def _keyword_score(self, text: str, query_terms: set) -> float:
        """Calculate keyword match score."""
        text_terms = set(text.lower().split())
        matches = query_terms.intersection(text_terms)
        return len(matches) / len(query_terms)
        
    def _merge_results(
        self,
        semantic_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        weights: List[float]
    ) -> List[Tuple[Document, float]]:
        """Merge and deduplicate results."""
        # Normalize scores
        sem_max = max(score for _, score in semantic_results) if semantic_results else 1
        key_max = max(score for _, score in keyword_results) if keyword_results else 1
        
        # Create score map
        scores = {}
        
        # Add semantic scores
        for doc, score in semantic_results:
            scores[doc.metadata["source"]] = {
                "doc": doc,
                "score": (score / sem_max) * weights[0]
            }
            
        # Add/update with keyword scores
        for doc, score in keyword_results:
            source = doc.metadata["source"]
            if source in scores:
                scores[source]["score"] += (score / key_max) * weights[1]
            else:
                scores[source] = {
                    "doc": doc,
                    "score": (score / key_max) * weights[1]
                }
                
        # Convert back to list and sort
        results = [
            (data["doc"], data["score"])
            for data in scores.values()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

async def custom_rag_example():
    """Demonstrate custom RAG usage."""
    print("\n=== Custom RAG Example ===")
    
    # Initialize components
    model = LLM.from_provider("openai", model_name="gpt-4")
    store = ChromaStore(
        collection_name="custom_docs",
        persist_directory="chroma_db"
    )
    
    # Create custom RAG
    rag = CustomRAG(
        llm=model,
        vector_store=store
    )
    
    # Add document with sections
    doc = Document(
        content=(
            "# Introduction\n"
            "This document covers quantum computing basics.\n\n"
            "# Key Concepts\n"
            "Quantum bits are the fundamental unit...\n\n"
            "# Important Applications\n"
            "Quantum computers excel at certain tasks...\n\n"
            "# Future Directions\n"
            "Several breakthrough advances are expected..."
        ),
        metadata={"source": "quantum_overview.md"}
    )
    
    await rag.add_documents([doc])
    
    # Query with custom processing
    result = await rag.query(
        "What are the key applications of quantum computing?"
    )
    
    print("\nCustom Processing Result:")
    print(result["answer"])
    print("\nSections Used:")
    for source in result["sources"]:
        print(f"- {source['metadata']['section']}")

async def hybrid_rag_example():
    """Demonstrate hybrid search RAG."""
    print("\n=== Hybrid Search Example ===")
    
    # Initialize components
    model = LLM.from_provider("openai", model_name="gpt-4")
    store = PineconeStore(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment="us-west1-gcp",
        index_name="hybrid-search"
    )
    
    # Create hybrid RAG
    rag = HybridRAG(
        llm=model,
        vector_store=store
    )
    
    # Add test documents
    docs = [
        Document(
            content="Python is a popular programming language...",
            metadata={"source": "python.txt"}
        ),
        Document(
            content="JavaScript runs in web browsers...",
            metadata={"source": "javascript.txt"}
        ),
        Document(
            content="Python's simple syntax makes it ideal for beginners...",
            metadata={"source": "python_intro.txt"}
        )
    ]
    
    await rag.add_documents(docs)
    
    # Test hybrid search
    result = await rag.query(
        "What makes Python good for beginners?"
    )
    
    print("\nHybrid Search Result:")
    print(result["answer"])
    print("\nSources (with scores):")
    for source in result["sources"]:
        print(f"- {source['metadata']['source']}: {source['score']:.2f}")

async def main():
    """Run all examples."""
    # Basic RAG
    await basic_rag_example()
    
    # Custom RAG
    await custom_rag_example()
    
    # Hybrid RAG
    await hybrid_rag_example()

if __name__ == "__main__":
    asyncio.run(main())