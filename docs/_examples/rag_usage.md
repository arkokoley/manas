---
layout: example
title: Working with RAG
description: Learn how to use Retrieval-Augmented Generation (RAG) with various vector stores
nav_order: 3
parent: Examples
difficulty: Intermediate
time: 25 minutes
source_file: rag_usage_example.py
related_docs:
  - title: RAG API Reference
    url: /api/rag/
  - title: Vector Stores
    url: /api/vectorstores/
  - title: QANode with RAG
    url: /api/nodes/qa_node/
---

# Working with RAG

This tutorial demonstrates how to use Retrieval-Augmented Generation (RAG) to enhance LLM responses with relevant context from your documents.

## Overview

We'll learn how to:
1. Set up RAG with different vector stores
2. Process and index documents
3. Create RAG-enabled nodes
4. Build flows with RAG support

## Prerequisites

```bash
# Install with vector store support
pip install "manas-ai[vector-stores]"

# Optional: Install specific vector store backends
pip install "manas-ai[faiss]"  # For FAISS support
pip install "manas-ai[chroma]"  # For Chroma support
pip install "manas-ai[pinecone]"  # For Pinecone support
```

## Basic RAG Setup

```python
import os
from core import RAG, LLM
from core.vectorstores import FaissVectorStore

# Initialize LLM
model = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create vector store
vector_store = FaissVectorStore(dimension=1536)  # OpenAI embedding dimension

# Initialize RAG system
rag = RAG(
    llm=model,
    vector_store=vector_store,
    chunk_size=500,     # Document chunk size
    chunk_overlap=50    # Overlap between chunks
)

# Add documents
await rag.add_documents([
    "path/to/document1.pdf",
    "path/to/document2.txt",
    "path/to/documents/*.md"
])

# Query with context enhancement
result = await rag.query(
    "What are the key findings?",
    max_sources=3  # Number of relevant chunks to include
)

print("Answer:", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"- {source['content'][:200]}...")
```

## Using Different Vector Stores

### Chroma

```python
from core.vectorstores import ChromaStore

store = ChromaStore(
    collection_name="research_docs",
    persist_directory="path/to/chroma_db"
)

rag = RAG(
    llm=model,
    vector_store=store,
    embedding_model="text-embedding-ada-002"
)
```

### Pinecone

```python
from core.vectorstores import PineconeStore

store = PineconeStore(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment="us-west1-gcp",
    index_name="research-index"
)

rag = RAG(llm=model, vector_store=store)
```

## RAG in Nodes

### QANode with RAG

```python
from core.nodes import QANode, RAGConfig

# Create QA node with RAG support
qa_node = QANode(
    name="researcher",
    llm=model,
    config=RAGConfig(
        use_rag=True,
        system_prompt=(
            "You are an expert researcher. Use the provided context "
            "to give detailed, accurate answers."
        ),
        rag_config={
            "chunk_size": 500,
            "chunk_overlap": 50,
            "max_sources": 3,
            "min_relevance": 0.7
        }
    )
)

# Add knowledge base
await qa_node.add_documents([
    "knowledge_base/research_papers/",
    "knowledge_base/articles/"
])

# Process queries
result = await qa_node.process({
    "question": "Explain recent quantum computing breakthroughs"
})
```

## Advanced RAG Usage

### Custom Document Processing

```python
from core.models import Document
from typing import List

class CustomRAG(RAG):
    """RAG with custom document processing."""
    
    async def process_document(self, doc: Document) -> List[Document]:
        """Custom document processing logic."""
        chunks = []
        
        # Custom chunking strategy
        sections = self.split_into_sections(doc.content)
        for section in sections:
            # Create chunk with metadata
            chunk = Document(
                content=section.text,
                metadata={
                    "source": doc.metadata["source"],
                    "section": section.title,
                    "page": section.page
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_query_context(self, query: str, docs: List[Document]) -> str:
        """Custom context building."""
        # Sort by relevance and section importance
        sorted_docs = self.sort_by_relevance_and_section(docs)
        
        # Build context with section headers
        context = []
        for doc in sorted_docs:
            context.append(f"[{doc.metadata['section']}]")
            context.append(doc.content)
        
        return "\n\n".join(context)

# Usage
custom_rag = CustomRAG(
    llm=model,
    vector_store=vector_store
)
```

### Hybrid Search

```python
from core import RAG
from typing import List, Tuple

class HybridRAG(RAG):
    """RAG with hybrid search capabilities."""
    
    async def retrieve_context(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Hybrid semantic + keyword search."""
        # Get semantic search results
        semantic_results = await self.vector_store.similarity_search(
            query, k=k
        )
        
        # Get keyword search results
        keyword_results = self.keyword_search(query, k=k)
        
        # Combine and deduplicate results
        all_results = self.merge_results(
            semantic_results,
            keyword_results,
            weights=[0.7, 0.3]  # Weight semantic vs keyword results
        )
        
        return all_results[:k]

# Usage
hybrid_rag = HybridRAG(
    llm=model,
    vector_store=vector_store
)
```

## Best Practices

1. **Document Processing**
   - Choose appropriate chunk sizes
   - Maintain document structure
   - Include relevant metadata
   - Handle different formats properly

2. **Vector Store Selection**
   - Consider dataset size
   - Evaluate update frequency
   - Check hosting requirements
   - Monitor performance

3. **Query Optimization**
   - Tune relevance thresholds
   - Balance context length
   - Consider hybrid approaches
   - Cache frequent queries

4. **Resource Management**
   - Clean up unused embeddings
   - Monitor memory usage
   - Implement proper persistence
   - Handle updates efficiently

## Notes

- Initialize vector stores before adding documents
- Clean up resources when done
- Monitor embedding costs
- Consider privacy implications
- Test with representative queries
- Validate retrieved context quality