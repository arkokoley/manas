---
title: RAG
description: Retrieval-Augmented Generation system
parent: API Reference
---

# RAG

The `RAG` class provides functionality for enhancing LLM responses with relevant context from a knowledge base using vector similarity search.

## Import

```python
from core import RAG
from manas_ai.models import Document
```

## Constructor

```python
def __init__(
    self,
    llm: LLM,
    vector_store: VectorStore,
    embedding_model: str = "text-embedding-ada-002",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_sources: int = 3,
    min_relevance: float = 0.7
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| llm | LLM | Required | Language model to use |
| vector_store | VectorStore | Required | Vector store for embeddings |
| embedding_model | str | "text-embedding-ada-002" | Model for embeddings |
| chunk_size | int | 500 | Size of document chunks |
| chunk_overlap | int | 50 | Overlap between chunks |
| max_sources | int | 3 | Max sources to include |
| min_relevance | float | 0.7 | Minimum relevance score |

## Core Methods

### add_documents

```python
async def add_documents(
    self,
    docs: Union[str, List[str], Document, List[Document]]
) -> int
```

Add documents to the knowledge base. Returns number of chunks added.

### query

```python
async def query(
    self,
    query: str,
    max_sources: Optional[int] = None,
    min_relevance: Optional[float] = None
) -> Dict[str, Any]
```

Query the RAG system.

Returns:
```python
{
    "query": str,         # Original query
    "answer": str,        # Generated answer
    "sources": List[Dict] # Relevant sources
}
```

### remove_documents

```python
async def remove_documents(
    self,
    doc_ids: List[str]
) -> None
```

Remove documents from the knowledge base.

### search

```python
async def search(
    self,
    query: str,
    k: int = 3,
    min_relevance: Optional[float] = None
) -> List[Tuple[Document, float]]
```

Search for relevant documents without generating an answer.

## Advanced Methods

### process_document

```python
async def process_document(self, doc: Document) -> List[Document]
```

Process a document into chunks. Override for custom chunking.

### get_query_context

```python
def get_query_context(
    self,
    query: str,
    docs: List[Document]
) -> str
```

Build context from retrieved documents. Override for custom context building.

### format_sources

```python
def format_sources(
    self,
    docs: List[Tuple[Document, float]]
) -> List[Dict[str, Any]]
```

Format source documents for output. Override for custom formatting.

## Example Usage

### Basic Usage

```python
# Initialize RAG
rag = RAG(
    llm=model,
    vector_store=FaissVectorStore(dimension=1536)
)

# Add documents
await rag.add_documents([
    "path/to/documents/*.pdf",
    Document(content="Custom document", metadata={"source": "manual"})
])

# Query
result = await rag.query(
    "What are the key findings?",
    max_sources=5
)
```

### Custom Processing

```python
from manas_ai.models import Document
from typing import List

class CustomRAG(RAG):
    """RAG with custom document processing."""
    
    async def process_document(self, doc: Document) -> List[Document]:
        # Custom chunking logic
        chunks = []
        sections = self.split_into_sections(doc.content)
        
        for section in sections:
            chunk = Document(
                content=section.text,
                metadata={
                    **doc.metadata,
                    "section": section.title
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_query_context(self, query: str, docs: List[Document]) -> str:
        # Custom context building
        return "\n\n".join([
            f"[{doc.metadata['section']}]\n{doc.content}"
            for doc, _ in docs
        ])
```

### Batched Processing

```python
async def process_documents_batch(
    rag: RAG,
    docs: List[str],
    batch_size: int = 10
) -> None:
    """Process documents in batches."""
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        await rag.add_documents(batch)
        print(f"Processed batch {i//batch_size + 1}")
```

## Best Practices

1. **Document Processing**
   - Choose chunk size based on model context
   - Use appropriate chunk overlap
   - Include relevant metadata
   - Clean text before processing

2. **Query Optimization**
   - Set appropriate max_sources
   - Tune min_relevance threshold
   - Consider using hybrid search
   - Cache frequent queries

3. **Resource Management**
   - Initialize vector store first
   - Clean up unused embeddings
   - Monitor memory usage
   - Implement persistence

4. **Performance**
   - Use batched processing
   - Optimize chunking strategy
   - Consider caching
   - Monitor embedding costs

## Notes

- Initialize the vector store before adding documents
- Clean up resources when done
- Monitor embedding API usage
- Consider privacy implications
- Test with representative queries
- Handle document updates properly