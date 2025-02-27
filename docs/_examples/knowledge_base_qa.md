---
title: Building a Knowledge Base QA System
description: Create a question-answering system with RAG and flow-based document processing
nav_order: 4
parent: Examples
difficulty: Advanced
time: 45 minutes
---

# Building a Knowledge Base QA System

This tutorial demonstrates how to build a complete knowledge base question-answering system that processes documents, maintains a searchable index, and answers questions using RAG.

## Overview

We'll build a system that:
1. Processes and indexes documents from multiple sources
2. Uses specialized nodes for different tasks
3. Maintains a vector store for efficient retrieval
4. Provides accurate answers with source citations

## Prerequisites

```bash
pip install "manas-ai[all]" "python-magic" "chardet"
```

## Implementation

### 1. Document Processing Node

```python
from core.nodes import DocumentNode
from core.models import Document
import magic
import chardet
from pathlib import Path
from typing import List, Dict, Any

class DocumentProcessorNode(DocumentNode):
    """Node for processing documents of various formats."""
    
    async def process_file(self, file_path: str) -> Document:
        """Process a single file."""
        path = Path(file_path)
        
        # Detect file type
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(str(path))
        
        # Read and detect encoding
        with open(path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding']
        
        # Read content based on file type
        if file_type == 'text/plain':
            content = raw.decode(encoding)
        elif file_type == 'application/pdf':
            content = await self.extract_pdf_text(path)
        else:
            content = await self.extract_text(path)
            
        return Document(
            content=content,
            metadata={
                "source": str(path),
                "type": file_type,
                "encoding": encoding
            }
        )
    
    async def _process_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process input files."""
        files = inputs.get("files", [])
        if isinstance(files, str):
            files = [files]
            
        documents = []
        for file_path in files:
            doc = await self.process_file(file_path)
            documents.append(doc)
            
        return {
            "documents": documents,
            "count": len(documents)
        }
```

### 2. Knowledge Base Node

```python
from core.nodes import QANode
from core.models import RAGConfig
from core.vectorstores import ChromaStore

class KnowledgeBaseNode(QANode):
    """Node for maintaining and querying a knowledge base."""
    
    def __init__(self, name: str, persist_dir: str):
        # Initialize with Chroma for persistence
        store = ChromaStore(
            collection_name=name,
            persist_directory=persist_dir
        )
        
        super().__init__(
            name=name,
            config=RAGConfig(
                use_rag=True,
                system_prompt=(
                    "You are a knowledgeable assistant. Use the provided context "
                    "to answer questions accurately. Always cite your sources."
                ),
                rag_config={
                    "vector_store": store,
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "max_sources": 5,
                    "min_relevance": 0.75
                }
            )
        )
        
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the knowledge base."""
        await self.rag_node.add_documents(documents)
        
    async def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search the knowledge base without generating an answer."""
        docs = await self.rag_node.search(query, k=k)
        return [
            {
                "content": doc.content,
                "relevance": score,
                "source": doc.metadata["source"]
            }
            for doc, score in docs
        ]
```

### 3. Building the Flow

```python
from core import Flow, LLM
import os

async def create_qa_system(
    docs_dir: str,
    kb_dir: str,
    model_name: str = "gpt-4"
) -> Flow:
    """Create a complete QA system."""
    
    # Initialize LLM
    model = LLM.from_provider(
        "openai",
        model_name=model_name,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create nodes
    processor = DocumentProcessorNode(
        name="doc_processor",
        llm=model
    )
    
    knowledge_base = KnowledgeBaseNode(
        name="knowledge_base",
        persist_dir=kb_dir
    )
    
    # Create flow
    flow = Flow(name="qa_system")
    
    # Add and connect nodes
    processor_id = flow.add_node(processor)
    kb_id = flow.add_node(knowledge_base)
    flow.add_edge(processor_id, kb_id)
    
    # Initialize nodes
    await processor.initialize()
    await knowledge_base.initialize()
    
    # Process initial documents
    docs = [str(p) for p in Path(docs_dir).glob("**/*")]
    result = await processor.process({"files": docs})
    await knowledge_base.add_documents(result["documents"])
    
    return flow

# Usage
async def main():
    # Create system
    flow = await create_qa_system(
        docs_dir="path/to/documents",
        kb_dir="path/to/knowledge_base"
    )
    
    try:
        # Ask questions
        result = await flow.process({
            "question": "What are the key principles of quantum computing?"
        })
        
        # Print answer with sources
        print("Answer:", result["knowledge_base"]["answer"])
        print("\nSources:")
        for source in result["knowledge_base"]["sources"]:
            print(f"- {source['content'][:200]}...")
            print(f"  From: {source['metadata']['source']}\n")
            
    finally:
        # Clean up
        await flow.cleanup()
```

## Advanced Features

### 1. Adding Real-Time Updates

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentUpdateHandler(FileSystemEventHandler):
    """Handle document updates in real-time."""
    
    def __init__(self, kb_node: KnowledgeBaseNode):
        self.kb_node = kb_node
        self.processor = DocumentProcessorNode("update_processor")
    
    async def process_file(self, path: str) -> None:
        """Process and add/update a document."""
        result = await self.processor.process({"files": [path]})
        await self.kb_node.add_documents(result["documents"])
    
    def on_created(self, event):
        if event.is_directory:
            return
        asyncio.create_task(self.process_file(event.src_path))

# Usage
handler = DocumentUpdateHandler(knowledge_base)
observer = Observer()
observer.schedule(handler, docs_dir, recursive=True)
observer.start()
```

### 2. Adding Caching

```python
from functools import lru_cache
from typing import Optional

class CachedKnowledgeBase(KnowledgeBaseNode):
    """Knowledge base with result caching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
    
    @lru_cache(maxsize=1000)
    async def get_cached_answer(
        self,
        question: str,
        max_age: int = 3600
    ) -> Optional[Dict[str, Any]]:
        """Get cached answer if available and fresh."""
        cache_key = self.normalize_question(question)
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < max_age:
                return result
        
        # Get fresh answer
        result = await super().answer(question)
        self.cache[cache_key] = (result, time.time())
        return result
    
    def normalize_question(self, question: str) -> str:
        """Normalize question for cache key."""
        return " ".join(question.lower().split())
```

### 3. Adding Feedback Loop

```python
class FeedbackEnabledKB(KnowledgeBaseNode):
    """Knowledge base that learns from feedback."""
    
    async def record_feedback(
        self,
        question: str,
        answer: Dict[str, Any],
        feedback: Dict[str, Any]
    ) -> None:
        """Record user feedback for an answer."""
        # Store feedback
        await self.store_feedback(question, answer, feedback)
        
        # If feedback indicates answer was wrong
        if not feedback.get("correct", True):
            # Get correct answer if provided
            correct_answer = feedback.get("correct_answer")
            if correct_answer:
                # Create training example
                await self.add_training_example(
                    question=question,
                    context=answer["context"],
                    correct_answer=correct_answer
                )
                
                # Retrain if needed
                if self.should_retrain():
                    await self.retrain_model()
    
    async def get_answer_with_confidence(
        self,
        question: str
    ) -> Dict[str, Any]:
        """Get answer with confidence based on feedback history."""
        result = await self.answer(question)
        
        # Check similar questions' feedback
        similar = await self.find_similar_questions(question)
        confidence = self.calculate_confidence(result, similar)
        
        return {
            **result,
            "confidence": confidence
        }
```

## Best Practices

1. **Document Processing**
   - Handle different file types properly
   - Detect and handle encodings
   - Extract text efficiently
   - Maintain document structure

2. **Knowledge Base Management**
   - Use persistent vector stores
   - Implement proper updates
   - Monitor index size
   - Regular maintenance

3. **Answer Generation**
   - Validate context relevance
   - Include source citations
   - Handle edge cases
   - Provide confidence scores

4. **System Operations**
   - Proper resource cleanup
   - Error handling
   - Monitoring
   - Regular backups

## Notes

- Initialize all components properly
- Handle errors gracefully
- Monitor system resources
- Validate document processing
- Test with various file types
- Consider scaling needs