---
title: Custom Components
description: Guide to creating custom components and extending Manas
parent: API Reference
---

# Custom Components

This guide explains how to create custom components by extending Manas base classes.

## Custom Nodes

### Base Classes

All custom nodes should inherit from appropriate base classes:
- `BaseNode`: Basic node functionality
- `QANode`: Question-answering capabilities
- `DocumentNode`: Document processing 
- `ToolNode`: Tool integration
- `APINode`: External API integration

### Example: Custom QA Node

```python
from core.nodes import QANode
from core.models import Document
from typing import Dict, Any, List

class CustomQANode(QANode):
    """Custom QA node with specialized processing."""
    
    async def process_question(
        self,
        question: str,
        context: str = None
    ) -> Dict[str, Any]:
        """Custom question processing logic."""
        # Add preprocessing
        enhanced_question = self.enhance_question(question)
        
        # Get base response
        response = await super().process_question(
            enhanced_question,
            context
        )
        
        # Add postprocessing
        response["enhanced"] = True
        response["original_question"] = question
        return response
    
    def enhance_question(self, question: str) -> str:
        """Add custom question enhancement."""
        # Add your enhancement logic
        return f"{question} (Enhanced)"
```

### Example: Custom Document Node

```python
from core.nodes import DocumentNode
from core.models import Document
from typing import List

class CustomDocumentNode(DocumentNode):
    """Custom document processing node."""
    
    async def process_document(self, doc: Document) -> List[Document]:
        """Custom document processing."""
        # Implement custom chunking
        chunks = self.custom_chunk_document(doc)
        
        # Process each chunk
        processed = []
        for chunk in chunks:
            # Add custom processing
            enhanced = self.enhance_chunk(chunk)
            processed.append(enhanced)
            
        return processed
    
    def custom_chunk_document(self, doc: Document) -> List[str]:
        """Custom document chunking strategy."""
        # Implement your chunking logic
        return doc.content.split("\n\n")
    
    def enhance_chunk(self, chunk: str) -> Document:
        """Enhance a document chunk."""
        # Add your enhancement logic
        return Document(
            content=f"Enhanced: {chunk}",
            metadata={"enhanced": True}
        )
```

## Custom Vector Stores

### Base Class

Custom vector stores should inherit from `BaseVectorStore`:

```python
from core.vectorstores import BaseVectorStore
from core.models import Document
from typing import List, Tuple

class CustomVectorStore(BaseVectorStore):
    """Custom vector store implementation."""
    
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.initialize_store()
    
    def initialize_store(self):
        """Initialize storage backend."""
        # Set up your storage
        self.vectors = []
        self.documents = []
    
    async def add_documents(
        self,
        documents: List[Document]
    ) -> int:
        """Add documents to store."""
        # Get embeddings
        embeddings = await self.get_embeddings(
            [doc.content for doc in documents]
        )
        
        # Store vectors and documents
        for vec, doc in zip(embeddings, documents):
            self.vectors.append(vec)
            self.documents.append(doc)
            
        return len(documents)
    
    async def similarity_search(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        # Get query embedding
        query_vec = await self.get_embeddings([query])[0]
        
        # Calculate similarities
        scores = self.calculate_similarities(query_vec)
        
        # Get top k results
        results = []
        for idx, score in self.get_top_k(scores, k):
            results.append((self.documents[idx], score))
            
        return results
```

## Custom Providers 

### Base Class

Custom LLM providers should inherit from `BaseProvider`:

```python
from core.providers import BaseProvider
from core.models import ChatMessage
from typing import List, Dict, Any

class CustomProvider(BaseProvider):
    """Custom LLM provider implementation."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.api_key = kwargs.get("api_key")
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize API client."""
        # Set up your API client
        self.client = YourAPIClient(self.api_key)
    
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate text completion."""
        response = await self.client.complete(
            prompt,
            max_tokens=kwargs.get("max_tokens", 100),
            temperature=kwargs.get("temperature", 0.7)
        )
        return response.text
    
    async def chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> str:
        """Generate chat completion."""
        formatted = self.format_messages(messages)
        response = await self.client.chat(
            formatted,
            **kwargs
        )
        return response.message
```

## Custom Tools

### Tool Definition

Custom tools can be created using the `Tool` class:

```python
from core import Tool
from typing import Dict, Any

def custom_tool_function(
    param1: str,
    param2: int = 10
) -> Dict[str, Any]:
    """
    Custom tool implementation.
    
    Args:
        param1: First parameter
        param2: Second parameter (default: 10)
        
    Returns:
        Tool execution results
    """
    result = your_implementation(param1, param2)
    return {
        "status": "success",
        "result": result
    }

# Create tool
custom_tool = Tool(
    name="custom_tool",
    description="A custom tool that does something",
    function=custom_tool_function,
    metadata={
        "parameters": {
            "param1": {
                "type": "string",
                "description": "First parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Second parameter",
                "default": 10
            }
        },
        "returns": {
            "type": "object",
            "description": "Tool results"
        }
    }
)
```

## Testing Custom Components

### Node Tests

```python
import pytest
from core.models import Document

async def test_custom_node():
    # Create node
    node = CustomQANode(name="test_node")
    
    # Test processing
    result = await node.process_question(
        "Test question?"
    )
    
    assert result["enhanced"]
    assert result["original_question"] == "Test question?"
```

### Vector Store Tests

```python
import numpy as np

async def test_custom_store():
    # Create store
    store = CustomVectorStore(dimension=3)
    
    # Add documents
    docs = [
        Document(content="Test 1"),
        Document(content="Test 2")
    ]
    count = await store.add_documents(docs)
    assert count == 2
    
    # Test search
    results = await store.similarity_search(
        "Test query",
        k=1
    )
    assert len(results) == 1
    assert isinstance(results[0][1], float)
```

### Provider Tests

```python
import pytest
from core.models import ChatMessage

async def test_custom_provider():
    # Create provider
    provider = CustomProvider(api_key="test_key")
    
    # Test generation
    response = await provider.generate(
        "Test prompt",
        max_tokens=50
    )
    assert isinstance(response, str)
    
    # Test chat
    messages = [
        ChatMessage(role="user", content="Hello")
    ]
    response = await provider.chat(messages)
    assert isinstance(response, str)
```

## Best Practices

1. **Inheritance**
   - Extend appropriate base classes
   - Override necessary methods
   - Call super() when needed
   - Maintain interface contracts

2. **Configuration**
   - Use descriptive parameters
   - Provide sensible defaults
   - Validate inputs
   - Document options

3. **Error Handling**
   - Use custom exceptions
   - Provide error context
   - Handle edge cases
   - Clean up resources

4. **Testing**
   - Write comprehensive tests
   - Test edge cases
   - Use async test functions
   - Mock external services

5. **Documentation**
   - Document public interfaces
   - Provide usage examples
   - Explain custom behavior
   - Note requirements

## Notes

- Initialize components properly
- Clean up resources
- Follow async patterns
- Maintain type hints
- Consider performance
- Test thoroughly