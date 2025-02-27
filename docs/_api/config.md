---
title: Configuration
description: Configuration options and settings for Manas components
layout: reference
parent: API Reference
api_metadata:
  since: "0.1.0"
  status: "stable"
  type: "Core"
---

# Configuration

This page documents all configuration options available in Manas.

## Global Configuration

### Environment Variables

```bash
# OpenAI configuration
OPENAI_API_KEY=your-api-key
OPENAI_ORG_ID=your-org-id    # Optional

# Anthropic configuration
ANTHROPIC_API_KEY=your-api-key

# HuggingFace configuration
HUGGINGFACE_API_KEY=your-api-key

# Pinecone configuration
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=your-environment

# Logging configuration
MANAS_LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR
MANAS_LOG_FORMAT=detailed   # simple, detailed, json
```

## LLM Configuration

### LLMConfig

Configuration for language models:

```python
from core.models import LLMConfig

config = LLMConfig(
    provider="openai",           # Provider name
    model_name="gpt-4",         # Model identifier
    temperature=0.7,            # Response randomness (0-1)
    max_tokens=100,             # Maximum response length
    top_p=1.0,                  # Nucleus sampling parameter
    frequency_penalty=0.0,      # Repetition penalty
    presence_penalty=0.0,       # Topic steering
    stop=None,                  # Stop sequences
    timeout=30,                 # Request timeout
    retry_attempts=3,           # Retry failed requests
    cache_ttl=3600,            # Cache lifetime
    streaming=False            # Enable streaming
)
```

## Node Configuration

### NodeConfig

Base configuration for all nodes:

```python
from core.models import NodeConfig

config = NodeConfig(
    name="my_node",            # Node name
    description=None,          # Node description
    metadata={},               # Custom metadata
    timeout=30,                # Processing timeout
    retry_attempts=3,          # Retry attempts
    cache_enabled=True,        # Enable caching
    cache_ttl=3600,           # Cache lifetime
    async_execution=True,      # Run asynchronously
    error_handler=None        # Custom error handler
)
```

### QAConfig

Configuration for QA nodes:

```python
from core.models import QAConfig

config = QAConfig(
    system_prompt="You are...",  # System instructions
    use_rag=True,               # Enable RAG
    rag_config={                # RAG settings
        "chunk_size": 500,
        "chunk_overlap": 50,
        "max_sources": 3,
        "min_relevance": 0.7
    },
    conversation_window=10,     # Message history
    temperature=0.7,            # Model temperature
    max_tokens=500             # Max response length
)
```

### DocumentConfig

Configuration for document processing:

```python
from core.models import DocumentConfig

config = DocumentConfig(
    chunk_size=500,            # Chunk size
    chunk_overlap=50,          # Overlap between chunks
    chunk_strategy="token",    # token or character
    preserve_structure=True,   # Maintain formatting
    extract_metadata=True,     # Extract doc metadata
    supported_types=[          # Allowed file types
        "text/plain",
        "application/pdf",
        "text/markdown"
    ],
    processors={               # Custom processors
        "pdf": pdf_processor,
        "markdown": md_processor
    }
)
```

## Flow Configuration

### FlowConfig

Configuration for flows:

```python
from core.models import FlowConfig

config = FlowConfig(
    name="my_flow",             # Flow name
    description=None,           # Flow description
    parallel_execution=True,    # Allow parallel
    max_concurrency=4,         # Max parallel nodes
    timeout=60,                # Flow timeout
    error_handler=None,        # Error handler
    middleware=[],             # Flow middleware
    state_manager=None,        # State management
    visualization={            # Vis settings
        "enabled": True,
        "format": "mermaid",
        "details": "minimal"
    }
)
```

## RAG Configuration

### RAGConfig

Configuration for RAG systems:

```python
from core.models import RAGConfig

config = RAGConfig(
    chunk_size=500,            # Chunk size
    chunk_overlap=50,          # Chunk overlap
    embedding_model="text-embedding-ada-002",
    embedding_dimension=1536,   # Vector dimension
    similarity_metric="cosine", # Similarity measure
    min_relevance=0.7,         # Relevance threshold
    max_sources=3,             # Sources per query
    reranking_enabled=False,   # Enable reranking
    cache_embeddings=True,     # Cache vectors
    preprocessing=[            # Text processors
        clean_text,
        remove_duplicates
    ],
    postprocessing=[          # Result processors
        format_sources,
        add_citations
    ]
)
```

## Vector Store Configuration

### VectorStoreConfig

Configuration for vector stores:

```python
from core.models import VectorStoreConfig

config = VectorStoreConfig(
    store_type="faiss",        # Store backend
    dimension=1536,            # Vector dimension
    metric="cosine",           # Distance metric
    index_type="flat",         # Index structure
    serialization={            # Storage settings
        "enabled": True,
        "path": "vectors.idx",
        "format": "binary"
    },
    optimization={             # Performance
        "nprobe": 10,
        "quantization": None
    }
)
```

## Provider Configuration

### ProviderConfig

Configuration for LLM providers:

```python
from core.models import ProviderConfig

config = ProviderConfig(
    api_key="your-key",        # API key
    organization=None,         # Org ID
    base_url=None,            # API endpoint
    timeout=30,               # Request timeout
    rate_limit=60,            # Requests per minute
    retry={                   # Retry settings
        "attempts": 3,
        "delay": 1,
        "backoff": 2
    },
    proxy=None,               # Proxy settings
    ssl_verify=True          # SSL verification
)
```

## Tool Configuration

### ToolConfig

Configuration for tools:

```python
from core.models import ToolConfig

config = ToolConfig(
    name="my_tool",            # Tool name
    description=None,          # Tool description
    async_execution=True,      # Run async
    timeout=30,               # Execution timeout
    retry_attempts=3,         # Retry attempts
    validation={              # Input validation
        "enabled": True,
        "schema": {...}
    },
    rate_limit=None,          # Rate limiting
    cache_config={            # Caching
        "enabled": True,
        "ttl": 3600
    }
)
```

## Best Practices

1. **Environment Variables**
   - Use for sensitive data
   - Set in deployment
   - Document requirements
   - Provide examples

2. **Configuration Files**
   - Use for static config
   - Version control safe
   - Environment specific
   - Well documented

3. **Runtime Configuration**
   - Validate inputs
   - Provide defaults
   - Allow overrides
   - Type safety

4. **Security**
   - Protect API keys
   - SSL verification
   - Rate limiting
   - Input validation

5. **Performance**
   - Cache settings
   - Timeouts
   - Retries
   - Resource limits

## Notes

- Validate configurations
- Use type hints
- Document options
- Provide examples
- Consider security
- Test configurations