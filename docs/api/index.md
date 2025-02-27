---
layout: default
title: API Reference
nav_order: 4
permalink: /api/
has_children: true
---

# API Reference

This section provides detailed documentation for all major components of the Manas framework.

## Core Components

### Agents and Flows

- [Agent](/api/agent/) - Base agent class for creating autonomous agents
- [Flow](/api/flow/) - Flow orchestration and management
- [Node](/api/node/) - Base node class for all specialized nodes

### Node Types

- [QANode](/api/nodes/qa_node/) - Question answering node with RAG support
- [DocumentNode](/api/nodes/document_node/) - Document processing node
- [ToolNode](/api/nodes/tool_node/) - Tool integration node
- [APINode](/api/nodes/api_node/) - External API integration node

### LLM Integration

- [LLM](/api/llm/) - Language model interface
- [LLMConfig](/api/llm_config/) - LLM configuration
- [LLMNode](/api/llm_node/) - Base LLM node

### Provider System

- [BaseProvider](/api/providers/base/) - Base provider class
- [OpenAI](/api/providers/openai/) - OpenAI integration
- [Anthropic](/api/providers/anthropic/) - Claude models
- [HuggingFace](/api/providers/huggingface/) - Hugging Face models
- [Ollama](/api/providers/ollama/) - Local model integration

### Vector Stores

- [VectorStore](/api/vectorstores/base/) - Base vector store
- [FAISS](/api/vectorstores/faiss/) - FAISS integration
- [Chroma](/api/vectorstores/chroma/) - Chroma integration
- [Pinecone](/api/vectorstores/pinecone/) - Pinecone integration

### RAG System

- [RAG](/api/rag/) - Retrieval augmented generation
- [RAGConfig](/api/rag_config/) - RAG configuration
- [RAGNode](/api/rag_node/) - RAG node implementation

### Utilities

- [Tool](/api/tool/) - Tool definition and management
- [Memory](/api/memory/) - Memory management
- [Middleware](/api/middleware/) - Provider middleware
- [Errors](/api/errors/) - Error types and handling

## Using the API Reference

Each component's documentation includes:
- Description and purpose
- Configuration options
- Constructor parameters
- Available methods
- Usage examples
- Important notes

For example, see the [QANode](/api/nodes/qa_node/) documentation for a comprehensive reference of a specific component.