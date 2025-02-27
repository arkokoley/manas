---
layout: page
title: API Reference
nav_order: 4
permalink: /api/
has_children: true
has_toc: true
---

# API Reference

This section provides detailed documentation for all major components of the Manas framework.

## Core Components

### Agents and Flows

- [Agent]({{ site.baseurl }}/api/agent/) - Base agent class for creating autonomous agents
- [Flow]({{ site.baseurl }}/api/flow/) - Flow orchestration and management
- [Node]({{ site.baseurl }}/api/node/) - Base node class for all specialized nodes

### Node Types

- [QANode]({{ site.baseurl }}/api/nodes/qa_node/) - Question answering node with RAG support
- [DocumentNode]({{ site.baseurl }}/api/nodes/document_node/) - Document processing node
- [ToolNode]({{ site.baseurl }}/api/nodes/tool_node/) - Tool integration node
- [APINode]({{ site.baseurl }}/api/nodes/api_node/) - External API integration node

### LLM Integration

- [LLM]({{ site.baseurl }}/api/llm/) - Language model interface
- [LLMConfig]({{ site.baseurl }}/api/config/) - LLM configuration
- [LLMNode]({{ site.baseurl }}/api/llm_node/) - Base LLM node

### Provider System

- [BaseProvider]({{ site.baseurl }}/api/providers/base/) - Base provider class
- [OpenAI]({{ site.baseurl }}/api/providers/openai/) - OpenAI integration
- [Anthropic]({{ site.baseurl }}/api/providers/anthropic/) - Claude models
- [HuggingFace]({{ site.baseurl }}/api/providers/huggingface/) - Hugging Face models
- [Ollama]({{ site.baseurl }}/api/providers/ollama/) - Local model integration

### Vector Stores

- [VectorStore]({{ site.baseurl }}/api/vectorstores/base/) - Base vector store
- [FAISS]({{ site.baseurl }}/api/vectorstores/faiss/) - FAISS integration
- [Chroma]({{ site.baseurl }}/api/vectorstores/chroma/) - Chroma integration
- [Pinecone]({{ site.baseurl }}/api/vectorstores/pinecone/) - Pinecone integration

### RAG System

- [RAG]({{ site.baseurl }}/api/rag/) - Retrieval augmented generation
- [RAGConfig]({{ site.baseurl }}/api/rag_config/) - RAG configuration
- [RAGNode]({{ site.baseurl }}/api/rag_node/) - RAG node implementation

### Utilities

- [Tool]({{ site.baseurl }}/api/tool/) - Tool definition and management
- [Memory]({{ site.baseurl }}/api/memory/) - Memory management
- [Middleware]({{ site.baseurl }}/api/middleware/) - Provider middleware
- [Errors]({{ site.baseurl }}/api/errors/) - Error types and handling

## Using the API Reference

Each component's documentation includes:
- Description and purpose
- Configuration options
- Constructor parameters
- Available methods
- Usage examples
- Important notes

For example, see the [QANode]({{ site.baseurl }}/api/nodes/qa_node/) documentation for a comprehensive reference of a specific component.