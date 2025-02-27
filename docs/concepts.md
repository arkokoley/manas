---
layout: default
title: Core Concepts
nav_order: 3
permalink: /concepts/
---

# Core Concepts

This page explains the fundamental concepts and architecture of the Manas framework. Understanding these concepts will help you design and build effective applications.

## Framework Architecture

Manas is built around a modular architecture that allows components to work together while maintaining separation of concerns:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Agents   │◄────┤    Flows    │────►│    Nodes    │
└─────▲───────┘     └─────────────┘     └─────▲───────┘
      │                                       │
      │             ┌─────────────┐           │
      └─────────────┤    LLMs     ├───────────┘
                    └─────▲───────┘
                          │
                    ┌─────┴───────┐
                    │  Providers  │
                    └─────────────┘
```

## Key Components

### LLMs (Language Models)

LLMs are the foundation of the framework. They represent the AI models responsible for generating text, reasoning, and decision-making.

- **Providers**: Abstractions for different LLM services (OpenAI, Anthropic, HuggingFace)
- **Model Management**: Unified interface regardless of the underlying provider
- **Tokenization**: Built-in handling of token counting and context windows

```python
# Example of LLM initialization
from core import LLM

# OpenAI model
openai_model = LLM.from_provider("openai", model_name="gpt-4")

# Anthropic model
claude_model = LLM.from_provider("anthropic", model_name="claude-3-opus")

# Local model via Ollama
local_model = LLM.from_provider("ollama", model_name="llama3")
```

### Agents

Agents are intelligent entities that use LLMs to perform tasks. They encapsulate:

- **Reasoning**: Processing information and making decisions
- **Memory**: Maintaining conversational context
- **Behavior**: Following instructions via system prompts
- **Tools**: Accessing external functionality

```python
from core import Agent

# Basic agent
agent = Agent(llm=model, system_prompt="You are a helpful assistant.")

# Tool-using agent
calculator_agent = Agent(
    llm=model,
    system_prompt="You help with math calculations.",
    tools=[calculator_tool]
)
```

### Flows

Flows orchestrate complex processes by connecting multiple nodes in a directed graph:

- **Nodes**: Components that perform specific tasks
- **Edges**: Connections that define how data flows between nodes
- **Execution**: Managed processing of inputs through the graph

```python
from core import Flow
from core.nodes import QANode

# Create a flow
flow = Flow()

# Add nodes
node1 = QANode(name="researcher", llm=model1)
node2 = QANode(name="writer", llm=model2)

# Connect nodes
flow.add_node(node1)
flow.add_node(node2)
flow.add_edge(node1, node2)

# Run the flow
result = flow.process("Research quantum computing")
```

### Nodes

Nodes are specialized components designed for specific tasks within flows:

- **QANode**: For question-answering tasks
- **DocumentNode**: For document processing and analysis
- **ToolNode**: For integrating with external tools and APIs
- **APINode**: For making API calls and processing responses

```python
from core.nodes import DocumentNode, ToolNode

# Document processing node
doc_node = DocumentNode(
    name="document_processor",
    llm=model
)

# Tool integration node
api_tool = ToolNode(
    name="weather_api",
    tool=weather_api_tool
)
```

### RAG (Retrieval-Augmented Generation)

RAG combines LLMs with information retrieval to enhance responses with external knowledge:

- **Vector Stores**: Efficient storage and retrieval of embeddings
- **Document Processing**: Conversion, chunking, and embedding of data
- **Retrieval**: Finding relevant information based on semantic similarity
- **Generation**: Producing responses informed by retrieved context

```python
from core import RAG
from core.vectorstores import FaissVectorStore

# Create a vector store
vector_store = FaissVectorStore(dimension=1536)

# Initialize RAG
rag_system = RAG(
    llm=model,
    vector_store=vector_store
)

# Add documents
rag_system.add_file("knowledge_base.pdf")

# Query
response = rag_system.query("What are the key findings?")
```

## Design Principles

Manas is built on these core design principles:

### 1. Modularity

Components can be used independently or combined for complex applications.

### 2. Extensibility

The framework is designed to be extended with new capabilities and integrations.

### 3. Provider Agnosticism

Applications can work with multiple LLM providers seamlessly.

### 4. Flow-Based Architecture

Complex processes are represented as flows of interconnected nodes.

### 5. Strong Typing

Type hints throughout the codebase ensure reliability and enable IDE support.

## Next Steps

Now that you understand the core concepts, explore:

- [API Reference](/api/) for detailed documentation
- [Examples](/examples/) for practical applications
- [Project Structure](/structure/) to understand the codebase organization