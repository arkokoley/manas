---
layout: page
title: Getting Started with Manas
nav_order: 2
permalink: /getting-started/
has_toc: true
---

# Getting Started with Manas

This guide will walk you through installing Manas, setting up your first project, and building a simple LLM-powered application.

## Prerequisites

Before you begin, ensure you have:

- Python 3.11 or newer
- pip (Python package installer)
- (Optional) Poetry for dependency management

## Installation

### Using pip

```bash
pip install manas-ai
```

### Using Poetry

```bash
poetry add manas-ai
```

### Optional Features

Manas supports different LLM providers and vector stores through extras:

```bash
# For OpenAI support
pip install "manas-ai[openai]"

# For Anthropic support
pip install "manas-ai[anthropic]"

# For HuggingFace models
pip install "manas-ai[huggingface]"

# For all supported providers
pip install "manas-ai[all-cpu]"

# With GPU support for vector operations
pip install "manas-ai[all-gpu]"
```

## Your First Manas Application

Let's build a simple question-answering application using Manas:

### 1. Setting up the LLM

```python
from core import LLM

# Initialize a model (replace with your API key)
openai_model = LLM.from_provider(
    "openai", 
    model_name="gpt-4",
    api_key="your-api-key-here"  # Or set OPENAI_API_KEY environment variable
)
```

### 2. Creating an Agent

```python
from core import Agent

# Create a basic agent
agent = Agent(llm=openai_model)
```

### 3. Generating Responses

```python
# Simple generation
response = agent.generate("Explain the concept of reinforcement learning in under 100 words.")
print(response)
```

### 4. Creating a Multi-Node Flow

Let's create a simple flow with two agents that collaborate:

```python
from core import Flow
from core.nodes import QANode

# Create two specialized nodes
research_node = QANode(
    name="researcher",
    llm=openai_model,
    system_prompt="You are a scientific researcher who provides in-depth analysis."
)

summarizer_node = QANode(
    name="summarizer", 
    llm=openai_model,
    system_prompt="You summarize complex information in simple terms for general audience."
)

# Create a flow
flow = Flow()
flow.add_node(research_node)
flow.add_node(summarizer_node)
flow.add_edge(research_node, summarizer_node)

# Process a query through the flow
result = flow.process("What are the implications of quantum computing for cryptography?")

# The result will contain the final output from the summarizer
print(result)
```

## Adding RAG Capabilities

Let's enhance our application with RAG (Retrieval-Augmented Generation):

```python
from core import RAG
from core.vectorstores import FaissVectorStore

# Create a vector store
vector_store = FaissVectorStore(dimension=1536)  # OpenAI embeddings dimension

# Add some documents
documents = [
    "Quantum computing uses quantum bits or qubits which can represent 0, 1, or both simultaneously.",
    "Modern cryptography relies on mathematical problems that are difficult for classical computers.",
    "Shor's algorithm, when run on quantum computers, can break RSA encryption efficiently."
]

# Create a RAG system
rag_system = RAG(
    llm=openai_model,
    vector_store=vector_store
)

# Add documents to the RAG system
rag_system.add_texts(documents)

# Query the RAG system
answer = rag_system.query("How does quantum computing affect encryption?")
print(answer)
```

## Next Steps

Now that you've built your first Manas application, explore:

- [Core Concepts]({{ site.baseurl }}/concepts/) to understand the architecture
- [API Reference]({{ site.baseurl }}/api/) for detailed documentation
- [Examples]({{ site.baseurl }}/examples/) for more complex use cases
- [Project Structure]({{ site.baseurl }}/structure/) to learn about the codebase organization

For any issues or questions, check the [FAQ]({{ site.baseurl }}/faq/) or visit our [GitHub repository](https://github.com/arkokoley/manas).