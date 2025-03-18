# Manas - Multi-Agent System Framework for LLM Applications

A robust, modular, and extensible framework for building LLM-powered applications with intelligent agents, tool integration, task decomposition, and dynamic workflows.

[![PyPI version](https://badge.fury.io/py/manas-ai.svg)](https://badge.fury.io/py/manas-ai)
[![Python Version](https://img.shields.io/pypi/pyversions/manas-ai.svg)](https://pypi.org/project/manas-ai/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://arkokoley.github.io/manas/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🤖 **Intelligent Agents** - Create autonomous agents with think-act-observe cycle and built-in state management
- 🛠️ **Tool System** - Flexible registry for tools with automatic parameter discovery and documentation
- 🧩 **Task Decomposition** - Break down complex tasks into manageable subtasks
- 📚 **Retrieval Augmented Generation (RAG)** - Enhance LLM responses with relevant context and document chunking
- 🔄 **Dynamic Flows** - Create and modify workflows with visualization and dependency management
- 🔌 **Provider Architecture** - Modular support for OpenAI, Anthropic, HuggingFace, Ollama, and custom providers
- 💾 **Vector Store Integration** - FAISS, Chroma, Pinecone support with consistent interfaces
- 🧠 **Memory and Middleware** - Built-in memory middleware and extensible middleware architecture
- ⚡ **Async First** - Fully asynchronous architecture with proper error handling
- ✅ **Formally Verified** - Core flow execution logic validated through formal verification

## Installation

### Basic Installation

```bash
# Install using pip
pip install manas-ai

# Install using poetry
poetry add manas-ai
```

### Installing with Specific Features

```bash
# Install with OpenAI support
pip install "manas-ai[openai]"

# Install with Anthropic support
pip install "manas-ai[anthropic]"

# Install with HuggingFace support
pip install "manas-ai[huggingface]"

# Install with Vector Store support
pip install "manas-ai[vector-stores]"

# Install all features with CPU support
pip install "manas-ai[all-cpu]"

# Install all features with GPU support
pip install "manas-ai[all-gpu]"
```

## Quick Start

Here's a simple example of a question-answering agent:

```python
import os
from core import LLM, Agent

# Initialize a model with your API key
model = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create an agent
agent = Agent(llm=model, system_prompt="You are a helpful assistant.")

# Generate a response
response = agent.generate("What is the capital of France?")
print(response)
```

## Creating a Multi-Agent Flow

Here's how to create a flow with multiple specialized agents:

```python
from core import Flow
from manas_ai.nodes import QANode

# Initialize a model
model = LLM.from_provider("openai", model_name="gpt-4")

# Create specialized nodes
researcher = QANode(
    name="researcher",
    llm=model,
    system_prompt="You are an expert researcher who provides factual information."
)

writer = QANode(
    name="writer",
    llm=model,
    system_prompt="You are a skilled writer who creates engaging content."
)

# Create a flow
flow = Flow()
flow.add_node(researcher)
flow.add_node(writer)
flow.add_edge(researcher, writer)

# Process a query
result = flow.process("Explain quantum computing.")
print(result)
```

## Documentation

For complete documentation, visit [https://arkokoley.github.io/manas/](https://arkokoley.github.io/manas/).

- [Getting Started](https://arkokoley.github.io/manas/getting-started/)
- [Core Concepts](https://arkokoley.github.io/manas/concepts/)
- [API Reference](https://arkokoley.github.io/manas/api/)
- [Examples](https://arkokoley.github.io/manas/examples/)
- [FAQ](https://arkokoley.github.io/manas/faq/)

## Core Components

### Agent System

Agents in Manas follow a think-act-observe cycle with improved state management:

```python
from core import Agent, LLM

# Create an agent with tools
agent = Agent(
    llm=model,
    system_prompt="You are an agent with access to tools.",
    tools=[calculator_tool, search_tool]
)

# Process a query
response = agent.generate("Calculate 125 * 37 and find information about Mars.")
```

### RAG Integration

Manas provides robust support for Retrieval-Augmented Generation:

```python
from core import RAG
from manas_ai.vectorstores import FaissVectorStore

# Create a vector store
vector_store = FaissVectorStore(dimension=1536)

# Initialize RAG
rag_system = RAG(
    llm=model,
    vector_store=vector_store
)

# Add documents
rag_system.add_file("knowledge_base.pdf")

# Query the RAG system
response = rag_system.query("What are the key findings in the document?")
```

### Flow Orchestration

Create complex workflows with multiple specialized nodes:

```python
from core import Flow
from manas_ai.nodes import QANode, ToolNode, DocumentNode

# Create nodes
nodes = [
    QANode(name="planner", llm=model),
    DocumentNode(name="document_processor", llm=model),
    ToolNode(name="calculator", tool=calculator_tool)
]

# Create flow
flow = Flow()
for node in nodes:
    flow.add_node(node)

# Connect nodes
flow.add_edge(nodes[0], nodes[1])
flow.add_edge(nodes[0], nodes[2])
flow.add_edge(nodes[1], nodes[0])
flow.add_edge(nodes[2], nodes[0])

# Process flow
result = flow.process("Analyze this document and calculate the totals.")
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](https://arkokoley.github.io/manas/contributing/) for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.