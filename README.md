# MAS - Multi-Agent System Framework for LLM Applications

A powerful framework for building LLM-powered applications with intelligent agents, task decomposition, and Retrieval Augmented Generation (RAG).

## Features

- 🤖 **Intelligent Agents** - Create autonomous agents with think-act-observe cycle
- 🧩 **Task Decomposition** - Break down complex tasks into manageable subtasks
- 📚 **Retrieval Augmented Generation (RAG)** - Enhance LLM responses with relevant context
- 🔄 **Flow-based Architecture** - Model workflows as nested directed graphs
- 🔌 **Multiple LLM Providers** - Support for OpenAI, HuggingFace, Ollama, and more
- 💾 **Vector Store Integration** - FAISS and Chroma support for efficient similarity search
- 🧠 **Memory and Context Management** - Built-in support for maintaining conversation state
- ⚡ **Async First** - Built for high-performance async operations

## Installation

### Basic Installation

```bash
# Install using poetry
poetry install

# Or using pip
pip install .
```

### Installing with Specific Features

```bash
# Install with OpenAI support
poetry install --extras openai

# Install with HuggingFace support
poetry install --extras huggingface

# Install with Vector Store support
poetry install --extras vector-stores

# Install all features
poetry install --extras all
```

### Vector Store Dependencies

To use specific vector stores, install the corresponding extras:

```bash
# For FAISS
poetry install --extras faiss

# For Chroma
poetry install --extras chroma

# For all vector stores
poetry install --extras vector-stores
```

## Quick Start

Here's a simple example using Ollama:

```python
import asyncio
from mas.core.agent import Agent
from mas.core.llm import LLMNode, LLMConfig

async def main():
    # Initialize LLM node with Ollama
    llm = LLMNode(
        name="ollama_node",
        config=LLMConfig(
            provider="ollama",
            provider_config={
                "model": "llama2",
                "base_url": "http://localhost:11434/v1"
            }
        )
    )
    
    # Process a prompt
    result = await llm.process({
        "prompt": "Explain quantum computing in simple terms"
    })
    
    print(result["response"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Components

### Agents

Agents are the building blocks of autonomous behavior:

```python
from mas.core.agent import Agent

class MyAgent(Agent):
    async def think(self, context):
        # Process information and make decisions
        return {"decision": "next_action"}
    
    async def act(self, decision):
        # Execute actions based on decisions
        return {"result": "action_outcome"}
    
    async def observe(self, result):
        # Process results and update state
        return {"observation": "updated_state"}
```

### RAG Integration

Add context to your LLM responses:

```python
from mas.core.rag import RAGNode, RAGConfig
from mas.core.models import Document

# Initialize RAG node
rag = RAGNode(
    name="research_rag",
    config=RAGConfig(
        vectorstore_type="faiss",
        vectorstore_config={
            "dimension": 384,
            "index_type": "Cosine"
        }
    ),
    embedding_node=llm_node
)

# Add documents
await rag.add_documents([
    Document(
        content="Your document content here",
        metadata={"source": "document.txt"}
    )
])

# Query with context
result = await rag.process({
    "query": "Your question here"
})
```

### Flow Orchestration

Create complex workflows:

```python
from mas.core.flow import Flow
from mas.core.base import Edge

# Create a flow
flow = Flow(name="research_flow")

# Add nodes
flow.add_node(rag_node)
flow.add_node(llm_node)

# Connect nodes
flow.add_edge(Edge(
    source_node=rag_node.id,
    target_node=llm_node.id,
    name="rag_to_llm"
))

# Process flow
result = await flow.process({
    "input": "Your query here"
})
```

## Advanced Features

### Memory Management

Use built-in memory middleware:

```python
from mas.core.chat import MemoryMiddleware, SimpleMemory

# Create memory middleware
memory = SimpleMemory()
middleware = MemoryMiddleware(memory)

# Add to provider
provider.add_middleware(middleware)
```

### Batch Processing

Process multiple inputs efficiently:

```python
results = await flow.batch_process([
    {"input": "Query 1"},
    {"input": "Query 2"},
], batch_size=5)
```

### Streaming Support

Stream LLM responses:

```python
async for chunk in llm.stream_generate("Your prompt here"):
    print(chunk, end="", flush=True)
```

## Contributing

Contributions are welcome! Please read our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.