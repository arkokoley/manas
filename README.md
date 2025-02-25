# MAS - Multi-Agent System Framework for LLM Applications

A powerful framework for building LLM-powered applications with intelligent agents, tool integration, task decomposition, and dynamic workflows.

## Features

- 🤖 **Intelligent Agents** - Create autonomous agents with think-act-observe cycle
- 🛠️ **Tool-Using Agents** - Agents that can use tools to solve complex tasks
- 🧩 **Task Decomposition** - Break down complex tasks into manageable subtasks
- 📚 **Retrieval Augmented Generation (RAG)** - Enhance LLM responses with relevant context
- 🔄 **Dynamic Flows** - Create and modify workflows at runtime
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

Here's a simple example using a tool-using agent with Ollama:

```python
import asyncio
from mas.core.agent import Agent
from mas.core.llm import LLMNode, LLMConfig
from tool_using_agent import ToolUsingAgent, Tool

async def main():
    # Initialize tool-using agent with Ollama
    agent = ToolUsingAgent(
        name="research_assistant",
        provider="ollama",
        provider_config={
            "model": "llama2",
            "base_url": "http://localhost:11434/v1"
        }
    )
    
    # Add tools
    agent.add_tool(Tool(
        name="read_file",
        description="Read content from a file",
        func=lambda path: open(path).read()
    ))
    
    # Process a task
    result = await agent.process({
        "task": "Read README.md and summarize its contents"
    })
    
    print(result["observation"]["summary"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Components

### Tool-Using Agents

Agents that can use tools to solve complex tasks:

```python
from mas.core.agent import Agent
from tool_using_agent import ToolUsingAgent, Tool

# Create an agent
agent = ToolUsingAgent(name="file_processor")

# Add tools
agent.add_tool(Tool(
    name="read_file",
    description="Read file content",
    func=read_file_function
))

agent.add_tool(Tool(
    name="analyze_text",
    description="Analyze text content",
    func=analyze_text_function
))

# Process a complex task
result = await agent.process({
    "task": "Read config.json and analyze its structure"
})
```

### Dynamic Flows

Create and modify workflows at runtime:

```python
from mas.core.flow import Flow
from ollama_tool_test import DynamicOllamaFlow

# Create a dynamic flow
flow = DynamicOllamaFlow(model="llama2")

# Execute a complex task
result = await flow.execute_plan(
    "Research quantum computing concepts and summarize findings"
)

print(result["final_analysis"])
```

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

Create complex, dynamic workflows:

```python
from mas.core.flow import Flow
from mas.core.base import Edge

# Create a flow
flow = Flow(name="research_flow")

# Dynamically add nodes based on task requirements
for subtask in subtasks:
    node = create_node_for_subtask(subtask)
    flow.add_node(node)

# Connect nodes based on dependencies
for i in range(len(nodes) - 1):
    flow.add_edge(Edge(
        source_node=nodes[i].id,
        target_node=nodes[i + 1].id,
        name=f"step_{i}_to_{i+1}"
    ))

# Process flow
result = await flow.process({
    "input": "Your query here"
})
```

## Advanced Features

### Tool Integration

Create custom tools for agents:

```python
from tool_using_agent import Tool

# Create a custom tool
async def custom_tool(arg1: str, arg2: int) -> str:
    # Tool implementation
    return result

# Add tool to agent
agent.add_tool(Tool(
    name="custom_tool",
    description="Description of what the tool does",
    func=custom_tool
))
```

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