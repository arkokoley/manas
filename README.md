# MAS - Multi-Agent System Framework for LLM Applications

A robust, modular, and extensible framework for building LLM-powered applications with intelligent agents, tool integration, task decomposition, and dynamic workflows.

## Features

- 🤖 **Intelligent Agents** - Create autonomous agents with think-act-observe cycle and built-in state management
- 🛠️ **Tool System** - Flexible registry for tools with automatic parameter discovery and documentation
- 🧩 **Task Decomposition** - Break down complex tasks into manageable subtasks
- 📚 **Retrieval Augmented Generation (RAG)** - Enhance LLM responses with relevant context and document chunking
- 🔄 **Dynamic Flows** - Create and modify workflows with visualization and dependency management
- 🔌 **Provider Architecture** - Modular support for OpenAI, HuggingFace, Ollama, and custom providers
- 💾 **Vector Store Integration** - FAISS and Chroma support with consistent interfaces
- 🧠 **Memory and Middleware** - Built-in memory middleware and extensible middleware architecture
- ⚡ **Async First** - Fully asynchronous architecture with proper error handling
- ✅ **Formally Verified** - Core flow execution logic validated through formal verification

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
from mas.core.agent import Agent, AgentRegistry, Tool
from mas.core.llm import LLMNode, LLMConfig

# Define a custom agent
@AgentRegistry.register
class ResearchAgent(Agent):
    async def think(self, context):
        # Process input and plan next steps
        query = context.get("query", "")
        return {"plan": f"Research information about {query}"}
    
    async def act(self, decision):
        # Execute the research plan
        plan = decision.get("plan", "")
        result = await self.tools["search"](plan)
        return {"search_results": result}
    
    async def observe(self, result):
        # Process and summarize the results
        search_results = result.get("search_results", "")
        return {"summary": f"Based on research: {search_results}"}

async def main():
    # Create LLM configuration
    config = LLMConfig(
        provider_name="ollama",
        provider_config={
            "model": "llama3",
            "base_url": "http://localhost:11434/v1"
        },
        temperature=0.7
    )
    
    # Create LLM node for the agent
    llm_node = LLMNode("llm", config)
    
    # Create agent
    agent = ResearchAgent("researcher")
    
    # Register tools
    @AgentRegistry.register_tool(name="search", description="Search for information")
    async def search(query):
        # In a real application, this would perform an actual search
        return f"Search results for {query}"
    
    # Add tool to agent
    agent.add_capability("search")
    
    # Process a query
    result = await agent.process({"query": "quantum computing"})
    print(result["observation"]["summary"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Components

### Agent System

Agents follow a think-act-observe cycle with improved state management:

```python
from mas.core.agent import Agent, AgentRegistry, Tool

@AgentRegistry.register
class AnalysisAgent(Agent):
    async def think(self, context):
        # Process information and make decisions
        return {"decision": "analyze_data"}
    
    async def act(self, decision):
        # Execute actions based on decisions
        if decision["decision"] == "analyze_data":
            # Use registered tools
            result = await self.tools["data_analysis"](context["data"])
            return {"result": result}
    
    async def observe(self, result):
        # Process results and update state
        self.set_state({"last_analysis": result["result"]})
        return {"observation": "Analysis complete"}

# Register a tool
@AgentRegistry.register_tool(name="data_analysis", description="Analyze data")
async def analyze_data(data):
    # Analysis implementation
    return {"insights": "Data analysis results"}
```

### RAG Integration

Enhanced RAG with document chunking and reranking:

```python
from mas.core.rag import RAGNode, RAGConfig, DocumentLoader, chunk_document
from mas.core.models import Document
from mas.core.llm import LLMNode, LLMConfig
from mas.core.vectorstores.factory import create_vectorstore

# Create embedding model
embedding_config = LLMConfig(
    provider_name="openai",
    provider_config={"model": "text-embedding-ada-002"},
)
embedding_node = LLMNode("embeddings", embedding_config)

# Create LLM for generation
llm_config = LLMConfig(
    provider_name="openai",
    provider_config={"model": "gpt-4o-mini"},
)
llm_node = LLMNode("llm", llm_config)

# Create document loader with chunking preprocessor
document_loader = DocumentLoader(
    name="loader",
    preprocessors=[chunk_document(max_length=500, overlap=50)]
)

# Initialize RAG node
rag = RAGNode(
    name="research_rag",
    config=RAGConfig(
        vectorstore_type="faiss",
        vectorstore_config={
            "dimension": 1536,
            "similarity_metric": "cosine"
        },
        num_results=5,
        rerank_results=True
    ),
    embedding_node=embedding_node,
    llm_node=llm_node
)

# Load and add documents
documents = await document_loader.process({
    "documents": ["Document 1 content", "Document 2 content"]
})
await rag.add_documents(documents["documents"])

# Query with RAG
result = await rag.process({
    "query": "What are the key concepts?"
})

print(result["response"])
```

### Flow Orchestration

Create complex, dynamic workflows with dependency management and validation:

```python
from mas.core.flow import Flow
from mas.core.llm import LLMNode, LLMConfig
from mas.core.rag import RAGNode, RAGConfig

# Create a flow
flow = Flow(name="research_flow", description="Research and analysis flow")

# Add nodes
reader = DocumentLoader(name="reader")
flow.add_node(reader)

embedder = LLMNode("embedder", embedding_config)
flow.add_node(embedder)

rag = RAGNode(name="rag", config=rag_config, embedding_node=embedder)
flow.add_node(rag)

analyzer = LLMNode("analyzer", llm_config)
flow.add_node(analyzer)

# Connect nodes by name (simplified API)
flow.connect_nodes("reader", "rag", "docs_to_rag")
flow.connect_nodes("rag", "analyzer", "context_to_analyzer")

# Process flow
result = await flow.process({
    "documents": ["Document 1", "Document 2"],
    "query": "Analyze the trend in these documents"
})

# Visualize flow
flow_viz = flow.visualize()
```

## Advanced Features

### Middleware System

Use middleware to enhance provider capabilities:

```python
from mas.core.chat import MemoryMiddleware, SimpleMemory
from mas.core.providers.middleware import MiddlewareProvider

# Create memory and middleware
memory = SimpleMemory()
middleware = MemoryMiddleware(memory)

# Add to provider
provider.add_middleware(middleware)

# Configure memory operations in message metadata
message = Message(
    role="user",
    content="Recall what I told you about preferences",
    metadata={
        "requires_memory": ["user_preferences"]
    }
)

# Memory will be automatically injected into prompt
```

### Provider Registration

Create and register custom LLM providers:

```python
from mas.core.providers.factory import register_provider
from mas.core.providers.base import BaseLLMProvider

@register_provider("custom")
class CustomProvider(BaseLLMProvider):
    """Custom provider implementation."""
    provider_name = "custom"
    supports_streaming = True
    supports_embeddings = True
    default_embedding_dimension = 768
    
    async def initialize(self):
        # Initialize resources
        self._client = await setup_client(self.config)
        await super().initialize()  # Set _initialized flag
    
    async def cleanup(self):
        # Clean up resources
        await self._client.close()
        await super().cleanup()  # Clear _initialized flag
    
    async def generate(self, prompt, temperature=0.7, **kwargs):
        await self._ensure_initialized()
        # Generate completion
        response = await self._client.complete(prompt, temperature)
        return response.text
    
    async def stream_generate(self, prompt, **kwargs):
        await self._ensure_initialized()
        # Stream completion
        async for chunk in self._client.stream(prompt):
            yield chunk
    
    async def embed(self, text):
        await self._ensure_initialized()
        # Generate embeddings
        embedding = await self._client.embed(text)
        return embedding
```

### Vector Store Registration

Create and register custom vector stores:

```python
from mas.core.vectorstores.factory import register_vectorstore
from mas.core.vectorstores.base import VectorStoreProvider
from mas.core.models import Document
from mas.core.llm import LLMNode

@register_vectorstore("custom_store")
class CustomVectorStore(VectorStoreProvider):
    """Custom vector store implementation."""
    
    def __init__(self, config: Dict[str, Any], embedding_node: LLMNode):
        super().__init__(config, embedding_node)
        # Initialize your vector store-specific attributes
        
    async def initialize(self):
        # Initialize the vector store
        # Your initialization logic here
        await super().initialize()
    
    async def cleanup(self):
        # Cleanup resources
        # Your cleanup logic here
        await super().cleanup()
    
    async def add_documents(self, documents: List[Document]) -> int:
        await self._ensure_initialized()
        # Add documents to the vector store
        # Return number of documents added
        return len(documents)
    
    async def similarity_search(self, query: str, k: int = 4,
                              filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        await self._ensure_initialized()
        # Perform similarity search
        # Return list of matching documents
        
    async def delete(self, ids_or_filter: Any) -> int:
        await self._ensure_initialized()
        # Delete documents
        # Return number of documents deleted
```

### Batch and Stream Processing

Efficient batch processing and streaming:

```python
# Batch processing
results = await flow.batch_process([
    {"query": "Question 1", "documents": ["Doc A", "Doc B"]},
    {"query": "Question 2", "documents": ["Doc C", "Doc D"]},
], batch_size=5)

# Streaming
async for chunk in llm_node.stream_generate("Explain quantum computing"):
    print(chunk, end="", flush=True)
```

## Formal Verification

The flow execution engine has been formally verified to guarantee several critical properties:

- **✓ acyclic**: All flows are guaranteed to be acyclic, preventing infinite execution loops
- **✓ reachability**: All nodes in the flow can be reached from input nodes
- **✓ deterministic_execution**: Flow execution is deterministic given the same inputs
- **✓ resource_safety**: All resources are properly initialized and cleaned up
- **✓ parallel_safety**: Parallel node execution is safe and properly synchronized

## Error Handling

Improved error handling with custom exceptions:

```python
try:
    result = await flow.process(inputs)
except FlowExecutionError as e:
    print(f"Error in flow: {e}")
    print(f"Failed node: {e.node_name}")
    print(f"Details: {e.details}")
except RAGError as e:
    print(f"RAG error: {e}")
except AgentError as e:
    print(f"Agent error: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Contributing

Contributions are welcome! Please read our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.