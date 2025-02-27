---
layout: default
title: Manas - A Framework for LLM-Powered Applications
nav_order: 1
permalink: /
---

# Manas

Manas is a powerful, flexible framework for building LLM-powered applications with intelligent agents, task decomposition, and Retrieval-Augmented Generation (RAG). The framework is designed to help developers create complex AI applications by orchestrating multiple AI agents working together to solve problems.

## Why Manas?

In the rapidly evolving landscape of LLM applications, developers face several challenges:

- **Complexity Management**: Coordinating multiple AI components and tasks
- **Model Flexibility**: Supporting various LLM providers and models
- **RAG Integration**: Integrating knowledge bases effectively
- **Scalability**: Building systems that can grow with your needs
- **Tooling**: Connecting AI with external tools and APIs

Manas addresses these challenges with a unified framework that lets you define multi-agent systems, manage workflows, integrate knowledge bases, and connect to external tools—all with a clean, consistent API.

## Key Features

- **Multi-Agent Orchestration**: Define and coordinate multiple agents with different roles and capabilities
- **Flexible LLM Integration**: Support for OpenAI, Anthropic, HuggingFace, and more
- **Built-in RAG**: First-class support for Retrieval-Augmented Generation with various vector stores
- **Flow-Based Architecture**: Create complex workflows with directed graphs
- **Tool Integration**: Connect agents to external tools and APIs
- **Extensible Design**: Easily add new capabilities and integrations

## Documentation Sections

### Getting Started
- [Installation & Setup]({{ site.baseurl }}/getting-started/#installation)
- [Quick Start Guide]({{ site.baseurl }}/getting-started/#quick-start)
- [Basic Concepts]({{ site.baseurl }}/getting-started/#basic-concepts)
- [First Application]({{ site.baseurl }}/getting-started/#your-first-manas-application)

### Core Documentation
- [Architecture Overview]({{ site.baseurl }}/concepts/#framework-architecture)
- [Key Components]({{ site.baseurl }}/concepts/#key-components)
- [Design Principles]({{ site.baseurl }}/concepts/#design-principles)
- [Best Practices]({{ site.baseurl }}/concepts/#best-practices)

### Components
- [Agents]({{ site.baseurl }}/api/agent/)
- [Flows]({{ site.baseurl }}/api/flow/)
- [Nodes]({{ site.baseurl }}/api/nodes/)
- [Providers]({{ site.baseurl }}/api/providers/)
- [Vector Stores]({{ site.baseurl }}/api/vectorstores/)

### Features
- [RAG Implementation]({{ site.baseurl }}/api/rag/)
- [Tool Integration]({{ site.baseurl }}/examples/tool-usage/)
- [Provider Architecture]({{ site.baseurl }}/llm-integration/)
- [Middleware System]({{ site.baseurl }}/concepts/#middleware-system)

### Examples & Tutorials
- [Basic Examples]({{ site.baseurl }}/examples/#basic-examples)
- [Flow Examples]({{ site.baseurl }}/examples/#flow-examples)
- [RAG Examples]({{ site.baseurl }}/examples/#rag-examples)
- [Advanced Examples]({{ site.baseurl }}/examples/#advanced-examples)

### Advanced Topics
- [Performance & Benchmarking]({{ site.baseurl }}/benchmarking/)
- [LLM Integration Guide]({{ site.baseurl }}/llm-integration/)
- [Project Structure]({{ site.baseurl }}/structure/)
- [Custom Components]({{ site.baseurl }}/api/custom/)

### Reference
- [API Reference]({{ site.baseurl }}/api/)
- [Configuration Options]({{ site.baseurl }}/api/config/)
- [Error Handling]({{ site.baseurl }}/api/errors/)
- [Utility Functions]({{ site.baseurl }}/api/utils/)

### Development
- [Contributing Guide]({{ site.baseurl }}/contributing/)
- [Development Setup]({{ site.baseurl }}/contributing/#development-setup)
- [Testing Guidelines]({{ site.baseurl }}/contributing/#testing)
- [Documentation Guide]({{ site.baseurl }}/contributing/#documentation)

### Support
- [FAQ]({{ site.baseurl }}/faq/)
- [Troubleshooting]({{ site.baseurl }}/faq/#troubleshooting)
- [Known Issues]({{ site.baseurl }}/faq/#known-issues)
- [Getting Help]({{ site.baseurl }}/contributing/#getting-help)

## Getting Started

Here's a simple example to get you started:

```python
from core import LLM, Agent

# Initialize a model
model = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    api_key="your-api-key"  # Or use environment variable
)

# Create an agent
agent = Agent(llm=model)

# Generate a response
response = agent.generate("Explain what the Manas framework does")
print(response)
```

## Installation

```bash
pip install manas-ai
```

For specific features:

```bash
# OpenAI support
pip install "manas-ai[openai]"

# Full installation with CPU support
pip install "manas-ai[all-cpu]"

# Full installation with GPU support
pip install "manas-ai[all-gpu]"
```

## Next Steps

1. Follow the [Getting Started Guide]({{ site.baseurl }}/getting-started/) for a complete introduction
2. Explore [Core Concepts]({{ site.baseurl }}/concepts/) to understand the architecture
3. Check out [Examples]({{ site.baseurl }}/examples/) for practical use cases
4. Read the [API Reference]({{ site.baseurl }}/api/) for detailed documentation

## Community

- [GitHub Repository](https://github.com/arkokoley/manas)
- [Issue Tracker](https://github.com/arkokoley/manas/issues)
- [Contributing Guide]({{ site.baseurl }}/contributing/)

## License

Manas is open-source software licensed under the MIT license.