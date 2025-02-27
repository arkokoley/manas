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

## Who Is Manas For?

Manas is designed for:

- **AI Application Developers**: Building complex LLM-powered applications
- **Research Engineers**: Experimenting with multi-agent systems
- **Product Teams**: Creating production-ready AI features
- **LLM Enthusiasts**: Learning about agent-based AI architecture

## Getting Started

Ready to build with Manas? Start with our [Getting Started Guide](/getting-started/) or explore our [Examples](/examples/).

```python
# A simple Manas example
from core import LLM, Agent

# Initialize a model
model = LLM.from_provider("openai", model_name="gpt-4")

# Create an agent
agent = Agent(llm=model)

# Have a conversation
response = agent.generate("Explain what the Manas framework does")
print(response)
```

## Installation

```bash
pip install manas-ai
```

## License

Manas is open-source software licensed under the MIT license.