---
layout: default
title: Examples
nav_order: 5
permalink: /examples/
has_children: true
has_toc: false
---

# Examples

This section contains a variety of practical examples showing how to build applications with Manas. Each example demonstrates specific features of the framework and includes complete code that you can run and adapt.

## Getting Started Examples

- [RAG Usage]({{ site.baseurl }}/examples/rag-usage)
  Learn how to use the Retrieval-Augmented Generation (RAG) capabilities to enhance LLM outputs with relevant information.

- [Tool Usage]({{ site.baseurl }}/examples/tool-usage)
  Build an agent that can use external tools to solve problems and answer questions by interacting with APIs and services.

## Advanced Examples

- [Research Assistant]({{ site.baseurl }}/examples/research-assistant)
  Implement a multi-agent system that can research topics, synthesize information, and generate reports.

- [Knowledge Base QA]({{ site.baseurl }}/examples/knowledge-base-qa)
  Create a question-answering system over your own documents and knowledge bases.

## By Feature

### Agent Examples

- [Simple Agent]({{ site.baseurl }}/examples/simple-agent) - Creating and configuring basic agents
- [Tool-Using Agent]({{ site.baseurl }}/examples/tool-using-agent) - Building agents that use external tools
- [Memory-Enhanced Agent]({{ site.baseurl }}/examples/memory-agent) - Agents with conversation memory

### Flow Examples

- [Two-Node Flow]({{ site.baseurl }}/examples/two-node-flow) - Simple sequential flow with two nodes
- [Research Flow]({{ site.baseurl }}/examples/research-flow) - Complex flow for research tasks
- [Feedback Flow]({{ site.baseurl }}/examples/feedback-flow) - Flow with feedback loops for refinement

### RAG Examples

- [Basic RAG]({{ site.baseurl }}/examples/basic-rag) - Simple retrieval-augmented generation
- [Chunking Strategies]({{ site.baseurl }}/examples/chunking-strategies) - Different document chunking approaches
- [Multi-Vector RAG]({{ site.baseurl }}/examples/multi-vector-rag) - Advanced RAG with multiple vector indices

## Running the Examples

Most examples can be run with:

```bash
# Clone the repository
git clone https://github.com/arkokoley/manas.git
cd manas

# Install dependencies
pip install -e ".[all-cpu]"  # or .[all-gpu] for GPU support

# Run an example
python examples/simple_agent.py
```

Be sure to set the required environment variables for API keys:

```bash
export OPENAI_API_KEY=your_api_key_here
export ANTHROPIC_API_KEY=your_api_key_here
```

## Next Steps

After exploring these examples, check out:

- [Getting Started Guide]({{ site.baseurl }}/getting-started/) for a proper introduction
- [Core Concepts]({{ site.baseurl }}/concepts/) to understand the framework architecture
- [API Reference]({{ site.baseurl }}/api/) for detailed documentation