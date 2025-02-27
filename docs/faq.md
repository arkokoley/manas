---
layout: default
title: FAQ & Troubleshooting
nav_order: 6
permalink: /faq/
has_toc: true
---

# Frequently Asked Questions

This page addresses common questions and issues you might encounter when working with the Manas framework.

## General Questions

### What is Manas?

Manas is a framework for building LLM-powered applications with intelligent agents, task decomposition, and retrieval-augmented generation (RAG). It provides a unified API for working with different LLM providers, orchestrating multi-agent workflows, and integrating with vector stores.

### Is Manas free to use?

Yes, Manas itself is open source and free to use under the MIT license. However, you are responsible for any costs associated with the LLM providers you choose to use (e.g., OpenAI, Anthropic).

### What Python versions are supported?

Manas requires Python 3.11 or newer.

### How does Manas differ from other LLM frameworks?

Manas focuses on multi-agent orchestration and flow-based architectures, making it particularly well-suited for complex AI applications that involve multiple specialized agents working together.

## Installation Issues

### Why am I seeing dependency conflicts when installing Manas?

This might happen if you have incompatible versions of libraries already installed. Try using a clean virtual environment:

```bash
python -m venv manas-env
source manas-env/bin/activate  # On Windows: manas-env\Scripts\activate
pip install manas-ai
```

### Can I use Manas with GPU acceleration?

Yes, for features that support GPU acceleration (like FAISS vector storage), install with the GPU extras:

```bash
pip install "manas-ai[all-gpu]"
```

Make sure you have the appropriate CUDA drivers installed for your GPU.

## API Keys and Configuration

### How do I set up API keys for different providers?

You can either pass API keys directly when initializing a provider or set them as environment variables:

```python
# Direct passing
llm = LLM.from_provider("openai", api_key="your-api-key")

# Or use environment variables (recommended)
# Set OPENAI_API_KEY in your environment
llm = LLM.from_provider("openai")
```

### How can I configure default parameters for models?

You can set default parameters when initializing an LLM:

```python
llm = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=500
)
```

## Using Multiple Models

### Can I mix different providers in the same application?

Yes, you can create multiple LLM instances using different providers and use them in the same application:

```python
openai_model = LLM.from_provider("openai", model_name="gpt-4")
anthropic_model = LLM.from_provider("anthropic", model_name="claude-3-opus")

research_agent = Agent(llm=openai_model)
writing_agent = Agent(llm=anthropic_model)
```

### How do I choose the right model for each task?

Consider these factors:
- Task complexity (use more capable models for complex reasoning)
- Speed requirements (smaller models are faster)
- Cost considerations (larger models are more expensive)
- Specialization (some models perform better at specific tasks)

## Flow Management

### How can I debug a complex flow?

Flows support detailed logging to help trace the execution path:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your flow will now produce detailed logs
flow = Flow(verbose=True)
```

You can also inspect the intermediate outputs of each node during execution.

### Can flows execute in parallel?

Yes, you can configure flows to execute nodes in parallel when their dependencies allow:

```python
flow = Flow(parallel_execution=True)
```

Note that this requires proper handling of concurrency and might be affected by rate limits of LLM providers.

## RAG Implementation

### What document formats can Manas process for RAG?

Manas supports various document formats including:
- Plain text (.txt)
- PDF files (.pdf) 
- Markdown (.md)
- Word documents (.docx)
- HTML (.html)

### How can I optimize vector search for large collections?

For large document collections:
1. Choose an appropriate chunking strategy
2. Use a vector store with efficient indexing (like FAISS)
3. Consider using dimensionality reduction techniques
4. Implement filtering to narrow search space

```python
# Example of optimized vector store setup
vector_store = FaissVectorStore(
    dimension=1536,
    index_type="IVF100,Flat",  # IVF index for faster search
    metric="l2"
)
```

## Performance Optimization

### How can I reduce token usage?

To optimize token usage:
1. Use smaller context windows when possible
2. Implement effective chunking strategies for documents
3. Use summarization for long conversations
4. Choose models with better performance/token ratio

### My application is slow. How can I speed it up?

Performance improvements:
1. Use caching for common queries
2. Implement batch processing for embeddings
3. Use smaller, faster models for less complex tasks
4. Enable parallel processing where appropriate
5. Optimize prompt design to reduce token usage

## Error Handling

### How should I handle API rate limiting?

Implement exponential backoff and retry logic:

```python
from time import sleep
import random

def call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.random()
                print(f"Rate limit hit. Retrying in {sleep_time:.2f} seconds...")
                sleep(sleep_time)
            else:
                raise
```

### What should I do if I encounter "context length exceeded" errors?

1. Reduce the size of your prompts
2. Use more efficient chunking strategies
3. Implement summarization of previous context
4. Use models with larger context windows
5. Implement a sliding window approach for processing long documents

## Contributing to Manas

### How can I contribute to the project?

We welcome contributions! Check out our [Contributing Guide](/contributing/) for details on:
1. Setting up a development environment
2. Finding issues to work on
3. Submitting pull requests
4. Coding standards

### How do I report bugs or request features?

You can report bugs or request features by opening an issue on our [GitHub repository](https://github.com/arkokoley/manas/issues).