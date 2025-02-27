---
layout: default
title: LLM Integration Guide
nav_order: 8
permalink: /llm-integration/
---

# LLM Integration Guide

This guide explains how to effectively integrate and work with different Language Learning Models (LLMs) in the Manas framework.

## Supported LLM Providers

Manas supports multiple LLM providers out of the box:

- **OpenAI** - GPT-3.5, GPT-4, and text embeddings
- **Anthropic** - Claude models
- **HuggingFace** - Open source models and Inference API
- **Ollama** - Local models including Llama, Mistral, and more

## Provider Configuration

### OpenAI

```python
from core import LLM

# Basic configuration
openai_model = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    api_key="your-api-key",  # Or use OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=500
)

# With additional parameters
openai_model = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    api_key="your-api-key",
    model_config={
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
)
```

### Anthropic

```python
# Claude configuration
claude_model = LLM.from_provider(
    "anthropic",
    model_name="claude-3-opus",
    api_key="your-api-key",  # Or use ANTHROPIC_API_KEY env var
    max_tokens=1000
)
```

### HuggingFace

```python
# Using Inference API
hf_model = LLM.from_provider(
    "huggingface",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    api_key="your-api-key",  # Or use HF_API_KEY env var
)

# Local model
local_hf_model = LLM.from_provider(
    "huggingface",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda",  # or "cpu"
    model_config={
        "load_in_8bit": True,  # For memory efficiency
        "torch_dtype": "float16"
    }
)
```

### Ollama

```python
# Local Ollama setup
ollama_model = LLM.from_provider(
    "ollama",
    model_name="llama2",
    base_url="http://localhost:11434/v1"
)
```

## Advanced Configuration

### Custom Provider Settings

```python
from core import LLM
from core.providers import ProviderConfig

config = ProviderConfig(
    name="openai",
    model_name="gpt-4",
    api_key="your-api-key",
    retry_strategy={
        "max_retries": 3,
        "initial_delay": 1,
        "max_delay": 30,
        "exponential_base": 2
    },
    timeout_settings={
        "request_timeout": 30,
        "connection_timeout": 10
    },
    caching={
        "enabled": True,
        "ttl": 3600,  # 1 hour
        "max_size": 1000
    }
)

model = LLM.from_provider_config(config)
```

### Streaming Support

```python
async def handle_stream():
    # Initialize streaming model
    model = LLM.from_provider("openai", model_name="gpt-4")
    
    # Stream responses
    async for chunk in model.stream_generate(
        "Explain quantum computing step by step"
    ):
        print(chunk, end="", flush=True)
```

### Provider Middleware

```python
from core.providers.middleware import MemoryMiddleware, LoggingMiddleware
from core.chat import SimpleMemory

# Set up memory and logging
memory = SimpleMemory()
memory_middleware = MemoryMiddleware(memory)
logging_middleware = LoggingMiddleware()

# Add to model
model = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    middleware=[memory_middleware, logging_middleware]
)
```

## Optimizing LLM Usage

### 1. Prompt Engineering

```python
# Good prompt example
prompt = """
Role: You are a technical expert in Python programming.
Task: Explain the following code snippet and suggest improvements.
Format: Provide your response in the following structure:
1. Code explanation
2. Potential issues
3. Suggested improvements

Code to analyze:
{code}
"""

# Bad prompt example (too vague)
prompt = "What do you think about this code?"
```

### 2. Token Management

```python
from core.tokenizers import count_tokens

# Check token count before sending
text = "Your long input text here..."
token_count = count_tokens(text)

if token_count > 4000:
    # Split or truncate text
    chunks = split_text(text, max_tokens=4000)
```

### 3. Batching and Caching

```python
# Batch processing
results = await model.batch_generate([
    "Question 1",
    "Question 2",
    "Question 3"
], batch_size=3)

# With caching
from core.cache import LLMCache

cache = LLMCache(max_size=1000)
model = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    cache=cache
)
```

## Error Handling

### Common Errors and Solutions

```python
from core.exceptions import (
    ProviderError,
    RateLimitError,
    TokenLimitError,
    InvalidRequestError
)

try:
    response = await model.generate(prompt)
except RateLimitError as e:
    # Implement exponential backoff
    await handle_rate_limit(e)
except TokenLimitError as e:
    # Split input or use a model with larger context
    chunks = split_text(prompt)
    responses = []
    for chunk in chunks:
        response = await model.generate(chunk)
        responses.append(response)
except InvalidRequestError as e:
    # Handle malformed requests
    logger.error(f"Invalid request: {e}")
except ProviderError as e:
    # Handle general provider errors
    logger.error(f"Provider error: {e}")
```

### Retry Strategies

```python
from core.utils.retry import retry_with_exponential_backoff

@retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=1,
    max_delay=30
)
async def generate_with_retry(model, prompt):
    return await model.generate(prompt)
```

## Best Practices

1. **Model Selection**
   - Use smaller models (e.g., GPT-3.5) for simpler tasks
   - Reserve larger models (e.g., GPT-4) for complex reasoning
   - Consider using local models for latency-sensitive operations

2. **Prompt Design**
   - Be specific and clear in instructions
   - Use consistent formatting
   - Include examples for complex tasks
   - Structure output format requirements

3. **Error Handling**
   - Implement proper retry logic
   - Handle rate limits gracefully
   - Monitor and log errors
   - Have fallback strategies

4. **Performance**
   - Use streaming for long responses
   - Implement caching where appropriate
   - Batch similar requests
   - Monitor token usage

5. **Security**
   - Never expose API keys in code
   - Validate and sanitize user inputs
   - Implement proper access controls
   - Monitor usage patterns

## Troubleshooting

### Common Issues

1. **Rate Limiting**
   ```python
   # Implement rate limiting handler
   async def handle_rate_limit(error):
       delay = error.retry_after or 1
       logger.warning(f"Rate limited. Waiting {delay} seconds")
       await asyncio.sleep(delay)
   ```

2. **Context Length**
   ```python
   # Handle context length issues
   def handle_context_length(text, max_length):
       if len(text) > max_length:
           return text[:max_length]
       return text
   ```

3. **Model Availability**
   ```python
   # Implement fallback models
   async def generate_with_fallback(prompt):
       models = [
           ("gpt-4", 1.0),      # Primary
           ("gpt-3.5", 0.7),    # Fallback 1
           ("claude-instant", 0.5)  # Fallback 2
       ]
       
       for model_name, temp in models:
           try:
               model = LLM.from_provider("openai", model_name=model_name)
               return await model.generate(prompt)
           except ProviderError:
               continue
       raise Exception("All models failed")
   ```

## Using Multiple Providers

### Load Balancing

```python
from core.providers import ProviderPool

# Create provider pool
pool = ProviderPool([
    LLM.from_provider("openai", model_name="gpt-4"),
    LLM.from_provider("anthropic", model_name="claude-3"),
    LLM.from_provider("ollama", model_name="llama2")
])

# Use pool with automatic failover
response = await pool.generate(
    prompt,
    strategy="round_robin"  # or "random", "weighted"
)
```

### Provider Selection

```python
def select_provider(task_type, input_length, priority):
    if task_type == "creative" and priority == "high":
        return LLM.from_provider("openai", model_name="gpt-4")
    elif input_length > 10000:
        return LLM.from_provider("anthropic", model_name="claude-3")
    else:
        return LLM.from_provider("openai", model_name="gpt-3.5-turbo")
```

## Monitoring and Analytics

```python
from core.monitoring import LLMMetrics

# Initialize metrics
metrics = LLMMetrics()

# Track usage
async with metrics.track():
    response = await model.generate(prompt)

# Get statistics
stats = metrics.get_statistics()
print(f"Average latency: {stats.avg_latency}ms")
print(f"Token usage: {stats.total_tokens}")
print(f"Success rate: {stats.success_rate}%")
```