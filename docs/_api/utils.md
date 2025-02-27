---
title: Utilities
description: Utility functions and helper classes provided by Manas
parent: API Reference
---

# Utilities

This page documents utility functions and helper classes provided by Manas.

## Text Processing

### Tokenization

```python
from core.tokenizers import count_tokens, split_by_tokens

# Count tokens in text
token_count = count_tokens("Your text here")

# Split text by token count
chunks = split_by_tokens(
    text="Long text to split",
    chunk_size=500,
    overlap=50
)
```

### Text Cleaning

```python
from core.utils.text import (
    clean_text,
    remove_extra_whitespace,
    normalize_line_endings
)

# Clean text
cleaned = clean_text("Your  messy\ttext\n\nhere")

# Remove extra whitespace
normalized = remove_extra_whitespace("Too   many    spaces")

# Normalize line endings
fixed = normalize_line_endings("Mixed\r\nline\nends")
```

## Document Processing

### Document Loading

```python
from core.utils.docs import load_document, detect_format

# Detect document format
format_info = detect_format("document.pdf")

# Load document with auto-detection
doc = load_document("document.pdf")

# Load with specific format
doc = load_document(
    "document.txt",
    format="text",
    encoding="utf-8"
)
```

### Document Chunking

```python
from core.utils.docs import chunk_document, merge_chunks

# Split document into chunks
chunks = chunk_document(
    document,
    chunk_size=500,
    overlap=50,
    strategy="sentence"
)

# Merge small chunks
merged = merge_chunks(
    chunks,
    min_size=200,
    max_size=1000
)
```

## Async Utilities

### Concurrency Helpers

```python
from core.utils.async_utils import (
    run_concurrently,
    with_timeout,
    retry_async
)

# Run tasks concurrently
results = await run_concurrently(
    tasks=[task1, task2, task3],
    max_concurrency=3
)

# Add timeout to coroutine
result = await with_timeout(
    coro=long_running_task(),
    timeout=30
)

# Retry with backoff
result = await retry_async(
    coro=flaky_operation(),
    max_attempts=3,
    backoff_base=2
)
```

### Resource Management

```python
from core.utils.async_utils import (
    AsyncResourceManager,
    cleanup_resources
)

# Manage async resources
async with AsyncResourceManager() as resources:
    # Add resources
    await resources.add(resource1)
    await resources.add(resource2)
    
    # Resources auto-cleanup after block

# Manual cleanup
await cleanup_resources([resource1, resource2])
```

## Cache Management

### Memory Cache

```python
from core.utils.cache import (
    MemoryCache,
    TTLCache,
    LRUCache
)

# Simple memory cache
cache = MemoryCache()
cache.set("key", "value")
value = cache.get("key")

# Cache with TTL
cache = TTLCache(ttl=3600)  # 1 hour
cache.set("key", "value")

# LRU cache with max size
cache = LRUCache(maxsize=1000)
cache.set("key", "value")
```

### Disk Cache

```python
from core.utils.cache import DiskCache

# Initialize disk cache
cache = DiskCache(
    directory="cache",
    ttl=86400,  # 1 day
    cleanup_interval=3600
)

# Cache operations
await cache.set("key", "value")
value = await cache.get("key")
await cache.delete("key")
```

## Configuration Utilities

### Environment Helpers

```python
from core.utils.config import (
    load_env,
    get_env,
    parse_bool
)

# Load .env file
load_env()

# Get environment variable with type
api_key = get_env("API_KEY", required=True)
debug = parse_bool(get_env("DEBUG", default="false"))
port = get_env("PORT", default="8080", cast=int)
```

### Config Loading

```python
from core.utils.config import (
    load_config,
    merge_configs,
    validate_config
)

# Load configuration
config = load_config("config.yml")

# Merge configurations
merged = merge_configs(default_config, user_config)

# Validate configuration
errors = validate_config(config, schema)
```

## Logging Utilities

### Logging Setup

```python
from core.utils.logging import setup_logging

# Configure logging
setup_logging(
    level="INFO",
    format="detailed",
    output="logs/app.log",
    rotation="1 day"
)
```

### Log Formatting

```python
from core.utils.logging import (
    format_error,
    format_request,
    format_response
)

# Format error for logging
error_log = format_error(error, include_trace=True)

# Format API request/response
req_log = format_request(request)
resp_log = format_response(response)
```

## Type Utilities

### Type Checking

```python
from core.utils.types import (
    is_coroutine,
    is_generator,
    is_async_generator
)

# Check types
if is_coroutine(obj):
    result = await obj
    
if is_generator(obj):
    results = list(obj)
    
if is_async_generator(obj):
    async for item in obj:
        process(item)
```

### Type Conversion

```python
from core.utils.types import (
    to_bool,
    to_int,
    to_float,
    to_list
)

# Convert types safely
bool_val = to_bool("true")     # True
int_val = to_int("123")        # 123
float_val = to_float("12.34")  # 12.34
list_val = to_list("a,b,c")    # ["a", "b", "c"]
```

## Debug Utilities

### Performance Monitoring

```python
from core.utils.debug import (
    timer,
    memory_usage,
    profile_function
)

# Measure execution time
with timer() as t:
    long_operation()
print(f"Took {t.elapsed:.2f}s")

# Monitor memory usage
with memory_usage() as mem:
    memory_intensive_operation()
print(f"Peak memory: {mem.peak_mb}MB")

# Profile function
stats = profile_function(target_function)
```

### Debug Information

```python
from core.utils.debug import (
    get_stack_trace,
    object_info,
    memory_snapshot
)

# Get stack trace
trace = get_stack_trace()

# Get object information
info = object_info(obj)

# Get memory snapshot
snapshot = memory_snapshot()
```

## Best Practices

1. **Error Handling**
   - Use appropriate error types
   - Handle edge cases
   - Provide context
   - Clean up resources

2. **Performance**
   - Cache expensive operations
   - Use async where appropriate
   - Monitor resource usage
   - Clean up properly

3. **Type Safety**
   - Use type hints
   - Validate inputs
   - Convert safely
   - Handle edge cases

4. **Configuration**
   - Use environment variables
   - Validate configs
   - Provide defaults
   - Document options

5. **Debugging**
   - Log appropriately
   - Monitor performance
   - Profile when needed
   - Clean up resources

## Notes

- Handle errors gracefully
- Clean up resources
- Use type hints
- Document functions
- Test utilities
- Monitor performance