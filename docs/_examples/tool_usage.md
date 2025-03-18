---
layout: example
title: Working with Tools
description: Learn how to integrate external tools with Manas agents and flows
nav_order: 2
parent: Examples
difficulty: Beginner
time: 20 minutes
source_file: tool_usage_example.py
related_docs:
  - title: Agent API Reference
    url: /api/agent/
  - title: ToolNode Reference
    url: /api/nodes/tool_node/
  - title: Flow Documentation
    url: /api/flow/
---

# Working with Tools

This tutorial shows how to create agents that can use external tools to enhance their capabilities.

## Overview

We'll create:
1. A calculator tool for mathematical operations
2. A web search tool for information retrieval
3. An agent that can use both tools effectively

## Prerequisites

```bash
pip install "manas-ai[all-cpu]" aiohttp
```

## Implementation

```python
import os
import aiohttp
import json
from core import LLM, Agent, Tool
from manas_ai.nodes import ToolNode

# Define a calculator tool
def calculator_tool(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.
    
    Args:
        expression: A string containing a mathematical expression (e.g., "2 + 2")
        
    Returns:
        The result as a string
    """
    try:
        # Safe evaluation with limited operations
        allowed_names = {
            "abs": abs, 
            "round": round,
            "sum": sum,
            "min": min,
            "max": max
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Define a web search tool
async def web_search_tool(query: str) -> str:
    """
    Performs a web search using a search API.
    
    Args:
        query: The search query
        
    Returns:
        Search results as a JSON string
    """
    try:
        # Replace with your preferred search API
        search_url = "https://api.search.example.com/v1/search"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                search_url,
                params={"q": query},
                headers={"Authorization": f"Bearer {os.environ.get('SEARCH_API_KEY')}"}
            ) as response:
                data = await response.json()
                return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error performing search: {str(e)}"

# Create an agent with tools
agent = Agent(
    llm=LLM.from_provider("openai", model_name="gpt-4"),
    name="assistant",
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use the calculator for math operations and web search for information."
    )
)

# Register tools with the agent
agent.add_tool(Tool("calculator", calculator_tool))
agent.add_tool(Tool("web_search", web_search_tool))

# Example usage
async def main():
    try:
        # Initialize the agent
        await agent.initialize()
        
        # Process queries that require tools
        result = await agent.process({
            "prompt": "What is 157 * 23? Also, find information about quantum computing."
        })
        
        print("Agent Response:", result["response"])
        
    finally:
        await agent.cleanup()

# Run with asyncio
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Creating Tool Nodes in a Flow

For more complex applications, you can create dedicated tool nodes in a flow:

```python
from core import Flow
from manas_ai.nodes import QANode, ToolNode

# Create nodes
qa_node = QANode(
    name="answerer",
    llm=model,
    system_prompt="You answer questions using available tools."
)

calc_node = ToolNode(
    name="calculator",
    tool=calculator_tool
)

search_node = ToolNode(
    name="web_search", 
    tool=web_search_tool
)

# Create flow
flow = Flow()
flow.add_node(qa_node)
flow.add_node(calc_node)
flow.add_node(search_node)

# Connect nodes (bidirectional since QA node needs tool results)
flow.add_edge(qa_node, calc_node)
flow.add_edge(calc_node, qa_node)
flow.add_edge(qa_node, search_node)
flow.add_edge(search_node, qa_node)

# Process queries
result = await flow.process({
    "prompt": "Calculate 125 * 37 and find recent quantum computing breakthroughs."
})
```

## Advanced Tool Configuration

For more control over tool behavior:

```python
from typing import Dict, Any

# Define a tool with metadata
weather_tool = Tool(
    name="get_weather",
    description="Gets current weather for a location",
    function=lambda location: f"Weather in {location}: Sunny, 72°F",
    metadata={
        "parameters": {
            "location": {
                "type": "string",
                "description": "City and state/country (e.g., 'New York, NY')"
            }
        },
        "rate_limit": 60,  # calls per minute
        "requires_auth": True
    }
)

# Add validation
def validate_weather_params(params: Dict[str, Any]) -> bool:
    return bool(params.get("location", "").strip())

weather_tool.set_validator(validate_weather_params)

# Add to agent with configuration
agent.add_tool(
    weather_tool,
    config={
        "cache_ttl": 300,  # 5 minutes
        "retry_attempts": 3
    }
)
```

## Error Handling

```python
try:
    result = await agent.process({
        "prompt": "What's 123 * 456 and the weather in London?"
    })
except Exception as e:
    print(f"Error: {e}")
    # Handle specific error types
    if isinstance(e, ToolExecutionError):
        print("Tool execution failed")
    elif isinstance(e, ToolNotFoundError):
        print("Required tool not available")
finally:
    await agent.cleanup()
```

## Complete Example

The complete example with additional features and error handling is available in the [examples directory](https://github.com/arkokoley/manas/blob/main/examples/tool_usage.py).