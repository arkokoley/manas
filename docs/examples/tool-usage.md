---
layout: tutorial
title: Tool Integration with Agents
nav_order: 2
parent: Examples
permalink: /examples/tool-usage/
difficulty: Beginner
time: 15 minutes
---

# Tool Integration with Manas Agents

This example demonstrates how to create agents that can use external tools to enhance their capabilities.

## Objective

Build an agent that can:
1. Access web search capabilities
2. Perform calculations
3. Use these tools to solve complex problems

## Prerequisites

```bash
pip install "manas[all-cpu]" requests
```

## Implementation

```python
import os
import requests
import json
from core import LLM, Agent
from manas_ai.nodes import ToolNode
from typing import Callable, Dict, Any

# Define a simple calculator tool
def calculator_tool(expression: str) -> str:
    """Evaluates a mathematical expression.
    
    Args:
        expression: A string containing a mathematical expression (e.g., "2 + 2")
        
    Returns:
        The result of the evaluated expression as a string
    """
    try:
        # Safe evaluation of mathematical expressions
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "sum": sum})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Define a simple web search tool
def web_search_tool(query: str) -> str:
    """Performs a web search using a search API.
    
    Args:
        query: The search query
        
    Returns:
        JSON string containing search results
    """
    # This is a mock implementation - in a real application, 
    # you would use an actual search API like Google or Bing
    try:
        # Mock response - in real code, make an API call here
        mock_results = [
            {"title": f"Result 1 for {query}", "snippet": "This is the first result."},
            {"title": f"Result 2 for {query}", "snippet": "This is the second result."}
        ]
        return json.dumps(mock_results, indent=2)
    except Exception as e:
        return f"Error performing search: {str(e)}"

# Initialize the LLM
llm = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create an agent with tools
agent = Agent(
    llm=llm,
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use the calculator for mathematical operations and "
        "the web search for finding information online."
    ),
    tools=[calculator_tool, web_search_tool]
)

# Use the agent to solve a problem
response = agent.generate(
    "What is the square root of 144 plus 25? Also, can you find some information about quantum computing?"
)
print(response)
```

## Explanation

This example demonstrates how to integrate tools with Manas agents:

1. **Tool Definitions**: We define two tools - a calculator and a web search function. Each tool has a docstring that describes its purpose and parameters, which the LLM uses to understand when and how to use the tool.

2. **Agent Creation**: We create an agent with access to the tools, instructing it via the system prompt to use them appropriately.

3. **Tool Usage**: When the agent receives a query that requires calculation or information retrieval, it automatically decides which tool to use and how to use it.

### How Tool Invocation Works

1. The agent analyzes the user query to determine if tools are needed
2. If needed, the agent formats a proper tool call with parameters
3. Manas executes the tool function with the provided parameters
4. The tool returns results which are sent back to the agent
5. The agent incorporates the tool results into its final response

## Creating a Tool Node in a Flow

In more complex applications, you can create specialized ToolNodes within a flow:

```python
from core import Flow
from manas_ai.nodes import QANode, ToolNode

# Create a QA node for general questions
qa_node = QANode(
    name="answerer",
    llm=llm,
    system_prompt="You answer questions based on your knowledge and tool results."
)

# Create a tool node specifically for calculations
calculator_node = ToolNode(
    name="calculator",
    tool=calculator_tool
)

# Create a search tool node
search_node = ToolNode(
    name="web_search",
    tool=web_search_tool
)

# Create a flow
tool_flow = Flow()
tool_flow.add_node(qa_node)
tool_flow.add_node(calculator_node)
tool_flow.add_node(search_node)

# Connect the nodes (bidirectional connections since the QA node needs tool results)
tool_flow.add_edge(qa_node, calculator_node)
tool_flow.add_edge(calculator_node, qa_node)
tool_flow.add_edge(qa_node, search_node)
tool_flow.add_edge(search_node, qa_node)

# Process a query
result = tool_flow.process(
    "What is 157 multiplied by 23? Also, find information about Mars."
)
print(result)
```

## Advanced Tool Configuration

For more complex tools, you can specify additional metadata:

```python
from manas_ai.models import Tool

# Define a weather API tool with more structured metadata
weather_tool = Tool(
    name="get_weather",
    description="Gets the current weather for a specified location",
    function=lambda location: f"Weather for {location}: Sunny, 72°F",
    parameters={
        "location": {
            "type": "string",
            "description": "The city and state/country, e.g., 'New York, NY'"
        }
    }
)

# Add to an agent
weather_agent = Agent(
    llm=llm,
    system_prompt="You provide weather information for locations worldwide.",
    tools=[weather_tool]
)
```

## Complete Example

You can find the complete example in the [examples directory](https://github.com/arkokoley/manas/blob/main/examples/tool_using_agent.py) of the Manas repository.