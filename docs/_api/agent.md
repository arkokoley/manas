---
title: Agent
description: Base agent class for creating autonomous agents
parent: API Reference
---

# Agent

The `Agent` class extends the base `Node` class to implement autonomous agents that can think, act, and learn from their actions. Agents follow a think-act-observe cycle and can maintain state across interactions.

## Import

```python
from core import Agent
from manas_ai.models import Tool
```

## Constructor

```python
def __init__(
    self,
    name: str,
    llm: Optional[LLM] = None,
    system_prompt: Optional[str] = None,
    memory: Optional[Dict[str, Any]] = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | Required | Agent identifier |
| llm | Optional[LLM] | None | Language model to use |
| system_prompt | Optional[str] | None | Agent's system prompt |
| memory | Optional[Dict[str, Any]] | None | Initial memory state |

## Core Methods

### think

```python
async def think(self, context: Dict[str, Any]) -> Dict[str, Any]
```

Analyze context and plan actions. Override this to customize agent's thinking process.

### act

```python
async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]
```

Execute planned actions. Override this to implement custom actions.

### observe

```python
async def observe(self, result: Dict[str, Any]) -> Dict[str, Any]
```

Process action results and update state. Override to customize learning.

### process

```python
async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]
```

Execute the think-act-observe cycle.

## Tool Management

### add_tool

```python
def add_tool(self, tool: Tool) -> None
```

Add a tool to the agent's capabilities.

### remove_tool

```python
def remove_tool(self, tool_name: str) -> None
```

Remove a tool from the agent.

### get_tool

```python
def get_tool(self, tool_name: str) -> Optional[Tool]
```

Get a tool by name.

## Memory Management

### remember

```python
def remember(self, key: str, value: Any) -> None
```

Store information in agent's memory.

### recall

```python
def recall(self, key: str) -> Optional[Any]
```

Retrieve information from memory.

### forget

```python
def forget(self, key: str) -> None
```

Remove information from memory.

## Example Usage

### Basic Agent

```python
# Create a simple agent
agent = Agent(
    name="assistant",
    llm=LLM.from_provider("openai", model_name="gpt-4"),
    system_prompt="You are a helpful assistant."
)

# Initialize and use
await agent.initialize()
try:
    result = await agent.process({
        "prompt": "What is quantum computing?"
    })
    print(result["response"])
finally:
    await agent.cleanup()
```

### Tool-Using Agent

```python
# Create tools
calculator = Tool(
    name="calculator",
    description="Performs calculations",
    function=lambda x: eval(x)
)

search = Tool(
    name="web_search",
    description="Searches the web",
    function=async_search_function
)

# Create agent with tools
agent = Agent(
    name="researcher",
    llm=model,
    system_prompt=(
        "You are a research assistant with access to tools. "
        "Use the calculator for math and search for information."
    )
)

# Add tools
agent.add_tool(calculator)
agent.add_tool(search)

# Process with tools
result = await agent.process({
    "prompt": "What is 157 * 23? Also, find information about Mars."
})
```

### Custom Agent Implementation

```python
class ResearchAgent(Agent):
    """Specialized research agent with custom behavior."""
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Custom thinking process
        topic = context["prompt"]
        return {
            "plan": [
                f"Research {topic}",
                "Analyze findings",
                "Summarize results"
            ],
            "next_action": "research",
            "parameters": {"topic": topic}
        }
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        # Custom action execution
        if decision["next_action"] == "research":
            return await self.do_research(decision["parameters"])
        # ... handle other actions
    
    async def observe(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Custom learning/observation
        self.remember(
            f"research_{result['topic']}",
            result["findings"]
        )
        return {
            "learned": True,
            "summary": result["summary"]
        }

# Usage
agent = ResearchAgent(
    name="specialized_researcher",
    llm=model,
    system_prompt="You are a specialized research agent."
)
```

## Best Practices

1. **Think Phase**
   - Break down complex tasks
   - Plan actions systematically
   - Consider available tools
   - Use memory for context

2. **Act Phase**
   - Execute one step at a time
   - Handle tool errors gracefully
   - Track action progress
   - Validate tool inputs

3. **Observe Phase**
   - Learn from results
   - Update memory appropriately
   - Identify patterns
   - Adapt future behavior

4. **Memory Usage**
   - Store relevant information
   - Clean up old/unused data
   - Use structured memory
   - Consider memory limits

## Notes

- Agents maintain state across interactions
- Tools must be added before use
- Memory persists until cleared
- Clean up resources properly
- Override core methods for custom behavior
- Use type hints for better IDE support