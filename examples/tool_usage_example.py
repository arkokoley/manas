"""
Example of creating and using tools with Manas agents and flows.
Demonstrates:
1. Basic tool creation and usage
2. Tool integration in flows
3. Advanced tool configuration and error handling
"""

import os
import asyncio
import aiohttp
import json
from typing import Dict, Any, List
from datetime import datetime

from core import LLM, Agent, Tool, Flow
from core.nodes import ToolNode, QANode
from core.models import ToolConfig

# Basic Calculator Tool
def calculator_tool(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")
        
    Returns:
        Result as a string
    """
    try:
        # Safe evaluation with limited operations
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }
        result = eval(
            expression,
            {"__builtins__": {}},
            allowed_names
        )
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Web Search Tool
async def web_search_tool(query: str) -> str:
    """
    Perform a web search using a search API.
    
    Args:
        query: Search query string
        
    Returns:
        Search results as JSON string
    """
    try:
        # Replace with your preferred search API
        search_url = "https://api.search.example.com/v1/search"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                search_url,
                params={"q": query},
                headers={
                    "Authorization": f"Bearer {os.environ.get('SEARCH_API_KEY')}"
                }
            ) as response:
                data = await response.json()
                return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error performing search: {str(e)}"

# Weather Tool with Validation
def validate_weather_params(params: Dict[str, Any]) -> bool:
    """Validate weather tool parameters."""
    location = params.get("location", "").strip()
    if not location:
        return False
        
    # Optional date validation
    date = params.get("date")
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return False
            
    return True

async def weather_tool(
    location: str,
    date: str = None
) -> Dict[str, Any]:
    """
    Get weather information for a location.
    
    Args:
        location: City and state/country
        date: Optional date (YYYY-MM-DD)
        
    Returns:
        Weather information
    """
    # Simulate weather API call
    await asyncio.sleep(1)
    return {
        "location": location,
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "temperature": 72,
        "conditions": "Sunny",
        "humidity": 45
    }

# Basic Agent Example
async def basic_agent_example():
    """Demonstrate basic agent with tools."""
    print("\n=== Basic Agent Example ===")
    
    # Create agent
    agent = Agent(
        llm=LLM.from_provider("openai", model_name="gpt-4"),
        name="assistant",
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use the calculator for math and search for information."
        )
    )
    
    # Add tools
    agent.add_tool(Tool("calculator", calculator_tool))
    agent.add_tool(Tool("web_search", web_search_tool))
    
    try:
        # Initialize agent
        await agent.initialize()
        
        # Process queries
        result = await agent.process({
            "prompt": "What is 157 * 23? Also, find information about quantum computing."
        })
        
        print("\nAgent Response:")
        print(result["response"])
        
    finally:
        await agent.cleanup()

# Flow with Tools Example
async def tool_flow_example():
    """Demonstrate tools in a flow."""
    print("\n=== Tool Flow Example ===")
    
    # Create nodes
    qa_node = QANode(
        name="answerer",
        llm=LLM.from_provider("openai", model_name="gpt-4"),
        system_prompt="You answer questions using available tools."
    )
    
    calc_node = ToolNode(
        name="calculator",
        tool=Tool("calculator", calculator_tool)
    )
    
    search_node = ToolNode(
        name="web_search",
        tool=Tool("web_search", web_search_tool)
    )
    
    # Create flow
    flow = Flow()
    flow.add_node(qa_node)
    flow.add_node(calc_node)
    flow.add_node(search_node)
    
    # Connect nodes (bidirectional for tool access)
    flow.add_edge(qa_node, calc_node)
    flow.add_edge(calc_node, qa_node)
    flow.add_edge(qa_node, search_node)
    flow.add_edge(search_node, qa_node)
    
    try:
        # Initialize flow
        await flow.initialize()
        
        # Process query
        result = await flow.process({
            "prompt": "Calculate 125 * 37 and find recent quantum computing breakthroughs."
        })
        
        print("\nFlow Response:")
        print(result["answerer"]["response"])
        
    finally:
        await flow.cleanup()

# Advanced Tool Configuration
async def advanced_tool_example():
    """Demonstrate advanced tool features."""
    print("\n=== Advanced Tool Example ===")
    
    # Create weather tool with configuration
    weather = Tool(
        name="get_weather",
        description="Gets weather for a location and date",
        function=weather_tool,
        config=ToolConfig(
            async_execution=True,
            timeout=30,
            retry_attempts=3,
            validation={
                "enabled": True,
                "validator": validate_weather_params
            },
            cache_config={
                "enabled": True,
                "ttl": 3600  # 1 hour
            }
        ),
        metadata={
            "parameters": {
                "location": {
                    "type": "string",
                    "description": "City and state/country"
                },
                "date": {
                    "type": "string",
                    "description": "Date (YYYY-MM-DD)",
                    "optional": True
                }
            },
            "returns": {
                "type": "object",
                "description": "Weather information"
            }
        }
    )
    
    # Create agent with weather tool
    agent = Agent(
        llm=LLM.from_provider("openai", model_name="gpt-4"),
        name="weather_assistant",
        system_prompt=(
            "You are a weather assistant. Use the weather tool "
            "to provide weather information."
        )
    )
    
    agent.add_tool(weather)
    
    try:
        # Initialize agent
        await agent.initialize()
        
        # Test various queries
        queries = [
            "What's the weather in London?",
            "What's the weather in Paris for 2024-01-01?",
            "What's the weather in Tokyo tomorrow?",  # Should fail validation
            "What's the weather?",  # Should fail validation
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            try:
                result = await agent.process({"prompt": query})
                print("Response:", result["response"])
            except Exception as e:
                print("Error:", str(e))
                
    finally:
        await agent.cleanup()

async def main():
    """Run all examples."""
    # Basic agent example
    await basic_agent_example()
    
    # Flow example
    await tool_flow_example()
    
    # Advanced example
    await advanced_tool_example()

if __name__ == "__main__":
    asyncio.run(main())