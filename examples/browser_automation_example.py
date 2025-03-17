"""
Example of using BrowserNode for web automation tasks using browser-use Agent.
Demonstrates browser automation with deepseek-r1 model.
"""

import asyncio
from core.nodes import create_browser_node

async def browser_agent_example():
    """Demonstrate browser automation using Agent."""
    print("\n=== Browser Agent Example ===")
    
    # Create browser node with deepseek-r1 model
    browser = create_browser_node(
        name="browser",
        model="deepseek-r1",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        use_vision=False
    )
    
    try:
        # Initialize browser
        await browser.initialize()
        
        # Task: Research pricing
        print("\nTask 1: Researching AI model pricing...")
        result = await browser.process({
            "task": "Research and compare pricing for GPT-4 and DeepSeek models. Focus on token costs and any usage limitations.",
            "url": "https://www.together.ai/pricing"
        })
        print("\nPricing Research Result:", result.get("result", "No result"))
        
    finally:
        await browser.cleanup()

async def main():
    """Run the example."""
    await browser_agent_example()

if __name__ == "__main__":
    asyncio.run(main())