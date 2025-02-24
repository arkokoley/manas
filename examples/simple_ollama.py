"""Simple example using Ollama provider."""
import asyncio
from mas.core.llm import LLMNode, LLMConfig

async def main():
    # Create LLM node with Ollama config
    node = LLMNode(
        name="ollama_node",
        config=LLMConfig(
            provider_name="ollama",
            provider_config={
                "model": "qwen2.5:0.5b",  # or your preferred model
                "base_url": "http://localhost:11434/v1"
            },
            temperature=0.7
        )
    )
    
    # Initialize the node
    await node.initialize()
    
    try:
        # Process a simple prompt
        result = await node.process({
            "prompt": "What is the capital of France?"
        })
        print(f"Response: {result['response']}")
        
    finally:
        # Always cleanup
        await node.cleanup()

if __name__ == "__main__":
    asyncio.run(main())