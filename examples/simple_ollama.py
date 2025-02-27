"""Simple example using Ollama provider."""
import asyncio
import logging
from examples.utils import create_ollama_node, ensure_ollama_available, AsyncContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Create LLM node with Ollama config using the utility function
    node = create_ollama_node(
        name="ollama_node",
        model="qwen2.5:0.5b",  # or your preferred model
    )
    
    # Use AsyncContext for proper resource management
    async with AsyncContext(
        enter_func=node.initialize,
        exit_func=node.cleanup
    ):
        try:
            # Process a simple prompt
            result = await node.process({
                "prompt": "What is the capital of France?"
            })
            print(f"Response: {result['response']}")
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")

if __name__ == "__main__":
    asyncio.run(main())