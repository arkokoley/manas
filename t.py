import asyncio
from mas.core.agent import Agent
from mas.core.llm import LLMNode, LLMConfig

async def main():
    # Initialize LLM node with Ollama
    llm = LLMNode(
        name="ollama_node",
        config=LLMConfig(
            provider_name="ollama",
            provider_config={
                "model": "deepseek-r1:latest",
                "base_url": "http://localhost:11434/v1"
            }
        )
    )
    
    # Process a prompt
    result = await llm.process({
        "prompt": "Explain quantum computing in simple terms"
    })
    
    print(result["response"])

if __name__ == "__main__":
    asyncio.run(main())
