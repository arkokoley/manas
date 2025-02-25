"""Test script for tool-using agent."""
import asyncio
import json
from pathlib import Path

from examples.utils import (
    logger, ensure_ollama_available, AsyncContext,
    get_common_tools, analyze_text, transform_text
)
from examples.tool_using_agent import ToolUsingAgent, Tool

async def main():
    # Check if Ollama is available
    ollama_available = await ensure_ollama_available()
    if not ollama_available:
        logger.error("Ollama service not available. Please ensure Ollama is running.")
        return
    
    # Initialize agent with Ollama
    agent = ToolUsingAgent(
        name="file_processor",
        provider="ollama",
        provider_config={
            "model": "llama2",
            "base_url": "http://localhost:11434/v1"
        }
    )
    
    # Add common tools for file and text operations
    agent.add_common_tools()
    
    # Add specialized text analysis tool
    async def get_text_stats(text: str) -> str:
        """Get detailed statistics about text."""
        words = len(text.split())
        lines = len(text.splitlines())
        key_terms = set(word.lower() for word in text.split() 
                       if len(word) > 5 and word.isalnum())
        
        stats = {
            "words": words,
            "lines": lines,
            "chars": len(text),
            "avg_word_length": sum(len(w) for w in text.split()) / words if words else 0,
            "key_terms": sorted(list(key_terms))[:10]
        }
        
        return json.dumps(stats, indent=2)
    
    # Add specialized tool
    agent.add_tool(Tool(
        name="get_text_stats",
        description="Get detailed statistics about text in JSON format. Args: text(str)",
        func=get_text_stats
    ))
    
    # Initialize
    await agent.llm.initialize()
    
    try:
        # Test with a complex multi-step task
        task = """
        Perform the following analysis on README.md:
        1. Read the file
        2. Get detailed statistics about the text
        3. Extract key terms
        4. Transform the content to title case
        5. Write the results to reports/analysis_report.txt
        Include all the information collected in the final report.
        """
        
        logger.info("Starting complex task: %s", task)
        
        # Process the task
        result = await agent.process({"task": task})
        
        logger.info("\nInitial Plan:")
        for i, step in enumerate(result["decision"]["plan"], 1):
            logger.info(f"{i}. {step}")
        
        logger.info("\nFirst Step:")
        logger.info(result["decision"]["next_step"])
        
        logger.info("\nExecution Results:")
        logger.info(result["result"].get("result", "No result")[:200] + "...")
        
        logger.info("\nAgent's Analysis:")
        logger.info(result["observation"]["summary"])
        
        logger.info("\nRemaining Plan:")
        logger.info(result["observation"]["remaining_plan"])
        
    finally:
        await agent.llm.cleanup()

if __name__ == "__main__":
    asyncio.run(main())