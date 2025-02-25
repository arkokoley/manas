"""Test script for tool-using agent."""
import asyncio
import json
import os
from pathlib import Path
from tool_using_agent import ToolUsingAgent, Tool

async def main():
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Initialize agent with Ollama
    agent = ToolUsingAgent(
        name="file_processor",
        provider="ollama",
        provider_config={
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1"
        }
    )
    
    # Define more sophisticated tools with proper path handling
    def read_file(path: str) -> str:
        try:
            # Convert relative paths to absolute using project root
            if not os.path.isabs(path):
                path = os.path.join(project_root, path)
            with open(path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(path: str, content: str) -> str:
        try:
            # Convert relative paths to absolute using project root
            if not os.path.isabs(path):
                path = os.path.join(project_root, path)
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def analyze_text(text: str, operation: str) -> str:
        ops = {
            "count_words": lambda x: len(x.split()),
            "count_lines": lambda x: len(x.splitlines()),
            "get_stats": lambda x: json.dumps({
                "words": len(x.split()),
                "lines": len(x.splitlines()),
                "chars": len(x),
                "avg_word_length": sum(len(w) for w in x.split()) / len(x.split()) if x.split() else 0
            }),
            "extract_keywords": lambda x: ", ".join(sorted(set(w.lower() for w in x.split() if len(w) > 5)))[:200]
        }
        if operation not in ops:
            return f"Error: Unknown operation. Available operations: {', '.join(ops.keys())}"
        return str(ops[operation](text))
    
    def transform_text(text: str, operation: str) -> str:
        ops = {
            "uppercase": str.upper,
            "lowercase": str.lower,
            "titlecase": str.title,
            "reverse_lines": lambda x: "\n".join(reversed(x.splitlines())),
            "sort_lines": lambda x: "\n".join(sorted(x.splitlines())),
            "deduplicate_lines": lambda x: "\n".join(dict.fromkeys(x.splitlines()))
        }
        if operation not in ops:
            return f"Error: Unknown operation. Available operations: {', '.join(ops.keys())}"
        return ops[operation](text)
    
    # Register tools with improved descriptions
    agent.add_tool(Tool(
        name="read_file",
        description="Read content from a file. Paths can be relative to project root or absolute. Args: path(str)",
        func=read_file
    ))
    
    agent.add_tool(Tool(
        name="write_file",
        description="Write content to a file. Creates directories if needed. Args: path(str), content(str)",
        func=write_file
    ))
    
    agent.add_tool(Tool(
        name="analyze_text",
        description="Analyze text with operations (count_words/count_lines/get_stats/extract_keywords). Args: text(str), operation(str)",
        func=analyze_text
    ))
    
    agent.add_tool(Tool(
        name="transform_text",
        description="Transform text with operations (uppercase/lowercase/titlecase/reverse_lines/sort_lines/deduplicate_lines). Args: text(str), operation(str)",
        func=transform_text
    ))
    
    # Initialize
    await agent.llm.initialize()
    
    try:
        # Test with a complex multi-step task using relative path
        task = """
        Perform the following analysis on README.md:
        1. Read the file
        2. Get detailed statistics about the text
        3. Extract key terms
        4. Transform the content to title case
        5. Write the results to reports/analysis_report.txt
        Include all the information collected in the final report.
        """
        
        print("Starting complex task:", task)
        
        # Process the task
        result = await agent.process({"task": task})
        
        print("\nInitial Plan:")
        for i, step in enumerate(result["decision"]["plan"], 1):
            print(f"{i}. {step}")
        
        print("\nFirst Step:")
        print(result["decision"]["next_step"])
        
        print("\nExecution Results:")
        print(result["result"].get("result", "No result"))
        
        print("\nAgent's Analysis:")
        print(result["observation"]["summary"])
        
        print("\nRemaining Plan:")
        print(result["observation"]["remaining_plan"])
        
    finally:
        await agent.llm.cleanup()

if __name__ == "__main__":
    asyncio.run(main())