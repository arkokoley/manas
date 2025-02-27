"""Common utilities for example scripts."""
import asyncio
import logging
import os
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

from core.llm import LLMNode, LLMConfig
from core.flow import Flow
from core.base import Node, Edge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_ollama_node(
    name: str,
    model: str = "llama2",
    base_url: str = "http://localhost:11434/v1",
    temperature: float = 0.7
) -> LLMNode:
    """Create an LLM node using Ollama.
    
    Args:
        name: Name of the node
        model: Model name to use
        base_url: Ollama API base URL
        temperature: Generation temperature
        
    Returns:
        Configured LLMNode using Ollama
    """
    return LLMNode(
        name=name,
        config=LLMConfig(
            provider_name="ollama",
            provider_config={
                "model": model,
                "base_url": base_url
            },
            temperature=temperature
        )
    )

async def ensure_ollama_available(base_url: str = "http://localhost:11434/v1") -> bool:
    """Check if Ollama service is available.
    
    Args:
        base_url: Ollama API base URL
        
    Returns:
        True if available, False otherwise
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags") as response:
                available = response.status == 200
                if not available:
                    logger.warning("Ollama API is not accessible. Please ensure Ollama is running.")
                return available
    except Exception as e:
        logger.warning(f"Error connecting to Ollama: {e}")
        return False

class Tool:
    """Represents a tool that can be used by agents."""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
        
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)

def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    # Find the mas module path and go up two levels
    current_file = Path(__file__).resolve()
    examples_dir = current_file.parent
    return examples_dir.parent

# Common file operations
async def read_file(path: str) -> str:
    """Read contents from a file.
    
    Args:
        path: File path relative to project root or absolute
        
    Returns:
        File contents as string
    """
    try:
        # Remove any extra quotes that might be in the path
        path = path.strip("'\"")
        
        if not os.path.isabs(path):
            path = os.path.join(get_project_root(), path)
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

async def write_file(path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        path: File path relative to project root or absolute
        content: Content to write
        
    Returns:
        Success message or error description
    """
    try:
        if not os.path.isabs(path):
            path = os.path.join(get_project_root(), path)
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

# Text processing utilities
def analyze_text(text: str, operation: str) -> str:
    """Analyze text with various operations.
    
    Args:
        text: Text to analyze
        operation: Operation to perform (count_words, count_lines, etc.)
        
    Returns:
        Analysis results
    """
    ops = {
        "count_words": lambda x: str(len(x.split())),
        "count_lines": lambda x: str(len(x.splitlines())),
        "count_chars": lambda x: str(len(x)),
        "get_stats": lambda x: (
            f"Words: {len(x.split())}, "
            f"Lines: {len(x.splitlines())}, "
            f"Characters: {len(x)}"
        ),
        "extract_keywords": lambda x: ", ".join(
            sorted(set(w.lower() for w in x.split() if len(w) > 5))[:20]
        )
    }
    
    if operation not in ops:
        return f"Unknown operation. Available operations: {', '.join(ops.keys())}"
    
    return ops[operation](text)

def transform_text(text: str, operation: str) -> str:
    """Transform text with various operations.
    
    Args:
        text: Text to transform
        operation: Operation to perform (uppercase, lowercase, etc.)
        
    Returns:
        Transformed text
    """
    ops = {
        "uppercase": str.upper,
        "lowercase": str.lower,
        "titlecase": str.title,
        "reverse_lines": lambda x: "\n".join(reversed(x.splitlines())),
        "sort_lines": lambda x: "\n".join(sorted(x.splitlines())),
        "deduplicate_lines": lambda x: "\n".join(dict.fromkeys(x.splitlines()))
    }
    
    if operation not in ops:
        return f"Unknown operation. Available operations: {', '.join(ops.keys())}"
    
    return ops[operation](text)

# Common file tools
def get_common_tools() -> List[Tool]:
    """Get a set of common tools for file and text operations.
    
    Returns:
        List of common tools
    """
    return [
        Tool(
            name="read_file",
            description="Read content from a file. Args: path(str)",
            func=read_file
        ),
        Tool(
            name="write_file",
            description="Write content to a file. Args: path(str), content(str)",
            func=write_file
        ),
        Tool(
            name="analyze_text",
            description="Analyze text with operations (count_words/count_lines/count_chars/get_stats/extract_keywords). Args: text(str), operation(str)",
            func=analyze_text
        ),
        Tool(
            name="transform_text",
            description="Transform text with operations (uppercase/lowercase/titlecase/reverse_lines/sort_lines/deduplicate_lines). Args: text(str), operation(str)",
            func=transform_text
        )
    ]

class AsyncContext:
    """Context manager for async resource management."""
    
    def __init__(self, enter_func=None, exit_func=None):
        self.enter_func = enter_func or (lambda: None)
        self.exit_func = exit_func or (lambda: None)
    
    async def __aenter__(self):
        if asyncio.iscoroutinefunction(self.enter_func):
            return await self.enter_func()
        return self.enter_func()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if asyncio.iscoroutinefunction(self.exit_func):
            await self.exit_func()
        else:
            self.exit_func()