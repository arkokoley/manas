"""Common utilities for tests and benchmarks."""
import asyncio
import logging
import time
import sys
import subprocess
import importlib.util
from typing import Dict, Any, List, Optional, Callable, Union, Type
from pathlib import Path

from core.flow import Flow
from core.base import Node, Edge
from core.llm import LLMNode, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies(required_packages: Dict[str, str]) -> None:
    """Check and install required dependencies.
    
    Args:
        required_packages: Dict mapping module names to pip package names
    """
    missing = []
    for package, pip_name in required_packages.items():
        if importlib.util.find_spec(package) is None:
            missing.append(pip_name)
    
    if missing:
        logger.info(f"Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            logger.info("Successfully installed dependencies")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
            sys.exit(1)

async def check_ollama_availability(url: str = "http://localhost:11434/v1") -> bool:
    """Check if Ollama is available at the specified URL.
    
    Args:
        url: Base URL for Ollama API
        
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/api/tags") as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return False

def create_llm_node(name: str, 
                   model: str = "llama3.2", 
                   base_url: str = "http://localhost:11434/v1",
                   temperature: float = 0.7) -> LLMNode:
    """Create an LLM node with standard configuration.
    
    Args:
        name: Name for the node
        model: Model name to use
        base_url: Base URL for Ollama API
        temperature: Temperature setting for generation
        
    Returns:
        Configured LLMNode
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

def create_linear_flow(name: str, n_nodes: int, 
                      node_factory: Callable[[str], Node]) -> Flow:
    """Create a linear flow with the specified number of nodes.
    
    Args:
        name: Name for the flow
        n_nodes: Number of nodes to create
        node_factory: Function that creates nodes given a name
        
    Returns:
        Flow with linearly connected nodes
    """
    flow = Flow(name=name)
    prev_node_id = None
    
    for i in range(n_nodes):
        node = node_factory(f"node_{i}")
        node_id = flow.add_node(node)
        
        if prev_node_id is not None:
            flow.add_edge(Edge(
                source_node=prev_node_id,
                target_node=node_id,
                name=f"edge_{i}"
            ))
        prev_node_id = node_id
    
    return flow

async def measure_execution_time(flow: Flow, 
                              inputs: Dict[str, Any]) -> float:
    """Measure execution time for a flow.
    
    Args:
        flow: Flow to execute
        inputs: Inputs for the flow
        
    Returns:
        Execution time in seconds
    """
    start_time = time.time()
    await flow.process(inputs)
    return time.time() - start_time

class TimingContext:
    """Context manager for timing code execution."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.execution_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.execution_time = time.time() - self.start_time
        logger.info(f"{self.description} completed in {self.execution_time:.2f}s")