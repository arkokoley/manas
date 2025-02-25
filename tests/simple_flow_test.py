"""Simple benchmark test for core flow functionality."""
import asyncio
from typing import Dict, Any
from uuid import UUID

from mas.core.flow import Flow
from mas.core.base import Node, Edge
from mas.core.llm import LLMNode, LLMConfig

from tests.utils import (
    logger, check_ollama_availability, create_llm_node, 
    create_linear_flow, measure_execution_time, TimingContext
)

class SimpleNode(Node):
    """A simple node for testing that processes text."""
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.1)  # Simulate some work
        text = inputs.get("text", "")
        return {"result": text.upper()}

async def test_linear_flow(n_nodes: int = 3):
    """Test a simple linear flow with n nodes."""
    flow = create_linear_flow(
        name="linear_test",
        n_nodes=n_nodes,
        node_factory=lambda name: SimpleNode(name=name)
    )
    
    # Process some test data
    execution_time = await measure_execution_time(
        flow=flow,
        inputs={"text": "test input"}
    )
    
    logger.info(f"Linear flow ({n_nodes} nodes) completed in {execution_time:.2f}s")
    return execution_time

async def test_branching_flow(depth: int = 2, branch_factor: int = 2):
    """Test a branching flow with specified depth and branching factor."""
    flow = Flow(name="branch_test")
    
    # Create root node
    root = SimpleNode(name="root")
    root_id = flow.add_node(root)
    
    # Create branches recursively
    async def create_branch(parent_id: UUID, current_depth: int):
        if current_depth >= depth:
            return
        
        for i in range(branch_factor):
            node = SimpleNode(name=f"node_d{current_depth}_b{i}")
            node_id = flow.add_node(node)
            
            flow.add_edge(Edge(
                source_node=parent_id,
                target_node=node_id,
                name=f"edge_d{current_depth}_b{i}"
            ))
            
            await create_branch(node_id, current_depth + 1)
    
    await create_branch(root_id, 0)
    
    # Process test data using timing context
    async with TimingContext(f"Branching flow ({len(flow.nodes)} nodes)") as timing:
        await flow.process({"text": "test input"})
    
    return timing.execution_time

async def test_ollama_flow():
    """Test a simple flow using Ollama nodes."""
    flow = Flow(name="ollama_test")
    
    # Create two Ollama nodes
    node1 = create_llm_node("query_node")
    node2 = create_llm_node("analysis_node")
    
    # Add nodes and connect them
    node1_id = flow.add_node(node1)
    node2_id = flow.add_node(node2)
    
    flow.add_edge(Edge(
        source_node=node1_id,
        target_node=node2_id,
        name="query_to_analysis"
    ))
    
    try:
        # Initialize nodes
        await node1.initialize()
        await node2.initialize()
        
        # Process test query
        async with TimingContext("Ollama flow") as timing:
            result = await flow.process({
                "prompt": "What is quantum computing?",
                "analysis_prompt": lambda r: f"Summarize this explanation: {r['response']}"
            })
        
        logger.info(f"Query response: {result[node1_id]['response'][:100]}...")
        logger.info(f"Analysis: {result[node2_id]['response'][:100]}...")
        
        return timing.execution_time
        
    finally:
        # Cleanup
        await node1.cleanup()
        await node2.cleanup()

async def main():
    """Run a series of simple flow tests."""
    logger.info("Starting flow tests...")
    
    # Check if Ollama is available
    ollama_available = await check_ollama_availability()
    if not ollama_available:
        logger.warning("Ollama is not accessible. Will skip Ollama tests.")
    
    # Test linear scaling
    linear_times = []
    for n in [2, 4, 8]:
        time = await test_linear_flow(n)
        linear_times.append((n, time))
    
    # Test branch scaling
    branch_times = []
    for d in [2, 3]:
        time = await test_branching_flow(depth=d, branch_factor=2)
        branch_times.append((d, time))
    
    # Test Ollama integration
    ollama_time = None
    if ollama_available:
        logger.info("\nTesting Ollama integration...")
        try:
            ollama_time = await test_ollama_flow()
        except Exception as e:
            logger.error(f"Ollama test failed: {e}")
    
    # Print summary
    logger.info("\nTest Results:")
    logger.info("Linear Flow Tests:")
    for nodes, time in linear_times:
        logger.info(f"  {nodes} nodes: {time:.2f}s")
    
    logger.info("\nBranching Flow Tests:")
    for depth, time in branch_times:
        logger.info(f"  Depth {depth}: {time:.2f}s")
    
    if ollama_time is not None:
        logger.info(f"\nOllama Flow Test: {ollama_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())