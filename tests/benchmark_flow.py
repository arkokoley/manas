"""Benchmarking framework for testing MAS flow scalability."""
import asyncio
import time
import psutil
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from uuid import UUID

from core.flow import Flow
from core.base import Node, Edge
from core.llm import LLMNode, LLMConfig

from tests.utils import (
    logger, check_dependencies, check_ollama_availability,
    create_llm_node, create_linear_flow, measure_execution_time
)

@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmarking."""
    execution_time: float
    memory_usage: float
    node_count: int
    edge_count: int
    max_depth: int
    total_tokens: int
    concurrent_tasks: int
    success_rate: float
    errors: List[str]
    throughput: float  # tasks per second

class FlowBenchmark:
    """Benchmark framework for testing flow scalability."""
    
    def __init__(self, 
                 base_model: str = "llama3.2",
                 base_url: str = "http://localhost:11434/v1",
                 results_dir: str = "benchmark_results"):
        self.base_model = base_model
        self.base_url = base_url
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.metrics: List[BenchmarkMetrics] = []
    
    def _create_test_node(self, name: str, complexity: int = 1) -> Node:
        """Create a test node with configurable complexity."""
        return create_llm_node(name, self.base_model, self.base_url)
    
    def _create_linear_flow(self, n_nodes: int) -> Flow:
        """Create a linear flow with n nodes."""
        return create_linear_flow(
            f"linear_flow_{n_nodes}", 
            n_nodes,
            self._create_test_node
        )
    
    def _create_branching_flow(self, depth: int, branch_factor: int) -> Flow:
        """Create a tree-like flow with specified depth and branching factor."""
        flow = Flow(name=f"branch_flow_d{depth}_b{branch_factor}")
        
        def add_level(parent_id: Optional[UUID], current_depth: int):
            if current_depth >= depth:
                return
            
            for i in range(branch_factor):
                node = self._create_test_node(f"node_d{current_depth}_b{i}")
                node_id = flow.add_node(node)
                
                if parent_id is not None:
                    flow.add_edge(Edge(
                        source_node=parent_id,
                        target_node=node_id,
                        name=f"edge_d{current_depth}_b{i}"
                    ))
                
                add_level(node_id, current_depth + 1)
        
        # Create root node
        root = self._create_test_node("root")
        root_id = flow.add_node(root)
        add_level(root_id, 0)
        
        return flow
    
    async def _measure_execution(self, 
                               flow: Flow, 
                               inputs: List[Dict[str, Any]],
                               concurrency: int = 1) -> BenchmarkMetrics:
        """Execute flow and measure performance metrics."""
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        errors = []
        completed = 0
        
        async def process_batch(batch: List[Dict[str, Any]]):
            nonlocal completed
            try:
                results = await flow.batch_process(batch, batch_size=concurrency)
                completed += len(results)
                return results
            except Exception as e:
                errors.append(str(e))
                return []
        
        # Process inputs in batches
        batches = [inputs[i:i + concurrency] for i in range(0, len(inputs), concurrency)]
        results = []
        
        for batch in batches:
            batch_results = await process_batch(batch)
            results.extend(batch_results)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        success_rate = completed / len(inputs) if inputs else 1.0
        throughput = completed / execution_time if execution_time > 0 else 0
        
        return BenchmarkMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            node_count=len(flow.nodes),
            edge_count=len(flow.edges),
            max_depth=self._calculate_max_depth(flow),
            total_tokens=0,  # TODO: Implement token counting
            concurrent_tasks=concurrency,
            success_rate=success_rate,
            errors=errors,
            throughput=throughput
        )
    
    def _calculate_max_depth(self, flow: Flow) -> int:
        """Calculate the maximum depth of the flow graph."""
        depths = {}
        
        def calc_depth(node_id: UUID) -> int:
            if node_id in depths:
                return depths[node_id]
            
            incoming_edges = [e for e in flow.edges if e.target_node == node_id]
            if not incoming_edges:
                depths[node_id] = 0
                return 0
            
            max_parent_depth = max(calc_depth(e.source_node) for e in incoming_edges)
            depths[node_id] = max_parent_depth + 1
            return depths[node_id]
        
        return max(calc_depth(node_id) for node_id in flow.nodes.keys()) if flow.nodes else 0
    
    async def run_horizontal_scaling_test(self, 
                                        node_counts: List[int],
                                        inputs_per_node: int = 10,
                                        concurrency: int = 1):
        """Test scalability with increasing number of nodes."""
        for n_nodes in node_counts:
            flow = self._create_linear_flow(n_nodes)
            
            # Generate test inputs
            inputs = [
                {"prompt": f"Test input {i} for {n_nodes} nodes"}
                for i in range(n_nodes * inputs_per_node)
            ]
            
            metrics = await self._measure_execution(flow, inputs, concurrency)
            self.metrics.append(metrics)
            
            logger.info(f"Completed horizontal scaling test with {n_nodes} nodes:")
            logger.info(f"Execution time: {metrics.execution_time:.2f}s")
            logger.info(f"Throughput: {metrics.throughput:.2f} tasks/s")
            logger.info(f"Memory usage: {metrics.memory_usage:.2f}MB")
    
    async def run_depth_scaling_test(self,
                                   depths: List[int],
                                   branch_factor: int = 2,
                                   inputs_per_level: int = 5,
                                   concurrency: int = 1):
        """Test scalability with increasing flow depth."""
        for depth in depths:
            flow = self._create_branching_flow(depth, branch_factor)
            
            # Generate test inputs
            inputs = [
                {"prompt": f"Test input {i} for depth {depth}"}
                for i in range(depth * inputs_per_level)
            ]
            
            metrics = await self._measure_execution(flow, inputs, concurrency)
            self.metrics.append(metrics)
            
            logger.info(f"Completed depth scaling test with depth {depth}:")
            logger.info(f"Execution time: {metrics.execution_time:.2f}s")
            logger.info(f"Throughput: {metrics.throughput:.2f} tasks/s")
            logger.info(f"Memory usage: {metrics.memory_usage:.2f}MB")
    
    async def run_concurrent_scaling_test(self,
                                        concurrency_levels: List[int],
                                        n_nodes: int = 5,
                                        inputs_per_node: int = 10):
        """Test scalability with increasing concurrency."""
        flow = self._create_linear_flow(n_nodes)
        inputs = [
            {"prompt": f"Test input {i} for concurrency test"}
            for i in range(n_nodes * inputs_per_node)
        ]
        
        for concurrency in concurrency_levels:
            metrics = await self._measure_execution(flow, inputs, concurrency)
            self.metrics.append(metrics)
            
            logger.info(f"Completed concurrency test with {concurrency} concurrent tasks:")
            logger.info(f"Execution time: {metrics.execution_time:.2f}s")
            logger.info(f"Throughput: {metrics.throughput:.2f} tasks/s")
            logger.info(f"Memory usage: {metrics.memory_usage:.2f}MB")
    
    def save_results(self, test_name: str):
        """Save benchmark results to file."""
        results = {
            "test_name": test_name,
            "timestamp": time.time(),
            "metrics": [vars(m) for m in self.metrics]
        }
        
        output_file = self.results_dir / f"{test_name}_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved benchmark results to {output_file}")
    
    def plot_results(self, test_name: str):
        """Generate plots of benchmark results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figures directory
            figures_dir = self.results_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            # Plot execution time vs different scaling factors
            plt.figure(figsize=(10, 6))
            
            if "horizontal" in test_name.lower():
                x = [m.node_count for m in self.metrics]
                xlabel = "Number of Nodes"
            elif "depth" in test_name.lower():
                x = [m.max_depth for m in self.metrics]
                xlabel = "Flow Depth"
            else:
                x = [m.concurrent_tasks for m in self.metrics]
                xlabel = "Concurrency Level"
            
            # Execution time
            plt.subplot(2, 2, 1)
            plt.plot(x, [m.execution_time for m in self.metrics], marker='o')
            plt.xlabel(xlabel)
            plt.ylabel("Execution Time (s)")
            plt.title("Execution Time Scaling")
            
            # Throughput
            plt.subplot(2, 2, 2)
            plt.plot(x, [m.throughput for m in self.metrics], marker='o')
            plt.xlabel(xlabel)
            plt.ylabel("Throughput (tasks/s)")
            plt.title("Throughput Scaling")
            
            # Memory usage
            plt.subplot(2, 2, 3)
            plt.plot(x, [m.memory_usage for m in self.metrics], marker='o')
            plt.xlabel(xlabel)
            plt.ylabel("Memory Usage (MB)")
            plt.title("Memory Usage Scaling")
            
            # Success rate
            plt.subplot(2, 2, 4)
            plt.plot(x, [m.success_rate for m in self.metrics], marker='o')
            plt.xlabel(xlabel)
            plt.ylabel("Success Rate")
            plt.title("Success Rate Scaling")
            
            plt.tight_layout()
            plt.savefig(figures_dir / f"{test_name}_scaling.png")
            plt.close()
            
            logger.info(f"Generated plots in {figures_dir}")
            
        except ImportError:
            logger.warning("matplotlib and seaborn required for plotting")

async def main():
    """Run a complete set of scaling tests."""
    # Check required dependencies
    check_dependencies({
        "psutil": "psutil",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pandas": "pandas",
        "networkx": "networkx"
    })
    
    # Verify Ollama is running
    ollama_available = await check_ollama_availability()
    if not ollama_available:
        logger.error("Ollama is not accessible. Please ensure Ollama is running.")
        return
    
    benchmark = FlowBenchmark(base_model="llama3.2")
    
    # Test horizontal scaling
    await benchmark.run_horizontal_scaling_test(
        node_counts=[2, 4, 8, 16, 32],
        inputs_per_node=5,
        concurrency=2
    )
    benchmark.save_results("horizontal_scaling")
    benchmark.plot_results("horizontal_scaling")
    benchmark.metrics.clear()
    
    # Test depth scaling
    await benchmark.run_depth_scaling_test(
        depths=[2, 3, 4, 5],
        branch_factor=2,
        inputs_per_level=5,
        concurrency=2
    )
    benchmark.save_results("depth_scaling")
    benchmark.plot_results("depth_scaling")
    benchmark.metrics.clear()
    
    # Test concurrent scaling
    await benchmark.run_concurrent_scaling_test(
        concurrency_levels=[1, 2, 4, 8],
        n_nodes=5,
        inputs_per_node=5
    )
    benchmark.save_results("concurrent_scaling")
    benchmark.plot_results("concurrent_scaling")

if __name__ == "__main__":
    asyncio.run(main())