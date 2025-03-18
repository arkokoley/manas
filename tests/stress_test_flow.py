"""Stress testing framework for MAS flow architecture."""
import asyncio
import random
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import signal
from contextlib import contextmanager

from manas_ai.flow import Flow
from manas_ai.base import Node, Edge
from manas_ai.llm import LLMNode, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    max_nodes: int = 100
    max_depth: int = 10
    max_concurrent_tasks: int = 20
    test_duration: int = 300  # seconds
    fault_injection_rate: float = 0.1
    recovery_delay: float = 1.0
    timeout: float = 30.0

@dataclass
class StressTestResult:
    """Results from a stress test run."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    timeouts: int
    recovered_failures: int
    avg_response_time: float
    max_response_time: float
    memory_peak: float
    error_distribution: Dict[str, int]

class FaultInjector:
    """Injects faults into the system for stress testing."""
    
    def __init__(self, rate: float = 0.1):
        self.rate = rate
        self.injected_faults = 0
        self.recovered_faults = 0
    
    async def maybe_inject_fault(self, node: Node) -> bool:
        """Potentially inject a fault into a node."""
        if random.random() < self.rate:
            fault_type = random.choice([
                'timeout',
                'error',
                'memory_leak',
                'high_latency'
            ])
            
            if fault_type == 'timeout':
                await asyncio.sleep(30)  # Simulate timeout
            elif fault_type == 'error':
                raise Exception("Injected fault: Random error")
            elif fault_type == 'memory_leak':
                # Simulate memory leak
                node.metadata['leak'] = ['x' * 1000000 for _ in range(100)]
            elif fault_type == 'high_latency':
                await asyncio.sleep(random.uniform(5, 15))
            
            self.injected_faults += 1
            return True
        return False
    
    async def recover_fault(self, node: Node):
        """Attempt to recover from a fault."""
        if 'leak' in node.metadata:
            del node.metadata['leak']
        self.recovered_faults += 1
        await asyncio.sleep(1)  # Recovery time

class ChaosFlow(Flow):
    """Flow implementation with chaos engineering capabilities."""
    
    def __init__(self, name: str, fault_injector: FaultInjector):
        super().__init__(name=name)
        self.fault_injector = fault_injector
        self.metrics: Dict[str, List[float]] = {
            'response_times': [],
            'error_counts': [],
            'memory_usage': []
        }
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs with fault injection and recovery."""
        start_time = time.time()
        
        try:
            # Maybe inject fault before processing
            for node in self.nodes.values():
                if await self.fault_injector.maybe_inject_fault(node):
                    logger.warning(f"Fault injected in node {node.name}")
                    # Attempt recovery
                    await self.fault_injector.recover_fault(node)
            
            # Normal processing
            result = await super().process(inputs)
            
            # Record metrics
            response_time = time.time() - start_time
            self.metrics['response_times'].append(response_time)
            
            return result
            
        except Exception as e:
            self.metrics['error_counts'].append(1)
            logger.error(f"Error in flow processing: {str(e)}")
            raise

class StressTester:
    """Orchestrates stress testing of flow architecture."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.fault_injector = FaultInjector(config.fault_injection_rate)
        self.results_dir = Path("stress_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _create_complex_flow(self, n_nodes: int, max_depth: int) -> ChaosFlow:
        """Create a complex flow topology for stress testing."""
        flow = ChaosFlow(f"stress_flow_{n_nodes}", self.fault_injector)
        
        # Create nodes with varying complexity
        nodes = []
        for i in range(n_nodes):
            node = LLMNode(
                name=f"node_{i}",
                config=LLMConfig(
                    provider_name="ollama",
                    provider_config={
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434/v1"
                    },
                    temperature=0.7
                )
            )
            node_id = flow.add_node(node)
            nodes.append(node_id)
        
        # Create complex connections
        for i in range(1, len(nodes)):
            # Create primary connections
            flow.add_edge(Edge(
                source_node=nodes[i-1],
                target_node=nodes[i],
                name=f"edge_{i}"
            ))
            
            # Add some cross-connections for complexity
            if i > 2 and i < len(nodes) - 1:
                # Add skip connections
                flow.add_edge(Edge(
                    source_node=nodes[i-2],
                    target_node=nodes[i],
                    name=f"skip_edge_{i}"
                ))
                
                # Add feedback connections (within depth limit)
                if i > max_depth:
                    flow.add_edge(Edge(
                        source_node=nodes[i],
                        target_node=nodes[i-max_depth],
                        name=f"feedback_edge_{i}"
                    ))
        
        return flow
    
    @contextmanager
    def _timeout_handler(self, timeout: float):
        """Handle timeouts for operations."""
        def handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        # Set timeout
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))
        
        try:
            yield
        finally:
            # Disable timeout
            signal.alarm(0)
    
    async def run_single_test(self, 
                            n_nodes: int, 
                            n_tasks: int) -> StressTestResult:
        """Run a single stress test with specified parameters."""
        flow = self._create_complex_flow(n_nodes, self.config.max_depth)
        
        successful_tasks = 0
        failed_tasks = 0
        timeouts = 0
        response_times = []
        errors: Dict[str, int] = {}
        
        # Generate test tasks
        tasks = [
            {
                "prompt": f"Complex task {i} with multiple steps",
                "complexity": random.randint(1, 5)
            }
            for i in range(n_tasks)
        ]
        
        async def process_task(task: Dict[str, Any]) -> None:
            nonlocal successful_tasks, failed_tasks, timeouts
            
            try:
                with self._timeout_handler(self.config.timeout):
                    start_time = time.time()
                    await flow.process(task)
                    response_times.append(time.time() - start_time)
                    successful_tasks += 1
            except TimeoutError:
                timeouts += 1
                failed_tasks += 1
            except Exception as e:
                error_type = type(e).__name__
                errors[error_type] = errors.get(error_type, 0) + 1
                failed_tasks += 1
        
        # Process tasks with controlled concurrency
        task_groups = [tasks[i:i + self.config.max_concurrent_tasks] 
                      for i in range(0, len(tasks), self.config.max_concurrent_tasks)]
        
        for task_group in task_groups:
            await asyncio.gather(
                *(process_task(task) for task in task_group),
                return_exceptions=True
            )
        
        return StressTestResult(
            total_tasks=n_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            timeouts=timeouts,
            recovered_failures=self.fault_injector.recovered_faults,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            memory_peak=psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            error_distribution=errors
        )
    
    async def run_stress_test_suite(self):
        """Run a comprehensive stress test suite."""
        test_configs = [
            # Test increasing node count
            {"n_nodes": n, "n_tasks": 50} 
            for n in [10, 20, 50, 100]
        ] + [
            # Test high concurrency
            {"n_nodes": 20, "n_tasks": n} 
            for n in [100, 200, 500]
        ]
        
        results = []
        for config in test_configs:
            logger.info(f"Running stress test with config: {config}")
            result = await self.run_single_test(**config)
            results.append({
                "config": config,
                "result": dataclasses.asdict(result)
            })
            
            # Save intermediate results
            self._save_results(results)
            
            # Optional: Add cooldown period between tests
            await asyncio.sleep(5)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save test results to file."""
        timestamp = int(time.time())
        output_file = self.results_dir / f"stress_test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved stress test results to {output_file}")
    
    def analyze_results(self, results: List[Dict[str, Any]]):
        """Analyze stress test results and generate report."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            # Convert results to DataFrame for analysis
            df_rows = []
            for r in results:
                config = r["config"]
                result = r["result"]
                df_rows.append({
                    "n_nodes": config["n_nodes"],
                    "n_tasks": config["n_tasks"],
                    "success_rate": result["successful_tasks"] / result["total_tasks"],
                    "avg_response_time": result["avg_response_time"],
                    "memory_peak": result["memory_peak"],
                    "timeouts": result["timeouts"],
                    "recovered_failures": result["recovered_failures"]
                })
            
            df = pd.DataFrame(df_rows)
            
            # Create visualization directory
            viz_dir = self.results_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Plot key metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # Success rate vs Node Count
            sns.scatterplot(data=df, x="n_nodes", y="success_rate", ax=axes[0, 0])
            axes[0, 0].set_title("Success Rate vs Node Count")
            
            # Response Time vs Task Count
            sns.scatterplot(data=df, x="n_tasks", y="avg_response_time", ax=axes[0, 1])
            axes[0, 1].set_title("Response Time vs Task Count")
            
            # Memory Usage vs Node Count
            sns.scatterplot(data=df, x="n_nodes", y="memory_peak", ax=axes[1, 0])
            axes[1, 0].set_title("Memory Usage vs Node Count")
            
            # Recovery Rate
            recovery_rate = df["recovered_failures"] / (df["timeouts"] + df["recovered_failures"])
            sns.scatterplot(data=df, x="n_nodes", y=recovery_rate, ax=axes[1, 1])
            axes[1, 1].set_title("Recovery Rate vs Node Count")
            
            plt.tight_layout()
            plt.savefig(viz_dir / "stress_test_metrics.png")
            
            # Generate statistical analysis
            stats = {
                "overall_success_rate": df["success_rate"].mean(),
                "avg_response_time": df["avg_response_time"].mean(),
                "max_memory_usage": df["memory_peak"].max(),
                "recovery_rate": recovery_rate.mean(),
                "correlation_nodes_success": df["n_nodes"].corr(df["success_rate"]),
                "correlation_tasks_response": df["n_tasks"].corr(df["avg_response_time"])
            }
            
            with open(viz_dir / "statistical_analysis.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            return stats
            
        except ImportError:
            logger.warning("matplotlib, seaborn, and pandas required for analysis")
            return None

async def main():
    """Run a complete stress test suite."""
    config = StressTestConfig(
        max_nodes=100,
        max_depth=10,
        max_concurrent_tasks=20,
        test_duration=300,
        fault_injection_rate=0.1
    )
    
    tester = StressTester(config)
    results = await tester.run_stress_test_suite()
    
    # Analyze results
    stats = tester.analyze_results(results)
    if stats:
        logger.info("Stress Test Results:")
        logger.info(f"Overall Success Rate: {stats['overall_success_rate']:.2%}")
        logger.info(f"Average Response Time: {stats['avg_response_time']:.2f}s")
        logger.info(f"Max Memory Usage: {stats['max_memory_usage']:.2f}MB")
        logger.info(f"Recovery Rate: {stats['recovery_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())