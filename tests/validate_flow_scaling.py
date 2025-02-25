"""Real-world validation tests for flow architecture scaling."""
import asyncio
import time
import logging
from typing import Dict, Any, List
from pathlib import Path
import json

from mas.core.flow import Flow
from mas.core.base import Node, Edge
from mas.core.llm import LLMNode, LLMConfig
from mas.core.rag import RAGNode, RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationTest:
    """Runs real-world validation tests for flow architecture."""
    
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        self.base_url = base_url
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
    
    def _create_llm_node(self, name: str, model: str = "llama3.2") -> LLMNode:
        """Create an LLM node with specified configuration."""
        return LLMNode(
            name=name,
            config=LLMConfig(
                provider_name="ollama",
                provider_config={
                    "model": model,
                    "base_url": self.base_url
                }
            )
        )
    
    def _create_rag_node(self, name: str, embedding_node: LLMNode) -> RAGNode:
        """Create a RAG node for context-aware processing."""
        return RAGNode(
            name=name,
            config=RAGConfig(
                vectorstore_type="faiss",
                vectorstore_config={
                    "dimension": 384,
                    "metric": "cosine"
                }
            ),
            embedding_node=embedding_node
        )
    
    async def test_query_complexity_scaling(self) -> Dict[str, Any]:
        """Test how the system scales with increasing query complexity."""
        flow = Flow(name="complexity_test")
        
        # Create nodes for different aspects of processing
        query_node = self._create_llm_node("query_processor")
        research_node = self._create_llm_node("research")
        analysis_node = self._create_llm_node("analyzer")
        summary_node = self._create_llm_node("summarizer")
        
        # Add nodes
        query_id = flow.add_node(query_node)
        research_id = flow.add_node(research_node)
        analysis_id = flow.add_node(analysis_node)
        summary_id = flow.add_node(summary_node)
        
        # Connect nodes
        flow.add_edge(Edge(source_node=query_id, target_node=research_id))
        flow.add_edge(Edge(source_node=research_id, target_node=analysis_id))
        flow.add_edge(Edge(source_node=analysis_id, target_node=summary_id))
        
        # Test queries of increasing complexity
        queries = [
            "What is quantum computing?",  # Simple
            "Explain how quantum entanglement affects quantum computing performance.",  # Medium
            "Compare and contrast different quantum computing architectures, including their advantages and limitations for specific applications.",  # Complex
            "Analyze the potential impact of quantum supremacy on current cryptographic systems and propose mitigation strategies.",  # Very Complex
        ]
        
        results = []
        for query in queries:
            start_time = time.time()
            result = await flow.process({
                "prompt": query,
                "research_prompt": lambda r: f"Research deeply: {r['response']}",
                "analysis_prompt": lambda r: f"Analyze findings: {r['response']}",
                "summary_prompt": lambda r: f"Summarize analysis: {r['response']}"
            })
            execution_time = time.time() - start_time
            
            results.append({
                "query_complexity": len(query.split()),
                "execution_time": execution_time,
                "response_length": len(result[summary_id]["response"])
            })
        
        return {
            "test_name": "query_complexity_scaling",
            "results": results
        }
    
    async def test_context_scaling(self) -> Dict[str, Any]:
        """Test how the system scales with increasing context size."""
        flow = Flow(name="context_test")
        
        # Create nodes
        embed_node = self._create_llm_node("embedder")
        rag_node = self._create_rag_node("rag_processor", embed_node)
        query_node = self._create_llm_node("query_processor")
        
        # Add nodes
        rag_id = flow.add_node(rag_node)
        query_id = flow.add_node(query_node)
        
        # Connect nodes
        flow.add_edge(Edge(source_node=rag_id, target_node=query_id))
        
        # Test with increasing context sizes
        context_sizes = [1, 5, 10, 20]  # Number of documents
        results = []
        
        for size in context_sizes:
            # Generate test documents
            docs = [
                {"content": f"Document {i} with important information about quantum computing and its applications in various fields.",
                 "metadata": {"source": f"doc_{i}.txt"}}
                for i in range(size)
            ]
            
            # Add documents to RAG
            await rag_node.add_documents(docs)
            
            # Test query with context
            start_time = time.time()
            result = await flow.process({
                "query": "What are the main applications of quantum computing?",
                "k": size  # Number of relevant documents to retrieve
            })
            execution_time = time.time() - start_time
            
            results.append({
                "context_size": size,
                "execution_time": execution_time,
                "response_length": len(result[query_id]["response"])
            })
        
        return {
            "test_name": "context_scaling",
            "results": results
        }
    
    async def test_parallel_scaling(self) -> Dict[str, Any]:
        """Test how the system scales with parallel processing."""
        flow = Flow(name="parallel_test")
        
        # Create parallel processing branches
        branches = 4
        branch_nodes = []
        
        # Create nodes for each branch
        for i in range(branches):
            processor = self._create_llm_node(f"processor_{i}")
            analyzer = self._create_llm_node(f"analyzer_{i}")
            
            proc_id = flow.add_node(processor)
            anal_id = flow.add_node(analyzer)
            
            flow.add_edge(Edge(source_node=proc_id, target_node=anal_id))
            branch_nodes.append((proc_id, anal_id))
        
        # Create aggregator node
        aggregator = self._create_llm_node("aggregator")
        agg_id = flow.add_node(aggregator)
        
        # Connect all branches to aggregator
        for _, anal_id in branch_nodes:
            flow.add_edge(Edge(source_node=anal_id, target_node=agg_id))
        
        # Test with increasing concurrent tasks
        concurrency_levels = [1, 2, 4, 8]
        results = []
        
        for concurrency in concurrency_levels:
            tasks = [
                {
                    "prompt": f"Task {i}: Analyze quantum computing application",
                    "branch_assignments": {
                        node_id: f"Subtask {i} for branch {j}"
                        for j, (node_id, _) in enumerate(branch_nodes)
                    }
                }
                for i in range(concurrency)
            ]
            
            start_time = time.time()
            batch_results = await flow.batch_process(tasks, batch_size=concurrency)
            execution_time = time.time() - start_time
            
            results.append({
                "concurrency": concurrency,
                "execution_time": execution_time,
                "tasks_completed": len(batch_results)
            })
        
        return {
            "test_name": "parallel_scaling",
            "results": results
        }
    
    async def run_all_tests(self):
        """Run all validation tests."""
        logger.info("Starting validation tests...")
        
        tests = [
            self.test_query_complexity_scaling(),
            self.test_context_scaling(),
            self.test_parallel_scaling()
        ]
        
        results = await asyncio.gather(*tests)
        
        # Save results
        timestamp = int(time.time())
        output_file = self.results_dir / f"validation_results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation results saved to {output_file}")
        return results

async def main():
    """Run the validation test suite."""
    validator = ValidationTest()
    results = await validator.run_all_tests()
    
    # Print summary
    print("\nValidation Test Summary:")
    for test in results:
        print(f"\n{test['test_name']}:")
        for result in test["results"]:
            print(f"  {result}")

if __name__ == "__main__":
    asyncio.run(main())