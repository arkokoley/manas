"""Formal proof runner for validating flow correctness properties."""
import asyncio
import json
import time
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
from uuid import UUID
from pathlib import Path

import networkx as nx

from mas.core.flow import Flow
from mas.core.base import Node, Edge
from mas.core.llm import LLMNode, LLMConfig

from tests.utils import (
    logger, check_dependencies, create_llm_node, TimingContext
)

@dataclass
class CorrectnessProperty:
    """Represents a correctness property to verify."""
    name: str
    description: str
    checker: callable
    severity: str  # 'critical', 'warning', 'info'

@dataclass
class ProofResult:
    """Result of a correctness proof."""
    property_name: str
    valid: bool
    counterexample: Optional[Dict[str, Any]] = None
    proof_time: float = 0.0

class FlowProofRunner:
    """Validates correctness properties of flows."""
    
    def __init__(self):
        self.properties: List[CorrectnessProperty] = []
        self._register_properties()
    
    def _register_properties(self):
        """Register all correctness properties to verify."""
        
        def check_acyclic(flow: Flow) -> ProofResult:
            """Verify that the flow graph is acyclic."""
            with TimingContext("Acyclic proof") as timing:
                # Convert flow to networkx graph
                G = nx.DiGraph()
                for node_id in flow.nodes:
                    G.add_node(node_id)
                for edge in flow.edges:
                    G.add_edge(edge.source_node, edge.target_node)
                
                try:
                    cycle = nx.find_cycle(G)
                    return ProofResult(
                        property_name="acyclic",
                        valid=False,
                        counterexample={"cycle": [str(n) for n in cycle]},
                        proof_time=timing.execution_time or 0.0
                    )
                except nx.NetworkXNoCycle:
                    return ProofResult(
                        property_name="acyclic",
                        valid=True,
                        proof_time=timing.execution_time or 0.0
                    )
        
        def check_reachability(flow: Flow) -> ProofResult:
            """Verify that all nodes are reachable from entry points."""
            with TimingContext("Reachability proof") as timing:
                # Find entry nodes (nodes with no incoming edges)
                entry_nodes = set(flow.nodes.keys())
                for edge in flow.edges:
                    entry_nodes.discard(edge.target_node)
                
                if not entry_nodes:
                    return ProofResult(
                        property_name="reachability",
                        valid=False,
                        counterexample={"error": "No entry nodes found"},
                        proof_time=timing.execution_time or 0.0
                    )
                
                # Check reachability from each entry node
                G = nx.DiGraph()
                for node_id in flow.nodes:
                    G.add_node(node_id)
                for edge in flow.edges:
                    G.add_edge(edge.source_node, edge.target_node)
                
                reachable = set()
                for entry in entry_nodes:
                    reachable.update(nx.descendants(G, entry))
                    reachable.add(entry)
                
                unreachable = set(flow.nodes.keys()) - reachable
                if unreachable:
                    return ProofResult(
                        property_name="reachability",
                        valid=False,
                        counterexample={"unreachable_nodes": [str(n) for n in unreachable]},
                        proof_time=timing.execution_time or 0.0
                    )
                
                return ProofResult(
                    property_name="reachability",
                    valid=True,
                    proof_time=timing.execution_time or 0.0
                )
        
        def check_deterministic_execution(flow: Flow) -> ProofResult:
            """Verify that execution paths are deterministic."""
            with TimingContext("Determinism proof") as timing:
                # Check for nodes with multiple outgoing edges without explicit conditions
                G = nx.DiGraph()
                for edge in flow.edges:
                    G.add_edge(edge.source_node, edge.target_node)
                
                for node in G.nodes():
                    successors = list(G.successors(node))
                    if len(successors) > 1:
                        # Check if edges have conditions
                        edges_from_node = [e for e in flow.edges if e.source_node == node]
                        unconditioned_edges = [e for e in edges_from_node if "condition" not in e.metadata]
                        if len(unconditioned_edges) > 1:
                            return ProofResult(
                                property_name="deterministic_execution",
                                valid=False,
                                counterexample={
                                    "node": str(node),
                                    "unconditioned_branches": len(unconditioned_edges)
                                },
                                proof_time=timing.execution_time or 0.0
                            )
                
                return ProofResult(
                    property_name="deterministic_execution",
                    valid=True,
                    proof_time=timing.execution_time or 0.0
                )
        
        def check_resource_safety(flow: Flow) -> ProofResult:
            """Verify that the flow properly manages resources."""
            with TimingContext("Resource safety proof") as timing:
                # Check that all LLM nodes have cleanup hooks
                nodes_without_cleanup = []
                for node_id, node in flow.nodes.items():
                    if not hasattr(node, 'cleanup') or not callable(node.cleanup):
                        nodes_without_cleanup.append(str(node_id))
                
                if nodes_without_cleanup:
                    return ProofResult(
                        property_name="resource_safety",
                        valid=False,
                        counterexample={"nodes_without_cleanup": nodes_without_cleanup},
                        proof_time=timing.execution_time or 0.0
                    )
                
                return ProofResult(
                    property_name="resource_safety",
                    valid=True,
                    proof_time=timing.execution_time or 0.0
                )
        
        def check_parallel_safety(flow: Flow) -> ProofResult:
            """Verify that parallel execution is safe."""
            with TimingContext("Parallel safety proof") as timing:
                G = nx.DiGraph()
                for edge in flow.edges:
                    G.add_edge(edge.source_node, edge.target_node)
                
                # Check for shared state between parallel branches
                parallel_violations = []
                
                # Find entry nodes
                entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
                if not entry_nodes:
                    # Use any node as entry if no clear entry point
                    entry_nodes = list(G.nodes())[:1] if G.nodes() else []
                
                if entry_nodes:
                    for node in G.nodes():
                        if node in entry_nodes:
                            continue
                            
                        paths_to_node = []
                        for entry in entry_nodes:
                            try:
                                paths_to_node.extend(nx.all_simple_paths(G, entry, node))
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                continue
                                
                        if len(paths_to_node) > 1:
                            # Check if node has mutable state
                            node_obj = flow.nodes[node]
                            if hasattr(node_obj, 'state') or any(
                                not name.startswith('_') and not callable(getattr(node_obj, name))
                                and name not in ('name',)  # Exclude standard attributes
                                for name in dir(node_obj) if hasattr(getattr(node_obj, name), '__set__')
                            ):
                                parallel_violations.append(str(node))
                
                if parallel_violations:
                    return ProofResult(
                        property_name="parallel_safety",
                        valid=False,
                        counterexample={"nodes_with_shared_state": parallel_violations},
                        proof_time=timing.execution_time or 0.0
                    )
                
                return ProofResult(
                    property_name="parallel_safety",
                    valid=True,
                    proof_time=timing.execution_time or 0.0
                )
        
        # Register all properties
        self.properties.extend([
            CorrectnessProperty(
                name="acyclic",
                description="Flow graph must be acyclic",
                checker=check_acyclic,
                severity="critical"
            ),
            CorrectnessProperty(
                name="reachability",
                description="All nodes must be reachable from entry points",
                checker=check_reachability,
                severity="critical"
            ),
            CorrectnessProperty(
                name="deterministic_execution",
                description="Execution paths must be deterministic",
                checker=check_deterministic_execution,
                severity="warning"
            ),
            CorrectnessProperty(
                name="resource_safety",
                description="Resources must be properly managed",
                checker=check_resource_safety,
                severity="critical"
            ),
            CorrectnessProperty(
                name="parallel_safety",
                description="Parallel execution must be safe",
                checker=check_parallel_safety,
                severity="warning"
            )
        ])
    
    async def verify_flow(self, flow: Flow) -> List[ProofResult]:
        """Verify all correctness properties for a flow."""
        results = []
        
        for prop in self.properties:
            logger.info(f"Verifying property: {prop.name}")
            try:
                result = prop.checker(flow)
                results.append(result)
                
                status = "✓" if result.valid else "✗"
                logger.info(f"{status} {prop.name}: {'Valid' if result.valid else 'Invalid'}")
                if not result.valid:
                    level = logging.ERROR if prop.severity == 'critical' else logging.WARNING
                    logger.log(level, f"Counterexample: {result.counterexample}")
                
            except Exception as e:
                logger.error(f"Error verifying {prop.name}: {e}")
                results.append(ProofResult(
                    property_name=prop.name,
                    valid=False,
                    counterexample={"error": str(e)}
                ))
        
        return results
    
    def save_proof_results(self, results: List[ProofResult], output_dir: Path):
        """Save proof results to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        output_file = output_dir / f"flow_proofs_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(
                [{
                    "property": r.property_name,
                    "valid": r.valid,
                    "counterexample": r.counterexample,
                    "proof_time": r.proof_time
                } for r in results],
                f,
                indent=2
            )
        
        logger.info(f"Saved proof results to {output_file}")
    
    def generate_proof_report(self, results: List[ProofResult], output_dir: Path):
        """Generate a detailed proof report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        report = []
        report.append("# Flow Correctness Proof Report\n")
        
        # Summary
        valid_count = sum(1 for r in results if r.valid)
        total_count = len(results)
        report.append(f"## Summary\n")
        report.append(f"- Properties Verified: {total_count}")
        report.append(f"- Properties Valid: {valid_count}")
        report.append(f"- Properties Invalid: {total_count - valid_count}\n")
        
        # Detailed Results
        report.append("## Detailed Results\n")
        for result in results:
            status = "✓ Valid" if result.valid else "✗ Invalid"
            report.append(f"### {result.property_name} ({status})")
            report.append(f"- Proof Time: {result.proof_time:.3f}s")
            if not result.valid:
                report.append(f"- Counterexample:")
                report.append(f"```json")
                report.append(json.dumps(result.counterexample, indent=2))
                report.append(f"```\n")
            report.append("")
        
        # Write report
        report_file = output_dir / f"proof_report_{int(time.time())}.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Generated proof report at {report_file}")

async def main():
    """Run proofs on example flows."""
    # Check required dependencies
    check_dependencies({
        "networkx": "networkx",
        "matplotlib": "matplotlib"
    })
    
    logger.info("Starting flow proof validation...")
    
    try:
        from mas.core.llm import LLMNode, LLMConfig
        from mas.core.flow import Flow
        from mas.core.base import Edge
        
        # Create a test flow
        flow = Flow(name="test_flow")
        
        # Add some nodes
        nodes = []
        for i in range(5):
            node = create_llm_node(f"node_{i}")
            nodes.append(flow.add_node(node))
        
        # Add edges
        for i in range(len(nodes) - 1):
            flow.add_edge(Edge(
                source_node=nodes[i],
                target_node=nodes[i + 1],
                name=f"edge_{i}"
            ))
        
        # Run proofs
        proof_runner = FlowProofRunner()
        results = await proof_runner.verify_flow(flow)
        
        # Save results and generate report
        output_dir = Path("proof_results")
        proof_runner.save_proof_results(results, output_dir)
        proof_runner.generate_proof_report(results, output_dir)
        
        # Print summary
        valid_count = sum(1 for r in results if r.valid)
        logger.info(f"\nProof Summary:")
        logger.info(f"Total Properties: {len(results)}")
        logger.info(f"Valid Properties: {valid_count}")
        logger.info(f"Invalid Properties: {len(results) - valid_count}")
        
    except Exception as e:
        logger.error(f"Error during proof validation: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())