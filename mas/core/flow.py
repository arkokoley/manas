from typing import Dict, List, Set, Any, Iterable
from uuid import UUID
import asyncio
from collections import defaultdict

from .base import Node, Edge

class Flow(Node):
    """Orchestrates execution of a graph of nodes."""
    
    def __init__(self, name: str, description: str = None):
        super().__init__(name=name, description=description)
        self.nodes: Dict[UUID, Node] = {}
        self.edges: List[Edge] = []
        self._adjacency_list: Dict[UUID, Set[UUID]] = defaultdict(set)
        
    def add_node(self, node: Node) -> UUID:
        """Add a node to the flow."""
        self.nodes[node.id] = node
        return node.id
        
    def add_edge(self, edge: Edge):
        """Add an edge connecting two nodes."""
        if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the flow")
        self.edges.append(edge)
        self._adjacency_list[edge.source_node].add(edge.target_node)
        
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the flow, running nodes in dependency order."""
        results = {}
        in_degree = self._calculate_in_degree()
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        node_results = {}  # Store individual node results
        
        while queue:
            # Process nodes that can be executed in parallel
            current_batch = queue.copy()
            queue.clear()
            
            # Execute current batch of nodes
            tasks = []
            for node_id in current_batch:
                node = self.nodes[node_id]
                node_inputs = await self._gather_inputs(node_id, node_results)
                node_inputs.update(inputs)  # Include original inputs for each node
                tasks.append(self._execute_node(node, node_inputs))
            
            batch_results = await asyncio.gather(*tasks)
            
            # Store results and update queue
            for node_id, result in zip(current_batch, batch_results):
                node_results[node_id] = result
                
                # Update in_degree and queue for next nodes
                for next_node in self._adjacency_list[node_id]:
                    in_degree[next_node] -= 1
                    if in_degree[next_node] == 0:
                        queue.append(next_node)
        
        return node_results
    
    def _calculate_in_degree(self) -> Dict[UUID, int]:
        """Calculate initial in-degree for all nodes."""
        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_node] += 1
        return in_degree
    
    async def _gather_inputs(self, node_id: UUID, results: Dict[UUID, Dict[str, Any]]) -> Dict[str, Any]:
        """Gather inputs for a node from its incoming edges."""
        inputs = {}
        for edge in self.edges:
            if edge.target_node == node_id and edge.source_node in results:
                transformed_data = await edge.transform(results[edge.source_node])
                inputs.update(transformed_data)
        return inputs
    
    async def _execute_node(self, node: Node, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node with its inputs."""
        if not node.validate_inputs(inputs):
            raise ValueError(f"Invalid inputs for node {node.name}")
        return await node.process(inputs)

    async def batch_process(self, batch_inputs: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process multiple inputs in batches."""
        results = []
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]
            tasks = [self.process(inputs) for inputs in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    async def stream_process(self, input_stream: Iterable[Dict[str, Any]], batch_size: int = 10):
        """Stream process inputs, yielding results as they become available."""
        batch = []
        for inputs in input_stream:
            batch.append(inputs)
            if len(batch) >= batch_size:
                results = await self.batch_process(batch, batch_size)
                for result in results:
                    yield result
                batch = []
        
        if batch:
            results = await self.batch_process(batch, batch_size)
            for result in results:
                yield result