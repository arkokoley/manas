"""Flow orchestration for executing graph-based workflows."""
from typing import Dict, List, Set, Any, Iterable, Optional, Callable, Tuple
from uuid import UUID
import asyncio
import logging
from collections import defaultdict
from core.base import Node, Edge

# Set up logging
logger = logging.getLogger(__name__)


class FlowExecutionError(Exception):
    """Error raised when flow execution fails."""
    def __init__(self, message: str, node_id: UUID = None, node_name: str = None, details: Dict = None):
        self.node_id = node_id
        self.node_name = node_name
        self.details = details or {}
        super().__init__(message)


class Flow(Node):
    """Orchestrates execution of a graph of nodes."""
    
    def __init__(self, name: str, description: str = None):
        super().__init__(name=name, description=description)
        self.nodes: Dict[UUID, Node] = {}
        self.edges: List[Edge] = []
        self._adjacency_list: Dict[UUID, Set[UUID]] = defaultdict(set)
        self._node_names: Dict[str, UUID] = {}  # For looking up nodes by name
        
    def add_node(self, node: Node) -> UUID:
        """
        Add a node to the flow.
        
        Args:
            node: The node to add
            
        Returns:
            UUID of the added node
            
        Raises:
            ValueError: If a node with the same name already exists
        """
        if node.name in self._node_names:
            raise ValueError(f"A node with name '{node.name}' already exists in this flow")
            
        self.nodes[node.id] = node
        self._node_names[node.name] = node.id
        logger.debug(f"Added node: {node.name} ({node.id})")
        return node.id
        
    def add_edge(self, edge: Edge):
        """
        Add an edge connecting two nodes.
        
        Args:
            edge: The edge to add
            
        Raises:
            ValueError: If source or target nodes don't exist in the flow
        """
        if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the flow")
            
        self.edges.append(edge)
        self._adjacency_list[edge.source_node].add(edge.target_node)
        logger.debug(f"Added edge: {edge.name} from {self.nodes[edge.source_node].name} to {self.nodes[edge.target_node].name}")
        
    def get_node_by_name(self, name: str) -> Optional[Node]:
        """
        Get a node by name.
        
        Args:
            name: Name of the node
            
        Returns:
            The node if found, None otherwise
        """
        node_id = self._node_names.get(name)
        if node_id:
            return self.nodes.get(node_id)
        return None
        
    def connect_nodes(self, source_name: str, target_name: str, edge_name: str = None, description: str = None) -> Edge:
        """
        Connect two nodes by name with an edge.
        
        Args:
            source_name: Name of the source node
            target_name: Name of the target node
            edge_name: Name for the edge (defaults to f"{source_name}_to_{target_name}")
            description: Optional description for the edge
            
        Returns:
            The created edge
            
        Raises:
            ValueError: If source or target nodes don't exist
        """
        source = self.get_node_by_name(source_name)
        target = self.get_node_by_name(target_name)
        
        if not source:
            raise ValueError(f"Source node '{source_name}' not found")
        if not target:
            raise ValueError(f"Target node '{target_name}' not found")
            
        edge_name = edge_name or f"{source_name}_to_{target_name}"
        edge = Edge(source.id, target.id, edge_name, description)
        self.add_edge(edge)
        return edge
        
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the flow, running nodes in dependency order.
        
        Args:
            inputs: Input data for the flow
            
        Returns:
            Dictionary containing output from all nodes
            
        Raises:
            FlowExecutionError: If execution fails
        """
        try:
            results = {}
            in_degree = self._calculate_in_degree()
            queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
            node_results = {}  # Store individual node results
            executed_nodes = set()
            
            if not queue:
                # Check for cycles in the graph
                if self.nodes:
                    raise FlowExecutionError("No starting nodes found. The flow may contain cycles.")
                else:
                    raise FlowExecutionError("Flow contains no nodes to execute")
            
            while queue:
                # Process nodes that can be executed in parallel
                current_batch = queue.copy()
                queue.clear()
                
                logger.debug(f"Executing batch of {len(current_batch)} nodes")
                
                # Execute current batch of nodes
                tasks = []
                for node_id in current_batch:
                    node = self.nodes[node_id]
                    node_inputs = await self._gather_inputs(node_id, node_results)
                    node_inputs.update(inputs)  # Include original inputs for each node
                    tasks.append(self._execute_node_with_context(node, node_inputs))
                
                # Execute nodes and collect results
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results, handling any exceptions
                for node_id, result in zip(current_batch, batch_results):
                    node = self.nodes[node_id]
                    
                    if isinstance(result, Exception):
                        # Handle node execution failure
                        err_msg = f"Error executing node '{node.name}': {str(result)}"
                        logger.error(err_msg)
                        raise FlowExecutionError(err_msg, node_id=node_id, node_name=node.name, 
                                                details={"error": str(result)})
                    
                    # Store successful result
                    node_results[node_id] = result
                    executed_nodes.add(node_id)
                    logger.debug(f"Node '{node.name}' executed successfully")
                    
                    # Update in-degree and queue for next nodes
                    for next_node in self._adjacency_list[node_id]:
                        in_degree[next_node] -= 1
                        if in_degree[next_node] == 0:
                            queue.append(next_node)
            
            # Check if all nodes were executed
            if len(executed_nodes) < len(self.nodes):
                unexecuted = [self.nodes[nid].name for nid in self.nodes if nid not in executed_nodes]
                logger.warning(f"Not all nodes were executed. Unexecuted nodes: {unexecuted}")
            
            return node_results
            
        except FlowExecutionError:
            # Re-raise FlowExecutionError as is
            raise
        except Exception as e:
            # Wrap other exceptions
            logger.error(f"Unexpected error in flow execution: {str(e)}")
            raise FlowExecutionError(f"Flow execution failed: {str(e)}")
    
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
    
    async def _execute_node_with_context(self, node: Node, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node with its inputs and handle context properly."""
        node_name = node.name
        logger.debug(f"Executing node: {node_name}")
        
        try:
            if not node.validate_inputs(inputs):
                raise ValueError(f"Invalid inputs for node {node_name}")
                
            # Initialize node if needed
            if hasattr(node, 'initialize'):
                await node.initialize()
                
            # Process node
            result = await node.process(inputs)
            
            # Clean up node resources if needed
            if hasattr(node, 'cleanup'):
                await node.cleanup()
                
            return result
        except Exception as e:
            logger.error(f"Error executing node {node_name}: {str(e)}")
            # Ensure cleanup on error
            if hasattr(node, 'cleanup'):
                try:
                    await node.cleanup()
                except Exception as cleanup_err:
                    logger.warning(f"Error during node cleanup: {str(cleanup_err)}")
            raise
    
    async def batch_process(self, batch_inputs: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in batches.
        
        Args:
            batch_inputs: List of input dictionaries
            batch_size: Number of inputs to process in each batch
            
        Returns:
            List of results, one for each input
        """
        results = []
        total = len(batch_inputs)
        
        for i in range(0, total, batch_size):
            batch = batch_inputs[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} inputs)")
            
            tasks = [self.process(inputs) for inputs in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in batch item {i+j}: {str(result)}")
                    # Include error info in result
                    processed_results.append({"error": str(result)})
                else:
                    processed_results.append(result)
            
            results.extend(processed_results)
            
        return results
    
    async def stream_process(self, input_stream: Iterable[Dict[str, Any]], batch_size: int = 10):
        """
        Stream process inputs, yielding results as they become available.
        
        Args:
            input_stream: Iterable of input dictionaries
            batch_size: Number of inputs to process in each batch
            
        Yields:
            Result dictionaries as they become available
        """
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
                
    def visualize(self) -> Dict[str, Any]:
        """
        Generate visualization data for the flow.
        
        Returns:
            Dictionary with nodes and edges in a format suitable for visualization
        """
        nodes_data = []
        for node_id, node in self.nodes.items():
            nodes_data.append({
                "id": str(node_id),
                "name": node.name,
                "type": node.__class__.__name__,
                "description": node.metadata.description
            })
            
        edges_data = []
        for edge in self.edges:
            edges_data.append({
                "id": str(edge.id),
                "name": edge.name,
                "source": str(edge.source_node),
                "target": str(edge.target_node),
                "description": edge.metadata.description
            })
            
        return {
            "nodes": nodes_data,
            "edges": edges_data
        }


# Export classes
__all__ = ["Flow", "FlowExecutionError"]