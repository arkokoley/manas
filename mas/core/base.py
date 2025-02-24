"""Base classes for all nodes in the graph system."""
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

class Node:
    """Base class for all nodes in the graph system."""
    def __init__(self, name: str, description: Optional[str] = None):
        self.id: UUID = uuid4()
        self.name: str = name
        self.description: Optional[str] = description
        self.metadata: Dict[str, Any] = {}
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return outputs."""
        raise NotImplementedError
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate that required inputs are present."""
        return True

class Edge:
    """Represents a directed connection between nodes."""
    def __init__(self, source_node: UUID, target_node: UUID, name: str):
        self.id: UUID = uuid4()
        self.source_node: UUID = source_node
        self.target_node: UUID = target_node
        self.name: str = name
        self.metadata: Dict[str, Any] = {}
    
    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data as it flows through the edge."""
        return data