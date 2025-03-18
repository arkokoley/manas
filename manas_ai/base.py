"""Base classes for all nodes in the graph system."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from uuid import UUID, uuid4

@dataclass
class NodeMetadata:
    """Model for node metadata with type validation."""
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)


class Node:
    """Base class for all nodes in the graph system."""
    def __init__(self, name: str, description: Optional[str] = None):
        self.id: UUID = uuid4()
        self.name: str = name
        self.metadata = NodeMetadata(description=description)
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return outputs."""
        raise NotImplementedError
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate that required inputs are present."""
        return True
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        
    async def initialize(self):
        """Initialize the node (setup resources, etc)."""
        pass
        
    async def cleanup(self):
        """Cleanup node resources."""
        pass


@dataclass
class EdgeMetadata:
    """Model for edge metadata with type validation."""
    description: Optional[str] = None
    weight: float = 1.0
    custom: Dict[str, Any] = field(default_factory=dict)


class Edge:
    """Represents a directed connection between nodes."""
    def __init__(self, source_node: UUID, target_node: UUID, name: str, description: Optional[str] = None):
        self.id: UUID = uuid4()
        self.source_node: UUID = source_node
        self.target_node: UUID = target_node
        self.name: str = name
        self.metadata = EdgeMetadata(description=description)
    
    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data as it flows through the edge."""
        return data


# Export all classes
__all__ = ['Node', 'Edge', 'NodeMetadata', 'EdgeMetadata']