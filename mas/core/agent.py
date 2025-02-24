from typing import Any, Dict, List, Optional
from .base import Node

class Agent(Node):
    """
    Base Agent class that can be used in flows.
    Agents are specialized nodes that can maintain state and make decisions.
    """
    def __init__(self, name: str, description: Optional[str] = None):
        super().__init__(name=name, description=description)
        self.memory: Dict[str, Any] = {}
        self.capabilities: List[str] = []
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process information and make decisions."""
        raise NotImplementedError
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actions based on decisions."""
        raise NotImplementedError
    
    async def observe(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process results of actions and update internal state."""
        raise NotImplementedError
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implement the agent's think-act-observe cycle."""
        decision = await self.think(inputs)
        result = await self.act(decision)
        observation = await self.observe(result)
        return {
            "decision": decision,
            "result": result,
            "observation": observation
        }

class AgentRegistry:
    """
    Registry for managing and discovering agents.
    """
    _agents: Dict[str, type[Agent]] = {}
    
    @classmethod
    def register(cls, agent_class: type[Agent]):
        """Register a new agent type."""
        cls._agents[agent_class.__name__] = agent_class
        return agent_class
    
    @classmethod
    def get_agent(cls, name: str) -> type[Agent]:
        """Get an agent class by name."""
        if name not in cls._agents:
            raise KeyError(f"Agent {name} not found in registry")
        return cls._agents[name]
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent types."""
        return list(cls._agents.keys())