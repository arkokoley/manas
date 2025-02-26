"""Agent implementation for decision making in the multi-agent system."""
from typing import Any, Dict, List, Optional, Callable, Set, Type
import logging
import inspect
from uuid import UUID
from core.base import Node

# Set up logging
logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class Agent(Node):
    """
    Base Agent class that can be used in flows.
    Agents are specialized nodes that follow a think-act-observe cycle
    and can maintain state across interactions.
    """
    def __init__(self, name: str, description: Optional[str] = None):
        super().__init__(name=name, description=description)
        self.memory: Dict[str, Any] = {}
        self.capabilities: List[str] = []
        self._state: Dict[str, Any] = {}
        
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process information and make decisions.
        
        Args:
            context: Current context for decision making
            
        Returns:
            Decision dictionary with reasoning and planned actions
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement think()")
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actions based on decisions.
        
        Args:
            decision: Decision dictionary from think phase
            
        Returns:
            Result dictionary containing action outcomes
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement act()")
    
    async def observe(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process results of actions and update internal state.
        
        Args:
            result: Result dictionary from act phase
            
        Returns:
            Observation dictionary with insights and state updates
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement observe()")
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement the agent's think-act-observe cycle.
        
        This is the main entry point for agent execution in a flow.
        
        Args:
            inputs: Input data for the agent
            
        Returns:
            Dictionary containing decision, result, and observation
            
        Raises:
            AgentError: If an error occurs during agent execution
        """
        try:
            # Merge inputs into state for persistence across calls
            self._update_state(inputs)
            
            # Execute think-act-observe cycle
            context = self._prepare_context()
            decision = await self.think(context)
            result = await self.act(decision)
            observation = await self.observe(result)
            
            # Store results in state
            self._state["last_decision"] = decision
            self._state["last_result"] = result
            self._state["last_observation"] = observation
            
            return {
                "decision": decision,
                "result": result,
                "observation": observation
            }
        except Exception as e:
            logger.error(f"Error in agent {self.name}: {str(e)}")
            raise AgentError(f"Agent execution failed: {str(e)}") from e
    
    def _update_state(self, inputs: Dict[str, Any]):
        """Update agent state with new inputs."""
        # Only update with keys that don't start with underscore (private)
        for key, value in inputs.items():
            if not key.startswith('_'):
                self._state[key] = value
    
    def _prepare_context(self) -> Dict[str, Any]:
        """Prepare context for think phase using current state."""
        return self._state.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get a copy of the current agent state."""
        return self._state.copy()
    
    def set_state(self, state: Dict[str, Any]):
        """Update the agent state."""
        self._state.update(state)
    
    def clear_state(self):
        """Clear the agent state."""
        self._state.clear()
    
    def add_capability(self, capability: str):
        """Add a capability to the agent."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities
    
    def list_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return self.capabilities.copy()


class Tool:
    """
    Represents a tool that agents can use.
    
    Tools provide specific capabilities to agents and can be registered
    to make them discoverable.
    """
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
        
        # Extract parameter info for documentation
        sig = inspect.signature(func)
        self.parameters = {
            name: {
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "default": None if param.default == inspect.Parameter.empty else param.default,
                "required": param.default == inspect.Parameter.empty and name != "self"
            }
            for name, param in sig.parameters.items()
            if name != "self"  # Skip self parameter for methods
        }
    
    async def __call__(self, *args, **kwargs) -> Any:
        """Call the tool function."""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(*args, **kwargs)
        return self.func(*args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class AgentRegistry:
    """
    Registry for managing and discovering agents.
    """
    _agents: Dict[str, Type[Agent]] = {}
    _tools: Dict[str, Tool] = {}
    
    @classmethod
    def register(cls, agent_class: Optional[Type[Agent]] = None):
        """
        Register a new agent type.
        
        Can be used as a decorator:
        
        @AgentRegistry.register
        class MyAgent(Agent):
            pass
        
        Args:
            agent_class: Agent class to register
            
        Returns:
            The registered agent class
        """
        def _register(cls_to_register):
            cls._agents[cls_to_register.__name__] = cls_to_register
            logger.info(f"Registered agent: {cls_to_register.__name__}")
            return cls_to_register
            
        # Handle both decorator with and without arguments
        if agent_class is not None:
            return _register(agent_class)
            
        return _register
    
    @classmethod
    def register_tool(cls, name: Optional[str] = None, description: Optional[str] = None):
        """
        Register a tool for agent use.
        
        Can be used as a decorator:
        
        @AgentRegistry.register_tool(name="calculator", description="Performs calculations")
        def calculate(expression):
            return eval(expression)
            
        Args:
            name: Tool name (defaults to function name)
            description: Tool description
            
        Returns:
            Decorator function
        """
        def decorator(func):
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or "No description provided"
            tool = Tool(tool_name, tool_desc, func)
            
            cls._tools[tool_name] = tool
            logger.info(f"Registered tool: {tool_name}")
            return func
            
        return decorator
    
    @classmethod
    def get_agent(cls, name: str) -> Type[Agent]:
        """Get an agent class by name."""
        if name not in cls._agents:
            raise KeyError(f"Agent {name} not found in registry")
        return cls._agents[name]
    
    @classmethod
    def get_tool(cls, name: str) -> Tool:
        """Get a tool by name."""
        if name not in cls._tools:
            raise KeyError(f"Tool {name} not found in registry")
        return cls._tools[name]
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent types."""
        return list(cls._agents.keys())
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tools."""
        return list(cls._tools.keys())
    
    @classmethod
    def get_tool_descriptions(cls) -> List[Dict[str, Any]]:
        """Get descriptions of all registered tools."""
        return [tool.to_dict() for tool in cls._tools.values()]


# Export classes
__all__ = ["Agent", "AgentError", "Tool", "AgentRegistry"]