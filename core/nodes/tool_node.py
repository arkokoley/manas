"""Node implementation for executing tools."""
from typing import Dict, Any, List, Callable, Optional, Union
import asyncio
import logging
import inspect
from dataclasses import dataclass
from core.base import Node
from .factory import register_node
from .protocol import ToolNodeProtocol

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Tool:
    """Represents a tool that can be executed."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Dict[str, Any]]
    is_async: bool = False

    def __post_init__(self):
        """Verify that function is callable and set is_async."""
        if not callable(self.function):
            raise ValueError(f"Tool function must be callable: {self.name}")
        self.is_async = asyncio.iscoroutinefunction(self.function)

@register_node("tool")
class ToolNode(Node):
    """Node for executing registered tools."""
    
    def __init__(self, name: str, tools: Optional[List[Tool]] = None):
        super().__init__(name=name)
        self.tools: Dict[str, Tool] = {}
        
        # Register tools if provided
        if tools:
            for tool in tools:
                self.register_tool(tool)
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool for use with this node.
        
        Args:
            tool: Tool to register
        """
        if tool.name in self.tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered tools.
        
        Returns:
            Dictionary mapping tool names to their descriptions and parameters
        """
        return {
            name: {
                "description": tool.description,
                "parameters": tool.parameters
            }
            for name, tool in self.tools.items()
        }
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a specific tool with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool is not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        
        try:
            # Execute the tool asynchronously or synchronously
            if tool.is_async:
                return await tool.function(**kwargs)
            else:
                return tool.function(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs by executing the specified tool.
        
        Args:
            inputs: Dictionary containing at least 'tool' (name) and tool arguments
            
        Returns:
            Dictionary containing tool execution result
            
        Raises:
            ValueError: If required inputs are missing or invalid
        """
        tool_name = inputs.get("tool")
        if not tool_name:
            raise ValueError("Input must contain 'tool' with the tool name to execute")
        
        # Extract tool arguments, removing 'tool' key
        tool_args = {k: v for k, v in inputs.items() if k != "tool"}
        
        # Execute the tool and return result
        result = await self.execute_tool(tool_name, **tool_args)
        return {"result": result}

def create_tool(
    name: str, 
    description: str, 
    function: Callable,
    parameters: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tool:
    """
    Create a tool from a function, automatically detecting parameters if not provided.
    
    Args:
        name: Tool name
        description: Tool description
        function: Function to execute
        parameters: Optional parameter specifications
        
    Returns:
        Tool instance
    """
    if parameters is None:
        parameters = {}
        sig = inspect.signature(function)
        
        for param_name, param in sig.parameters.items():
            # Skip self and cls parameters
            if param_name in ('self', 'cls'):
                continue
                
            param_info = {
                "type": "string",  # Default type
                "description": f"Parameter: {param_name}"
            }
            
            # Handle default values
            if param.default is not inspect.Parameter.empty:
                param_info["default"] = param.default
                
            # Handle annotations
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is str:
                    param_info["type"] = "string"
                elif param.annotation is int:
                    param_info["type"] = "integer"
                elif param.annotation is float:
                    param_info["type"] = "number"
                elif param.annotation is bool:
                    param_info["type"] = "boolean"
                elif param.annotation is list or param.annotation is List:
                    param_info["type"] = "array"
                elif param.annotation is dict or param.annotation is Dict:
                    param_info["type"] = "object"
            
            parameters[param_name] = param_info
    
    return Tool(name=name, description=description, function=function, parameters=parameters)
