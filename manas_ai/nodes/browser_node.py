"""Node implementation for browser automation using browser-use Agent."""
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field
from browser_use import Agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from manas_ai.base import Node
from .factory import register_node

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class BrowserConfig:
    """Configuration for browser nodes."""
    model: str = "deepseek-r1"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"  # Default for Ollama
    use_vision: bool = False
    timeout: float = 30.0

@register_node("browser")
class BrowserNode(Node):
    """Node for browser automation tasks using browser-use Agent."""
    
    def __init__(self, name: str, config: Optional[BrowserConfig] = None):
        super().__init__(name=name)
        self.config = config or BrowserConfig()
        self._agent = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize browser agent."""
        if not self._initialized:
            # Create ChatOpenAI instance with proper configuration
            llm = ChatOpenAI(
                base_url=self.config.base_url,
                model=self.config.model,
                api_key=SecretStr(self.config.api_key),
                timeout=self.config.timeout
            )
            
            # Initialize Agent with the LLM
            self._agent = Agent(
                task=None,  # Will be set during process
                llm=llm,
                use_vision=self.config.use_vision
            )
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup browser resources."""
        if self._agent:
            # await self._agent.close()
            self._agent = None
            self._initialized = False
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process browser automation tasks.
        
        Args:
            inputs: Dictionary containing:
                - task: The task description for the agent
                - url: Optional URL to include in the task
                
        Returns:
            Dictionary containing task results
            
        Raises:
            ValueError: If required inputs are missing
        """
        if not self._initialized:
            await self.initialize()
            
        task = inputs.get("task")
        if not task:
            raise ValueError("Input must contain 'task' description")
            
        try:
            # Include URL in task if provided
            url = inputs.get("url")
            if url:
                task = f"Go to {url} and {task}"
            
            # Update the agent's task and run
            self._agent.task = task
            result = await self._agent.run()
            return {"status": "success", "result": result}
                
        except Exception as e:
            logger.error("Browser task failed: %s", str(e))
            return {"status": "error", "error": str(e)}

def create_browser_node(
    name: str,
    model: str = "deepseek-r1",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    use_vision: bool = False,
    timeout: float = 30.0
) -> BrowserNode:
    """
    Convenience function to create a browser node.
    
    Args:
        name: Node name
        model: Model name to use
        base_url: API base URL
        api_key: API key (default: "ollama")
        use_vision: Whether to enable vision capabilities
        timeout: Operation timeout in seconds
        
    Returns:
        Configured BrowserNode instance
    """
    config = BrowserConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        use_vision=use_vision,
        timeout=timeout
    )
    return BrowserNode(name=name, config=config)