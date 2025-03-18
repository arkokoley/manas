"""Node implementation for Model Context Protocol (MCP) servers."""
from typing import Dict, Any, Optional, List, Union
import logging
import json
import aiohttp
from dataclasses import dataclass, field
from core.base import Node
from .factory import register_node
from core.llm import LLMNode, LLMConfig

logger = logging.getLogger(__name__)

@dataclass
class MCPConfig:
    """Configuration for MCP nodes."""
    model: str
    _servers: List[str] = field(default_factory=list)
    provider_name: str = "ollama"
    provider_config: Dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    @property
    def servers(self) -> List[str]:
        return self._servers

    @servers.setter
    def servers(self, value: List[str]):
        self._servers = value

@register_node("mcp")
class MCPNode(Node):
    """Node for communicating with MCP servers."""
    
    def __init__(self, name: str, mcp_config: Dict, config: MCPConfig):
        super().__init__(name=name)
        self.config = config
        self.mcp_config = mcp_config
        self._servers = {}
        self._initialized = False
        self._session = None
        
        llm_config = LLMConfig(
            provider_name=config.provider_name,
            provider_config={"model": config.model, **config.provider_config},
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        self.llm = LLMNode(f"{name}_llm", llm_config)

    async def initialize(self):
        """Initialize MCP server connections."""
        if not self._initialized:
            try:
                await self.llm.initialize()
                self._session = aiohttp.ClientSession()
                
                for server_name, server_config in self.mcp_config["mcpServers"].items():
                    if server_name in self.config.servers:
                        try:
                            port = None
                            for arg in server_config["args"]:
                                if isinstance(arg, str):
                                    if "--port=" in arg:
                                        port = int(arg.split("=")[1])
                                    elif arg.isdigit():
                                        port = int(arg)
                            
                            if not port:
                                continue
                                
                            self._servers[server_name] = {
                                "name": server_name,
                                "port": port,
                                "config": server_config,
                                "url": f"http://localhost:{port}/api"
                            }
                            logger.info(f"Configured MCP server {server_name} at port {port}")
                            
                        except Exception as e:
                            logger.error(f"Error configuring server {server_name}: {e}")
                            continue
                            
                if not self._servers:
                    raise RuntimeError("No MCP servers were successfully configured")
                    
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize MCP node: {e}")
                if self._session:
                    await self._session.close()
                raise

    async def cleanup(self):
        """Cleanup resources."""
        if self._initialized:
            if self._session:
                await self._session.close()
            await self.llm.cleanup()
            self._servers.clear()
            self._initialized = False

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs using MCP servers."""
        if not self._initialized:
            await self.initialize()

        action = inputs.get("action")
        if not action:
            raise ValueError("Input must contain 'action'")

        data = inputs.get("data", {})
        servers = inputs.get("servers", self.config.servers)
        prompt = inputs.get("prompt")

        if not all(s in self.config.servers for s in servers):
            raise ValueError("Requested servers must be subset of configured servers")

        responses = {}
        
        for server_name in servers:
            if server_name in self._servers:
                try:
                    url = f"{self._servers[server_name]['url']}/{action}"
                    headers = {"Content-Type": "application/json"}
                    
                    async with self._session.post(url, json=data, headers=headers) as response:
                        response.raise_for_status()
                        result = await response.json()
                        responses[server_name] = {
                            "status": "success",
                            "result": result
                        }
                        
                except Exception as e:
                    logger.error(f"Error from server {server_name}: {e}")
                    responses[server_name] = {"error": str(e)}

        if prompt and responses:
            try:
                # Format server results for LLM
                results = []
                for server_name, response in responses.items():
                    if "result" in response:
                        results.append({
                            "server": server_name,
                            "data": response["result"]
                        })
                
                if results:
                    context = json.dumps(results, indent=2)
                    enhanced_prompt = f"{prompt}\n\nContext from MCP servers:\n{context}"
                    llm_response = await self.llm.process({"prompt": enhanced_prompt})
                    responses["llm"] = llm_response["response"]
                    
            except Exception as e:
                logger.error(f"LLM processing error: {e}")
                responses["llm"] = {"error": str(e)}

        return responses
