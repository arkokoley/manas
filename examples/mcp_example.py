"""Example of using multiple Model Context Protocol servers with LLM integration."""
import asyncio
import datetime
import json
import subprocess
import os
from typing import Dict, List
from core.nodes import MCPNode, MCPConfig

async def start_mcp_servers(servers_config: Dict) -> List[subprocess.Popen]:
    """Start MCP servers from configuration."""
    processes = []
    for name, config in servers_config["mcpServers"].items():
        env = os.environ.copy()
        if "env" in config:
            env.update(config["env"])
        
        process = subprocess.Popen(
            [config["command"]] + config["args"],
            env=env
        )
        processes.append(process)
        print(f"Started {name} MCP server")
    return processes

async def cleanup_servers(processes: List[subprocess.Popen]):
    """Cleanup MCP server processes."""
    for process in processes:
        process.terminate()
        process.wait()

async def main():
    # Load MCP server configuration
    with open("/home/arkokoley/code/mas/examples/mcp_config.json", encoding='utf-8') as f:
        servers_config = json.load(f)

    # Start MCP servers
    server_processes = await start_mcp_servers(servers_config)

    try:
        # Create specialized MCP nodes for different tasks
        research = MCPNode(
            name="research",
            mcp_config=servers_config,
            config=MCPConfig(
                model="deepseek-r1",  # Using deepseek-coder model
                _servers=["arxiv-mcp-server", "zotero", "brave-search"],
                provider_name="ollama",  # Using Ollama as the provider
                temperature=0.7
            )
        )

        browser = MCPNode(
            name="browser",
            mcp_config=servers_config,
            config=MCPConfig(
                model="deepseek-r1",
                _servers=["puppeteer", "youtube-transcript"],
                provider_name="ollama",
                temperature=0.7
            )
        )

        storage = MCPNode(
            name="storage",
            mcp_config=servers_config,
            config=MCPConfig(
                model="deepseek-r1",
                _servers=["memory"],
                provider_name="ollama",
                temperature=0.7
            )
        )

        # Initialize nodes
        await asyncio.gather(
            research.initialize(),
            browser.initialize(),
            storage.initialize()
        )

        try:
            # Example: Research task using multiple servers and LLM
            research_result = await research.process({
                "action": "search",
                "data": {
                    "query": "Latest developments in LLM architecture",
                    "limit": 5
                },
                "servers": ["arxiv-mcp-server", "brave-search"],
                "prompt": "Analyze the search results and provide a concise summary of the latest developments in LLM architecture."
            })

            # Store research results with LLM analysis
            await storage.process({
                "action": "store",
                "data": research_result,
                "prompt": "Generate metadata tags and a structured summary for the stored research results."
            })

            # Example: Get video transcript and analyze it with LLM
            browser_result = await browser.process({
                "action": "get_transcript",
                "data": {
                    "url": "https://www.youtube.com/watch?v=Wmv9lIjsUIU"
                },
                "servers": ["youtube-transcript"],
                "prompt": "Extract key points and insights from this video transcript."
            })

            # Example: Browse web content and analyze with LLM
            web_result = await browser.process({
                "action": "browse",
                "data": {
                    "url": "https://www.wired.co.uk/article/china-social-credit-system-explained",
                    "elements": ["article", ".content"]
                },
                "servers": ["puppeteer"],
                "prompt": "Provide a comprehensive analysis of the social credit system based on this article."
            })

            # Print results including LLM analysis
            print("\nResearch Results:")
            print("-" * 50)
            if "llm" in research_result:
                print("\nLLM Analysis:")
                print(research_result["llm"])
            for server, response in research_result.items():
                if server != "llm":
                    print(f"\nFrom {server}:")
                    print(response.get("result", "No results"))

            print("\nTranscript Analysis:")
            print("-" * 50)
            if "llm" in browser_result:
                print("\nLLM Analysis:")
                print(browser_result["llm"])
            print("\nRaw Transcript:")
            print(browser_result.get("youtube-transcript", {}).get("result", "No transcript found"))

            print("\nWeb Content Analysis:")
            print("-" * 50)
            if "llm" in web_result:
                print("\nLLM Analysis:")
                print(web_result["llm"])
            print("\nRaw Content:")
            print(web_result.get("puppeteer", {}).get("result", "No content found"))

        finally:
            # Cleanup node resources
            await asyncio.gather(
                research.cleanup(),
                browser.cleanup(),
                storage.cleanup()
            )

    finally:
        # Cleanup MCP servers
        await cleanup_servers(server_processes)

if __name__ == "__main__":
    asyncio.run(main())