"""Example implementation of a tool-using agent."""
import asyncio
from typing import Dict, Any, List, Callable, Optional
from mas.core.agent import Agent
from mas.core.llm import LLMNode, LLMConfig

class Tool:
    """Represents a tool that the agent can use."""
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)

class ToolUsingAgent(Agent):
    """Agent that can use tools to solve tasks."""
    def __init__(self, name: str, provider: str = "ollama", 
                 provider_config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name)
        
        # Default to ollama if no provider config
        if provider_config is None:
            provider_config = {
                "model": "llama2",
                "base_url": "http://localhost:11434/v1"
            }
        
        # Initialize LLM
        self.llm = LLMNode(
            name=f"{name}_llm",
            config=LLMConfig(
                provider_name=provider,
                provider_config=provider_config,
                temperature=0.7
            )
        )
        
        # Initialize tools registry
        self.tools: Dict[str, Tool] = {}
        
        # Initialize memory
        self.memory["conversation"] = []
        self.memory["current_plan"] = []
        self.memory["execution_state"] = {}
        
    def add_tool(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"{tool.name}: {tool.description}")
        return "\n".join(descriptions)
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the task and decide which tools to use."""
        tools_desc = self._get_tools_description()
        
        # Include execution state in prompt if available
        execution_state = ""
        if self.memory["execution_state"]:
            execution_state = "\nCurrent Execution State:\n"
            for key, value in self.memory["execution_state"].items():
                execution_state += f"- {key}: {value}\n"
        
        prompt = f"""Given the following task and available tools, create a plan and decide which tool to use next:

Available Tools:
{tools_desc}

Task: {context['task']}

Previous Actions: {self.memory.get('conversation', [])}
Current Plan: {self.memory.get('current_plan', [])}
{execution_state}

Your goal is to break down complex tasks into steps and use tools appropriately.
If the task requires multiple steps, plan them out and execute one at a time.
Use the execution state to keep track of intermediate results and use them in subsequent steps.

Respond in this format:
Plan: <list of steps to take>
Next Step: <current step being executed>
Tool: <tool_name or 'none'>
Arguments: <key=value pairs or 'none'>
Reasoning: <your reasoning>
"""
        
        result = await self.llm.process({"prompt": prompt})
        response = result["response"]
        
        # Parse the response
        lines = response.split("\n")
        decision = {
            "plan": [],
            "next_step": "",
            "tool": None,
            "arguments": {},
            "reasoning": ""
        }
        
        current_section = None
        for line in lines:
            if line.startswith("Plan:"):
                current_section = "plan"
                continue
            elif line.startswith("Next Step:"):
                current_section = "next_step"
                decision["next_step"] = line.replace("Next Step:", "").strip()
            elif line.startswith("Tool:"):
                tool = line.replace("Tool:", "").strip()
                decision["tool"] = None if tool == "none" else tool
            elif line.startswith("Arguments:"):
                args = line.replace("Arguments:", "").strip()
                if args != "none":
                    # Parse key=value pairs
                    pairs = [pair.strip() for pair in args.split(",")]
                    for pair in pairs:
                        if "=" in pair:
                            key, value = pair.split("=")
                            decision["arguments"][key.strip()] = value.strip()
            elif line.startswith("Reasoning:"):
                decision["reasoning"] = line.replace("Reasoning:", "").strip()
            elif current_section == "plan" and line.strip():
                # Add non-empty lines to plan
                if line.strip().startswith("- "):
                    decision["plan"].append(line.strip()[2:])
                elif line.strip().startswith("* "):
                    decision["plan"].append(line.strip()[2:])
                else:
                    decision["plan"].append(line.strip())
        
        # Update current plan in memory
        if decision["plan"]:
            self.memory["current_plan"] = decision["plan"]
        
        return decision

    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the chosen tool with arguments."""
        if not decision["tool"] or decision["tool"] not in self.tools:
            return {
                "error": f"Invalid or no tool specified: {decision['tool']}",
                "plan": decision.get("plan", []),
                "next_step": decision.get("next_step", "")
            }
        
        tool = self.tools[decision["tool"]]
        try:
            result = await tool.execute(**decision["arguments"])
            
            # Update execution state with tool result
            self.memory["execution_state"][decision["next_step"]] = {
                "tool": tool.name,
                "result": result
            }
            
            return {
                "success": True, 
                "result": result, 
                "tool": tool.name,
                "plan": decision.get("plan", []),
                "next_step": decision.get("next_step", "")
            }
        except Exception as e:
            return {
                "success": False, 
                "error": str(e), 
                "tool": tool.name,
                "plan": decision.get("plan", []),
                "next_step": decision.get("next_step", "")
            }

    async def observe(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the results and update memory."""
        # Add to conversation history
        self.memory["conversation"].append({
            "action": result.get("tool"),
            "success": result.get("success", False),
            "result": result.get("result", None),
            "error": result.get("error", None),
            "next_step": result.get("next_step", ""),
            "remaining_plan": result.get("plan", [])[1:] if result.get("plan") else []
        })
        
        # Generate observation summary with execution state context
        execution_state_str = "\n".join(
            f"{step}: {details['result']}"
            for step, details in self.memory["execution_state"].items()
        )
        
        prompt = f"""Analyze the result of the tool execution and plan next steps:

Tool Used: {result.get('tool')}
Success: {result.get('success', False)}
Result: {result.get('result', 'No result')}
Error: {result.get('error', 'No error')}
Current Step: {result.get('next_step', 'No step')}
Remaining Plan: {result.get('plan', [])[1:] if result.get('plan') else []}

Execution State:
{execution_state_str}

Provide:
1. A summary of what happened
2. Whether the current step was completed successfully
3. What should be done next based on the current execution state
"""
        
        analysis = await self.llm.process({"prompt": prompt})
        return {
            "summary": analysis["response"],
            "tool_result": result,
            "remaining_plan": result.get("plan", [])[1:] if result.get("plan") else []
        }

# Example usage with practical tools
async def main():
    # Create an agent
    agent = ToolUsingAgent(
        name="research_assistant",
        provider="ollama",
        provider_config={
            "model": "llama2",
            "base_url": "http://localhost:11434/v1"
        }
    )
    
    # File reading tool
    async def read_file_tool(path: str) -> str:
        try:
            with open(path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    # File writing tool
    async def write_file_tool(path: str, content: str) -> str:
        try:
            with open(path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    # HTTP request tool
    async def http_get_tool(url: str) -> str:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()
        except Exception as e:
            return f"Error making HTTP request: {str(e)}"
    
    # System command tool
    async def shell_tool(command: str) -> str:
        try:
            import subprocess
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30
            )
            return f"Exit code: {result.returncode}\nOutput:\n{result.stdout}\nErrors:\n{result.stderr}"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    # Text processing tool
    def process_text_tool(text: str, operation: str) -> str:
        ops = {
            "upper": str.upper,
            "lower": str.lower,
            "title": str.title,
            "words": lambda x: str(len(x.split())),
            "chars": len,
            "lines": lambda x: len(x.splitlines())
        }
        try:
            if operation not in ops:
                return f"Unknown operation. Available: {', '.join(ops.keys())}"
            return str(ops[operation](text))
        except Exception as e:
            return f"Error processing text: {str(e)}"
    
    # Add tools to agent
    agent.add_tool(Tool(
        name="read_file",
        description="Read content from a file. Args: path(str)",
        func=read_file_tool
    ))
    
    agent.add_tool(Tool(
        name="write_file",
        description="Write content to a file. Args: path(str), content(str)",
        func=write_file_tool
    ))
    
    agent.add_tool(Tool(
        name="http_get",
        description="Make HTTP GET request to a URL. Args: url(str)",
        func=http_get_tool
    ))
    
    agent.add_tool(Tool(
        name="shell_command",
        description="Execute a shell command. Args: command(str)",
        func=shell_tool
    ))
    
    agent.add_tool(Tool(
        name="process_text",
        description="Process text with operations (upper/lower/title/words/chars/lines). Args: text(str), operation(str)",
        func=process_text_tool
    ))
    
    # Initialize the agent
    await agent.llm.initialize()
    
    try:
        # Test tasks
        tasks = [
            "Read the contents of README.md, count the number of words, and write a summary to summary.txt",
            "Get the current system time using a shell command and convert it to uppercase",
            "Make an HTTP GET request to http://example.com and count the number of lines in the response"
        ]
        
        for task in tasks:
            print(f"\nExecuting task: {task}")
            result = await agent.process({"task": task})
            print("\nObservation:")
            print(result["observation"]["summary"])
            print("-" * 50)
            
    finally:
        await agent.llm.cleanup()

if __name__ == "__main__":
    asyncio.run(main())