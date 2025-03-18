"""Example implementation of a tool-using agent."""
import asyncio
from typing import Dict, Any, List, Callable, Optional
from manas_ai.agent import Agent
from manas_ai.llm import LLMNode, LLMConfig
from .utils import Tool, logger, create_ollama_node, get_common_tools

class ToolUsingAgent(Agent):
    """Agent that can use tools to solve tasks."""
    def __init__(self, name: str, provider: str = "ollama", 
                 provider_config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name)
        
        # Default to ollama if no provider config
        if provider_config is None:
            provider_config = {
                "model": "llama3.2",
                "base_url": "http://localhost:11434/v1"
            }
        
        # Initialize LLM
        self.llm = create_ollama_node(
            name=f"{name}_llm",
            model=provider_config.get("model", "llama3.2"),
            base_url=provider_config.get("base_url", "http://localhost:11434/v1")
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
    
    def add_common_tools(self):
        """Add common file and text operation tools."""
        for tool in get_common_tools():
            self.add_tool(tool)
    
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


async def main():
    """Example usage of the ToolUsingAgent."""
    # Create an agent
    agent = ToolUsingAgent(
        name="research_assistant",
        provider="ollama",
        provider_config={
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1"
        }
    )
    
    # Add common tools
    agent.add_common_tools()
    
    # Initialize the agent
    await agent.llm.initialize()
    
    try:
        # Process a task
        task = "Read the README.md file, analyze it to count words and lines, then create a summary in a new file called summary.txt"
        logger.info(f"Processing task: {task}")
        
        result = await agent.process({"task": task})
        
        # Print the results
        logger.info("\nExecution Plan:")
        for i, step in enumerate(result["decision"]["plan"], 1):
            logger.info(f"  {i}. {step}")
        
        logger.info(f"\nSelected Tool: {result['result']['tool']}")
        logger.info(f"Result: {result['result'].get('result', '')[:100]}...")
        
        logger.info("\nAgent's Analysis:")
        logger.info(result["observation"]["summary"])
        
    finally:
        # Always cleanup
        await agent.llm.cleanup()

if __name__ == "__main__":
    asyncio.run(main())