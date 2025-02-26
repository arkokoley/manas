"""Example demonstrating a dynamic Ollama-based task execution system."""
import asyncio
from typing import Dict, Any, List
from core.llm import LLMNode, LLMConfig
from core.flow import Flow
from .tool_using_agent import ToolUsingAgent, Tool

class OllamaQueryNode:
    """A node that can query Ollama and process results."""
    
    def __init__(self, name: str, model: str = "llama3.2", base_url: str = "http://localhost:11434/v1"):
        self.llm = LLMNode(
            name=f"{name}_llm",
            config=LLMConfig(
                provider_name="ollama",
                provider_config={
                    "model": model,
                    "base_url": base_url
                },
                temperature=0.7
            )
        )
        self._initialized = False
        self.results = []
    
    async def initialize(self):
        """Initialize the LLM."""
        if not self._initialized:
            await self.llm.initialize()
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._initialized:
            await self.llm.cleanup()
            self._initialized = False
    
    async def query(self, question: str) -> Dict[str, Any]:
        """Send a query to Ollama and get response."""
        if not self._initialized:
            await self.initialize()
        
        result = await self.llm.process({"prompt": question})
        self.results.append({"question": question, "response": result["response"]})
        return result

class DynamicOllamaFlow:
    """Manages a dynamic flow of Ollama queries based on a plan."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.base_url = base_url
        self.flow = Flow(name="dynamic_ollama_flow")
        self.nodes: List[OllamaQueryNode] = []
        self.task_agent = ToolUsingAgent(
            name="planner",
            provider="ollama",
            provider_config={
                "model": model,
                "base_url": base_url
            }
        )
        
    async def initialize(self):
        """Initialize the flow system."""
        await self.task_agent.llm.initialize()
        
        # Add planning capabilities to the agent
        self.task_agent.add_tool(Tool(
            name="execute_query",
            description="Execute a query and get response. Args: question(str)",
            func=self._execute_query
        ))
        
        self.task_agent.add_tool(Tool(
            name="analyze_results",
            description="Analyze multiple responses and create a summary. Args: responses(str)",
            func=self._analyze_results
        ))
    
    async def _execute_query(self, question: str) -> str:
        """Execute a single query."""
        node = OllamaQueryNode(f"query_{len(self.nodes)}", self.model, self.base_url)
        self.nodes.append(node)
        await node.initialize()
        result = await node.query(question)
        return result["response"]
    
    async def _analyze_results(self, responses: str) -> str:
        """Analyze a set of responses and create a summary."""
        analysis_node = OllamaQueryNode("analysis", self.model, self.base_url)
        await analysis_node.initialize()
        
        prompt = f"""Analyze these responses and provide a comprehensive summary:

{responses}

Focus on:
1. Key concepts and their relationships
2. Main insights and findings
3. Practical implications

Provide a structured summary."""
        
        result = await analysis_node.query(prompt)
        await analysis_node.cleanup()
        return result["response"]
    
    async def cleanup(self):
        """Cleanup all resources."""
        await self.task_agent.cleanup()
        for node in self.nodes:
            await node.cleanup()
    
    async def execute_plan(self, task: str) -> Dict[str, Any]:
        """Execute a task by generating and following a plan."""
        print("Generating execution plan...")
        
        # First, get the plan
        plan_result = await self.task_agent.process({"task": task})
        
        print("\nExecuting plan steps...")
        all_responses = []
        
        # Execute each step in the plan
        for step in plan_result["decision"]["plan"]:
            if "ask" in step.lower() or "query" in step.lower():
                # Extract the question from the step
                question = step.split("ask", 1)[-1].strip()
                if not question:
                    question = step
                
                print(f"\nExecuting step: {step}")
                response = await self._execute_query(question)
                all_responses.append(f"Question: {question}\nAnswer: {response}\n")
        
        # Analyze results
        print("\nAnalyzing results...")
        final_analysis = await self._analyze_results("\n".join(all_responses))
        
        return {
            "plan": plan_result["decision"]["plan"],
            "individual_responses": all_responses,
            "final_analysis": final_analysis
        }

async def main():
    flow_system = DynamicOllamaFlow(model="llama3.2")
    
    try:
        await flow_system.initialize()
        
        task = """
        Help me understand the following concepts:
        1. What is quantum entanglement?
        2. How does it relate to quantum computing?
        3. What are some practical applications?
        Analyze each answer and provide a summary of key points.
        """
        
        print("Starting task execution:", task)
        
        result = await flow_system.execute_plan(task)
        
        print("\nExecution Plan:")
        for i, step in enumerate(result["plan"], 1):
            print(f"{i}. {step}")
        
        print("\nIndividual Responses:")
        for response in result["individual_responses"]:
            print("-" * 50)
            print(response)
        
        print("\nFinal Analysis:")
        print(result["final_analysis"])
        
    finally:
        await flow_system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())