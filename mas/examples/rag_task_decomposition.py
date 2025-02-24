"""Example demonstrating RAG-enabled task decomposition with Ollama."""
import asyncio
from typing import List, Dict, Any
from pathlib import Path

from mas.core.agent import Agent
from mas.core.base import Edge
from mas.core.llm import LLMNode, LLMConfig, PromptTemplate
from mas.core.rag import RAGNode, RAGConfig
from mas.core.models import Document
from mas.core.flow import Flow

class RAGEnabledTaskAgent(Agent):
    """Agent that uses RAG for informed task decomposition and execution."""
    
    def __init__(self, 
        name: str,
        documents: List[Document],
        provider_config: Dict[str, Any],
        vectorstore_config: Dict[str, Any]
    ):
        super().__init__(name=name)
        
        # Set up LLM node with Ollama
        self.llm = LLMNode(
            name=f"{name}_llm",
            config=LLMConfig(
                provider="ollama",
                provider_config=provider_config,
                temperature=0.7
            )
        )
        
        # Set up RAG node with specified vector store
        self.rag = RAGNode(
            name=f"{name}_rag",
            config=RAGConfig(
                vectorstore_type="faiss",  # or "chroma"
                vectorstore_config=vectorstore_config,
                num_results=4,
                rerank_results=True
            ),
            embedding_node=self.llm,
            llm_node=self.llm
        )
        
        # Store documents for initialization
        self.documents = documents
        
        # Templates for different steps
        self.templates = {
            "decompose": PromptTemplate(
                template="Based on the context and task, break this down into subtasks:"
                        "\nContext: {context}"
                        "\nTask: {task}"
                        "\n\nSubtasks:",
                input_variables=["context", "task"]
            ),
            "execute": PromptTemplate(
                template="Complete this subtask using the provided context:"
                        "\nContext: {context}"
                        "\nTask: {subtask}"
                        "\n\nResponse:",
                input_variables=["context", "subtask"]
            ),
            "summarize": PromptTemplate(
                template="Synthesize the subtask results into a cohesive response:"
                        "\nTask: {task}"
                        "\nSubtask Results: {results}"
                        "\n\nFinal Response:",
                input_variables=["task", "results"]
            )
        }
    
    async def initialize(self):
        """Initialize RAG with documents."""
        await self.rag.initialize()
        await self.rag.add_documents(self.documents)
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use RAG to inform task decomposition."""
        # Get relevant context for the task
        rag_result = await self.rag.process({"query": context["task"]})
        
        # Use context to decompose task
        prompt = self.templates["decompose"].format(
            context=rag_result["context"],
            task=context["task"]
        )
        
        result = await self.llm.process({"prompt": prompt})
        subtasks = [s.strip() for s in result["response"].split("\n") if s.strip()]
        
        return {
            "subtasks": subtasks,
            "task_context": rag_result["context"]
        }
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute each subtask with RAG support."""
        results = []
        for subtask in decision["subtasks"]:
            # Get specific context for subtask
            rag_result = await self.rag.process({"query": subtask})
            
            # Execute subtask with context
            prompt = self.templates["execute"].format(
                context=rag_result["context"],
                subtask=subtask
            )
            result = await self.llm.process({"prompt": prompt})
            results.append({
                "subtask": subtask,
                "result": result["response"],
                "context_used": rag_result["context"]
            })
        
        return {"subtask_results": results}
    
    async def observe(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results into final response."""
        results_text = "\n".join(
            f"Subtask: {r['subtask']}\nResult: {r['result']}"
            for r in result["subtask_results"]
        )
        
        prompt = self.templates["summarize"].format(
            task=self.memory.get("task", ""),
            results=results_text
        )
        
        summary = await self.llm.process({"prompt": prompt})
        return {
            "summary": summary["response"],
            "detailed_results": result["subtask_results"]
        }

async def main():
    # Example documents
    documents = [
        Document(
            content="Quantum computing leverages quantum mechanics principles...",
            metadata={"source": "quantum_basics.txt"}
        ),
        Document(
            content="Recent breakthroughs in error correction...",
            metadata={"source": "recent_advances.txt"}
        )
    ]
    
    # Configure agent
    agent = RAGEnabledTaskAgent(
        name="quantum_research_agent",
        documents=documents,
        provider_config={
            "model": "llama2",
            "base_url": "http://localhost:11434/v1"
        },
        vectorstore_config={
            "dimension": 384,  # Depends on the embedding model
            "index_type": "Cosine"
        }
    )
    
    # Initialize agent
    await agent.initialize()
    
    # Process a research task
    result = await agent.process({
        "task": "Analyze recent breakthroughs in quantum error correction"
    })
    
    print("Task Summary:", result["observation"]["summary"])
    print("\nDetailed Results:")
    for r in result["observation"]["detailed_results"]:
        print(f"\nSubtask: {r['subtask']}")
        print(f"Result: {r['result']}")
        
if __name__ == "__main__":
    asyncio.run(main())