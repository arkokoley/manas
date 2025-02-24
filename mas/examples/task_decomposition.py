"""Example implementation showcasing the framework's capabilities."""
import asyncio
from typing import Dict, Any, List
from ..core.agent import Agent
from ..core.base import Edge
from ..core.llm import LLMNode, PromptTemplate, LLMConfig
from ..core.rag import RAGNode, RAGConfig, Document
from ..core.flow import Flow
from ..core.providers.ollama import OllamaProvider

class TaskDecompositionAgent(Agent):
    """Agent that breaks down complex tasks into subtasks."""
    def __init__(self, name: str, provider: str, provider_config: Dict[str, Any]):
        super().__init__(name=name)
        # Create LLM config with embedding dimension from provider
        self.llm = LLMNode(
            name=f"{name}_llm",
            config=LLMConfig(
                provider_name=provider,
                provider_config=provider_config,
                temperature=0.7,
                embedding_dimension=OllamaProvider.EMBEDDING_DIMENSIONS.get(
                    provider_config.get("model", "").lower().split(":")[0],
                    OllamaProvider.EMBEDDING_DIMENSIONS["default"]
                )
            )
        )
        self.decomposition_template = PromptTemplate(
            template="Break down the following task into subtasks:\n{task}\n\nSubtasks:",
            input_variables=["task"]
        )
        # Initialize RAG config with same dimension as LLM
        self.rag_config = RAGConfig(
            vectorstore_type="faiss",
            vectorstore_config={
                "dimension": self.llm.config.embedding_dimension,
                "index_type": "Cosine"
            }
        )
        self.has_documents = False
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.decomposition_template.format(task=context["task"])
        result = await self.llm.process({"prompt": prompt})
        subtasks = [s.strip() for s in result["response"].split("\n") if s.strip()]
        return {"subtasks": subtasks, "task": context["task"]}
    
    async def initialize_rag_nodes(self, documents: List[Document]) -> bool:
        """Initialize RAG nodes with documents. Returns True if documents were added."""
        if not documents:
            return False
            
        # Convert to Documents if they're strings
        docs = [
            Document(content=doc) if isinstance(doc, str) else doc
            for doc in documents
        ]
        
        # Add documents to the first RAG node of each flow
        for flow_info in self.memory.get("flows", []):
            flow = flow_info["flow"]
            rag_node = flow.nodes[0]
            await rag_node.add_documents(docs)
        
        self.has_documents = True
        return True
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        subtasks = decision["subtasks"]
        subtask_flows = []
        
        for subtask in subtasks:
            # Create a flow for each subtask
            flow = Flow(name=f"subtask_{len(subtask_flows)}")
            
            # Add RAG node for context retrieval
            rag_node = RAGNode(
                name="retrieval",
                config=self.rag_config,
                embedding_node=self.llm,
                llm_node=self.llm
            )
            rag_node_id = flow.add_node(rag_node)
            
            # Add LLM node for processing
            llm_node = LLMNode(name="processor", config=self.llm.config)
            llm_node_id = flow.add_node(llm_node)
            
            # Connect nodes
            flow.add_edge(Edge(
                source_node=rag_node_id,
                target_node=llm_node_id,
                name="retrieval_to_processing"
            ))
            
            subtask_flows.append({
                "flow": flow, 
                "subtask": subtask,
                "rag_node_id": rag_node_id,
                "llm_node_id": llm_node_id
            })
        
        # Store flows in memory for initialization
        self.memory["flows"] = subtask_flows
            
        return {"flows": subtask_flows}
    
    async def observe(self, result: Dict[str, Any]) -> Dict[str, Any]:
        all_results = []
        for flow_info in result["flows"]:
            flow = flow_info["flow"]
            subtask = flow_info["subtask"]
            rag_node_id = flow_info["rag_node_id"]
            llm_node_id = flow_info["llm_node_id"]
            
            if self.has_documents:
                # Only use RAG if we have documents
                rag_result = await flow.nodes[rag_node_id].process({
                    "query": subtask,
                    "generate_response": False
                })
                context = rag_result.get('context', '')
            else:
                context = ''
            
            # Use context to process with LLM if available, otherwise just use LLM
            prompt = (
                f"Complete this task. Use the context if provided:\n\n"
                f"{f'Context: {context}' if context else 'No additional context available.'}\n\n"
                f"Task: {subtask}"
            )
            llm_result = await flow.nodes[llm_node_id].process({"prompt": prompt})
            
            all_results.append({
                "subtask": subtask,
                "context_used": bool(context),
                "response": llm_result.get("response", "No response")
            })
        
        # Format final response
        final_response = "Task Results:\n\n"
        for result in all_results:
            final_response += f"Subtask: {result['subtask']}\n"
            final_response += f"Using Context: {'Yes' if result['context_used'] else 'No'}\n"
            final_response += f"Response: {result['response']}\n\n"
        
        return {"response": final_response, "detailed_results": all_results}

# Example usage with different providers:
async def openai_example():
    # Using OpenAI
    agent = TaskDecompositionAgent(
        name="openai_decomposer",
        provider="openai",
        provider_config={
            "api_key": "your-api-key",
            "model": "gpt-3.5-turbo"
        }
    )
    
    # Only add documents if you have them
    documents = []  # Add your documents here if needed
    if documents:
        await agent.initialize_rag_nodes(documents)
        
    return await agent.process({
        "task": "Research and summarize the latest developments in quantum computing"
    })

async def huggingface_example():
    # Using Hugging Face
    agent = TaskDecompositionAgent(
        name="hf_decomposer",
        provider="huggingface",
        provider_config={
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "device": "cuda",
            "torch_dtype": "float16"
        }
    )
    return await agent.process({
        "task": "Research and summarize the latest developments in quantum computing"
    })

async def ollama_example():
    # Using Ollama
    agent = TaskDecompositionAgent(
        name="ollama_decomposer",
        provider="ollama",
        provider_config={
            "model": "deepseek-r1:latest",
            "base_url": "http://localhost:11434/v1"
        }
    )
    
    # Only add documents if you have them
    documents = []  # Add your documents here if needed
    if documents:
        await agent.initialize_rag_nodes(documents)
    
    # Process the task
    result = await agent.process({
        "task": "What are the fundamental concepts of quantum computing and its main challenges?"
    })
    
    # Print the final response
    if "observation" in result and "response" in result["observation"]:
        print(result["observation"]["response"])
    else:
        print("No response generated")

if __name__ == "__main__":
    asyncio.run(ollama_example())
