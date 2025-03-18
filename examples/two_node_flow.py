"""Example demonstrating a two-node flow using Ollama."""
import asyncio
from manas_ai.llm import LLMNode, LLMConfig
from manas_ai.flow import Flow
from manas_ai.base import Edge

async def main():
    # Create first LLM node for question answering
    qa_node = LLMNode(
        name="qa_node",
        config=LLMConfig(
            provider_name="ollama",
            provider_config={
                "model": "qwen2.5:0.5b",
                "base_url": "http://localhost:11434/v1"
            },
            temperature=0.7
        )
    )
    
    # Create second LLM node for summarization
    summary_node = LLMNode(
        name="summary_node",
        config=LLMConfig(
            provider_name="ollama",
            provider_config={
                "model": "deepseek-r1:latest",
                "base_url": "http://localhost:11434/v1"
            },
            temperature=0.3  # Lower temperature for more focused summary
        )
    )
    
    # Create a flow
    flow = Flow(name="qa_summary_flow")
    
    # Add nodes to flow
    qa_node_id = flow.add_node(qa_node)
    summary_node_id = flow.add_node(summary_node)
    
    # Connect nodes with an edge
    flow.add_edge(Edge(
        source_node=qa_node_id,
        target_node=summary_node_id,
        name="qa_to_summary"
    ))
    
    try:
        # Process with the flow
        result = await flow.process({
            "prompt": "What are the key principles of quantum computing?",
            # For the summary node, create a prompt that uses the QA response
            "summary_prompt": lambda qa_result: f"Summarize this explanation in 2-3 sentences: {qa_result['response']}"
        })
        
        # Print both the detailed answer and summary
        print("\nDetailed Answer:")
        print(result[qa_node_id]["response"])
        print("\nSummary:")
        print(result[summary_node_id]["response"])
        
    finally:
        # Cleanup both nodes
        await qa_node.cleanup()
        await summary_node.cleanup()

if __name__ == "__main__":
    asyncio.run(main())