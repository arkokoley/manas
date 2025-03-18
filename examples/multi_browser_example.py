"""
Example of using multiple BrowserNodes in a flow for parallel research and analysis tasks.
Demonstrates:
1. Market research node
2. Technical analysis node
3. News monitoring node
Working together in a coordinated flow
"""

import asyncio
from manas_ai.flow import Flow
from manas_ai.base import Edge
from manas_ai.nodes import create_browser_node, QANode, QAConfig

async def create_research_flow():
    """Create a flow with multiple browser nodes for different research aspects."""
    
    # Create specialized browser nodes
    market_research = create_browser_node(
        name="market_research",
        model="deepseek-r1",
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    tech_analysis = create_browser_node(
        name="tech_analysis",
        model="deepseek-r1",
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    news_monitor = create_browser_node(
        name="news_monitor",
        model="deepseek-r1",
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    # Create QA node for analysis with proper configuration
    analyzer = QANode(
        name="analyzer",
        config=QAConfig(
            model="deepseek-r1",
            temperature=0.7,
            prompt_template=(
                "You are an expert analyst. Analyze and synthesize the following "
                "information from multiple sources to provide comprehensive insights:\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )
    )
    
    # Create and configure flow
    flow = Flow(name="research_flow")
    
    # Add nodes and store their IDs
    market_id = flow.add_node(market_research)
    tech_id = flow.add_node(tech_analysis)
    news_id = flow.add_node(news_monitor)
    analyzer_id = flow.add_node(analyzer)
    
    # Connect nodes to analyzer using Edge objects
    flow.add_edge(Edge(source_node=market_id, target_node=analyzer_id, name="market_to_analyzer"))
    flow.add_edge(Edge(source_node=tech_id, target_node=analyzer_id, name="tech_to_analyzer"))
    flow.add_edge(Edge(source_node=news_id, target_node=analyzer_id, name="news_to_analyzer"))
    
    return flow

async def run_research_example():
    """Run an example using multiple browser nodes for comprehensive research."""
    print("\n=== Multi-Browser Research Example ===")
    
    # Create and initialize flow
    flow = await create_research_flow()
    await flow.initialize()
    
    try:
        # Start parallel research tasks
        market_task = flow.nodes[flow._node_names["market_research"]].process({
            "task": "Research current AI model pricing and business models. "
                   "Focus on GPT-4, Claude, and open source alternatives.",
            "url": "https://openai.com/pricing"
        })
        
        tech_task = flow.nodes[flow._node_names["tech_analysis"]].process({
            "task": "Analyze the top AI repositories on GitHub. "
                   "Look for trending open source models and their capabilities.",
            "url": "https://github.com/trending"
        })
        
        news_task = flow.nodes[flow._node_names["news_monitor"]].process({
            "task": "Find recent significant AI news and developments "
                   "from major tech news sources.",
            "url": "https://techcrunch.com/category/artificial-intelligence/"
        })
        
        # Gather all results
        results = await asyncio.gather(market_task, tech_task, news_task)
        
        # Analyze combined findings
        analysis = await flow.nodes[flow._node_names["analyzer"]].process({
            "question": (
                "Based on the research findings, what are the key trends "
                "in AI model development, pricing, and market adoption? "
                "Provide a comprehensive analysis."
            ),
            "context": str(results)
        })
        
        # Print results
        print("\nMarket Research:", results[0].get("result", "No result"))
        print("\nTechnical Analysis:", results[1].get("result", "No result"))
        print("\nNews Monitoring:", results[2].get("result", "No result"))
        print("\nCombined Analysis:", analysis.get("response", "No analysis"))
        
    finally:
        await flow.cleanup()

async def main():
    """Run the multi-browser example."""
    await run_research_example()

if __name__ == "__main__":
    asyncio.run(main())