"""
Example implementation of a research flow that:
1. Researches a topic using RAG
2. Analyzes the findings
3. Generates a well-structured report
"""

import os
import asyncio
from typing import Dict, Any, List
from pathlib import Path

from core import Flow, LLM
from manas_ai.nodes import QANode
from manas_ai.models import RAGConfig, Document
from manas_ai.vectorstores import ChromaStore

class ResearchFlow:
    """A flow for researching topics and generating reports."""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        docs_dir: str = None,
        kb_dir: str = "knowledge_base"
    ):
        """Initialize the research flow."""
        # Create LLM
        self.model = LLM.from_provider(
            "openai",
            model_name=model_name,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Create vector store
        self.store = ChromaStore(
            collection_name="research",
            persist_directory=kb_dir
        )
        
        # Initialize nodes
        self.researcher = self._create_researcher()
        self.analyst = self._create_analyst()
        self.writer = self._create_writer()
        
        # Create flow
        self.flow = self._create_flow()
        
        # Add initial documents if provided
        if docs_dir:
            self.add_documents(docs_dir)
            
    def _create_researcher(self) -> QANode:
        """Create the researcher node."""
        return QANode(
            name="researcher",
            llm=self.model,
            config=RAGConfig(
                system_prompt=(
                    "You are an expert researcher specializing in comprehensive "
                    "investigation. For each topic:\n"
                    "1. Gather relevant information from available sources\n"
                    "2. Include key facts, statistics, and context\n"
                    "3. Cite sources for important claims\n"
                    "4. Note any gaps or uncertainties in the research"
                ),
                use_rag=True,
                rag_config={
                    "vector_store": self.store,
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "max_sources": 5,
                    "min_relevance": 0.7
                }
            )
        )
        
    def _create_analyst(self) -> QANode:
        """Create the analyst node."""
        return QANode(
            name="analyst",
            llm=self.model,
            config=RAGConfig(
                system_prompt=(
                    "You are a skilled analyst who excels at finding patterns "
                    "and deriving insights. For the research provided:\n"
                    "1. Identify key patterns and trends\n"
                    "2. Analyze implications and significance\n"
                    "3. Note relationships between findings\n"
                    "4. Highlight areas needing further research"
                )
            )
        )
        
    def _create_writer(self) -> QANode:
        """Create the writer node."""
        return QANode(
            name="writer",
            llm=self.model,
            config=RAGConfig(
                system_prompt=(
                    "You are a technical writer who creates clear, well-structured "
                    "reports. For the analysis provided:\n"
                    "1. Create a logical structure with sections\n"
                    "2. Use clear, concise language\n"
                    "3. Include relevant examples\n"
                    "4. Format using Markdown with proper headers"
                )
            )
        )
        
    def _create_flow(self) -> Flow:
        """Create and configure the flow."""
        flow = Flow(name="research_flow")
        
        # Add nodes
        flow.add_node(self.researcher)
        flow.add_node(self.analyst)
        flow.add_node(self.writer)
        
        # Connect nodes
        flow.add_edge(self.researcher, self.analyst)
        flow.add_edge(self.analyst, self.writer)
        
        return flow
        
    async def add_documents(self, docs_dir: str) -> None:
        """Add documents to the knowledge base."""
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise ValueError(f"Documents directory not found: {docs_dir}")
            
        # Get all documents
        files = []
        for ext in ["*.txt", "*.md", "*.pdf"]:
            files.extend(docs_path.glob(f"**/{ext}"))
            
        # Process and add documents
        docs = []
        for file in files:
            content = await self._read_file(file)
            doc = Document(
                content=content,
                metadata={
                    "source": str(file),
                    "type": file.suffix[1:]
                }
            )
            docs.append(doc)
            
        # Add to vector store
        await self.store.add_documents(docs)
        
    async def _read_file(self, file_path: Path) -> str:
        """Read file content with appropriate handling."""
        if file_path.suffix == ".pdf":
            return await self._read_pdf(file_path)
        else:
            async with open(file_path, "r", encoding="utf-8") as f:
                return await f.read()
                
    async def _read_pdf(self, file_path: Path) -> str:
        """Extract text from PDF."""
        # Add your PDF processing logic here
        raise NotImplementedError("PDF processing not implemented")
        
    async def research(
        self,
        topic: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Research a topic and generate a report.
        
        Args:
            topic: The research topic
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing results from each node
        """
        try:
            # Initialize nodes
            await self.researcher.initialize()
            await self.analyst.initialize()
            await self.writer.initialize()
            
            # Process topic
            result = await self.flow.process({
                "prompt": topic,
                **kwargs
            })
            
            return result
            
        finally:
            # Clean up
            await self.flow.cleanup()
            
async def main():
    """Example usage of research flow."""
    # Create flow
    flow = ResearchFlow(
        model_name="gpt-4",
        docs_dir="research_docs",
        kb_dir="knowledge_base"
    )
    
    # Research topics
    topics = [
        "Recent advances in quantum computing",
        "Environmental impact of AI model training",
        "Progress in fusion energy research"
    ]
    
    for topic in topics:
        print(f"\nResearching: {topic}")
        result = await flow.research(
            topic,
            max_sources=5
        )
        
        # Print report
        print("\nFinal Report:")
        print(result["writer"]["response"])
        
        # Print sources if available
        if "sources" in result["researcher"]:
            print("\nSources Used:")
            for source in result["researcher"]["sources"]:
                print(f"- {source['metadata']['source']}")

if __name__ == "__main__":
    asyncio.run(main())