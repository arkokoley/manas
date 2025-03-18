"""
Example implementation of a knowledge base QA system that:
1. Processes and indexes documents
2. Maintains a searchable knowledge base
3. Answers questions with citations
"""

import os
import asyncio
import magic
import chardet
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache
import time

from core import Flow, LLM, RAG
from manas_ai.nodes import DocumentNode, QANode
from manas_ai.models import Document, RAGConfig
from manas_ai.vectorstores import ChromaStore
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentProcessorNode(DocumentNode):
    """Node for processing documents of various formats."""
    
    async def process_file(self, file_path: str) -> Document:
        """Process a single file."""
        path = Path(file_path)
        
        # Detect file type
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(str(path))
        
        # Read and detect encoding
        with open(path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding']
        
        # Read content based on file type
        if file_type == 'text/plain':
            content = raw.decode(encoding)
        elif file_type == 'application/pdf':
            content = await self.extract_pdf_text(path)
        else:
            content = await self.extract_text(path)
            
        return Document(
            content=content,
            metadata={
                "source": str(path),
                "type": file_type,
                "encoding": encoding,
                "modified": path.stat().st_mtime
            }
        )
    
    async def extract_pdf_text(self, path: Path) -> str:
        """Extract text from PDF."""
        # Add your PDF extraction logic here
        raise NotImplementedError("PDF extraction not implemented")
    
    async def extract_text(self, path: Path) -> str:
        """Extract text from other formats."""
        with open(path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding']
            return raw.decode(encoding)
    
    async def _process_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process input files."""
        files = inputs.get("files", [])
        if isinstance(files, str):
            files = [files]
            
        documents = []
        for file_path in files:
            doc = await self.process_file(file_path)
            documents.append(doc)
            
        return {
            "documents": documents,
            "count": len(documents)
        }

class KnowledgeBaseNode(QANode):
    """Node for maintaining and querying a knowledge base."""
    
    def __init__(
        self,
        name: str,
        persist_dir: str,
        llm: Optional[LLM] = None
    ):
        # Create vector store
        store = ChromaStore(
            collection_name=name,
            persist_directory=persist_dir
        )
        
        # Create LLM if not provided
        if not llm:
            llm = LLM.from_provider(
                "openai",
                model_name="gpt-4",
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        
        super().__init__(
            name=name,
            llm=llm,
            config=RAGConfig(
                system_prompt=(
                    "You are a knowledgeable assistant with access to a document "
                    "knowledge base. When answering questions:\n"
                    "1. Use the provided context\n"
                    "2. Cite your sources\n"
                    "3. Indicate if information is missing\n"
                    "4. Be objective and accurate"
                ),
                use_rag=True,
                rag_config={
                    "vector_store": store,
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "max_sources": 5,
                    "min_relevance": 0.75
                }
            )
        )
        
        # Initialize cache
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the knowledge base."""
        await self.rag_node.add_documents(documents)
    
    @lru_cache(maxsize=1000)
    async def get_cached_answer(
        self,
        question: str,
        max_age: int = 3600  # 1 hour
    ) -> Optional[Dict[str, Any]]:
        """Get cached answer if available and fresh."""
        cache_key = self.normalize_question(question)
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < max_age:
                return result
        
        # Get fresh answer
        result = await super().answer(question)
        self.cache[cache_key] = (result, time.time())
        return result
    
    def normalize_question(self, question: str) -> str:
        """Normalize question for cache key."""
        return " ".join(question.lower().split())
    
    async def search(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base without generating an answer."""
        docs = await self.rag_node.search(query, k=k)
        return [
            {
                "content": doc.content,
                "relevance": score,
                "source": doc.metadata["source"]
            }
            for doc, score in docs
        ]

class DocumentUpdateHandler(FileSystemEventHandler):
    """Handle document updates in real-time."""
    
    def __init__(self, kb_node: KnowledgeBaseNode):
        self.kb_node = kb_node
        self.processor = DocumentProcessorNode("update_processor")
    
    async def process_file(self, path: str) -> None:
        """Process and add/update a document."""
        result = await self.processor.process({"files": [path]})
        await self.kb_node.add_documents(result["documents"])
    
    def on_created(self, event):
        if event.is_directory:
            return
        asyncio.create_task(self.process_file(event.src_path))
    
    def on_modified(self, event):
        if event.is_directory:
            return
        asyncio.create_task(self.process_file(event.src_path))

class QASystem:
    """Complete QA system with document processing and knowledge base."""
    
    def __init__(
        self,
        docs_dir: str,
        kb_dir: str,
        model_name: str = "gpt-4"
    ):
        """Initialize the QA system."""
        self.docs_dir = Path(docs_dir)
        self.kb_dir = Path(kb_dir)
        
        # Create LLM
        self.model = LLM.from_provider(
            "openai",
            model_name=model_name,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Create nodes
        self.processor = DocumentProcessorNode(
            name="doc_processor",
            llm=self.model
        )
        
        self.knowledge_base = KnowledgeBaseNode(
            name="knowledge_base",
            persist_dir=str(self.kb_dir),
            llm=self.model
        )
        
        # Create flow
        self.flow = self._create_flow()
        
        # Set up file watching
        self.observer = None
        
    def _create_flow(self) -> Flow:
        """Create and configure the flow."""
        flow = Flow(name="qa_system")
        flow.add_node(self.processor)
        flow.add_node(self.knowledge_base)
        flow.add_edge(self.processor, self.knowledge_base)
        return flow
    
    async def initialize(self) -> None:
        """Initialize the system."""
        # Create directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize nodes
        await self.processor.initialize()
        await self.knowledge_base.initialize()
        
        # Process initial documents
        await self.process_documents()
        
        # Start file watching
        self.start_file_watching()
    
    async def process_documents(self) -> None:
        """Process all documents in the docs directory."""
        docs = [str(p) for p in self.docs_dir.glob("**/*")]
        if docs:
            result = await self.processor.process({"files": docs})
            await self.knowledge_base.add_documents(result["documents"])
    
    def start_file_watching(self) -> None:
        """Start watching for document changes."""
        handler = DocumentUpdateHandler(self.knowledge_base)
        self.observer = Observer()
        self.observer.schedule(
            handler,
            str(self.docs_dir),
            recursive=True
        )
        self.observer.start()
    
    async def ask(
        self,
        question: str,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ask a question to the knowledge base.
        
        Args:
            question: The question to ask
            use_cache: Whether to use cached results
            **kwargs: Additional parameters
            
        Returns:
            Answer with sources
        """
        if use_cache:
            return await self.knowledge_base.get_cached_answer(
                question,
                **kwargs
            )
        else:
            return await self.knowledge_base.answer(
                question,
                **kwargs
            )
    
    async def search(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        return await self.knowledge_base.search(query, k=k)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        await self.flow.cleanup()

async def main():
    """Example usage of QA system."""
    # Create system
    qa = QASystem(
        docs_dir="documents",
        kb_dir="knowledge_base"
    )
    
    try:
        # Initialize
        await qa.initialize()
        
        # Example questions
        questions = [
            "What are the key principles of quantum computing?",
            "How do neural networks learn?",
            "What are the latest developments in fusion energy?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            
            # Get answer
            result = await qa.ask(question)
            print("\nA:", result["answer"])
            
            # Show sources
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['metadata']['source']}")
            
            # Example search
            print("\nRelated Documents:")
            docs = await qa.search(question, k=2)
            for doc in docs:
                print(f"- {doc['source']} (relevance: {doc['relevance']:.2f})")
                
    finally:
        await qa.cleanup()

if __name__ == "__main__":
    asyncio.run(main())