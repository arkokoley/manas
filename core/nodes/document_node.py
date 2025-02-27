"""Node implementation for document processing."""
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from dataclasses import dataclass, field
from core.base import Node
from core.llm import LLMNode, LLMConfig
from core.models import Document
from .factory import register_node
from .protocol import DocumentNodeProtocol

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class DocumentProcessorConfig:
    """Configuration for document processor nodes."""
    model: str = "llama3.2"
    temperature: float = 0.2  # Lower temperature for more deterministic output
    max_chunk_size: int = 4000
    overlap: int = 200
    formats: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "text": {"mime_type": "text/plain"},
        "markdown": {"mime_type": "text/markdown"},
        "html": {"mime_type": "text/html"},
        "json": {"mime_type": "application/json"}
    })
    templates: Dict[str, str] = field(default_factory=lambda: {
        "summarize": "Summarize the following document:\n\n{content}\n\nSummary:",
        "extract_keywords": "Extract key concepts and terms from the following document:\n\n{content}\n\nKeywords:",
        "generate_title": "Generate a concise but descriptive title for the following document:\n\n{content}\n\nTitle:"
    })
    preprocessors: List[Callable[[str], str]] = field(default_factory=list)
    postprocessors: List[Callable[[str], str]] = field(default_factory=list)

@register_node("document")
class DocumentNode(Node):
    """Node for processing and generating documents."""
    
    def __init__(self, name: str, config: DocumentProcessorConfig, llm_node: Optional[LLMNode] = None):
        super().__init__(name=name)
        self.config = config
        self.llm_node = llm_node
        self._initialized = False
    
    async def initialize(self):
        """Initialize document processor resources."""
        if not self._initialized:
            # Create LLM node if not provided
            if self.llm_node is None:
                self.llm_node = LLMNode(
                    name=f"{self.name}_llm",
                    config=LLMConfig(
                        provider_name="ollama" if self.config.model.startswith("llama") else "openai", 
                        provider_config={"model": self.config.model},
                        temperature=self.config.temperature
                    )
                )
                await self.llm_node.initialize()
            
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._initialized:
            if self.llm_node and not self.llm_node._initialized:
                await self.llm_node.cleanup()
            self._initialized = False
    
    def _chunk_document(self, content: str) -> List[str]:
        """
        Split document content into chunks.
        
        Args:
            content: Document content
            
        Returns:
            List of content chunks
        """
        size = self.config.max_chunk_size
        overlap = self.config.overlap
        
        if len(content) <= size:
            return [content]
            
        chunks = []
        i = 0
        while i < len(content):
            chunk_end = min(i + size, len(content))
            chunks.append(content[i:chunk_end])
            i += size - overlap
            
        return chunks
    
    def _apply_preprocessors(self, content: str) -> str:
        """Apply configured preprocessors to content."""
        result = content
        for processor in self.config.preprocessors:
            result = processor(result)
        return result
    
    def _apply_postprocessors(self, content: str) -> str:
        """Apply configured postprocessors to content."""
        result = content
        for processor in self.config.postprocessors:
            result = processor(result)
        return result
    
    async def process_document(self, 
        content: Union[str, Dict[str, Any]],
        operations: List[str] = None,
        format: str = "text"
    ) -> Dict[str, Any]:
        """
        Process a document with specified operations.
        
        Args:
            content: Document content or document dict
            operations: List of operations to perform (summarize, extract_keywords, etc.)
            format: Content format (text, markdown, html, json)
            
        Returns:
            Dictionary with operation results
        """
        if not self._initialized:
            await self.initialize()
        
        # Extract content from different input formats
        if isinstance(content, dict):
            doc_content = content.get("content", "")
            metadata = content.get("metadata", {})
        else:
            doc_content = content
            metadata = {}
        
        # Preprocess content
        doc_content = self._apply_preprocessors(doc_content)
        
        # Default operations if none specified
        if not operations:
            operations = ["summarize"]
        
        # Chunk document if necessary
        chunks = self._chunk_document(doc_content)
        
        # Process each operation
        results = {}
        for operation in operations:
            if operation in self.config.templates:
                template = self.config.templates[operation]
                
                # Process in chunks if multiple chunks exist
                if len(chunks) > 1:
                    chunk_results = []
                    for i, chunk in enumerate(chunks):
                        prompt = template.format(content=chunk)
                        result = await self.llm_node.process({"prompt": prompt})
                        chunk_results.append(result["response"])
                    
                    # Combine chunk results if needed
                    if operation == "summarize":
                        combined = "\n\n".join(chunk_results)
                        # Create a summary of summaries for large documents
                        if len(combined) > self.config.max_chunk_size:
                            prompt = template.format(content=combined)
                            result = await self.llm_node.process({"prompt": prompt})
                            combined = result["response"]
                        results[operation] = combined
                    else:
                        results[operation] = chunk_results
                else:
                    # Process single chunk
                    prompt = template.format(content=doc_content)
                    result = await self.llm_node.process({"prompt": prompt})
                    results[operation] = result["response"]
            else:
                results[operation] = f"Operation '{operation}' not supported"
        
        # Create output document
        output_doc = {
            "content": doc_content,
            "metadata": {
                **metadata,
                "processed_at": self._get_timestamp(),
                "operations": operations,
                "format": format
            },
            "results": results
        }
        
        # Apply postprocessors to results
        for key, value in results.items():
            if isinstance(value, str):
                results[key] = self._apply_postprocessors(value)
        
        return output_doc
    
    async def generate_document(self,
        parameters: Dict[str, Any],
        format: str = "text"
    ) -> Dict[str, Any]:
        """
        Generate a new document from parameters.
        
        Args:
            parameters: Document generation parameters
            format: Output format (text, markdown, html, json)
            
        Returns:
            Dictionary with generated document
        """
        if not self._initialized:
            await self.initialize()
        
        # Extract parameters
        template = parameters.get("template", "")
        variables = parameters.get("variables", {})
        instructions = parameters.get("instructions", "")
        length = parameters.get("length", "medium")
        style = parameters.get("style", "informative")
        
        # Build prompt
        prompt_parts = []
        
        if instructions:
            prompt_parts.append(instructions)
        
        prompt_parts.append(f"Generate a {length} document in {format} format with a {style} style.")
        
        if template:
            try:
                filled_template = template.format(**variables)
                prompt_parts.append(filled_template)
            except KeyError as e:
                missing_vars = str(e)
                prompt_parts.append(f"Template requires variables: {missing_vars}")
                prompt_parts.append(template)
        
        # Process with LLM
        prompt = "\n\n".join(prompt_parts)
        result = await self.llm_node.process({"prompt": prompt})
        content = result["response"]
        
        # Apply postprocessors
        content = self._apply_postprocessors(content)
        
        # Create output document
        output_doc = {
            "content": content,
            "metadata": {
                "generated_at": self._get_timestamp(),
                "format": format,
                "parameters": {
                    "length": length,
                    "style": style
                }
            }
        }
        
        return output_doc
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs by either processing or generating a document.
        
        Args:
            inputs: Dictionary containing either 'content' (for processing) 
                   or 'parameters' (for generation)
            
        Returns:
            Processed or generated document
            
        Raises:
            ValueError: If required inputs are missing or invalid
        """
        content = inputs.get("content")
        parameters = inputs.get("parameters")
        
        if content is not None:
            # Process existing document
            operations = inputs.get("operations", ["summarize"])
            format = inputs.get("format", "text")
            return await self.process_document(content, operations, format)
        elif parameters is not None:
            # Generate new document
            format = inputs.get("format", "text")
            return await self.generate_document(parameters, format)
        else:
            raise ValueError("Input must contain either 'content' or 'parameters'")
