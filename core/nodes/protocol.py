"""Protocol definitions for specialized node types."""
from typing import Protocol, Dict, Any, List, Optional, Union, AsyncIterator, runtime_checkable

@runtime_checkable
class ToolNodeProtocol(Protocol):
    """Protocol for tools that can be executed by nodes."""
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        ...

@runtime_checkable
class APINodeProtocol(Protocol):
    """Protocol for API-accessing nodes."""
    async def request(self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Dict[str, Any]] = None, 
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make API request with specified parameters."""
        ...

@runtime_checkable
class QANodeProtocol(Protocol):
    """Protocol for question answering nodes."""
    async def answer(self, 
        question: str, 
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer a question with optional context."""
        ...

@runtime_checkable
class DocumentNodeProtocol(Protocol):
    """Protocol for document processing nodes."""
    async def process_document(self, 
        content: Union[str, Dict[str, Any]], 
        format: str = "text"
    ) -> Dict[str, Any]:
        """Process a document with specified format."""
        ...
    
    async def generate_document(self, 
        parameters: Dict[str, Any], 
        format: str = "text"
    ) -> Dict[str, Any]:
        """Generate a new document with specified parameters."""
        ...
