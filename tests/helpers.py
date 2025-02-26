"""Test helpers and fixtures."""
import pytest
import asyncio
from typing import Dict, Any, Optional
from core.models import Document
from core.llm import LLMNode, LLMConfig
from core.providers.base import BaseLLMProvider

class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {
            "default": "This is a mock response"
        }
        self.calls = []
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass
    
    async def generate(self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop_sequences": stop_sequences,
            "kwargs": kwargs
        })
        return self.responses.get(prompt, self.responses["default"])
    
    async def stream_generate(self, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ):
        response = self.responses.get(prompt, self.responses["default"])
        for chunk in response.split():
            yield chunk + " "
    
    async def embed(self, text: str) -> list[float]:
        """Return mock embeddings."""
        return [0.1] * 384  # Common embedding dimension

@pytest.fixture
def mock_documents():
    """Fixture providing test documents."""
    return [
        Document(
            content="Test document one",
            metadata={"source": "test1.txt"}
        ),
        Document(
            content="Test document two",
            metadata={"source": "test2.txt"}
        )
    ]

@pytest.fixture
def mock_llm_node():
    """Fixture providing a LLM node with mock provider."""
    config = LLMConfig(
        provider="mock",
        provider_config={},
        temperature=0.7
    )
    node = LLMNode(name="test_llm", config=config)
    node._provider = MockLLMProvider()
    return node

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()