"""Test helpers and fixtures."""
import pytest
import asyncio
from typing import Dict, Any, Optional
from core.models import Document
from core.llm import LLMNode, LLMConfig
from core.providers.mock import MockLLMProvider

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
async def mock_llm_node():
    """Fixture providing a LLM node with mock provider."""
    config = LLMConfig(
        provider_name="mock",
        provider_config={
            "responses": {
                "default": "This is a mock response",
                "What is quantum computing?": "Quantum computing is a type of computing that uses quantum mechanics principles."
            }
        },
        temperature=0.7
    )
    
    # Create and initialize the node
    node = LLMNode(name="test_llm", config=config)
    await node.initialize()
    yield node
    await node.cleanup()

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()