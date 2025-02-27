"""Tests for specialized node implementations."""
import pytest
import asyncio
from typing import Dict, Any

from core.nodes.tool_node import ToolNode, Tool, create_tool
from core.nodes.api_node import APINode, APIConfig
from core.nodes.qa_node import QANode, QAConfig
from core.nodes.document_node import DocumentNode, DocumentProcessorConfig
from core.llm import LLMNode, LLMConfig
from tests.helpers import MockLLMProvider

@pytest.fixture
async def mock_llm_node():
    """Fixture providing a mock LLM node."""
    # Create a proper config with the mock provider
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
    node = LLMNode(name="test_llm", config=config)
    await node.initialize()
    yield node  # Using yield instead of return for proper cleanup
    await node.cleanup()

@pytest.mark.asyncio
async def test_tool_node():
    """Test ToolNode functionality."""
    # Define a test tool
    async def add_numbers(a: int, b: int) -> int:
        return a + b
    
    tool = create_tool(
        name="add",
        description="Add two numbers",
        function=add_numbers
    )
    
    # Create tool node
    node = ToolNode(name="math_tools", tools=[tool])
    
    # Test tool execution
    result = await node.process({
        "tool": "add",
        "a": 5,
        "b": 3
    })
    
    assert result["result"] == 8
    
    # Test tool metadata
    tools_info = node.get_tools()
    assert "add" in tools_info
    assert tools_info["add"]["description"] == "Add two numbers"

@pytest.mark.asyncio
async def test_qa_node(mock_llm_node):
    """Test QANode functionality."""
    # Create QA node
    config = QAConfig(
        model="mock_model",
        temperature=0.7,
        prompt_template="Question: {question}\nContext: {context}\nAnswer:"
    )
    
    # Get the actual node instead of the generator
    llm = mock_llm_node
    
    qa_node = QANode(
        name="test_qa",
        config=config,
        llm_node=llm
    )
    await qa_node.initialize()  # Make sure the node is initialized
    
    # Test simple question
    result = await qa_node.process({
        "question": "What is quantum computing?"
    })
    
    assert "answer" in result
    assert "quantum" in result["answer"].lower()
    
    # Test with custom context
    result = await qa_node.process({
        "question": "What is quantum computing?",
        "context": "Quantum computing uses qubits and superposition.",
        "session_id": "test_session"
    })
    
    assert "answer" in result
    
    # Test follow-up question using session
    result = await qa_node.process({
        "question": "How does it work?",
        "session_id": "test_session"
    })
    
    assert "answer" in result
    await qa_node.cleanup()  # Clean up resources

@pytest.mark.asyncio
async def test_document_node(mock_llm_node):
    """Test DocumentNode functionality."""
    # Create document node
    config = DocumentProcessorConfig(
        model="mock_model",
        templates={
            "summarize": "Summarize: {content}",
            "extract_keywords": "Keywords: {content}"
        }
    )
    
    # Get the actual node instead of the generator
    llm = mock_llm_node
    
    doc_node = DocumentNode(
        name="test_doc",
        config=config,
        llm_node=llm
    )
    await doc_node.initialize()  # Make sure the node is initialized
    
    # Test document processing
    result = await doc_node.process({
        "content": "Quantum computing is a rapidly emerging technology that uses quantum mechanics to solve problems too complex for classical computers.",
        "operations": ["summarize", "extract_keywords"],
        "format": "text"
    })
    
    assert "results" in result
    assert "summarize" in result["results"]
    assert "extract_keywords" in result["results"]
    assert "content" in result
    assert "metadata" in result
    
    # Test document generation
    result = await doc_node.process({
        "parameters": {
            "template": "Write about {topic}.",
            "variables": {"topic": "quantum computing"},
            "length": "short",
            "style": "educational"
        },
        "format": "markdown"
    })
    
    assert "content" in result
    assert result["metadata"]["format"] == "markdown"
    assert len(result["content"]) > 0
    await doc_node.cleanup()  # Clean up resources

@pytest.mark.asyncio
async def test_node_factory():
    """Test node factory functionality."""
    from core.nodes.factory import register_node, create_node
    
    # Register a test node
    @register_node("test")
    class TestNode(ToolNode):
        def __init__(self, name, config=None, custom_param=None):
            super().__init__(name)
            self.custom_param = custom_param
    
    # Create node using factory
    node = create_node(
        node_type="test",
        name="factory_test",
        custom_param="custom_value"
    )
    
    assert node.name == "factory_test"
    assert node.custom_param == "custom_value"
    assert isinstance(node, TestNode)
