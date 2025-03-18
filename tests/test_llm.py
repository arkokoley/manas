"""Tests for core LLM components."""
import pytest
from pydantic import ValidationError
from manas_ai.llm import LLMNode, LLMConfig, PromptTemplate, ChainNode
from manas_ai.models import ModelProviderConfig

@pytest.fixture
def mock_provider_config():
    return {
        "provider": "ollama",
        "provider_config": {
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1"
        },
        "temperature": 0.7
    }

@pytest.fixture
def llm_node(mock_provider_config):
    return LLMNode(
        name="test_llm",
        config=LLMConfig(**mock_provider_config)
    )

@pytest.mark.asyncio
async def test_llm_node_initialization(llm_node):
    """Test LLM node initialization."""
    assert llm_node.name == "test_llm"
    assert llm_node._provider is not None
    assert not llm_node._initialized

@pytest.mark.asyncio
async def test_prompt_template():
    """Test prompt template functionality."""
    template = PromptTemplate(
        template="Hello {name}!",
        input_variables=["name"]
    )
    
    # Test valid formatting
    result = template.format(name="World")
    assert result == "Hello World!"
    
    # Test missing variables
    with pytest.raises(ValueError):
        template.format(wrong_var="World")

@pytest.mark.asyncio
async def test_chain_node(llm_node):
    """Test chain node functionality."""
    chain = ChainNode(
        name="test_chain",
        nodes=[llm_node],
        prompt_templates=[
            PromptTemplate(
                template="Process: {input}",
                input_variables=["input"]
            )
        ]
    )
    
    assert len(chain.nodes) == 1
    assert len(chain.prompt_templates) == 1

@pytest.mark.asyncio
async def test_llm_config_validation():
    """Test LLM configuration validation."""
    # Test valid config
    config = LLMConfig(
        provider="ollama",
        provider_config={"model": "llama3.2"},
        temperature=0.7
    )
    assert config.temperature == 0.7
    
    # Test invalid temperature
    with pytest.raises(ValidationError):
        LLMConfig(
            provider="ollama",
            provider_config={"model": "llama3.2"},
            temperature=2.0
        )

@pytest.mark.asyncio
async def test_llm_node_lifecycle(llm_node):
    """Test LLM node lifecycle management."""
    # Test initialization
    await llm_node.initialize()
    assert llm_node._initialized
    
    # Test cleanup
    await llm_node.cleanup()
    assert not llm_node._initialized

@pytest.mark.asyncio
async def test_llm_node_error_handling(llm_node):
    """Test LLM node error handling."""
    # Test missing prompt
    with pytest.raises(ValueError):
        await llm_node.process({})
    
    # Test invalid provider
    with pytest.raises(ValueError):
        LLMNode(
            name="invalid",
            config=LLMConfig(
                provider="invalid_provider",
                provider_config={}
            )
        )