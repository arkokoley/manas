"""Tests for MCP node functionality."""
import pytest
from pydantic import ValidationError
from core.nodes import MCPNode, MCPConfig
from core.llm import LLMNode, LLMConfig

@pytest.fixture(scope="module")
def base_schemas():
    """Return all test schemas in one fixture to avoid redefinition."""
    return {
        "measurement": {
            "type": "object",
            "properties": {
                "temperature": {"type": "float"},
                "humidity": {"type": "float"},
                "time": {"type": "str"}
            },
            "required": ["temperature", "humidity"]
        },
        "condition": {
            "type": "object",
            "properties": {
                "location": {"type": "str"},
                "season": {"type": "str"}
            }
        },
        "prediction": {
            "type": "object",
            "properties": {
                "future_temperature": {"type": "float"},
                "confidence": {"type": "float"}
            },
            "required": ["future_temperature"]
        }
    }

@pytest.fixture(scope="function")
async def mock_llm_node():
    """Fixture providing a mock LLM node."""
    config = LLMConfig(
        provider_name="mock",
        provider_config={
            "responses": {
                "default": '{"future_temperature": 28.5, "confidence": 0.85}'
            }
        },
        temperature=0.7
    )
    node = LLMNode(name="test_llm", config=config)
    await node.initialize()
    yield node
    await node.cleanup()

@pytest.mark.asyncio
async def test_mcp_node_initialization():
    """Test MCP node initialization."""
    config = MCPConfig(
        model="claude-2",
        measurement_schema={
            "type": "object",
            "properties": {"value": {"type": "float"}},
            "required": ["value"]
        }
    )
    node = MCPNode(name="test_mcp", config=config)
    
    assert node.name == "test_mcp"
    assert not node.is_initialized()
    
    await node.initialize()
    assert node.is_initialized()
    assert node.llm_node is not None
    
    await node.cleanup()
    assert not node.is_initialized()

@pytest.mark.asyncio
async def test_mcp_prediction_with_schema_validation(mock_llm_node, base_schemas):
    """Test MCP prediction with schema validation."""
    config = MCPConfig(
        model="mock_model",
        temperature=0.7,
        measurement_schema=base_schemas["measurement"],
        condition_schema=base_schemas["condition"],
        prediction_schema=base_schemas["prediction"]
    )
    
    node = MCPNode(
        name="test_mcp",
        config=config,
        llm_node=mock_llm_node
    )
    await node.initialize()
    
    # Test valid input
    result = await node.process({
        "measurement": {
            "temperature": 25.0,
            "humidity": 60.0,
            "time": "2024-03-19T12:00:00Z"
        }
    })
    
    assert "predictions" in result
    assert len(result["predictions"]) == 1
    assert "future_temperature" in result["predictions"][0]
    assert "confidence" in result
    
    # Test with conditions
    result = await node.process({
        "measurement": {
            "temperature": 25.0,
            "humidity": 60.0,
        },
        "conditions": {
            "location": "San Francisco",
            "season": "summer"
        }
    })
    
    assert "predictions" in result
    assert "conditions" in result
    assert result["conditions"] is not None
    
    # Test invalid measurement
    with pytest.raises(ValidationError):
        await node.process({
            "measurement": {
                "temperature": "invalid",  # Should be float
                "humidity": 60.0
            }
        })
    
    await node.cleanup()

@pytest.mark.asyncio
async def test_mcp_multiple_samples(mock_llm_node, base_schemas):
    """Test MCP prediction with multiple samples."""
    config = MCPConfig(
        model="mock_model",
        num_samples=3,
        measurement_schema=base_schemas["measurement"],
        prediction_schema=base_schemas["prediction"]
    )
    
    node = MCPNode(
        name="test_mcp",
        config=config,
        llm_node=mock_llm_node
    )
    await node.initialize()
    
    result = await node.process({
        "measurement": {
            "temperature": 25.0,
            "humidity": 60.0
        }
    })
    
    assert len(result["predictions"]) == 3
    assert result["confidence"] == 1.0 / 3
    
    # Test num_samples override
    result = await node.process({
        "measurement": {
            "temperature": 25.0,
            "humidity": 60.0
        },
        "num_samples": 2
    })
    
    assert len(result["predictions"]) == 2
    assert result["confidence"] == 0.5
    
    await node.cleanup()