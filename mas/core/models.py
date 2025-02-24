"""Common data models and configuration classes."""
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID, uuid4

class Document:
    """Represents a document with content and metadata."""
    def __init__(self, content: str, metadata: Dict[str, Any] = None, embedding: Optional[List[float]] = None):
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding

class BaseConfig:
    """Base configuration class with common settings."""
    def __init__(self, debug: bool = False, log_level: str = "INFO", 
                 batch_size: int = 32, max_retries: int = 3, timeout: float = 30.0):
        self.debug = debug
        self.log_level = log_level
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

class ModelProviderConfig(BaseConfig):
    """Base configuration for model providers."""
    def __init__(self, provider_name: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.7, streaming: bool = False,
                 retry_on_failure: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.provider_name = provider_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.streaming = streaming
        self.retry_on_failure = retry_on_failure

__all__ = [
    "Document",
    "BaseConfig",
    "ModelProviderConfig",
]