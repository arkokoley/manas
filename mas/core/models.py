"""Common data models and configuration classes."""
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass
class Document:
    """Represents a document with content and metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    id: UUID = field(default_factory=uuid4)


class BaseConfig:
    """Base configuration class with common settings."""
    def __init__(
        self, 
        debug: bool = False,
        log_level: str = "INFO",
        batch_size: int = 32,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.debug = debug
        self._log_level = None
        self.log_level = log_level  # Uses the property setter for validation
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

    @property
    def log_level(self) -> str:
        return self._log_level

    @log_level.setter
    def log_level(self, value: str):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        self._log_level = value


class ModelProviderConfig(BaseConfig):
    """Base configuration for model providers."""
    def __init__(
        self,
        provider_name: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        streaming: bool = False,
        retry_on_failure: bool = True,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.provider_name = provider_name
        self.max_tokens = max_tokens
        self._temperature = None
        self.temperature = temperature  # Uses property setter for validation
        self.streaming = streaming
        self.retry_on_failure = retry_on_failure
        self.stop_sequences = stop_sequences or []

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        self._temperature = value


class VectorStoreConfig(BaseConfig):
    """Configuration for vector stores."""
    def __init__(
        self,
        store_type: str,
        dimension: int = 1536,
        similarity_metric: str = "cosine",
        persist_directory: Optional[str] = None,
        collection_name: str = "default",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.store_type = store_type
        self.dimension = dimension
        self.similarity_metric = similarity_metric
        self.persist_directory = persist_directory
        self.collection_name = collection_name


# Export all classes
__all__ = [
    "Document",
    "BaseConfig",
    "ModelProviderConfig",
    "VectorStoreConfig",
]