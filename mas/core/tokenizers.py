"""Tokenizer implementations for different model providers."""
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import tiktoken
from transformers import AutoTokenizer
from pydantic import BaseModel

class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids back to text."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        pass
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum number of tokens."""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])

class OpenAITokenizer(BaseTokenizer):
    """Tokenizer for OpenAI models using tiktoken."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fall back to cl100k_base encoding for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

class HuggingFaceTokenizer(BaseTokenizer):
    """Tokenizer wrapper for Hugging Face tokenizers."""
    
    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

class OllamaTokenizer(BaseTokenizer):
    """Tokenizer for Ollama models using tiktoken cl100k_base."""
    
    def __init__(self):
        # Use cl100k_base as default encoding for Ollama models
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

TOKENIZERS = {
    "openai": OpenAITokenizer,
    "huggingface": HuggingFaceTokenizer,
    "ollama": OllamaTokenizer,
}

def get_tokenizer(provider: str, model_name: Optional[str] = None) -> BaseTokenizer:
    """Get appropriate tokenizer for provider and model."""
    tokenizer_cls = TOKENIZERS.get(provider)
    if not tokenizer_cls:
        raise ValueError(f"No tokenizer available for provider: {provider}")
        
    if provider == "openai":
        return tokenizer_cls(model_name) if model_name else tokenizer_cls()
    elif provider == "huggingface":
        if not model_name:
            raise ValueError("model_name required for HuggingFace tokenizer")
        return tokenizer_cls(model_name)
    else:
        return tokenizer_cls()