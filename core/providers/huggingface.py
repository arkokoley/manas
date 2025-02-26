"""Hugging Face provider implementation."""
from typing import Any, Dict, Optional, Union, AsyncIterator
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, TextIteratorStreamer
from torch import cuda, float32
import torch
from pydantic import Field

from core.providers.base import BaseLLMProvider
from core.models import ModelProviderConfig

class HuggingFaceConfig(ModelProviderConfig):
    """Configuration for Hugging Face provider."""
    model_name: str = Field(..., description="Name or path of the model to load")
    tokenizer_name: Optional[str] = Field(None, description="Name or path of tokenizer if different from model")
    device: str = Field("cuda" if cuda.is_available() else "cpu", description="Device to load model on")
    torch_dtype: str = Field("float32", description="Data type for model weights")
    embedding_model: Optional[str] = Field(None, description="Model to use for embeddings")
    
    model_config = dict(protected_namespaces=())
    provider: str = "huggingface"

class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face provider implementation."""
    
    def __init__(self, config: HuggingFaceConfig):
        self.config = config

    async def initialize(self):
        """Initialize models and move to specified device."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name or self.config.model_name
        )
        dtype = getattr(torch, self.config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device
        )
        
        if self.config.embedding_model:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model
            )
            self.embedding_model = AutoModel.from_pretrained(
                self.config.embedding_model
            ).to(self.config.device)
    
    async def cleanup(self):
        """Free GPU memory and cleanup resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'embedding_model'):
            del self.embedding_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, prompt, temperature, max_tokens, stop_sequences, kwargs
        )
    
    def _generate_sync(self, prompt, temperature, max_tokens, stop_sequences, kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            temperature=temperature,
            max_new_tokens=max_tokens,
            stopping_criteria=self._make_stopping_criteria(stop_sequences) if stop_sequences else None,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    async def stream_generate(self, 
        prompt: Union[str, Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer)
        
        generation_kwargs = dict(
            **inputs,
            temperature=temperature,
            max_new_tokens=max_tokens,
            streamer=streamer,
            **kwargs
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
            if stop_sequences and any(stop in text for stop in stop_sequences):
                break
    
    async def embed(self, text: str) -> list[float]:
        if not hasattr(self, 'embedding_model'):
            raise ValueError("No embedding model configured")
            
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)
    
    def _embed_sync(self, text: str) -> list[float]:
        inputs = self.embedding_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.embedding_model.device)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use CLS token embedding or mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings[0].tolist()
    
    def _make_stopping_criteria(self, stop_sequences: list[str]):
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopOnTokens(StoppingCriteria):
            def __init__(self, stops = [], tokenizer=None):
                super().__init__()
                self.stops = [
                    tokenizer(stop, return_tensors='pt').input_ids.squeeze() 
                    for stop in stops
                ]
            
            def __call__(self, input_ids, scores):
                for stop in self.stops:
                    if torch.all((input_ids[0][-len(stop):] == stop)).item():
                        return True
                return False
                
        return StoppingCriteriaList([StopOnTokens(stop_sequences, self.tokenizer)])