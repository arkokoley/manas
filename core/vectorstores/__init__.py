"""Vector store provider registry."""
from typing import Dict, Type
from .base import VectorStoreProvider
from .faiss import FAISSVectorStore
from .chroma import ChromaVectorStore

VECTORSTORES = {
    "faiss": FAISSVectorStore,
    "chroma": ChromaVectorStore,
}

__all__ = [
    "VectorStoreProvider",
    "FAISSVectorStore",
    "ChromaVectorStore",
    "VECTORSTORES"
]