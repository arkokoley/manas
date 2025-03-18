"""Vector store provider registry."""
# Import protocol and factory
from .protocol import VectorStoreProtocol, VectorStoreError
from .factory import register_vectorstore, create_vectorstore, list_vectorstores

# Import base provider
from .base import VectorStoreProvider

# Import implementations to register them
from .faiss import FAISSVectorStore
from .chroma import ChromaVectorStore
from .pinecone import PineconeVectorStore

# For backward compatibility
VECTORSTORES = {
    "faiss": FAISSVectorStore,
    "chroma": ChromaVectorStore,
    "pinecone": PineconeVectorStore
}

# Export all
__all__ = [
    "VectorStoreProtocol",
    "VectorStoreProvider",
    "VectorStoreError",
    "register_vectorstore",
    "create_vectorstore", 
    "list_vectorstores",
    "FAISSVectorStore",
    "ChromaVectorStore",
    "PineconeVectorStore",
    "VECTORSTORES"  # Keep for backward compatibility
]