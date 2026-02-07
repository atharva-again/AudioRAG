"""Vector store providers."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import with clear error messages for missing dependencies."""
    if name == "ChromaDBVectorStore":
        try:
            from audiorag.store.chromadb import ChromaDBVectorStore

            return ChromaDBVectorStore
        except ImportError:
            raise ImportError(
                "ChromaDBVectorStore requires 'chromadb'. "
                "Install with: pip install audiorag[chromadb]"
            ) from None
    if name == "PineconeVectorStore":
        try:
            from audiorag.store.pinecone import PineconeVectorStore

            return PineconeVectorStore
        except ImportError:
            raise ImportError(
                "PineconeVectorStore requires 'pinecone-client'. "
                "Install with: pip install audiorag[pinecone]"
            ) from None
    if name == "WeaviateVectorStore":
        try:
            from audiorag.store.weaviate import WeaviateVectorStore

            return WeaviateVectorStore
        except ImportError:
            raise ImportError(
                "WeaviateVectorStore requires 'weaviate-client'. "
                "Install with: pip install audiorag[weaviate]"
            ) from None
    if name == "SupabasePgVectorStore":
        try:
            from audiorag.store.supabase import SupabasePgVectorStore

            return SupabasePgVectorStore
        except ImportError:
            raise ImportError(
                "SupabasePgVectorStore requires 'supabase'. "
                "Install with: pip install audiorag[supabase]"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChromaDBVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "SupabasePgVectorStore",
]
