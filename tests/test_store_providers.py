"""Unit tests for vector store providers - instantiation tests."""

import pytest
from tempfile import TemporaryDirectory


class TestVectorStoreProvidersInstantiation:
    """Test vector store provider instantiation through lazy imports."""

    def test_chromadb_instantiation(self) -> None:
        """Test ChromaDB vector store can be instantiated."""
        from audiorag.store.chromadb import ChromaDBVectorStore

        with TemporaryDirectory() as tmpdir:
            store = ChromaDBVectorStore(persist_directory=tmpdir)
            assert store._provider_name == "chromadb_vector_store"
            assert store._collection_name == "audiorag"

    def test_chromadb_custom_collection(self) -> None:
        """Test ChromaDB with custom collection name."""
        from audiorag.store.chromadb import ChromaDBVectorStore

        with TemporaryDirectory() as tmpdir:
            store = ChromaDBVectorStore(
                persist_directory=tmpdir,
                collection_name="custom_collection",
            )
            assert store._collection_name == "custom_collection"
