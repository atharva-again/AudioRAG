from __future__ import annotations

import pytest

from audiorag.core.protocols.vector_store import VectorStoreProvider


def test_verifiable_protocol_runtime_checkable() -> None:
    from audiorag.core.protocols.vector_store import VerifiableVectorStoreProvider

    assert hasattr(VerifiableVectorStoreProvider, "_is_protocol")
    assert VerifiableVectorStoreProvider._is_protocol is True


def test_verifiable_protocol_has_verify() -> None:
    from audiorag.core.protocols.vector_store import VerifiableVectorStoreProvider

    assert hasattr(VerifiableVectorStoreProvider, "verify")


@pytest.mark.asyncio
async def test_verifiable_vector_store_isinstance_passes() -> None:
    from audiorag.core.protocols.vector_store import VerifiableVectorStoreProvider

    class FakeStore:
        async def add(
            self,
            ids: list[str],
            embeddings: list[list[float]],
            metadatas: list[dict],
            documents: list[str],
        ) -> None:
            return None

        async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
            return []

        async def delete_by_source_id(self, source_id: str) -> None:
            return None

        async def verify(self, ids: list[str]) -> bool:
            return True

    store = FakeStore()
    assert isinstance(store, VectorStoreProvider)
    assert isinstance(store, VerifiableVectorStoreProvider)
