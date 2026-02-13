from __future__ import annotations

import uuid
from typing import Literal

import pytest

from audiorag.core.id_strategy import (
    resolve_vector_id_format,
    resolve_vector_id_strategy,
    to_provider_vector_ids,
)


class _VectorStoreWithUuidDefault:
    vector_id_default_format: Literal["sha256", "uuid5"] = "uuid5"

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


def test_resolve_vector_id_format_auto_weaviate() -> None:
    assert resolve_vector_id_format("weaviate", "auto") == "uuid5"


def test_resolve_vector_id_format_auto_defaults_to_sha256() -> None:
    assert resolve_vector_id_format("supabase", "auto") == "sha256"


def test_resolve_vector_id_format_prefers_vector_store_capability() -> None:
    vector_store = _VectorStoreWithUuidDefault()
    assert resolve_vector_id_format("supabase", "auto", vector_store=vector_store) == "uuid5"


def test_resolve_vector_id_strategy_sha256() -> None:
    strategy = resolve_vector_id_strategy(
        provider_name="supabase",
        configured_format="sha256",
        uuid5_namespace=None,
    )
    assert strategy == "sha256"


def test_resolve_vector_id_strategy_uuid5_with_default_namespace() -> None:
    strategy = resolve_vector_id_strategy(
        provider_name="weaviate",
        configured_format="auto",
        uuid5_namespace=None,
    )
    assert strategy.startswith("uuid5:")
    assert strategy.count(":") == 1


def test_to_provider_vector_ids_sha256_returns_canonical_ids() -> None:
    canonical_ids = ["abc123", "def456"]

    result = to_provider_vector_ids(
        canonical_ids=canonical_ids,
        provider_name="chromadb",
        configured_format="sha256",
        uuid5_namespace=None,
    )

    assert result == canonical_ids


def test_to_provider_vector_ids_uuid5_is_deterministic() -> None:
    canonical_ids = ["a" * 64, "b" * 64]

    first = to_provider_vector_ids(
        canonical_ids=canonical_ids,
        provider_name="weaviate",
        configured_format="auto",
        uuid5_namespace=None,
    )
    second = to_provider_vector_ids(
        canonical_ids=canonical_ids,
        provider_name="weaviate",
        configured_format="auto",
        uuid5_namespace=None,
    )

    assert first == second
    assert all(uuid.UUID(vector_id).version == 5 for vector_id in first)


def test_to_provider_vector_ids_uuid5_namespace_override_changes_output() -> None:
    canonical_ids = ["a" * 64]

    default_ids = to_provider_vector_ids(
        canonical_ids=canonical_ids,
        provider_name="weaviate",
        configured_format="uuid5",
        uuid5_namespace=None,
    )
    custom_ids = to_provider_vector_ids(
        canonical_ids=canonical_ids,
        provider_name="weaviate",
        configured_format="uuid5",
        uuid5_namespace="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    )

    assert default_ids != custom_ids


def test_to_provider_vector_ids_invalid_namespace_raises() -> None:
    with pytest.raises(ValueError, match="vector_id_uuid5_namespace must be a valid UUID string"):
        to_provider_vector_ids(
            canonical_ids=["a" * 64],
            provider_name="weaviate",
            configured_format="uuid5",
            uuid5_namespace="not-a-uuid",
        )
