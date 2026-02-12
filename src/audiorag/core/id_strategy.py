from __future__ import annotations

import uuid
from typing import Literal

from audiorag.core.protocols import VectorIdFormatAwareProvider, VectorStoreProvider

VectorIdFormat = Literal["auto", "sha256", "uuid5"]
ResolvedVectorIdFormat = Literal["sha256", "uuid5"]

_DEFAULT_UUID5_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "audiorag/vector-id/v1")

_AUTO_PROVIDER_FORMATS: dict[str, ResolvedVectorIdFormat] = {
    "weaviate": "uuid5",
}


def resolve_vector_id_format(
    provider_name: str,
    configured_format: VectorIdFormat,
    vector_store: VectorStoreProvider | None = None,
) -> ResolvedVectorIdFormat:
    if configured_format != "auto":
        return configured_format

    if vector_store is not None and isinstance(vector_store, VectorIdFormatAwareProvider):
        return vector_store.vector_id_default_format

    return _AUTO_PROVIDER_FORMATS.get(provider_name.lower(), "sha256")


def _parse_uuid5_namespace(namespace: str | None) -> uuid.UUID:
    if namespace is None:
        return _DEFAULT_UUID5_NAMESPACE
    try:
        return uuid.UUID(namespace)
    except ValueError as exc:
        raise ValueError("vector_id_uuid5_namespace must be a valid UUID string") from exc


def to_provider_vector_ids(
    canonical_ids: list[str],
    provider_name: str,
    configured_format: VectorIdFormat,
    uuid5_namespace: str | None,
    vector_store: VectorStoreProvider | None = None,
) -> list[str]:
    effective_format = resolve_vector_id_format(
        provider_name=provider_name,
        configured_format=configured_format,
        vector_store=vector_store,
    )
    if effective_format == "sha256":
        return canonical_ids

    namespace_uuid = _parse_uuid5_namespace(uuid5_namespace)
    return [str(uuid.uuid5(namespace_uuid, canonical_id)) for canonical_id in canonical_ids]


def resolve_vector_id_strategy(
    provider_name: str,
    configured_format: VectorIdFormat,
    uuid5_namespace: str | None,
    vector_store: VectorStoreProvider | None = None,
) -> str:
    effective_format = resolve_vector_id_format(
        provider_name=provider_name,
        configured_format=configured_format,
        vector_store=vector_store,
    )
    if effective_format == "sha256":
        return "sha256"

    namespace_uuid = _parse_uuid5_namespace(uuid5_namespace)
    return f"uuid5:{namespace_uuid}"
