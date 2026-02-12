from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from audiorag import AudioRAGPipeline
from audiorag.core.exceptions import PipelineError
from audiorag.core.id_strategy import resolve_vector_id_strategy
from audiorag.core.models import IndexingStatus


@pytest.mark.asyncio
async def test_pipeline_auto_weaviate_transforms_vector_ids_to_uuid5(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
) -> None:
    url = "https://youtube.com/watch?v=test123"
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_provider = "weaviate"
    mock_config.vector_id_format = "auto"
    all_mock_providers["vector_store"].verify = AsyncMock(return_value=True)

    pipeline = AudioRAGPipeline(
        config=mock_config,
        audio_source=all_mock_providers["audio_source"],
        stt=all_mock_providers["stt"],
        embedder=all_mock_providers["embedding"],
        vector_store=all_mock_providers["vector_store"],
        generator=all_mock_providers["generation"],
        reranker=all_mock_providers["reranker"],
    )

    try:
        await pipeline._ensure_initialized()
        await pipeline.index(url)

        stored_chunks = await pipeline._state.get_chunks_for_source(url)
        canonical_ids = [chunk["chunk_id"] for chunk in stored_chunks]

        added_ids = all_mock_providers["vector_store"].add.await_args.kwargs["ids"]
        verify_args = all_mock_providers["vector_store"].verify.await_args
        assert verify_args is not None
        verified_ids = verify_args.args[0]

        assert added_ids == verified_ids
        assert added_ids != canonical_ids
        assert all(uuid.UUID(vector_id).version == 5 for vector_id in added_ids)
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_pipeline_weaviate_respects_sha256_override(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
) -> None:
    url = "https://youtube.com/watch?v=test123"
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_provider = "weaviate"
    mock_config.vector_id_format = "sha256"
    all_mock_providers["vector_store"].verify = AsyncMock(return_value=True)

    pipeline = AudioRAGPipeline(
        config=mock_config,
        audio_source=all_mock_providers["audio_source"],
        stt=all_mock_providers["stt"],
        embedder=all_mock_providers["embedding"],
        vector_store=all_mock_providers["vector_store"],
        generator=all_mock_providers["generation"],
        reranker=all_mock_providers["reranker"],
    )

    try:
        await pipeline._ensure_initialized()
        await pipeline.index(url)

        stored_chunks = await pipeline._state.get_chunks_for_source(url)
        canonical_ids = [chunk["chunk_id"] for chunk in stored_chunks]

        added_ids = all_mock_providers["vector_store"].add.await_args.kwargs["ids"]
        verify_args = all_mock_providers["vector_store"].verify.await_args
        assert verify_args is not None
        verified_ids = verify_args.args[0]

        assert added_ids == verified_ids
        assert added_ids == canonical_ids
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_index_raises_on_vector_id_strategy_change_without_force(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
) -> None:
    url = "https://youtube.com/watch?v=test123"
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_provider = "weaviate"
    mock_config.vector_id_format = "sha256"

    pipeline = AudioRAGPipeline(
        config=mock_config,
        audio_source=all_mock_providers["audio_source"],
        stt=all_mock_providers["stt"],
        embedder=all_mock_providers["embedding"],
        vector_store=all_mock_providers["vector_store"],
        generator=all_mock_providers["generation"],
        reranker=all_mock_providers["reranker"],
    )

    try:
        await pipeline._ensure_initialized()
        previous_vector_id_strategy = resolve_vector_id_strategy(
            provider_name="weaviate",
            configured_format="uuid5",
            uuid5_namespace=None,
            vector_store=pipeline._vector_store,
        )
        await pipeline._state.upsert_source(
            url,
            IndexingStatus.FAILED,
            metadata={"vector_id_strategy": previous_vector_id_strategy},
        )

        with pytest.raises(PipelineError, match="Vector ID strategy changed"):
            await pipeline.index(url)
    finally:
        await pipeline._state.close()
