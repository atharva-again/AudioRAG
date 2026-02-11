from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from audiorag import AudioRAGPipeline
from audiorag.core.exceptions import PipelineError
from audiorag.core.models import IndexingStatus


@pytest.fixture
def vector_store_with_verification(mock_vector_store_provider: AsyncMock) -> AsyncMock:
    mock_vector_store_provider.verify = AsyncMock(return_value=True)
    return mock_vector_store_provider


@pytest.mark.asyncio
async def test_embed_stage_verifies_on_success(
    mock_config,
    all_mock_providers,
    vector_store_with_verification,
    tmp_audio_dir,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_verify_mode = "best_effort"
    all_mock_providers["vector_store"] = vector_store_with_verification

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
        await pipeline.index("https://youtube.com/watch?v=test123")
        all_mock_providers["vector_store"].verify.assert_called_once()
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_embed_stage_skips_verification_when_missing(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_verify_mode = "best_effort"

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
        await pipeline.index("https://youtube.com/watch?v=test123")
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_embed_stage_strict_mode_requires_verification_method(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_verify_mode = "strict"

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
        with pytest.raises(PipelineError):
            await pipeline.index("https://youtube.com/watch?v=test123")
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_embed_stage_strict_mode_blocks_on_failed_verification(
    mock_config,
    all_mock_providers,
    vector_store_with_verification,
    tmp_audio_dir,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_verify_mode = "strict"
    vector_store_with_verification.verify = AsyncMock(return_value=False)
    all_mock_providers["vector_store"] = vector_store_with_verification

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
        with pytest.raises(PipelineError):
            await pipeline.index("https://youtube.com/watch?v=test123")
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_embed_stage_best_effort_allows_failed_verification(
    mock_config,
    all_mock_providers,
    vector_store_with_verification,
    tmp_audio_dir,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_verify_mode = "best_effort"
    vector_store_with_verification.verify = AsyncMock(return_value=False)
    all_mock_providers["vector_store"] = vector_store_with_verification

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
        await pipeline.index("https://youtube.com/watch?v=test123")
        status = await pipeline._state.get_source_status("https://youtube.com/watch?v=test123")
        assert status is not None
        assert status["status"] == IndexingStatus.COMPLETED
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_embed_stage_verification_retries_until_success(
    mock_config,
    all_mock_providers,
    vector_store_with_verification,
    tmp_audio_dir,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_verify_mode = "strict"
    mock_config.vector_store_verify_max_attempts = 3
    mock_config.vector_store_verify_wait_seconds = 0.0
    vector_store_with_verification.verify = AsyncMock(side_effect=[False, False, True])
    all_mock_providers["vector_store"] = vector_store_with_verification

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
        await pipeline.index("https://youtube.com/watch?v=test123")
        assert vector_store_with_verification.verify.await_count == 3
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_embed_stage_strict_mode_raises_after_retry_exhaustion(
    mock_config,
    all_mock_providers,
    vector_store_with_verification,
    tmp_audio_dir,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.vector_store_verify_mode = "strict"
    mock_config.vector_store_verify_max_attempts = 2
    mock_config.vector_store_verify_wait_seconds = 0.0
    vector_store_with_verification.verify = AsyncMock(return_value=False)
    all_mock_providers["vector_store"] = vector_store_with_verification

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
        with pytest.raises(PipelineError):
            await pipeline.index("https://youtube.com/watch?v=test123")
        assert vector_store_with_verification.verify.await_count == 2
    finally:
        await pipeline._state.close()
