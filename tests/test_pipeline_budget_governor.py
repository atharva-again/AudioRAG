from __future__ import annotations

import pytest

from audiorag import AudioRAGPipeline
from audiorag.core.exceptions import BudgetExceededError, PipelineError


@pytest.mark.asyncio
async def test_index_reserves_full_audio_budget_before_transcription(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
    mocker,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.budget_enabled = True
    mock_config.budget_audio_seconds_per_hour = 300

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
        spy = mocker.spy(pipeline._budget_governor, "reserve")
        await pipeline.index("https://youtube.com/watch?v=test123")

        audio_reservations = [
            call.kwargs for call in spy.call_args_list if call.kwargs.get("audio_seconds", 0) > 0
        ]
        assert len(audio_reservations) == 1
        assert audio_reservations[0]["audio_seconds"] == 120
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_index_fails_fast_when_audio_budget_insufficient(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.budget_enabled = True
    mock_config.budget_audio_seconds_per_hour = 60

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

        with pytest.raises(PipelineError) as exc_info:
            await pipeline.index("https://youtube.com/watch?v=test123")

        assert isinstance(exc_info.value.__cause__, BudgetExceededError)
        assert all_mock_providers["stt"].transcribe.await_count == 0
        status = await pipeline._state.get_source_status("https://youtube.com/watch?v=test123")
        assert status is not None
        assert status["status"] == "failed"
    finally:
        await pipeline._state.close()
