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
        await pipeline.index("file:///tmp/test_audio.mp3")

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
            await pipeline.index("file:///tmp/test_audio.mp3")

        assert isinstance(exc_info.value.__cause__, BudgetExceededError)
        assert all_mock_providers["stt"].transcribe.await_count == 0
        status = await pipeline._state.get_source_status("file:///tmp/test_audio.mp3")
        assert status is not None
        assert status["status"] == "failed"
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_index_releases_budget_when_download_fails(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
    mocker,
) -> None:
    mock_config.work_dir = tmp_audio_dir
    mock_config.budget_enabled = True
    mock_config.budget_audio_seconds_per_hour = 300

    all_mock_providers["audio_source"].download.side_effect = Exception("Download failed")

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
        spy = mocker.spy(pipeline._budget_governor, "release")

        with pytest.raises(PipelineError):
            await pipeline.index("file:///tmp/test_audio.mp3")

        assert spy.call_count == 1
        assert spy.call_args.kwargs["audio_seconds"] == 120
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_index_reconciles_duration_when_actual_exceeds_estimate(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
    tmp_path,
    mocker,
) -> None:
    from audiorag.core.models import AudioFile, SourceMetadata

    mock_config.work_dir = tmp_audio_dir
    mock_config.budget_enabled = True
    mock_config.budget_audio_seconds_per_hour = 300

    audio_path = tmp_path / "mock_audio.mp3"
    audio_path.write_bytes(b"dummy audio content")

    all_mock_providers["audio_source"].get_metadata.return_value = SourceMetadata(
        duration=100.0,
        title="Test Video",
    )
    all_mock_providers["audio_source"].download.return_value = AudioFile(
        path=audio_path,
        source_url="file:///tmp/test_audio.mp3",
        title="Test Video Title",
        duration=150.0,
    )

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

        await pipeline.index("file:///tmp/test_audio.mp3")

        audio_reservations = [
            call.kwargs for call in spy.call_args_list if call.kwargs.get("audio_seconds", 0) > 0
        ]
        assert len(audio_reservations) == 2
        assert audio_reservations[0]["audio_seconds"] == 100
        assert audio_reservations[1]["audio_seconds"] == 50
    finally:
        await pipeline._state.close()


@pytest.mark.asyncio
async def test_index_reconciles_duration_when_actual_less_than_estimate(
    mock_config,
    all_mock_providers,
    tmp_audio_dir,
    tmp_path,
    mocker,
) -> None:
    from audiorag.core.models import AudioFile, SourceMetadata

    mock_config.work_dir = tmp_audio_dir
    mock_config.budget_enabled = True
    mock_config.budget_audio_seconds_per_hour = 300

    audio_path = tmp_path / "mock_audio.mp3"
    audio_path.write_bytes(b"dummy audio content")

    all_mock_providers["audio_source"].get_metadata.return_value = SourceMetadata(
        duration=150.0,
        title="Test Video",
    )
    all_mock_providers["audio_source"].download.return_value = AudioFile(
        path=audio_path,
        source_url="file:///tmp/test_audio.mp3",
        title="Test Video Title",
        duration=100.0,
    )

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
        reserve_spy = mocker.spy(pipeline._budget_governor, "reserve")
        release_spy = mocker.spy(pipeline._budget_governor, "release")

        await pipeline.index("file:///tmp/test_audio.mp3")

        audio_reservations = [
            call.kwargs
            for call in reserve_spy.call_args_list
            if call.kwargs.get("audio_seconds", 0) > 0
        ]
        assert len(audio_reservations) == 1
        assert audio_reservations[0]["audio_seconds"] == 150

        audio_releases = [
            call.kwargs for call in release_spy.call_args_list if call.kwargs.get("audio_seconds")
        ]
        assert len(audio_releases) == 1
        assert audio_releases[0]["audio_seconds"] == 50
    finally:
        await pipeline._state.close()
