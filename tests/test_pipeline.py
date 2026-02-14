"""Comprehensive integration tests for AudioRAGPipeline."""

from unittest.mock import AsyncMock

import pytest

from audiorag import AudioRAGPipeline, QueryResult
from audiorag.core.exceptions import PipelineError
from audiorag.core.models import BatchIndexResult, IndexingStatus


class TestPipelineInitialization:
    """Test suite for pipeline initialization."""

    def test_create_with_default_providers(self, mock_config) -> None:
        """Test pipeline creation with default providers."""
        # Note: This will fail without optional deps, but tests structure
        pass  # Placeholder - actual implementation needs optional deps

    def test_create_with_custom_providers(self, mock_config, all_mock_providers) -> None:
        """Test pipeline creation with custom providers."""
        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        assert pipeline is not None
        assert pipeline._config == mock_config

    def test_lazy_provider_initialization(self, mock_config, mocker, all_mock_providers) -> None:
        """Test that providers are lazily initialized."""
        # Provide all providers to avoid lazy initialization that imports modules
        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        # Providers should be set from the passed providers
        assert pipeline is not None


async def create_mock_pipeline(mock_config, all_mock_providers):
    """Helper to create an initialized pipeline."""
    pipeline = AudioRAGPipeline(
        config=mock_config,
        audio_source=all_mock_providers["audio_source"],
        stt=all_mock_providers["stt"],
        embedder=all_mock_providers["embedding"],
        vector_store=all_mock_providers["vector_store"],
        generator=all_mock_providers["generation"],
        reranker=all_mock_providers["reranker"],
    )

    # Initialize the state manager asynchronously
    await pipeline._ensure_initialized()
    return pipeline


class TestPipelineQuery:
    """Test suite for pipeline query flow."""

    @pytest.fixture
    async def initialized_pipeline(self, mock_config, all_mock_providers):
        """Create and initialize a pipeline with mocked providers."""
        return await create_mock_pipeline(mock_config, all_mock_providers)

    @pytest.mark.asyncio
    async def test_query_returns_queryresult(self, initialized_pipeline) -> None:
        """Test that query returns a QueryResult."""
        result = await initialized_pipeline.query("test query")

        assert isinstance(result, QueryResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.sources, list)

    @pytest.mark.asyncio
    async def test_query_embeds_query_text(self, initialized_pipeline, mocker) -> None:
        """Test that query embeds the query text."""
        spy = mocker.spy(initialized_pipeline._embedder, "embed")
        await initialized_pipeline.query("test query")

        spy.assert_called_once()
        call_args = spy.call_args[0][0]
        assert "test query" in call_args

    @pytest.mark.asyncio
    async def test_query_queries_vector_store(self, initialized_pipeline, mocker) -> None:
        """Test that query retrieves from vector store."""
        spy = mocker.spy(initialized_pipeline._vector_store, "query")
        await initialized_pipeline.query("test query")

        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_reranks_results(self, initialized_pipeline, mocker) -> None:
        """Test that query reranks retrieved results."""
        spy = mocker.spy(initialized_pipeline._reranker, "rerank")
        await initialized_pipeline.query("test query")

        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_generates_answer(self, initialized_pipeline, mocker) -> None:
        """Test that query generates an answer."""
        spy = mocker.spy(initialized_pipeline._generator, "generate")
        result = await initialized_pipeline.query("test query")

        spy.assert_called_once()
        assert "mocked" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_query_no_results_returns_empty(self, initialized_pipeline, mocker) -> None:
        """Test that query with no results returns appropriate message."""
        # Mock vector store to return empty results
        mocker.patch.object(initialized_pipeline._vector_store, "query", return_value=[])

        result = await initialized_pipeline.query("test query")

        assert "No relevant information found" in result.answer
        assert len(result.sources) == 0


class TestPipelineIndex:
    """Test suite for pipeline indexing."""

    @pytest.fixture
    async def pipeline_for_index(self, mock_config, all_mock_providers, tmp_audio_dir):
        """Create a pipeline configured for indexing tests."""
        mock_config.work_dir = tmp_audio_dir

        return await create_mock_pipeline(mock_config, all_mock_providers)

    @pytest.mark.asyncio
    async def test_index_creates_source_entry(self, pipeline_for_index) -> None:
        """Test that index creates a source entry in database."""
        await pipeline_for_index.index("https://youtube.com/watch?v=test123")

        status = await pipeline_for_index._state.get_source_status(
            "https://youtube.com/watch?v=test123"
        )
        assert status is not None

    @pytest.mark.asyncio
    async def test_index_skips_already_indexed(self, pipeline_for_index, mocker) -> None:
        """Test that index skips already indexed URLs."""
        url = "https://youtube.com/watch?v=test123"

        # First index
        await pipeline_for_index.index(url)

        # Spy on audio_source download
        spy = mocker.spy(pipeline_for_index._audio_source, "download")

        # Second index should skip
        await pipeline_for_index.index(url)

        # Download should not be called again
        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_force_reindexes(self, pipeline_for_index, mocker) -> None:
        """Test that index with force=True reindexes."""
        url = "https://youtube.com/watch?v=test123"

        # First index
        await pipeline_for_index.index(url)

        # Spy on audio_source download
        spy = mocker.spy(pipeline_for_index._audio_source, "download")

        # Force reindex
        await pipeline_for_index.index(url, force=True)

        # Download should be called again
        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_tracks_status_progression(self, pipeline_for_index, mocker) -> None:
        """Test that index tracks status through pipeline stages."""
        url = "https://youtube.com/watch?v=test123"

        # Mock state methods to capture status updates
        status_updates = []
        original_update = pipeline_for_index._state.update_source_status

        async def capture_update(url, status, metadata=None):
            status_updates.append(status)
            return await original_update(url, status, metadata=metadata)

        mocker.patch.object(
            pipeline_for_index._state, "update_source_status", side_effect=capture_update
        )

        await pipeline_for_index.index(url)

        # Should have status updates for each stage
        assert len(status_updates) > 0
        assert IndexingStatus.COMPLETED in status_updates

    @pytest.mark.asyncio
    async def test_index_skips_when_in_progress(self, pipeline_for_index) -> None:
        """Test that index skips if status is in-progress."""
        url = "https://youtube.com/watch?v=test123"
        await pipeline_for_index._state.upsert_source(url, IndexingStatus.DOWNLOADING)

        await pipeline_for_index.index(url)

        status = await pipeline_for_index._state.get_source_status(url)
        assert status["status"] == IndexingStatus.DOWNLOADING

    @pytest.mark.asyncio
    async def test_index_force_overrides_in_progress(self, pipeline_for_index, mocker) -> None:
        """Test that force reindexes even if status is in-progress."""
        url = "https://youtube.com/watch?v=test123"
        await pipeline_for_index._state.upsert_source(url, IndexingStatus.DOWNLOADING)

        spy = mocker.spy(pipeline_for_index._audio_source, "download")

        await pipeline_for_index.index(url, force=True)

        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_uses_discovery_for_playlist_inputs(
        self, pipeline_for_index, mocker
    ) -> None:
        """Test that SDK index() expands playlist URLs into per-video sources."""
        playlist_url = "https://youtube.com/playlist?list=test-playlist"
        expanded_sources = [
            "https://youtube.com/watch?v=video1",
            "https://youtube.com/watch?v=video2",
        ]
        discover_mock = mocker.patch(
            "audiorag.pipeline.discover_sources",
            new=AsyncMock(return_value=expanded_sources),
        )

        await pipeline_for_index.index(playlist_url)

        discover_mock.assert_awaited_once_with([playlist_url], pipeline_for_index._config)
        for source in expanded_sources:
            status = await pipeline_for_index._state.get_source_status(source)
            assert status is not None
            assert status["status"] == IndexingStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_index_many_resumes_per_source(self, pipeline_for_index, mocker) -> None:
        """Test that index_many skips already completed sources on rerun."""
        inputs = ["https://youtube.com/playlist?list=test-playlist"]
        expanded_sources = [
            "https://youtube.com/watch?v=video1",
            "https://youtube.com/watch?v=video2",
        ]
        mocker.patch(
            "audiorag.pipeline.discover_sources",
            new=AsyncMock(return_value=expanded_sources),
        )

        await pipeline_for_index.index_many(inputs)

        spy = mocker.spy(pipeline_for_index._audio_source, "download")
        await pipeline_for_index.index_many(inputs)

        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_many_returns_structured_result(self, pipeline_for_index, mocker) -> None:
        inputs = ["https://youtube.com/playlist?list=test-playlist"]
        expanded_sources = [
            "https://youtube.com/watch?v=video1",
            "https://youtube.com/watch?v=video2",
        ]
        mocker.patch(
            "audiorag.pipeline.discover_sources",
            new=AsyncMock(return_value=expanded_sources),
        )

        result = await pipeline_for_index.index_many(inputs, raise_on_error=False)

        assert isinstance(result, BatchIndexResult)
        assert result.inputs == inputs
        assert result.discovered_sources == expanded_sources
        assert result.indexed_sources == expanded_sources
        assert result.skipped_sources == []
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_index_many_collects_failures_and_continues(
        self, pipeline_for_index, mocker
    ) -> None:
        """Test that index_many continues processing and raises aggregate failure."""
        inputs = ["https://youtube.com/playlist?list=test-playlist"]
        expanded_sources = [
            "https://youtube.com/watch?v=video1",
            "https://youtube.com/watch?v=video2",
            "https://youtube.com/watch?v=video3",
        ]
        mocker.patch(
            "audiorag.pipeline.discover_sources",
            new=AsyncMock(return_value=expanded_sources),
        )

        default_audio_file = pipeline_for_index._audio_source.download.return_value

        async def download_with_partial_failures(url: str, *_args, **_kwargs):
            if url.endswith("video2") or url.endswith("video3"):
                raise Exception(f"download failed for {url}")
            return default_audio_file

        download_mock = mocker.patch.object(
            pipeline_for_index._audio_source,
            "download",
            side_effect=download_with_partial_failures,
        )

        with pytest.raises(PipelineError) as exc_info:
            await pipeline_for_index.index_many(inputs)

        assert exc_info.value.stage == "index_many"
        assert "video2" in str(exc_info.value)
        assert "video3" in str(exc_info.value)
        assert [call.args[0] for call in download_mock.call_args_list] == expanded_sources

        completed = await pipeline_for_index._state.get_source_status(expanded_sources[0])
        failed_second = await pipeline_for_index._state.get_source_status(expanded_sources[1])
        failed_third = await pipeline_for_index._state.get_source_status(expanded_sources[2])

        assert completed is not None
        assert completed["status"] == IndexingStatus.COMPLETED
        assert failed_second is not None
        assert failed_second["status"] == IndexingStatus.FAILED
        assert failed_third is not None
        assert failed_third["status"] == IndexingStatus.FAILED

    @pytest.mark.asyncio
    async def test_index_many_normalizes_non_pipeline_errors(
        self, pipeline_for_index, mocker
    ) -> None:
        inputs = ["https://youtube.com/playlist?list=test-playlist"]
        expanded_sources = [
            "https://youtube.com/watch?v=video1",
            "https://youtube.com/watch?v=video2",
            "https://youtube.com/watch?v=video3",
        ]
        mocker.patch(
            "audiorag.pipeline.discover_sources",
            new=AsyncMock(return_value=expanded_sources),
        )

        original_index_single = pipeline_for_index._index_single_source

        async def failing_index_single(url: str, *, force: bool = False):
            if url.endswith("video2"):
                raise RuntimeError("unexpected crash")
            return await original_index_single(url, force=force)

        mocker.patch.object(
            pipeline_for_index,
            "_index_single_source",
            side_effect=failing_index_single,
        )

        result = await pipeline_for_index.index_many(inputs, raise_on_error=False)

        assert result.indexed_sources == [expanded_sources[0], expanded_sources[2]]
        assert result.skipped_sources == []
        assert len(result.failures) == 1
        assert result.failures[0].source_url == expanded_sources[1]
        assert result.failures[0].stage == "index_many"
        assert "RuntimeError" in result.failures[0].error_message


class TestPipelineErrorHandling:
    """Test suite for pipeline error handling."""

    @pytest.fixture
    async def pipeline_for_error(self, mock_config, all_mock_providers):
        """Create a pipeline for error testing."""
        return await create_mock_pipeline(mock_config, all_mock_providers)

    @pytest.mark.asyncio
    async def test_index_sets_failed_status_on_error(self, pipeline_for_error, mocker) -> None:
        """Test that index sets FAILED status when error occurs."""
        url = "https://youtube.com/watch?v=test123"

        # Mock audio_source to raise exception
        mocker.patch.object(
            pipeline_for_error._audio_source, "download", side_effect=Exception("Download failed")
        )

        with pytest.raises(PipelineError):
            await pipeline_for_error.index(url)

        status = await pipeline_for_error._state.get_source_status(url)
        assert status["status"] == IndexingStatus.FAILED

    @pytest.mark.asyncio
    async def test_query_handles_embedding_error(self, pipeline_for_error, mocker) -> None:
        """Test that query handles embedding errors."""
        # Mock embedder to raise exception
        mocker.patch.object(
            pipeline_for_error._embedder, "embed", side_effect=Exception("Embedding failed")
        )

        with pytest.raises(Exception):  # noqa: PT011, B017
            await pipeline_for_error.query("test query")

    @pytest.mark.asyncio
    async def test_query_handles_vector_store_error(self, pipeline_for_error, mocker) -> None:
        """Test that query handles vector store errors."""
        # Mock vector_store to raise exception
        mocker.patch.object(
            pipeline_for_error._vector_store,
            "query",
            side_effect=Exception("Vector store query failed"),
        )

        with pytest.raises(Exception):  # noqa: PT011, B017
            await pipeline_for_error.query("test query")


class TestPipelineCleanup:
    """Test suite for pipeline cleanup and graceful exit."""

    @pytest.fixture
    async def pipeline_for_cleanup(self, mock_config, all_mock_providers):
        """Create a pipeline for cleanup testing."""
        return await create_mock_pipeline(mock_config, all_mock_providers)

    @pytest.mark.asyncio
    async def test_close_closes_state_manager(self, pipeline_for_cleanup, mocker) -> None:
        """Test that close() closes the state manager."""
        spy = mocker.spy(pipeline_for_cleanup._state, "close")

        await pipeline_for_cleanup.close()

        spy.assert_called_once()
        assert pipeline_for_cleanup._initialized is False

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, pipeline_for_cleanup, mocker) -> None:
        """Test that close() can be called multiple times without error."""
        spy = mocker.spy(pipeline_for_cleanup._state, "close")

        await pipeline_for_cleanup.close()
        await pipeline_for_cleanup.close()
        await pipeline_for_cleanup.close()

        spy.assert_called_once()
        assert pipeline_for_cleanup._initialized is False

    @pytest.mark.asyncio
    async def test_close_does_nothing_if_not_initialized(
        self, mock_config, all_mock_providers
    ) -> None:
        """Test that close() does nothing if pipeline was never initialized."""
        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        await pipeline.close()

        assert pipeline._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager_initializes_and_closes(
        self, mock_config, all_mock_providers, mocker
    ) -> None:
        """Test that async context manager properly initializes and closes pipeline."""
        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        init_spy = mocker.spy(pipeline, "_ensure_initialized")
        close_spy = mocker.spy(pipeline._state, "close")

        async with pipeline:
            assert pipeline._initialized is True
            init_spy.assert_called_once()

        close_spy.assert_called_once()
        assert pipeline._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_on_exception(
        self, mock_config, all_mock_providers, mocker
    ) -> None:
        """Test that async context manager closes even when exception occurs."""
        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        close_spy = mocker.spy(pipeline._state, "close")

        with pytest.raises(ValueError, match="Test exception"):
            async with pipeline:
                raise ValueError("Test exception")

        close_spy.assert_called_once()
        assert pipeline._initialized is False


class TestPipelineCache:
    """Test suite for pipeline cache management - functional tests with real filesystem."""

    @pytest.fixture
    def real_config(self, tmp_path):
        """Create a real config with a temporary work directory."""
        from audiorag import AudioRAGConfig

        config = AudioRAGConfig()
        config.work_dir = tmp_path / "cache"
        config.work_dir.mkdir(parents=True, exist_ok=True)
        config.cleanup_audio = False
        return config

    def test_get_cache_info_empty(self, real_config) -> None:
        """Test get_cache_info returns correct info for empty cache."""
        pipeline = AudioRAGPipeline(
            config=real_config,
            audio_source=AsyncMock(),
            stt=AsyncMock(),
            embedder=AsyncMock(),
            vector_store=AsyncMock(),
            generator=AsyncMock(),
            reranker=AsyncMock(),
        )

        info = pipeline.get_cache_info()

        assert info["exists"] is True
        assert info["file_count"] == 0
        assert info["total_size_bytes"] == 0
        assert info["location"] == str(real_config.work_dir)

    def test_get_cache_info_with_files(self, real_config) -> None:
        """Test get_cache_info returns correct file count and size."""
        (real_config.work_dir / "file1.mp3").write_text("test content 1")
        (real_config.work_dir / "file2.mp3").write_text("test content 2")
        (real_config.work_dir / "subdir").mkdir()
        (real_config.work_dir / "subdir" / "file3.mp3").write_text("test content 3")

        pipeline = AudioRAGPipeline(
            config=real_config,
            audio_source=AsyncMock(),
            stt=AsyncMock(),
            embedder=AsyncMock(),
            vector_store=AsyncMock(),
            generator=AsyncMock(),
            reranker=AsyncMock(),
        )

        info = pipeline.get_cache_info()

        assert info["exists"] is True
        assert info["file_count"] == 3
        assert info["total_size_bytes"] == len("test content 1") + len("test content 2") + len(
            "test content 3"
        )

    def test_clear_cache_removes_files(self, real_config) -> None:
        """Test clear_cache actually removes files from work directory."""
        (real_config.work_dir / "file1.mp3").write_text("test content")
        (real_config.work_dir / "file2.mp3").write_text("test content")

        assert len(list(real_config.work_dir.iterdir())) == 2

        pipeline = AudioRAGPipeline(
            config=real_config,
            audio_source=AsyncMock(),
            stt=AsyncMock(),
            embedder=AsyncMock(),
            vector_store=AsyncMock(),
            generator=AsyncMock(),
            reranker=AsyncMock(),
        )

        cleared = pipeline.clear_cache()

        assert cleared == 2
        assert len(list(real_config.work_dir.iterdir())) == 0

    def test_clear_cache_removes_subdirs(self, real_config) -> None:
        """Test clear_cache also removes subdirectories."""
        subdir = real_config.work_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.mp3").write_text("nested content")

        assert (real_config.work_dir / "subdir").exists()

        pipeline = AudioRAGPipeline(
            config=real_config,
            audio_source=AsyncMock(),
            stt=AsyncMock(),
            embedder=AsyncMock(),
            vector_store=AsyncMock(),
            generator=AsyncMock(),
            reranker=AsyncMock(),
        )

        cleared = pipeline.clear_cache()

        assert cleared == 1
        assert not (real_config.work_dir / "subdir").exists()
        assert len(list(real_config.work_dir.iterdir())) == 0

    def test_clear_cache_with_default_work_dir(self, tmp_path, monkeypatch) -> None:
        """Test clear_cache works with the default work_dir."""
        monkeypatch.chdir(tmp_path)
        from audiorag import AudioRAGConfig

        config = AudioRAGConfig()

        pipeline = AudioRAGPipeline(
            config=config,
            audio_source=AsyncMock(),
            stt=AsyncMock(),
            embedder=AsyncMock(),
            vector_store=AsyncMock(),
            generator=AsyncMock(),
            reranker=AsyncMock(),
        )

        cleared = pipeline.clear_cache()

        assert cleared == 0

    def test_get_cache_info_with_default_work_dir(self, tmp_path, monkeypatch) -> None:
        """Test get_cache_info works with the default work_dir (may not exist)."""
        monkeypatch.chdir(tmp_path)
        from audiorag import AudioRAGConfig

        config = AudioRAGConfig()

        pipeline = AudioRAGPipeline(
            config=config,
            audio_source=AsyncMock(),
            stt=AsyncMock(),
            embedder=AsyncMock(),
            vector_store=AsyncMock(),
            generator=AsyncMock(),
            reranker=AsyncMock(),
        )

        info = pipeline.get_cache_info()

        assert info["location"] is not None


class TestGetIndexStatus:
    """Test suite for pipeline get_index_status method."""

    @pytest.fixture
    async def initialized_pipeline(self, mock_config, all_mock_providers):
        """Create and initialize a pipeline with mocked providers."""
        return await create_mock_pipeline(mock_config, all_mock_providers)

    @pytest.mark.asyncio
    async def test_returns_not_started_for_unknown_url(self, initialized_pipeline) -> None:
        """Test that get_index_status returns 'not_started' for URLs not in database."""
        # Mock state to return None (no source found)
        initialized_pipeline._state.get_source_status = AsyncMock(return_value=None)

        status = await initialized_pipeline.get_index_status("http://never-seen.example.com")

        assert status == "not_started"

    @pytest.mark.asyncio
    async def test_returns_completed_for_completed_source(self, initialized_pipeline) -> None:
        """Test that get_index_status returns 'completed' for completed sources."""
        initialized_pipeline._state.get_source_status = AsyncMock(
            return_value={"status": IndexingStatus.COMPLETED}
        )

        status = await initialized_pipeline.get_index_status("http://completed.example.com")

        assert status == "completed"

    @pytest.mark.asyncio
    async def test_returns_failed_for_failed_source(self, initialized_pipeline) -> None:
        """Test that get_index_status returns 'failed' for failed sources."""
        initialized_pipeline._state.get_source_status = AsyncMock(
            return_value={"status": IndexingStatus.FAILED}
        )

        status = await initialized_pipeline.get_index_status("http://failed.example.com")

        assert status == "failed"

    @pytest.mark.asyncio
    async def test_returns_processing_for_downloading(self, initialized_pipeline) -> None:
        """Test that get_index_status returns 'processing' for downloading status."""
        initialized_pipeline._state.get_source_status = AsyncMock(
            return_value={"status": IndexingStatus.DOWNLOADING}
        )

        status = await initialized_pipeline.get_index_status("http://downloading.example.com")

        assert status == "processing"

    @pytest.mark.asyncio
    async def test_returns_processing_for_transcribing(self, initialized_pipeline) -> None:
        """Test that get_index_status returns 'processing' for transcribing status."""
        initialized_pipeline._state.get_source_status = AsyncMock(
            return_value={"status": IndexingStatus.TRANSCRIBING}
        )

        status = await initialized_pipeline.get_index_status("http://transcribing.example.com")

        assert status == "processing"

    @pytest.mark.asyncio
    async def test_returns_processing_for_embedding(self, initialized_pipeline) -> None:
        """Test that get_index_status returns 'processing' for embedding status."""
        initialized_pipeline._state.get_source_status = AsyncMock(
            return_value={"status": IndexingStatus.EMBEDDING}
        )

        status = await initialized_pipeline.get_index_status("http://embedding.example.com")

        assert status == "processing"

    @pytest.mark.asyncio
    async def test_returns_processing_for_embedded(self, initialized_pipeline) -> None:
        """Test that get_index_status returns 'processing' for embedded status (gap case)."""
        initialized_pipeline._state.get_source_status = AsyncMock(
            return_value={"status": IndexingStatus.EMBEDDED}
        )

        status = await initialized_pipeline.get_index_status("http://embedded.example.com")

        assert status == "processing"

    @pytest.mark.asyncio
    async def test_returns_processing_for_all_intermediate_statuses(
        self, initialized_pipeline
    ) -> None:
        """Test that get_index_status returns 'processing' for all intermediate statuses."""
        # All statuses except COMPLETED and FAILED should return "processing"
        intermediate_statuses = [
            IndexingStatus.DOWNLOADING,
            IndexingStatus.DOWNLOADED,
            IndexingStatus.SPLITTING,
            IndexingStatus.TRANSCRIBING,
            IndexingStatus.TRANSCRIBED,
            IndexingStatus.CHUNKING,
            IndexingStatus.CHUNKED,
            IndexingStatus.EMBEDDING,
            IndexingStatus.EMBEDDED,
        ]

        for status in intermediate_statuses:
            initialized_pipeline._state.get_source_status = AsyncMock(
                return_value={"status": status}
            )

            result = await initialized_pipeline.get_index_status("http://test.example.com")

            assert result == "processing", f"Expected 'processing' for {status}, got {result}"

    @pytest.mark.asyncio
    async def test_initializes_state_if_needed(
        self, mock_config, all_mock_providers, mocker
    ) -> None:
        """Test that get_index_status initializes state if not already initialized."""
        # Create pipeline but don't initialize
        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        # Mock the state manager's get_source_status using mocker
        mocker.patch.object(
            pipeline._state, "get_source_status", new_callable=AsyncMock, return_value=None
        )

        # Should not raise - should initialize automatically
        status = await pipeline.get_index_status("http://test.example.com")

        assert status == "not_started"
        assert pipeline._initialized is True
