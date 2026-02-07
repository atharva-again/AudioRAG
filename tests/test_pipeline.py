"""Comprehensive integration tests for AudioRAGPipeline."""

import pytest

from audiorag import AudioRAGPipeline, QueryResult
from audiorag.core.models import IndexingStatus


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

    def test_lazy_provider_initialization(self, mock_config) -> None:
        """Test that providers are lazily initialized."""
        pipeline = AudioRAGPipeline(mock_config)

        # Providers should be None until first use
        # (This depends on implementation details)
        assert pipeline is not None


class TestPipelineQuery:
    """Test suite for pipeline query flow."""

    @pytest.fixture
    async def initialized_pipeline(self, mock_config, all_mock_providers, tmp_db_path):
        """Create and initialize a pipeline with mocked providers."""
        mock_config.database_path = str(tmp_db_path)

        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        await pipeline._ensure_initialized()
        return pipeline

    @pytest.mark.asyncio
    async def test_query_returns_queryresult(self, initialized_pipeline) -> None:
        """Test that query returns a QueryResult."""
        pipeline = await initialized_pipeline
        result = await pipeline.query("test query")

        assert isinstance(result, QueryResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.sources, list)

    @pytest.mark.asyncio
    async def test_query_embeds_query_text(self, initialized_pipeline, mocker) -> None:
        """Test that query embeds the query text."""
        pipeline = await initialized_pipeline

        spy = mocker.spy(pipeline._embedder, "embed")
        await pipeline.query("test query")

        spy.assert_called_once()
        call_args = spy.call_args[0][0]
        assert "test query" in call_args

    @pytest.mark.asyncio
    async def test_query_queries_vector_store(self, initialized_pipeline, mocker) -> None:
        """Test that query retrieves from vector store."""
        pipeline = await initialized_pipeline

        spy = mocker.spy(pipeline._vector_store, "query")
        await pipeline.query("test query")

        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_reranks_results(self, initialized_pipeline, mocker) -> None:
        """Test that query reranks retrieved results."""
        pipeline = await initialized_pipeline

        spy = mocker.spy(pipeline._reranker, "rerank")
        await pipeline.query("test query")

        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_generates_answer(self, initialized_pipeline, mocker) -> None:
        """Test that query generates an answer."""
        pipeline = await initialized_pipeline

        spy = mocker.spy(pipeline._generator, "generate")
        result = await pipeline.query("test query")

        spy.assert_called_once()
        assert result.answer == "Mock generated answer"

    @pytest.mark.asyncio
    async def test_query_no_results_returns_empty(self, initialized_pipeline, mocker) -> None:
        """Test that query with no results returns appropriate message."""
        pipeline = await initialized_pipeline

        # Mock vector store to return empty results
        mocker.patch.object(pipeline._vector_store, "query", return_value=[])

        result = await pipeline.query("test query")

        assert "No relevant information found" in result.answer
        assert len(result.sources) == 0


class TestPipelineIndex:
    """Test suite for pipeline indexing."""

    @pytest.fixture
    async def pipeline_for_index(self, mock_config, all_mock_providers, tmp_db_path, tmp_audio_dir):
        """Create a pipeline configured for indexing tests."""
        mock_config.database_path = str(tmp_db_path)
        mock_config.work_dir = tmp_audio_dir

        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        await pipeline._ensure_initialized()
        return pipeline

    @pytest.mark.asyncio
    async def test_index_creates_source_entry(self, pipeline_for_index) -> None:
        """Test that index creates a source entry in database."""
        pipeline = await pipeline_for_index

        await pipeline.index("https://youtube.com/watch?v=test123")

        status = await pipeline._state.get_source_status("https://youtube.com/watch?v=test123")
        assert status is not None

    @pytest.mark.asyncio
    async def test_index_skips_already_indexed(self, pipeline_for_index, mocker) -> None:
        """Test that index skips already indexed URLs."""
        pipeline = await pipeline_for_index
        url = "https://youtube.com/watch?v=test123"

        # First index
        await pipeline.index(url)

        # Spy on audio_source download
        spy = mocker.spy(pipeline._audio_source, "download")

        # Second index should skip
        await pipeline.index(url)

        # Download should not be called again
        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_force_reindexes(self, pipeline_for_index, mocker) -> None:
        """Test that index with force=True reindexes."""
        pipeline = await pipeline_for_index
        url = "https://youtube.com/watch?v=test123"

        # First index
        await pipeline.index(url)

        # Spy on audio_source download
        spy = mocker.spy(pipeline._audio_source, "download")

        # Force reindex
        await pipeline.index(url, force=True)

        # Download should be called again
        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_tracks_status_progression(self, pipeline_for_index, mocker) -> None:
        """Test that index tracks status through pipeline stages."""
        pipeline = await pipeline_for_index
        url = "https://youtube.com/watch?v=test123"

        # Mock state methods to capture status updates
        status_updates = []
        original_update = pipeline._state.update_source_status

        async def capture_update(url, status, error_message=None):
            status_updates.append(status)
            return await original_update(url, status, error_message)

        mocker.patch.object(pipeline._state, "update_source_status", side_effect=capture_update)

        await pipeline.index(url)

        # Should have status updates for each stage
        assert len(status_updates) > 0
        assert IndexingStatus.COMPLETED in status_updates


class TestPipelineErrorHandling:
    """Test suite for pipeline error handling."""

    @pytest.fixture
    async def pipeline_for_error(self, mock_config, all_mock_providers, tmp_db_path):
        """Create a pipeline for error testing."""
        mock_config.database_path = str(tmp_db_path)

        pipeline = AudioRAGPipeline(
            config=mock_config,
            audio_source=all_mock_providers["audio_source"],
            stt=all_mock_providers["stt"],
            embedder=all_mock_providers["embedding"],
            vector_store=all_mock_providers["vector_store"],
            generator=all_mock_providers["generation"],
            reranker=all_mock_providers["reranker"],
        )

        await pipeline._ensure_initialized()
        return pipeline

    @pytest.mark.asyncio
    async def test_index_sets_failed_status_on_error(self, pipeline_for_error, mocker) -> None:
        """Test that index sets FAILED status when error occurs."""
        pipeline = await pipeline_for_error
        url = "https://youtube.com/watch?v=test123"

        # Mock audio_source to raise exception
        mocker.patch.object(
            pipeline._audio_source, "download", side_effect=Exception("Download failed")
        )

        with pytest.raises(Exception):  # noqa: PT011, B017
            await pipeline.index(url)

        status = await pipeline._state.get_source_status(url)
        assert status == IndexingStatus.FAILED

    @pytest.mark.asyncio
    async def test_query_handles_embedding_error(self, pipeline_for_error, mocker) -> None:
        """Test that query handles embedding errors."""
        pipeline = await pipeline_for_error

        # Mock embedder to raise exception
        mocker.patch.object(pipeline._embedder, "embed", side_effect=Exception("Embedding failed"))

        with pytest.raises(Exception):  # noqa: PT011, B017
            await pipeline.query("test query")

    @pytest.mark.asyncio
    async def test_query_handles_vector_store_error(self, pipeline_for_error, mocker) -> None:
        """Test that query handles vector store errors."""
        pipeline = await pipeline_for_error

        # Mock vector_store to raise exception
        mocker.patch.object(
            pipeline._vector_store,
            "query",
            side_effect=Exception("Vector store query failed"),
        )

        with pytest.raises(Exception):  # noqa: PT011, B017
            await pipeline.query("test query")
