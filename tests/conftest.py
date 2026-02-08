"""Shared pytest fixtures for AudioRAG test suite."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from audiorag.core.models import AudioFile, ChunkMetadata, TranscriptionSegment

# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for audio files.

    Returns:
        Path: Temporary directory path for audio test files.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database file path.

    Returns:
        Path: Path to a temporary SQLite database file.
    """
    return tmp_path / "test.db"


@pytest.fixture
def tmp_vector_store_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for vector store data.

    Returns:
        Path: Temporary directory path for vector store files.
    """
    store_dir = tmp_path / "vector_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for cache files.

    Returns:
        Path: Temporary directory path for cache files.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_transcription_segment() -> TranscriptionSegment:
    """Create a sample transcription segment.

    Returns:
        TranscriptionSegment: A sample segment with typical values.
    """
    return TranscriptionSegment(
        start_time=0.0,
        end_time=5.5,
        text="This is a sample transcription segment for testing purposes.",
    )


@pytest.fixture
def sample_transcription_segments() -> list[TranscriptionSegment]:
    """Create multiple sample transcription segments.

    Returns:
        list[TranscriptionSegment]: List of sample segments covering 30 seconds.
    """
    return [
        TranscriptionSegment(
            start_time=0.0,
            end_time=5.5,
            text="This is the first segment of the transcription.",
        ),
        TranscriptionSegment(
            start_time=5.5,
            end_time=11.2,
            text="This is the second segment with more content.",
        ),
        TranscriptionSegment(
            start_time=11.2,
            end_time=18.8,
            text="The third segment continues the discussion.",
        ),
        TranscriptionSegment(
            start_time=18.8,
            end_time=25.0,
            text="And here is the fourth and final segment.",
        ),
    ]


@pytest.fixture
def sample_chunk_metadata() -> ChunkMetadata:
    """Create a sample chunk metadata object.

    Returns:
        ChunkMetadata: A sample chunk with typical metadata.
    """
    return ChunkMetadata(
        start_time=0.0,
        end_time=5.5,
        text="This is a sample text chunk for testing.",
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Sample Video Title",
    )


@pytest.fixture
def sample_chunks_metadata() -> list[ChunkMetadata]:
    """Create multiple sample chunk metadata objects.

    Returns:
        list[ChunkMetadata]: List of sample chunks with different timestamps.
    """
    return [
        ChunkMetadata(
            start_time=0.0,
            end_time=5.5,
            text="First chunk of text content.",
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Sample Video Title",
        ),
        ChunkMetadata(
            start_time=5.5,
            end_time=11.2,
            text="Second chunk with more information.",
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Sample Video Title",
        ),
        ChunkMetadata(
            start_time=11.2,
            end_time=18.8,
            text="Third chunk continuing the narrative.",
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Sample Video Title",
        ),
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create sample embedding vectors.

    Returns:
        list[list[float]]: List of sample embeddings (3 vectors of dimension 384).
    """
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5] * 77,  # 385 dimensions (simulating real embeddings)
        [0.2, 0.3, 0.4, 0.5, 0.6] * 77,
        [0.3, 0.4, 0.5, 0.6, 0.7] * 77,
    ]


@pytest.fixture
def sample_query_embedding() -> list[float]:
    """Create a sample query embedding vector.

    Returns:
        list[float]: A single embedding vector for query testing.
    """
    return [0.15, 0.25, 0.35, 0.45, 0.55] * 77  # 385 dimensions


# ============================================================================
# Mock Provider Fixtures
# ============================================================================


@pytest.fixture
def mock_stt_provider() -> AsyncMock:
    """Create a mock STT (Speech-to-Text) provider.

    Returns:
        AsyncMock: Mock provider with transcribe method.
    """
    mock = AsyncMock()
    mock.transcribe = AsyncMock(
        return_value=[
            TranscriptionSegment(
                start_time=0.0,
                end_time=5.5,
                text="Mocked transcription segment one.",
            ),
            TranscriptionSegment(
                start_time=5.5,
                end_time=10.0,
                text="Mocked transcription segment two.",
            ),
        ]
    )
    return mock


@pytest.fixture
def mock_embedding_provider() -> AsyncMock:
    """Create a mock Embedding provider.

    Returns:
        AsyncMock: Mock provider with embed method.
    """
    mock = AsyncMock()
    mock.embed = AsyncMock(
        return_value=[
            [0.1, 0.2, 0.3, 0.4, 0.5] * 77,
        ]
    )
    return mock


@pytest.fixture
def mock_vector_store_provider() -> AsyncMock:
    """Create a mock Vector Store provider.

    Returns:
        AsyncMock: Mock provider with add, query, and delete_by_source methods.
    """
    mock = AsyncMock()
    mock.add = AsyncMock(return_value=None)
    mock.query = AsyncMock(
        return_value=[
            {
                "id": "chunk_1",
                "document": "First matching chunk",
                "metadata": {
                    "start_time": 0.0,
                    "end_time": 5.5,
                    "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "title": "Sample Video",
                },
                "distance": 0.1,
            },
            {
                "id": "chunk_2",
                "document": "Second matching chunk",
                "metadata": {
                    "start_time": 5.5,
                    "end_time": 11.2,
                    "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "title": "Sample Video",
                },
                "distance": 0.2,
            },
        ]
    )
    mock.delete_by_source = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_reranker_provider() -> AsyncMock:
    """Create a mock Reranker provider.

    Returns:
        AsyncMock: Mock provider with rerank method.
    """
    mock = AsyncMock()
    mock.rerank = AsyncMock(
        return_value=[
            (0, 0.95),  # (index, score) - only 2 results in vector store
            (1, 0.87),
        ]
    )
    return mock


@pytest.fixture
def mock_generation_provider() -> AsyncMock:
    """Create a mock Generation provider.

    Returns:
        AsyncMock: Mock provider with generate method.
    """
    mock = AsyncMock()
    mock.generate = AsyncMock(
        return_value="This is a mocked generated response based on the provided context."
    )
    return mock


@pytest.fixture
def mock_audio_source_provider(tmp_path) -> AsyncMock:
    """Create a mock Audio Source provider.

    Returns:
        AsyncMock: Mock provider with download method that creates a real audio file.
    """
    mock = AsyncMock()
    audio_path = tmp_path / "mock_audio.mp3"
    # Create a small dummy file so the splitter doesn't fail
    audio_path.write_bytes(b"dummy audio content")

    mock.download = AsyncMock(
        return_value=AudioFile(
            path=audio_path,
            source_url="https://youtube.com/watch?v=test123",
            title="Test Video Title",
            duration=120.0,
        )
    )
    return mock


# ============================================================================
# Composite Provider Fixtures
# ============================================================================


@pytest.fixture
def all_mock_providers(
    mock_stt_provider: AsyncMock,
    mock_embedding_provider: AsyncMock,
    mock_vector_store_provider: AsyncMock,
    mock_reranker_provider: AsyncMock,
    mock_generation_provider: AsyncMock,
    mock_audio_source_provider: AsyncMock,
) -> dict:
    """Create a dictionary of all mock providers.

    Returns:
        dict: Dictionary with keys for each provider type.
    """
    return {
        "stt": mock_stt_provider,
        "embedding": mock_embedding_provider,
        "vector_store": mock_vector_store_provider,
        "reranker": mock_reranker_provider,
        "generation": mock_generation_provider,
        "audio_source": mock_audio_source_provider,
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def mock_config(tmp_db_path: Path, tmp_vector_store_dir: Path):
    """Create a mock configuration object that behaves like AudioRAGConfig.

    Args:
        tmp_db_path: Temporary database path.
        tmp_vector_store_dir: Temporary vector store directory.

    Returns:
        MagicMock: Mock configuration with typical settings matching AudioRAGConfig.
    """
    config = MagicMock()
    config.database_path = str(tmp_db_path)
    config.stt_provider = "openai"
    config.embedding_provider = "openai"
    config.vector_store_provider = "chromadb"
    config.generation_provider = "openai"
    config.reranker_provider = "cohere"
    config.openai_api_key = ""
    config.chunk_duration_seconds = 30
    config.stt_language = None
    config.retrieval_top_k = 10
    config.rerank_top_n = 3
    config.audio_format = "mp3"
    config.audio_split_max_size_mb = 24
    config.cleanup_audio = True
    config.work_dir = None
    config.log_level = "INFO"
    config.log_format = "colored"
    config.log_timestamps = True
    config.chromadb_persist_directory = str(tmp_vector_store_dir)
    config.chromadb_collection_name = "audiorag"
    config.youtube_download_archive = None
    config.youtube_concurrent_fragments = 3
    config.youtube_skip_after_errors = 3
    config.youtube_cookie_file = None
    config.youtube_po_token = None
    config.youtube_impersonate = None
    config.youtube_player_clients = ["tv", "web", "mweb"]
    config.js_runtime = None
    config.retry_max_attempts = 3
    config.retry_min_wait_seconds = 4.0
    config.retry_max_wait_seconds = 60.0
    config.retry_exponential_multiplier = 1.0

    # Model getter methods
    config.get_stt_model = MagicMock(return_value="whisper-1")
    config.get_embedding_model = MagicMock(return_value="text-embedding-3-small")
    config.get_generation_model = MagicMock(return_value="gpt-4o-mini")

    return config


# ============================================================================
# Async Context Manager Fixtures
# ============================================================================


@pytest.fixture
async def async_context_manager() -> AsyncGenerator[None, None]:
    """Provide an async context for tests that need async setup/teardown.

    Yields:
        None: Allows test to run with async context.
    """
    # Setup
    yield
    # Teardown
    await asyncio.sleep(0)  # Allow pending tasks to complete


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger for testing logging behavior.

    Returns:
        MagicMock: Mock logger with standard logging methods.
    """
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    return logger


@pytest.fixture
def sample_metadata_dict() -> dict:
    """Create a sample metadata dictionary.

    Returns:
        dict: Dictionary with typical chunk metadata.
    """
    return {
        "start_time": 0.0,
        "end_time": 5.5,
        "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Sample Video Title",
    }


@pytest.fixture
def sample_metadata_dicts() -> list[dict]:
    """Create multiple sample metadata dictionaries.

    Returns:
        list[dict]: List of metadata dictionaries for multiple chunks.
    """
    return [
        {
            "start_time": 0.0,
            "end_time": 5.5,
            "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Sample Video Title",
        },
        {
            "start_time": 5.5,
            "end_time": 11.2,
            "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Sample Video Title",
        },
        {
            "start_time": 11.2,
            "end_time": 18.8,
            "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Sample Video Title",
        },
    ]
