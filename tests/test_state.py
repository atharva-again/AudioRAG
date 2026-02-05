"""Comprehensive async tests for StateManager."""

import hashlib
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from audiorag.state import StateManager

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
async def state_manager(tmp_db_path: Path) -> AsyncGenerator[StateManager, None]:
    """Create and initialize a StateManager instance.

    Args:
        tmp_db_path: Temporary database path from conftest.

    Returns:
        StateManager: Initialized state manager instance.
    """
    manager = StateManager(tmp_db_path)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def sample_source_path() -> str:
    """Provide a sample source path for testing.

    Returns:
        str: Sample YouTube URL.
    """
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Provide sample chunks for testing.

    Returns:
        list[dict]: List of chunk dictionaries with required fields.
    """
    return [
        {
            "chunk_index": 0,
            "start_time": 0.0,
            "end_time": 5.5,
            "text": "First chunk of transcribed text.",
            "embedding": b"\x00\x01\x02\x03",
            "metadata": {"speaker": "John", "confidence": 0.95},
        },
        {
            "chunk_index": 1,
            "start_time": 5.5,
            "end_time": 11.2,
            "text": "Second chunk with more content.",
            "embedding": b"\x04\x05\x06\x07",
            "metadata": {"speaker": "Jane", "confidence": 0.92},
        },
        {
            "chunk_index": 2,
            "start_time": 11.2,
            "end_time": 18.8,
            "text": "Third chunk continuing the discussion.",
            "embedding": None,  # Test optional embedding
            "metadata": None,  # Test optional metadata
        },
    ]


# ============================================================================
# TestStateManagerInitialization
# ============================================================================


class TestStateManagerInitialization:
    """Test database initialization and schema creation."""

    @pytest.mark.asyncio
    async def test_database_file_creation(self, tmp_db_path: Path):
        """Test that database file is created on initialization."""
        assert not tmp_db_path.exists()

        manager = StateManager(tmp_db_path)
        await manager.initialize()

        assert tmp_db_path.exists()
        assert tmp_db_path.is_file()

        await manager.close()

    @pytest.mark.asyncio
    async def test_parent_directory_creation(self, tmp_path: Path):
        """Test that parent directories are created if they don't exist."""
        nested_db_path = tmp_path / "nested" / "dirs" / "test.db"
        assert not nested_db_path.parent.exists()

        manager = StateManager(nested_db_path)
        await manager.initialize()

        assert nested_db_path.parent.exists()
        assert nested_db_path.exists()

        await manager.close()

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, state_manager: StateManager):
        """Test that WAL (Write-Ahead Logging) mode is enabled."""
        assert state_manager._db is not None
        async with state_manager._db.execute("PRAGMA journal_mode") as cursor:
            result = await cursor.fetchone()
            assert result is not None
            assert result[0].upper() == "WAL"

    @pytest.mark.asyncio
    async def test_sources_table_created(self, state_manager: StateManager):
        """Test that sources table is created with correct schema."""
        assert state_manager._db is not None
        async with state_manager._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sources'"
        ) as cursor:
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == "sources"

        # Verify columns
        assert state_manager._db is not None
        async with state_manager._db.execute("PRAGMA table_info(sources)") as cursor:
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            assert "source_id" in column_names
            assert "source_path" in column_names
            assert "status" in column_names
            assert "metadata" in column_names
            assert "created_at" in column_names
            assert "updated_at" in column_names

    @pytest.mark.asyncio
    async def test_chunks_table_created(self, state_manager: StateManager):
        """Test that chunks table is created with correct schema."""
        assert state_manager._db is not None
        async with state_manager._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ) as cursor:
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == "chunks"

        # Verify columns
        assert state_manager._db is not None
        async with state_manager._db.execute("PRAGMA table_info(chunks)") as cursor:
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            assert "chunk_id" in column_names
            assert "source_id" in column_names
            assert "chunk_index" in column_names
            assert "start_time" in column_names
            assert "end_time" in column_names
            assert "text" in column_names
            assert "embedding" in column_names
            assert "metadata" in column_names
            assert "created_at" in column_names

    @pytest.mark.asyncio
    async def test_foreign_key_constraint(self, state_manager: StateManager):
        """Test that foreign key constraint exists on chunks table."""
        assert state_manager._db is not None
        async with state_manager._db.execute("PRAGMA foreign_key_list(chunks)") as cursor:
            foreign_keys = await cursor.fetchall()
            foreign_keys_list = list(foreign_keys)
            assert len(foreign_keys_list) > 0
            # Check that source_id references sources table
            fk = foreign_keys_list[0]
            assert fk[2] == "sources"  # Referenced table
            assert fk[3] == "source_id"  # From column
            assert fk[4] == "source_id"  # To column

    @pytest.mark.asyncio
    async def test_indices_created(self, state_manager: StateManager):
        """Test that required indices are created."""
        assert state_manager._db is not None
        async with state_manager._db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ) as cursor:
            indices = await cursor.fetchall()
            index_names = [idx[0] for idx in indices]
            assert "idx_chunks_source_id" in index_names
            assert "idx_sources_status" in index_names

    @pytest.mark.asyncio
    async def test_context_manager_initialization(self, tmp_db_path: Path):
        """Test that async context manager initializes and closes properly."""
        async with StateManager(tmp_db_path) as manager:
            assert manager._db is not None
            # Verify database is functional
            async with manager._db.execute("SELECT 1") as cursor:
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == 1

        # After context exit, connection should be closed
        assert manager._db is None


# ============================================================================
# TestSourceOperations
# ============================================================================


class TestSourceOperations:
    """Test source CRUD operations."""

    @pytest.mark.asyncio
    async def test_upsert_source_new(self, state_manager: StateManager, sample_source_path: str):
        """Test inserting a new source."""
        source_id = await state_manager.upsert_source(
            source_path=sample_source_path,
            status="pending",
            metadata={"title": "Test Video", "duration": 300},
        )

        assert source_id is not None
        assert len(source_id) == 64  # SHA-256 hex length

        # Verify source was inserted
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["source_id"] == source_id
        assert status["source_path"] == sample_source_path
        assert status["status"] == "pending"
        assert status["metadata"]["title"] == "Test Video"
        assert status["metadata"]["duration"] == 300

    @pytest.mark.asyncio
    async def test_upsert_source_update_existing(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test updating an existing source."""
        # Insert initial source
        source_id_1 = await state_manager.upsert_source(
            source_path=sample_source_path,
            status="pending",
            metadata={"title": "Original Title"},
        )

        # Update the same source
        source_id_2 = await state_manager.upsert_source(
            source_path=sample_source_path,
            status="processing",
            metadata={"title": "Updated Title", "progress": 50},
        )

        # Should return same source_id
        assert source_id_1 == source_id_2

        # Verify update
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["status"] == "processing"
        assert status["metadata"]["title"] == "Updated Title"
        assert status["metadata"]["progress"] == 50

    @pytest.mark.asyncio
    async def test_upsert_source_without_metadata(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test inserting source without metadata."""
        _source_id = await state_manager.upsert_source(
            source_path=sample_source_path, status="pending"
        )

        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["metadata"] is None

    @pytest.mark.asyncio
    async def test_get_source_status_not_found(self, state_manager: StateManager):
        """Test getting status for non-existent source."""
        status = await state_manager.get_source_status(
            "https://www.youtube.com/watch?v=nonexistent"
        )
        assert status is None

    @pytest.mark.asyncio
    async def test_update_source_status(self, state_manager: StateManager, sample_source_path: str):
        """Test updating source status."""
        # Insert source
        await state_manager.upsert_source(
            source_path=sample_source_path,
            status="pending",
            metadata={"title": "Test Video"},
        )

        # Update status
        await state_manager.update_source_status(source_path=sample_source_path, status="completed")

        # Verify update
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["status"] == "completed"
        assert status["metadata"]["title"] == "Test Video"  # Metadata unchanged

    @pytest.mark.asyncio
    async def test_update_source_status_with_metadata_merge(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test updating source status with metadata merge."""
        # Insert source with initial metadata
        await state_manager.upsert_source(
            source_path=sample_source_path,
            status="pending",
            metadata={"title": "Test Video", "duration": 300},
        )

        # Update status and add new metadata
        await state_manager.update_source_status(
            source_path=sample_source_path,
            status="processing",
            metadata={"progress": 50, "stage": "transcription"},
        )

        # Verify merge
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["status"] == "processing"
        assert status["metadata"]["title"] == "Test Video"  # Original preserved
        assert status["metadata"]["duration"] == 300  # Original preserved
        assert status["metadata"]["progress"] == 50  # New added
        assert status["metadata"]["stage"] == "transcription"  # New added

    @pytest.mark.asyncio
    async def test_status_transitions(self, state_manager: StateManager, sample_source_path: str):
        """Test typical status transitions through pipeline stages."""
        # Initial state
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["status"] == "pending"

        # Downloading
        await state_manager.update_source_status(
            source_path=sample_source_path, status="downloading"
        )
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["status"] == "downloading"

        # Transcribing
        await state_manager.update_source_status(
            source_path=sample_source_path, status="transcribing"
        )
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["status"] == "transcribing"

        # Embedding
        await state_manager.update_source_status(source_path=sample_source_path, status="embedding")
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["status"] == "embedding"

        # Completed
        await state_manager.update_source_status(source_path=sample_source_path, status="completed")
        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_timestamps_updated(self, state_manager: StateManager, sample_source_path: str):
        """Test that created_at and updated_at timestamps are set correctly."""
        # Insert source
        _source_id = await state_manager.upsert_source(
            source_path=sample_source_path, status="pending"
        )

        status1 = await state_manager.get_source_status(sample_source_path)
        assert status1 is not None
        created_at = status1["created_at"]
        updated_at_1 = status1["updated_at"]

        assert created_at is not None
        assert updated_at_1 is not None
        assert created_at == updated_at_1  # Should be same on creation

        # Update source
        await state_manager.update_source_status(source_path=sample_source_path, status="completed")

        status2 = await state_manager.get_source_status(sample_source_path)
        assert status2 is not None
        updated_at_2 = status2["updated_at"]

        assert status2["created_at"] == created_at  # created_at unchanged
        assert updated_at_2 >= updated_at_1  # updated_at should be newer or same


# ============================================================================
# TestChunkOperations
# ============================================================================


class TestChunkOperations:
    """Test chunk storage and retrieval operations."""

    @pytest.mark.asyncio
    async def test_store_chunks(
        self,
        state_manager: StateManager,
        sample_source_path: str,
        sample_chunks: list[dict],
    ):
        """Test storing chunks for a source."""
        # Create source first
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")

        # Store chunks
        chunk_ids = await state_manager.store_chunks(sample_source_path, sample_chunks)

        assert len(chunk_ids) == len(sample_chunks)
        assert all(len(cid) == 64 for cid in chunk_ids)  # SHA-256 hex length

    @pytest.mark.asyncio
    async def test_get_chunks_for_source(
        self,
        state_manager: StateManager,
        sample_source_path: str,
        sample_chunks: list[dict],
    ):
        """Test retrieving chunks for a source."""
        # Create source and store chunks
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")
        await state_manager.store_chunks(sample_source_path, sample_chunks)

        # Retrieve chunks
        retrieved_chunks = await state_manager.get_chunks_for_source(sample_source_path)

        assert len(retrieved_chunks) == len(sample_chunks)

        # Verify chunk data
        for i, chunk in enumerate(retrieved_chunks):
            assert chunk["chunk_index"] == sample_chunks[i]["chunk_index"]
            assert chunk["start_time"] == sample_chunks[i]["start_time"]
            assert chunk["end_time"] == sample_chunks[i]["end_time"]
            assert chunk["text"] == sample_chunks[i]["text"]
            assert chunk["embedding"] == sample_chunks[i]["embedding"]

            # Verify metadata
            if sample_chunks[i]["metadata"]:
                assert chunk["metadata"] == sample_chunks[i]["metadata"]
            else:
                assert chunk["metadata"] is None

    @pytest.mark.asyncio
    async def test_chunks_ordered_by_index(
        self,
        state_manager: StateManager,
        sample_source_path: str,
    ):
        """Test that chunks are returned in order by chunk_index."""
        # Create source
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")

        # Store chunks in random order
        chunks = [
            {
                "chunk_index": 2,
                "start_time": 10.0,
                "end_time": 15.0,
                "text": "Third chunk",
            },
            {
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "First chunk",
            },
            {
                "chunk_index": 1,
                "start_time": 5.0,
                "end_time": 10.0,
                "text": "Second chunk",
            },
        ]
        await state_manager.store_chunks(sample_source_path, chunks)

        # Retrieve chunks
        retrieved = await state_manager.get_chunks_for_source(sample_source_path)

        # Verify order
        assert retrieved[0]["chunk_index"] == 0
        assert retrieved[1]["chunk_index"] == 1
        assert retrieved[2]["chunk_index"] == 2

    @pytest.mark.asyncio
    async def test_store_chunks_replace_existing(
        self,
        state_manager: StateManager,
        sample_source_path: str,
    ):
        """Test that storing chunks with same index replaces existing."""
        # Create source
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")

        # Store initial chunk
        initial_chunks = [
            {
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Original text",
            }
        ]
        await state_manager.store_chunks(sample_source_path, initial_chunks)

        # Store updated chunk with same index
        updated_chunks = [
            {
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Updated text",
            }
        ]
        await state_manager.store_chunks(sample_source_path, updated_chunks)

        # Retrieve chunks
        retrieved = await state_manager.get_chunks_for_source(sample_source_path)

        # Should only have one chunk with updated text
        assert len(retrieved) == 1
        assert retrieved[0]["text"] == "Updated text"

    @pytest.mark.asyncio
    async def test_get_chunks_empty_source(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test retrieving chunks for source with no chunks."""
        # Create source without chunks
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")

        # Retrieve chunks
        chunks = await state_manager.get_chunks_for_source(sample_source_path)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_chunk_id_generation(
        self,
        state_manager: StateManager,
        sample_source_path: str,
    ):
        """Test that chunk IDs are generated consistently."""
        source_id = state_manager._generate_source_id(sample_source_path)

        # Generate chunk IDs
        chunk_id_0 = state_manager._generate_chunk_id(source_id, 0)
        chunk_id_1 = state_manager._generate_chunk_id(source_id, 1)

        # Should be different
        assert chunk_id_0 != chunk_id_1

        # Should be deterministic
        chunk_id_0_again = state_manager._generate_chunk_id(source_id, 0)
        assert chunk_id_0 == chunk_id_0_again


# ============================================================================
# TestDeleteOperations
# ============================================================================


class TestDeleteOperations:
    """Test delete operations with cascading."""

    @pytest.mark.asyncio
    async def test_delete_source_with_chunks(
        self,
        state_manager: StateManager,
        sample_source_path: str,
        sample_chunks: list[dict],
    ):
        """Test that deleting source cascades to chunks."""
        # Create source and chunks
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")
        await state_manager.store_chunks(sample_source_path, sample_chunks)

        # Verify chunks exist
        chunks_before = await state_manager.get_chunks_for_source(sample_source_path)
        assert len(chunks_before) == len(sample_chunks)

        # Delete source
        deleted = await state_manager.delete_source(sample_source_path)
        assert deleted is True

        # Verify source is gone
        status = await state_manager.get_source_status(sample_source_path)
        assert status is None

        # Verify chunks are gone (cascaded)
        chunks_after = await state_manager.get_chunks_for_source(sample_source_path)
        assert chunks_after == []

    @pytest.mark.asyncio
    async def test_delete_source_without_chunks(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test deleting source without chunks."""
        # Create source without chunks
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")

        # Delete source
        deleted = await state_manager.delete_source(sample_source_path)
        assert deleted is True

        # Verify source is gone
        status = await state_manager.get_source_status(sample_source_path)
        assert status is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_source(self, state_manager: StateManager):
        """Test deleting a source that doesn't exist."""
        deleted = await state_manager.delete_source("https://www.youtube.com/watch?v=nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_cascade_delete_verification(
        self,
        state_manager: StateManager,
        sample_source_path: str,
        sample_chunks: list[dict],
    ):
        """Test that CASCADE DELETE is properly configured."""
        # Create source and chunks
        source_id = await state_manager.upsert_source(
            source_path=sample_source_path, status="pending"
        )
        await state_manager.store_chunks(sample_source_path, sample_chunks)

        # Manually delete source (bypassing delete_source method)
        assert state_manager._db is not None
        await state_manager._db.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
        await state_manager._db.commit()

        # Verify chunks were cascaded
        async with state_manager._db.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_id = ?", (source_id,)
        ) as cursor:
            count = await cursor.fetchone()
            assert count is not None
            assert count[0] == 0


# ============================================================================
# TestIdempotency
# ============================================================================


class TestIdempotency:
    """Test idempotency of operations."""

    @pytest.mark.asyncio
    async def test_same_url_produces_same_source_id(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test that same URL always produces same source_id."""
        # Insert source twice
        source_id_1 = await state_manager.upsert_source(
            source_path=sample_source_path, status="pending"
        )
        source_id_2 = await state_manager.upsert_source(
            source_path=sample_source_path, status="processing"
        )

        assert source_id_1 == source_id_2

    @pytest.mark.asyncio
    async def test_source_id_generation_deterministic(self, state_manager: StateManager):
        """Test that source ID generation is deterministic."""
        url = "https://www.youtube.com/watch?v=test123"

        source_id_1 = state_manager._generate_source_id(url)
        source_id_2 = state_manager._generate_source_id(url)

        assert source_id_1 == source_id_2
        assert len(source_id_1) == 64  # SHA-256 hex length

    @pytest.mark.asyncio
    async def test_different_urls_produce_different_ids(self, state_manager: StateManager):
        """Test that different URLs produce different source IDs."""
        url1 = "https://www.youtube.com/watch?v=test123"
        url2 = "https://www.youtube.com/watch?v=test456"

        source_id_1 = state_manager._generate_source_id(url1)
        source_id_2 = state_manager._generate_source_id(url2)

        assert source_id_1 != source_id_2

    @pytest.mark.asyncio
    async def test_sha256_hash_format(self, state_manager: StateManager):
        """Test that source IDs are valid SHA-256 hashes."""
        url = "https://www.youtube.com/watch?v=test123"
        source_id = state_manager._generate_source_id(url)

        # Verify it's a valid hex string
        assert all(c in "0123456789abcdef" for c in source_id)
        assert len(source_id) == 64

        # Verify it matches manual SHA-256 calculation
        expected = hashlib.sha256(url.encode()).hexdigest()
        assert source_id == expected

    @pytest.mark.asyncio
    async def test_multiple_upserts_no_duplicates(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test that multiple upserts don't create duplicate sources."""
        # Upsert same source multiple times
        for i in range(5):
            await state_manager.upsert_source(source_path=sample_source_path, status=f"status_{i}")

        # Count sources in database
        assert state_manager._db is not None
        async with state_manager._db.execute("SELECT COUNT(*) FROM sources") as cursor:
            count = await cursor.fetchone()
            assert count is not None
            assert count[0] == 1  # Should only have one source


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_operations_without_initialization(self, tmp_db_path: Path):
        """Test that operations fail without initialization."""
        manager = StateManager(tmp_db_path)
        # Don't call initialize()

        with pytest.raises(RuntimeError, match="Database not initialized"):
            await manager.get_source_status("https://example.com")

        with pytest.raises(RuntimeError, match="Database not initialized"):
            await manager.upsert_source("https://example.com", "pending")

        with pytest.raises(RuntimeError, match="Database not initialized"):
            await manager.update_source_status("https://example.com", "completed")

        with pytest.raises(RuntimeError, match="Database not initialized"):
            await manager.store_chunks("https://example.com", [])

        with pytest.raises(RuntimeError, match="Database not initialized"):
            await manager.get_chunks_for_source("https://example.com")

        with pytest.raises(RuntimeError, match="Database not initialized"):
            await manager.delete_source("https://example.com")

    @pytest.mark.asyncio
    async def test_close_idempotent(self, state_manager: StateManager):
        """Test that close() can be called multiple times safely."""
        await state_manager.close()
        await state_manager.close()  # Should not raise

        assert state_manager._db is None

    @pytest.mark.asyncio
    async def test_empty_metadata_handling(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test handling of empty metadata dictionaries."""
        # Insert with empty dict
        await state_manager.upsert_source(
            source_path=sample_source_path, status="pending", metadata={}
        )

        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["metadata"] == {}

    @pytest.mark.asyncio
    async def test_special_characters_in_text(
        self, state_manager: StateManager, sample_source_path: str
    ):
        """Test handling of special characters in chunk text."""
        await state_manager.upsert_source(source_path=sample_source_path, status="pending")

        chunks = [
            {
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Text with 'quotes', \"double quotes\", and\nnewlines\t\ttabs",
            }
        ]
        await state_manager.store_chunks(sample_source_path, chunks)

        retrieved = await state_manager.get_chunks_for_source(sample_source_path)
        assert retrieved[0]["text"] == chunks[0]["text"]

    @pytest.mark.asyncio
    async def test_unicode_in_metadata(self, state_manager: StateManager, sample_source_path: str):
        """Test handling of Unicode characters in metadata."""
        metadata = {
            "title": "Test 测试 テスト тест",
            "emoji": "🎵🎶🎤",
            "special": "café résumé naïve",
        }

        await state_manager.upsert_source(
            source_path=sample_source_path, status="pending", metadata=metadata
        )

        status = await state_manager.get_source_status(sample_source_path)
        assert status is not None
        assert status["metadata"] == metadata
