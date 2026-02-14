"""State management for AudioRAG using async SQLite."""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from audiorag.core.exceptions import StateError
from audiorag.core.logging_config import get_logger

logger = get_logger(__name__)

SOURCES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        source_path TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL,
        metadata TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
"""

CHUNKS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL NOT NULL,
        text TEXT NOT NULL,
        embedding BLOB,
        metadata TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
        UNIQUE(source_id, chunk_index)
    )
"""

TRANSCRIPTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS transcripts (
        source_id TEXT NOT NULL,
        part_index INTEGER NOT NULL,
        raw_response TEXT,
        segments TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (source_id, part_index)
    )
"""

CREATE_INDEX_SQL: tuple[str, ...] = (
    """
    CREATE INDEX IF NOT EXISTS idx_chunks_source_id
    ON chunks(source_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sources_status
    ON sources(status)
    """,
)

GET_SOURCE_SQL = (
    "SELECT source_id, source_path, status, metadata, created_at, updated_at "
    "FROM sources WHERE source_id = ?"
)
UPDATE_SOURCE_WITH_METADATA_SQL = (
    "UPDATE sources SET status = ?, metadata = ?, updated_at = ? WHERE source_id = ?"
)
UPDATE_SOURCE_STATUS_SQL = "UPDATE sources SET status = ?, updated_at = ? WHERE source_id = ?"
INSERT_SOURCE_SQL = (
    "INSERT INTO sources (source_id, source_path, status, metadata, created_at, updated_at) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)
INSERT_OR_REPLACE_CHUNK_SQL = (
    "INSERT OR REPLACE INTO chunks "
    "(chunk_id, source_id, chunk_index, start_time, end_time, "
    "text, embedding, metadata, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
GET_CHUNKS_FOR_SOURCE_SQL = (
    "SELECT chunk_id, chunk_index, start_time, end_time, text, embedding, metadata, created_at "
    "FROM chunks WHERE source_id = ? ORDER BY chunk_index"
)
DELETE_CHUNKS_FOR_SOURCE_SQL = "DELETE FROM chunks WHERE source_id = ?"
DELETE_SOURCE_SQL = "DELETE FROM sources WHERE source_id = ?"

# Transcript SQL queries
INSERT_OR_REPLACE_TRANSCRIPT_SQL = (
    "INSERT OR REPLACE INTO transcripts "
    "(source_id, part_index, raw_response, segments, created_at) "
    "VALUES (?, ?, ?, ?, ?)"
)
GET_TRANSCRIPTS_FOR_SOURCE_SQL = (
    "SELECT part_index, raw_response, segments, created_at "
    "FROM transcripts WHERE source_id = ? ORDER BY part_index"
)
GET_TRANSCRIPT_FOR_PART_SQL = (
    "SELECT part_index, raw_response, segments, created_at "
    "FROM transcripts WHERE source_id = ? AND part_index = ?"
)
GET_TRANSCRIBED_PART_INDICES_SQL = (
    "SELECT part_index FROM transcripts WHERE source_id = ? ORDER BY part_index"
)
DELETE_TRANSCRIPTS_FOR_SOURCE_SQL = "DELETE FROM transcripts WHERE source_id = ?"


class StateManager:
    """Manages persistent state for audio sources and chunks using SQLite."""

    def __init__(self, db_path: str | Path):
        """Initialize StateManager with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database with WAL mode and create schema."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        self._db = await aiosqlite.connect(str(self.db_path))

        # Enable foreign keys for CASCADE DELETE support
        await self._db.execute("PRAGMA foreign_keys = ON")

        # Enable WAL mode for better concurrency
        await self._db.execute("PRAGMA journal_mode=WAL")

        await self._db.execute(SOURCES_TABLE_SQL)
        await self._db.execute(CHUNKS_TABLE_SQL)
        await self._db.execute(TRANSCRIPTS_TABLE_SQL)
        for statement in CREATE_INDEX_SQL:
            await self._db.execute(statement)

        await self._db.commit()

    def _generate_source_id(self, source_path: str) -> str:
        """Generate SHA-256 hash ID for source path.

        Args:
            source_path: Path to source file

        Returns:
            Hexadecimal SHA-256 hash
        """
        return hashlib.sha256(source_path.encode()).hexdigest()

    def _generate_chunk_id(self, source_id: str, chunk_index: int) -> str:
        """Generate SHA-256 hash ID for chunk.

        Args:
            source_id: Source identifier
            chunk_index: Index of chunk within source

        Returns:
            Hexadecimal SHA-256 hash
        """
        data = f"{source_id}:{chunk_index}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _now_iso8601(self) -> str:
        """Get current UTC timestamp in ISO 8601 format.

        Returns:
            ISO 8601 formatted timestamp
        """
        return datetime.now(UTC).isoformat()

    def _ensure_initialized(self) -> None:
        """Ensure the database connection is initialized."""
        if not self._db:
            logger.error("state_manager_not_initialized")
            raise StateError("Database not initialized. Call initialize() first.")

    def _get_db(self) -> aiosqlite.Connection:
        """Return the active database connection.

        Raises:
            StateError: If the database is not initialized.
        """
        self._ensure_initialized()
        if not self._db:
            logger.error("state_manager_not_initialized")
            raise StateError("Database not initialized. Call initialize() first.")
        return self._db

    def _json_dumps(self, data: dict[str, Any] | None) -> str | None:
        if data is None:
            return None
        return json.dumps(data)

    def _json_loads(self, data: str | None) -> dict[str, Any] | None:
        if data is None:
            return None
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else None

    def _source_from_row(self, row: Sequence[Any]) -> dict[str, Any]:
        return {
            "source_id": row[0],
            "source_path": row[1],
            "status": row[2],
            "metadata": self._json_loads(row[3]),
            "created_at": row[4],
            "updated_at": row[5],
        }

    def _chunk_from_row(self, row: Sequence[Any]) -> dict[str, Any]:
        return {
            "chunk_id": row[0],
            "chunk_index": row[1],
            "start_time": row[2],
            "end_time": row[3],
            "text": row[4],
            "embedding": row[5],
            "metadata": self._json_loads(row[6]),
            "created_at": row[7],
        }

    def _transcript_from_row(self, row: Sequence[Any]) -> dict[str, Any]:
        segments_data = json.loads(row[2]) if row[2] else []
        return {
            "part_index": row[0],
            "raw_response": self._json_loads(row[1]),
            "segments": segments_data,
            "created_at": row[3],
        }

    async def get_source_status(self, source_path: str) -> dict[str, Any] | None:
        """Get status information for a source.

        Args:
            source_path: Path to source file

        Returns:
            Dictionary with source information or None if not found
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)

        async with db.execute(
            GET_SOURCE_SQL,
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return self._source_from_row(row)

    async def upsert_source(
        self, source_path: str, status: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Insert or update a source record.

        Args:
            source_path: Path to source file
            status: Processing status (e.g., 'pending', 'processing', 'completed', 'failed')
            metadata: Optional metadata dictionary

        Returns:
            Source ID
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)
        now = self._now_iso8601()
        metadata_json = self._json_dumps(metadata)

        # Check if source exists
        existing = await self.get_source_status(source_path)

        if existing:
            # Update existing source
            await db.execute(
                UPDATE_SOURCE_WITH_METADATA_SQL,
                (status, metadata_json, now, source_id),
            )
        else:
            # Insert new source
            await db.execute(
                INSERT_SOURCE_SQL,
                (source_id, source_path, status, metadata_json, now, now),
            )

        await db.commit()
        return source_id

    async def update_source_status(
        self, source_path: str, status: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Update status and optionally metadata for a source.

        Args:
            source_path: Path to source file
            status: New processing status
            metadata: Optional metadata to merge with existing
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)
        now = self._now_iso8601()

        # Get existing metadata if we need to merge
        if metadata is not None:
            existing = await self.get_source_status(source_path)
            if existing and existing.get("metadata"):
                merged_metadata = {**existing["metadata"], **metadata}
            else:
                merged_metadata = metadata
            metadata_json = self._json_dumps(merged_metadata)

            await db.execute(
                UPDATE_SOURCE_WITH_METADATA_SQL,
                (status, metadata_json, now, source_id),
            )
        else:
            await db.execute(
                UPDATE_SOURCE_STATUS_SQL,
                (status, now, source_id),
            )

        await db.commit()

    async def store_chunks(self, source_path: str, chunks: list[dict[str, Any]]) -> list[str]:
        """Store chunks for a source.

        Args:
            source_path: Path to source file
            chunks: List of chunk dictionaries with keys:
                - chunk_index: Integer index
                - start_time: Start time in seconds
                - end_time: End time in seconds
                - text: Transcribed text
                - embedding: Optional embedding bytes
                - metadata: Optional metadata dictionary

        Returns:
            List of chunk IDs
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)
        now = self._now_iso8601()
        chunk_ids = []

        for chunk in chunks:
            chunk_id = self._generate_chunk_id(source_id, chunk["chunk_index"])
            chunk_ids.append(chunk_id)

            metadata_json = self._json_dumps(chunk.get("metadata"))

            await db.execute(
                INSERT_OR_REPLACE_CHUNK_SQL,
                (
                    chunk_id,
                    source_id,
                    chunk["chunk_index"],
                    chunk["start_time"],
                    chunk["end_time"],
                    chunk["text"],
                    chunk.get("embedding"),
                    metadata_json,
                    now,
                ),
            )

        await db.commit()
        return chunk_ids

    async def get_chunks_for_source(self, source_path: str) -> list[dict[str, Any]]:
        """Retrieve all chunks for a source.

        Args:
            source_path: Path to source file

        Returns:
            List of chunk dictionaries
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)

        async with db.execute(
            GET_CHUNKS_FOR_SOURCE_SQL,
            (source_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [self._chunk_from_row(row) for row in rows]

    async def store_transcript(
        self,
        source_path: str,
        part_index: int,
        segments: list[dict[str, Any]],
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        """Store transcription result for a single audio part.

        Args:
            source_path: Path to source file
            part_index: Index of the audio part (0-based)
            segments: List of segment dictionaries with start_time, end_time, text
            raw_response: Optional raw STT provider response (for re-chunking)
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)
        now = self._now_iso8601()
        segments_json = json.dumps(segments)
        raw_response_json = self._json_dumps(raw_response)

        await db.execute(
            INSERT_OR_REPLACE_TRANSCRIPT_SQL,
            (source_id, part_index, raw_response_json, segments_json, now),
        )
        await db.commit()

    async def get_transcripts(self, source_path: str) -> dict[int, dict[str, Any]]:
        """Retrieve all transcripts for a source.

        Args:
            source_path: Path to source file

        Returns:
            Dictionary mapping part_index to transcript data
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)

        async with db.execute(
            GET_TRANSCRIPTS_FOR_SOURCE_SQL,
            (source_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return {row[0]: self._transcript_from_row(row) for row in rows}

    async def get_transcribed_part_indices(self, source_path: str) -> set[int]:
        """Get set of already transcribed part indices for a source.

        Args:
            source_path: Path to source file

        Returns:
            Set of part indices that already have transcripts
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)

        async with db.execute(
            GET_TRANSCRIBED_PART_INDICES_SQL,
            (source_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return {row[0] for row in rows}

    async def delete_source(self, source_path: str) -> bool:
        """Delete a source and all its chunks and transcripts.

        Args:
            source_path: Path to source file

        Returns:
            True if source was deleted, False if not found
        """
        db = self._get_db()

        source_id = self._generate_source_id(source_path)

        existing = await self.get_source_status(source_path)
        if not existing:
            return False

        await db.execute(DELETE_CHUNKS_FOR_SOURCE_SQL, (source_id,))
        await db.execute(DELETE_TRANSCRIPTS_FOR_SOURCE_SQL, (source_id,))
        await db.execute(DELETE_SOURCE_SQL, (source_id,))

        await db.commit()
        return True

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
