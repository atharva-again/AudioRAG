"""State management for AudioRAG using async SQLite."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite


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

        # Create sources table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Create chunks table
        await self._db.execute("""
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
        """)

        # Create indices for efficient queries
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_source_id
            ON chunks(source_id)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sources_status
            ON sources(status)
        """)

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

    async def get_source_status(self, source_path: str) -> dict[str, Any] | None:
        """Get status information for a source.

        Args:
            source_path: Path to source file

        Returns:
            Dictionary with source information or None if not found
        """
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        source_id = self._generate_source_id(source_path)

        async with self._db.execute(
            "SELECT source_id, source_path, status, metadata, created_at, updated_at "
            "FROM sources WHERE source_id = ?",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return {
            "source_id": row[0],
            "source_path": row[1],
            "status": row[2],
            "metadata": json.loads(row[3]) if row[3] else None,
            "created_at": row[4],
            "updated_at": row[5],
        }

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
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        source_id = self._generate_source_id(source_path)
        now = self._now_iso8601()
        metadata_json = json.dumps(metadata) if metadata is not None else None

        # Check if source exists
        existing = await self.get_source_status(source_path)

        if existing:
            # Update existing source
            await self._db.execute(
                "UPDATE sources SET status = ?, metadata = ?, updated_at = ? WHERE source_id = ?",
                (status, metadata_json, now, source_id),
            )
        else:
            # Insert new source
            await self._db.execute(
                "INSERT INTO sources (source_id, source_path, status, "
                "metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, source_path, status, metadata_json, now, now),
            )

        await self._db.commit()
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
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        source_id = self._generate_source_id(source_path)
        now = self._now_iso8601()

        # Get existing metadata if we need to merge
        if metadata is not None:
            existing = await self.get_source_status(source_path)
            if existing and existing.get("metadata"):
                merged_metadata = {**existing["metadata"], **metadata}
            else:
                merged_metadata = metadata
            metadata_json = json.dumps(merged_metadata)

            await self._db.execute(
                "UPDATE sources SET status = ?, metadata = ?, updated_at = ? WHERE source_id = ?",
                (status, metadata_json, now, source_id),
            )
        else:
            await self._db.execute(
                "UPDATE sources SET status = ?, updated_at = ? WHERE source_id = ?",
                (status, now, source_id),
            )

        await self._db.commit()

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
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        source_id = self._generate_source_id(source_path)
        now = self._now_iso8601()
        chunk_ids = []

        for chunk in chunks:
            chunk_id = self._generate_chunk_id(source_id, chunk["chunk_index"])
            chunk_ids.append(chunk_id)

            metadata_json = json.dumps(chunk.get("metadata")) if chunk.get("metadata") else None

            await self._db.execute(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, source_id, chunk_index, start_time, end_time, "
                "text, embedding, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

        await self._db.commit()
        return chunk_ids

    async def get_chunks_for_source(self, source_path: str) -> list[dict[str, Any]]:
        """Retrieve all chunks for a source.

        Args:
            source_path: Path to source file

        Returns:
            List of chunk dictionaries
        """
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        source_id = self._generate_source_id(source_path)

        async with self._db.execute(
            "SELECT chunk_id, chunk_index, start_time, end_time, "
            "text, embedding, metadata, created_at "
            "FROM chunks WHERE source_id = ? ORDER BY chunk_index",
            (source_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        chunks = []
        for row in rows:
            chunks.append(
                {
                    "chunk_id": row[0],
                    "chunk_index": row[1],
                    "start_time": row[2],
                    "end_time": row[3],
                    "text": row[4],
                    "embedding": row[5],
                    "metadata": json.loads(row[6]) if row[6] else None,
                    "created_at": row[7],
                }
            )

        return chunks

    async def delete_source(self, source_path: str) -> bool:
        """Delete a source and all its chunks.

        Args:
            source_path: Path to source file

        Returns:
            True if source was deleted, False if not found
        """
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        source_id = self._generate_source_id(source_path)

        # Check if source exists
        existing = await self.get_source_status(source_path)
        if not existing:
            return False

        # Delete chunks (CASCADE should handle this, but explicit is better)
        await self._db.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))

        # Delete source
        await self._db.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))

        await self._db.commit()
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
