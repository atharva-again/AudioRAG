"""Supabase pgvector vector store implementation using vecs client."""

from __future__ import annotations

import asyncio
from typing import Any

import vecs  # type: ignore
from vecs import Client as VecsClient  # type: ignore
from vecs.collection import Collection  # type: ignore

from audiorag.logging_config import get_logger
from audiorag.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class SupabasePgVectorStore:
    """Supabase pgvector-based vector store implementation using vecs client.

    This provider uses Supabase's vecs Python client to store and query
    vectors in PostgreSQL with the pgvector extension.

    Requires:
        - vecs package: pip install vecs
        - Supabase project with pgvector extension enabled
        - Connection string from Supabase dashboard
    """

    def __init__(
        self,
        connection_string: str,
        collection_name: str = "audiorag",
        dimension: int = 1536,
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize Supabase pgvector store.

        Args:
            connection_string: PostgreSQL connection string from Supabase.
                Format: postgresql://user:password@host:port/database
            collection_name: Name of the collection (table) to use.
            dimension: Dimension of embedding vectors. Defaults to 1536
                (OpenAI text-embedding-3-small).
            retry_config: Retry configuration. Uses default if not provided.
        """
        self._connection_string = connection_string
        self._collection_name = collection_name
        self._dimension = dimension
        self._client: VecsClient | None = None
        self._collection: Collection | None = None
        self._logger = logger.bind(
            provider="supabase_pgvector",
            collection_name=collection_name,
            dimension=dimension,
        )
        self._retry_config = retry_config or RetryConfig()

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Supabase operations."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(ConnectionError, TimeoutError, RuntimeError),
        )

    def _ensure_initialized(self) -> Collection:
        """Lazy initialization of vecs client and collection."""
        if self._collection is None:
            self._logger.debug("initializing_supabase_pgvector")
            self._client = vecs.create_client(self._connection_string)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                dimension=self._dimension,
            )
            self._logger.info("supabase_pgvector_initialized")
        return self._collection

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        """Add embeddings to the vector store.

        Args:
            ids: List of unique identifiers for each embedding.
            embeddings: List of embedding vectors.
            metadatas: List of metadata dictionaries.
            documents: List of document texts.
        """
        operation_logger = self._logger.bind(
            operation="add",
            documents_count=len(documents),
            ids_count=len(ids),
        )
        operation_logger.debug("adding_documents")

        # Prepare records for vecs upsert
        # vecs expects: list of (id, vector, metadata) tuples
        records = [
            (ids[i], embeddings[i], {**metadatas[i], "document": documents[i]})
            for i in range(len(ids))
        ]

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _add_sync() -> None:
            collection = self._ensure_initialized()
            collection.upsert(records=records)

        try:
            await asyncio.to_thread(_add_sync)
            operation_logger.info("documents_added")
        except Exception as e:
            operation_logger.error(
                "add_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        """Query the vector store for similar embeddings.

        Args:
            embedding: Query embedding vector.
            top_k: Number of top results to return.

        Returns:
            List of dictionaries containing query results with metadata.
        """
        operation_logger = self._logger.bind(
            operation="query",
            top_k=top_k,
        )
        operation_logger.debug("querying_documents")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _query_sync() -> list[tuple[str, float]]:
            collection = self._ensure_initialized()
            # vecs.query returns list of (id, distance) tuples
            return collection.query(
                data=embedding,
                limit=top_k,
                include_metadata=True,
            )

        try:
            results = await asyncio.to_thread(_query_sync)

            # Transform vecs results to expected format
            output = []
            for result in results:
                if len(result) >= 2:
                    id_val = result[0]
                    distance = result[1]
                    # Metadata is included in the result tuple if include_metadata=True
                    metadata = result[2] if len(result) > 2 else {}

                    output.append(
                        {
                            "id": id_val,
                            "metadata": metadata,
                            "document": metadata.get("document", ""),
                            "distance": distance,
                        }
                    )

            operation_logger.info("query_completed", results_count=len(output))
            return output

        except Exception as e:
            operation_logger.error(
                "query_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def delete_by_source(self, source_url: str) -> None:
        """Delete all embeddings associated with a source URL.

        Note: vecs doesn't support metadata filtering in delete operations.
        This implementation queries for all records and deletes matching ones.

        Args:
            source_url: Source URL to filter deletions.
        """
        operation_logger = self._logger.bind(
            operation="delete_by_source",
            source_url=source_url,
        )
        operation_logger.debug("deleting_documents")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _delete_sync() -> int:
            collection = self._ensure_initialized()
            # Query all records to find matching ones
            # Note: This is not efficient for large collections
            # Consider using a metadata index or separate tracking table
            all_records = collection.query(
                data=[0.0] * self._dimension,  # Dummy query vector
                limit=10000,  # Adjust based on expected collection size
                include_metadata=True,
            )

            # Find IDs matching the source_url
            ids_to_delete = []
            for record in all_records:
                if len(record) > 2:
                    metadata = record[2]
                    if metadata.get("source_url") == source_url:
                        ids_to_delete.append(record[0])

            # Delete matching records
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                return len(ids_to_delete)
            return 0

        try:
            deleted_count = await asyncio.to_thread(_delete_sync)
            operation_logger.info("documents_deleted", deleted_count=deleted_count)
        except Exception as e:
            operation_logger.error(
                "delete_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
