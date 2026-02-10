"""Supabase pgvector vector store provider using vecs client."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from audiorag.core.logging_config import get_logger
from audiorag.store._base import VectorStoreMixin

if TYPE_CHECKING:
    from vecs import Client as VecsClient  # type: ignore
    from vecs.collection import Collection  # type: ignore

logger = get_logger(__name__)


class SupabasePgVectorStore(VectorStoreMixin):
    """Supabase pgvector-based vector store using vecs client."""

    _provider_name: str = "supabase_pgvector"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )

    def __init__(
        self,
        *,
        connection_string: str,
        collection_name: str = "audiorag",
        dimension: int = 1536,
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Supabase pgvector store."""
        super().__init__(retry_config=retry_config)
        self._connection_string = connection_string
        self._collection_name = collection_name
        self._dimension = dimension
        self._client: VecsClient | None = None
        self._collection: Collection | None = None
        self._logger = self._logger.bind(
            collection_name=collection_name,
            dimension=dimension,
        )

    def _ensure_initialized(self) -> Collection:
        """Lazy initialization of vecs client and collection."""
        if self._collection is None:
            import vecs  # type: ignore

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
        """Add embeddings to the vector store."""
        operation_logger = self._logger.bind(
            operation="add",
            documents_count=len(documents),
            ids_count=len(ids),
        )
        operation_logger.debug("adding_documents")

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
            raise await self._wrap_error(e, "add")

    async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        """Query the vector store for similar embeddings."""
        operation_logger = self._logger.bind(
            operation="query",
            top_k=top_k,
        )
        operation_logger.debug("querying_documents")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _query_sync() -> list[tuple[str, float]]:
            collection = self._ensure_initialized()
            return collection.query(
                data=embedding,
                limit=top_k,
                include_metadata=True,
            )

        try:
            results = await asyncio.to_thread(_query_sync)
            return self._format_results(results)
        except Exception as e:
            raise await self._wrap_error(e, "query")

    async def delete_by_source(self, source_url: str) -> None:
        """Delete all embeddings associated with a source URL."""
        operation_logger = self._logger.bind(
            operation="delete_by_source",
            source_url=source_url,
        )
        operation_logger.debug("deleting_documents")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _delete_sync() -> int:
            collection = self._ensure_initialized()
            all_records = collection.query(
                data=[0.0] * self._dimension,
                limit=10000,
                include_metadata=True,
            )
            ids_to_delete = []
            for record in all_records:
                if len(record) > 2 and record[2].get("source_url") == source_url:
                    ids_to_delete.append(record[0])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                return len(ids_to_delete)
            return 0

        try:
            deleted_count = await asyncio.to_thread(_delete_sync)
            operation_logger.info("documents_deleted", deleted_count=deleted_count)
        except Exception as e:
            raise await self._wrap_error(e, "delete_by_source")

    def _format_results(self, results: list[tuple[str, float, dict]]) -> list[dict]:
        """Transform vecs results to expected format."""
        output = []
        for result in results:
            if len(result) >= 2:
                metadata = result[2] if len(result) > 2 else {}
                output.append(
                    {
                        "id": result[0],
                        "metadata": metadata,
                        "document": metadata.get("document", ""),
                        "distance": result[1],
                    }
                )
        return output
