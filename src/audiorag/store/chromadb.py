"""ChromaDB vector store provider."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from audiorag.core.logging_config import get_logger
from audiorag.store._base import VectorStoreMixin

if TYPE_CHECKING:
    from chromadb.api import ClientAPI  # type: ignore
    from chromadb.api.models.Collection import Collection  # type: ignore

logger = get_logger(__name__)


class ChromaDBVectorStore(VectorStoreMixin):
    """ChromaDB-based vector store."""

    vector_id_default_format: str = "sha256"
    _provider_name: str = "chromadb_vector_store"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )

    def __init__(
        self,
        *,
        persist_directory: str | Path,
        collection_name: str = "audiorag",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize ChromaDB persistent client."""
        super().__init__(retry_config=retry_config)
        self._persist_directory = Path(persist_directory)
        self._collection_name = collection_name
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None
        self._logger = self._logger.bind(
            collection_name=collection_name,
            persist_directory=str(persist_directory),
        )

    def _ensure_initialized(self) -> Collection:
        """Lazy initialization of ChromaDB client and collection."""
        if self._collection is None:
            import chromadb  # type: ignore

            self._logger.debug("initializing_chromadb")
            client: ClientAPI = chromadb.PersistentClient(path=str(self._persist_directory))
            self._client = client
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._logger.info("chromadb_initialized")
        return self._collection

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """Add embeddings to the vector store."""
        operation_logger = self._logger.bind(
            operation="add",
            documents_count=len(documents),
            ids_count=len(ids),
        )
        operation_logger.debug("adding_documents")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _add_sync() -> None:
            collection = self._ensure_initialized()
            collection.add(
                ids=ids,
                embeddings=cast(Any, embeddings),
                metadatas=cast(Any, metadatas),
                documents=documents,
            )

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
        def _query_sync() -> list[dict]:
            collection = self._ensure_initialized()
            results = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
            return self._format_results(results)

        try:
            results = await asyncio.to_thread(_query_sync)
            operation_logger.info("query_completed", results_count=len(results))
            return results
        except Exception as e:
            raise await self._wrap_error(e, "query")

    async def delete_by_source_id(self, source_id: str) -> None:
        """Delete all embeddings associated with a source ID."""
        operation_logger = self._logger.bind(
            operation="delete_by_source_id",
            source_id=source_id,
        )
        operation_logger.debug("deleting_documents")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _delete_sync() -> int:
            collection = self._ensure_initialized()
            results = collection.get(
                where={"source_id": source_id},
                include=["metadatas"],
            )
            if results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
            return 0

        try:
            deleted_count = await asyncio.to_thread(_delete_sync)
            operation_logger.info("documents_deleted", deleted_count=deleted_count)
        except Exception as e:
            raise await self._wrap_error(e, "delete_by_source_id")

    async def verify(self, ids: list[str]) -> bool:
        if not ids:
            return True

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _verify_sync() -> bool:
            collection = self._ensure_initialized()
            results = collection.get(ids=ids, include=[])
            found_ids = results.get("ids") or []
            return len(found_ids) == len(ids)

        try:
            return await asyncio.to_thread(_verify_sync)
        except Exception as e:
            raise await self._wrap_error(e, "verify")

    def _format_results(self, results: Any) -> list[dict]:
        """Transform ChromaDB results to expected format."""
        output = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                result_dict = {
                    "id": results["ids"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                }
                output.append(result_dict)
        return output
