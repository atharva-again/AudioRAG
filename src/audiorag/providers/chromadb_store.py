"""ChromaDB vector store implementation."""

import asyncio
from pathlib import Path
from typing import Any

import chromadb  # type: ignore
from chromadb.api import ClientAPI  # type: ignore
from chromadb.api.models.Collection import Collection  # type: ignore

from audiorag.logging_config import get_logger
from audiorag.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class ChromaDBVectorStore:
    """ChromaDB-based vector store implementation satisfying VectorStoreProvider protocol."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "audiorag",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize ChromaDB persistent client.

        Args:
            persist_directory: Directory path for ChromaDB persistence
            collection_name: Name of the collection to use
            retry_config: Retry configuration. Uses default if not provided.
        """
        self._persist_directory = Path(persist_directory)
        self._collection_name = collection_name
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None
        self._logger = logger.bind(
            provider="chromadb_vector_store",
            collection_name=collection_name,
            persist_directory=str(persist_directory),
        )
        self._retry_config = retry_config or RetryConfig()

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for ChromaDB operations."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(ConnectionError, TimeoutError, RuntimeError),
        )

    def _ensure_initialized(self) -> Collection:
        """Lazy initialization of ChromaDB client and collection."""
        if self._collection is None:
            self._logger.debug("initializing_chromadb")
            self._client = chromadb.PersistentClient(path=str(self._persist_directory))
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
            self._logger.info("chromadb_initialized")
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
            ids: List of unique identifiers for each embedding
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            documents: List of document texts
        """
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
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )

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
            embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing query results with metadata
        """
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

            # Transform ChromaDB results to expected format
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

        try:
            results = await asyncio.to_thread(_query_sync)
            operation_logger.info("query_completed", results_count=len(results))
            return results
        except Exception as e:
            operation_logger.error(
                "query_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def delete_by_source(self, source_url: str) -> None:
        """Delete all embeddings associated with a source URL.

        Args:
            source_url: Source URL to filter deletions
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
            # Query for all items with matching source_url in metadata
            results = collection.get(
                where={"source_url": source_url},
                include=["metadatas"],
            )

            # Delete if any results found
            if results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
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
