"""Pinecone vector store implementation."""

from __future__ import annotations

from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.core.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class PineconeVectorStore:
    """Pinecone-based vector store implementation satisfying VectorStoreProvider protocol.

    Pinecone is a leading managed vector database in 2026, offering
    high performance, scalability, and hybrid search capabilities.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str = "audiorag",
        namespace: str = "default",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize Pinecone client.

        Args:
            api_key: Pinecone API key. If None, uses PINECONE_API_KEY environment variable.
            index_name: Name of the Pinecone index to use
            namespace: Namespace for the vectors
            retry_config: Retry configuration. Uses default if not provided.
        """
        # Lazy import to avoid ModuleNotFoundError when optional dep not installed
        from pinecone import Pinecone  # noqa: PLC0415 # type: ignore

        self._pc = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._namespace = namespace
        self._index = None
        self._logger = logger.bind(
            provider="pinecone_vector_store",
            index_name=index_name,
            namespace=namespace,
        )
        self._retry_config = retry_config or RetryConfig()

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Pinecone operations."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(ConnectionError, TimeoutError, RuntimeError),
        )

    def _ensure_initialized(self) -> Any:
        """Lazy initialization of Pinecone index."""
        if self._index is None:
            self._logger.debug("initializing_pinecone")
            self._index = self._pc.Index(self._index_name)
            self._logger.info("pinecone_initialized")
        return self._index

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
            index = self._ensure_initialized()

            # Prepare vectors for upsert
            vectors = []
            for _i, (id_, embedding, metadata, document) in enumerate(
                zip(ids, embeddings, metadatas, documents, strict=False)
            ):
                # Include document text in metadata for retrieval
                vector_metadata = {
                    **metadata,
                    "text": document,
                }
                vectors.append(
                    {
                        "id": id_,
                        "values": embedding,
                        "metadata": vector_metadata,
                    }
                )

            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                index.upsert(vectors=batch, namespace=self._namespace)

        try:
            _add_sync()
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
            index = self._ensure_initialized()
            results = index.query(
                vector=embedding,
                top_k=top_k,
                namespace=self._namespace,
                include_metadata=True,
            )

            # Transform Pinecone results to expected format
            output = []
            if results.matches:
                for match in results.matches:
                    metadata = match.metadata or {}
                    # Extract text from metadata
                    document = metadata.pop("text", "")
                    result_dict = {
                        "id": match.id,
                        "metadata": metadata,
                        "document": document,
                        "distance": match.score or 0.0,
                    }
                    output.append(result_dict)

            return output

        try:
            results = _query_sync()
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
            index = self._ensure_initialized()

            # Query for all items with matching source_url in metadata
            results = index.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata-only query
                top_k=10000,
                namespace=self._namespace,
                filter={"source_url": {"$eq": source_url}},
                include_metadata=False,
            )

            # Delete if any results found
            if results.matches:
                ids_to_delete = [match.id for match in results.matches]
                index.delete(ids=ids_to_delete, namespace=self._namespace)
                return len(ids_to_delete)
            return 0

        try:
            deleted_count = _delete_sync()
            operation_logger.info("documents_deleted", deleted_count=deleted_count)
        except Exception as e:
            operation_logger.error(
                "delete_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
