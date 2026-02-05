"""Weaviate vector store implementation."""

from __future__ import annotations

from typing import Any

from audiorag.logging_config import get_logger
from audiorag.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class WeaviateVectorStore:
    """Weaviate-based vector store implementation satisfying VectorStoreProvider protocol.

    Weaviate is a popular open-source vector database in 2026, offering
    semantic search, vector search, and hybrid search capabilities.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "AudioRAG",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize Weaviate client.

        Args:
            url: Weaviate instance URL. If None, uses WEAVIATE_URL environment variable.
            api_key: Weaviate API key. If None, uses WEAVIATE_API_KEY environment variable.
            collection_name: Name of the collection/class to use
            retry_config: Retry configuration. Uses default if not provided.
        """
        # Lazy import to avoid ModuleNotFoundError when optional dep not installed
        import weaviate  # noqa: PLC0415 # type: ignore

        if api_key:
            self._client = weaviate.connect_to_wcs(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key),
            )
        else:
            # Connect to local instance
            self._client = weaviate.connect_to_local()

        self._collection_name = collection_name
        self._collection = None
        self._logger = logger.bind(
            provider="weaviate_vector_store",
            collection_name=collection_name,
        )
        self._retry_config = retry_config or RetryConfig()

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Weaviate operations."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(ConnectionError, TimeoutError, RuntimeError),
        )

    def _ensure_initialized(self) -> Any:
        """Lazy initialization of Weaviate collection."""
        if self._collection is None:
            self._logger.debug("initializing_weaviate")

            # Check if collection exists, create if not
            if not self._client.collections.exists(self._collection_name):
                from weaviate.classes.config import Configure, DataType, Property  # noqa: PLC0415 # type: ignore

                self._client.collections.create(
                    name=self._collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="source_url", data_type=DataType.TEXT),
                        Property(name="video_title", data_type=DataType.TEXT),
                        Property(name="start_time", data_type=DataType.NUMBER),
                        Property(name="end_time", data_type=DataType.NUMBER),
                    ],
                )

            self._collection = self._client.collections.get(self._collection_name)
            self._logger.info("weaviate_initialized")
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

            # Prepare objects for insertion
            with collection.batch.dynamic() as batch:
                for id_, embedding, metadata, document in zip(
                    ids, embeddings, metadatas, documents, strict=False
                ):
                    batch.add_object(
                        properties={
                            "text": document,
                            "source_url": metadata.get("source_url", ""),
                            "video_title": metadata.get("video_title", ""),
                            "start_time": metadata.get("start_time", 0.0),
                            "end_time": metadata.get("end_time", 0.0),
                        },
                        vector=embedding,
                        uuid=id_,
                    )

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
            collection = self._ensure_initialized()
            results = collection.query.near_vector(
                near_vector=embedding,
                limit=top_k,
                return_metadata=["distance"],
            )

            # Transform Weaviate results to expected format
            output = []
            if results.objects:
                for obj in results.objects:
                    result_dict = {
                        "id": str(obj.uuid),
                        "metadata": {
                            "source_url": obj.properties.get("source_url", ""),
                            "video_title": obj.properties.get("video_title", ""),
                            "start_time": obj.properties.get("start_time", 0.0),
                            "end_time": obj.properties.get("end_time", 0.0),
                        },
                        "document": obj.properties.get("text", ""),
                        "distance": obj.metadata.distance if obj.metadata else 0.0,
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
            collection = self._ensure_initialized()

            # Delete all objects with matching source_url
            result = collection.data.delete_many(
                where={
                    "path": ["source_url"],
                    "operator": "Equal",
                    "valueText": source_url,
                }
            )

            return result.matches

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
