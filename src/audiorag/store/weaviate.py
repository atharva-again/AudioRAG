"""Weaviate vector store provider."""

from __future__ import annotations

from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.store._base import VectorStoreMixin

logger = get_logger(__name__)


class WeaviateVectorStore(VectorStoreMixin):
    """Weaviate-based vector store."""

    _provider_name: str = "weaviate_vector_store"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        RuntimeError,
    )

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "AudioRAG",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Weaviate client."""
        super().__init__(retry_config=retry_config)
        import weaviate  # type: ignore[import]

        if api_key:
            self._client = weaviate.connect_to_wcs(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key),
            )
        else:
            self._client = weaviate.connect_to_local()

        self._collection_name = collection_name
        self._collection = None
        self._logger = self._logger.bind(collection_name=collection_name)

    def _ensure_initialized(self) -> Any:
        """Lazy initialization of Weaviate collection."""
        if self._collection is None:
            self._logger.debug("initializing_weaviate")

            if not self._client.collections.exists(self._collection_name):
                from weaviate.classes.config import (  # type: ignore[import]
                    Configure,
                    DataType,
                    Property,
                )

                self._client.collections.create(
                    name=self._collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="source_url", data_type=DataType.TEXT),
                        Property(name="title", data_type=DataType.TEXT),
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
            with collection.batch.dynamic() as batch:
                for id_, embedding, metadata, document in zip(
                    ids, embeddings, metadatas, documents, strict=False
                ):
                    batch.add_object(
                        properties={
                            "text": document,
                            "source_url": metadata.get("source_url", ""),
                            "title": metadata.get("title", ""),
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
            results = collection.query.near_vector(
                near_vector=embedding,
                limit=top_k,
                return_metadata=["distance"],
            )
            return self._format_results(results)

        try:
            results = _query_sync()
            operation_logger.info("query_completed", results_count=len(results))
            return results
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
            raise await self._wrap_error(e, "delete_by_source")

    def _format_results(self, results: Any) -> list[dict]:
        """Transform Weaviate results to expected format."""
        output = []
        if results.objects:
            for obj in results.objects:
                output.append(
                    {
                        "id": str(obj.uuid),
                        "metadata": {
                            "source_url": obj.properties.get("source_url", ""),
                            "title": obj.properties.get("title", ""),
                            "start_time": obj.properties.get("start_time", 0.0),
                            "end_time": obj.properties.get("end_time", 0.0),
                        },
                        "document": obj.properties.get("text", ""),
                        "distance": obj.metadata.distance if obj.metadata else 0.0,
                    }
                )
        return output
