"""Pinecone vector store provider."""

from __future__ import annotations

from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.store._base import VectorStoreMixin

logger = get_logger(__name__)


class PineconeVectorStore(VectorStoreMixin):
    """Pinecone-based vector store."""

    vector_id_default_format: str = "sha256"
    _provider_name: str = "pinecone_vector_store"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        index_name: str = "audiorag",
        namespace: str = "default",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Pinecone client."""
        super().__init__(retry_config=retry_config)
        from pinecone import Pinecone  # type: ignore[import]

        self._pc = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._namespace = namespace
        self._index = None
        self._logger = self._logger.bind(
            index_name=index_name,
            namespace=namespace,
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
        """Add embeddings to Pinecone."""
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
            vectors = []
            for _i, (id_, embedding, metadata, document) in enumerate(
                zip(ids, embeddings, metadatas, documents, strict=False)
            ):
                vectors.append(
                    {
                        "id": id_,
                        "values": embedding,
                        "metadata": {**metadata, "document": document},
                    }
                )
            index.upsert(vectors=vectors, namespace=self._namespace)

        try:
            await self._run_sync(_add_sync)
            operation_logger.info("documents_added")
        except Exception as e:
            raise await self._wrap_error(e, "add")

    async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        """Query Pinecone for similar embeddings."""
        operation_logger = self._logger.bind(
            operation="query",
            top_k=top_k,
        )
        operation_logger.debug("querying_documents")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _query_sync() -> Any:
            index = self._ensure_initialized()
            return index.query(
                vector=embedding,
                top_k=top_k,
                namespace=self._namespace,
                include_metadata=True,
            )

        try:
            results = await self._run_sync(_query_sync)
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
            index = self._ensure_initialized()
            results = index.query(
                vector=[0.0] * len(index.describe_index_stats().dimension),
                filter={"source_url": source_url},
                top_k=1000,
                namespace=self._namespace,
            )
            if results.matches:
                ids_to_delete = [m.id for m in results.matches]
                index.delete(ids=ids_to_delete, namespace=self._namespace)
                return len(ids_to_delete)
            return 0

        try:
            deleted_count = await self._run_sync(_delete_sync)
            operation_logger.info("documents_deleted", deleted_count=deleted_count)
        except Exception as e:
            raise await self._wrap_error(e, "delete_by_source")

    async def verify(self, ids: list[str]) -> bool:
        if not ids:
            return True

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _verify_sync() -> bool:
            index = self._ensure_initialized()
            response = index.fetch(ids=ids, namespace=self._namespace)
            vectors = getattr(response, "vectors", {})
            if isinstance(vectors, dict):
                return len(vectors) == len(ids)
            return False

        try:
            return await self._run_sync(_verify_sync)
        except Exception as e:
            raise await self._wrap_error(e, "verify")

    async def _run_sync(self, func: Any) -> Any:
        """Run sync function in executor."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)

    def _format_results(self, results: Any) -> list[dict]:
        """Transform Pinecone results to expected format."""
        output = []
        if results.matches:
            for match in results.matches:
                output.append(
                    {
                        "id": match.id,
                        "metadata": match.metadata or {},
                        "document": match.metadata.get("document", "") if match.metadata else "",
                        "distance": match.score,
                    }
                )
        return output
