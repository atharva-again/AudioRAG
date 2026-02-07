"""Voyage AI embedding provider."""

from __future__ import annotations

from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.embed._base import EmbedderMixin

logger = get_logger(__name__)


class VoyageEmbeddingProvider(EmbedderMixin):
    """Embedding provider using Voyage AI's embedding models."""

    MODEL_VOYAGE_3_5 = "voyage-3.5"
    MODEL_VOYAGE_3_5_LITE = "voyage-3.5-lite"
    MODEL_VOYAGE_4 = "voyage-4"
    MODEL_VOYAGE_4_LITE = "voyage-4-lite"
    MODEL_VOYAGE_4_LARGE = "voyage-4-large"
    MODEL_VOYAGE_4_NANO = "voyage-4-nano"

    _provider_name: str = "voyage_embedding"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        RuntimeError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "voyage-3.5",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Voyage AI embedding provider."""
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        import voyageai  # noqa: PLC0415

        self.client = voyageai.AsyncClient(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Voyage AI."""
        operation_logger = self._logger.bind(
            texts_count=len(texts),
            operation="embed",
        )
        operation_logger.debug("embedding_started")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _embed_with_retry() -> Any:
            all_embeddings = []
            batch_size = 128
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="document",
                )
                all_embeddings.extend(response.embeddings)
            return all_embeddings

        try:
            embeddings = await _embed_with_retry()
            operation_logger.info(
                "embedding_completed",
                embeddings_count=len(embeddings),
                dimensions=len(embeddings[0]) if embeddings else 0,
            )
            return embeddings
        except Exception as e:
            raise await self._wrap_error(e, "embed")
