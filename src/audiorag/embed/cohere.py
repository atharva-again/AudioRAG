"""Cohere embedding provider."""

from __future__ import annotations

from typing import Any, cast

from audiorag.core.logging_config import get_logger
from audiorag.embed._base import EmbedderMixin

logger = get_logger(__name__)


class CohereEmbeddingProvider(EmbedderMixin):
    """Embedding provider using Cohere's embed models."""

    MODEL_EMBED_V4 = "embed-v4.0"
    MODEL_EMBED_V3 = "embed-v3.0"
    MODEL_EMBED_ENGLISH_V3 = "embed-english-v3.0"
    MODEL_EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    MODEL_EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    MODEL_EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    INPUT_TYPE_SEARCH_DOCUMENT = "search_document"
    INPUT_TYPE_SEARCH_QUERY = "search_query"

    _provider_name: str = "cohere_embedding"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "embed-v4.0",
        input_type: str = "search_document",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Cohere embedding provider."""
        from cohere import AsyncClientV2  # type: ignore[import]
        from cohere.errors import (  # type: ignore[import]
            InternalServerError,
            ServiceUnavailableError,
            TooManyRequestsError,
        )

        self._retryable_exceptions: tuple[type[Exception], ...] = (
            TooManyRequestsError,
            ServiceUnavailableError,
            InternalServerError,
            ConnectionError,
        )
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        self.client = AsyncClientV2(api_key=api_key)
        self.input_type = input_type

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Cohere."""
        operation_logger = self._logger.bind(
            texts_count=len(texts),
            operation="embed",
        )
        operation_logger.debug("embedding_started")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _embed_with_retry() -> Any:
            all_embeddings = []
            batch_size = 96
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=self.input_type,
                    embedding_types=["float"],
                )
                all_embeddings.extend(cast(Any, response.embeddings).float)
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
