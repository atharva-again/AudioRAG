"""OpenAI embedding provider."""

from __future__ import annotations

from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from audiorag.core.logging_config import get_logger
from audiorag.embed._base import EmbedderMixin

logger = get_logger(__name__)


class OpenAIEmbeddingProvider(EmbedderMixin):
    """Embedding provider using OpenAI's embedding models."""

    MODEL_TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    MODEL_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    MODEL_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    _provider_name: str = "openai_embedding"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        RateLimitError,
        APIError,
        APITimeoutError,
        ConnectionError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize the OpenAI embedding provider."""
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        self.client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI."""
        operation_logger = self._logger.bind(
            texts_count=len(texts),
            operation="embed",
        )
        operation_logger.debug("embedding_started")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _embed_with_retry() -> Any:
            return await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )

        try:
            response = await _embed_with_retry()
            embeddings = [item.embedding for item in response.data]
            operation_logger.info(
                "embedding_completed",
                embeddings_count=len(embeddings),
                dimensions=len(embeddings[0]) if embeddings else 0,
            )
            return embeddings
        except Exception as e:
            raise await self._wrap_error(e, "embed")
