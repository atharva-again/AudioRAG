"""OpenAI embedding provider implementation."""

from __future__ import annotations

from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError  # type: ignore

from audiorag.core import (
    RetryConfig,
    create_retry_decorator,
    get_logger,
)

logger = get_logger(__name__)


class OpenAIEmbeddingProvider:
    """Embedding provider using OpenAI's embedding models.

    Satisfies the EmbeddingProvider Protocol by implementing the async embed method.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str = "text-embedding-3-small",
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            client: AsyncOpenAI client instance. If None, a new client will be created.
            model: The embedding model to use. Defaults to "text-embedding-3-small".
            retry_config: Retry configuration. Uses default if not provided.
        """
        self.client = client or AsyncOpenAI()
        self.model = model
        self._logger = logger.bind(provider="openai_embedding", model=model)
        self._retry_config = retry_config or RetryConfig()

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for OpenAI API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(
                RateLimitError,
                APIError,
                APITimeoutError,
                ConnectionError,
            ),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using OpenAI.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, where each vector is a list of floats.
        """
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

            # Extract embeddings from response and return as list[list[float]]
            embeddings = [item.embedding for item in response.data]

            operation_logger.info(
                "embedding_completed",
                embeddings_count=len(embeddings),
                dimensions=len(embeddings[0]) if embeddings else 0,
            )
            return embeddings

        except Exception as e:
            operation_logger.error(
                "embedding_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
