"""Voyage AI embedding provider implementation."""

from __future__ import annotations

from typing import Any

from audiorag.logging_config import get_logger
from audiorag.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class VoyageEmbeddingProvider:
    """Embedding provider using Voyage AI's embedding models.

    Voyage AI offers state-of-the-art embedding models in 2026, including
    the voyage-3.5 and voyage-4 series, known for superior retrieval quality
    and competitive pricing.

    Available models:
        - "voyage-3.5" (default): Best general-purpose model
        - "voyage-3.5-lite": Lighter, faster version
        - "voyage-4": Latest generation model
        - "voyage-4-lite": Lighter version of voyage-4
        - "voyage-4-large": Highest accuracy version
        - "voyage-4-nano": Open-weighted, smallest model
    """

    MODEL_VOYAGE_3_5 = "voyage-3.5"
    MODEL_VOYAGE_3_5_LITE = "voyage-3.5-lite"
    MODEL_VOYAGE_4 = "voyage-4"
    MODEL_VOYAGE_4_LITE = "voyage-4-lite"
    MODEL_VOYAGE_4_LARGE = "voyage-4-large"
    MODEL_VOYAGE_4_NANO = "voyage-4-nano"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-3.5",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize the Voyage AI embedding provider.

        Args:
            api_key: Voyage AI API key. If None, uses VOYAGE_API_KEY environment variable.
            model: The embedding model to use. Defaults to "voyage-3.5".
                Options: "voyage-3.5", "voyage-3.5-lite", "voyage-4", "voyage-4-lite",
                "voyage-4-large", "voyage-4-nano".
            retry_config: Retry configuration. Uses default if not provided.
        """
        import voyageai  # type: ignore  # noqa: PLC0415

        self.client = voyageai.AsyncClient(api_key=api_key)
        self._model = model
        self._logger = logger.bind(provider="voyage_embedding", model=model)
        self._retry_config = retry_config or RetryConfig()

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name - switch models instantly.

        Args:
            value: Model name. Any valid Voyage AI model string.
        """
        self._model = value
        self._logger = logger.bind(provider="voyage_embedding", model=value)

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Voyage AI API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using Voyage AI.

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
            # Voyage AI supports batching up to 128 texts
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
            operation_logger.error(
                "embedding_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
