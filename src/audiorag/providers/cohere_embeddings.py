"""Cohere embedding provider implementation."""

from __future__ import annotations

from typing import Any

from audiorag.logging_config import get_logger
from audiorag.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class CohereEmbeddingProvider:
    """Embedding provider using Cohere's embed models.

    Cohere's embed-v4 and related models are popular in 2026 for their
    multilingual capabilities and enterprise features.

    Available models:
        - "embed-v4.0" (default): Latest v4 model with best accuracy
        - "embed-v3.0": Previous generation v3 model
        - "embed-english-v3.0": English-only v3 model
        - "embed-multilingual-v3.0": Multilingual v3 model
        - "embed-english-light-v3.0": Lightweight English model
        - "embed-multilingual-light-v3.0": Lightweight multilingual model
    """

    MODEL_EMBED_V4 = "embed-v4.0"
    MODEL_EMBED_V3 = "embed-v3.0"
    MODEL_EMBED_ENGLISH_V3 = "embed-english-v3.0"
    MODEL_EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    MODEL_EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    MODEL_EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    INPUT_TYPE_SEARCH_DOCUMENT = "search_document"
    INPUT_TYPE_SEARCH_QUERY = "search_query"
    INPUT_TYPE_CLASSIFICATION = "classification"
    INPUT_TYPE_CLUSTERING = "clustering"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-v4.0",
        input_type: str = "search_document",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize the Cohere embedding provider.

        Args:
            api_key: Cohere API key. If None, uses COHERE_API_KEY environment variable.
            model: The embedding model to use. Defaults to "embed-v4.0".
                Options: "embed-v4.0", "embed-v3.0", "embed-english-v3.0",
                "embed-multilingual-v3.0", "embed-english-light-v3.0",
                "embed-multilingual-light-v3.0".
            input_type: Type of input for embeddings. Defaults to "search_document".
                Options: "search_document", "search_query", "classification", "clustering".
            retry_config: Retry configuration. Uses default if not provided.
        """
        from cohere import AsyncClientV2  # type: ignore  # noqa: PLC0415

        self.client = AsyncClientV2(api_key=api_key)
        self._model = model
        self.input_type = input_type
        self._logger = logger.bind(provider="cohere_embedding", model=model)
        self._retry_config = retry_config or RetryConfig()

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name - switch models instantly.

        Args:
            value: Model name. Any valid Cohere embed model string.
        """
        self._model = value
        self._logger = logger.bind(provider="cohere_embedding", model=value)

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Cohere API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using Cohere.

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
            # Cohere supports batching up to 96 texts per request
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
                all_embeddings.extend(response.embeddings.float)

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
