"""Cohere reranker provider implementation."""

from __future__ import annotations

from typing import Any

from cohere import AsyncClientV2  # type: ignore
from cohere.errors import RateLimitError, ServiceUnavailableError  # type: ignore

from audiorag.core.logging_config import get_logger
from audiorag.core.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class CohereReranker:
    """Reranker provider using Cohere's reranking models.

    Satisfies the RerankerProvider Protocol by implementing the async rerank method.
    """

    def __init__(
        self,
        client: AsyncClientV2 | None = None,
        model: str = "rerank-english-v3.0",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize the Cohere reranker provider.

        Args:
            client: AsyncClientV2 client instance. If None, a new client will be created.
            model: The reranking model to use. Defaults to "rerank-english-v3.0".
            retry_config: Retry configuration. Uses default if not provided.
        """
        self.client = client or AsyncClientV2()
        self.model = model
        self._logger = logger.bind(provider="cohere_reranker", model=model)
        self._retry_config = retry_config or RetryConfig()

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Cohere API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(RateLimitError, ServiceUnavailableError, ConnectionError),
        )

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 3
    ) -> list[tuple[int, float]]:
        """Rerank documents based on relevance to the query using Cohere.

        Args:
            query: The query string to rerank documents against.
            documents: List of document strings to rerank.
            top_n: Number of top results to return. Defaults to 3.

        Returns:
            List of tuples containing (original_index, relevance_score) for the top_n documents,
            sorted by relevance score in descending order.
        """
        operation_logger = self._logger.bind(
            documents_count=len(documents),
            top_n=top_n,
            operation="rerank",
        )
        operation_logger.debug("reranking_started")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _rerank_with_retry() -> Any:
            return await self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
            )

        try:
            response = await _rerank_with_retry()

            # Extract results and return as list[tuple[int, float]]
            results = [(result.index, result.relevance_score) for result in response.results]
            operation_logger.info(
                "reranking_completed",
                results_count=len(results),
            )
            return results

        except Exception as e:
            operation_logger.error(
                "reranking_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
