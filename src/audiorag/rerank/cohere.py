"""Cohere reranker provider."""

from __future__ import annotations

from typing import Any

from cohere import AsyncClientV2  # type: ignore
from cohere.errors import RateLimitError, ServiceUnavailableError  # type: ignore

from audiorag.core.logging_config import get_logger
from audiorag.rerank._base import RerankerMixin

logger = get_logger(__name__)


class CohereReranker(RerankerMixin):
    """Cohere reranking provider."""

    _provider_name: str = "cohere_reranker"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        RateLimitError,
        ServiceUnavailableError,
        ConnectionError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Cohere reranker."""
        super().__init__(api_key=api_key, retry_config=retry_config)
        self.client = AsyncClientV2(api_key=api_key)
        self.model = model
        self._logger = self._logger.bind(model=model)

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 3
    ) -> list[tuple[int, float]]:
        """Rerank documents using Cohere API."""
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
            results = [(result.index, result.relevance_score) for result in response.results]
            operation_logger.info("reranking_completed", results_count=len(results))
            return results
        except Exception as e:
            raise await self._wrap_error(e, "rerank")
