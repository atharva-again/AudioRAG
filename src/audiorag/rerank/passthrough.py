"""Passthrough reranker provider (no-op)."""

from __future__ import annotations

from audiorag.core.logging_config import get_logger

logger = get_logger(__name__)


class PassthroughReranker:
    """No-op reranker returning first N documents with score 1.0."""

    def __init__(self) -> None:
        """Initialize passthrough reranker."""
        self._logger = logger.bind(provider="passthrough_reranker")

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 3,
    ) -> list[tuple[int, float]]:
        """Return first N documents with score 1.0."""
        operation_logger = self._logger.bind(
            documents_count=len(documents),
            top_n=top_n,
            operation="rerank",
        )
        operation_logger.debug("passthrough_rerank_started")

        num_docs = min(top_n, len(documents))
        results = [(i, 1.0) for i in range(num_docs)]

        operation_logger.info("passthrough_rerank_completed", results_count=len(results))
        return results
