"""Passthrough reranker provider implementation."""

from audiorag.logging_config import get_logger

logger = get_logger(__name__)


class PassthroughReranker:
    """No-op reranker that returns the first N documents with score 1.0.

    Satisfies the RerankerProvider Protocol by implementing the async rerank method.
    Used as a fallback when no Cohere API key is configured.
    """

    def __init__(self) -> None:
        """Initialize passthrough reranker."""
        self._logger = logger.bind(provider="passthrough_reranker")

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 3  # noqa: ARG002
    ) -> list[tuple[int, float]]:
        """Return the first N documents with score 1.0.

        Args:
            query: The query string (unused in passthrough mode).
            documents: List of document strings to rerank.
            top_n: Number of top documents to return. Defaults to 3.

        Returns:
            List of tuples containing (document_index, score) for the first N documents.
            Each score is 1.0 to indicate equal relevance.
        """
        operation_logger = self._logger.bind(
            documents_count=len(documents),
            top_n=top_n,
            operation="rerank",
        )
        operation_logger.debug("passthrough_rerank_started")

        # Return indices and scores for the first top_n documents
        num_docs = min(top_n, len(documents))
        results = [(i, 1.0) for i in range(num_docs)]

        operation_logger.info(
            "passthrough_rerank_completed", results_count=len(results)
        )
        return results
