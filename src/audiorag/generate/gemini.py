"""Google Gemini generation provider."""

from __future__ import annotations

from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.generate._base import GeneratorMixin

logger = get_logger(__name__)


class GeminiGenerator(GeneratorMixin):
    """Google Gemini LLM generation provider."""

    _provider_name: str = "gemini_generation"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash-001",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Gemini generator."""
        import httpx
        from google import genai
        from google.genai import errors as genai_errors

        # Retryable exceptions for google.genai SDK
        self._retryable_exceptions: tuple[type[Exception], ...] = (
            genai_errors.APIError,
            genai_errors.ServerError,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.TimeoutException,
            ConnectionError,
        )
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    async def generate(self, query: str, context: list[str]) -> str:
        """Generate answer using Google Gemini API."""
        operation_logger = self._logger.bind(
            query_length=len(query),
            context_count=len(context),
            operation="generate",
        )
        operation_logger.debug("generation_started")

        context_text = "\n".join(context)
        prompt = (
            "You are a helpful assistant. Use the provided context to answer. "
            "If the context doesn't contain relevant information, say so.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}"
        )

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _generate_with_retry() -> Any:
            async with self._client.aio as aclient:
                return await aclient.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )

        try:
            response = await _generate_with_retry()
            answer = response.text if hasattr(response, "text") else ""
            operation_logger.info("generation_completed", answer_length=len(answer))
            return answer
        except Exception as e:
            raise await self._wrap_error(e, "generate")
