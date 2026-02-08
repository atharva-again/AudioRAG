"""OpenAI generation provider."""

from __future__ import annotations

from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError  # type: ignore

from audiorag.core.logging_config import get_logger
from audiorag.generate._base import GeneratorMixin

logger = get_logger(__name__)


class OpenAIGenerator(GeneratorMixin):
    """OpenAI LLM generation provider."""

    _provider_name: str = "openai_generation"
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
        model: str = "gpt-4o-mini",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize OpenAI generator."""
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, query: str, context: list[str]) -> str:
        """Generate answer using OpenAI chat completion."""
        operation_logger = self._logger.bind(
            query_length=len(query),
            context_count=len(context),
            operation="generate",
        )
        operation_logger.debug("generation_started")

        context_text = "\n".join(context)
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer. "
            "If the context doesn't contain relevant information, say so."
        )
        user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _generate_with_retry() -> Any:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )

        try:
            response = await _generate_with_retry()
            answer = response.choices[0].message.content
            operation_logger.info(
                "generation_completed", answer_length=len(answer) if answer else 0
            )
            return answer
        except Exception as e:
            raise await self._wrap_error(e, "generate")
