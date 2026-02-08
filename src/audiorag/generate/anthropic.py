"""Anthropic Claude generation provider."""

from __future__ import annotations

from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.generate._base import GeneratorMixin

logger = get_logger(__name__)


class AnthropicGenerator(GeneratorMixin):
    """Anthropic Claude LLM generation provider."""

    _provider_name: str = "anthropic_generation"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        RuntimeError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 1024,
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Anthropic generator."""
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        from anthropic import AsyncAnthropic  # type: ignore[import]

        self.client = AsyncAnthropic(api_key=api_key)
        self.max_tokens = max_tokens

    async def generate(self, query: str, context: list[str]) -> str:
        """Generate answer using Anthropic Claude API."""
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
            return await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

        try:
            response = await _generate_with_retry()
            answer = response.content[0].text if response.content else ""
            operation_logger.info("generation_completed", answer_length=len(answer))
            return answer
        except Exception as e:
            raise await self._wrap_error(e, "generate")
