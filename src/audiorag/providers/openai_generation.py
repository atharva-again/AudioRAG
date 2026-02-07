"""OpenAI generation provider implementation."""

from __future__ import annotations

from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError  # type: ignore

from audiorag.core.logging_config import get_logger
from audiorag.core.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class OpenAIGenerationProvider:
    """Generation provider using OpenAI's chat completion models.

    Satisfies the GenerationProvider Protocol by implementing the async generate method.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str = "gpt-4o-mini",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize the OpenAI generation provider.

        Args:
            client: AsyncOpenAI client instance. If None, a new client will be created.
            model: The chat completion model to use. Defaults to "gpt-4o-mini".
            retry_config: Retry configuration. Uses default if not provided.
        """
        self.client = client or AsyncOpenAI()
        self.model = model
        self._logger = logger.bind(provider="openai_generation", model=model)
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

    async def generate(self, query: str, context: list[str]) -> str:
        """Generate an answer using OpenAI's chat completion API.

        Args:
            query: The user's query or question.
            context: List of context strings to use for generating the answer.

        Returns:
            The generated answer as a string.
        """
        operation_logger = self._logger.bind(
            query_length=len(query),
            context_count=len(context),
            operation="generate",
        )
        operation_logger.debug("generation_started")

        # Combine context into a single string
        context_text = "\n".join(context)

        # Create the system prompt
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the context doesn't contain relevant information, say so."
        )

        # Create the user message with query and context
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

            # Extract and return the generated answer
            answer = response.choices[0].message.content
            operation_logger.info(
                "generation_completed",
                answer_length=len(answer) if answer else 0,
            )
            return answer

        except Exception as e:
            operation_logger.error(
                "generation_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
