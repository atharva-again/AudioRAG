"""Anthropic Claude generation provider implementation."""

from __future__ import annotations

from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.core.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class AnthropicGenerationProvider:
    """Generation provider using Anthropic's Claude models.

    Claude 3.7 Opus/Sonnet/Haiku are among the top LLMs in 2026, known for
    strong reasoning, safety features, and large context windows.

    Available models:
        - "claude-3-7-sonnet-20250219" (default): Best balance of speed and intelligence
        - "claude-3-7-opus-20250219": Most powerful model for complex tasks
        - "claude-3-7-haiku-20250219": Fastest model for lightweight tasks
        - "claude-3-5-sonnet-20241022": Previous generation Sonnet
        - "claude-3-5-haiku-20241022": Previous generation Haiku
        - "claude-3-opus-20240229": Claude 3 Opus
        - "claude-3-sonnet-20240229": Claude 3 Sonnet
        - "claude-3-haiku-20240307": Claude 3 Haiku
    """

    MODEL_CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    MODEL_CLAUDE_3_7_OPUS = "claude-3-7-opus-20250219"
    MODEL_CLAUDE_3_7_HAIKU = "claude-3-7-haiku-20250219"
    MODEL_CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    MODEL_CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    MODEL_CLAUDE_3_OPUS = "claude-3-opus-20240229"
    MODEL_CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    MODEL_CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 1024,
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize the Anthropic generation provider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY environment variable.
            model: The Claude model to use. Defaults to "claude-3-7-sonnet-20250219".
                Options: "claude-3-7-sonnet-20250219", "claude-3-7-opus-20250219",
                "claude-3-7-haiku-20250219", "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022", "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", "claude-3-haiku-20240307".
            max_tokens: Maximum tokens to generate. Defaults to 1024.
            retry_config: Retry configuration. Uses default if not provided.
        """
        from anthropic import AsyncAnthropic  # noqa: PLC0415 # type: ignore

        self.client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self.max_tokens = max_tokens
        self._logger = logger.bind(provider="anthropic_generation", model=model)
        self._retry_config = retry_config or RetryConfig()

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name - switch models instantly.

        Args:
            value: Model name. Any valid Claude model string.
        """
        self._model = value
        self._logger = logger.bind(provider="anthropic_generation", model=value)

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Anthropic API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ),
        )

    async def generate(self, query: str, context: list[str]) -> str:
        """Generate an answer using Anthropic's Claude API.

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
            return await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )

        try:
            response = await _generate_with_retry()

            # Extract and return the generated answer
            answer = response.content[0].text if response.content else ""
            operation_logger.info(
                "generation_completed",
                answer_length=len(answer),
            )
            return answer

        except Exception as e:
            operation_logger.error(
                "generation_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
