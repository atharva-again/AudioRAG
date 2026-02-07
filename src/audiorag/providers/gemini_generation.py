"""Google Gemini generation provider implementation."""

from __future__ import annotations

from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.core.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class GeminiGenerationProvider:
    """Generation provider using Google's Gemini models.

    Gemini 2 Ultra/Pro/Flash are among the top LLMs in 2026, offering
    strong multimodal capabilities and integration with Google Cloud.

    Available models:
        - "gemini-2.0-flash-001" (default): Fastest, most efficient model
        - "gemini-2.0-pro-001": Advanced reasoning and coding tasks
        - "gemini-2.0-ultra-001": Highest quality, most capable
        - "gemini-2.0-flash-lite-001": Ultra-fast, cost-effective
        - "gemini-1.5-pro-002": Previous generation Pro
        - "gemini-1.5-flash-002": Previous generation Flash
    """

    MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash-001"
    MODEL_GEMINI_2_0_PRO = "gemini-2.0-pro-001"
    MODEL_GEMINI_2_0_ULTRA = "gemini-2.0-ultra-001"
    MODEL_GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite-001"
    MODEL_GEMINI_1_5_PRO = "gemini-1.5-pro-002"
    MODEL_GEMINI_1_5_FLASH = "gemini-1.5-flash-002"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash-001",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize the Gemini generation provider.

        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY environment variable.
            model: The Gemini model to use. Defaults to "gemini-2.0-flash-001".
                Options: "gemini-2.0-flash-001", "gemini-2.0-pro-001",
                "gemini-2.0-ultra-001", "gemini-2.0-flash-lite-001",
                "gemini-1.5-pro-002", "gemini-1.5-flash-002".
            retry_config: Retry configuration. Uses default if not provided.
        """
        import google.generativeai as genai  # type: ignore  # noqa: PLC0415

        if api_key:
            genai.configure(api_key=api_key)

        self._genai = genai
        self._model = model
        self._logger = logger.bind(provider="gemini_generation", model=model)
        self._retry_config = retry_config or RetryConfig()
        self._client = genai.GenerativeModel(model_name=model)

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name - switch models instantly.

        Args:
            value: Model name. Any valid Gemini model string.
        """
        self._model = value
        self._logger = logger.bind(provider="gemini_generation", model=value)
        self._client = self._genai.GenerativeModel(model_name=value)

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Google API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ),
        )

    async def generate(self, query: str, context: list[str]) -> str:
        """Generate an answer using Google's Gemini API.

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

        # Create the prompt
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the context doesn't contain relevant information, say so.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}"
        )

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _generate_with_retry() -> Any:
            return await self._client.generate_content_async(prompt)

        try:
            response = await _generate_with_retry()

            # Extract and return the generated answer
            answer = response.text if hasattr(response, "text") else ""
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
