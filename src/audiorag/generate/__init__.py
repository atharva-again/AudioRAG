"""Generation (LLM) providers."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import with clear error messages for missing dependencies."""
    if name == "OpenAIGenerator":
        try:
            from audiorag.generate.openai import OpenAIGenerator

            return OpenAIGenerator
        except ImportError:
            raise ImportError(
                "OpenAIGenerator requires 'openai'. Install with: uv pip install audiorag[openai]"
            ) from None
    if name == "AnthropicGenerator":
        try:
            from audiorag.generate.anthropic import AnthropicGenerator

            return AnthropicGenerator
        except ImportError:
            raise ImportError(
                "AnthropicGenerator requires 'anthropic'. "
                "Install with: uv pip install audiorag[anthropic]"
            ) from None
    if name == "GeminiGenerator":
        try:
            from audiorag.generate.gemini import GeminiGenerator

            return GeminiGenerator
        except ImportError:
            raise ImportError(
                "GeminiGenerator requires 'google-generativeai'. "
                "Install with: uv pip install audiorag[gemini]"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicGenerator",
    "GeminiGenerator",
    "OpenAIGenerator",
]
