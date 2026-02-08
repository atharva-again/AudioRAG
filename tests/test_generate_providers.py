"""Unit tests for generation providers - instantiation tests."""

import pytest


class TestGenerationProvidersInstantiation:
    """Test generation provider instantiation through lazy imports."""

    def test_openai_generator_instantiation(self) -> None:
        """Test OpenAI generator can be instantiated."""
        from audiorag.generate import OpenAIGenerator

        provider = OpenAIGenerator(api_key="test-key")
        assert provider.model == "gpt-4o-mini"

    def test_openai_generator_custom_model(self) -> None:
        """Test OpenAI generator with custom model."""
        from audiorag.generate import OpenAIGenerator

        provider = OpenAIGenerator(api_key="test-key", model="gpt-4o")
        assert provider.model == "gpt-4o"

    def test_anthropic_generator_instantiation(self) -> None:
        """Test Anthropic generator can be instantiated."""
        from audiorag.generate import AnthropicGenerator

        provider = AnthropicGenerator(api_key="test-key")
        assert provider._provider_name == "anthropic_generation"

    def test_anthropic_generator_custom_model(self) -> None:
        """Test Anthropic generator with custom model."""
        from audiorag.generate.anthropic import AnthropicGenerator

        provider = AnthropicGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        assert provider.model == "claude-sonnet-4-20250514"

    def test_gemini_generator_instantiation(self) -> None:
        """Test Gemini generator can be instantiated."""
        from audiorag.generate import GeminiGenerator

        provider = GeminiGenerator(api_key="test-key")
        assert provider._provider_name == "gemini_generation"
