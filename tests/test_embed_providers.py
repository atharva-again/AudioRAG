"""Unit tests for embedding providers - instantiation tests."""

import pytest


class TestEmbeddingProvidersInstantiation:
    """Test embedding provider instantiation through lazy imports."""

    def test_openai_embedder_instantiation(self) -> None:
        """Test OpenAI embedder can be instantiated."""
        from audiorag.embed import OpenAIEmbedder

        provider = OpenAIEmbedder(api_key="test-key")
        assert provider.model == "text-embedding-3-small"

    def test_openai_embedder_custom_model(self) -> None:
        """Test OpenAI embedder with custom model."""
        from audiorag.embed import OpenAIEmbedder

        provider = OpenAIEmbedder(api_key="test-key", model="text-embedding-3-large")
        assert provider.model == "text-embedding-3-large"

    def test_cohere_embedder_instantiation(self) -> None:
        """Test Cohere embedder can be instantiated."""
        from audiorag.embed import CohereEmbedder

        provider = CohereEmbedder(api_key="test-key")
        assert provider.model == "embed-v4.0"

    def test_cohere_embedder_custom_model(self) -> None:
        """Test Cohere embedder with custom model."""
        from audiorag.embed import CohereEmbedder

        provider = CohereEmbedder(api_key="test-key", model="embed-english-v3.0")
        assert provider.model == "embed-english-v3.0"

    def test_voyage_embedder_instantiation(self) -> None:
        """Test Voyage embedder can be instantiated."""
        from audiorag.embed import VoyageEmbedder

        provider = VoyageEmbedder(api_key="test-key")
        assert provider.model == "voyage-3.5"

    def test_voyage_embedder_custom_model(self) -> None:
        """Test Voyage embedder with custom model."""
        from audiorag.embed import VoyageEmbedder

        provider = VoyageEmbedder(api_key="test-key", model="voyage-4")
        assert provider.model == "voyage-4"
