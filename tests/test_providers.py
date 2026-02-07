"""Comprehensive tests for provider implementations."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audiorag.core.protocols import (
    EmbeddingProvider,
    GenerationProvider,
    RerankerProvider,
    STTProvider,
    VectorStoreProvider,
)
from audiorag.providers.passthrough_reranker import PassthroughReranker
from audiorag.retry_config import RetryConfig


class TestPassthroughReranker:
    """Test suite for PassthroughReranker implementation."""

    @pytest.fixture
    def reranker(self) -> PassthroughReranker:
        """Create a PassthroughReranker instance."""
        return PassthroughReranker()

    @pytest.mark.asyncio
    async def test_returns_first_n_documents(
        self, reranker: PassthroughReranker
    ) -> None:
        """Test that reranker returns first N documents."""
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        result = await reranker.rerank("query", documents, top_n=3)

        assert len(result) == 3
        assert result[0] == (0, 1.0)
        assert result[1] == (1, 1.0)
        assert result[2] == (2, 1.0)

    @pytest.mark.asyncio
    async def test_returns_all_if_fewer_than_top_n(
        self, reranker: PassthroughReranker
    ) -> None:
        """Test that reranker returns all documents if fewer than top_n."""
        documents = ["doc1", "doc2"]
        result = await reranker.rerank("query", documents, top_n=5)

        assert len(result) == 2
        assert result[0] == (0, 1.0)
        assert result[1] == (1, 1.0)

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_documents(
        self, reranker: PassthroughReranker
    ) -> None:
        """Test that reranker returns empty list for empty documents."""
        result = await reranker.rerank("query", [], top_n=3)

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_single_document(self, reranker: PassthroughReranker) -> None:
        """Test that reranker handles single document."""
        documents = ["doc1"]
        result = await reranker.rerank("query", documents, top_n=3)

        assert len(result) == 1
        assert result[0] == (0, 1.0)

    @pytest.mark.asyncio
    async def test_ignores_query_content(self, reranker: PassthroughReranker) -> None:
        """Test that query content doesn't affect results."""
        documents = ["doc1", "doc2"]
        result1 = await reranker.rerank("query1", documents, top_n=2)
        result2 = await reranker.rerank(
            "completely different query", documents, top_n=2
        )

        assert result1 == result2

    def test_satisfies_reranker_protocol(self, reranker: PassthroughReranker) -> None:
        """Test that PassthroughReranker satisfies RerankerProvider protocol."""
        assert isinstance(reranker, RerankerProvider)


class TestMockProviderCompliance:
    """Test suite for mock provider protocol compliance."""

    def test_mock_stt_provider_protocol(self, mock_stt_provider) -> None:
        """Test that mock STT provider satisfies protocol."""
        from audiorag.core.protocols import STTProvider  # noqa: PLC0415

        assert isinstance(mock_stt_provider, STTProvider)

    def test_mock_embedding_provider_protocol(self, mock_embedding_provider) -> None:
        """Test that mock embedding provider satisfies protocol."""
        from audiorag.core.protocols import EmbeddingProvider  # noqa: PLC0415

        assert isinstance(mock_embedding_provider, EmbeddingProvider)

    def test_mock_vector_store_provider_protocol(
        self, mock_vector_store_provider
    ) -> None:
        """Test that mock vector store provider satisfies protocol."""
        from audiorag.core.protocols import VectorStoreProvider  # noqa: PLC0415

        assert isinstance(mock_vector_store_provider, VectorStoreProvider)

    def test_mock_reranker_provider_protocol(self, mock_reranker_provider) -> None:
        """Test that mock reranker provider satisfies protocol."""
        from audiorag.core.protocols import RerankerProvider  # noqa: PLC0415

        assert isinstance(mock_reranker_provider, RerankerProvider)

    def test_mock_generation_provider_protocol(self, mock_generation_provider) -> None:
        """Test that mock generation provider satisfies protocol."""
        from audiorag.core.protocols import GenerationProvider  # noqa: PLC0415

        assert isinstance(mock_generation_provider, GenerationProvider)

    def test_mock_audio_source_provider_protocol(
        self, mock_audio_source_provider
    ) -> None:
        """Test that mock audio source provider satisfies protocol."""
        from audiorag.core.protocols import AudioSourceProvider  # noqa: PLC0415

        assert isinstance(mock_audio_source_provider, AudioSourceProvider)


class TestMockProviderBehavior:
    """Test suite for mock provider behavior."""

    @pytest.mark.asyncio
    async def test_mock_stt_returns_segments(self, mock_stt_provider) -> None:
        """Test that mock STT provider returns transcription segments."""
        from audiorag.core.models import TranscriptionSegment  # noqa: PLC0415

        segments = await mock_stt_provider.transcribe(Path("/tmp/test.mp3"))

        assert isinstance(segments, list)
        assert len(segments) > 0
        assert isinstance(segments[0], TranscriptionSegment)

    @pytest.mark.asyncio
    async def test_mock_embedding_returns_vectors(
        self, mock_embedding_provider
    ) -> None:
        """Test that mock embedding provider returns vectors."""
        texts = ["text1", "text2"]
        embeddings = await mock_embedding_provider.embed(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert isinstance(embeddings[0], list)
        assert isinstance(embeddings[0][0], float)

    @pytest.mark.asyncio
    async def test_mock_vector_store_add_and_query(
        self, mock_vector_store_provider
    ) -> None:
        """Test that mock vector store supports add and query."""
        await mock_vector_store_provider.add(
            ids=["id1"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"key": "value"}],
            documents=["doc1"],
        )

        results = await mock_vector_store_provider.query([0.1, 0.2], top_k=1)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_mock_reranker_returns_tuples(self, mock_reranker_provider) -> None:
        """Test that mock reranker returns index-score tuples."""
        documents = ["doc1", "doc2", "doc3"]
        results = await mock_reranker_provider.rerank("query", documents, top_n=2)

        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2
        assert isinstance(results[0][0], int)
        assert isinstance(results[0][1], float)

    @pytest.mark.asyncio
    async def test_mock_generation_returns_string(
        self, mock_generation_provider
    ) -> None:
        """Test that mock generation provider returns string."""
        answer = await mock_generation_provider.generate(
            "query", ["context1", "context2"]
        )

        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_mock_audio_source_returns_audiofile(
        self, mock_audio_source_provider, tmp_audio_dir
    ) -> None:
        """Test that mock audio source returns AudioFile."""
        from audiorag.core.models import AudioFile  # noqa: PLC0415

        audio_file = await mock_audio_source_provider.download(
            "https://youtube.com/watch?v=test", tmp_audio_dir
        )

        assert isinstance(audio_file, AudioFile)
        assert audio_file.source_url == "https://youtube.com/watch?v=test"


class TestProviderProtocolCompliance:
    """Test suite for new provider protocol compliance using mocks."""

    @pytest.fixture
    def retry_config(self):
        """Create a retry config for testing."""
        return RetryConfig(max_attempts=1, min_wait_seconds=0.1, max_wait_seconds=1.0)

    def test_deepgram_stt_provider_protocol(self, retry_config):
        """Test that DeepgramSTTProvider satisfies STTProvider protocol."""
        with patch("audiorag.providers.deepgram_stt.DeepgramClient") as _mock_client:
            from audiorag.providers.deepgram_stt import DeepgramSTTProvider  # noqa: PLC0415

            provider = DeepgramSTTProvider(
                api_key="test-key", retry_config=retry_config
            )
            assert isinstance(provider, STTProvider)

    def test_assemblyai_stt_provider_protocol(self, retry_config):
        """Test that AssemblyAISTTProvider satisfies STTProvider protocol."""
        with patch("audiorag.providers.assemblyai_stt.assemblyai") as _mock_aai:
            from audiorag.providers.assemblyai_stt import AssemblyAISTTProvider  # noqa: PLC0415

            provider = AssemblyAISTTProvider(
                api_key="test-key", retry_config=retry_config
            )
            assert isinstance(provider, STTProvider)

    def test_voyage_embedding_provider_protocol(self, retry_config):
        """Test that VoyageEmbeddingProvider satisfies EmbeddingProvider protocol."""
        with patch("audiorag.providers.voyage_embeddings.voyageai") as _mock_voyage:
            from audiorag.providers.voyage_embeddings import (  # noqa: PLC0415
                VoyageEmbeddingProvider,
            )

            provider = VoyageEmbeddingProvider(
                api_key="test-key", retry_config=retry_config
            )
            assert isinstance(provider, EmbeddingProvider)

    def test_cohere_embedding_provider_protocol(self, retry_config):
        """Test that CohereEmbeddingProvider satisfies EmbeddingProvider protocol."""
        with patch("audiorag.providers.cohere_embeddings.AsyncClientV2") as _mock_cohere:
            from audiorag.providers.cohere_embeddings import (  # noqa: PLC0415
                CohereEmbeddingProvider,
            )

            provider = CohereEmbeddingProvider(
                api_key="test-key", retry_config=retry_config
            )
            assert isinstance(provider, EmbeddingProvider)

    def test_anthropic_generation_provider_protocol(self, retry_config):
        """Test that AnthropicGenerationProvider satisfies GenerationProvider protocol."""
        with patch(
            "audiorag.providers.anthropic_generation.AsyncAnthropic"
        ) as _mock_anthropic:
            from audiorag.providers.anthropic_generation import (  # noqa: PLC0415
                AnthropicGenerationProvider,
            )

            provider = AnthropicGenerationProvider(
                api_key="test-key", retry_config=retry_config
            )
            assert isinstance(provider, GenerationProvider)

    def test_gemini_generation_provider_protocol(self, retry_config):
        """Test that GeminiGenerationProvider satisfies GenerationProvider protocol."""
        with patch("audiorag.providers.gemini_generation.genai") as _mock_genai:
            from audiorag.providers.gemini_generation import (  # noqa: PLC0415
                GeminiGenerationProvider,
            )

            provider = GeminiGenerationProvider(
                api_key="test-key", retry_config=retry_config
            )
            assert isinstance(provider, GenerationProvider)

    def test_pinecone_vector_store_provider_protocol(self, retry_config):
        """Test that PineconeVectorStore satisfies VectorStoreProvider protocol."""
        with patch("audiorag.providers.pinecone_store.Pinecone") as _mock_pinecone:
            from audiorag.providers.pinecone_store import PineconeVectorStore  # noqa: PLC0415

            provider = PineconeVectorStore(
                api_key="test-key", retry_config=retry_config
            )
            assert isinstance(provider, VectorStoreProvider)

    def test_weaviate_vector_store_provider_protocol(self, retry_config):
        """Test that WeaviateVectorStore satisfies VectorStoreProvider protocol."""
        with patch("audiorag.providers.weaviate_store.weaviate") as _mock_weaviate:
            from audiorag.providers.weaviate_store import WeaviateVectorStore  # noqa: PLC0415

            provider = WeaviateVectorStore(
                url="http://localhost:8080", retry_config=retry_config
            )
            assert isinstance(provider, VectorStoreProvider)


class TestProviderInitialization:
    """Test suite for provider initialization with various configurations."""

    def test_deepgram_provider_default_model(self):
        """Test Deepgram provider with default model."""
        with patch("audiorag.providers.deepgram_stt.DeepgramClient"):
            from audiorag.providers.deepgram_stt import DeepgramSTTProvider  # noqa: PLC0415

            provider = DeepgramSTTProvider(api_key="test-key")
            assert provider.model == "nova-2"

    def test_deepgram_provider_custom_model(self):
        """Test Deepgram provider with custom model."""
        with patch("audiorag.providers.deepgram_stt.DeepgramClient"):
            from audiorag.providers.deepgram_stt import DeepgramSTTProvider  # noqa: PLC0415

            provider = DeepgramSTTProvider(api_key="test-key", model="nova-2-general")
            assert provider.model == "nova-2-general"

    def test_assemblyai_provider_default_model(self):
        """Test AssemblyAI provider with default model."""
        with patch("audiorag.providers.assemblyai_stt.assemblyai"):
            from audiorag.providers.assemblyai_stt import AssemblyAISTTProvider  # noqa: PLC0415

            provider = AssemblyAISTTProvider(api_key="test-key")
            assert provider.model == "best"

    def test_voyage_provider_default_model(self):
        """Test Voyage provider with default model."""
        with patch("audiorag.providers.voyage_embeddings.voyageai"):
            from audiorag.providers.voyage_embeddings import (  # noqa: PLC0415
                VoyageEmbeddingProvider,
            )

            provider = VoyageEmbeddingProvider(api_key="test-key")
            assert provider.model == "voyage-3.5"

    def test_cohere_provider_default_model(self):
        """Test Cohere embedding provider with default model."""
        with patch("audiorag.providers.cohere_embeddings.AsyncClientV2"):
            from audiorag.providers.cohere_embeddings import (  # noqa: PLC0415
                CohereEmbeddingProvider,
            )

            provider = CohereEmbeddingProvider(api_key="test-key")
            assert provider.model == "embed-v4.0"

    def test_anthropic_provider_default_model(self):
        """Test Anthropic provider with default model."""
        with patch("audiorag.providers.anthropic_generation.AsyncAnthropic"):
            from audiorag.providers.anthropic_generation import (  # noqa: PLC0415
                AnthropicGenerationProvider,
            )

            provider = AnthropicGenerationProvider(api_key="test-key")
            assert provider.model == "claude-3-7-sonnet-20250219"

    def test_gemini_provider_default_model(self):
        """Test Gemini provider with default model."""
        with patch("audiorag.providers.gemini_generation.genai"):
            from audiorag.providers.gemini_generation import (  # noqa: PLC0415
                GeminiGenerationProvider,
            )

            provider = GeminiGenerationProvider(api_key="test-key")
            assert provider.model == "gemini-2.0-flash-001"

    def test_pinecone_provider_default_settings(self):
        """Test Pinecone provider with default settings."""
        with patch("audiorag.providers.pinecone_store.Pinecone"):
            from audiorag.providers.pinecone_store import PineconeVectorStore  # noqa: PLC0415

            provider = PineconeVectorStore(api_key="test-key")
            assert provider._index_name == "audiorag"
            assert provider._namespace == "default"

    def test_weaviate_provider_default_settings(self):
        """Test Weaviate provider with default settings."""
        with patch("audiorag.providers.weaviate_store.weaviate"):
            from audiorag.providers.weaviate_store import WeaviateVectorStore  # noqa: PLC0415

            provider = WeaviateVectorStore()
            assert provider._collection_name == "AudioRAG"


class TestProviderFactoryMethods:
    """Test suite for pipeline provider factory methods."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.stt_provider = "deepgram"
        config.embedding_provider = "voyage"
        config.vector_store_provider = "pinecone"
        config.generation_provider = "anthropic"
        config.reranker_provider = "cohere"
        config.deepgram_api_key = "test-deepgram-key"
        config.voyage_api_key = "test-voyage-key"
        config.pinecone_api_key = "test-pinecone-key"
        config.anthropic_api_key = "test-anthropic-key"
        config.cohere_api_key = "test-cohere-key"
        config.pinecone_index_name = "test-index"
        config.pinecone_namespace = "test-namespace"
        config.get_stt_model = MagicMock(return_value="nova-2")
        config.get_embedding_model = MagicMock(return_value="voyage-3.5")
        config.get_generation_model = MagicMock(
            return_value="claude-3-7-sonnet-20250219"
        )
        return config

    @pytest.fixture
    def retry_config(self):
        """Create a retry config for testing."""
        return RetryConfig(max_attempts=1, min_wait_seconds=0.1, max_wait_seconds=1.0)

    def test_create_deepgram_stt_provider(self, mock_config, retry_config):
        """Test factory method creates Deepgram provider."""
        with patch("audiorag.providers.deepgram_stt.DeepgramClient"):
            from audiorag.pipeline import AudioRAGPipeline  # noqa: PLC0415

            pipeline = object.__new__(AudioRAGPipeline)
            provider = pipeline._create_stt_provider(mock_config, retry_config)
            assert provider is not None

    def test_create_assemblyai_stt_provider(self, mock_config, retry_config):
        """Test factory method creates AssemblyAI provider."""
        mock_config.stt_provider = "assemblyai"
        mock_config.assemblyai_api_key = "test-assemblyai-key"

        with patch("audiorag.providers.assemblyai_stt.assemblyai"):
            from audiorag.pipeline import AudioRAGPipeline  # noqa: PLC0415

            pipeline = object.__new__(AudioRAGPipeline)
            provider = pipeline._create_stt_provider(mock_config, retry_config)
            assert provider is not None

    def test_create_voyage_embedding_provider(self, mock_config, retry_config):
        """Test factory method creates Voyage provider."""
        with patch("audiorag.providers.voyage_embeddings.voyageai"):
            from audiorag.pipeline import AudioRAGPipeline  # noqa: PLC0415

            pipeline = object.__new__(AudioRAGPipeline)
            provider = pipeline._create_embedding_provider(mock_config, retry_config)
            assert provider is not None

    def test_create_cohere_embedding_provider(self, mock_config, retry_config):
        """Test factory method creates Cohere embedding provider."""
        mock_config.embedding_provider = "cohere"
        mock_config.cohere_api_key = "test-cohere-key"

        with patch("audiorag.providers.cohere_embeddings.AsyncClientV2"):
            from audiorag.pipeline import AudioRAGPipeline  # noqa: PLC0415

            pipeline = object.__new__(AudioRAGPipeline)
            provider = pipeline._create_embedding_provider(mock_config, retry_config)
            assert provider is not None

    def test_create_pinecone_vector_store(self, mock_config, retry_config):
        """Test factory method creates Pinecone vector store."""
        with patch("audiorag.providers.pinecone_store.Pinecone"):
            from audiorag.pipeline import AudioRAGPipeline  # noqa: PLC0415

            pipeline = object.__new__(AudioRAGPipeline)
            provider = pipeline._create_vector_store_provider(mock_config, retry_config)
            assert provider is not None

    def test_create_weaviate_vector_store(self, mock_config, retry_config):
        """Test factory method creates Weaviate vector store."""
        mock_config.vector_store_provider = "weaviate"
        mock_config.weaviate_url = "http://localhost:8080"

        with patch("audiorag.providers.weaviate_store.weaviate"):
            from audiorag.pipeline import AudioRAGPipeline  # noqa: PLC0415

            pipeline = object.__new__(AudioRAGPipeline)
            provider = pipeline._create_vector_store_provider(mock_config, retry_config)
            assert provider is not None

    def test_create_anthropic_generation_provider(self, mock_config, retry_config):
        """Test factory method creates Anthropic generation provider."""
        with patch("audiorag.providers.anthropic_generation.AsyncAnthropic"):
            from audiorag.pipeline import AudioRAGPipeline  # noqa: PLC0415

            pipeline = object.__new__(AudioRAGPipeline)
            provider = pipeline._create_generation_provider(mock_config, retry_config)
            assert provider is not None

    def test_create_gemini_generation_provider(self, mock_config, retry_config):
        """Test factory method creates Gemini generation provider."""
        mock_config.generation_provider = "gemini"
        mock_config.google_api_key = "test-google-key"

        with patch("audiorag.providers.gemini_generation.genai"):
            from audiorag.pipeline import AudioRAGPipeline  # noqa: PLC0415

            pipeline = object.__new__(AudioRAGPipeline)
            provider = pipeline._create_generation_provider(mock_config, retry_config)
            assert provider is not None
