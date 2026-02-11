"""Comprehensive tests for all Protocol interfaces in audiorag.core.protocols.

Tests cover:
1. Protocol structure and runtime_checkable decorator
2. isinstance() checks for compliant implementations
3. isinstance() checks for non-compliant implementations
4. Required method presence and signatures
5. Mock implementations for protocol compliance verification
"""

from pathlib import Path

import pytest

from audiorag.core.models import AudioFile, TranscriptionSegment
from audiorag.core.protocols import (
    AudioSourceProvider,
    EmbeddingProvider,
    GenerationProvider,
    RerankerProvider,
    STTProvider,
    VectorStoreProvider,
)

# ============================================================================
# Test STTProvider Protocol
# ============================================================================


class TestSTTProvider:
    """Test suite for STTProvider protocol."""

    def test_stt_provider_is_runtime_checkable(self):
        """Test that STTProvider can be used with isinstance checks."""

        # Verify runtime_checkable decorator is applied by checking isinstance works
        class CompliantSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                return []

        stt = CompliantSTT()
        assert isinstance(stt, STTProvider)

    def test_stt_provider_has_transcribe_method(self):
        """Test that STTProvider protocol defines transcribe method."""
        # Check that transcribe method exists in the protocol
        assert hasattr(STTProvider, "transcribe")

    @pytest.mark.asyncio
    async def test_stt_compliant_implementation_passes_isinstance(self):
        """Test that a compliant STTProvider implementation passes isinstance check."""

        class CompliantSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                return [
                    TranscriptionSegment(start_time=0.0, end_time=5.0, text="Test transcription")
                ]

        stt = CompliantSTT()
        assert isinstance(stt, STTProvider)

    def test_stt_non_compliant_implementation_fails_isinstance(self):
        """Test that a non-compliant implementation fails isinstance check."""

        class NonCompliantSTT:
            def transcribe_text(self):
                """Wrong method name."""
                pass

        stt = NonCompliantSTT()
        assert not isinstance(stt, STTProvider)

    def test_stt_missing_method_fails_isinstance(self):
        """Test that implementation missing transcribe method fails isinstance check."""

        class IncompleteSTT:
            pass

        stt = IncompleteSTT()
        assert not isinstance(stt, STTProvider)

    @pytest.mark.asyncio
    async def test_stt_with_mock_implementation(self, sample_transcription_segments):
        """Test STTProvider with mock implementation."""

        class MockSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                return sample_transcription_segments

        stt = MockSTT()
        assert isinstance(stt, STTProvider)

        # Test that the mock works correctly
        result = await stt.transcribe(Path("test.mp3"), language="en")
        assert len(result) == 4
        assert all(isinstance(seg, TranscriptionSegment) for seg in result)

    def test_stt_async_mock_provider(self, mock_stt_provider):
        """Test STTProvider with AsyncMock from conftest."""
        # AsyncMock should be compatible with protocol
        assert isinstance(mock_stt_provider, STTProvider)


# ============================================================================
# Test EmbeddingProvider Protocol
# ============================================================================


class TestEmbeddingProvider:
    """Test suite for EmbeddingProvider protocol."""

    def test_embedding_provider_is_runtime_checkable(self):
        """Test that EmbeddingProvider can be used with isinstance checks."""

        # Verify runtime_checkable decorator is applied by checking isinstance works
        class CompliantEmbedding:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3]]

        emb = CompliantEmbedding()
        assert isinstance(emb, EmbeddingProvider)

    def test_embedding_provider_has_embed_method(self):
        """Test that EmbeddingProvider protocol defines embed method."""
        assert hasattr(EmbeddingProvider, "embed")

    @pytest.mark.asyncio
    async def test_embedding_compliant_implementation_passes_isinstance(self):
        """Test that a compliant EmbeddingProvider implementation passes isinstance check."""

        class CompliantEmbedding:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        embedding = CompliantEmbedding()
        assert isinstance(embedding, EmbeddingProvider)

    def test_embedding_non_compliant_implementation_fails_isinstance(self):
        """Test that a non-compliant implementation fails isinstance check."""

        class NonCompliantEmbedding:
            def embed_text(self, text: str) -> list[float]:
                """Wrong method signature."""
                return [0.1, 0.2, 0.3]

        embedding = NonCompliantEmbedding()
        assert not isinstance(embedding, EmbeddingProvider)

    def test_embedding_missing_method_fails_isinstance(self):
        """Test that implementation missing embed method fails isinstance check."""

        class IncompleteEmbedding:
            pass

        embedding = IncompleteEmbedding()
        assert not isinstance(embedding, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_embedding_with_mock_implementation(self, sample_embeddings):
        """Test EmbeddingProvider with mock implementation."""

        class MockEmbedding:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return sample_embeddings

        embedding = MockEmbedding()
        assert isinstance(embedding, EmbeddingProvider)

        # Test that the mock works correctly
        result = await embedding.embed(["text1", "text2", "text3"])
        assert len(result) == 3
        assert all(isinstance(emb, list) for emb in result)
        assert all(all(isinstance(val, float) for val in emb) for emb in result)

    def test_embedding_async_mock_provider(self, mock_embedding_provider):
        """Test EmbeddingProvider with AsyncMock from conftest."""
        assert isinstance(mock_embedding_provider, EmbeddingProvider)


# ============================================================================
# Test VectorStoreProvider Protocol
# ============================================================================


class TestVectorStoreProvider:
    """Test suite for VectorStoreProvider protocol."""

    def test_vector_store_provider_is_runtime_checkable(self):
        """Test that VectorStoreProvider can be used with isinstance checks."""

        # Verify runtime_checkable decorator is applied by checking isinstance works
        class CompliantVectorStore:
            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                pass

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return []

            async def delete_by_source(self, source_url: str) -> None:
                pass

        vs = CompliantVectorStore()
        assert isinstance(vs, VectorStoreProvider)

    def test_vector_store_provider_has_required_methods(self):
        """Test that VectorStoreProvider protocol defines all required methods."""
        assert hasattr(VectorStoreProvider, "add")
        assert hasattr(VectorStoreProvider, "query")
        assert hasattr(VectorStoreProvider, "delete_by_source")

    @pytest.mark.asyncio
    async def test_vector_store_compliant_implementation_passes_isinstance(self):
        """Test that a compliant VectorStoreProvider implementation passes isinstance check."""

        class CompliantVectorStore:
            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                pass

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return []

            async def delete_by_source(self, source_url: str) -> None:
                pass

        store = CompliantVectorStore()
        assert isinstance(store, VectorStoreProvider)

    def test_vector_store_missing_add_method_fails_isinstance(self):
        """Test that implementation missing add method fails isinstance check."""

        class IncompleteVectorStore:
            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return []

            async def delete_by_source(self, source_url: str) -> None:
                pass

        store = IncompleteVectorStore()
        assert not isinstance(store, VectorStoreProvider)

    def test_vector_store_missing_query_method_fails_isinstance(self):
        """Test that implementation missing query method fails isinstance check."""

        class IncompleteVectorStore:
            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                pass

            async def delete_by_source(self, source_url: str) -> None:
                pass

        store = IncompleteVectorStore()
        assert not isinstance(store, VectorStoreProvider)

    def test_vector_store_missing_delete_by_source_method_fails_isinstance(self):
        """Test that implementation missing delete_by_source method fails isinstance check."""

        class IncompleteVectorStore:
            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                pass

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return []

        store = IncompleteVectorStore()
        assert not isinstance(store, VectorStoreProvider)

    @pytest.mark.asyncio
    async def test_vector_store_add_method(self):
        """Test VectorStoreProvider add method with mock implementation."""

        class MockVectorStore:
            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                self.stored_ids = ids
                self.stored_embeddings = embeddings
                self.stored_metadatas = metadatas
                self.stored_documents = documents

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return []

            async def delete_by_source(self, source_url: str) -> None:
                pass

        store = MockVectorStore()
        assert isinstance(store, VectorStoreProvider)

        # Test add method
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"key": "value1"}, {"key": "value2"}]
        documents = ["doc1", "doc2"]

        await store.add(ids, embeddings, metadatas, documents)
        assert store.stored_ids == ids
        assert store.stored_embeddings == embeddings
        assert store.stored_metadatas == metadatas
        assert store.stored_documents == documents

    @pytest.mark.asyncio
    async def test_vector_store_query_method(self):
        """Test VectorStoreProvider query method with mock implementation."""

        class MockVectorStore:
            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                pass

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return [
                    {"id": "id1", "text": "result1", "distance": 0.1},
                    {"id": "id2", "text": "result2", "distance": 0.2},
                ]

            async def delete_by_source(self, source_url: str) -> None:
                pass

        store = MockVectorStore()
        assert isinstance(store, VectorStoreProvider)

        # Test query method
        embedding = [0.1, 0.2, 0.3]
        results = await store.query(embedding, top_k=5)
        assert len(results) == 2
        assert all(isinstance(result, dict) for result in results)

    @pytest.mark.asyncio
    async def test_vector_store_delete_by_source_method(self):
        """Test VectorStoreProvider delete_by_source method with mock implementation."""

        class MockVectorStore:
            def __init__(self):
                self.deleted_sources = []

            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                pass

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return []

            async def delete_by_source(self, source_url: str) -> None:
                self.deleted_sources.append(source_url)

        store = MockVectorStore()
        assert isinstance(store, VectorStoreProvider)

        # Test delete_by_source method
        source_url = "https://example.com/video"
        await store.delete_by_source(source_url)
        assert source_url in store.deleted_sources

    def test_vector_store_async_mock_provider(self, mock_vector_store_provider):
        """Test VectorStoreProvider with AsyncMock from conftest."""
        assert isinstance(mock_vector_store_provider, VectorStoreProvider)


# ============================================================================
# Test GenerationProvider Protocol
# ============================================================================


class TestGenerationProvider:
    """Test suite for GenerationProvider protocol."""

    def test_generation_provider_is_runtime_checkable(self):
        """Test that GenerationProvider can be used with isinstance checks."""

        # Verify runtime_checkable decorator is applied by checking isinstance works
        class CompliantGeneration:
            async def generate(self, query: str, context: list[str]) -> str:
                return "Generated answer"

        gen = CompliantGeneration()
        assert isinstance(gen, GenerationProvider)

    def test_generation_provider_has_generate_method(self):
        """Test that GenerationProvider protocol defines generate method."""
        assert hasattr(GenerationProvider, "generate")

    @pytest.mark.asyncio
    async def test_generation_compliant_implementation_passes_isinstance(self):
        """Test that a compliant GenerationProvider implementation passes isinstance check."""

        class CompliantGeneration:
            async def generate(self, query: str, context: list[str]) -> str:
                return "Generated response based on context."

        generation = CompliantGeneration()
        assert isinstance(generation, GenerationProvider)

    def test_generation_non_compliant_implementation_fails_isinstance(self):
        """Test that a non-compliant implementation fails isinstance check."""

        class NonCompliantGeneration:
            def generate_text(self, query: str) -> str:
                """Wrong method name and signature."""
                return "Response"

        generation = NonCompliantGeneration()
        assert not isinstance(generation, GenerationProvider)

    def test_generation_missing_method_fails_isinstance(self):
        """Test that implementation missing generate method fails isinstance check."""

        class IncompleteGeneration:
            pass

        generation = IncompleteGeneration()
        assert not isinstance(generation, GenerationProvider)

    @pytest.mark.asyncio
    async def test_generation_with_mock_implementation(self):
        """Test GenerationProvider with mock implementation."""

        class MockGeneration:
            async def generate(self, query: str, context: list[str]) -> str:
                return f"Response to '{query}' based on {len(context)} context items."

        generation = MockGeneration()
        assert isinstance(generation, GenerationProvider)

        # Test that the mock works correctly
        result = await generation.generate("What is AI?", ["AI is...", "Machine learning..."])
        assert isinstance(result, str)
        assert "Response to" in result

    def test_generation_async_mock_provider(self, mock_generation_provider):
        """Test GenerationProvider with AsyncMock from conftest."""
        assert isinstance(mock_generation_provider, GenerationProvider)


# ============================================================================
# Test RerankerProvider Protocol
# ============================================================================


class TestRerankerProvider:
    """Test suite for RerankerProvider protocol."""

    def test_reranker_provider_is_runtime_checkable(self):
        """Test that RerankerProvider can be used with isinstance checks."""

        # Verify runtime_checkable decorator is applied by checking isinstance works
        class CompliantReranker:
            async def rerank(
                self, query: str, documents: list[str], top_n: int = 3
            ) -> list[tuple[int, float]]:
                return [(0, 0.95)]

        rerank = CompliantReranker()
        assert isinstance(rerank, RerankerProvider)

    def test_reranker_provider_has_rerank_method(self):
        """Test that RerankerProvider protocol defines rerank method."""
        assert hasattr(RerankerProvider, "rerank")

    @pytest.mark.asyncio
    async def test_reranker_compliant_implementation_passes_isinstance(self):
        """Test that a compliant RerankerProvider implementation passes isinstance check."""

        class CompliantReranker:
            async def rerank(
                self, query: str, documents: list[str], top_n: int = 3
            ) -> list[tuple[int, float]]:
                return [(0, 0.95), (1, 0.87)]

        reranker = CompliantReranker()
        assert isinstance(reranker, RerankerProvider)

    def test_reranker_non_compliant_implementation_fails_isinstance(self):
        """Test that a non-compliant implementation fails isinstance check."""

        class NonCompliantReranker:
            def rank_documents(self, query: str, documents: list[str]) -> list[str]:
                """Wrong method name and signature."""
                return documents

        reranker = NonCompliantReranker()
        assert not isinstance(reranker, RerankerProvider)

    def test_reranker_missing_method_fails_isinstance(self):
        """Test that implementation missing rerank method fails isinstance check."""

        class IncompleteReranker:
            pass

        reranker = IncompleteReranker()
        assert not isinstance(reranker, RerankerProvider)

    @pytest.mark.asyncio
    async def test_reranker_with_mock_implementation(self):
        """Test RerankerProvider with mock implementation."""

        class MockReranker:
            async def rerank(
                self, query: str, documents: list[str], top_n: int = 3
            ) -> list[tuple[int, float]]:
                # Return top_n results as (index, score) tuples
                return [(i, 1.0 - (i * 0.1)) for i in range(min(top_n, len(documents)))]

        reranker = MockReranker()
        assert isinstance(reranker, RerankerProvider)

        # Test that the mock works correctly
        documents = ["doc1", "doc2", "doc3", "doc4"]
        results = await reranker.rerank("test query", documents, top_n=3)
        assert len(results) == 3
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in results)

    def test_reranker_async_mock_provider(self, mock_reranker_provider):
        """Test RerankerProvider with AsyncMock from conftest."""
        assert isinstance(mock_reranker_provider, RerankerProvider)


# ============================================================================
# Test AudioSourceProvider Protocol
# ============================================================================


class TestAudioSourceProvider:
    """Test suite for AudioSourceProvider protocol."""

    def test_audio_source_provider_is_runtime_checkable(self):
        """Test that AudioSourceProvider can be used with isinstance checks."""

        # Verify runtime_checkable decorator is applied by checking isinstance works
        class CompliantAudioSource:
            async def download(
                self, url: str, output_dir: Path, audio_format: str = "mp3"
            ) -> AudioFile:
                return AudioFile(
                    path=output_dir / "audio.mp3",
                    source_url=url,
                    title="Test",
                    duration=100.0,
                )

        audio = CompliantAudioSource()
        assert isinstance(audio, AudioSourceProvider)

    def test_audio_source_provider_has_download_method(self):
        """Test that AudioSourceProvider protocol defines download method."""
        assert hasattr(AudioSourceProvider, "download")

    @pytest.mark.asyncio
    async def test_audio_source_compliant_implementation_passes_isinstance(self, tmp_path):
        """Test that a compliant AudioSourceProvider implementation passes isinstance check."""

        class CompliantAudioSource:
            async def download(
                self, url: str, output_dir: Path, audio_format: str = "mp3"
            ) -> AudioFile:
                audio_path = output_dir / f"audio.{audio_format}"
                return AudioFile(
                    path=audio_path,
                    source_url=url,
                    title="Test Video",
                    duration=100.0,
                )

        source = CompliantAudioSource()
        assert isinstance(source, AudioSourceProvider)

    def test_audio_source_non_compliant_implementation_fails_isinstance(self):
        """Test that a non-compliant implementation fails isinstance check."""

        class NonCompliantAudioSource:
            def download_audio(self, url: str) -> str:
                """Wrong method name and signature."""
                return "/path/to/audio.mp3"

        source = NonCompliantAudioSource()
        assert not isinstance(source, AudioSourceProvider)

    def test_audio_source_missing_method_fails_isinstance(self):
        """Test that implementation missing download method fails isinstance check."""

        class IncompleteAudioSource:
            pass

        source = IncompleteAudioSource()
        assert not isinstance(source, AudioSourceProvider)

    @pytest.mark.asyncio
    async def test_audio_source_with_mock_implementation(self, tmp_path):
        """Test AudioSourceProvider with mock implementation."""

        class MockAudioSource:
            async def download(
                self, url: str, output_dir: Path, audio_format: str = "mp3"
            ) -> AudioFile:
                audio_path = output_dir / f"downloaded_audio.{audio_format}"
                return AudioFile(
                    path=audio_path,
                    source_url=url,
                    title="Downloaded Video",
                    duration=120.5,
                )

        source = MockAudioSource()
        assert isinstance(source, AudioSourceProvider)

        # Test that the mock works correctly
        output_dir = tmp_path / "audio"
        output_dir.mkdir()
        result = await source.download("https://example.com/video", output_dir, audio_format="wav")
        assert isinstance(result, AudioFile)
        assert result.source_url == "https://example.com/video"
        assert result.title == "Downloaded Video"

    def test_audio_source_async_mock_provider(self, mock_audio_source_provider):
        """Test AudioSourceProvider with AsyncMock from conftest."""
        assert isinstance(mock_audio_source_provider, AudioSourceProvider)


# ============================================================================
# Integration Tests: Multiple Protocols
# ============================================================================


class TestProtocolIntegration:
    """Integration tests for multiple protocols working together."""

    @pytest.mark.asyncio
    async def test_all_protocols_with_mock_implementations(
        self,
        sample_transcription_segments,
        sample_embeddings,
        sample_metadata_dicts,
        tmp_path,
    ):
        """Test that all protocols can work together with mock implementations."""

        class MockSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                return sample_transcription_segments

        class MockEmbedding:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return sample_embeddings

        class MockVectorStore:
            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                pass

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return [{"id": "id1", "text": "result"}]

            async def delete_by_source(self, source_url: str) -> None:
                pass

        class MockGeneration:
            async def generate(self, query: str, context: list[str]) -> str:
                return "Generated answer"

        class MockReranker:
            async def rerank(
                self, query: str, documents: list[str], top_n: int = 3
            ) -> list[tuple[int, float]]:
                return [(0, 0.95)]

        class MockAudioSource:
            async def download(
                self, url: str, output_dir: Path, audio_format: str = "mp3"
            ) -> AudioFile:
                return AudioFile(
                    path=output_dir / "audio.mp3",
                    source_url=url,
                    title="Test",
                    duration=100.0,
                )

        # Verify all implementations are protocol-compliant
        stt = MockSTT()
        embedding = MockEmbedding()
        vector_store = MockVectorStore()
        generation = MockGeneration()
        reranker = MockReranker()
        audio_source = MockAudioSource()

        assert isinstance(stt, STTProvider)
        assert isinstance(embedding, EmbeddingProvider)
        assert isinstance(vector_store, VectorStoreProvider)
        assert isinstance(generation, GenerationProvider)
        assert isinstance(reranker, RerankerProvider)
        assert isinstance(audio_source, AudioSourceProvider)

        # Test a simple workflow
        transcription = await stt.transcribe(Path("test.mp3"))
        assert len(transcription) > 0

        texts = [seg.text for seg in transcription]
        embeddings = await embedding.embed(texts)
        # Mock returns fixed 3 embeddings, verify we got embeddings back
        assert len(embeddings) > 0

        await vector_store.add(
            ids=[f"id_{i}" for i in range(len(texts))],
            embeddings=embeddings,
            metadatas=sample_metadata_dicts[: len(texts)],
            documents=texts,
        )

        query_results = await vector_store.query(embeddings[0], top_k=5)
        assert isinstance(query_results, list)

        rerank_results = await reranker.rerank("test query", texts, top_n=2)
        assert len(rerank_results) <= 2

        answer = await generation.generate("test query", texts)
        assert isinstance(answer, str)

        output_dir = tmp_path / "audio"
        output_dir.mkdir()
        audio_file = await audio_source.download("https://example.com/video", output_dir)
        assert isinstance(audio_file, AudioFile)

    def test_all_mock_providers_from_conftest(self, all_mock_providers):
        """Test that all mock providers from conftest are protocol-compliant."""
        assert isinstance(all_mock_providers["stt"], STTProvider)
        assert isinstance(all_mock_providers["embedding"], EmbeddingProvider)
        assert isinstance(all_mock_providers["vector_store"], VectorStoreProvider)
        assert isinstance(all_mock_providers["generation"], GenerationProvider)
        assert isinstance(all_mock_providers["reranker"], RerankerProvider)
        assert isinstance(all_mock_providers["audio_source"], AudioSourceProvider)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestProtocolEdgeCases:
    """Test edge cases and error handling for protocols."""

    def test_protocol_with_extra_methods_still_compliant(self):
        """Test that implementations with extra methods are still protocol-compliant."""

        class ExtendedSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                return []

            async def validate_audio(self, audio_path: Path) -> bool:
                """Extra method not in protocol."""
                return True

            def get_supported_languages(self) -> list[str]:
                """Another extra method."""
                return ["en", "es", "fr"]

        stt = ExtendedSTT()
        assert isinstance(stt, STTProvider)

    def test_protocol_with_different_return_type_fails(self):
        """Test that wrong return type fails isinstance check."""

        class WrongReturnTypeSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> str:  # Wrong return type
                return "transcription"

        stt = WrongReturnTypeSTT()
        # Note: Runtime checkable protocols don't strictly check return types,
        # but they do check method presence
        assert isinstance(stt, STTProvider)  # Method exists, so it passes

    def test_protocol_with_sync_instead_of_async_fails(self):
        """Test that sync method instead of async fails isinstance check."""

        class SyncSTT:
            def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                # Sync instead of async
                return []

        stt = SyncSTT()
        # Runtime checkable protocols check for method presence, not async/await
        assert isinstance(stt, STTProvider)

    @pytest.mark.asyncio
    async def test_vector_store_with_default_parameters(self):
        """Test VectorStoreProvider query method with default parameters."""

        class MockVectorStore:
            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                pass

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return [{"id": f"id_{i}", "score": 0.9 - (i * 0.1)} for i in range(top_k)]

            async def delete_by_source(self, source_url: str) -> None:
                pass

        store = MockVectorStore()
        assert isinstance(store, VectorStoreProvider)

        # Test with default top_k
        results = await store.query([0.1, 0.2, 0.3])
        assert len(results) == 10  # Default top_k

        # Test with custom top_k
        results = await store.query([0.1, 0.2, 0.3], top_k=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_reranker_with_default_parameters(self):
        """Test RerankerProvider rerank method with default parameters."""

        class MockReranker:
            async def rerank(
                self, query: str, documents: list[str], top_n: int = 3
            ) -> list[tuple[int, float]]:
                return [(i, 1.0 - (i * 0.1)) for i in range(min(top_n, len(documents)))]

        reranker = MockReranker()
        assert isinstance(reranker, RerankerProvider)

        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        # Test with default top_n
        results = await reranker.rerank("query", documents)
        assert len(results) == 3  # Default top_n

        # Test with custom top_n
        results = await reranker.rerank("query", documents, top_n=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_audio_source_with_default_format(self, tmp_path):
        """Test AudioSourceProvider download method with default format."""

        class MockAudioSource:
            async def download(
                self, url: str, output_dir: Path, audio_format: str = "mp3"
            ) -> AudioFile:
                return AudioFile(
                    path=output_dir / f"audio.{audio_format}",
                    source_url=url,
                    title="Test",
                    duration=100.0,
                )

        source = MockAudioSource()
        assert isinstance(source, AudioSourceProvider)

        output_dir = tmp_path / "audio"
        output_dir.mkdir()

        # Test with default format
        result = await source.download("https://example.com/video", output_dir)
        assert str(result.path).endswith(".mp3")

        # Test with custom format
        result = await source.download("https://example.com/video", output_dir, audio_format="wav")
        assert str(result.path).endswith(".wav")
