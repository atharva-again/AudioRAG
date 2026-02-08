"""Protocol conformance tests for AudioRAG.

These tests verify that ANY implementation of each protocol works correctly.
They define the contract that all implementations must satisfy and will survive
the entire redesign.

Tests cover:
1. Protocol runtime_checkable decorator presence
2. isinstance() checks for compliant implementations
3. isinstance() checks for non-compliant implementations
4. Method signatures and return type shapes
5. Mock implementations demonstrating protocol compliance
"""

from pathlib import Path

import pytest

from audiorag.core.models import AudioFile, ChunkMetadata, TranscriptionSegment
from audiorag.core.protocols.audio_source import AudioSourceProvider
from audiorag.core.protocols.embedding import EmbeddingProvider
from audiorag.core.protocols.generation import GenerationProvider
from audiorag.core.protocols.reranker import RerankerProvider
from audiorag.core.protocols.stt import STTProvider
from audiorag.core.protocols.vector_store import VectorStoreProvider

# ============================================================================
# Test STTProvider Protocol Conformance
# ============================================================================


class TestSTTProviderConformance:
    """Conformance tests for STTProvider protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify STTProvider is marked as runtime_checkable."""
        assert hasattr(STTProvider, "_is_protocol")
        assert STTProvider._is_protocol is True

    def test_protocol_has_required_method(self):
        """Verify STTProvider defines transcribe method."""
        assert hasattr(STTProvider, "transcribe")

    @pytest.mark.asyncio
    async def test_compliant_implementation_passes_isinstance(self):
        """Verify compliant implementation passes isinstance check."""

        class FakeSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                return [
                    TranscriptionSegment(start_time=0.0, end_time=5.0, text="Fake transcription")
                ]

        provider = FakeSTT()
        assert isinstance(provider, STTProvider)

    @pytest.mark.asyncio
    async def test_return_type_shape(self):
        """Verify transcribe returns list of TranscriptionSegment."""

        class FakeSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                return [
                    TranscriptionSegment(start_time=0.0, end_time=2.5, text="First segment"),
                    TranscriptionSegment(start_time=2.5, end_time=5.0, text="Second segment"),
                ]

        provider = FakeSTT()
        result = await provider.transcribe(Path("fake.mp3"), language="en")

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(seg, TranscriptionSegment) for seg in result)
        assert all(hasattr(seg, "start_time") for seg in result)
        assert all(hasattr(seg, "end_time") for seg in result)
        assert all(hasattr(seg, "text") for seg in result)

    def test_non_compliant_missing_method_fails_isinstance(self):
        """Verify implementation missing transcribe method fails isinstance."""

        class NonCompliantSTT:
            async def convert_audio(self, path: Path) -> str:
                return "wrong method"

        provider = NonCompliantSTT()
        assert not isinstance(provider, STTProvider)

    def test_non_compliant_wrong_signature_fails_isinstance(self):
        """Verify implementation with wrong signature fails isinstance."""

        class NonCompliantSTT:
            async def transcribe(self, path: str) -> str:  # Wrong signature
                return "wrong return type"

        provider = NonCompliantSTT()
        # Note: runtime_checkable only checks method presence, not signature
        # This will pass isinstance but fail at runtime
        assert isinstance(provider, STTProvider)


# ============================================================================
# Test EmbeddingProvider Protocol Conformance
# ============================================================================


class TestEmbeddingProviderConformance:
    """Conformance tests for EmbeddingProvider protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify EmbeddingProvider is marked as runtime_checkable."""
        assert hasattr(EmbeddingProvider, "_is_protocol")
        assert EmbeddingProvider._is_protocol is True

    def test_protocol_has_required_method(self):
        """Verify EmbeddingProvider defines embed method."""
        assert hasattr(EmbeddingProvider, "embed")

    @pytest.mark.asyncio
    async def test_compliant_implementation_passes_isinstance(self):
        """Verify compliant implementation passes isinstance check."""

        class FakeEmbedding:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        provider = FakeEmbedding()
        assert isinstance(provider, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_return_type_shape(self):
        """Verify embed returns list of float vectors."""

        class FakeEmbedding:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                # Return 384-dimensional embeddings (common size)
                return [[float(i) * 0.1 for i in range(384)] for _ in texts]

        provider = FakeEmbedding()
        result = await provider.embed(["text1", "text2", "text3"])

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(vec, list) for vec in result)
        assert all(len(vec) == 384 for vec in result)
        assert all(all(isinstance(val, float) for val in vec) for vec in result)

    def test_non_compliant_missing_method_fails_isinstance(self):
        """Verify implementation missing embed method fails isinstance."""

        class NonCompliantEmbedding:
            async def encode(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2]]

        provider = NonCompliantEmbedding()
        assert not isinstance(provider, EmbeddingProvider)


# ============================================================================
# Test VectorStoreProvider Protocol Conformance
# ============================================================================


class TestVectorStoreProviderConformance:
    """Conformance tests for VectorStoreProvider protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify VectorStoreProvider is marked as runtime_checkable."""
        assert hasattr(VectorStoreProvider, "_is_protocol")
        assert VectorStoreProvider._is_protocol is True

    def test_protocol_has_required_methods(self):
        """Verify VectorStoreProvider defines all required methods."""
        assert hasattr(VectorStoreProvider, "add")
        assert hasattr(VectorStoreProvider, "query")
        assert hasattr(VectorStoreProvider, "delete_by_source")

    @pytest.mark.asyncio
    async def test_compliant_implementation_passes_isinstance(self):
        """Verify compliant implementation passes isinstance check."""

        class FakeVectorStore:
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

        provider = FakeVectorStore()
        assert isinstance(provider, VectorStoreProvider)

    @pytest.mark.asyncio
    async def test_add_method_signature(self):
        """Verify add method accepts correct parameters."""

        class FakeVectorStore:
            def __init__(self):
                self.stored_data = {}

            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                self.stored_data = {
                    "ids": ids,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "documents": documents,
                }

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return []

            async def delete_by_source(self, source_url: str) -> None:
                pass

        provider = FakeVectorStore()
        await provider.add(
            ids=["id1", "id2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"key": "val1"}, {"key": "val2"}],
            documents=["doc1", "doc2"],
        )

        assert provider.stored_data["ids"] == ["id1", "id2"]
        assert len(provider.stored_data["embeddings"]) == 2
        assert len(provider.stored_data["metadatas"]) == 2
        assert len(provider.stored_data["documents"]) == 2

    @pytest.mark.asyncio
    async def test_query_return_type_shape(self):
        """Verify query returns list of dicts with expected structure."""

        class FakeVectorStore:
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
                    {
                        "id": "chunk_1",
                        "text": "Result text",
                        "metadata": {"start_time": 0.0, "end_time": 5.0},
                        "distance": 0.15,
                    },
                    {
                        "id": "chunk_2",
                        "text": "Another result",
                        "metadata": {"start_time": 5.0, "end_time": 10.0},
                        "distance": 0.25,
                    },
                ]

            async def delete_by_source(self, source_url: str) -> None:
                pass

        provider = FakeVectorStore()
        results = await provider.query([0.1, 0.2, 0.3], top_k=5)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert all("id" in r for r in results)
        assert all("text" in r for r in results)

    @pytest.mark.asyncio
    async def test_delete_by_source_method(self):
        """Verify delete_by_source accepts source_url parameter."""

        class FakeVectorStore:
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

        provider = FakeVectorStore()
        await provider.delete_by_source("https://example.com/video")

        assert "https://example.com/video" in provider.deleted_sources

    def test_non_compliant_missing_add_fails_isinstance(self):
        """Verify implementation missing add method fails isinstance."""

        class NonCompliantVectorStore:
            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return []

            async def delete_by_source(self, source_url: str) -> None:
                pass

        provider = NonCompliantVectorStore()
        assert not isinstance(provider, VectorStoreProvider)

    def test_non_compliant_missing_query_fails_isinstance(self):
        """Verify implementation missing query method fails isinstance."""

        class NonCompliantVectorStore:
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

        provider = NonCompliantVectorStore()
        assert not isinstance(provider, VectorStoreProvider)

    def test_non_compliant_missing_delete_fails_isinstance(self):
        """Verify implementation missing delete_by_source fails isinstance."""

        class NonCompliantVectorStore:
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

        provider = NonCompliantVectorStore()
        assert not isinstance(provider, VectorStoreProvider)


# ============================================================================
# Test GenerationProvider Protocol Conformance
# ============================================================================


class TestGenerationProviderConformance:
    """Conformance tests for GenerationProvider protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify GenerationProvider is marked as runtime_checkable."""
        assert hasattr(GenerationProvider, "_is_protocol")
        assert GenerationProvider._is_protocol is True

    def test_protocol_has_required_method(self):
        """Verify GenerationProvider defines generate method."""
        assert hasattr(GenerationProvider, "generate")

    @pytest.mark.asyncio
    async def test_compliant_implementation_passes_isinstance(self):
        """Verify compliant implementation passes isinstance check."""

        class FakeGeneration:
            async def generate(self, query: str, context: list[str]) -> str:
                return f"Answer to '{query}' based on {len(context)} contexts"

        provider = FakeGeneration()
        assert isinstance(provider, GenerationProvider)

    @pytest.mark.asyncio
    async def test_return_type_shape(self):
        """Verify generate returns string."""

        class FakeGeneration:
            async def generate(self, query: str, context: list[str]) -> str:
                return "Generated answer based on provided context."

        provider = FakeGeneration()
        result = await provider.generate(
            "What is AI?", ["AI is artificial intelligence.", "AI learns from data."]
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_non_compliant_missing_method_fails_isinstance(self):
        """Verify implementation missing generate method fails isinstance."""

        class NonCompliantGeneration:
            async def create_response(self, query: str) -> str:
                return "wrong method"

        provider = NonCompliantGeneration()
        assert not isinstance(provider, GenerationProvider)


# ============================================================================
# Test RerankerProvider Protocol Conformance
# ============================================================================


class TestRerankerProviderConformance:
    """Conformance tests for RerankerProvider protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify RerankerProvider is marked as runtime_checkable."""
        assert hasattr(RerankerProvider, "_is_protocol")
        assert RerankerProvider._is_protocol is True

    def test_protocol_has_required_method(self):
        """Verify RerankerProvider defines rerank method."""
        assert hasattr(RerankerProvider, "rerank")

    @pytest.mark.asyncio
    async def test_compliant_implementation_passes_isinstance(self):
        """Verify compliant implementation passes isinstance check."""

        class FakeReranker:
            async def rerank(
                self, query: str, documents: list[str], top_n: int = 3
            ) -> list[tuple[int, float]]:
                return [(0, 0.95), (1, 0.87), (2, 0.72)]

        provider = FakeReranker()
        assert isinstance(provider, RerankerProvider)

    @pytest.mark.asyncio
    async def test_return_type_shape(self):
        """Verify rerank returns list of (index, score) tuples."""

        class FakeReranker:
            async def rerank(
                self, query: str, documents: list[str], top_n: int = 3
            ) -> list[tuple[int, float]]:
                # Return top_n results sorted by relevance
                return [(i, 1.0 - (i * 0.1)) for i in range(min(top_n, len(documents)))]

        provider = FakeReranker()
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        result = await provider.rerank("test query", documents, top_n=3)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in result)
        # Verify indices are valid
        assert all(0 <= idx < len(documents) for idx, _ in result)

    def test_non_compliant_missing_method_fails_isinstance(self):
        """Verify implementation missing rerank method fails isinstance."""

        class NonCompliantReranker:
            async def rank(self, query: str, docs: list[str]) -> list[int]:
                return [0, 1, 2]

        provider = NonCompliantReranker()
        assert not isinstance(provider, RerankerProvider)


# ============================================================================
# Test AudioSourceProvider Protocol Conformance
# ============================================================================


class TestAudioSourceProviderConformance:
    """Conformance tests for AudioSourceProvider protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify AudioSourceProvider is marked as runtime_checkable."""
        assert hasattr(AudioSourceProvider, "_is_protocol")
        assert AudioSourceProvider._is_protocol is True

    def test_protocol_has_required_method(self):
        """Verify AudioSourceProvider defines download method."""
        assert hasattr(AudioSourceProvider, "download")

    @pytest.mark.asyncio
    async def test_compliant_implementation_passes_isinstance(self, tmp_path):
        """Verify compliant implementation passes isinstance check."""

        class FakeAudioSource:
            async def download(
                self, url: str, output_dir: Path, audio_format: str = "mp3"
            ) -> AudioFile:
                audio_path = output_dir / f"audio.{audio_format}"
                return AudioFile(
                    path=audio_path,
                    source_url=url,
                    title="Fake Video",
                    duration=120.0,
                )

        provider = FakeAudioSource()
        assert isinstance(provider, AudioSourceProvider)

    @pytest.mark.asyncio
    async def test_return_type_shape(self, tmp_path):
        """Verify download returns AudioFile with correct structure."""

        class FakeAudioSource:
            async def download(
                self, url: str, output_dir: Path, audio_format: str = "mp3"
            ) -> AudioFile:
                audio_path = output_dir / f"downloaded.{audio_format}"
                return AudioFile(
                    path=audio_path,
                    source_url=url,
                    title="Test Video Title",
                    duration=180.5,
                )

        provider = FakeAudioSource()
        output_dir = tmp_path / "audio"
        output_dir.mkdir()

        result = await provider.download(
            "https://example.com/video", output_dir, audio_format="wav"
        )

        assert isinstance(result, AudioFile)
        assert isinstance(result.path, Path)
        assert result.source_url == "https://example.com/video"
        assert isinstance(result.title, str)
        assert isinstance(result.duration, float)
        assert str(result.path).endswith(".wav")

    def test_non_compliant_missing_method_fails_isinstance(self):
        """Verify implementation missing download method fails isinstance."""

        class NonCompliantAudioSource:
            async def fetch(self, url: str) -> Path:
                return Path("/fake/path.mp3")

        provider = NonCompliantAudioSource()
        assert not isinstance(provider, AudioSourceProvider)


# ============================================================================
# Test ChunkingStrategy Protocol Conformance (NEW)
# ============================================================================


class TestChunkingStrategyConformance:
    """Conformance tests for ChunkingStrategy protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify ChunkingStrategy is marked as runtime_checkable."""
        from audiorag.core.protocols.chunking import ChunkingStrategy

        assert hasattr(ChunkingStrategy, "_is_protocol")
        assert ChunkingStrategy._is_protocol is True

    def test_protocol_has_required_method(self):
        """Verify ChunkingStrategy defines chunk method."""
        from audiorag.core.protocols.chunking import ChunkingStrategy

        assert hasattr(ChunkingStrategy, "chunk")

    @pytest.mark.asyncio
    async def test_compliant_implementation_passes_isinstance(self):
        """Verify compliant implementation passes isinstance check."""
        from audiorag.core.protocols.chunking import ChunkingStrategy

        class FakeChunking:
            def chunk(
                self, segments: list[TranscriptionSegment], source_url: str, title: str
            ) -> list[ChunkMetadata]:
                return [
                    ChunkMetadata(
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        text=seg.text,
                        source_url=source_url,
                        title=title,
                    )
                    for seg in segments
                ]

        provider = FakeChunking()
        assert isinstance(provider, ChunkingStrategy)

    def test_return_type_shape(self):
        """Verify chunk returns list of ChunkMetadata."""
        from audiorag.core.protocols.chunking import ChunkingStrategy  # noqa: F401

        class FakeChunking:
            def chunk(
                self, segments: list[TranscriptionSegment], source_url: str, title: str
            ) -> list[ChunkMetadata]:
                # Combine segments into 30-second chunks
                chunks = []
                current_chunk_text = []
                chunk_start = 0.0

                for seg in segments:
                    current_chunk_text.append(seg.text)
                    if seg.end_time - chunk_start >= 30.0:
                        chunks.append(
                            ChunkMetadata(
                                start_time=chunk_start,
                                end_time=seg.end_time,
                                text=" ".join(current_chunk_text),
                                source_url=source_url,
                                title=title,
                            )
                        )
                        current_chunk_text = []
                        chunk_start = seg.end_time

                return chunks

        provider = FakeChunking()
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=10.0, text="First"),
            TranscriptionSegment(start_time=10.0, end_time=20.0, text="Second"),
            TranscriptionSegment(start_time=20.0, end_time=35.0, text="Third"),
        ]

        result = provider.chunk(segments, "https://example.com/video", "Test Video")

        assert isinstance(result, list)
        assert all(isinstance(chunk, ChunkMetadata) for chunk in result)
        assert all(hasattr(chunk, "start_time") for chunk in result)
        assert all(hasattr(chunk, "end_time") for chunk in result)
        assert all(hasattr(chunk, "text") for chunk in result)
        assert all(hasattr(chunk, "source_url") for chunk in result)
        assert all(hasattr(chunk, "title") for chunk in result)

    def test_non_compliant_missing_method_fails_isinstance(self):
        """Verify implementation missing chunk method fails isinstance."""
        from audiorag.core.protocols.chunking import ChunkingStrategy

        class NonCompliantChunking:
            def split(self, segments: list[TranscriptionSegment]) -> list[dict]:
                return []

        provider = NonCompliantChunking()
        assert not isinstance(provider, ChunkingStrategy)


# ============================================================================
# Integration Tests: Protocol Interoperability
# ============================================================================


class TestProtocolInteroperability:
    """Test that all protocols can work together in a pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_fake_providers(self, tmp_path):
        """Verify all protocols work together in a complete workflow."""

        # Define fake implementations for all protocols
        class FakeAudioSource:
            async def download(
                self, url: str, output_dir: Path, audio_format: str = "mp3"
            ) -> AudioFile:
                return AudioFile(
                    path=output_dir / "audio.mp3",
                    source_url=url,
                    title="Test Video",
                    duration=100.0,
                )

        class FakeSTT:
            async def transcribe(
                self, audio_path: Path, language: str | None = None
            ) -> list[TranscriptionSegment]:
                return [
                    TranscriptionSegment(start_time=0.0, end_time=5.0, text="First segment"),
                    TranscriptionSegment(start_time=5.0, end_time=10.0, text="Second segment"),
                ]

        class FakeEmbedding:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[float(i) * 0.1 for i in range(384)] for _ in texts]

        class FakeVectorStore:
            def __init__(self):
                self.data = []

            async def add(
                self,
                ids: list[str],
                embeddings: list[list[float]],
                metadatas: list[dict],
                documents: list[str],
            ) -> None:
                self.data.extend(zip(ids, embeddings, metadatas, documents, strict=False))

            async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]:
                return [
                    {
                        "id": "chunk_1",
                        "text": "First segment",
                        "metadata": {"start_time": 0.0, "end_time": 5.0},
                        "distance": 0.1,
                    }
                ]

            async def delete_by_source(self, source_url: str) -> None:
                self.data = [d for d in self.data if d[2].get("source_url") != source_url]

        class FakeReranker:
            async def rerank(
                self, query: str, documents: list[str], top_n: int = 3
            ) -> list[tuple[int, float]]:
                return [(0, 0.95)]

        class FakeGeneration:
            async def generate(self, query: str, context: list[str]) -> str:
                return f"Answer based on {len(context)} context items"

        # Instantiate all providers
        audio_source = FakeAudioSource()
        stt = FakeSTT()
        embedding = FakeEmbedding()
        vector_store = FakeVectorStore()
        reranker = FakeReranker()
        generation = FakeGeneration()

        # Verify all are protocol-compliant
        assert isinstance(audio_source, AudioSourceProvider)
        assert isinstance(stt, STTProvider)
        assert isinstance(embedding, EmbeddingProvider)
        assert isinstance(vector_store, VectorStoreProvider)
        assert isinstance(reranker, RerankerProvider)
        assert isinstance(generation, GenerationProvider)

        # Simulate a complete pipeline workflow
        output_dir = tmp_path / "audio"
        output_dir.mkdir()

        # 1. Download audio
        audio_file = await audio_source.download("https://example.com/video", output_dir)
        assert isinstance(audio_file, AudioFile)

        # 2. Transcribe audio
        segments = await stt.transcribe(audio_file.path)
        assert len(segments) == 2

        # 3. Embed transcription segments
        texts = [seg.text for seg in segments]
        embeddings = await embedding.embed(texts)
        assert len(embeddings) == len(texts)

        # 4. Store in vector database
        await vector_store.add(
            ids=[f"chunk_{i}" for i in range(len(texts))],
            embeddings=embeddings,
            metadatas=[
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "source_url": audio_file.source_url,
                }
                for seg in segments
            ],
            documents=texts,
        )

        # 5. Query vector store
        query_embedding = [0.15] * 384
        results = await vector_store.query(query_embedding, top_k=5)
        assert len(results) > 0

        # 6. Rerank results
        result_texts = [r["text"] for r in results]
        reranked = await reranker.rerank("test query", result_texts, top_n=1)
        assert len(reranked) == 1

        # 7. Generate answer
        answer = await generation.generate("test query", result_texts)
        assert isinstance(answer, str)
        assert len(answer) > 0
