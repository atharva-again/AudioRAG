"""AudioRAG provider implementations with lazy loading."""

__all__ = [
    "AnthropicGenerationProvider",
    "AssemblyAISTTProvider",
    "AudioSplitter",
    "ChromaDBVectorStore",
    "CohereEmbeddingProvider",
    "CohereReranker",
    "DeepgramSTTProvider",
    "GeminiGenerationProvider",
    "GroqSTTProvider",
    "OpenAIEmbeddingProvider",
    "OpenAIGenerationProvider",
    "OpenAISTTProvider",
    "PassthroughReranker",
    "PineconeVectorStore",
    "SupabasePgVectorStore",
    "VideoInfo",
    "VoyageEmbeddingProvider",
    "WeaviateVectorStore",
    "YouTubeScraper",
]


def __getattr__(name: str) -> object:  # noqa: PLR0911, PLR0912
    """Lazy load providers to avoid import errors when optional deps not installed."""
    # Audio Source
    if name == "YouTubeScraper":
        from .youtube_scraper import YouTubeScraper  # noqa: PLC0415

        return YouTubeScraper
    if name == "AudioSplitter":
        from .audio_splitter import AudioSplitter  # noqa: PLC0415

        return AudioSplitter

    # STT Providers
    if name == "OpenAISTTProvider":
        from .openai_stt import OpenAISTTProvider  # noqa: PLC0415

        return OpenAISTTProvider
    if name == "DeepgramSTTProvider":
        from .deepgram_stt import DeepgramSTTProvider  # noqa: PLC0415

        return DeepgramSTTProvider
    if name == "AssemblyAISTTProvider":
        from .assemblyai_stt import AssemblyAISTTProvider  # noqa: PLC0415

        return AssemblyAISTTProvider
    if name == "GroqSTTProvider":
        from .groq_stt import GroqSTTProvider  # noqa: PLC0415

        return GroqSTTProvider

    # Embedding Providers
    if name == "OpenAIEmbeddingProvider":
        from .openai_embeddings import OpenAIEmbeddingProvider  # noqa: PLC0415

        return OpenAIEmbeddingProvider
    if name == "VoyageEmbeddingProvider":
        from .voyage_embeddings import VoyageEmbeddingProvider  # noqa: PLC0415

        return VoyageEmbeddingProvider
    if name == "CohereEmbeddingProvider":
        from .cohere_embeddings import CohereEmbeddingProvider  # noqa: PLC0415

        return CohereEmbeddingProvider

    # Vector Store Providers
    if name == "ChromaDBVectorStore":
        from .chromadb_store import ChromaDBVectorStore  # noqa: PLC0415

        return ChromaDBVectorStore
    if name == "PineconeVectorStore":
        from .pinecone_store import PineconeVectorStore  # noqa: PLC0415

        return PineconeVectorStore
    if name == "WeaviateVectorStore":
        from .weaviate_store import WeaviateVectorStore  # noqa: PLC0415

        return WeaviateVectorStore
    if name == "SupabasePgVectorStore":
        from .supabase_pgvector import SupabasePgVectorStore  # noqa: PLC0415

        return SupabasePgVectorStore

    # Audio Source
    if name == "VideoInfo":
        from .youtube_scraper import VideoInfo  # noqa: PLC0415

        return VideoInfo

    # Generation Providers
    if name == "OpenAIGenerationProvider":
        from .openai_generation import OpenAIGenerationProvider  # noqa: PLC0415

        return OpenAIGenerationProvider
    if name == "AnthropicGenerationProvider":
        from .anthropic_generation import AnthropicGenerationProvider  # noqa: PLC0415

        return AnthropicGenerationProvider
    if name == "GeminiGenerationProvider":
        from .gemini_generation import GeminiGenerationProvider  # noqa: PLC0415

        return GeminiGenerationProvider

    # Reranker Providers
    if name == "CohereReranker":
        from .cohere_reranker import CohereReranker  # noqa: PLC0415

        return CohereReranker
    if name == "PassthroughReranker":
        from .passthrough_reranker import PassthroughReranker  # noqa: PLC0415

        return PassthroughReranker

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
