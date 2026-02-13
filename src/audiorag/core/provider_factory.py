from __future__ import annotations

from pathlib import Path

from audiorag.core.config import AudioRAGConfig
from audiorag.core.protocols import (
    AudioSourceProvider,
    EmbeddingProvider,
    GenerationProvider,
    RerankerProvider,
    STTProvider,
    VectorStoreProvider,
)
from audiorag.core.retry_config import RetryConfig


def create_audio_source_provider(config: AudioRAGConfig) -> AudioSourceProvider:
    """Create an audio source provider based on config."""
    provider_name = config.audio_source_provider.lower()

    if provider_name == "local":
        from audiorag.source.local import LocalSource

        return LocalSource()

    from audiorag.pipeline import _build_ydl_opts
    from audiorag.source.youtube import YouTubeSource

    archive_path = (
        Path(config.youtube_download_archive) if config.youtube_download_archive else None
    )
    ydl_opts = _build_ydl_opts(config)
    return YouTubeSource(
        download_archive=archive_path,
        ydl_opts=ydl_opts,
    )


def create_stt_provider(config: AudioRAGConfig, retry_config: RetryConfig) -> STTProvider:
    provider_name = config.stt_provider.lower()

    if provider_name == "groq":
        from audiorag.transcribe.groq import GroqTranscriber

        return GroqTranscriber(
            api_key=config.groq_api_key or None,
            model=config.get_stt_model(),
            retry_config=retry_config,
        )
    if provider_name == "deepgram":
        from audiorag.transcribe.deepgram import DeepgramTranscriber

        return DeepgramTranscriber(
            api_key=config.deepgram_api_key or None,
            model=config.get_stt_model(),
            retry_config=retry_config,
        )
    if provider_name == "assemblyai":
        from audiorag.transcribe.assemblyai import AssemblyAITranscriber

        return AssemblyAITranscriber(
            api_key=config.assemblyai_api_key or None,
            model=config.get_stt_model(),
            retry_config=retry_config,
        )
    from audiorag.transcribe.openai import OpenAITranscriber

    return OpenAITranscriber(
        api_key=config.openai_api_key or None,
        model=config.stt_model,
        retry_config=retry_config,
    )


def create_embedding_provider(
    config: AudioRAGConfig, retry_config: RetryConfig
) -> EmbeddingProvider:
    provider_name = config.embedding_provider.lower()

    if provider_name == "voyage":
        from audiorag.embed.voyage import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider(
            api_key=config.voyage_api_key or None,
            model=config.get_embedding_model(),
            retry_config=retry_config,
        )
    if provider_name == "cohere":
        from audiorag.embed.cohere import CohereEmbeddingProvider

        return CohereEmbeddingProvider(
            api_key=config.cohere_api_key or None,
            model=config.get_embedding_model(),
            retry_config=retry_config,
        )
    from audiorag.embed.openai import OpenAIEmbeddingProvider

    return OpenAIEmbeddingProvider(
        api_key=config.openai_api_key or None,
        model=config.embedding_model,
        retry_config=retry_config,
    )


def create_vector_store_provider(
    config: AudioRAGConfig, retry_config: RetryConfig
) -> VectorStoreProvider:
    provider_name = config.vector_store_provider.lower()

    if provider_name == "supabase":
        try:
            from audiorag.store.supabase import SupabasePgVectorStore
        except ImportError as e:
            raise ImportError(
                "supabase vector store requires 'supabase' package. "
                "Install with: pip install audiorag[supabase]"
            ) from e
        return SupabasePgVectorStore(
            connection_string=config.supabase_connection_string or "",
            collection_name=config.supabase_collection_name or "audiorag",
            dimension=config.supabase_vector_dimension,
            retry_config=retry_config,
        )
    if provider_name == "pinecone":
        try:
            from audiorag.store.pinecone import PineconeVectorStore
        except ImportError as e:
            raise ImportError(
                "pinecone vector store requires 'pinecone-client' package. "
                "Install with: pip install audiorag[pinecone]"
            ) from e
        return PineconeVectorStore(
            api_key=config.pinecone_api_key or "",
            index_name=config.pinecone_index_name or "audiorag",
            namespace=config.pinecone_namespace or "default",
            retry_config=retry_config,
        )
    if provider_name == "weaviate":
        try:
            from audiorag.store.weaviate import WeaviateVectorStore
        except ImportError as e:
            raise ImportError(
                "weaviate vector store requires 'weaviate-client' package. "
                "Install with: pip install audiorag[weaviate]"
            ) from e
        return WeaviateVectorStore(
            url=config.weaviate_url or None,
            api_key=config.weaviate_api_key or None,
            collection_name=config.weaviate_collection_name or "AudioRAG",
            retry_config=retry_config,
        )
    try:
        from audiorag.store.chromadb import ChromaDBVectorStore
    except ImportError as e:
        raise ImportError(
            "chromadb vector store requires 'chromadb' package. "
            "Install with: pip install audiorag[chromadb]"
        ) from e
    return ChromaDBVectorStore(
        persist_directory=config.chromadb_persist_directory or "./chroma_db",
        collection_name=config.chromadb_collection_name or "audiorag",
        retry_config=retry_config,
    )


def create_generation_provider(
    config: AudioRAGConfig, retry_config: RetryConfig
) -> GenerationProvider:
    provider_name = config.generation_provider.lower()

    if provider_name == "anthropic":
        from audiorag.generate.anthropic import AnthropicGenerator

        return AnthropicGenerator(
            api_key=config.anthropic_api_key or None,
            model=config.generation_model or "claude-3-7-sonnet-20250219",
            retry_config=retry_config,
        )
    if provider_name == "gemini":
        from audiorag.generate.gemini import GeminiGenerator

        return GeminiGenerator(
            api_key=config.google_api_key or None,
            model=config.generation_model or "gemini-2.0-flash-001",
            retry_config=retry_config,
        )
    from audiorag.generate.openai import OpenAIGenerator

    return OpenAIGenerator(
        api_key=config.openai_api_key or None,
        model=config.generation_model or "gpt-4o-mini",
        retry_config=retry_config,
    )


def create_reranker_provider(config: AudioRAGConfig, retry_config: RetryConfig) -> RerankerProvider:
    provider_name = config.reranker_provider.lower()

    if provider_name == "passthrough" or not config.cohere_api_key:
        from audiorag.rerank.passthrough import PassthroughReranker

        return PassthroughReranker()

    from audiorag.rerank.cohere import CohereReranker

    return CohereReranker(
        api_key=config.cohere_api_key or None,
        model=config.reranker_model or "rerank-v3.5",
        retry_config=retry_config,
    )
