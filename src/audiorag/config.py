"""Configuration management for AudioRAG using pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioRAGConfig(BaseSettings):
    """AudioRAG configuration with environment variable support.

    Environment Variables (all prefixed with AUDIORAG_):
        Provider Selection:
            STT_PROVIDER: openai, deepgram, assemblyai, groq
            EMBEDDING_PROVIDER: openai, voyage, cohere
            VECTOR_STORE_PROVIDER: chromadb, pinecone, weaviate, supabase
            GENERATION_PROVIDER: openai, anthropic, gemini

        STT Models:
            Groq: whisper-large-v3 (default)
            Deepgram: nova-2 (default), nova-2-general, nova-2-meeting,
                      nova-2-phonecall, nova-2-voicemail, nova-2-finance,
                      nova-2-conversationalai, nova-2-video, nova-2-medical,
                      nova-1, enhanced, base
            AssemblyAI: best (default), nano, universal
            OpenAI: whisper-1 (default)

        Embedding Models:
            Voyage: voyage-3.5 (default), voyage-3.5-lite, voyage-4,
                    voyage-4-lite, voyage-4-large, voyage-4-nano
            Cohere: embed-v4.0 (default), embed-v3.0, embed-english-v3.0,
                    embed-multilingual-v3.0, embed-english-light-v3.0,
                    embed-multilingual-light-v3.0
            OpenAI: text-embedding-3-small (default), text-embedding-3-large,
                    text-embedding-ada-002

        Generation Models:
            Anthropic: claude-3-7-sonnet-20250219 (default), claude-3-7-opus-20250219,
                       claude-3-7-haiku-20250219, claude-3-5-sonnet-20241022,
                       claude-3-5-haiku-20241022
            Gemini: gemini-2.0-flash-001 (default), gemini-2.0-pro-001,
                    gemini-2.0-ultra-001, gemini-2.0-flash-lite-001
            OpenAI: gpt-4o-mini (default), gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo
    """

    model_config = SettingsConfigDict(
        env_prefix="AUDIORAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================================================================
    # Provider Selection
    # =========================================================================
    stt_provider: str = "openai"  # openai | deepgram | assemblyai | groq
    embedding_provider: str = "openai"  # openai | voyage | cohere
    vector_store_provider: str = "chromadb"  # chromadb | pinecone | weaviate | supabase
    generation_provider: str = "openai"  # openai | anthropic | gemini
    reranker_provider: str = "cohere"  # cohere | passthrough

    # =========================================================================
    # API Keys
    # =========================================================================
    # OpenAI (used for STT, embeddings, and generation)
    openai_api_key: str = ""

    # Deepgram (STT)
    deepgram_api_key: str = ""

    # AssemblyAI (STT)
    assemblyai_api_key: str = ""

    # Groq (STT)
    groq_api_key: str = ""

    # Voyage AI (embeddings)
    voyage_api_key: str = ""

    # Cohere (embeddings and reranking)
    cohere_api_key: str = ""

    # Anthropic (generation)
    anthropic_api_key: str = ""

    # Google (Gemini generation)
    google_api_key: str = ""

    # Pinecone (vector store)
    pinecone_api_key: str = ""

    # Weaviate (vector store)
    weaviate_url: str = ""
    weaviate_api_key: str = ""

    # Supabase (vector store)
    supabase_connection_string: str = ""
    supabase_collection_name: str = "audiorag"
    supabase_vector_dimension: int = 1536

    # =========================================================================
    # Vector Store Settings
    # =========================================================================
    pinecone_index_name: str = "audiorag"
    pinecone_namespace: str = "default"
    weaviate_collection_name: str = "AudioRAG"
    chromadb_persist_directory: str = "./chroma_db"
    chromadb_collection_name: str = "audiorag"

    # =========================================================================
    # Database and Storage
    # =========================================================================
    database_path: str = "audiorag.db"
    work_dir: Path | None = None

    # =========================================================================
    # YouTube Scraping (Large-Scale)
    # =========================================================================
    # Download archive file tracks already processed videos (resumable scraping)
    youtube_download_archive: str | None = None
    # Concurrent fragments per download (higher = faster but more bandwidth)
    youtube_concurrent_fragments: int = 3
    # Skip playlist after N consecutive errors
    youtube_skip_after_errors: int = 3
    # Batch size for channel scraping (videos per batch)
    youtube_batch_size: int = 100
    # Max concurrent downloads within a batch
    youtube_max_concurrent: int = 3

    # =========================================================================
    # Audio Processing
    # =========================================================================
    chunk_duration_seconds: int = 300
    audio_format: str = "mp3"
    audio_split_max_size_mb: int = 24

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # STT Models:
    #   - Groq: "whisper-large-v3"
    #   - Deepgram: "nova-2", "nova-2-general", "nova-2-meeting", "nova-2-phonecall",
    #               "nova-2-voicemail", "nova-2-finance", "nova-2-conversationalai",
    #               "nova-2-video", "nova-2-medical", "nova-1", "enhanced", "base"
    #   - AssemblyAI: "best", "nano", "universal"
    #   - OpenAI: "whisper-1"
    stt_model: str = "whisper-1"
    stt_language: str | None = None  # e.g., "en", "es", "fr"

    # Embedding Models:
    #   - Voyage: "voyage-3.5", "voyage-3.5-lite", "voyage-4", "voyage-4-lite",
    #             "voyage-4-large", "voyage-4-nano"
    #   - Cohere: "embed-v4.0", "embed-v3.0", "embed-english-v3.0",
    #             "embed-multilingual-v3.0", "embed-english-light-v3.0",
    #             "embed-multilingual-light-v3.0"
    #   - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
    embedding_model: str = "text-embedding-3-small"

    # Generation Models:
    #   - Anthropic: "claude-3-7-sonnet-20250219", "claude-3-7-opus-20250219",
    #                "claude-3-7-haiku-20250219", "claude-3-5-sonnet-20241022",
    #                "claude-3-5-haiku-20241022"
    #   - Gemini: "gemini-2.0-flash-001", "gemini-2.0-pro-001", "gemini-2.0-ultra-001",
    #             "gemini-2.0-flash-lite-001"
    #   - OpenAI: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
    generation_model: str = "gpt-4o-mini"

    # Reranker Models:
    #   - Cohere: "rerank-v3.5", "rerank-v3", "rerank-english-v2.0", "rerank-multilingual-v2.0"
    reranker_model: str = "rerank-v3.5"

    # =========================================================================
    # Retrieval Settings
    # =========================================================================
    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    cleanup_audio: bool = True

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: str = "INFO"
    log_format: str = "colored"
    log_timestamps: bool = True

    # =========================================================================
    # Retry Configuration
    # =========================================================================
    retry_max_attempts: int = 3
    retry_min_wait_seconds: float = 4.0
    retry_max_wait_seconds: float = 60.0
    retry_exponential_multiplier: float = 1.0

    def get_stt_model(self) -> str:
        """Get the appropriate STT model based on provider."""
        if self.stt_provider == "groq":
            return self.stt_model if self.stt_model != "whisper-1" else "whisper-large-v3"
        if self.stt_provider == "deepgram":
            return self.stt_model if self.stt_model != "whisper-1" else "nova-2"
        if self.stt_provider == "assemblyai":
            return self.stt_model if self.stt_model != "whisper-1" else "best"
        return self.stt_model

    def get_embedding_model(self) -> str:
        """Get the appropriate embedding model based on provider."""
        if self.embedding_provider == "voyage":
            return (
                self.embedding_model
                if self.embedding_model != "text-embedding-3-small"
                else "voyage-3.5"
            )
        if self.embedding_provider == "cohere":
            return (
                self.embedding_model
                if self.embedding_model != "text-embedding-3-small"
                else "embed-v4.0"
            )
        return self.embedding_model

    def get_generation_model(self) -> str:
        """Get the appropriate generation model based on provider."""
        if self.generation_provider == "anthropic":
            return (
                self.generation_model
                if self.generation_model != "gpt-4o-mini"
                else "claude-3-7-sonnet-20250219"
            )
        if self.generation_provider == "gemini":
            return (
                self.generation_model
                if self.generation_model != "gpt-4o-mini"
                else "gemini-2.0-flash-001"
            )
        return self.generation_model
