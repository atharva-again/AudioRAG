"""Configuration management for AudioRAG using pydantic-settings."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioRAGConfig(BaseSettings):
    """AudioRAG configuration with environment variable support.

    All settings use the AUDIORAG_ env prefix. Provider-specific model
    defaults are handled by provider constructors, not this config.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUDIORAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Provider Selection --
    stt_provider: str = "openai"
    embedding_provider: str = "openai"
    vector_store_provider: str = "chromadb"
    generation_provider: str = "openai"
    reranker_provider: str = "cohere"

    # -- API Keys --
    openai_api_key: str = ""
    deepgram_api_key: str = ""
    assemblyai_api_key: str = ""
    groq_api_key: str = ""
    voyage_api_key: str = ""
    cohere_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    pinecone_api_key: str = ""
    weaviate_url: str = ""
    weaviate_api_key: str = ""
    supabase_connection_string: str = ""
    supabase_collection_name: str = "audiorag"
    supabase_vector_dimension: int = 1536

    # -- Vector Store Settings --
    pinecone_index_name: str = "audiorag"
    pinecone_namespace: str = "default"
    weaviate_collection_name: str = "AudioRAG"
    chromadb_persist_directory: str = "./chroma_db"
    chromadb_collection_name: str = "audiorag"

    # -- Database and Storage --
    database_path: str = "audiorag.db"
    work_dir: Path | None = None

    # -- YouTube Scraping --
    youtube_download_archive: str | None = None
    youtube_concurrent_fragments: int = 3
    youtube_skip_after_errors: int = 3
    youtube_batch_size: int = 100
    youtube_max_concurrent: int = 3
    youtube_cookie_file: str | None = None
    youtube_po_token: str | None = None
    youtube_impersonate: str | None = "chrome"
    youtube_player_clients: list[str] = ["tv", "web", "mweb"]
    js_runtime: str | None = "deno:~/.deno/bin/deno"

    # -- Audio Processing --
    chunk_duration_seconds: int = 30
    audio_format: str = "mp3"
    audio_split_max_size_mb: int = 24

    # -- Model Configuration --
    stt_model: str = "whisper-1"
    stt_language: str | None = None
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    reranker_model: str = "rerank-v3.5"

    # -- Retrieval Settings --
    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    cleanup_audio: bool = True

    # -- Logging --
    log_level: str = "INFO"
    log_format: str = "colored"
    log_timestamps: bool = True

    # -- Retry Configuration --
    retry_max_attempts: int = 3
    retry_min_wait_seconds: float = 4.0
    retry_max_wait_seconds: float = 60.0
    retry_exponential_multiplier: float = 1.0

    budget_enabled: bool = False
    budget_rpm: int | None = None
    budget_tpm: int | None = None
    budget_audio_seconds_per_hour: int | None = None
    budget_token_chars_per_token: int = 4
    budget_provider_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    vector_store_verify_mode: Literal["off", "best_effort", "strict"] = "best_effort"
    vector_store_verify_max_attempts: int = 5
    vector_store_verify_wait_seconds: float = 0.5

    # -- Model Getter Methods (for backward compatibility) --
    def get_stt_model(self) -> str:
        """Get STT model based on provider."""
        provider = self.stt_provider.lower()
        if provider == "groq":
            return "whisper"
        if provider == "deepgram":
            return "nova-2"
        if provider == "assemblyai":
            return "best"
        return self.stt_model

    def get_embedding_model(self) -> str:
        """Get embedding model based on provider."""
        return self.embedding_model

    def get_generation_model(self) -> str:
        """Get generation model based on provider."""
        return self.generation_model
