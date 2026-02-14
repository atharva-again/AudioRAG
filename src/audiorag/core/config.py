"""Configuration management for AudioRAG using pydantic-settings."""

from __future__ import annotations

import os
import platform
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from audiorag.core.config import AdvancedConfig


def _get_default_work_dir() -> Path:
    """Get platform-appropriate default work directory for audio caching."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    else:  # Linux and other POSIX
        base = Path.home() / ".cache"
    return base / "audiorag"


class AdvancedConfig(BaseSettings):
    """Advanced configuration - rarely changed but available.

    Access via config.advanced.* or AUDIORAG_ADVANCED_* env vars.
    """

    model_config = SettingsConfigDict(env_prefix="AUDIORAG_ADVANCED_", extra="ignore")

    youtube_download_archive: str | None = None
    youtube_concurrent_fragments: int = 3
    youtube_skip_after_errors: int = 3
    youtube_batch_size: int = 100
    youtube_max_concurrent: int = 3
    youtube_cookie_file: str | None = None
    youtube_cookies_from_browser: str | None = None
    youtube_po_token: str | None = None
    youtube_visitor_data: str | None = None
    youtube_data_sync_id: str | None = None
    youtube_impersonate: str | None = "chrome-120"
    youtube_player_clients: list[str] = ["tv", "web", "mweb"]
    js_runtime: str | None = "deno"

    pinecone_index_name: str = "audiorag"
    pinecone_namespace: str = "default"
    weaviate_collection_name: str = "AudioRAG"
    chromadb_persist_directory: str = "./chroma_db"
    chromadb_collection_name: str = "audiorag"
    supabase_collection_name: str = "audiorag"
    supabase_vector_dimension: int = 1536

    stt_model: str = "whisper-1"
    stt_language: str | None = None
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    reranker_model: str = "rerank-v3.5"

    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    cleanup_audio: bool = True

    log_level: str = "INFO"
    log_format: str = "colored"
    log_timestamps: bool = True

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
    vector_id_format: Literal["auto", "sha256", "uuid5"] = "auto"
    vector_id_uuid5_namespace: str | None = None

    @field_validator("vector_id_uuid5_namespace")
    @classmethod
    def _validate_uuid_ns(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return str(uuid.UUID(v))


_ADVANCED_KEYS: set[str] = set(AdvancedConfig.model_fields)


class AudioRAGConfig(BaseSettings):
    """AudioRAG main configuration.

    Top-level settings (most common):
    - Providers: stt, embedding, vector_store, generation, reranker
    - API Keys: openai_api_key, deepgram_api_key, etc.
    - Core: chunk_duration, audio_format, database_path, work_dir

    Advanced settings via config.advanced.* or AUDIORAG_ADVANCED_* env vars.
    Backward compatibility: old flat keys (e.g. config.stt_model) also work.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUDIORAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    audio_source_provider: str = "youtube"
    stt_provider: str = "openai"
    embedding_provider: str = "openai"
    vector_store_provider: str = "chromadb"
    generation_provider: str = "openai"
    reranker_provider: str = "cohere"

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

    chunk_duration_seconds: int = 30
    audio_format: str = "mp3"
    audio_split_max_size_mb: int = 24
    database_path: str = "audiorag.db"
    work_dir: Path = Field(default_factory=_get_default_work_dir)

    youtube_download_archive: str | None = None
    youtube_concurrent_fragments: int = 3
    youtube_skip_after_errors: int = 3
    youtube_batch_size: int = 100
    youtube_max_concurrent: int = 3
    youtube_cookie_file: str | None = None
    youtube_cookies_from_browser: str | None = None
    youtube_po_token: str | None = None
    youtube_visitor_data: str | None = None
    youtube_data_sync_id: str | None = None
    youtube_impersonate: str | None = "chrome-120"
    youtube_player_clients: list[str] = ["tv", "web", "mweb"]
    js_runtime: str | None = "deno"

    pinecone_index_name: str = "audiorag"
    pinecone_namespace: str = "default"
    weaviate_collection_name: str = "AudioRAG"
    chromadb_persist_directory: str = "./chroma_db"
    chromadb_collection_name: str = "audiorag"
    supabase_collection_name: str = "audiorag"
    supabase_vector_dimension: int = 1536

    stt_model: str = "whisper-1"
    stt_language: str | None = None
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    reranker_model: str = "rerank-v3.5"

    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    cleanup_audio: bool = True

    log_level: str = "INFO"
    log_format: str = "colored"
    log_timestamps: bool = True

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
    vector_id_format: Literal["auto", "sha256", "uuid5"] = "auto"
    vector_id_uuid5_namespace: str | None = None

    @field_validator("vector_id_uuid5_namespace")
    @classmethod
    def _validate_uuid_ns(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return str(uuid.UUID(v))

    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)

    @model_validator(mode="after")
    def _sync_to_advanced(self) -> AudioRAGConfig:
        for key in _ADVANCED_KEYS:
            val = getattr(self, key, None)
            setattr(self.advanced, key, val)
        return self

    def get_stt_model(self) -> str:
        provider = self.stt_provider.lower()
        if provider == "groq":
            return "whisper"
        if provider == "deepgram":
            return "nova-2"
        if provider == "assemblyai":
            return "best"
        return self.advanced.stt_model

    def get_embedding_model(self) -> str:
        return self.advanced.embedding_model

    def get_generation_model(self) -> str:
        return self.advanced.generation_model
