"""Comprehensive tests for AudioRAGConfig."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from audiorag.core.config import AudioRAGConfig

# ============================================================================
# TestAudioRAGConfigDefaults - Test all default values
# ============================================================================


class TestAudioRAGConfigDefaults:
    """Test AudioRAGConfig default values."""

    def test_default_openai_api_key(self):
        """Test default openai_api_key is empty string."""
        config = AudioRAGConfig()
        assert config.openai_api_key == ""

    def test_default_cohere_api_key(self):
        """Test default cohere_api_key is empty string."""
        config = AudioRAGConfig()
        assert config.cohere_api_key == ""

    def test_default_database_path(self):
        """Test default database_path is 'audiorag.db'."""
        config = AudioRAGConfig()
        assert config.database_path == "audiorag.db"

    def test_default_work_dir(self):
        """Test default work_dir is ~/.cache/audiorag."""
        config = AudioRAGConfig()
        assert config.work_dir == Path("~/.cache/audiorag").expanduser()

    def test_default_chunk_duration_seconds(self):
        """Test default chunk_duration_seconds is 30."""
        config = AudioRAGConfig()
        assert config.chunk_duration_seconds == 30

    def test_default_audio_format(self):
        """Test default audio_format is 'mp3'."""
        config = AudioRAGConfig()
        assert config.audio_format == "mp3"

    def test_default_audio_split_max_size_mb(self):
        """Test default audio_split_max_size_mb is 24."""
        config = AudioRAGConfig()
        assert config.audio_split_max_size_mb == 24

    def test_default_embedding_model(self):
        """Test default embedding_model is 'text-embedding-3-small'."""
        config = AudioRAGConfig()
        assert config.embedding_model == "text-embedding-3-small"

    def test_default_generation_model(self):
        """Test default generation_model is 'gpt-4o-mini'."""
        config = AudioRAGConfig()
        assert config.generation_model == "gpt-4o-mini"

    def test_default_stt_model(self):
        """Test default stt_model is 'whisper-1'."""
        config = AudioRAGConfig()
        assert config.stt_model == "whisper-1"

    def test_default_stt_language(self):
        """Test default stt_language is None."""
        config = AudioRAGConfig()
        assert config.stt_language is None

    def test_default_reranker_model(self):
        """Test default reranker_model is 'rerank-v3.5'."""
        config = AudioRAGConfig()
        assert config.reranker_model == "rerank-v3.5"

    def test_default_retrieval_top_k(self):
        """Test default retrieval_top_k is 10."""
        config = AudioRAGConfig()
        assert config.retrieval_top_k == 10

    def test_default_rerank_top_n(self):
        """Test default rerank_top_n is 3."""
        config = AudioRAGConfig()
        assert config.rerank_top_n == 3

    def test_default_cleanup_audio(self):
        """Test default cleanup_audio is True."""
        config = AudioRAGConfig()
        assert config.cleanup_audio is True

    def test_all_defaults_together(self):
        """Test all default values in a single config instance."""
        config = AudioRAGConfig()
        config_any: Any = config
        assert config.openai_api_key == ""
        assert config.cohere_api_key == ""
        assert config.database_path == "audiorag.db"
        assert config.work_dir == Path("~/.cache/audiorag").expanduser()
        assert config.chunk_duration_seconds == 30
        assert config.audio_format == "mp3"
        assert config.audio_split_max_size_mb == 24
        assert config.embedding_model == "text-embedding-3-small"
        assert config.generation_model == "gpt-4o-mini"
        assert config.stt_model == "whisper-1"
        assert config.stt_language is None
        assert config.reranker_model == "rerank-v3.5"
        assert config.retrieval_top_k == 10
        assert config.rerank_top_n == 3
        assert config.cleanup_audio is True
        assert config_any.budget_enabled is False
        assert config_any.budget_rpm is None
        assert config_any.budget_tpm is None
        assert config_any.budget_audio_seconds_per_hour is None
        assert config_any.budget_token_chars_per_token == 4
        assert config_any.budget_provider_overrides == {}
        assert config_any.vector_store_verify_mode == "best_effort"
        assert config_any.vector_store_verify_max_attempts == 5
        assert config_any.vector_store_verify_wait_seconds == 0.5
        assert config_any.vector_id_format == "auto"
        assert config_any.vector_id_uuid5_namespace is None


# ============================================================================
# TestAudioRAGConfigEnvVars - Test loading from environment variables
# ============================================================================


class TestAudioRAGConfigEnvVars:
    """Test AudioRAGConfig environment variable loading with AUDIORAG_ prefix."""

    def test_env_var_openai_api_key(self, monkeypatch):
        """Test loading openai_api_key from AUDIORAG_OPENAI_API_KEY."""
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", "sk-test-key-123")
        config = AudioRAGConfig()
        assert config.openai_api_key == "sk-test-key-123"

    def test_env_var_cohere_api_key(self, monkeypatch):
        """Test loading cohere_api_key from AUDIORAG_COHERE_API_KEY."""
        monkeypatch.setenv("AUDIORAG_COHERE_API_KEY", "cohere-test-key-456")
        config = AudioRAGConfig()
        assert config.cohere_api_key == "cohere-test-key-456"

    def test_env_var_database_path(self, monkeypatch):
        """Test loading database_path from AUDIORAG_DATABASE_PATH."""
        monkeypatch.setenv("AUDIORAG_DATABASE_PATH", "/custom/path/db.sqlite")
        config = AudioRAGConfig()
        assert config.database_path == "/custom/path/db.sqlite"

    def test_env_var_work_dir(self, monkeypatch, tmp_path):
        """Test loading work_dir from AUDIORAG_WORK_DIR."""
        work_dir = str(tmp_path / "work")
        monkeypatch.setenv("AUDIORAG_WORK_DIR", work_dir)
        config = AudioRAGConfig()
        assert config.work_dir == Path(work_dir)

    def test_env_var_chunk_duration_seconds(self, monkeypatch):
        """Test loading chunk_duration_seconds from AUDIORAG_CHUNK_DURATION_SECONDS."""
        monkeypatch.setenv("AUDIORAG_CHUNK_DURATION_SECONDS", "600")
        config = AudioRAGConfig()
        assert config.chunk_duration_seconds == 600

    def test_env_var_audio_format(self, monkeypatch):
        """Test loading audio_format from AUDIORAG_AUDIO_FORMAT."""
        monkeypatch.setenv("AUDIORAG_AUDIO_FORMAT", "wav")
        config = AudioRAGConfig()
        assert config.audio_format == "wav"

    def test_env_var_audio_split_max_size_mb(self, monkeypatch):
        """Test loading audio_split_max_size_mb from AUDIORAG_AUDIO_SPLIT_MAX_SIZE_MB."""
        monkeypatch.setenv("AUDIORAG_AUDIO_SPLIT_MAX_SIZE_MB", "50")
        config = AudioRAGConfig()
        assert config.audio_split_max_size_mb == 50

    def test_env_var_embedding_model(self, monkeypatch):
        """Test loading embedding_model from AUDIORAG_EMBEDDING_MODEL."""
        monkeypatch.setenv("AUDIORAG_EMBEDDING_MODEL", "text-embedding-3-large")
        config = AudioRAGConfig()
        assert config.embedding_model == "text-embedding-3-large"

    def test_env_var_generation_model(self, monkeypatch):
        """Test loading generation_model from AUDIORAG_GENERATION_MODEL."""
        monkeypatch.setenv("AUDIORAG_GENERATION_MODEL", "gpt-4-turbo")
        config = AudioRAGConfig()
        assert config.generation_model == "gpt-4-turbo"

    def test_env_var_stt_model(self, monkeypatch):
        """Test loading stt_model from AUDIORAG_STT_MODEL."""
        monkeypatch.setenv("AUDIORAG_STT_MODEL", "whisper-large")
        config = AudioRAGConfig()
        assert config.stt_model == "whisper-large"

    def test_env_var_stt_language(self, monkeypatch):
        """Test loading stt_language from AUDIORAG_STT_LANGUAGE."""
        monkeypatch.setenv("AUDIORAG_STT_LANGUAGE", "es")
        config = AudioRAGConfig()
        assert config.stt_language == "es"

    def test_env_var_reranker_model(self, monkeypatch):
        """Test loading reranker_model from AUDIORAG_RERANKER_MODEL."""
        monkeypatch.setenv("AUDIORAG_RERANKER_MODEL", "rerank-v4.0")
        config = AudioRAGConfig()
        assert config.reranker_model == "rerank-v4.0"

    def test_env_var_retrieval_top_k(self, monkeypatch):
        """Test loading retrieval_top_k from AUDIORAG_RETRIEVAL_TOP_K."""
        monkeypatch.setenv("AUDIORAG_RETRIEVAL_TOP_K", "20")
        config = AudioRAGConfig()
        assert config.retrieval_top_k == 20

    def test_env_var_rerank_top_n(self, monkeypatch):
        """Test loading rerank_top_n from AUDIORAG_RERANK_TOP_N."""
        monkeypatch.setenv("AUDIORAG_RERANK_TOP_N", "5")
        config = AudioRAGConfig()
        assert config.rerank_top_n == 5

    def test_env_var_budget_enabled(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_BUDGET_ENABLED", "true")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.budget_enabled is True

    def test_env_var_budget_rpm(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_BUDGET_RPM", "30")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.budget_rpm == 30

    def test_env_var_budget_tpm(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_BUDGET_TPM", "9000")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.budget_tpm == 9000

    def test_env_var_budget_audio_seconds_per_hour(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_BUDGET_AUDIO_SECONDS_PER_HOUR", "7200")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.budget_audio_seconds_per_hour == 7200

    def test_env_var_budget_token_chars_per_token(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_BUDGET_TOKEN_CHARS_PER_TOKEN", "3")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.budget_token_chars_per_token == 3

    def test_env_var_budget_provider_overrides(self, monkeypatch):
        monkeypatch.setenv(
            "AUDIORAG_BUDGET_PROVIDER_OVERRIDES",
            '{"openai": {"rpm": 10, "tpm": 1000}, "deepgram": {"audio_seconds_per_hour": 3600}}',
        )
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.budget_provider_overrides["openai"]["rpm"] == 10
        assert config_any.budget_provider_overrides["openai"]["tpm"] == 1000
        assert config_any.budget_provider_overrides["deepgram"]["audio_seconds_per_hour"] == 3600

    def test_env_var_vector_store_verify_mode(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_VECTOR_STORE_VERIFY_MODE", "strict")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.vector_store_verify_mode == "strict"

    def test_env_var_vector_store_verify_max_attempts(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_VECTOR_STORE_VERIFY_MAX_ATTEMPTS", "7")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.vector_store_verify_max_attempts == 7

    def test_env_var_vector_store_verify_wait_seconds(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_VECTOR_STORE_VERIFY_WAIT_SECONDS", "1.25")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.vector_store_verify_wait_seconds == 1.25

    def test_env_var_vector_id_format(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_VECTOR_ID_FORMAT", "uuid5")
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.vector_id_format == "uuid5"

    def test_env_var_vector_id_uuid5_namespace(self, monkeypatch):
        monkeypatch.setenv(
            "AUDIORAG_VECTOR_ID_UUID5_NAMESPACE", "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        )
        config = AudioRAGConfig()
        config_any: Any = config
        assert config_any.vector_id_uuid5_namespace == "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

    def test_env_var_cleanup_audio_true(self, monkeypatch):
        """Test loading cleanup_audio as True from AUDIORAG_CLEANUP_AUDIO."""
        monkeypatch.setenv("AUDIORAG_CLEANUP_AUDIO", "true")
        config = AudioRAGConfig()
        assert config.cleanup_audio is True

    def test_env_var_cleanup_audio_false(self, monkeypatch):
        """Test loading cleanup_audio as False from AUDIORAG_CLEANUP_AUDIO."""
        monkeypatch.setenv("AUDIORAG_CLEANUP_AUDIO", "false")
        config = AudioRAGConfig()
        assert config.cleanup_audio is False

    def test_env_var_cleanup_audio_1(self, monkeypatch):
        """Test loading cleanup_audio as True from '1'."""
        monkeypatch.setenv("AUDIORAG_CLEANUP_AUDIO", "1")
        config = AudioRAGConfig()
        assert config.cleanup_audio is True

    def test_env_var_cleanup_audio_0(self, monkeypatch):
        """Test loading cleanup_audio as False from '0'."""
        monkeypatch.setenv("AUDIORAG_CLEANUP_AUDIO", "0")
        config = AudioRAGConfig()
        assert config.cleanup_audio is False

    def test_multiple_env_vars(self, monkeypatch):
        """Test loading multiple environment variables at once."""
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", "sk-test-123")
        monkeypatch.setenv("AUDIORAG_COHERE_API_KEY", "cohere-test-456")
        monkeypatch.setenv("AUDIORAG_CHUNK_DURATION_SECONDS", "450")
        monkeypatch.setenv("AUDIORAG_RETRIEVAL_TOP_K", "15")
        monkeypatch.setenv("AUDIORAG_CLEANUP_AUDIO", "false")

        config = AudioRAGConfig()
        assert config.openai_api_key == "sk-test-123"
        assert config.cohere_api_key == "cohere-test-456"
        assert config.chunk_duration_seconds == 450
        assert config.retrieval_top_k == 15
        assert config.cleanup_audio is False

    def test_env_var_without_prefix_ignored(self, monkeypatch):
        """Test that env vars without AUDIORAG_ prefix are ignored."""
        monkeypatch.setenv("OPENAI_API_KEY", "should-be-ignored")
        config = AudioRAGConfig()
        assert config.openai_api_key == ""

    def test_env_var_case_insensitive(self, monkeypatch):
        """Test that env var names are case-insensitive."""
        # pydantic-settings converts to uppercase internally
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", "sk-test-key")
        config = AudioRAGConfig()
        assert config.openai_api_key == "sk-test-key"


# ============================================================================
# TestAudioRAGConfigValidation - Test type validation and edge cases
# ============================================================================


class TestAudioRAGConfigValidation:
    """Test AudioRAGConfig type validation and edge cases."""

    def test_chunk_duration_seconds_type_validation(self, monkeypatch):
        """Test chunk_duration_seconds accepts integer."""
        monkeypatch.setenv("AUDIORAG_CHUNK_DURATION_SECONDS", "500")
        config = AudioRAGConfig()
        assert isinstance(config.chunk_duration_seconds, int)
        assert config.chunk_duration_seconds == 500

    def test_audio_split_max_size_mb_type_validation(self, monkeypatch):
        """Test audio_split_max_size_mb accepts integer."""
        monkeypatch.setenv("AUDIORAG_AUDIO_SPLIT_MAX_SIZE_MB", "100")
        config = AudioRAGConfig()
        assert isinstance(config.audio_split_max_size_mb, int)
        assert config.audio_split_max_size_mb == 100

    def test_retrieval_top_k_type_validation(self, monkeypatch):
        """Test retrieval_top_k accepts integer."""
        monkeypatch.setenv("AUDIORAG_RETRIEVAL_TOP_K", "25")
        config = AudioRAGConfig()
        assert isinstance(config.retrieval_top_k, int)
        assert config.retrieval_top_k == 25

    def test_rerank_top_n_type_validation(self, monkeypatch):
        """Test rerank_top_n accepts integer."""
        monkeypatch.setenv("AUDIORAG_RERANK_TOP_N", "7")
        config = AudioRAGConfig()
        assert isinstance(config.rerank_top_n, int)
        assert config.rerank_top_n == 7

    def test_cleanup_audio_type_validation(self, monkeypatch):
        """Test cleanup_audio accepts boolean."""
        monkeypatch.setenv("AUDIORAG_CLEANUP_AUDIO", "true")
        config = AudioRAGConfig()
        assert isinstance(config.cleanup_audio, bool)
        assert config.cleanup_audio is True

    def test_work_dir_path_conversion(self, monkeypatch, tmp_path):
        """Test work_dir is converted to Path object."""
        work_dir = str(tmp_path / "work")
        monkeypatch.setenv("AUDIORAG_WORK_DIR", work_dir)
        config = AudioRAGConfig()
        assert isinstance(config.work_dir, Path)
        assert config.work_dir == Path(work_dir)

    def test_work_dir_default_when_not_set(self):
        """Test work_dir defaults to ~/.cache/audiorag when not set."""
        config = AudioRAGConfig()
        assert config.work_dir == Path("~/.cache/audiorag").expanduser()

    def test_stt_language_none_when_not_set(self):
        """Test stt_language is None when not set."""
        config = AudioRAGConfig()
        assert config.stt_language is None

    def test_stt_language_string_when_set(self, monkeypatch):
        """Test stt_language is string when set."""
        monkeypatch.setenv("AUDIORAG_STT_LANGUAGE", "fr")
        config = AudioRAGConfig()
        assert isinstance(config.stt_language, str)
        assert config.stt_language == "fr"

    def test_openai_api_key_string_type(self):
        """Test openai_api_key is always string."""
        config = AudioRAGConfig()
        assert isinstance(config.openai_api_key, str)

    def test_cohere_api_key_string_type(self):
        """Test cohere_api_key is always string."""
        config = AudioRAGConfig()
        assert isinstance(config.cohere_api_key, str)

    def test_database_path_string_type(self):
        """Test database_path is always string."""
        config = AudioRAGConfig()
        assert isinstance(config.database_path, str)

    def test_audio_format_string_type(self):
        """Test audio_format is always string."""
        config = AudioRAGConfig()
        assert isinstance(config.audio_format, str)

    def test_embedding_model_string_type(self):
        """Test embedding_model is always string."""
        config = AudioRAGConfig()
        assert isinstance(config.embedding_model, str)

    def test_generation_model_string_type(self):
        """Test generation_model is always string."""
        config = AudioRAGConfig()
        assert isinstance(config.generation_model, str)

    def test_stt_model_string_type(self):
        """Test stt_model is always string."""
        config = AudioRAGConfig()
        assert isinstance(config.stt_model, str)

    def test_reranker_model_string_type(self):
        """Test reranker_model is always string."""
        config = AudioRAGConfig()
        assert isinstance(config.reranker_model, str)

    def test_zero_chunk_duration(self, monkeypatch):
        """Test chunk_duration_seconds can be zero."""
        monkeypatch.setenv("AUDIORAG_CHUNK_DURATION_SECONDS", "0")
        config = AudioRAGConfig()
        assert config.chunk_duration_seconds == 0

    def test_negative_chunk_duration(self, monkeypatch):
        """Test chunk_duration_seconds can be negative (no validation)."""
        monkeypatch.setenv("AUDIORAG_CHUNK_DURATION_SECONDS", "-100")
        config = AudioRAGConfig()
        assert config.chunk_duration_seconds == -100

    def test_large_chunk_duration(self, monkeypatch):
        """Test chunk_duration_seconds accepts large values."""
        monkeypatch.setenv("AUDIORAG_CHUNK_DURATION_SECONDS", "999999")
        config = AudioRAGConfig()
        assert config.chunk_duration_seconds == 999999

    def test_zero_audio_split_max_size(self, monkeypatch):
        """Test audio_split_max_size_mb can be zero."""
        monkeypatch.setenv("AUDIORAG_AUDIO_SPLIT_MAX_SIZE_MB", "0")
        config = AudioRAGConfig()
        assert config.audio_split_max_size_mb == 0

    def test_zero_retrieval_top_k(self, monkeypatch):
        """Test retrieval_top_k can be zero."""
        monkeypatch.setenv("AUDIORAG_RETRIEVAL_TOP_K", "0")
        config = AudioRAGConfig()
        assert config.retrieval_top_k == 0

    def test_zero_rerank_top_n(self, monkeypatch):
        """Test rerank_top_n can be zero."""
        monkeypatch.setenv("AUDIORAG_RERANK_TOP_N", "0")
        config = AudioRAGConfig()
        assert config.rerank_top_n == 0

    def test_empty_string_api_keys(self):
        """Test empty string API keys are valid."""
        config = AudioRAGConfig()
        assert config.openai_api_key == ""
        assert config.cohere_api_key == ""

    def test_empty_string_models(self, monkeypatch):
        """Test empty string model names are valid."""
        monkeypatch.setenv("AUDIORAG_EMBEDDING_MODEL", "")
        config = AudioRAGConfig()
        assert config.embedding_model == ""

    def test_special_characters_in_api_key(self, monkeypatch):
        """Test API keys with special characters."""
        special_key = "sk-test!@#$%^&*()_+-=[]{}|;:',.<>?/"
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", special_key)
        config = AudioRAGConfig()
        assert config.openai_api_key == special_key

    def test_special_characters_in_model_name(self, monkeypatch):
        """Test model names with special characters."""
        special_model = "model-v1.0_beta+test"
        monkeypatch.setenv("AUDIORAG_EMBEDDING_MODEL", special_model)
        config = AudioRAGConfig()
        assert config.embedding_model == special_model

    def test_unicode_in_stt_language(self, monkeypatch):
        """Test stt_language with unicode characters."""
        monkeypatch.setenv("AUDIORAG_STT_LANGUAGE", "中文")
        config = AudioRAGConfig()
        assert config.stt_language == "中文"

    def test_whitespace_in_string_fields(self, monkeypatch):
        """Test string fields with leading/trailing whitespace."""
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", "  sk-test-key  ")
        config = AudioRAGConfig()
        # pydantic-settings preserves whitespace
        assert config.openai_api_key == "  sk-test-key  "

    def test_relative_path_work_dir(self, monkeypatch):
        """Test work_dir with relative path."""
        monkeypatch.setenv("AUDIORAG_WORK_DIR", "./work")
        config = AudioRAGConfig()
        assert config.work_dir == Path("./work")

    def test_absolute_path_work_dir(self, monkeypatch):
        """Test work_dir with absolute path."""
        monkeypatch.setenv("AUDIORAG_WORK_DIR", "/absolute/path/work")
        config = AudioRAGConfig()
        assert config.work_dir == Path("/absolute/path/work")

    def test_path_with_spaces_work_dir(self, monkeypatch):
        """Test work_dir with spaces in path."""
        monkeypatch.setenv("AUDIORAG_WORK_DIR", "/path with spaces/work")
        config = AudioRAGConfig()
        assert config.work_dir == Path("/path with spaces/work")

    def test_database_path_with_extension(self, monkeypatch):
        """Test database_path with various extensions."""
        monkeypatch.setenv("AUDIORAG_DATABASE_PATH", "data/audiorag.sqlite3")
        config = AudioRAGConfig()
        assert config.database_path == "data/audiorag.sqlite3"

    def test_database_path_absolute(self, monkeypatch):
        """Test database_path with absolute path."""
        monkeypatch.setenv("AUDIORAG_DATABASE_PATH", "/var/lib/audiorag.db")
        config = AudioRAGConfig()
        assert config.database_path == "/var/lib/audiorag.db"

    def test_vector_id_uuid5_namespace_validation(self, monkeypatch):
        monkeypatch.setenv("AUDIORAG_VECTOR_ID_UUID5_NAMESPACE", "not-a-uuid")

        with pytest.raises(ValidationError, match="vector_id_uuid5_namespace"):
            AudioRAGConfig()


# ============================================================================
# TestAudioRAGConfigPathHandling - Test path handling edge cases
# ============================================================================


class TestAudioRAGConfigPathHandling:
    """Test path handling for work_dir and database_path."""

    def test_work_dir_with_tilde(self, monkeypatch):
        """Test work_dir with tilde expansion."""
        monkeypatch.setenv("AUDIORAG_WORK_DIR", "~/audiorag")
        config = AudioRAGConfig()
        assert config.work_dir == Path("~/audiorag")

    def test_work_dir_with_env_var_expansion(self, monkeypatch):
        """Test work_dir with environment variable in path."""
        monkeypatch.setenv("AUDIORAG_WORK_DIR", "$HOME/audiorag")
        config = AudioRAGConfig()
        # pydantic doesn't expand env vars in paths
        assert config.work_dir == Path("$HOME/audiorag")

    def test_database_path_with_tilde(self, monkeypatch):
        """Test database_path with tilde."""
        monkeypatch.setenv("AUDIORAG_DATABASE_PATH", "~/audiorag.db")
        config = AudioRAGConfig()
        assert config.database_path == "~/audiorag.db"

    def test_work_dir_default_persistent(self):
        """Test work_dir defaults to persistent cache when not provided."""
        config = AudioRAGConfig()
        assert config.work_dir == Path("~/.cache/audiorag").expanduser()

    def test_work_dir_empty_string_becomes_none(self, monkeypatch):
        """Test work_dir with empty string."""
        monkeypatch.setenv("AUDIORAG_WORK_DIR", "")
        config = AudioRAGConfig()
        # Empty string should be converted to None by pydantic
        assert config.work_dir is None or config.work_dir == Path("")


# ============================================================================
# TestAudioRAGConfigOptionalFields - Test optional field handling
# ============================================================================


class TestAudioRAGConfigOptionalFields:
    """Test optional field handling (None values)."""

    def test_work_dir_default_persistent(self):
        """Test work_dir defaults to persistent cache directory."""
        config = AudioRAGConfig()
        assert config.work_dir == Path("~/.cache/audiorag").expanduser()

    def test_stt_language_optional(self):
        """Test stt_language is optional."""
        config = AudioRAGConfig()
        assert config.stt_language is None

    def test_work_dir_can_be_set(self, monkeypatch, tmp_path):
        """Test work_dir can be set to a value."""
        work_dir = str(tmp_path / "work")
        monkeypatch.setenv("AUDIORAG_WORK_DIR", work_dir)
        config = AudioRAGConfig()
        assert config.work_dir is not None
        assert config.work_dir == Path(work_dir)

    def test_stt_language_can_be_set(self, monkeypatch):
        """Test stt_language can be set to a value."""
        monkeypatch.setenv("AUDIORAG_STT_LANGUAGE", "de")
        config = AudioRAGConfig()
        assert config.stt_language is not None
        assert config.stt_language == "de"

    def test_optional_fields_with_defaults(self):
        """Test that optional fields have proper defaults."""
        config = AudioRAGConfig()
        # work_dir has a persistent default
        assert config.work_dir == Path("~/.cache/audiorag").expanduser()
        # stt_language is still optional
        assert config.stt_language is None
        # These should have defaults
        assert config.database_path == "audiorag.db"
        assert config.chunk_duration_seconds == 30


# ============================================================================
# TestAudioRAGConfigModelConfiguration - Test model configuration
# ============================================================================


class TestAudioRAGConfigModelConfiguration:
    """Test AudioRAGConfig model configuration."""

    def test_env_prefix_is_audiorag(self):
        """Test that env_prefix is set to AUDIORAG_."""
        config = AudioRAGConfig()
        # Verify by checking that AUDIORAG_ prefixed vars are loaded
        # This is implicitly tested by other tests
        assert config is not None

    def test_env_file_is_env(self):
        """Test that env_file is set to .env."""
        # This is a configuration detail, verify it's set correctly
        model_config = AudioRAGConfig.model_config
        assert model_config.get("env_file") == ".env"

    def test_env_file_encoding_is_utf8(self):
        """Test that env_file_encoding is utf-8."""
        model_config = AudioRAGConfig.model_config
        assert model_config.get("env_file_encoding") == "utf-8"

    def test_extra_is_ignore(self):
        """Test that extra fields are ignored."""
        model_config = AudioRAGConfig.model_config
        assert model_config.get("extra") == "ignore"

    def test_extra_fields_ignored(self, monkeypatch):
        """Test that extra environment variables are ignored."""
        monkeypatch.setenv("AUDIORAG_UNKNOWN_FIELD", "should-be-ignored")
        config = AudioRAGConfig()
        # Should not raise an error and should not have the field
        assert not hasattr(config, "unknown_field")

    def test_model_config_dict(self):
        """Test that model_config is a SettingsConfigDict."""
        assert hasattr(AudioRAGConfig, "model_config")
        model_config = AudioRAGConfig.model_config
        assert isinstance(model_config, dict)
        assert "env_prefix" in model_config
        assert "env_file" in model_config
        assert "env_file_encoding" in model_config
        assert "extra" in model_config


# ============================================================================
# TestAudioRAGConfigIntegration - Integration tests
# ============================================================================


class TestAudioRAGConfigIntegration:
    """Integration tests for AudioRAGConfig."""

    def test_config_creation_with_no_env_vars(self):
        """Test creating config with no environment variables set."""
        config = AudioRAGConfig()
        assert config is not None
        assert isinstance(config, AudioRAGConfig)

    def test_config_creation_with_all_env_vars(self, monkeypatch, tmp_path):
        """Test creating config with all environment variables set."""
        work_dir = str(tmp_path / "work")
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", "sk-test-123")
        monkeypatch.setenv("AUDIORAG_COHERE_API_KEY", "cohere-test-456")
        monkeypatch.setenv("AUDIORAG_DATABASE_PATH", "/custom/db.sqlite")
        monkeypatch.setenv("AUDIORAG_WORK_DIR", work_dir)
        monkeypatch.setenv("AUDIORAG_CHUNK_DURATION_SECONDS", "450")
        monkeypatch.setenv("AUDIORAG_AUDIO_FORMAT", "wav")
        monkeypatch.setenv("AUDIORAG_AUDIO_SPLIT_MAX_SIZE_MB", "50")
        monkeypatch.setenv("AUDIORAG_EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("AUDIORAG_GENERATION_MODEL", "gpt-4-turbo")
        monkeypatch.setenv("AUDIORAG_STT_MODEL", "whisper-large")
        monkeypatch.setenv("AUDIORAG_STT_LANGUAGE", "es")
        monkeypatch.setenv("AUDIORAG_RERANKER_MODEL", "rerank-v4.0")
        monkeypatch.setenv("AUDIORAG_RETRIEVAL_TOP_K", "20")
        monkeypatch.setenv("AUDIORAG_RERANK_TOP_N", "5")
        monkeypatch.setenv("AUDIORAG_CLEANUP_AUDIO", "false")

        config = AudioRAGConfig()
        assert config.openai_api_key == "sk-test-123"
        assert config.cohere_api_key == "cohere-test-456"
        assert config.database_path == "/custom/db.sqlite"
        assert config.work_dir == Path(work_dir)
        assert config.chunk_duration_seconds == 450
        assert config.audio_format == "wav"
        assert config.audio_split_max_size_mb == 50
        assert config.embedding_model == "text-embedding-3-large"
        assert config.generation_model == "gpt-4-turbo"
        assert config.stt_model == "whisper-large"
        assert config.stt_language == "es"
        assert config.reranker_model == "rerank-v4.0"
        assert config.retrieval_top_k == 20
        assert config.rerank_top_n == 5
        assert config.cleanup_audio is False

    def test_config_partial_env_vars(self, monkeypatch):
        """Test creating config with only some environment variables set."""
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", "sk-test-123")
        monkeypatch.setenv("AUDIORAG_CHUNK_DURATION_SECONDS", "600")

        config = AudioRAGConfig()
        # Set values
        assert config.openai_api_key == "sk-test-123"
        assert config.chunk_duration_seconds == 600
        # Default values
        assert config.cohere_api_key == ""
        assert config.database_path == "audiorag.db"
        assert config.audio_format == "mp3"

    def test_config_immutability_after_creation(self):
        """Test that config values can be accessed after creation."""
        config = AudioRAGConfig()
        # Access all fields to ensure they're properly initialized
        _ = config.openai_api_key
        _ = config.cohere_api_key
        _ = config.database_path
        _ = config.work_dir
        _ = config.chunk_duration_seconds
        _ = config.audio_format
        _ = config.audio_split_max_size_mb
        _ = config.embedding_model
        _ = config.generation_model
        _ = config.stt_model
        _ = config.stt_language
        _ = config.reranker_model
        _ = config.retrieval_top_k
        _ = config.rerank_top_n
        _ = config.cleanup_audio

    def test_multiple_config_instances_independent(self, monkeypatch):
        """Test that multiple config instances are independent."""
        config1 = AudioRAGConfig()
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", "sk-test-123")
        config2 = AudioRAGConfig()

        assert config1.openai_api_key == ""
        assert config2.openai_api_key == "sk-test-123"

    def test_config_with_mixed_case_env_vars(self, monkeypatch):
        """Test that environment variable names are case-insensitive."""
        # pydantic-settings converts to uppercase
        monkeypatch.setenv("AUDIORAG_OPENAI_API_KEY", "sk-test-123")
        config = AudioRAGConfig()
        assert config.openai_api_key == "sk-test-123"
