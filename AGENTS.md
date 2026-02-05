# AudioRAG - Agent Guidelines

This document provides guidelines for AI agents working in the AudioRAG codebase.

## Project Overview

AudioRAG is a provider-agnostic RAG (Retrieval-Augmented Generation) pipeline for audio files. It downloads audio from sources (primarily YouTube), transcribes it, chunks it, embeds it, and enables semantic search with LLM-generated answers.

- **Language**: Python 3.12+
- **Package Manager**: `uv`
- **Architecture**: Protocol-based provider pattern with Pydantic models

## Build/Lint/Test Commands

```bash
# Install dependencies
uv sync

# Install with all optional dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_models.py

# Run single test class
uv run pytest tests/test_models.py::TestChunkMetadata

# Run single test method
uv run pytest tests/test_models.py::TestChunkMetadata::test_valid_creation

# Run with coverage (requires 80% coverage)
uv run pytest --cov=src/audiorag --cov-report=term-missing

# Build package
uv build
```

## Code Style Guidelines

### Imports
- Use `from __future__ import annotations` in all files for forward reference support
- Group imports: stdlib → third-party → local
- Use absolute imports for local modules (e.g., `from audiorag.models import X`)
- **Lazy imports required** for optional dependencies (e.g., `openai`, `cohere`, `chromadb`)
  - Import inside `__init__` or methods, NOT at module level
  - This prevents `ModuleNotFoundError` when optional deps aren't installed

### Type Hints
- Use modern Python syntax: `str | None` instead of `Optional[str]`
- Use `list[T]` instead of `List[T]` (no typing imports needed)
- Use `dict[str, Any]` instead of `Dict[str, Any]`
- Use `from __future__ import annotations` to avoid forward reference issues

### Naming Conventions
- **Classes**: PascalCase (e.g., `AudioRAGPipeline`, `StateManager`)
- **Functions/Methods**: snake_case (e.g., `chunk_transcription`, `get_source_status`)
- **Variables**: snake_case (e.g., `chunk_duration_seconds`, `source_path`)
- **Constants**: UPPER_SNAKE_CASE for module-level constants
- **Private methods**: prefix with underscore (e.g., `_ensure_initialized`)
- **Protocols**: Suffix with `Provider` (e.g., `STTProvider`, `EmbeddingProvider`)

### Pydantic Models
- All data models extend `BaseModel` from pydantic
- Use `computed_field` decorator for computed properties that should be serialized
- Use `StrEnum` from `enum` module for status enums
- Add docstrings to all model classes

### Async Patterns
- Use `async`/`await` throughout for I/O operations
- Database operations use `aiosqlite`
- External API calls use async clients (`AsyncOpenAI`, `AsyncClientV2`)
- Pytest uses `pytest-asyncio` with `asyncio_mode = "auto"`

### Error Handling
- **Use structured exceptions** from `audiorag.exceptions`:
  - `AudioRAGError` - Base exception for all errors
  - `PipelineError` - Pipeline execution errors (has `stage`, `source_url` attributes)
  - `ProviderError` - External provider errors (has `provider`, `retryable` attributes)
  - `ConfigurationError` - Configuration validation errors
  - `StateError` - Database/state management errors
- Log errors before raising: `logger.error("Pipeline failed for %s: %s", url, e)`
- Use `try/except/finally` for cleanup (e.g., temp directories)
- State tracking includes FAILED status with error messages
- Check `e.retryable` for ProviderError to decide if retry is appropriate

### Retry Logic
- All external API calls use tenacity retry decorators
- Configuration: 3 attempts, exponential backoff (4s min, 10s max)
- Retry on: `RateLimitError`, transient API errors
- Import tenacity: `from tenacity import retry, stop_after_attempt, wait_exponential, ...`

### Documentation
- Use Google-style docstrings
- Document all public methods with Args and Returns sections
- Include type information in docstrings
- Add module-level docstrings explaining purpose

### Protocols (Interfaces)
- Define protocols in `src/audiorag/protocols/`
- All providers implement protocol interfaces
- Use structural subtyping (duck typing) via Protocol

### State Management
- Use SQLite with WAL mode for concurrency
- Hash-based IDs using SHA-256
- ISO 8601 timestamps in UTC
- Support for async context managers (`__aenter__`/`__aexit__`)

### Testing
- Tests in `tests/` directory mirroring source structure
- Use `conftest.py` for shared fixtures
- Mock external providers using `AsyncMock`
- Use `pytest-mock` for spying and patching
- Class-based test organization by feature
- Tests should be async when testing async code

## Project Structure

```
src/audiorag/
  __init__.py          # Public API exports
  config.py            # Pydantic-settings configuration
  models.py            # Pydantic data models
  pipeline.py          # Main orchestrator
  state.py             # SQLite state management
  chunking.py          # Audio chunking logic
  providers/           # Provider implementations
    __init__.py
    openai_*.py        # OpenAI providers
    chromadb_store.py  # ChromaDB vector store
    youtube_scraper.py # YouTube audio source
    ...
  protocols/           # Protocol definitions
    __init__.py
    stt.py
    embedding.py
    vector_store.py
    generation.py
    reranker.py
    audio_source.py
tests/
  conftest.py          # Shared pytest fixtures
  test_*.py            # Test files
```

## Important Patterns

1. **Provider Pattern**: All external services (STT, embeddings, LLM, etc.) are abstracted behind protocols
2. **Lazy Initialization**: Default providers are created lazily in `__init__` to allow custom overrides
3. **Shared Clients**: Share clients between related providers (e.g., one `AsyncOpenAI` client for both embeddings and generation)
4. **State Tracking**: All pipeline stages track status in SQLite for resumability
5. **Config-driven**: All settings via `AudioRAGConfig` with environment variable support

## Environment Variables

All config uses `AUDIORAG_` prefix:
- `AUDIORAG_OPENAI_API_KEY`
- `AUDIORAG_COHERE_API_KEY`
- `AUDIORAG_DATABASE_PATH`
- `AUDIORAG_CHUNK_DURATION_SECONDS`
- etc.

## Common Gotchas

- Never import optional dependencies at module level
- Always call `initialize()` on `StateManager` before use
- Use `source_path` (URL) as the unique key, not video title
- `ChunkMetadata` is different from chunk dict stored in SQLite
- Vector store uses chunk IDs generated by `StateManager`
