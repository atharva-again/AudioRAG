# AudioRAG - Agent Guidelines

Provider-agnostic RAG pipeline for audio files. Downloads, transcribes, chunks, embeds, and enables semantic search over audio content.

- **Language**: Python 3.12+
- **Package Manager**: `uv`
- **Architecture**: Protocol-based provider pattern with Pydantic models

## Build/Lint/Test Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run single test file/class/method
uv run pytest tests/test_models.py
uv run pytest tests/test_models.py::TestChunkMetadata
uv run pytest tests/test_models.py::TestChunkMetadata::test_valid_creation

# Coverage (requires 80%)
uv run pytest --cov=src/audiorag --cov-report=term-missing

# Linting and formatting (Ruff)
uv run ruff check . --fix      # Check with auto-fix
uv run ruff format .           # Format all files

# Type checking (Ty - 100x faster than mypy)
uv run ty check
uv run ty check --watch        # Watch mode

# Pre-commit hooks (prek - Rust-based)
uv run prek run --all-files
uv run prek run ruff           # Run specific hook

# Build package
uv build
```

## Code Style Guidelines

### Imports
- `from __future__ import annotations` in all files
- Group: stdlib → third-party → local
- Absolute imports for local modules
- **Lazy imports** for optional dependencies inside methods, NOT module level

### Type Hints
- Modern syntax: `str | None`, `list[T]`, `dict[str, Any]`
- No `Optional`, `List`, `Dict` from typing

### Naming Conventions
- **Classes**: PascalCase (`AudioRAGPipeline`, `StateManager`)
- **Functions/Methods**: snake_case (`chunk_transcription`)
- **Variables**: snake_case (`chunk_duration_seconds`)
- **Constants**: UPPER_SNAKE_CASE
- **Private**: prefix underscore (`_ensure_initialized`)
- **Protocols**: Suffix with `Provider` (`STTProvider`, `EmbeddingProvider`)

### Pydantic Models
- Extend `BaseModel` from pydantic
- Use `computed_field` decorator for serialized computed properties
- Use `StrEnum` from `enum` module for status enums
- Docstrings for all model classes

### Async Patterns
- Use `async`/`await` throughout for I/O
- Database: `aiosqlite`
- External APIs: async clients (`AsyncOpenAI`, `AsyncClientV2`)
- Pytest: `pytest-asyncio` with `asyncio_mode = "auto"`

### Error Handling
Use structured exceptions from `audiorag.exceptions`:
- `AudioRAGError` - Base for all errors
- `PipelineError` - Pipeline failures (has `stage`, `source_url`)
- `ProviderError` - External provider errors (has `provider`, `retryable`)
- `ConfigurationError` - Config validation errors
- `StateError` - Database/state errors

Log before raising: `logger.error("Pipeline failed for %s: %s", url, e)`

### Retry Logic
- All external APIs use tenacity retry decorators
- 3 attempts, exponential backoff (4s min, 60s max)
- Retry on: `RateLimitError`, transient API errors

### Documentation
- Google-style docstrings
- Document public methods with Args/Returns
- Module-level docstrings explaining purpose
- **Line length**: 100 characters (Ruff config)

### Protocols
- Define in `src/audiorag/protocols/`
- Use `runtime_checkable` decorator
- Structural subtyping via Protocol

### State Management
- SQLite with WAL mode for concurrency
- Hash-based IDs using SHA-256
- ISO 8601 timestamps in UTC
- Async context managers support

### Logging
- Use structured logging: `from audiorag.logging_config import get_logger`
- Bind context: `logger.bind(provider="openai", operation="embed")`
- Levels: debug (verbose), info (operations), warning (retries), error (failures)

### Testing
- Tests mirror source structure in `tests/`
- Use `conftest.py` for fixtures
- Mock providers with `AsyncMock`
- Class-based organization by feature
- Async tests for async code

## Project Structure

```
src/audiorag/
  __init__.py          # Public API exports
  config.py            # Pydantic-settings configuration
  models.py            # Pydantic data models
  pipeline.py          # Main orchestrator
  state.py             # SQLite state management
  chunking.py          # Audio chunking logic
  retry_config.py      # Centralized retry logic
  logging_config.py    # Structured logging setup
  exceptions.py        # Exception hierarchy
  providers/           # Provider implementations
    openai_*.py        # OpenAI providers (embeddings, STT, generation)
    voyage_embeddings.py
    cohere_*.py        # Cohere providers
    deepgram_stt.py
    assemblyai_stt.py
    chromadb_store.py
    pinecone_store.py
    weaviate_store.py
    youtube_scraper.py
    audio_splitter.py
    passthrough_reranker.py
  protocols/           # Protocol definitions
    stt.py
    embedding.py
    vector_store.py
    generation.py
    reranker.py
    audio_source.py
tests/
  conftest.py          # Shared fixtures
  test_*.py            # Test files
```

## Environment Variables

All config uses `AUDIORAG_` prefix:
- `AUDIORAG_OPENAI_API_KEY`
- `AUDIORAG_COHERE_API_KEY`
- `AUDIORAG_DATABASE_PATH`
- `AUDIORAG_CHUNK_DURATION_SECONDS`
- `AUDIORAG_STT_PROVIDER`
- `AUDIORAG_EMBEDDING_PROVIDER`
- `AUDIORAG_VECTOR_STORE_PROVIDER`
- `AUDIORAG_GENERATION_PROVIDER`

## Important Patterns

1. **Provider Pattern**: All external services abstracted behind protocols
2. **Lazy Initialization**: Default providers created lazily in `__init__`
3. **Shared Clients**: One client per provider type (e.g., single `AsyncOpenAI`)
4. **State Tracking**: All pipeline stages tracked in SQLite for resumability
5. **Config-driven**: All settings via `AudioRAGConfig` with env var support
6. **Retry Decorators**: Use `create_retry_decorator()` from `retry_config.py`
7. **Logger Binding**: Always bind provider name and operation to loggers

## Commit Message Guidelines

- Use conventional commit format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Keep subject line under 72 characters
- **DO NOT include AI agent signatures** such as:
  - "Ultraworked with [Sisyphus]"
  - "Co-authored-by: Sisyphus"
  - Any similar attribution to AI assistants
- Focus on describing the change, not who made it
- **Never do unverified commits** - Always verify changes work before committing:
  - Run tests: `uv run pytest`
  - Run linting: `uv run ruff check .`
  - Run type checking: `uv run ty check`
  - Build package: `uv build && uv run twine check dist/*`
- **Never do unsigned commits** - All commits must be GPG signed

## Common Gotchas

- Never import optional dependencies at module level
- Always call `initialize()` on `StateManager` before use
- Use `source_path` (URL) as unique key, not video title
- `ChunkMetadata` is different from chunk dict stored in SQLite
- Vector store uses chunk IDs generated by `StateManager`
- Check `e.retryable` on `ProviderError` for retry decisions
- All tests are async - use `async def test_*()`
