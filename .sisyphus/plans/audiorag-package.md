# AudioRAG — Standalone Python Package for RAG over Audio

## TL;DR

> **Quick Summary**: Build a pure-Python async library (`audiorag`) that provides a complete RAG pipeline over YouTube audio: scrape → transcribe → chunk → embed → store → retrieve. Every pipeline stage is provider-agnostic via Protocol interfaces, with default implementations shipped (OpenAI, ChromaDB, Cohere).
> 
> **Deliverables**:
> - Installable Python package with `src/audiorag/` layout
> - 6 Protocol interfaces (AudioSource, STT, Embedding, VectorStore, Generation, Reranker)
> - 7 default provider implementations (YouTube scraper, audio splitter, OpenAI STT/embeddings/generation, ChromaDB, Cohere reranker)
> - SQLite state management for pipeline progress tracking
> - Time-based text chunking with timestamp preservation
> - Pipeline orchestrator with async-first API
> - Pydantic-based configuration with env var support
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: T1 (scaffold) → T2 (models) + T3 (protocols) → T4-T10 (infra + providers) → T11 (pipeline) → T12 (public API)

---

## Context

### Original Request
Build a standalone Python package that handles RAG over audio files. The package should scrape YouTube videos via yt-dlp, transcribe audio, chunk by time, embed chunks, store in a vector database, and support retrieval that returns generated answers plus reranked sources with YouTube URLs and timestamps. All provider stages must be agnostic/swappable. Use `uv` for everything — no pip.

### Interview Summary
**Key Discussions**:
- **API Surface**: Library-only (`import audiorag`), no CLI
- **Default Providers**: Ship working defaults, but make it trivial to swap via Protocols
- **Async**: Async-first API throughout
- **Configuration**: Pydantic models + pydantic-settings (env vars, `.env` support) — modern 2026 pattern
- **Chunk Metadata**: `start_time`, `end_time`, `text`, `source_url`, `video_title` — nothing else
- **Tests**: No test infrastructure for now — can be added later
- **Build**: `uv_build` backend (pure Python, zero config)

**Research Findings**:
- Repo is a minimal scaffold: `pyproject.toml` (empty deps) + `main.py` (hello world) + `.python-version` (3.12)
- `uv_build` is the correct build backend for pure Python packages
- `Protocol` classes with `@runtime_checkable` are the industry standard for provider abstraction (used by autogen, chunkhound, etc.)
- `EmbeddingProvider(Protocol)` with `embed(texts) -> list[list[float]]` is the dominant pattern

### Metis Review
**Identified Gaps** (all addressed):
- **Whisper 25MB limit**: Audio files from videos >20 min exceed the Whisper API's hard 25MB cap. Added mandatory audio splitting stage with `pydub` between download and transcription.
- **ChromaDB is synchronous**: `PersistentClient` is sync-only. Resolved by wrapping in `asyncio.to_thread()` — pragmatic for v1.
- **yt-dlp is synchronous**: Same solution — `asyncio.to_thread()` wrapping.
- **FFmpeg system dependency**: yt-dlp requires FFmpeg for audio extraction. Added runtime validation with clear error.
- **Idempotent indexing**: Same URL indexed twice would create duplicates. Added SQLite check — skip by default, `force=True` to re-index.
- **File cleanup**: Large audio downloads accumulate. Added configurable `work_dir` with auto-cleanup after transcription.
- **Concurrent SQLite**: Multiple async tasks would deadlock. Using `aiosqlite` with WAL mode.
- **Empty transcriptions**: Videos with no speech produce empty chunks. Filter before embedding.
- **Pipeline resumability**: Network failure mid-pipeline leaves inconsistent state. SQLite tracks per-stage status for resume capability.
- **Query response model**: Defined `QueryResult` Pydantic model with `Source` including computed `youtube_timestamp_url`.
- **Optional dependency groups**: Core package ships Protocol interfaces only; `audiorag[defaults]` installs all providers.
- **Reranker optionality**: Reranker is part of pipeline but can be a no-op passthrough if not configured.

---

## Work Objectives

### Core Objective
Build an async-first Python library that provides a complete, provider-agnostic RAG pipeline over YouTube audio with working default implementations out of the box.

### Concrete Deliverables
- `src/audiorag/` package installable via `uv pip install -e .` or `uv pip install audiorag[defaults]`
- 6 `@runtime_checkable` Protocol interfaces for every pipeline stage
- 7 default provider implementations
- SQLite-backed pipeline state management
- `AudioRAGPipeline` class as the main user-facing orchestrator
- `AudioRAGConfig` Pydantic settings class for configuration
- `QueryResult` / `Source` response models with YouTube timestamp URLs

### Definition of Done
- [ ] `uv build` produces a valid `.whl` file
- [ ] `python -c "from audiorag import AudioRAGPipeline, AudioRAGConfig, QueryResult"` succeeds
- [ ] All 6 Protocol interfaces are importable from `audiorag.protocols`
- [ ] All 7 default providers are importable from `audiorag.providers`
- [ ] A dummy class satisfying each Protocol passes `isinstance()` check
- [ ] SQLite schema creates correctly via `aiosqlite`
- [ ] `AudioRAGConfig` loads values from environment variables with `AUDIORAG_` prefix

### Must Have
- Async-first public API (every public method is `async`)
- Provider-agnostic design via Protocols for all 6 stages
- Default working implementations for all stages
- Time-based chunking with no overlap, preserving timestamps
- SQLite state tracking with per-stage pipeline status
- Audio splitting for files exceeding STT provider limits
- Idempotent indexing (skip already-indexed URLs by default)
- `QueryResult` with sources containing YouTube timestamp URLs
- Optional dependency groups in `pyproject.toml`
- FFmpeg runtime validation with clear error message
- File cleanup after transcription completes

### Must NOT Have (Guardrails)
- No CLI — library only
- No streaming responses — v1 returns complete answers
- No playlist/batch URL support in public API — single URL per `index()` call
- No audio preprocessing (noise reduction, normalization, silence removal, diarization)
- No custom chunking strategies — time-based only, configurable duration
- No progress callbacks, event systems, or job queues
- No retry/resilience logic in Protocol contracts (default providers may have simple retry)
- No metadata beyond 5 fields (start_time, end_time, text, source_url, video_title)
- No caching layer — SQLite tracks indexed/not-indexed, nothing more
- No export/import of indexed data
- No factory patterns or managers-of-managers — one level of abstraction max
- No `print()` statements — use `logging.getLogger("audiorag")` throughout
- No graph RAG
- No tests (deferred)

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan are verifiable WITHOUT any human action.
> Every criterion is checked by running a command or script.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: NONE (deferred by user decision)
- **Framework**: N/A

### Agent-Executed QA Scenarios (PRIMARY verification method)

Every task includes QA scenarios using:

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Package/Import** | Bash (`python -c "..."`) | Import modules, check types, verify attributes |
| **Config** | Bash (`AUDIORAG_X=y python -c "..."`) | Set env vars, load config, assert values |
| **SQLite** | Bash (`python -c "import aiosqlite; ..."`) | Create schema, insert, query, verify |
| **Build** | Bash (`uv build`) | Build wheel, check output |
| **Protocol** | Bash (`python -c "isinstance(MyClass(), Protocol)"`) | Structural subtyping check |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── T1: Project scaffolding (pyproject.toml, src layout, directory structure)

Wave 2 (After Wave 1):
├── T2: Pydantic models + config (models.py, config.py)
└── T3: Protocol definitions (all 6 protocols)

Wave 3 (After Wave 2):
├── T4: SQLite state management (state.py)
├── T5: Time-based text chunking (chunking.py)
├── T6: YouTube scraper + audio splitter providers
├── T7: OpenAI STT provider
├── T8: OpenAI embedding provider
├── T9: ChromaDB vector store provider
└── T10: OpenAI generation + Cohere reranker providers

Wave 4 (After Wave 3):
├── T11: Pipeline orchestrator (pipeline.py)
└── T12: Public API surface + final verification (__init__.py)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| T1 | None | T2, T3 | None (first) |
| T2 | T1 | T4-T12 | T3 |
| T3 | T1 | T4-T12 | T2 |
| T4 | T2, T3 | T11 | T5, T6, T7, T8, T9, T10 |
| T5 | T2 | T11 | T4, T6, T7, T8, T9, T10 |
| T6 | T2, T3 | T11 | T4, T5, T7, T8, T9, T10 |
| T7 | T2, T3 | T11 | T4, T5, T6, T8, T9, T10 |
| T8 | T2, T3 | T11 | T4, T5, T6, T7, T9, T10 |
| T9 | T2, T3 | T11 | T4, T5, T6, T7, T8, T10 |
| T10 | T2, T3 | T11 | T4, T5, T6, T7, T8, T9 |
| T11 | T4-T10 | T12 | None (integration) |
| T12 | T11 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Dispatch |
|------|-------|---------------------|
| 1 | T1 | `category="quick"` — single file changes |
| 2 | T2, T3 | Parallel: both `category="quick"` |
| 3 | T4-T10 | Parallel: all `category="unspecified-low"` |
| 4 | T11, T12 | Sequential: `category="unspecified-high"` then `category="quick"` |

---

## TODOs

- [ ] 1. Project Scaffolding — `pyproject.toml`, `src/` layout, directory structure

  **What to do**:
  - Replace the existing `pyproject.toml` with full configuration:
    - `[build-system]` using `uv_build>=0.9.18,<0.10.0`
    - `[project]` with name `audiorag`, version `0.1.0`, description, `requires-python = ">=3.12"`
    - Core dependencies: `pydantic>=2.0`, `pydantic-settings>=2.0`, `aiosqlite>=0.19`
    - `[project.optional-dependencies]`:
      - `defaults = ["openai>=1.0", "chromadb>=0.4", "yt-dlp>=2024.0", "pydub>=0.25", "cohere>=5.0"]`
      - `openai = ["openai>=1.0"]`
      - `chromadb = ["chromadb>=0.4"]`
      - `scraping = ["yt-dlp>=2024.0", "pydub>=0.25"]`
      - `cohere = ["cohere>=5.0"]`
  - Delete `main.py` (no longer needed)
  - Create the following directory structure with empty `__init__.py` files:
    ```
    src/audiorag/__init__.py
    src/audiorag/protocols/__init__.py
    src/audiorag/providers/__init__.py
    ```
  - Add a `.gitignore` with standard Python ignores (`.venv/`, `__pycache__/`, `dist/`, `*.egg-info/`, `.env`)

  **Must NOT do**:
  - Do not add any logic to `__init__.py` files yet — just empty files or minimal `"""audiorag package."""` docstrings
  - Do not install dependencies yet — just declare them
  - Do not create any non-package files (no docs, no tests, no CI)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File creation and pyproject.toml editing — no complex logic
  - **Skills**: []
    - No specialized skills needed for scaffolding

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: T2, T3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `pyproject.toml` (line 1-8) — Current minimal pyproject.toml to be replaced entirely
  - `main.py` (line 1-7) — File to delete

  **External References**:
  - uv build backend docs: `uv_build` requires `[build-system] requires = ["uv_build>=0.9.18,<0.10.0"]` and `build-backend = "uv_build"`
  - Python `src/` layout convention: package code lives under `src/packagename/`

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Directory structure exists
    Tool: Bash
    Steps:
      1. ls -la src/audiorag/
      2. Assert: __init__.py exists
      3. ls -la src/audiorag/protocols/
      4. Assert: __init__.py exists
      5. ls -la src/audiorag/providers/
      6. Assert: __init__.py exists
    Expected Result: All directories and __init__.py files present

  Scenario: pyproject.toml has correct build system
    Tool: Bash
    Steps:
      1. python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['build-system']['build-backend'])"
      2. Assert: output is "uv_build"
      3. python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['dependencies'])"
      4. Assert: output contains "pydantic", "pydantic-settings", "aiosqlite"
      5. python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(list(d['project']['optional-dependencies'].keys()))"
      6. Assert: output contains "defaults", "openai", "chromadb", "scraping", "cohere"
    Expected Result: Build system and dependencies correctly configured

  Scenario: main.py is deleted
    Tool: Bash
    Steps:
      1. test -f main.py && echo "EXISTS" || echo "DELETED"
      2. Assert: output is "DELETED"
    Expected Result: main.py no longer exists

  Scenario: .gitignore exists
    Tool: Bash
    Steps:
      1. cat .gitignore
      2. Assert: contains "__pycache__", ".venv", "dist", ".env"
    Expected Result: Standard Python gitignore present
  ```

  **Commit**: YES
  - Message: `feat(scaffold): set up src layout, pyproject.toml with uv_build, and dependency groups`
  - Files: `pyproject.toml`, `src/audiorag/__init__.py`, `src/audiorag/protocols/__init__.py`, `src/audiorag/providers/__init__.py`, `.gitignore`
  - Deleted: `main.py`

---

- [ ] 2. Pydantic Models and Configuration — `models.py`, `config.py`

  **What to do**:
  - Create `src/audiorag/models.py` with the following Pydantic models:
    - `ChunkMetadata(BaseModel)`: `start_time: float`, `end_time: float`, `text: str`, `source_url: str`, `video_title: str`
    - `Source(BaseModel)`: `text: str`, `start_time: float`, `end_time: float`, `source_url: str`, `video_title: str`, `relevance_score: float`, plus a `@computed_field` property `youtube_timestamp_url: str` that returns `f"{self.source_url}&t={int(self.start_time)}"` (handles both `?` and `&` for URL params)
    - `QueryResult(BaseModel)`: `answer: str`, `sources: list[Source]`
    - `AudioFile(BaseModel)`: `path: Path`, `source_url: str`, `video_title: str`, `duration: float | None = None` — represents a downloaded audio file
    - `TranscriptionSegment(BaseModel)`: `start_time: float`, `end_time: float`, `text: str` — raw segment from STT before chunking
    - `IndexingStatus(str, Enum)`: `DOWNLOADING = "downloading"`, `DOWNLOADED = "downloaded"`, `SPLITTING = "splitting"`, `TRANSCRIBING = "transcribing"`, `TRANSCRIBED = "transcribed"`, `CHUNKING = "chunking"`, `CHUNKED = "chunked"`, `EMBEDDING = "embedding"`, `EMBEDDED = "embedded"`, `COMPLETED = "completed"`, `FAILED = "failed"`
  - Create `src/audiorag/config.py` with:
    - `AudioRAGConfig(BaseSettings)` using `pydantic-settings`:
      - `model_config = SettingsConfigDict(env_prefix="AUDIORAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore")`
      - `openai_api_key: str = ""` — shared for STT, embeddings, generation
      - `cohere_api_key: str = ""` — for reranker
      - `database_path: str = "audiorag.db"` — SQLite path
      - `work_dir: Path | None = None` — temp dir for downloads; `None` = system temp
      - `chunk_duration_seconds: int = 300` — 5 minute chunks
      - `audio_format: str = "mp3"` — download format
      - `audio_split_max_size_mb: int = 24` — split threshold (under 25MB Whisper limit)
      - `embedding_model: str = "text-embedding-3-small"` — default embedding model
      - `generation_model: str = "gpt-4o-mini"` — default generation model
      - `stt_model: str = "whisper-1"` — default STT model
      - `stt_language: str | None = None` — language hint for STT (None = auto-detect)
      - `reranker_model: str = "rerank-v3.5"` — default Cohere reranker model
      - `retrieval_top_k: int = 10` — number of chunks to retrieve before reranking
      - `rerank_top_n: int = 3` — number of chunks after reranking
      - `cleanup_audio: bool = True` — delete audio files after transcription

  **Must NOT do**:
  - Do not add methods beyond properties — models are data containers
  - Do not import any provider-specific libraries (no `openai`, no `chromadb`)
  - Do not add validators beyond basic type checking — keep models simple
  - Do not add metadata fields beyond the 5 specified ones

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward Pydantic model definitions, no complex logic
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T3)
  - **Blocks**: T4, T5, T6, T7, T8, T9, T10, T11, T12
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - Pydantic `computed_field` pattern: `@computed_field @property def youtube_timestamp_url(self) -> str:`
  - `pydantic-settings` pattern: `class Config(BaseSettings): model_config = SettingsConfigDict(env_prefix="...")`

  **External References**:
  - Pydantic v2 docs: `BaseModel`, `computed_field`, `Field`
  - pydantic-settings docs: `BaseSettings`, `SettingsConfigDict`, env var loading

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All models are importable
    Tool: Bash
    Steps:
      1. Run: uv run python -c "from audiorag.models import ChunkMetadata, Source, QueryResult, AudioFile, TranscriptionSegment, IndexingStatus; print('OK')"
      2. Assert: output is "OK"
    Expected Result: All models import successfully

  Scenario: Source has computed youtube_timestamp_url
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.models import Source
         s = Source(text='hello', start_time=65.0, end_time=125.0, source_url='https://www.youtube.com/watch?v=abc123', video_title='Test', relevance_score=0.9)
         print(s.youtube_timestamp_url)
         "
      2. Assert: output contains "t=65"
      3. Assert: output contains "youtube.com"
    Expected Result: Timestamp URL correctly computed

  Scenario: QueryResult contains sources
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.models import QueryResult, Source
         s = Source(text='hi', start_time=0, end_time=10, source_url='https://youtube.com/watch?v=x', video_title='T', relevance_score=0.5)
         qr = QueryResult(answer='test answer', sources=[s])
         print(len(qr.sources), qr.answer)
         "
      2. Assert: output is "1 test answer"
    Expected Result: QueryResult holds answer and sources correctly

  Scenario: Config loads from environment variables
    Tool: Bash
    Steps:
      1. Run: AUDIORAG_OPENAI_API_KEY=test123 AUDIORAG_DATABASE_PATH=/tmp/test.db uv run python -c "
         from audiorag.config import AudioRAGConfig
         c = AudioRAGConfig()
         print(c.openai_api_key, c.database_path)
         "
      2. Assert: output is "test123 /tmp/test.db"
    Expected Result: Environment variables loaded with AUDIORAG_ prefix

  Scenario: Config has sensible defaults
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.config import AudioRAGConfig
         c = AudioRAGConfig()
         print(c.chunk_duration_seconds, c.audio_split_max_size_mb, c.retrieval_top_k)
         "
      2. Assert: output is "300 24 10"
    Expected Result: Default values are correct

  Scenario: IndexingStatus enum has all stages
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.models import IndexingStatus
         stages = [s.value for s in IndexingStatus]
         print(stages)
         "
      2. Assert: output contains "downloading", "transcribing", "embedding", "completed", "failed"
    Expected Result: All pipeline stages represented
  ```

  **Commit**: YES (groups with T3)
  - Message: `feat(core): add Pydantic data models, config, and protocol definitions`
  - Files: `src/audiorag/models.py`, `src/audiorag/config.py`

---

- [ ] 3. Protocol Definitions — All 6 provider interfaces

  **What to do**:
  - Create `src/audiorag/protocols/stt.py`:
    - `@runtime_checkable class STTProvider(Protocol)`:
      - `async def transcribe(self, audio_path: Path, language: str | None = None) -> list[TranscriptionSegment]` — returns list of segments with timestamps
  - Create `src/audiorag/protocols/embedding.py`:
    - `@runtime_checkable class EmbeddingProvider(Protocol)`:
      - `async def embed(self, texts: list[str]) -> list[list[float]]` — batch embed texts, return vectors
  - Create `src/audiorag/protocols/vector_store.py`:
    - `@runtime_checkable class VectorStoreProvider(Protocol)`:
      - `async def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]) -> None` — store embeddings with metadata
      - `async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]` — retrieve top-k similar; returns list of dicts with `id`, `document`, `metadata`, `distance`
      - `async def delete_by_source(self, source_url: str) -> None` — delete all chunks for a source URL (for re-indexing)
  - Create `src/audiorag/protocols/generation.py`:
    - `@runtime_checkable class GenerationProvider(Protocol)`:
      - `async def generate(self, query: str, context: list[str]) -> str` — generate answer given query and context chunks
  - Create `src/audiorag/protocols/reranker.py`:
    - `@runtime_checkable class RerankerProvider(Protocol)`:
      - `async def rerank(self, query: str, documents: list[str], top_n: int = 3) -> list[tuple[int, float]]` — returns list of `(original_index, relevance_score)` sorted by relevance
  - Create `src/audiorag/protocols/audio_source.py`:
    - `@runtime_checkable class AudioSourceProvider(Protocol)`:
      - `async def download(self, url: str, output_dir: Path, audio_format: str = "mp3") -> AudioFile` — download and return AudioFile model
  - Update `src/audiorag/protocols/__init__.py`:
    - Re-export all 6 protocols: `from .stt import STTProvider`, etc.

  **Must NOT do**:
  - Do not add more than 3 methods per Protocol
  - Do not import any provider-specific libraries — protocols are pure Python + typing
  - Do not add default implementations in protocol files
  - Do not add abstract base classes — use Protocol for structural subtyping only

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Protocol definitions are small, well-defined interfaces
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T2)
  - **Blocks**: T4, T5, T6, T7, T8, T9, T10, T11, T12
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `@runtime_checkable` Protocol pattern from autogen: `class EmbeddingProvider(Protocol): async def embed(self, text: str) -> list[float]: ...`
  - chunkhound Protocol pattern: uses `@runtime_checkable` + `@property` for name

  **API/Type References**:
  - `src/audiorag/models.py:AudioFile` — return type for AudioSourceProvider.download()
  - `src/audiorag/models.py:TranscriptionSegment` — return type for STTProvider.transcribe()

  **External References**:
  - Python `typing.Protocol` docs: structural subtyping, `@runtime_checkable` decorator
  - Python `typing.runtime_checkable` docs: enables `isinstance()` checks on Protocol classes

  **WHY Each Reference Matters**:
  - The autogen/chunkhound patterns show the exact async method signature style to follow
  - Models references are needed because Protocol methods return/accept these types

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All protocols are importable from audiorag.protocols
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import STTProvider, EmbeddingProvider, VectorStoreProvider, GenerationProvider, RerankerProvider, AudioSourceProvider
         print('OK')
         "
      2. Assert: output is "OK"
    Expected Result: All 6 protocols importable

  Scenario: Protocols are runtime_checkable
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import EmbeddingProvider
         class MyEmb:
             async def embed(self, texts: list[str]) -> list[list[float]]:
                 return [[0.1]*384 for _ in texts]
         print(isinstance(MyEmb(), EmbeddingProvider))
         "
      2. Assert: output is "True"
    Expected Result: isinstance check works with dummy class

  Scenario: Dummy class missing method fails isinstance
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import EmbeddingProvider
         class BadEmb:
             pass
         print(isinstance(BadEmb(), EmbeddingProvider))
         "
      2. Assert: output is "False"
    Expected Result: Incomplete class correctly fails check

  Scenario: Protocol methods have correct signatures
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import inspect
         from audiorag.protocols import STTProvider, EmbeddingProvider, VectorStoreProvider, GenerationProvider, RerankerProvider, AudioSourceProvider
         for P in [STTProvider, EmbeddingProvider, VectorStoreProvider, GenerationProvider, RerankerProvider, AudioSourceProvider]:
             methods = [m for m in dir(P) if not m.startswith('_')]
             print(f'{P.__name__}: {methods}')
         "
      2. Assert: STTProvider has "transcribe"
      3. Assert: EmbeddingProvider has "embed"
      4. Assert: VectorStoreProvider has "add", "query", "delete_by_source"
      5. Assert: GenerationProvider has "generate"
      6. Assert: RerankerProvider has "rerank"
      7. Assert: AudioSourceProvider has "download"
    Expected Result: Every protocol exposes the correct methods
  ```

  **Commit**: YES (groups with T2)
  - Message: `feat(core): add Pydantic data models, config, and protocol definitions`
  - Files: `src/audiorag/protocols/stt.py`, `src/audiorag/protocols/embedding.py`, `src/audiorag/protocols/vector_store.py`, `src/audiorag/protocols/generation.py`, `src/audiorag/protocols/reranker.py`, `src/audiorag/protocols/audio_source.py`, `src/audiorag/protocols/__init__.py`

---

- [ ] 4. SQLite State Management — `state.py`

  **What to do**:
  - Create `src/audiorag/state.py` with class `StateManager`:
    - `__init__(self, database_path: str)` — stores path, does not open connection
    - `async def initialize(self) -> None` — creates tables if not exist, enables WAL mode (`PRAGMA journal_mode=WAL`), creates the schema:
      - Table `sources`: `id TEXT PRIMARY KEY` (URL hash), `url TEXT UNIQUE NOT NULL`, `video_title TEXT`, `status TEXT NOT NULL DEFAULT 'downloading'`, `created_at TEXT NOT NULL`, `updated_at TEXT NOT NULL`, `error_message TEXT`
      - Table `chunks`: `id TEXT PRIMARY KEY`, `source_id TEXT NOT NULL REFERENCES sources(id)`, `text TEXT NOT NULL`, `start_time REAL NOT NULL`, `end_time REAL NOT NULL`, `embedding_id TEXT` (vector store reference), `created_at TEXT NOT NULL`
    - `async def get_source_status(self, url: str) -> IndexingStatus | None` — returns current status or None if not indexed
    - `async def upsert_source(self, url: str, video_title: str, status: IndexingStatus) -> str` — insert or update source, return source_id
    - `async def update_source_status(self, url: str, status: IndexingStatus, error_message: str | None = None) -> None`
    - `async def store_chunks(self, source_id: str, chunks: list[ChunkMetadata]) -> list[str]` — insert chunks, return chunk IDs
    - `async def get_chunks_for_source(self, source_id: str) -> list[dict]` — get all chunks for a source
    - `async def delete_source(self, url: str) -> None` — delete source and its chunks (for re-indexing)
    - Use `aiosqlite` for all database access
    - Generate source IDs as SHA-256 hash of the URL
    - Generate chunk IDs as SHA-256 hash of `source_id + start_time + end_time`
    - Use ISO 8601 format for timestamps

  **Must NOT do**:
  - Do not use an ORM — raw SQL with aiosqlite only
  - Do not add migration logic — schema is created fresh with `CREATE TABLE IF NOT EXISTS`
  - Do not add connection pooling — single connection per StateManager instance
  - Do not add caching on top of SQLite

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Moderate complexity — async SQLite with schema design, but well-defined scope
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T5-T10)
  - **Blocks**: T11
  - **Blocked By**: T2, T3

  **References**:

  **API/Type References**:
  - `src/audiorag/models.py:IndexingStatus` — enum used for status values
  - `src/audiorag/models.py:ChunkMetadata` — input type for store_chunks

  **External References**:
  - aiosqlite docs: `async with aiosqlite.connect(path) as db: await db.execute(...)`
  - SQLite WAL mode: `PRAGMA journal_mode=WAL` — enables concurrent reads during writes

  **WHY Each Reference Matters**:
  - IndexingStatus enum defines the valid pipeline stages the state manager tracks
  - ChunkMetadata model is the input format for persisting chunks — must match its fields

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Schema creates correctly
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio, aiosqlite
         from audiorag.state import StateManager
         async def test():
             sm = StateManager('/tmp/test_audiorag_state.db')
             await sm.initialize()
             async with aiosqlite.connect('/tmp/test_audiorag_state.db') as db:
                 cursor = await db.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
                 tables = [row[0] for row in await cursor.fetchall()]
                 print(sorted(tables))
         asyncio.run(test())
         "
      2. Assert: output contains "chunks" and "sources"
    Expected Result: Both tables created
    Evidence: Terminal output captured

  Scenario: WAL mode enabled
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio, aiosqlite
         from audiorag.state import StateManager
         async def test():
             sm = StateManager('/tmp/test_audiorag_wal.db')
             await sm.initialize()
             async with aiosqlite.connect('/tmp/test_audiorag_wal.db') as db:
                 cursor = await db.execute('PRAGMA journal_mode')
                 mode = (await cursor.fetchone())[0]
                 print(mode)
         asyncio.run(test())
         "
      2. Assert: output is "wal"
    Expected Result: WAL mode active

  Scenario: Upsert and retrieve source status
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio
         from audiorag.state import StateManager
         from audiorag.models import IndexingStatus
         async def test():
             sm = StateManager('/tmp/test_audiorag_upsert.db')
             await sm.initialize()
             sid = await sm.upsert_source('https://youtube.com/watch?v=abc', 'Test Video', IndexingStatus.DOWNLOADING)
             status = await sm.get_source_status('https://youtube.com/watch?v=abc')
             print(status, sid[:8])
         asyncio.run(test())
         "
      2. Assert: status is IndexingStatus.DOWNLOADING or "downloading"
      3. Assert: sid is an 8-char hex string (truncated SHA-256)
    Expected Result: Source created and retrievable

  Scenario: Idempotent upsert
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio
         from audiorag.state import StateManager
         from audiorag.models import IndexingStatus
         async def test():
             sm = StateManager('/tmp/test_audiorag_idempotent.db')
             await sm.initialize()
             sid1 = await sm.upsert_source('https://youtube.com/watch?v=abc', 'Test', IndexingStatus.DOWNLOADING)
             await sm.update_source_status('https://youtube.com/watch?v=abc', IndexingStatus.COMPLETED)
             sid2 = await sm.upsert_source('https://youtube.com/watch?v=abc', 'Test', IndexingStatus.DOWNLOADING)
             print(sid1 == sid2)
         asyncio.run(test())
         "
      2. Assert: output is "True"
    Expected Result: Same URL produces same source_id

  Scenario: Delete source removes source and chunks
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio
         from audiorag.state import StateManager
         from audiorag.models import IndexingStatus, ChunkMetadata
         async def test():
             sm = StateManager('/tmp/test_audiorag_delete.db')
             await sm.initialize()
             sid = await sm.upsert_source('https://youtube.com/watch?v=del', 'Del', IndexingStatus.COMPLETED)
             chunks = [ChunkMetadata(start_time=0, end_time=10, text='hello', source_url='https://youtube.com/watch?v=del', video_title='Del')]
             await sm.store_chunks(sid, chunks)
             await sm.delete_source('https://youtube.com/watch?v=del')
             status = await sm.get_source_status('https://youtube.com/watch?v=del')
             print(status)
         asyncio.run(test())
         "
      2. Assert: output is "None"
    Expected Result: Source and chunks fully removed
  ```

  **Commit**: YES
  - Message: `feat(state): add SQLite state management with aiosqlite and WAL mode`
  - Files: `src/audiorag/state.py`

---

- [ ] 5. Time-Based Text Chunking — `chunking.py`

  **What to do**:
  - Create `src/audiorag/chunking.py` with a single function:
    - `def chunk_transcription(segments: list[TranscriptionSegment], chunk_duration_seconds: int = 300) -> list[ChunkMetadata]`
      - Takes raw STT segments (each with `start_time`, `end_time`, `text`)
      - Groups segments into non-overlapping chunks of `chunk_duration_seconds`
      - Each chunk's `start_time` is the first segment's start, `end_time` is the last segment's end
      - Each chunk's `text` is the concatenation of all segment texts (space-joined)
      - Filter out chunks where `text.strip()` is empty (handles videos with silent sections)
      - `source_url` and `video_title` are passed as parameters and set on every chunk
    - Full signature: `def chunk_transcription(segments: list[TranscriptionSegment], chunk_duration_seconds: int, source_url: str, video_title: str) -> list[ChunkMetadata]`
    - This is a pure function — no async needed, no I/O, no side effects
    - The chunking algorithm: iterate segments in order, accumulate into current chunk until accumulated duration exceeds `chunk_duration_seconds`, then start a new chunk. The boundary is the segment boundary — never split a segment across chunks.

  **Must NOT do**:
  - Do not add overlap — zero overlap, as specified
  - Do not add semantic chunking, sentence-boundary chunking, or any other strategy
  - Do not make this async — it's pure CPU logic
  - Do not add configurability beyond `chunk_duration_seconds`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single pure function with clear algorithm
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T4, T6-T10)
  - **Blocks**: T11
  - **Blocked By**: T2

  **References**:

  **API/Type References**:
  - `src/audiorag/models.py:TranscriptionSegment` — input type (list of segments from STT)
  - `src/audiorag/models.py:ChunkMetadata` — output type (list of time-based chunks)

  **WHY Each Reference Matters**:
  - TranscriptionSegment defines the input contract: `start_time`, `end_time`, `text` fields
  - ChunkMetadata defines the output contract: the 5 metadata fields

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Basic chunking groups segments by duration
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.chunking import chunk_transcription
         from audiorag.models import TranscriptionSegment
         segments = [
             TranscriptionSegment(start_time=0, end_time=60, text='First minute.'),
             TranscriptionSegment(start_time=60, end_time=120, text='Second minute.'),
             TranscriptionSegment(start_time=120, end_time=180, text='Third minute.'),
             TranscriptionSegment(start_time=180, end_time=240, text='Fourth minute.'),
             TranscriptionSegment(start_time=240, end_time=300, text='Fifth minute.'),
             TranscriptionSegment(start_time=300, end_time=360, text='Sixth minute.'),
         ]
         chunks = chunk_transcription(segments, chunk_duration_seconds=300, source_url='https://yt.com/watch?v=x', video_title='Test')
         print(len(chunks), chunks[0].start_time, chunks[0].end_time, chunks[1].start_time)
         "
      2. Assert: 2 chunks total
      3. Assert: first chunk start_time=0, end_time=300
      4. Assert: second chunk start_time=300
    Expected Result: Segments grouped into 5-min chunks

  Scenario: Empty segments filtered out
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.chunking import chunk_transcription
         from audiorag.models import TranscriptionSegment
         segments = [
             TranscriptionSegment(start_time=0, end_time=60, text=''),
             TranscriptionSegment(start_time=60, end_time=120, text='   '),
             TranscriptionSegment(start_time=120, end_time=180, text='Real content.'),
         ]
         chunks = chunk_transcription(segments, chunk_duration_seconds=300, source_url='https://yt.com/watch?v=x', video_title='T')
         print(len(chunks), chunks[0].text.strip())
         "
      2. Assert: result has chunks with 'Real content.' (empty-only chunks filtered)
    Expected Result: Whitespace-only chunks excluded

  Scenario: Single segment produces single chunk
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.chunking import chunk_transcription
         from audiorag.models import TranscriptionSegment
         segments = [TranscriptionSegment(start_time=0, end_time=30, text='Short.')]
         chunks = chunk_transcription(segments, chunk_duration_seconds=300, source_url='https://yt.com/watch?v=x', video_title='T')
         print(len(chunks), chunks[0].source_url)
         "
      2. Assert: 1 chunk, source_url matches
    Expected Result: Single segment becomes single chunk

  Scenario: No overlap between chunks
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.chunking import chunk_transcription
         from audiorag.models import TranscriptionSegment
         segments = [TranscriptionSegment(start_time=i*30, end_time=(i+1)*30, text=f'Seg {i}') for i in range(20)]
         chunks = chunk_transcription(segments, chunk_duration_seconds=300, source_url='https://yt.com/watch?v=x', video_title='T')
         for i in range(len(chunks)-1):
             assert chunks[i].end_time <= chunks[i+1].start_time, f'Overlap at chunk {i}'
         print('No overlaps')
         "
      2. Assert: output is "No overlaps"
    Expected Result: Zero overlap confirmed
  ```

  **Commit**: YES
  - Message: `feat(chunking): add time-based text chunking with no overlap`
  - Files: `src/audiorag/chunking.py`

---

- [ ] 6. YouTube Scraper + Audio Splitter — `providers/youtube_scraper.py`, `providers/audio_splitter.py`

  **What to do**:
  - Create `src/audiorag/providers/youtube_scraper.py`:
    - `class YouTubeScraper` satisfying `AudioSourceProvider` Protocol:
      - `async def download(self, url: str, output_dir: Path, audio_format: str = "mp3") -> AudioFile`
      - Internally uses `yt_dlp.YoutubeDL` wrapped in `asyncio.to_thread()`:
        - Extract video info first (`extract_info` with `download=False`) to get title, duration
        - Then download with `FFmpegExtractAudio` postprocessor to extract audio
        - yt-dlp opts: `format: "bestaudio/best"`, `postprocessors: [{"key": "FFmpegExtractAudio", "preferredcodec": audio_format}]`, `outtmpl` set to `output_dir / "%(id)s.%(ext)s"`
      - Before any operation, validate FFmpeg is available: run `shutil.which("ffmpeg")` — if None, raise `RuntimeError("FFmpeg is required but not found. Install FFmpeg: https://ffmpeg.org/download.html")`
      - Return `AudioFile` with path, source_url, video_title, duration
      - Use `logging.getLogger("audiorag.providers.youtube_scraper")`
  - Create `src/audiorag/providers/audio_splitter.py`:
    - `class AudioSplitter`:
      - `async def split_if_needed(self, audio_file: AudioFile, max_size_mb: int = 24) -> list[Path]`
      - Check file size: if under `max_size_mb * 1024 * 1024`, return `[audio_file.path]`
      - If over limit, use `pydub.AudioSegment.from_file()` to load audio
      - Split into segments of duration calculated from: `segment_duration_ms = (max_size_mb / file_size_mb) * total_duration_ms * 0.95` (5% safety margin)
      - Export each segment to `output_dir / f"{stem}_part{i}.{ext}"`
      - Wrap pydub calls in `asyncio.to_thread()`
      - Return list of paths to split files
      - Use `logging.getLogger("audiorag.providers.audio_splitter")`

  **Must NOT do**:
  - Do not support playlists — reject URLs with `&list=` with a clear error
  - Do not add retry logic for downloads
  - Do not add progress callbacks
  - Do not add audio preprocessing (normalization, noise reduction)
  - Do not cache downloads

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Two provider implementations with real I/O wrapping and error handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T4, T5, T7-T10)
  - **Blocks**: T11
  - **Blocked By**: T2, T3

  **References**:

  **API/Type References**:
  - `src/audiorag/models.py:AudioFile` — return type for download method
  - `src/audiorag/protocols/audio_source.py:AudioSourceProvider` — Protocol to satisfy

  **External References**:
  - yt-dlp programmatic API: `YoutubeDL(opts).extract_info(url)`, `YoutubeDL(opts).download([url])`
  - pydub: `AudioSegment.from_file(path)`, `segment[start_ms:end_ms].export(path)`
  - shutil.which for FFmpeg detection

  **WHY Each Reference Matters**:
  - AudioFile model defines the return shape the scraper must produce
  - AudioSourceProvider Protocol defines the exact method signature to implement
  - yt-dlp docs show the correct opts dict structure and postprocessor config

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: YouTubeScraper satisfies AudioSourceProvider Protocol
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import AudioSourceProvider
         from audiorag.providers.youtube_scraper import YouTubeScraper
         print(isinstance(YouTubeScraper(), AudioSourceProvider))
         "
      2. Assert: output is "True"
    Expected Result: Protocol satisfied

  Scenario: AudioSplitter class is importable
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.providers.audio_splitter import AudioSplitter
         import inspect
         print(inspect.iscoroutinefunction(AudioSplitter().split_if_needed))
         "
      2. Assert: output is "True"
    Expected Result: split_if_needed is async

  Scenario: FFmpeg validation raises clear error
    Tool: Bash
    Steps:
      1. Run: PATH="" uv run python -c "
         import asyncio
         from audiorag.providers.youtube_scraper import YouTubeScraper
         from pathlib import Path
         async def test():
             yt = YouTubeScraper()
             try:
                 await yt.download('https://youtube.com/watch?v=test', Path('/tmp'))
             except RuntimeError as e:
                 print(str(e))
         asyncio.run(test())
         " 2>&1 || true
      2. Assert: output contains "FFmpeg" and "required"
    Expected Result: Clear FFmpeg missing error
  ```

  **Commit**: YES
  - Message: `feat(providers): add YouTube scraper with yt-dlp and audio splitter with pydub`
  - Files: `src/audiorag/providers/youtube_scraper.py`, `src/audiorag/providers/audio_splitter.py`

---

- [ ] 7. OpenAI STT Provider — `providers/openai_stt.py`

  **What to do**:
  - Create `src/audiorag/providers/openai_stt.py`:
    - `class OpenAISTTProvider` satisfying `STTProvider` Protocol:
      - `__init__(self, api_key: str, model: str = "whisper-1")`
      - `async def transcribe(self, audio_path: Path, language: str | None = None) -> list[TranscriptionSegment]`
      - Uses `openai.AsyncOpenAI(api_key=api_key)` client
      - Calls `client.audio.transcriptions.create()` with:
        - `model=self.model`
        - `file=open(audio_path, "rb")`
        - `response_format="verbose_json"`
        - `timestamp_granularities=["segment"]`
        - `language=language` if provided
      - Parses the response segments into `list[TranscriptionSegment]` with `start_time`, `end_time`, `text`
      - Use `logging.getLogger("audiorag.providers.openai_stt")`

  **Must NOT do**:
  - Do not handle audio splitting here — that's AudioSplitter's job (the pipeline orchestrator calls splitter first, then STT per chunk)
  - Do not add retry logic
  - Do not add rate limiting
  - Do not add local Whisper support — that would be a separate provider

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single API call wrapper with response parsing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T4-T6, T8-T10)
  - **Blocks**: T11
  - **Blocked By**: T2, T3

  **References**:

  **API/Type References**:
  - `src/audiorag/models.py:TranscriptionSegment` — output type, each segment has `start_time`, `end_time`, `text`
  - `src/audiorag/protocols/stt.py:STTProvider` — Protocol to satisfy

  **External References**:
  - OpenAI Whisper API: `client.audio.transcriptions.create(model=..., file=..., response_format="verbose_json", timestamp_granularities=["segment"])`
  - Response format: `response.segments` is a list of dicts with `start`, `end`, `text` keys

  **WHY Each Reference Matters**:
  - TranscriptionSegment is the exact output model — fields must map from Whisper's `start`/`end`/`text`
  - The `verbose_json` + `timestamp_granularities` flags are critical to getting segment-level timestamps

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: OpenAISTTProvider satisfies STTProvider Protocol
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import STTProvider
         from audiorag.providers.openai_stt import OpenAISTTProvider
         print(isinstance(OpenAISTTProvider(api_key='fake'), STTProvider))
         "
      2. Assert: output is "True"
    Expected Result: Protocol satisfied

  Scenario: Constructor accepts api_key and model
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.providers.openai_stt import OpenAISTTProvider
         p = OpenAISTTProvider(api_key='test', model='whisper-1')
         print(p.model)
         "
      2. Assert: output is "whisper-1"
    Expected Result: Config stored correctly
  ```

  **Commit**: YES (groups with T8, T9, T10)
  - Message: `feat(providers): add OpenAI STT, embeddings, generation, ChromaDB, and Cohere reranker providers`
  - Files: `src/audiorag/providers/openai_stt.py`

---

- [ ] 8. OpenAI Embedding Provider — `providers/openai_embeddings.py`

  **What to do**:
  - Create `src/audiorag/providers/openai_embeddings.py`:
    - `class OpenAIEmbeddingProvider` satisfying `EmbeddingProvider` Protocol:
      - `__init__(self, api_key: str, model: str = "text-embedding-3-small")`
      - `async def embed(self, texts: list[str]) -> list[list[float]]`
      - Uses `openai.AsyncOpenAI(api_key=api_key)` client
      - Calls `client.embeddings.create(input=texts, model=self.model)`
      - Returns `[item.embedding for item in response.data]`
      - Handle empty input: if `texts` is empty, return `[]`
      - Use `logging.getLogger("audiorag.providers.openai_embeddings")`

  **Must NOT do**:
  - Do not add batching logic (OpenAI API handles batch internally)
  - Do not add dimension configuration (model determines dimensions)
  - Do not add caching

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple API call wrapper
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T4-T7, T9, T10)
  - **Blocks**: T11
  - **Blocked By**: T2, T3

  **References**:

  **API/Type References**:
  - `src/audiorag/protocols/embedding.py:EmbeddingProvider` — Protocol to satisfy

  **External References**:
  - OpenAI Embeddings API: `client.embeddings.create(input=texts, model="text-embedding-3-small")`
  - Response format: `response.data[i].embedding` is `list[float]`

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: OpenAIEmbeddingProvider satisfies EmbeddingProvider Protocol
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import EmbeddingProvider
         from audiorag.providers.openai_embeddings import OpenAIEmbeddingProvider
         print(isinstance(OpenAIEmbeddingProvider(api_key='fake'), EmbeddingProvider))
         "
      2. Assert: output is "True"
    Expected Result: Protocol satisfied

  Scenario: embed method is async
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import inspect
         from audiorag.providers.openai_embeddings import OpenAIEmbeddingProvider
         print(inspect.iscoroutinefunction(OpenAIEmbeddingProvider(api_key='x').embed))
         "
      2. Assert: output is "True"
    Expected Result: Method is a coroutine
  ```

  **Commit**: YES (groups with T7, T9, T10)
  - Message: `feat(providers): add OpenAI STT, embeddings, generation, ChromaDB, and Cohere reranker providers`
  - Files: `src/audiorag/providers/openai_embeddings.py`

---

- [ ] 9. ChromaDB Vector Store Provider — `providers/chromadb_store.py`

  **What to do**:
  - Create `src/audiorag/providers/chromadb_store.py`:
    - `class ChromaDBVectorStore` satisfying `VectorStoreProvider` Protocol:
      - `__init__(self, persist_directory: str = "./chroma_db", collection_name: str = "audiorag")`
      - Internally creates `chromadb.PersistentClient(path=persist_directory)` and gets/creates collection `collection_name`
      - `async def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]) -> None`
        - Wraps `collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)` in `asyncio.to_thread()`
      - `async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]`
        - Wraps `collection.query(query_embeddings=[embedding], n_results=top_k)` in `asyncio.to_thread()`
        - Transforms ChromaDB's grouped result format into a flat list of dicts: `[{"id": id, "document": doc, "metadata": meta, "distance": dist}, ...]`
      - `async def delete_by_source(self, source_url: str) -> None`
        - Wraps `collection.delete(where={"source_url": source_url})` in `asyncio.to_thread()`
      - Use `logging.getLogger("audiorag.providers.chromadb_store")`

  **Must NOT do**:
  - Do not require a Chroma server — use `PersistentClient` only
  - Do not add collection management beyond get_or_create
  - Do not add embedding function to ChromaDB — we manage embeddings externally
  - Do not leak ChromaDB types into public API

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Moderate complexity — ChromaDB response format transformation and async wrapping
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T4-T8, T10)
  - **Blocks**: T11
  - **Blocked By**: T2, T3

  **References**:

  **API/Type References**:
  - `src/audiorag/protocols/vector_store.py:VectorStoreProvider` — Protocol to satisfy (add, query, delete_by_source methods)

  **External References**:
  - ChromaDB PersistentClient: `chromadb.PersistentClient(path="...")`
  - ChromaDB collection API: `collection.add(ids=, embeddings=, metadatas=, documents=)`, `collection.query(query_embeddings=, n_results=)`
  - ChromaDB query result format: `{"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}` — nested lists that need flattening

  **WHY Each Reference Matters**:
  - VectorStoreProvider Protocol defines the 3 exact methods to implement
  - ChromaDB's query result is nested (lists of lists) — must flatten to list of dicts
  - ChromaDB's `delete(where=...)` uses metadata filtering — source_url must be stored in metadata

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: ChromaDBVectorStore satisfies VectorStoreProvider Protocol
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import VectorStoreProvider
         from audiorag.providers.chromadb_store import ChromaDBVectorStore
         print(isinstance(ChromaDBVectorStore(persist_directory='/tmp/chroma_test'), VectorStoreProvider))
         "
      2. Assert: output is "True"
    Expected Result: Protocol satisfied

  Scenario: Add and query round-trip
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio
         from audiorag.providers.chromadb_store import ChromaDBVectorStore
         async def test():
             store = ChromaDBVectorStore(persist_directory='/tmp/chroma_roundtrip', collection_name='test')
             await store.add(
                 ids=['c1'],
                 embeddings=[[0.1]*384],
                 metadatas=[{'source_url': 'https://yt.com/watch?v=x', 'start_time': 0}],
                 documents=['hello world']
             )
             results = await store.query(embedding=[0.1]*384, top_k=1)
             print(len(results), results[0]['document'])
         asyncio.run(test())
         "
      2. Assert: 1 result with document "hello world"
    Expected Result: Data persists and queries correctly

  Scenario: Delete by source removes matching entries
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio
         from audiorag.providers.chromadb_store import ChromaDBVectorStore
         async def test():
             store = ChromaDBVectorStore(persist_directory='/tmp/chroma_delete', collection_name='deltest')
             await store.add(
                 ids=['c1', 'c2'],
                 embeddings=[[0.1]*384, [0.2]*384],
                 metadatas=[{'source_url': 'url1'}, {'source_url': 'url2'}],
                 documents=['doc1', 'doc2']
             )
             await store.delete_by_source('url1')
             results = await store.query(embedding=[0.1]*384, top_k=10)
             urls = [r['metadata']['source_url'] for r in results]
             print('url1' not in urls, len(results))
         asyncio.run(test())
         "
      2. Assert: url1 is gone, only url2 remains
    Expected Result: Selective deletion works
  ```

  **Commit**: YES (groups with T7, T8, T10)
  - Message: `feat(providers): add OpenAI STT, embeddings, generation, ChromaDB, and Cohere reranker providers`
  - Files: `src/audiorag/providers/chromadb_store.py`

---

- [ ] 10. OpenAI Generation + Cohere Reranker Providers — `providers/openai_generation.py`, `providers/cohere_reranker.py`

  **What to do**:
  - Create `src/audiorag/providers/openai_generation.py`:
    - `class OpenAIGenerationProvider` satisfying `GenerationProvider` Protocol:
      - `__init__(self, api_key: str, model: str = "gpt-4o-mini")`
      - `async def generate(self, query: str, context: list[str]) -> str`
      - Uses `openai.AsyncOpenAI(api_key=api_key)`
      - Builds a system prompt: `"You are a helpful assistant. Answer the user's question based ONLY on the following context from audio transcriptions. If the context doesn't contain relevant information, say so. Always reference the source when possible.\n\nContext:\n{joined_context}"`
      - Context is formatted as numbered chunks: `"[1] {chunk_text}\n[2] {chunk_text}\n..."`
      - Calls `client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}])`
      - Returns `response.choices[0].message.content`
      - Use `logging.getLogger("audiorag.providers.openai_generation")`
  - Create `src/audiorag/providers/cohere_reranker.py`:
    - `class CohereReranker` satisfying `RerankerProvider` Protocol:
      - `__init__(self, api_key: str, model: str = "rerank-v3.5")`
      - `async def rerank(self, query: str, documents: list[str], top_n: int = 3) -> list[tuple[int, float]]`
      - Uses `cohere.AsyncClientV2(api_key=api_key)` (Cohere v2 client)
      - Calls `client.rerank(model=self.model, query=query, documents=documents, top_n=top_n)`
      - Returns `[(result.index, result.relevance_score) for result in response.results]`
      - Handle edge case: if `documents` is empty, return `[]`
      - Use `logging.getLogger("audiorag.providers.cohere_reranker")`
  - Create `src/audiorag/providers/passthrough_reranker.py`:
    - `class PassthroughReranker` satisfying `RerankerProvider` Protocol:
      - No `__init__` args needed
      - `async def rerank(self, query: str, documents: list[str], top_n: int = 3) -> list[tuple[int, float]]`
      - Returns first `top_n` documents with score 1.0: `[(i, 1.0) for i in range(min(top_n, len(documents)))]`
      - This is the default reranker when no Cohere API key is configured

  **Must NOT do**:
  - Do not add streaming support to generation
  - Do not add conversation history / multi-turn
  - Do not add prompt customization beyond the system prompt
  - Do not add retry logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Three provider implementations, moderate complexity with API integration
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T4-T9)
  - **Blocks**: T11
  - **Blocked By**: T2, T3

  **References**:

  **API/Type References**:
  - `src/audiorag/protocols/generation.py:GenerationProvider` — Protocol to satisfy
  - `src/audiorag/protocols/reranker.py:RerankerProvider` — Protocol to satisfy

  **External References**:
  - OpenAI Chat Completions API: `client.chat.completions.create(model=, messages=[...])`
  - Cohere v2 Rerank API: `client.rerank(model=, query=, documents=, top_n=)`
  - Cohere response: `response.results[i].index`, `response.results[i].relevance_score`

  **WHY Each Reference Matters**:
  - GenerationProvider Protocol specifies `generate(query, context) -> str`
  - RerankerProvider Protocol specifies `rerank(query, documents, top_n) -> list[tuple[int, float]]`
  - PassthroughReranker is the fallback when no Cohere key is available

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All three classes satisfy their Protocols
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import GenerationProvider, RerankerProvider
         from audiorag.providers.openai_generation import OpenAIGenerationProvider
         from audiorag.providers.cohere_reranker import CohereReranker
         from audiorag.providers.passthrough_reranker import PassthroughReranker
         print(isinstance(OpenAIGenerationProvider(api_key='x'), GenerationProvider))
         print(isinstance(CohereReranker(api_key='x'), RerankerProvider))
         print(isinstance(PassthroughReranker(), RerankerProvider))
         "
      2. Assert: all three print "True"
    Expected Result: All protocols satisfied

  Scenario: PassthroughReranker returns first N items
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio
         from audiorag.providers.passthrough_reranker import PassthroughReranker
         async def test():
             r = PassthroughReranker()
             result = await r.rerank('query', ['a', 'b', 'c', 'd', 'e'], top_n=3)
             print(result)
         asyncio.run(test())
         "
      2. Assert: output is "[(0, 1.0), (1, 1.0), (2, 1.0)]"
    Expected Result: First 3 returned with score 1.0

  Scenario: PassthroughReranker handles empty docs
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import asyncio
         from audiorag.providers.passthrough_reranker import PassthroughReranker
         async def test():
             r = PassthroughReranker()
             result = await r.rerank('query', [], top_n=3)
             print(result)
         asyncio.run(test())
         "
      2. Assert: output is "[]"
    Expected Result: Empty input returns empty output
  ```

  **Commit**: YES (groups with T7, T8, T9)
  - Message: `feat(providers): add OpenAI STT, embeddings, generation, ChromaDB, and Cohere reranker providers`
  - Files: `src/audiorag/providers/openai_generation.py`, `src/audiorag/providers/cohere_reranker.py`, `src/audiorag/providers/passthrough_reranker.py`

---

- [ ] 11. Pipeline Orchestrator — `pipeline.py`

  **What to do**:
  - Create `src/audiorag/pipeline.py` with `class AudioRAGPipeline`:
    - `__init__(self, config: AudioRAGConfig, *, audio_source: AudioSourceProvider | None = None, stt: STTProvider | None = None, embedder: EmbeddingProvider | None = None, vector_store: VectorStoreProvider | None = None, generator: GenerationProvider | None = None, reranker: RerankerProvider | None = None)`:
      - Store config
      - For each provider param: if None, instantiate the default provider using config values (e.g., `OpenAISTTProvider(api_key=config.openai_api_key, model=config.stt_model)`)
      - For reranker: if None AND `config.cohere_api_key` is set, use `CohereReranker`; else use `PassthroughReranker`
      - Create `StateManager(config.database_path)` — store as `self._state`
      - Create `AudioSplitter()` — store as `self._splitter`
      - Set `self._initialized = False`
    - `async def _ensure_initialized(self) -> None`:
      - If not initialized, call `await self._state.initialize()`, set flag
    - `async def index(self, url: str, *, force: bool = False) -> None`:
      - Call `_ensure_initialized()`
      - Check idempotency: `status = await self._state.get_source_status(url)` — if status is `COMPLETED` and `force` is False, log "Already indexed" and return
      - If `force` and already exists, call `self._state.delete_source(url)` AND `self._vector_store.delete_by_source(url)`
      - **Stage 1 — Download**: Update status to `DOWNLOADING`. Resolve `work_dir` (config or `tempfile.mkdtemp(prefix="audiorag_")`). Call `audio_file = await self._audio_source.download(url, work_dir)`. Upsert source with title. Update status to `DOWNLOADED`.
      - **Stage 2 — Split**: Update status to `SPLITTING`. Call `audio_parts = await self._splitter.split_if_needed(audio_file, config.audio_split_max_size_mb)`. Log number of parts.
      - **Stage 3 — Transcribe**: Update status to `TRANSCRIBING`. For each audio part, call `segments = await self._stt.transcribe(part_path, config.stt_language)`. If multiple parts, adjust timestamps of subsequent parts: add cumulative offset. Merge all segments into one list. Update status to `TRANSCRIBED`.
      - **Stage 4 — Chunk**: Update status to `CHUNKING`. Call `chunks = chunk_transcription(all_segments, config.chunk_duration_seconds, url, audio_file.video_title)`. Store chunks in SQLite via `self._state.store_chunks(source_id, chunks)`. Update status to `CHUNKED`.
      - **Stage 5 — Embed**: Update status to `EMBEDDING`. Extract texts from chunks. Call `embeddings = await self._embedder.embed(texts)`. Generate chunk IDs. Store in vector store via `self._vector_store.add(ids, embeddings, metadatas, documents)`. Metadata for each chunk: `{"start_time": c.start_time, "end_time": c.end_time, "source_url": c.source_url, "video_title": c.video_title}`. Update status to `EMBEDDED`.
      - **Stage 6 — Complete**: Update status to `COMPLETED`.
      - **Cleanup**: If `config.cleanup_audio`, delete the work_dir (use `shutil.rmtree`)
      - **Error handling**: Wrap entire pipeline in try/except. On any exception, update status to `FAILED` with `error_message=str(e)`, cleanup if needed, re-raise.
      - Use `logging.getLogger("audiorag.pipeline")`
    - `async def query(self, query: str) -> QueryResult`:
      - Call `_ensure_initialized()`
      - **Step 1 — Embed query**: `query_embedding = (await self._embedder.embed([query]))[0]`
      - **Step 2 — Retrieve**: `raw_results = await self._vector_store.query(query_embedding, top_k=config.retrieval_top_k)`
      - If no results, return `QueryResult(answer="No relevant information found.", sources=[])`
      - **Step 3 — Rerank**: Extract documents from results. Call `reranked = await self._reranker.rerank(query, documents, top_n=config.rerank_top_n)`. Map back to original results using indices.
      - **Step 4 — Generate**: Extract reranked document texts as context. Call `answer = await self._generator.generate(query, context_texts)`.
      - **Step 5 — Build response**: Create `Source` objects from reranked results (text, start_time, end_time, source_url, video_title, relevance_score). Return `QueryResult(answer=answer, sources=sources)`.

  **Must NOT do**:
  - Do not add streaming
  - Do not add batch URL indexing
  - Do not add progress callbacks
  - Do not add conversation history
  - Do not add middleware or hooks
  - Do not catch and silence exceptions — update state to FAILED then re-raise

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Central orchestration logic tying all components together — most complex task
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: T12
  - **Blocked By**: T4, T5, T6, T7, T8, T9, T10

  **References**:

  **Pattern References**:
  - `src/audiorag/config.py:AudioRAGConfig` — config object passed to constructor, provides all settings
  - `src/audiorag/state.py:StateManager` — state tracking for pipeline stages
  - `src/audiorag/chunking.py:chunk_transcription` — chunking function to call in Stage 4

  **API/Type References**:
  - `src/audiorag/protocols/__init__.py` — all 6 Protocol types for constructor params
  - `src/audiorag/models.py:QueryResult` — return type for `query()`
  - `src/audiorag/models.py:Source` — individual source in QueryResult
  - `src/audiorag/models.py:AudioFile` — return from audio_source.download()
  - `src/audiorag/models.py:IndexingStatus` — enum for state updates
  - `src/audiorag/models.py:ChunkMetadata` — output from chunking, input to state.store_chunks()

  **Provider References** (default instantiation):
  - `src/audiorag/providers/youtube_scraper.py:YouTubeScraper` — default AudioSourceProvider
  - `src/audiorag/providers/openai_stt.py:OpenAISTTProvider` — default STTProvider
  - `src/audiorag/providers/openai_embeddings.py:OpenAIEmbeddingProvider` — default EmbeddingProvider
  - `src/audiorag/providers/chromadb_store.py:ChromaDBVectorStore` — default VectorStoreProvider
  - `src/audiorag/providers/openai_generation.py:OpenAIGenerationProvider` — default GenerationProvider
  - `src/audiorag/providers/cohere_reranker.py:CohereReranker` — default RerankerProvider (when cohere_api_key set)
  - `src/audiorag/providers/passthrough_reranker.py:PassthroughReranker` — fallback RerankerProvider
  - `src/audiorag/providers/audio_splitter.py:AudioSplitter` — audio splitting utility

  **WHY Each Reference Matters**:
  - Config is the single source of all settings — the pipeline reads everything from it
  - StateManager is called at every stage transition — the pipeline is the primary consumer
  - Each provider file is needed for default instantiation in `__init__` when user passes None
  - Models are used for every data handoff between stages
  - AudioSplitter is a utility (not a Protocol) used between download and transcription

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Pipeline instantiates with defaults from config
    Tool: Bash
    Steps:
      1. Run: AUDIORAG_OPENAI_API_KEY=fake AUDIORAG_DATABASE_PATH=/tmp/test_pipeline.db uv run python -c "
         from audiorag.pipeline import AudioRAGPipeline
         from audiorag.config import AudioRAGConfig
         config = AudioRAGConfig()
         pipeline = AudioRAGPipeline(config)
         print(type(pipeline).__name__)
         "
      2. Assert: output is "AudioRAGPipeline"
    Expected Result: Pipeline creates without error

  Scenario: Pipeline accepts custom providers
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.pipeline import AudioRAGPipeline
         from audiorag.config import AudioRAGConfig
         from audiorag.providers.passthrough_reranker import PassthroughReranker
         config = AudioRAGConfig()
         pipeline = AudioRAGPipeline(config, reranker=PassthroughReranker())
         print('Custom provider accepted')
         "
      2. Assert: output is "Custom provider accepted"
    Expected Result: Custom providers work via constructor injection

  Scenario: Pipeline has index and query methods that are async
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import inspect
         from audiorag.pipeline import AudioRAGPipeline
         print(inspect.iscoroutinefunction(AudioRAGPipeline.index))
         print(inspect.iscoroutinefunction(AudioRAGPipeline.query))
         "
      2. Assert: both print "True"
    Expected Result: Both public methods are coroutines

  Scenario: Pipeline query returns QueryResult type
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import inspect
         from audiorag.pipeline import AudioRAGPipeline
         sig = inspect.signature(AudioRAGPipeline.query)
         print(sig.return_annotation)
         "
      2. Assert: output references "QueryResult"
    Expected Result: Return type annotation is correct
  ```

  **Commit**: YES
  - Message: `feat(pipeline): add main AudioRAGPipeline orchestrator with index and query`
  - Files: `src/audiorag/pipeline.py`

---

- [ ] 12. Public API Surface + Final Verification — `__init__.py` and providers `__init__.py`

  **What to do**:
  - Update `src/audiorag/__init__.py`:
    - `__version__ = "0.1.0"`
    - Import and re-export: `AudioRAGPipeline` from `.pipeline`, `AudioRAGConfig` from `.config`, `QueryResult` and `Source` from `.models`
    - Define `__all__ = ["AudioRAGPipeline", "AudioRAGConfig", "QueryResult", "Source", "__version__"]`
  - Update `src/audiorag/providers/__init__.py`:
    - Lazy imports with try/except for optional dependencies:
      ```python
      def __getattr__(name):
          if name == "YouTubeScraper":
              from .youtube_scraper import YouTubeScraper
              return YouTubeScraper
          # ... similar for each provider
          raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
      ```
    - This pattern avoids ImportError when optional deps aren't installed
    - List all provider names in `__all__`
  - Update `src/audiorag/protocols/__init__.py` if not already done in T3: ensure clean re-exports
  - Run `uv sync --all-extras` to install all dependencies
  - Run `uv build` to verify the package builds correctly
  - Verify the full public API surface is importable

  **Must NOT do**:
  - Do not add convenience functions beyond the class exports
  - Do not add module-level side effects (no auto-configuration, no env var reading at import time)
  - Do not eagerly import provider implementations — use lazy loading to avoid dependency errors

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wiring up exports and running verification — no new logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (after T11)
  - **Blocks**: None (final task)
  - **Blocked By**: T11

  **References**:

  **Pattern References**:
  - Python lazy `__getattr__` pattern for optional dependency handling
  - `__all__` for explicit public API surface

  **API/Type References**:
  - `src/audiorag/pipeline.py:AudioRAGPipeline` — main class to export
  - `src/audiorag/config.py:AudioRAGConfig` — config class to export
  - `src/audiorag/models.py:QueryResult` — response model to export
  - `src/audiorag/models.py:Source` — source model to export

  **WHY Each Reference Matters**:
  - These 4 classes are the entire public API — users interact with nothing else
  - Lazy loading in providers/__init__.py prevents ImportError when e.g. chromadb isn't installed

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Public API imports work
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag import AudioRAGPipeline, AudioRAGConfig, QueryResult, Source, __version__
         print(__version__)
         print(AudioRAGPipeline.__name__)
         print(AudioRAGConfig.__name__)
         print(QueryResult.__name__)
         print(Source.__name__)
         "
      2. Assert: version is "0.1.0"
      3. Assert: all class names print correctly
    Expected Result: Full public API importable

  Scenario: Protocol imports work
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.protocols import STTProvider, EmbeddingProvider, VectorStoreProvider, GenerationProvider, RerankerProvider, AudioSourceProvider
         print('All 6 protocols imported')
         "
      2. Assert: output is "All 6 protocols imported"
    Expected Result: All protocols accessible

  Scenario: Provider imports work with lazy loading
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         from audiorag.providers import YouTubeScraper, OpenAISTTProvider, OpenAIEmbeddingProvider, ChromaDBVectorStore, OpenAIGenerationProvider, CohereReranker, PassthroughReranker
         print('All 7 providers imported')
         "
      2. Assert: output is "All 7 providers imported"
    Expected Result: All providers accessible via lazy imports

  Scenario: Package builds successfully
    Tool: Bash
    Steps:
      1. Run: uv build
      2. Assert: exit code 0
      3. Assert: dist/ contains audiorag-0.1.0-py3-none-any.whl
    Expected Result: Valid wheel file produced

  Scenario: Protocols submodule accessible from top level
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
         import audiorag.protocols
         import audiorag.providers
         print('Submodules accessible')
         "
      2. Assert: output is "Submodules accessible"
    Expected Result: Package structure navigable
  ```

  **Commit**: YES
  - Message: `feat(api): wire up public API surface with lazy provider loading`
  - Files: `src/audiorag/__init__.py`, `src/audiorag/providers/__init__.py`

---

## Commit Strategy

| After Task(s) | Message | Key Files | Verification |
|---------------|---------|-----------|--------------|
| T1 | `feat(scaffold): set up src layout, pyproject.toml with uv_build, and dependency groups` | pyproject.toml, src/audiorag/**/__init__.py, .gitignore | `ls src/audiorag/` shows structure |
| T2 + T3 | `feat(core): add Pydantic data models, config, and protocol definitions` | models.py, config.py, protocols/*.py | `uv run python -c "from audiorag.models import QueryResult"` |
| T4 | `feat(state): add SQLite state management with aiosqlite and WAL mode` | state.py | SQLite schema verification script |
| T5 | `feat(chunking): add time-based text chunking with no overlap` | chunking.py | Chunking verification script |
| T6 | `feat(providers): add YouTube scraper with yt-dlp and audio splitter with pydub` | providers/youtube_scraper.py, providers/audio_splitter.py | Protocol isinstance check |
| T7-T10 | `feat(providers): add OpenAI STT, embeddings, generation, ChromaDB, and Cohere reranker providers` | providers/openai_*.py, providers/chromadb_store.py, providers/cohere_reranker.py, providers/passthrough_reranker.py | Protocol isinstance checks |
| T11 | `feat(pipeline): add main AudioRAGPipeline orchestrator with index and query` | pipeline.py | Pipeline instantiation check |
| T12 | `feat(api): wire up public API surface with lazy provider loading` | __init__.py files | `uv build` + full import check |

---

## Success Criteria

### Verification Commands
```bash
# Package builds
uv build  # Expected: dist/audiorag-0.1.0-py3-none-any.whl

# Full public API import
uv run python -c "from audiorag import AudioRAGPipeline, AudioRAGConfig, QueryResult, Source, __version__; print(__version__)"
# Expected: 0.1.0

# All protocols importable
uv run python -c "from audiorag.protocols import STTProvider, EmbeddingProvider, VectorStoreProvider, GenerationProvider, RerankerProvider, AudioSourceProvider; print('OK')"
# Expected: OK

# All providers importable
uv run python -c "from audiorag.providers import YouTubeScraper, OpenAISTTProvider, OpenAIEmbeddingProvider, ChromaDBVectorStore, OpenAIGenerationProvider, CohereReranker, PassthroughReranker; print('OK')"
# Expected: OK

# Protocol check works
uv run python -c "from audiorag.protocols import EmbeddingProvider; class E:\n async def embed(self,t):return []\nprint(isinstance(E(),EmbeddingProvider))"
# Expected: True

# Config loads from env
AUDIORAG_OPENAI_API_KEY=test uv run python -c "from audiorag.config import AudioRAGConfig; print(AudioRAGConfig().openai_api_key)"
# Expected: test
```

### Final Checklist
- [ ] All "Must Have" items present in the codebase
- [ ] All "Must NOT Have" items absent from the codebase
- [ ] `uv build` produces a valid wheel
- [ ] All 6 protocols pass `isinstance` checks with dummy classes
- [ ] All 7 default providers pass `isinstance` checks against their protocols
- [ ] SQLite schema creates with WAL mode
- [ ] Config loads from `AUDIORAG_` prefixed env vars
- [ ] `AudioRAGPipeline` accepts custom providers via constructor
- [ ] `QueryResult.sources[].youtube_timestamp_url` computes correctly
- [ ] No `print()` statements — only `logging.getLogger("audiorag.*")`
- [ ] No eager provider imports — lazy loading via `__getattr__`
