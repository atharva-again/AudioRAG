# AudioRAG v2: Architectural Redesign

## TL;DR

> **Quick Summary**: Full architectural redesign of AudioRAG from a rigid YouTube-centric pipeline into a layered library where composable primitives (transcribe, chunk, embed, store) work standalone AND compose into a convenient pipeline. Goal: "people doing RAG on audio shouldn't have to reinvent the wheel."
> 
> **Deliverables**:
> - Source-agnostic core models (no YouTube-centrism)
> - Decomposed config system (no God Config)
> - Thin provider wrappers with shared base classes (~40-80 lines each)
> - All providers migrated from `providers/` to domain modules (`transcribe/`, `embed/`, `store/`, `generate/`)
> - Hookable, composable pipeline (<200 lines)
> - Clean public API with proper lazy imports and clear error messages
> - Protocol conformance tests as contracts
> - Updated tests for all components
> 
> **Estimated Effort**: Large (16 tasks)
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: Task 1 → Task 3 → Task 5 → Tasks 6-11 → Task 12 → Task 15

---

## Context

### Original Request
Make AudioRAG the go-to library for audio/video RAG. Assess adoptability, keeping up with advancements, and extensibility. Reference Pipecat and other best-in-class frameworks. Be honest about whether we're doing too many things.

### Interview Summary
**Key Discussions**:
- User's core mission: "People doing RAG on audio shouldn't reinvent the wheel"
- Pre-launch (v0.1.0), full freedom to break anything
- All providers maintained in-house (with thin wrapper pattern for sustainability)
- Audio-first, video-aware scope
- Solo maintainer hoping to grow community — architecture must invite contributions
- Tests after implementation

**Research Findings**:
- **No dedicated open-source audio RAG library exists** — clear market opportunity
- **Pipecat** (10K stars): Frame-based composable architecture, services as thin wrappers on FrameProcessor, per-provider optional deps
- **LangChain failures**: Over-abstraction, breaking changes, framework tax, dependency bloat. 45% of devs never use in production
- **Loved frameworks** (LiteLLM, Instructor, Haystack 2.0): Thin core, composability, work WITH user's code, stability over features
- **Multimodal threat**: Real but not fatal — text-based RAG wins for cost, precision, and scale

### Metis Review
**Identified Gaps** (addressed):
- Identity crisis is worse than initially assessed — THREE competing module systems coexist (old `providers/`, new `core/`, half-built domain modules)
- `video_title` appears 19 times across 5 files — deeply embedded, not a cosmetic rename
- Provider duplication already exists (`embed/openai.py` vs `providers/openai_embeddings.py`)
- Structured exceptions exist but aren't used — providers raise `RuntimeError`
- `try/except ImportError` → `None` pattern gives terrible error messages (`TypeError: 'NoneType' is not callable`)
- StateManager's `IndexingStatus` enum IS the pipeline — tightly coupled
- Chunking is purely time-based — need pluggable strategy protocol from day one
- SQLite schema migration: pre-launch = document "delete your .db file"

---

## Work Objectives

### Core Objective
Transform AudioRAG from a rigid YouTube-centric monolithic pipeline into a layered library with composable primitives (Layer 1) and a convenient pipeline (Layer 2), making it the go-to tool for audio RAG.

### Concrete Deliverables
- New module structure with domain-organized providers
- Source-agnostic core data models
- Decomposed configuration system
- Base classes per provider category with shared retry/logging/error handling
- All 14+ providers as thin wrappers in domain modules
- Composable, hookable pipeline under 200 lines
- Clean public API with proper lazy imports
- Protocol conformance tests
- Updated test suite

### Definition of Done
- [ ] `uv run pytest` passes with all tests
- [ ] `uv run ruff check src/ tests/` returns no errors
- [ ] Each primitive works standalone without pipeline imports
- [ ] No `video_title` or `youtube` references in core models
- [ ] `AudioRAGConfig` under 60 lines (orchestration only)
- [ ] Each provider file under 80 lines (excluding YouTube scraper)
- [ ] No `RuntimeError` raises in any provider — all use `ProviderError`
- [ ] Zero `TODO` stubs in shipped modules
- [ ] `providers/` directory deleted entirely

### Must Have
- Backward-compatible top-level API: `from audiorag import AudioRAG` still works
- Every shipped module has working implementations (no empty stubs)
- Each class importable from exactly one canonical path
- Clear error messages when optional deps missing (`Install audiorag[voyage] to use VoyageEmbedder`)
- Source-agnostic core models with extensible metadata
- Pluggable chunking strategy protocol

### Must NOT Have (Guardrails)
- No phantom/stub modules — if it's not implemented, it doesn't ship
- No God Config — providers take plain arguments, config objects are pipeline-layer convenience
- No YouTube references in core models — source-specific data lives in source providers
- No bare `RuntimeError`/`Exception` in providers — must use `ProviderError`
- No event bus / middleware / hook system — customization via function replacement
- No CLI, no web UI — pure library
- No new source types beyond YouTube + local file + HTTP URL
- No chunking strategy implementations beyond time-based — just the protocol
- No provider auto-detection/auto-configuration — explicit only
- No new hard dependencies

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks are verifiable without any human action.
> Every criterion is executable by running a command or using a tool.

### Test Decision
- **Infrastructure exists**: YES (pytest, pytest-asyncio, 80% coverage target)
- **Automated tests**: Tests-after (implementation first, tests follow each component)
- **Framework**: pytest + pytest-asyncio (existing)

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

Every task includes concrete verification scenarios using Bash commands.

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| Module structure | Bash (python -c, find) | Import checks, file existence |
| Models | Bash (pytest, grep) | Test run, content assertions |
| Config | Bash (python -c, pytest) | Instantiation, field checks |
| Providers | Bash (pytest, python -c) | Protocol conformance, import isolation |
| Pipeline | Bash (pytest) | Integration tests |
| Public API | Bash (python -c) | Import verification |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Define new module structure blueprint
└── Task 2: Write protocol conformance tests

Wave 2 (After Wave 1):
├── Task 3: Source-agnostic core models
└── Task 4: Decompose God Config + fix import paths

Wave 3 (After Wave 2 — LARGEST WAVE, parallelize internally):
├── Task 5:  Create base classes per provider category
├── Task 6:  Migrate STT providers → transcribe/
├── Task 7:  Migrate embedding providers → embed/
├── Task 8:  Migrate vector store providers → store/
├── Task 9:  Migrate generation providers → generate/
├── Task 10: Migrate reranker → rerank/
└── Task 11: Migrate audio sources → source/ + add local/URL

Wave 4 (After Wave 3):
├── Task 12: Rewrite pipeline as Layer 2 orchestrator
├── Task 13: Update StateManager for composable pipeline
└── Task 14: Clean public API + lazy imports with clear errors

Wave 5 (After Wave 4):
├── Task 15: Delete dead code + old providers/
└── Task 16: Comprehensive test suite + lint/type check pass

Critical Path: Task 1 → Task 3 → Task 5 → Task 6 → Task 12 → Task 15
Parallel Speedup: ~45% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3, 4, 5-11 | 2 |
| 2 | None | 6-11, 16 | 1 |
| 3 | 1 | 5-11, 12, 13 | 4 |
| 4 | 1 | 5-11, 12, 14 | 3 |
| 5 | 3, 4 | 6-11 | None |
| 6 | 5 | 12 | 7, 8, 9, 10, 11 |
| 7 | 5 | 12 | 6, 8, 9, 10, 11 |
| 8 | 5 | 12 | 6, 7, 9, 10, 11 |
| 9 | 5 | 12 | 6, 7, 8, 10, 11 |
| 10 | 5 | 12 | 6, 7, 8, 9, 11 |
| 11 | 5 | 12 | 6, 7, 8, 9, 10 |
| 12 | 6-11 | 15 | 13, 14 |
| 13 | 3 | 15 | 12, 14 |
| 14 | 4, 12 | 15 | 13 |
| 15 | 12, 13, 14 | 16 | None |
| 16 | 15 | None | None |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2 | Task 1: category=quick; Task 2: category=unspecified-low |
| 2 | 3, 4 | Both: category=unspecified-high |
| 3 | 5-11 | Task 5: category=unspecified-high; Tasks 6-11: category=quick (thin wrappers) |
| 4 | 12-14 | Task 12: category=deep; Tasks 13-14: category=unspecified-low |
| 5 | 15, 16 | Task 15: category=quick; Task 16: category=unspecified-high |

---

## TODOs

### WAVE 1: Foundations

- [x] 1. Define new module structure blueprint

  **What to do**:
  - Create the target directory structure as empty `__init__.py` files (just module stubs to establish the layout)
  - The new structure:
    ```
    src/audiorag/
    ├── __init__.py              # Public API: AudioRAG, core types
    ├── core/
    │   ├── __init__.py          # Core exports
    │   ├── models.py            # Source-agnostic data models
    │   ├── config.py            # Minimal orchestration config
    │   ├── exceptions.py        # Exception hierarchy (keep as-is)
    │   ├── state.py             # State management
    │   ├── logging_config.py    # Structured logging (keep as-is)
    │   ├── retry_config.py      # Retry config (keep as-is)
    │   └── protocols/           # Protocol definitions (keep as-is)
    │       ├── __init__.py
    │       ├── stt.py
    │       ├── embedding.py
    │       ├── vector_store.py
    │       ├── generation.py
    │       ├── reranker.py
    │       ├── audio_source.py
    │       └── chunking.py      # NEW: ChunkingStrategy protocol
    ├── transcribe/              # STT providers (replaces providers/openai_stt.py etc.)
    │   ├── __init__.py          # Lazy imports with clear error messages
    │   ├── _base.py             # BaseTranscriber (shared retry/logging/error)
    │   ├── openai.py            # Thin wrapper
    │   ├── deepgram.py          # Thin wrapper
    │   ├── assemblyai.py        # Thin wrapper
    │   └── groq.py              # Thin wrapper
    ├── embed/                   # Embedding providers
    │   ├── __init__.py          # Lazy imports
    │   ├── _base.py             # BaseEmbedder
    │   ├── openai.py            # Thin wrapper
    │   ├── voyage.py            # Thin wrapper
    │   └── cohere.py            # Thin wrapper
    ├── store/                   # Vector store providers
    │   ├── __init__.py          # Lazy imports
    │   ├── _base.py             # BaseVectorStore
    │   ├── chromadb.py          # Thin wrapper
    │   ├── pinecone.py          # Thin wrapper
    │   ├── weaviate.py          # Thin wrapper
    │   └── supabase.py          # Thin wrapper
    ├── generate/                # LLM generation providers
    │   ├── __init__.py          # Lazy imports
    │   ├── _base.py             # BaseGenerator
    │   ├── openai.py            # Thin wrapper
    │   ├── anthropic.py         # Thin wrapper
    │   └── gemini.py            # Thin wrapper
    ├── rerank/                  # Reranker providers
    │   ├── __init__.py          # Lazy imports
    │   ├── _base.py             # BaseReranker
    │   ├── cohere.py            # Thin wrapper
    │   └── passthrough.py       # No-op reranker
    ├── source/                  # Audio source providers
    │   ├── __init__.py          # Lazy imports
    │   ├── _base.py             # BaseAudioSource
    │   ├── youtube.py           # YouTube downloader (kept complex — it IS complex)
    │   ├── local.py             # NEW: Local file/directory source
    │   └── url.py               # NEW: Direct HTTP URL download
    ├── chunk/                   # Chunking strategies
    │   ├── __init__.py
    │   ├── time_based.py        # Current time-based chunker (moved from chunking.py)
    │   └── _protocol.py         # ChunkingStrategy protocol (also in core/protocols/)
    └── pipeline.py              # Layer 2: Composable pipeline orchestrator
    ```
  - Each domain `__init__.py` uses `__getattr__` pattern for lazy imports with clear error messages:
    ```python
    def __getattr__(name):
        if name == "VoyageEmbedder":
            try:
                from audiorag.embed.voyage import VoyageEmbeddingProvider
                return VoyageEmbeddingProvider
            except ImportError:
                raise ImportError(
                    "VoyageEmbedder requires 'voyageai'. "
                    "Install with: pip install audiorag[voyage]"
                ) from None
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ```
  - Remove the old `ingest/`, `query/`, `retrieve/` stub directories (they were TODO stubs)

  **Must NOT do**:
  - Don't implement any provider logic — just establish the file layout
  - Don't delete old `providers/` yet — that happens in Task 15
  - Don't rename any classes yet — just create the skeleton

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Creating directory structure and empty/skeleton files is straightforward
  - **Skills**: []
    - No special skills needed — file operations only

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 3, 4, 5-11
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/audiorag/embed/__init__.py:16-35` — Current lazy import pattern (the try/except approach to REPLACE with `__getattr__`)
  - `src/audiorag/store/__init__.py:19-47` — Another lazy loading approach (confused dual-export to AVOID)
  - `src/audiorag/embed/voyage.py` — Example of a domain module provider file (structure to follow)

  **API/Type References**:
  - `src/audiorag/core/protocols/__init__.py` — All protocol names that domain modules must export implementations for

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: New directory structure exists
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         expected = [
             'src/audiorag/transcribe/__init__.py',
             'src/audiorag/transcribe/_base.py',
             'src/audiorag/embed/__init__.py',
             'src/audiorag/embed/_base.py',
             'src/audiorag/store/__init__.py',
             'src/audiorag/store/_base.py',
             'src/audiorag/generate/__init__.py',
             'src/audiorag/generate/_base.py',
             'src/audiorag/rerank/__init__.py',
             'src/audiorag/rerank/_base.py',
             'src/audiorag/source/__init__.py',
             'src/audiorag/source/_base.py',
             'src/audiorag/chunk/__init__.py',
             'src/audiorag/chunk/time_based.py',
             'src/audiorag/core/protocols/chunking.py',
         ]
         for path in expected:
             assert pathlib.Path(path).exists(), f'Missing: {path}'
         print('PASS: All expected files exist')
         "
    Expected Result: All files exist
    Evidence: Terminal output captured

  Scenario: Old stub directories removed
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         stubs = ['src/audiorag/ingest', 'src/audiorag/query', 'src/audiorag/retrieve']
         for path in stubs:
             assert not pathlib.Path(path).exists(), f'Stub still exists: {path}'
         print('PASS: Old stub directories removed')
         "
    Expected Result: No stub directories
    Evidence: Terminal output captured

  Scenario: Lazy import __getattr__ pattern in domain __init__.py
    Tool: Bash
    Steps:
      1. grep -l "__getattr__" src/audiorag/embed/__init__.py src/audiorag/transcribe/__init__.py src/audiorag/store/__init__.py src/audiorag/generate/__init__.py
    Expected Result: __getattr__ found in all domain module __init__.py files
    Evidence: grep output captured
  ```

  **Commit**: YES
  - Message: `refactor(structure): establish new domain-organized module layout`
  - Files: All new `__init__.py`, `_base.py` skeleton files; deleted `ingest/`, `query/`, `retrieve/`
  - Pre-commit: `uv run ruff check src/audiorag/`

---

- [x] 2. Write protocol conformance tests

  **What to do**:
  - Create `tests/test_protocol_conformance.py` with tests that verify ANY implementation of each protocol works correctly
  - These tests define the **contract** — they survive the entire redesign
  - Test each protocol:
    - `STTProvider`: accepts `Path` + optional language, returns `list[TranscriptionSegment]`
    - `EmbeddingProvider`: accepts `list[str]`, returns `list[list[float]]`
    - `VectorStoreProvider`: `add()`, `query()`, `delete_by_source()` with correct types
    - `GenerationProvider`: accepts query + context, returns `str`
    - `RerankerProvider`: accepts query + documents + top_n, returns `list[tuple[int, float]]`
    - `AudioSourceProvider`: accepts URL + output_dir + format, returns `AudioFile`
  - NEW: Add `ChunkingStrategy` protocol test (accepts `list[TranscriptionSegment]`, returns `list[ChunkMetadata]`)
  - Use mock/fake implementations that satisfy the protocol
  - Verify `runtime_checkable` works with `isinstance()` checks
  - Verify return type shapes (not values — these are contract tests, not integration tests)

  **Must NOT do**:
  - Don't test specific providers (OpenAI, Voyage, etc.) — just the protocols
  - Don't require external API keys
  - Don't modify existing test files

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Standard test writing, follows existing test patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 6-11 (providers must pass conformance), Task 16
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `tests/test_protocols.py` — Existing protocol tests (extend, don't replace; study the patterns used here)

  **API/Type References**:
  - `src/audiorag/core/protocols/stt.py:8-11` — STTProvider protocol signature
  - `src/audiorag/core/protocols/embedding.py:5-6` — EmbeddingProvider protocol signature
  - `src/audiorag/core/protocols/vector_store.py:5-16` — VectorStoreProvider protocol signature
  - `src/audiorag/core/protocols/generation.py:5-6` — GenerationProvider protocol signature
  - `src/audiorag/core/protocols/reranker.py:5-8` — RerankerProvider protocol signature
  - `src/audiorag/core/protocols/audio_source.py` — AudioSourceProvider protocol signature
  - `src/audiorag/core/models.py:9-17` — ChunkMetadata model (chunking output type)
  - `src/audiorag/core/models.py:52-57` — TranscriptionSegment model (chunking input type)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Protocol conformance tests pass
    Tool: Bash
    Steps:
      1. uv run pytest tests/test_protocol_conformance.py -v
    Expected Result: All tests pass, including new ChunkingStrategy protocol test
    Evidence: pytest output captured

  Scenario: Tests use mock implementations (no API keys needed)
    Tool: Bash
    Steps:
      1. AUDIORAG_OPENAI_API_KEY="" uv run pytest tests/test_protocol_conformance.py -v
    Expected Result: All tests pass without any API keys
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `test(protocols): add protocol conformance tests as redesign contracts`
  - Files: `tests/test_protocol_conformance.py`
  - Pre-commit: `uv run pytest tests/test_protocol_conformance.py`

---

### WAVE 2: Core Foundations

- [x] 3. Make core models source-agnostic

  **What to do**:
  - Rename `video_title` → `title` in `ChunkMetadata`, `Source`, `AudioFile`
  - Remove `youtube_timestamp_url` computed field from `Source` — move to a utility function `audiorag.source.youtube.youtube_timestamp_url(source_url, start_time)` or keep as an optional method
  - Add `metadata: dict[str, Any] = {}` field to `ChunkMetadata` and `Source` for extensible source-specific data
  - Update `chunk_transcription()` signature: `video_title` → `title`
  - Update `AudioFile` model: `video_title` → `title`
  - Add `ChunkingStrategy` protocol to `src/audiorag/core/protocols/chunking.py`:
    ```python
    @runtime_checkable
    class ChunkingStrategy(Protocol):
        def chunk(
            self, segments: list[TranscriptionSegment], source_url: str, title: str
        ) -> list[ChunkMetadata]: ...
    ```
  - Use `ast_grep_search` to find ALL `video_title` references and update them
  - Update all callers in pipeline.py, providers, and tests
  - Document: "If you have existing SQLite databases, delete them and re-index" (pre-launch, clean break)

  **Must NOT do**:
  - Don't add new model classes (no `VideoSource`, `PodcastSource` subclasses yet)
  - Don't change the protocol signatures (STTProvider, EmbeddingProvider, etc.) — only models
  - Don't touch vector store metadata schema migration (handled by documentation)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Cross-cutting change touching 5+ files and 19+ occurrences. Requires careful find-and-replace with semantic understanding.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 4)
  - **Blocks**: Tasks 5-11, 12, 13
  - **Blocked By**: Task 1 (needs chunking protocol file location)

  **References**:

  **Pattern References**:
  - `src/audiorag/core/models.py:9-74` — All current models to modify (ChunkMetadata, Source, QueryResult, AudioFile, TranscriptionSegment, IndexingStatus)
  - `src/audiorag/chunking.py:6-11` — `chunk_transcription()` function signature that takes `video_title`

  **API/Type References**:
  - `src/audiorag/core/models.py:29-33` — `Source.youtube_timestamp_url` computed field to remove
  - `src/audiorag/core/models.py:43-49` — `AudioFile.video_title` field to rename

  **Tool guidance**:
  - Use `ast_grep_search(pattern="video_title", lang="python")` to find all 19 occurrences
  - Use `lsp_find_references` on `ChunkMetadata.video_title` to map callers

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: No YouTube references in core models
    Tool: Bash
    Steps:
      1. python3 -c "
         content = open('src/audiorag/core/models.py').read()
         assert 'youtube' not in content.lower(), 'YouTube reference found'
         assert 'video_title' not in content, 'video_title found'
         print('PASS: Core models are source-agnostic')
         "
    Expected Result: No YouTube-specific references
    Evidence: Terminal output

  Scenario: ChunkingStrategy protocol exists
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.core.protocols.chunking import ChunkingStrategy
         from typing import runtime_checkable
         assert hasattr(ChunkingStrategy, 'chunk'), 'Missing chunk method'
         print('PASS: ChunkingStrategy protocol defined')
         "
    Expected Result: Protocol importable and has chunk method
    Evidence: Terminal output

  Scenario: Existing tests still pass after rename
    Tool: Bash
    Steps:
      1. uv run pytest tests/ -v --tb=short
    Expected Result: All tests pass (may need updates for video_title → title)
    Evidence: pytest output
  ```

  **Commit**: YES
  - Message: `refactor(models): make core models source-agnostic, add ChunkingStrategy protocol`
  - Files: `src/audiorag/core/models.py`, `src/audiorag/chunking.py`, `src/audiorag/core/protocols/chunking.py`, `src/audiorag/pipeline.py`, affected tests
  - Pre-commit: `uv run pytest && uv run ruff check src/`

---

- [ ] 4. Decompose God Config + fix import paths

  **What to do**:
  - Split `AudioRAGConfig` (239 lines) into:
    - `AudioRAGConfig` (pipeline orchestration only): provider selection strings, retrieval settings, logging, retry config, state DB path. Target: <60 lines.
    - Provider-specific settings stay as constructor kwargs (providers take `api_key: str`, `model: str` — NOT config objects)
    - Remove `get_stt_model()`, `get_embedding_model()`, `get_generation_model()` methods — model fallback logic moves to provider constructors as default parameter values
  - Fix all import paths in `providers/*.py` files:
    - `from audiorag.logging_config import get_logger` → `from audiorag.core.logging_config import get_logger`
    - `from audiorag.models import TranscriptionSegment` → `from audiorag.core.models import TranscriptionSegment`
    - `from audiorag.retry_config import RetryConfig` → `from audiorag.core.retry_config import RetryConfig`
  - Verify no flat `audiorag.logging_config`, `audiorag.models`, `audiorag.retry_config` imports remain

  **Must NOT do**:
  - Don't create per-provider config dataclasses — providers take plain kwargs
  - Don't change provider constructors yet (that's Wave 3)
  - Don't delete any config fields — just restructure the class

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Touches 14+ provider files and the central config. Requires systematic find-and-replace.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: Tasks 5-11, 12, 14
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/audiorag/core/config.py:1-239` — The full God Config to decompose
  - `src/audiorag/providers/openai_stt.py:8-15` — Example of stale flat imports to fix
  - `src/audiorag/providers/chromadb_store.py:11-15` — Another file with stale imports

  **Tool guidance**:
  - Use `ast_grep_search(pattern="from audiorag.logging_config", lang="python")` to find all stale imports
  - Use `ast_grep_search(pattern="from audiorag.models", lang="python")` for model imports
  - Use `ast_grep_search(pattern="from audiorag.retry_config", lang="python")` for retry imports
  - Use `ast_grep_replace(dryRun=true)` to preview replacements before applying

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Config is under 60 lines
    Tool: Bash
    Steps:
      1. python3 -c "
         lines = open('src/audiorag/core/config.py').read().strip().split('\n')
         print(f'Config is {len(lines)} lines')
         assert len(lines) < 100, f'Config still too large: {len(lines)} lines'
         print('PASS')
         "
    Expected Result: Config significantly reduced
    Evidence: Terminal output

  Scenario: No stale flat imports remain
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         stale_patterns = ['from audiorag.logging_config', 'from audiorag.models import', 'from audiorag.retry_config']
         for f in pathlib.Path('src/audiorag/providers').rglob('*.py'):
             content = f.read_text()
             for pattern in stale_patterns:
                 assert pattern not in content, f'{f}: still has stale import \"{pattern}\"'
         print('PASS: No stale imports in providers/')
         "
    Expected Result: All imports use audiorag.core.* paths
    Evidence: Terminal output

  Scenario: get_stt_model / get_embedding_model / get_generation_model removed
    Tool: Bash
    Steps:
      1. python3 -c "
         content = open('src/audiorag/core/config.py').read()
         assert 'get_stt_model' not in content, 'get_stt_model still exists'
         assert 'get_embedding_model' not in content, 'get_embedding_model still exists'
         assert 'get_generation_model' not in content, 'get_generation_model still exists'
         print('PASS: Provider-specific methods removed from config')
         "
    Expected Result: No provider-specific methods in config
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `refactor(config): decompose God Config, fix stale import paths across providers`
  - Files: `src/audiorag/core/config.py`, all `src/audiorag/providers/*.py`
  - Pre-commit: `uv run ruff check src/ && uv run pytest`

---

### WAVE 3: Provider Migration

- [ ] 5. Create base classes per provider category

  **What to do**:
  - Create `_base.py` in each domain module with a base class that handles:
    - Retry decorator setup (shared `_get_retry_decorator()` method)
    - Structured logging (shared `_logger` setup with provider name binding)
    - Error wrapping (catch SDK exceptions, wrap in `ProviderError` with `retryable` flag)
    - Config validation (e.g., assert api_key provided)
  - Base classes:
    - `transcribe/_base.py` → `BaseTranscriber(api_key, model, retry_config)`
    - `embed/_base.py` → `BaseEmbedder(api_key, model, retry_config)`
    - `store/_base.py` → `BaseVectorStore(retry_config)` (no api_key — varies per store)
    - `generate/_base.py` → `BaseGenerator(api_key, model, retry_config)`
    - `rerank/_base.py` → `BaseReranker(retry_config)` 
    - `source/_base.py` → `BaseAudioSource(retry_config)`
  - Each base class handles the pattern currently duplicated across providers:
    ```python
    class BaseEmbedder:
        def __init__(self, *, api_key: str | None = None, model: str, retry_config: RetryConfig | None = None):
            self._model = model
            self._logger = get_logger(__name__).bind(provider=self._provider_name, model=model)
            self._retry_config = retry_config or RetryConfig()
        
        @property
        def _provider_name(self) -> str:
            raise NotImplementedError
        
        def _get_retry_decorator(self):
            return create_retry_decorator(config=self._retry_config, exception_types=self._retryable_exceptions)
        
        @property
        def _retryable_exceptions(self) -> tuple[type[Exception], ...]:
            return (ConnectionError, TimeoutError)
    ```
  - Providers will subclass these and only implement the actual API call

  **Must NOT do**:
  - Don't migrate providers yet — just create the base classes
  - Don't add abstract methods that force unnecessary overrides
  - Don't make base classes so complex they become their own abstraction problem

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires designing good base class abstractions that balance DRY with simplicity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (must complete before Tasks 6-11)
  - **Blocks**: Tasks 6-11
  - **Blocked By**: Tasks 3, 4

  **References**:

  **Pattern References**:
  - `src/audiorag/embed/voyage.py:39-85` — Current pattern: constructor + retry setup + logger binding (to extract into base)
  - `src/audiorag/providers/openai_stt.py:23-51` — Same pattern in STT (to extract into base)
  - `src/audiorag/providers/chromadb_store.py:23-52` — Same pattern in vector store (to extract into base)
  - `src/audiorag/core/retry_config.py` — RetryConfig and create_retry_decorator (used by all base classes)
  - `src/audiorag/core/logging_config.py` — get_logger (used by all base classes)
  - `src/audiorag/core/exceptions.py:77-105` — ProviderError class that base classes must raise

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All base classes importable
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.transcribe._base import BaseTranscriber
         from audiorag.embed._base import BaseEmbedder
         from audiorag.store._base import BaseVectorStore
         from audiorag.generate._base import BaseGenerator
         from audiorag.rerank._base import BaseReranker
         from audiorag.source._base import BaseAudioSource
         print('PASS: All base classes importable')
         "
    Expected Result: All imports succeed
    Evidence: Terminal output

  Scenario: Base classes have shared retry/logging infrastructure
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.embed._base import BaseEmbedder
         b = type('FakeEmbedder', (BaseEmbedder,), {'_provider_name': 'test'})
         # Should not crash on instantiation patterns
         assert hasattr(BaseEmbedder, '_get_retry_decorator')
         print('PASS: Base class has retry infrastructure')
         "
    Expected Result: Base classes have expected methods
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `refactor(providers): create shared base classes for all provider categories`
  - Files: All `_base.py` files in domain modules
  - Pre-commit: `uv run ruff check src/`

---

- [ ] 6. Migrate STT providers → transcribe/

  **What to do**:
  - Move and refactor each STT provider from `providers/` to `transcribe/`:
    - `providers/openai_stt.py` → `transcribe/openai.py` (extend `BaseTranscriber`, thin wrapper)
    - `providers/deepgram_stt.py` → `transcribe/deepgram.py`
    - `providers/assemblyai_stt.py` → `transcribe/assemblyai.py`
    - `providers/groq_stt.py` → `transcribe/groq.py`
  - Each provider should:
    - Extend `BaseTranscriber` from `_base.py`
    - Only implement the SDK-specific API call
    - Use `ProviderError` (not `RuntimeError`) for failures
    - Use `video_title` → `title` in any references
    - Target: ~40-60 lines per provider
  - Update `transcribe/__init__.py` with `__getattr__` lazy imports
  - Verify each provider passes protocol conformance test from Task 2

  **Must NOT do**:
  - Don't delete old `providers/openai_stt.py` etc. yet (Task 15)
  - Don't change the `STTProvider` protocol signature
  - Don't add new STT providers

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical migration — move code, extend base class, thin out duplication
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 9, 10, 11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `src/audiorag/providers/openai_stt.py:20-123` — Full OpenAI STT provider to migrate and thin out
  - `src/audiorag/providers/groq_stt.py` — Groq STT provider to migrate
  - `src/audiorag/providers/deepgram_stt.py` — Deepgram STT provider to migrate
  - `src/audiorag/providers/assemblyai_stt.py` — AssemblyAI STT provider to migrate
  - `src/audiorag/transcribe/_base.py` — Base class to extend (from Task 5)

  **API/Type References**:
  - `src/audiorag/core/protocols/stt.py:8-11` — STTProvider protocol to satisfy
  - `src/audiorag/core/models.py` — TranscriptionSegment (return type)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: STT providers importable from new location
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.transcribe import OpenAITranscriber, GroqTranscriber, DeepgramTranscriber, AssemblyAITranscriber
         print('PASS: All STT providers importable')
         "
    Expected Result: All imports succeed
    Evidence: Terminal output

  Scenario: STT providers satisfy protocol
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.core.protocols import STTProvider
         from audiorag.transcribe.openai import OpenAISTTProvider
         assert isinstance(OpenAISTTProvider(api_key='test'), STTProvider)
         print('PASS: OpenAI STT satisfies protocol')
         "
    Expected Result: Protocol check passes
    Evidence: Terminal output

  Scenario: No RuntimeError in STT providers
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         for f in pathlib.Path('src/audiorag/transcribe').glob('*.py'):
             if f.name.startswith('_'):
                 continue
             content = f.read_text()
             assert 'raise RuntimeError' not in content, f'{f.name} raises RuntimeError'
         print('PASS: No RuntimeError in STT providers')
         "
    Expected Result: All providers use ProviderError
    Evidence: Terminal output
  ```

  **Commit**: YES (groups with 7-11)
  - Message: `refactor(transcribe): migrate STT providers to domain module with thin wrappers`
  - Files: `src/audiorag/transcribe/*.py`
  - Pre-commit: `uv run ruff check src/audiorag/transcribe/`

---

- [ ] 7. Migrate embedding providers → embed/

  **What to do**:
  - Refactor existing `embed/` providers to extend `BaseEmbedder`:
    - `embed/openai.py` → extend `BaseEmbedder`, thin out
    - `embed/voyage.py` → extend `BaseEmbedder`, thin out
    - `embed/cohere.py` → extend `BaseEmbedder`, thin out
  - Remove old `providers/openai_embeddings.py`, `providers/voyage_embeddings.py`, `providers/cohere_embeddings.py` duplicates (or mark for Task 15 deletion)
  - Use `ProviderError` for all failures
  - Update `embed/__init__.py` with `__getattr__` pattern
  - Target: ~40-60 lines per provider

  **Must NOT do**:
  - Don't change the `EmbeddingProvider` protocol
  - Don't add new embedding providers

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical refactoring — extend base, thin out duplication
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6, 8, 9, 10, 11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `src/audiorag/embed/voyage.py:16-138` — Current Voyage embedder to thin out (best reference)
  - `src/audiorag/embed/openai.py` — Current OpenAI embedder to thin out
  - `src/audiorag/embed/cohere.py` — Current Cohere embedder to thin out
  - `src/audiorag/providers/openai_embeddings.py` — Old duplicate to be aware of (DO NOT use as reference)
  - `src/audiorag/embed/_base.py` — Base class to extend (from Task 5)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Embedding providers importable and protocol-compliant
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.core.protocols import EmbeddingProvider
         from audiorag.embed.voyage import VoyageEmbeddingProvider
         # Protocol check (constructor may need api_key which isn't available, 
         # so just verify class exists and has embed method)
         assert hasattr(VoyageEmbeddingProvider, 'embed')
         print('PASS')
         "
    Expected Result: Protocol satisfied
    Evidence: Terminal output

  Scenario: Each embedding provider under 80 lines
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         for f in pathlib.Path('src/audiorag/embed').glob('*.py'):
             if f.name.startswith('_') or f.name == '__init__.py':
                 continue
             lines = len(f.read_text().strip().split('\n'))
             assert lines < 80, f'{f.name} is {lines} lines (target: <80)'
             print(f'{f.name}: {lines} lines - OK')
         print('PASS')
         "
    Expected Result: All providers under 80 lines
    Evidence: Terminal output
  ```

  **Commit**: YES (groups with 6, 8-11)
  - Message: `refactor(embed): thin out embedding providers with shared base class`
  - Files: `src/audiorag/embed/*.py`
  - Pre-commit: `uv run ruff check src/audiorag/embed/`

---

- [ ] 8. Migrate vector store providers → store/

  **What to do**:
  - Migrate and refactor vector store providers:
    - `providers/chromadb_store.py` → `store/chromadb.py` (extend `BaseVectorStore`)
    - `providers/pinecone_store.py` → `store/pinecone.py`
    - `providers/weaviate_store.py` → `store/weaviate.py`
    - `providers/supabase_pgvector.py` → `store/supabase.py`
  - Note: ChromaDB uses `asyncio.to_thread` for sync bridging — this is OK per Metis guardrail G6 (ChromaDB is genuinely sync-only)
  - Update metadata key: `video_title` → `title` in store implementations (especially Weaviate schema)
  - Replace current confused dual-export in `store/__init__.py` with clean `__getattr__` pattern
  - Use `ProviderError` for all failures

  **Must NOT do**:
  - Don't change the `VectorStoreProvider` protocol (3 methods: add, query, delete_by_source)
  - Don't add schema migration logic — document clean break

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical migration
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6, 7, 9, 10, 11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `src/audiorag/providers/chromadb_store.py:20-200` — ChromaDB store to migrate (note: sync bridging is acceptable)
  - `src/audiorag/providers/pinecone_store.py` — Pinecone store to migrate
  - `src/audiorag/providers/weaviate_store.py` — Weaviate store (check for hardcoded `video_title` schema)
  - `src/audiorag/providers/supabase_pgvector.py` — Supabase store to migrate
  - `src/audiorag/store/__init__.py:19-79` — Current confused dual-export to REPLACE entirely
  - `src/audiorag/store/_base.py` — Base class to extend (from Task 5)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Vector store providers importable from new location
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.store import ChromaStore, PineconeStore, WeaviateStore, SupabaseStore
         print('PASS: All vector stores importable')
         "
    Expected Result: All imports succeed
    Evidence: Terminal output

  Scenario: No video_title in store implementations
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         for f in pathlib.Path('src/audiorag/store').glob('*.py'):
             content = f.read_text()
             assert 'video_title' not in content, f'{f.name} still references video_title'
         print('PASS: No video_title in stores')
         "
    Expected Result: All stores use generic metadata
    Evidence: Terminal output
  ```

  **Commit**: YES (groups with 6-7, 9-11)
  - Message: `refactor(store): migrate vector store providers to domain module`
  - Files: `src/audiorag/store/*.py`
  - Pre-commit: `uv run ruff check src/audiorag/store/`

---

- [ ] 9. Migrate generation providers → generate/

  **What to do**:
  - Refactor generation providers to extend `BaseGenerator`:
    - Existing `generate/__init__.py` already has lazy imports — refactor to use `__getattr__` pattern
    - `providers/openai_generation.py` → `generate/openai.py`
    - `providers/anthropic_generation.py` → `generate/anthropic.py`
    - `providers/gemini_generation.py` → `generate/gemini.py`
  - Thin out with base class. Use `ProviderError`.

  **Must NOT do**:
  - Don't change the `GenerationProvider` protocol

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6-8, 10, 11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `src/audiorag/providers/openai_generation.py` — OpenAI generation to migrate
  - `src/audiorag/providers/anthropic_generation.py` — Anthropic generation to migrate
  - `src/audiorag/providers/gemini_generation.py` — Gemini generation to migrate
  - `src/audiorag/generate/__init__.py:1-38` — Current init to replace with __getattr__ pattern
  - `src/audiorag/generate/_base.py` — Base class to extend (from Task 5)

  **Acceptance Criteria**:

  ```
  Scenario: Generation providers importable and thin
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.generate import OpenAIGenerator, AnthropicGenerator, GeminiGenerator
         print('PASS')
         "
    Expected Result: All imports succeed
    Evidence: Terminal output
  ```

  **Commit**: YES (groups with 6-8, 10-11)
  - Message: `refactor(generate): migrate generation providers to domain module`
  - Files: `src/audiorag/generate/*.py`
  - Pre-commit: `uv run ruff check src/audiorag/generate/`

---

- [ ] 10. Migrate reranker providers → rerank/

  **What to do**:
  - `providers/cohere_reranker.py` → `rerank/cohere.py` (extend `BaseReranker`)
  - `providers/passthrough_reranker.py` → `rerank/passthrough.py`
  - Update `rerank/__init__.py` with `__getattr__` pattern

  **Must NOT do**:
  - Don't change the `RerankerProvider` protocol

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6-9, 11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:
  - `src/audiorag/providers/cohere_reranker.py` — Cohere reranker to migrate
  - `src/audiorag/providers/passthrough_reranker.py` — Passthrough reranker to migrate
  - `src/audiorag/rerank/_base.py` — Base class (from Task 5)

  **Acceptance Criteria**:

  ```
  Scenario: Reranker providers importable
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.rerank import CohereReranker, PassthroughReranker
         print('PASS')
         "
    Expected Result: All imports succeed
  ```

  **Commit**: YES (groups with 6-9, 11)
  - Message: `refactor(rerank): migrate reranker providers to domain module`
  - Files: `src/audiorag/rerank/*.py`

---

- [ ] 11. Migrate audio sources → source/ + add local file & URL sources

  **What to do**:
  - `providers/youtube_scraper.py` → `source/youtube.py` (keep as-is — it's legitimately complex)
  - `providers/audio_splitter.py` → `source/splitter.py` (utility, used by pipeline)
  - Create `source/local.py` — new `LocalFileSource` that:
    - Takes a file path or directory path
    - Returns `AudioFile` with path, title (from filename), duration (using pydub if available)
    - Satisfies `AudioSourceProvider` protocol
  - Create `source/url.py` — new `URLSource` that:
    - Downloads audio from a direct HTTP URL
    - Returns `AudioFile`
    - Satisfies `AudioSourceProvider` protocol
  - Move `youtube_timestamp_url` helper to `source/youtube.py` as a standalone function
  - Update `source/__init__.py` with `__getattr__` lazy imports

  **Must NOT do**:
  - Don't add Spotify, SoundCloud, RSS, S3, or other sources — just YouTube + local + URL
  - Don't refactor the YouTube scraper internals — it works, leave it

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: YouTube is a move, but local + URL sources are new (simple) implementations
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6-10)
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `src/audiorag/providers/youtube_scraper.py` — YouTube scraper to move (keep complex — it IS complex)
  - `src/audiorag/providers/audio_splitter.py` — Audio splitter utility to move
  - `src/audiorag/source/_base.py` — Base class (from Task 5)

  **API/Type References**:
  - `src/audiorag/core/protocols/audio_source.py` — AudioSourceProvider protocol to satisfy
  - `src/audiorag/core/models.py` — AudioFile model (return type)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All source providers importable
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.source import YouTubeSource, LocalFileSource, URLSource
         print('PASS')
         "
    Expected Result: All imports succeed

  Scenario: LocalFileSource handles a real file
    Tool: Bash
    Steps:
      1. python3 -c "
         import asyncio, tempfile, pathlib
         from audiorag.source.local import LocalFileSource
         # Create a dummy audio file
         tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
         tmp.write(b'fake audio content')
         tmp.close()
         source = LocalFileSource()
         result = asyncio.run(source.download(tmp.name, pathlib.Path(tempfile.mkdtemp()), 'mp3'))
         assert result.path.exists()
         assert result.title  # Should derive from filename
         print(f'PASS: Got AudioFile with title={result.title}')
         "
    Expected Result: LocalFileSource returns AudioFile from local path
    Evidence: Terminal output
  ```

  **Commit**: YES (groups with 6-10)
  - Message: `refactor(source): migrate audio sources, add local file and URL support`
  - Files: `src/audiorag/source/*.py`
  - Pre-commit: `uv run ruff check src/audiorag/source/`

---

### WAVE 4: Pipeline & API

- [ ] 12. Rewrite pipeline as Layer 2 orchestrator

  **What to do**:
  - Rewrite `pipeline.py` to be <200 lines by:
    - Remove all 5 `_create_*_provider()` factory methods (~160 lines) — replace with a simple factory module or inline provider construction
    - Import from new domain modules instead of old `providers/`
    - Use composable primitives from Layer 1
    - Make pipeline stages hookable: accept custom transcriber, chunker, embedder, store, generator, reranker via constructor (already partially done — improve it)
    - Accept string shortcuts: `transcriber="openai"` maps to `OpenAITranscriber(api_key=config.openai_api_key)`
    - Extract provider factory logic into a separate `_factory.py` or handle inline with a clean registry pattern
  - Move `chunk_transcription` to use the `ChunkingStrategy` protocol — wrap current function in a `TimeBasedChunker` class
  - Pipeline should compose Layer 1 primitives, not contain business logic
  - Rename main class: `AudioRAGPipeline` → `AudioRAG` (shorter, cleaner)
  - Keep backward compat alias: `AudioRAGPipeline = AudioRAG`

  **Must NOT do**:
  - Don't add event hooks / middleware / before-after callbacks
  - Don't change the external API contract (`.index(url)`, `.query(question)`)
  - Don't implement pipeline branching or DAGs — keep it linear

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core architectural work — must balance simplicity with flexibility, avoid over-engineering
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 13, 14)
  - **Blocks**: Task 15
  - **Blocked By**: Tasks 6-11 (all providers must be in new locations)

  **References**:

  **Pattern References**:
  - `src/audiorag/pipeline.py:33-584` — Current monolithic pipeline to rewrite
  - `src/audiorag/pipeline.py:40-136` — Constructor + 5 factory methods to extract/simplify
  - `src/audiorag/pipeline.py:327-508` — `index()` method stages to compose from primitives
  - `src/audiorag/pipeline.py:509-583` — `query()` method to compose from primitives
  - `src/audiorag/chunking.py:6-74` — Current chunking function to wrap in TimeBasedChunker class

  **External References**:
  - Pipecat's pipeline pattern: `Pipeline([processor1, processor2, ...])` — simple list composition

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Pipeline is under 200 lines
    Tool: Bash
    Steps:
      1. python3 -c "
         lines = len(open('src/audiorag/pipeline.py').read().strip().split('\n'))
         print(f'Pipeline: {lines} lines')
         assert lines < 250, f'Pipeline too large: {lines} lines'
         print('PASS')
         "
    Expected Result: Pipeline significantly reduced
    Evidence: Terminal output

  Scenario: AudioRAG accepts string shortcuts
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.pipeline import AudioRAG
         # Should accept string provider names
         rag = AudioRAG.__new__(AudioRAG)  # Test class exists
         print('PASS: AudioRAG class exists')
         "
    Expected Result: Class importable with new name
    Evidence: Terminal output

  Scenario: Backward compat alias works
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag import AudioRAGPipeline
         print('PASS: AudioRAGPipeline still importable')
         "
    Expected Result: Old name still works
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `refactor(pipeline): rewrite as thin Layer 2 orchestrator composing primitives`
  - Files: `src/audiorag/pipeline.py`, `src/audiorag/chunk/time_based.py`
  - Pre-commit: `uv run ruff check src/audiorag/pipeline.py`

---

- [ ] 13. Update StateManager for composable pipeline

  **What to do**:
  - Update `IndexingStatus` enum to be stage-agnostic:
    - Keep: `COMPLETED`, `FAILED`
    - Generalize: Instead of `DOWNLOADING`, `TRANSCRIBING`, etc., use `IN_PROGRESS` with stage stored in metadata
    - Or: keep granular statuses but add a generic `PROCESSING` that custom stages can use
  - Update SQLite schema: `video_title` → `title` in metadata conventions (document this as a breaking change)
  - Add a `NullStateManager` class that implements the same interface but does nothing (for stateless/serverless use)
  - Keep `StateManager` as the default with SQLite persistence
  - Document: "Delete existing `.db` files before upgrading" in migration notes

  **Must NOT do**:
  - Don't add schema migration logic (pre-launch, clean break)
  - Don't make state management required — pipeline should work without it

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Focused update to existing code, not a full rewrite
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 12, 14)
  - **Blocks**: Task 15
  - **Blocked By**: Task 3 (needs new model names)

  **References**:
  - `src/audiorag/core/state.py:1-356` — Full StateManager to update
  - `src/audiorag/core/models.py:60-73` — IndexingStatus enum to generalize

  **Acceptance Criteria**:

  ```
  Scenario: NullStateManager exists and works
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag.core.state import NullStateManager
         import asyncio
         mgr = NullStateManager()
         asyncio.run(mgr.initialize())
         result = asyncio.run(mgr.get_source_status('test'))
         assert result is None
         print('PASS: NullStateManager works')
         "
    Expected Result: NullStateManager importable and functional
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `refactor(state): generalize IndexingStatus, add NullStateManager`
  - Files: `src/audiorag/core/state.py`, `src/audiorag/core/models.py`
  - Pre-commit: `uv run pytest tests/`

---

- [ ] 14. Clean public API + lazy imports with clear error messages

  **What to do**:
  - Rewrite `src/audiorag/__init__.py` to export clean public API:
    ```python
    from audiorag.pipeline import AudioRAG
    from audiorag.core.models import QueryResult, Source, ChunkMetadata
    from audiorag.core.config import AudioRAGConfig
    from audiorag.core.exceptions import AudioRAGError, ProviderError, PipelineError
    
    # Backward compatibility
    AudioRAGPipeline = AudioRAG
    ```
  - Ensure all domain module `__init__.py` files use `__getattr__` pattern with clear error messages when deps missing
  - Test the error message: `from audiorag.embed import VoyageEmbedder` without `voyageai` installed → should say "Install audiorag[voyage]"
  - Remove all old `try/except ImportError: X = None` patterns
  - Ensure `__all__` is defined in every public `__init__.py`

  **Must NOT do**:
  - Don't export internal classes (base classes, retry config, etc.)
  - Don't create "convenience" re-exports that add import path confusion

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 12, 13)
  - **Blocks**: Task 15
  - **Blocked By**: Tasks 4, 12

  **References**:
  - `src/audiorag/__init__.py:1-47` — Current init to rewrite
  - All domain `__init__.py` files — to verify __getattr__ pattern

  **Acceptance Criteria**:

  ```
  Scenario: Clean top-level imports work
    Tool: Bash
    Steps:
      1. python3 -c "
         from audiorag import AudioRAG, AudioRAGConfig, QueryResult, Source
         from audiorag import AudioRAGPipeline  # backward compat
         assert AudioRAG is AudioRAGPipeline
         print('PASS: Public API clean')
         "
    Expected Result: All imports work
    Evidence: Terminal output

  Scenario: Missing dep gives clear error message
    Tool: Bash
    Steps:
      1. python3 -c "
         try:
             from audiorag.embed import VoyageEmbedder
             # If voyageai is installed, this won't fail — that's OK
             print('PASS: VoyageEmbedder available (voyageai installed)')
         except ImportError as e:
             assert 'audiorag[voyage]' in str(e), f'Bad error message: {e}'
             print(f'PASS: Got helpful error: {e}')
         "
    Expected Result: Either works (dep installed) or gives clear install message
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `refactor(api): clean public API with lazy imports and clear error messages`
  - Files: `src/audiorag/__init__.py`, all domain `__init__.py`
  - Pre-commit: `uv run ruff check src/`

---

### WAVE 5: Cleanup & Verification

- [ ] 15. Delete dead code + old providers/

  **What to do**:
  - Delete the entire `src/audiorag/providers/` directory (all 14 files migrated to domain modules)
  - Delete `src/audiorag/chunking.py` (moved to `chunk/time_based.py`)
  - Verify no imports reference the deleted paths
  - Clean up any remaining `# TODO` stubs or unused imports
  - Run `uv run ruff check . --fix` to auto-fix import issues
  - Run `uv run ruff format .` to ensure consistent formatting
  - Update `pyproject.toml` if any source paths changed

  **Must NOT do**:
  - Don't delete test files (even if they reference old paths — fix them instead)
  - Don't touch `core/` (it's stable)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Deletion + verification, straightforward
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 12, 13, 14

  **References**:
  - `src/audiorag/providers/` — Entire directory to delete
  - `src/audiorag/chunking.py` — File to delete (moved to chunk/)

  **Acceptance Criteria**:

  ```
  Scenario: providers/ directory gone
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         assert not pathlib.Path('src/audiorag/providers').exists(), 'providers/ still exists'
         assert not pathlib.Path('src/audiorag/chunking.py').exists(), 'chunking.py still exists'
         print('PASS: Dead code removed')
         "
    Expected Result: Old files deleted
    Evidence: Terminal output

  Scenario: No imports reference deleted paths
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         dead_imports = ['from audiorag.providers', 'import audiorag.providers', 'from audiorag.chunking import']
         for f in pathlib.Path('src/audiorag').rglob('*.py'):
             content = f.read_text()
             for pattern in dead_imports:
                 assert pattern not in content, f'{f}: still imports from deleted path: {pattern}'
         print('PASS: No dead imports')
         "
    Expected Result: No references to deleted modules
    Evidence: Terminal output

  Scenario: Ruff passes clean
    Tool: Bash
    Steps:
      1. uv run ruff check src/ tests/
    Expected Result: Exit code 0
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `chore(cleanup): delete old providers/ directory and dead code`
  - Files: Deleted `src/audiorag/providers/`, `src/audiorag/chunking.py`
  - Pre-commit: `uv run ruff check src/`

---

- [ ] 16. Comprehensive test suite + lint/type check pass

  **What to do**:
  - Update ALL existing tests to use new import paths
  - Add tests for:
    - New `LocalFileSource` and `URLSource`
    - `TimeBasedChunker` class (wrapping the old function)
    - `NullStateManager`
    - `AudioRAG` (renamed pipeline) basic instantiation
    - Protocol conformance for ALL migrated providers
  - Run full verification:
    - `uv run pytest --cov=src/audiorag --cov-report=term-missing` — target 80%+
    - `uv run ruff check src/ tests/ --fix`
    - `uv run ruff format .`
    - `uv run ty check` (type checking)
  - Fix any failures until everything passes

  **Must NOT do**:
  - Don't add integration tests requiring real API keys
  - Don't change test infrastructure (pytest config, fixtures, etc.)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Comprehensive test update touching many files, requires understanding the full redesign
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (must be last)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 15

  **References**:

  **Pattern References**:
  - `tests/test_protocols.py` — Existing protocol tests (update import paths)
  - `tests/conftest.py` — Shared fixtures (may need updates)
  - `tests/test_protocol_conformance.py` — From Task 2 (should already pass)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All tests pass
    Tool: Bash
    Steps:
      1. uv run pytest -v --tb=short
    Expected Result: All tests pass, exit code 0
    Evidence: Full pytest output

  Scenario: Coverage meets 80% target
    Tool: Bash
    Steps:
      1. uv run pytest --cov=src/audiorag --cov-report=term-missing --cov-fail-under=80
    Expected Result: Coverage >= 80%, exit code 0
    Evidence: Coverage report

  Scenario: Ruff lint clean
    Tool: Bash
    Steps:
      1. uv run ruff check src/ tests/
    Expected Result: No errors
    Evidence: Terminal output

  Scenario: Type checking passes
    Tool: Bash
    Steps:
      1. uv run ty check
    Expected Result: No type errors (warnings acceptable)
    Evidence: Terminal output

  Scenario: No RuntimeError in any provider
    Tool: Bash
    Steps:
      1. python3 -c "
         import pathlib
         providers = list(pathlib.Path('src/audiorag/transcribe').glob('*.py')) + \
                     list(pathlib.Path('src/audiorag/embed').glob('*.py')) + \
                     list(pathlib.Path('src/audiorag/store').glob('*.py')) + \
                     list(pathlib.Path('src/audiorag/generate').glob('*.py')) + \
                     list(pathlib.Path('src/audiorag/rerank').glob('*.py'))
         for f in providers:
             if f.name.startswith('_') or f.name == '__init__.py':
                 continue
             content = f.read_text()
             if 'raise RuntimeError' in content:
                 print(f'FAIL: {f} raises RuntimeError')
                 exit(1)
         print('PASS: All providers use structured exceptions')
         "
    Expected Result: Zero RuntimeError raises in providers
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `test: comprehensive test suite update for v2 architecture`
  - Files: All test files
  - Pre-commit: `uv run pytest && uv run ruff check . && uv run ty check`

---

## Commit Strategy

| After Task | Message | Verification |
|------------|---------|--------------|
| 1 | `refactor(structure): establish new domain-organized module layout` | `uv run ruff check src/` |
| 2 | `test(protocols): add protocol conformance tests as redesign contracts` | `uv run pytest tests/test_protocol_conformance.py` |
| 3 | `refactor(models): make core models source-agnostic, add ChunkingStrategy protocol` | `uv run pytest && uv run ruff check` |
| 4 | `refactor(config): decompose God Config, fix stale import paths` | `uv run ruff check && uv run pytest` |
| 5 | `refactor(providers): create shared base classes for all provider categories` | `uv run ruff check src/` |
| 6-11 | Individual commits per domain module migration | `uv run ruff check src/` per domain |
| 12 | `refactor(pipeline): rewrite as thin Layer 2 orchestrator` | `uv run pytest` |
| 13 | `refactor(state): generalize IndexingStatus, add NullStateManager` | `uv run pytest tests/` |
| 14 | `refactor(api): clean public API with lazy imports and clear error messages` | `uv run ruff check` |
| 15 | `chore(cleanup): delete old providers/ directory and dead code` | `uv run ruff check` |
| 16 | `test: comprehensive test suite update for v2 architecture` | `uv run pytest --cov && uv run ruff check && uv run ty check` |

---

## Success Criteria

### Verification Commands
```bash
# All tests pass with coverage
uv run pytest --cov=src/audiorag --cov-report=term-missing --cov-fail-under=80

# Lint clean
uv run ruff check src/ tests/

# Type check
uv run ty check

# No dead imports or old paths
python3 -c "
import pathlib
dead = ['from audiorag.providers', 'from audiorag.logging_config', 'from audiorag.models import', 'from audiorag.retry_config']
for f in pathlib.Path('src/audiorag').rglob('*.py'):
    content = f.read_text()
    for d in dead:
        assert d not in content, f'{f}: {d}'
print('PASS: No dead imports')
"

# Core models source-agnostic
python3 -c "
content = open('src/audiorag/core/models.py').read()
assert 'youtube' not in content.lower()
assert 'video_title' not in content
print('PASS')
"

# Standalone usage works
python3 -c "
from audiorag import AudioRAG, AudioRAGConfig, QueryResult
from audiorag.core.protocols import STTProvider, EmbeddingProvider, VectorStoreProvider
print('PASS: Public API works')
"
```

### Final Checklist
- [ ] All "Must Have" requirements present
- [ ] All "Must NOT Have" guardrails respected
- [ ] All tests pass with 80%+ coverage
- [ ] No `providers/` directory exists
- [ ] No `video_title` in core models
- [ ] Config under 100 lines
- [ ] Pipeline under 250 lines
- [ ] Each provider under 80 lines (except YouTube scraper)
- [ ] Zero `TODO` stubs in shipped modules
- [ ] All domain modules use `__getattr__` lazy import pattern
- [ ] `from audiorag import AudioRAGPipeline` still works (backward compat)
