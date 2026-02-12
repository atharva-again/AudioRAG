# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-02-12 - Provider-aware Vector ID Strategy

### Added
- Provider-aware vector ID strategy for indexing with `AUDIORAG_VECTOR_ID_FORMAT`
  (`auto`, `sha256`, `uuid5`) and optional `AUDIORAG_VECTOR_ID_UUID5_NAMESPACE`.
- Deterministic UUID5 conversion path for UUID-oriented vector stores while
  preserving canonical SHA-256 IDs in SQLite state.
- Vector store capability hint protocol (`VectorIdFormatAwareProvider`) so
  provider defaults are explicit (`weaviate -> uuid5`, others -> `sha256`).
- New test coverage for ID strategy resolution, deterministic UUID behavior,
  and pipeline integration paths.

### Changed
- Embed stage now adapts provider-facing vector IDs before `add()` and verifies
  writes using the effective provider-facing IDs.
- Indexing now stores vector ID strategy metadata and guards against strategy
  changes on existing sources unless `force=True` is used for reindex.
- Configuration and docs updated across README and `docs/` for vector ID
  strategy behavior, provider defaults, and safe migration guidance.

## [0.3.1] - 2026-02-11 - Refactor and Release Pipeline Cleanup

### Changed
- Refactored provider construction by extracting factory helpers into
  `audiorag.core.provider_factory`, reducing `AudioRAGPipeline` complexity.
- Refactored YouTube source yt-dlp option assembly into focused helper methods
  for listing and download flows.
- Refactored `StateManager` SQL handling by centralizing statements and row/JSON
  mapping helpers for better readability.
- Simplified release automation to a single tag-driven publish path in CI,
  preventing duplicate PyPI publish attempts.

## [0.3.0] - 2026-02-11 - Proactive API Reliability

### Added
- **Proactive budget governor** for API usage control:
  - Global and per-provider limits for RPM, TPM, and audio-seconds/hour
  - Fail-fast budget checks before expensive operations
  - Persistent budget accounting in SQLite for restart/process safety
  - `BudgetExceededError` for explicit budget-limit failures
- **Atomic vector write verification** after embedding storage:
  - New `VerifiableVectorStoreProvider` protocol (`verify(ids) -> bool`)
  - Built-in verification support for ChromaDB, Pinecone, Weaviate, and Supabase
  - Verification modes: `off`, `best_effort`, `strict`
  - Configurable retry attempts and wait interval for verification checks
- **End-to-end test coverage** for reliability controls:
  - Budget governor unit/integration tests
  - Pipeline preflight budget reservation tests
  - Vector verification mode and retry behavior tests

### Changed
- **Index pipeline reliability flow**:
  - Preflight STT budget reservation now runs when source duration is known
  - Embed stage verifies vector persistence before marking embedded/completed
- **Query pipeline budget enforcement**:
  - Budget checks added before embed/retrieve/rerank/generate steps

### Documentation
- Updated `README.md` and docs (`quickstart`, `configuration`, `providers`, `architecture`, `api-reference`) to:
  - Document budget governor and vector verification configuration/behavior
  - Correct stale model field names (`title` instead of `video_title`)
  - Correct module/class references and provider examples to match current codebase

## [0.2.0] - 2026-02-11 - Multi-source Indexing

### Added
- **Multi-source indexing support** - CLI `index` command now accepts multiple inputs:
  - YouTube playlists (auto-expanded to individual videos)
  - Local directories (recursively finds audio files)
  - Multiple URLs/paths in a single command
  - Mixed inputs (URLs + local files)
- **Source discovery module** (`audiorag.source.discovery`):
  - `discover_sources()` function for automatic source type detection
  - Async source resolution with progress tracking
  - Support for playlist expansion and directory traversal
- **Enhanced CLI progress tracking** - Multi-stage progress bars for batch operations
- **Lazy import validation tests** - Ensure optional dependencies are properly isolated

### Changed
- **CLI interface improvements**:
  - `index` command now uses `nargs="+"` for multiple inputs instead of single URL
  - Better help text with examples for quoted paths
  - Progress tracking shows overall and per-source progress
  - More detailed error reporting for batch operations
- **YouTube source validation** - Moved ffmpeg check to lazy validation (only when downloading)
  - Faster CLI startup when ffmpeg is not in PATH
  - Clearer error messages when ffmpeg is missing during operations
- **Source module exports** - Added `discover_sources` to `audiorag.source` public API

### Fixed
- Fixed stale flat imports throughout codebase to use `audiorag.core.*` paths
- Resolved CI test failures with uv package management migration
- Excluded optional provider modules from coverage calculation

### Documentation
- Updated API reference documentation
- Updated architecture documentation
- Improved CLI help documentation with usage examples

## [0.1.0] - 2025-02-XX - Initial Release

### Added
- Initial release of AudioRAG
- Provider-agnostic RAG pipeline for audio content
- Multi-provider support:
  - **STT**: OpenAI, Deepgram, AssemblyAI, Groq
  - **Embeddings**: OpenAI, Voyage, Cohere
  - **Generation**: OpenAI, Anthropic, Gemini
  - **Vector Stores**: ChromaDB, Pinecone, Weaviate, Supabase
- Core pipeline with resumable processing (SQLite state tracking)
- Automatic time-based chunking
- Audio splitting for large files
- Structured logging with context-aware operation timing
- Interactive CLI with `setup`, `index`, and `query` commands
- Full type annotations for Python 3.12+
- Base mixin classes for all provider categories
- Protocol-based provider abstractions

[Unreleased]: https://github.com/atharva-again/audiorag/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/atharva-again/audiorag/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/atharva-again/audiorag/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/atharva-again/audiorag/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/atharva-again/audiorag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/atharva-again/audiorag/releases/tag/v0.1.0
