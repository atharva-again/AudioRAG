# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.0] - 2026-02-15 - YouTube Audio-Only Downloads

### Added
- **youtube_format config option**: New `youtube_format` config to enable audio-only YouTube downloads.
  - Set `AUDIORAG_YOUTUBE_FORMAT=bestaudio` to download audio-only (saves ~95% bandwidth).
  - Works with yt-dlp format selection: `bestaudio`, `bestaudio/best`, `worstaudio`.
  - Compatible with existing `-x --audio-format mp3` post-processing.

## [0.12.0] - 2026-02-14 - Budget Store, Doctor CLI, .env Discovery, Index Status

### Added
- **Persistent BudgetStore**: New `BudgetStore` protocol enables pluggable budget state management backends.
  - Added `SqliteBudgetStore` for persistent budget tracking across process restarts.
  - Added `atomic_reserve()` for atomic check-and-record in a single transaction.
- **Budget sipping**: New budget adjustment feature that reconciles estimated vs actual audio duration after download.
  - Prevents budget waste by reserving based on estimated duration, then "sipping" (releasing) excess after actual duration is known.
- **audiorag doctor CLI**: New `audiorag doctor` subcommand for diagnosing pipeline issues.
  - Verifies all required dependencies are installed.
  - Checks provider configurations.
  - Provides actionable troubleshooting advice.
- **.env auto-discovery**: Configuration now automatically discovers `.env` files by walking up from the current directory.
  - Supports nested project structures.
  - Falls back to default locations if not found.
- **get_index_status API**: New `pipeline.get_index_status(source_url)` method to check if a source is indexed without triggering re-indexing.
  - Returns indexing status: `not_indexed`, `indexed`, or `failed`.
  - Useful for checking pipeline state before queries.

### Changed
- **BudgetGovernor refactor**: Split into separate modules for better maintainability.
  - `budget.py` - In-memory budget tracking
  - `budget_store_sqlite.py` - SQLite-backed persistent storage
  - `protocols/budget_store.py` - Protocol definition for custom stores

## [0.11.0] - 2026-02-14 - YouTube Improvements

### Added
- **YouTube cookies-from-browser**: New config option `youtube_cookies_from_browser` to extract cookies directly from browser (e.g., `chrome`, `firefox:default`, `chrome+gnomekeyring:Profile1`).
- **Cookie file support**: Wired up existing `youtube_cookie_file` config option (was dead code).
- **CI improvements**: CI now installs all optional dependencies for better test coverage.

### Changed
- **Consolidated ydl_opts**: Created shared `audiorag.source.ydl_utils` module to consolidate yt-dlp option building (was duplicated in pipeline.py and discovery.py).

### Fixed
- **Type safety**: Resolved pre-existing type errors in weaviate.py, assemblyai.py, groq.py, and cohere.py exposed by full dependency installation.

## [0.10.0] - 2026-02-14 - Transcription Resumability

### Added
- **Transcription resumability**: Pipeline now tracks per-part transcription in SQLite, enabling resumption after partial failures without re-transcribing completed parts.
- **Transcripts table**: New database table stores transcription segments per audio part.
- **StateManager methods**: Added `store_transcript()`, `get_transcripts()`, and `get_transcribed_part_indices()` methods.

### Benefits
- **Saves money**: If transcription fails at part 8/10, next run skips parts 1-7 and only transcribes 8-10, saving Groq/STT budget.
- **Enables re-chunking**: Raw transcript storage infrastructure ready for future re-chunking without re-STT.
- **Resilient**: Each part is persisted immediately after successful transcription.

### Fixed
- **Timestamp alignment**: Stored timestamps are now adjusted with cumulative offset to ensure correct time alignment after resume.

## [0.9.0] - 2026-02-14 - Persistent Cache & Cache Management

### Added
- **Persistent work_dir default**: Default `work_dir` now uses platform-appropriate cache directory:
  - Linux: `~/.cache/audiorag`
  - macOS: `~/Library/Caches/audiorag`
  - Windows: `%LOCALAPPDATA%\audiorag`
- **Cache management CLI**: New commands to manage cached audio files:
  - `audiorag cache info` - Show cache location and size
  - `audiorag cache clear` - Clear all cached audio files
- **Cache management SDK**: New methods on `AudioRAGPipeline`:
  - `pipeline.clear_cache()` - Clear cache, returns count of items removed
  - `pipeline.get_cache_info()` - Get cache location, file count, and size

### Fixed
- **Type safety**: Fixed type checking for `metadata.duration` attribute access in budget reservation logic.

## [0.8.1] - 2026-02-14 - Vector Store Source ID Fix

### Fixed
- **Leaky abstraction in vector stores**: Vector store providers now use canonical `source_id` instead of raw `source_url`. This fixes issue #19 where backends had to parse Source IDs from URLs.
  - Added `source_id` to `StageContext` for pipeline-wide canonical ID
  - Renamed `VectorStoreProvider.delete_by_source()` to `delete_by_source_id()`
  - Updated all vector store implementations (ChromaDB, Pinecone, Weaviate, Supabase) to filter by `source_id`
  - Updated metadata to use `source_id` instead of `source_url`

### Migration Note
> ⚠️ Existing vector stores with `source_url` metadata will need to be re-indexed for `force=True` deletion to work. Alternatively, users can manually delete via the vector store's native tools.

## [0.8.0] - 2026-02-14 - Auto-detect File Protocol

### Added
- **Audio source auto-routing**: New `AudioSourceRouter` automatically detects URL protocol:
  - `file://` URLs → `LocalSource` (bypasses yt-dlp)
  - Local paths (`/home/user/audio.mp3`, `./audio.mp3`) → `LocalSource`
  - YouTube URLs → `YouTubeSource`
  - Other HTTP URLs → `URLSource`
- **URL source provider**: New `audio_source_provider` config option supports `url` to use URLSource directly.
- **Robust YouTube URL detection**: Uses proper URL parsing instead of substring matching to avoid false positives (e.g., `myyoutube.com` is not YouTube).

### Changed
- **Replaced pydub with ffprobe**: Duration detection in `LocalSource` and `URLSource` now uses ffprobe directly instead of pydub.

### Fixed
- **Protocol conformance**: `get_metadata()` return type now allows `None` to conform with implementations that don't support metadata extraction.
- **URL parameter naming**: Fixed parameter name mismatch (`source_url` → `url`) in `URLSource.download()` to match protocol.

### Configuration
```bash
# Select audio source provider (default: youtube - auto-routing enabled)
export AUDIORAG_AUDIO_SOURCE_PROVIDER="local"   # Force LocalSource
export AUDIORAG_AUDIO_SOURCE_PROVIDER="url"    # Force URLSource
export AUDIORAG_AUDIO_SOURCE_PROVIDER="youtube"  # YouTube + auto-routing (default)
```

## [0.7.0] - 2026-02-13 - Audio Source Provider Fixes

### Added
- **Audio source provider selection**: New `audio_source_provider` config option allows selecting between YouTube (`youtube`) and local file (`local`) sources. Defaults to `youtube` for backward compatibility.
- **Local file unique ID extraction**: LocalSource now generates a unique ID from the file path hash, fixing vector store unique constraint violations.
- **Pre-download budget checks for local files**: Budget reservation now works for local files (file:// URLs and local paths), not just YouTube URLs.

### Fixed
- **Impersonate handling**: String impersonate values (e.g., "chrome-120") are now properly converted to `ImpersonateTarget` objects before passing to yt-dlp, preventing crashes.

### Changed
- **Provider factory**: Added `create_audio_source_provider()` factory function for consistent provider instantiation.

### Configuration
```bash
# Select audio source provider (default: youtube)
export AUDIORAG_AUDIO_SOURCE_PROVIDER="local"  # or "youtube"
```

## [0.6.2] - 2026-02-13 - Budget Error Fix

### Fixed
- **BudgetExceededError handling**: Fixed bug where budget errors during pre-download metadata extraction were being swallowed by generic exception handler, causing wasted bandwidth and API calls. Budget errors now fail-fast immediately at the pre-download stage.

## [0.6.1] - 2026-02-13 - YouTubeSource Lean Refactor

### Changed
- **Simplified YouTubeSource**: Reduced from 632 to 242 lines (62% smaller) by removing 16 hardcoded yt-dlp constructor parameters
- **Unified yt-dlp options**: Now uses `ydl_opts` dict for all yt-dlp options instead of individual parameters
- **Delegated to yt-dlp**: PO tokens, cookies, JS runtime, and other advanced options are now handled by yt-dlp internally rather than being re-implemented

### Breaking
- `YouTubeSource` constructor now only accepts `ydl_opts` and `download_archive` parameters
- Users who were passing individual yt-dlp options (po_token, visitor_data, impersonate_client, etc.) need to pass them via `ydl_opts` dict

### How to migrate
```python
# Before
YouTubeSource(po_token="xxx", visitor_data="yyy", player_clients=["tv"])

# After
YouTubeSource(ydl_opts={
    "extractor_args": {
        "youtube": {
            "po_token": ["web.gvs+xxx"],
            "visitor_data": "yyy",
            "player_client": ["tv"]
        }
    }
})
```

For most users, YouTubeSource now works out of the box without any configuration.

## [0.6.0] - 2026-02-13 - Simplification Refactor

### Breaking
- **Removed pydub dependency**: Audio splitting now uses ffmpeg subprocess calls directly instead of pydub. Users must have ffmpeg installed on their system.

### Changed
- **Removed Tenacity retry wrapper**: YouTube source no longer wraps yt-dlp calls with Tenacity, simplifying the download flow.
- **Advanced config flattening**: Added `AdvancedConfig` nested class for advanced settings while maintaining flat `AUDIORAG_X` env var access with automatic sync.
- **Simplified provider factory**: Added fail-fast `ImportError` messages with installation hints when optional vector store packages are missing.
- **CI ffmpeg installation**: Added ffmpeg installation via apt-get in GitHub Actions workflow.

### Fixed
- **Lint warnings**: Fixed warnings by using `shutil.which()` for full ffmpeg/ffprobe paths.
- **Dead code removal**: Removed unused `retry_config` parameter and empty `_ensure_js_runtime()` method.

### Documentation
- Updated README.md and docs to reflect pydub → ffmpeg change.

## [0.5.5] - 2026-02-12 - pyproject.toml Refactor

### Added
- **Granular optional dependencies**: Added separate optional dependency groups for all providers (deepgram, assemblyai, groq, voyageai, pinecone, weaviate, supabase, anthropic, google-genai, rerank-cohere).

### Changed
- **Renamed scraping to youtube**: Optional dependency group `scraping` renamed to `youtube` for clarity.
- **Upper bounds on dev dependencies**: Added upper bounds to prevent supply chain issues.
- **Coverage target**: Set realistic coverage target (65%) with documented excludes for untestable provider modules.
- **Type checking**: Configured ty to handle optional dependencies gracefully.

### Fixed
- **Logger initialization**: Added `__init__` methods to LocalSource and URLSource for proper `_logger` attribute initialization.
- **Type errors**: Fixed unresolved attribute type errors in source modules.

## [0.5.4] - 2026-02-12 - Gemini SDK Migration

### Changed
- **Gemini SDK migration**: Migrated from deprecated `google-generativeai` to the new `google.genai` SDK for future compatibility.
- **Dependency updates**: Updated dependency and docs to reference `google-genai`.

## [0.5.3] - 2026-02-12 - YouTube reliability + budget safeguards

### Added
- **Bleeding-edge YouTube 2026 support**: Upgraded to yt-dlp 2026.02.04 with mandatory JS runtime helpers (yt-dlp-ejs, curl-cffi) and optimized player skip patterns for lower latency.
- **Visitor context binding**: Added PO token, visitor data, and data sync ID configuration so tokens are bound to the correct session and avoid extraction failures.
- **Pre-download budget checks**: YouTube metadata is fetched before download to reserve audio-seconds budget upfront and prevent wasted bandwidth.
- **Duration reconciliation**: Post-download adjustments correct budget deltas when actual duration differs from estimates, including automatic release on download failures.

### Changed
- **Source metadata model**: Introduced `SourceMetadata` and updated `AudioSourceProvider` to support pre-flight metadata extraction.
- **Budget governor release flow**: Added `release()` method and negative audio-seconds handling for compensating entries.
- **CLI setup**: Added interactive prompts for YouTube PO tokens, visitor data, and JS runtime selection.
- **Documentation updates**: README and docs updated for YouTube 2026 configuration and budget safeguards.

### Fixed
- **Protocol conformance tests**: Updated AudioSourceProvider mocks for `get_metadata()` and added coverage for release/reconciliation flows.

## [0.5.2] - 2026-02-12 - YouTube Source Reliability Improvements

### Fixed
- **Falsey list bug in `youtube.py`**: Passing `player_clients=[]` no longer gets overridden with defaults. Uses explicit `is not None` check to distinguish between `None` (use defaults) and `[]` (use yt-dlp defaults).
- **Playlist expansion failure handling**: Failed playlist/channel expansions now raise `DiscoveryError` immediately instead of returning the original URL, which would cause confusing "file not found" errors downstream.

### Added
- **New `DiscoveryError` exception**: Structured exception for source discovery failures with optional URL context, following existing exception hierarchy.
- **Robust URL detection**: `_is_youtube_collection()` helper uses `urllib.parse` for proper URL parsing instead of brittle substring matching.
- **Comprehensive test coverage**: 20 new tests in `tests/test_source_discovery.py` covering DiscoveryError, URL detection, playlist expansion, and player_clients handling.

## [0.5.1] - 2026-02-12 - Graceful Exit Handling

### Added
- **Graceful exit handling** for SDK and CLI:
  - Added `AudioRAGPipeline.close()` method for explicit resource cleanup (idempotent)
  - Added async context manager support (`__aenter__`/`__aexit__`) for automatic cleanup
  - Resources cleaned up automatically on success, failure, or interruption
  - SIGTERM/SIGINT signal handlers in CLI for graceful shutdown
- **New test coverage** for cleanup behavior (5 comprehensive tests)

### Changed
- Updated all documentation examples to use `async with` syntax
- CLI `index` and `query` commands now use async context managers

### Documentation
- Added "Graceful exit" to README.md features list
- Added Resource Cleanup section to quickstart.md
- Documented `close()` method and context manager in api-reference.md

## [0.5.0] - 2026-02-12 - Resilient Batch Indexing and SDK/CLI Parity

### Added
- Structured batch indexing result models for SDK consumers:
  - `BatchIndexResult` (inputs, discovered/indexed/skipped sources, failures)
  - `BatchIndexFailure` (source_url, stage, error_message)
- New `AudioRAGPipeline.index_many(inputs, force=False, raise_on_error=True)`
  contract that returns structured batch outcomes and supports tolerant mode
  (`raise_on_error=False`) for partial-failure reporting.

### Changed
- Core indexing boundary now performs source discovery for SDK calls too:
  - `index(url)` routes through discovery-backed batch indexing
  - Playlist/channel URLs and local directories are processed per source with
    independent state/resume semantics.
- Batch indexing error handling is now resilient across exception types:
  non-`PipelineError` failures are normalized and captured in batch results,
  while strict mode still raises with clear context.
- CLI `audiorag index` now uses the unified pipeline batch path and reports
  aggregate batch outcomes (indexed/skipped/failed) with per-source failures.

### Documentation
- Updated README and docs (`quickstart`, `api-reference`) to reflect:
  - SDK/CLI behavior parity for discovery-backed indexing
  - `raise_on_error` behavior and structured batch results
  - Updated batch indexing examples and output expectations

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

[Unreleased]: https://github.com/atharva-again/audiorag/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/atharva-again/audiorag/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/atharva-again/audiorag/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/atharva-again/audiorag/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/atharva-again/audiorag/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/atharva-again/audiorag/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/atharva-again/audiorag/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/atharva-again/audiorag/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/atharva-again/audiorag/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/atharva-again/audiorag/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/atharva-again/audiorag/compare/v0.5.5...v0.6.0
[0.5.5]: https://github.com/atharva-again/audiorag/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/atharva-again/audiorag/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/atharva-again/audiorag/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/atharva-again/audiorag/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/atharva-again/audiorag/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/atharva-again/audiorag/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/atharva-again/audiorag/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/atharva-again/audiorag/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/atharva-again/audiorag/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/atharva-again/audiorag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/atharva-again/audiorag/releases/tag/v0.1.0
