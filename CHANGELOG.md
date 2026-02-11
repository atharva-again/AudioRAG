# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-11

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

## [0.1.0] - 2025-02-XX

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

[Unreleased]: https://github.com/atharva-again/audiorag/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/atharva-again/audiorag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/atharva-again/audiorag/releases/tag/v0.1.0
