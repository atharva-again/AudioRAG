# AudioRAG

Provider-agnostic RAG pipeline for audio content. Download, transcribe, chunk, embed, and search audio from YouTube and other sources.

## Features

- **Multi-provider support**: OpenAI, Deepgram, AssemblyAI, Groq (STT); OpenAI, Voyage, Cohere (embeddings); OpenAI, Anthropic, Gemini (generation); ChromaDB, Pinecone, Weaviate, Supabase (vector stores)
- **Batch indexing**: Index multiple URLs, playlists, and local directories in one command
- **Source discovery**: Automatically expand playlists and recursively scan directories
- **Resumable processing**: SQLite state tracking with hash-based IDs
- **Provider-aware vector IDs**: Canonical SHA-256 chunk IDs with optional UUID5 conversion per vector store
- **Proactive budget governor**: Optional fail-fast limits for RPM, TPM, and audio-seconds/hour
- **Atomic vector verification**: Optional post-write verification with strict or best-effort modes
- **Automatic chunking**: Time-based segmentation with configurable duration
- **Audio splitting**: Handles large files by splitting before transcription
- **Structured logging**: Context-aware logging with operation timing
- **Graceful exit**: Automatic resource cleanup on completion, failure, or interruption
- **Type-safe**: Python 3.12+ with full type annotations

## Quick Start

```python
import asyncio
from audiorag import AudioRAGPipeline, AudioRAGConfig

async def main():
    # Configure with your chosen providers
    config = AudioRAGConfig(
        stt_provider="openai",
        stt_model="whisper-1",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        vector_store_provider="chromadb",
        generation_provider="openai",
        generation_model="gpt-4o-mini",
        # API keys can also be set via environment variables
        openai_api_key="sk-...",
    )
    
    # Use async context manager for automatic resource cleanup
    async with AudioRAGPipeline(config) as pipeline:
        # Index audio from YouTube
        await pipeline.index("https://youtube.com/watch?v=...")

    # Batch indexing with partial-failure reporting
        batch_result = await pipeline.index_many(
            [
                "https://youtube.com/playlist?list=...",
                "./podcasts/",
                "https://youtube.com/watch?v=singleVideo",
            ],
            raise_on_error=False,
        )
        print(
            f"Indexed={len(batch_result.indexed_sources)} "
            f"Skipped={len(batch_result.skipped_sources)} "
            f"Failed={len(batch_result.failures)}"
        )

        # Query the indexed content
        result = await pipeline.query("What are the main points discussed?")
        print(result.answer)

        # Access sources with timestamps
        for source in result.sources:
            print(f"{source.title} at {source.start_time}s")
            print(f"URL: {source.source_url}")

asyncio.run(main())
```

## Installation

```bash
# Install with uv (recommended)
uv add audiorag

# Or with pip
pip install audiorag
```

### Optional Dependencies

```bash
# Audio scraping utilities (yt-dlp, ffmpeg)
uv add audiorag[defaults]  # or: pip install audiorag[defaults]

# All providers and utilities
uv add audiorag[all]  # or: pip install audiorag[all]

# Specific providers only
uv add audiorag[openai,chromadb,youtube,cohere]
```

## Command Line Interface

AudioRAG includes a premium CLI for easy setup, indexing, and querying.

### Setup

Configure your providers and API keys interactively:

```bash
audiorag setup
```

This will guide you through selecting providers for STT, embeddings, vector stores, and generation, saving them to a `.env` file.

### Indexing

Index audio from multiple sources in a single command:

```bash
# Single YouTube video
audiorag index "https://youtube.com/watch?v=..."

# YouTube playlist (auto-expanded to individual videos)
audiorag index "https://youtube.com/playlist?list=..."

# Local audio files and folders
audiorag index "./podcast.mp3" "./audio_folder/"

# Multiple URLs at once
audiorag index "https://youtube.com/watch?v=video1" "https://youtube.com/watch?v=video2"

# Mixed inputs
audiorag index "./local_audio/" "https://youtube.com/watch?v=..." "./interview.wav"
```

**Note:** Always wrap URLs and paths containing spaces in quotes.

**Options:**
- `--force`: Re-process and re-index even if the URL has been processed before.

The CLI automatically:
- Expands YouTube playlists/channels into individual video URLs
- Recursively discovers audio files in directories
- Shows aggregate batch results (indexed/skipped/failed) with per-source failures
- Handles errors per source without stopping the entire batch

### Querying

Ask questions about your indexed audio content with a sophisticated results layout:

```bash
audiorag query "What are the main points discussed in the audio?"
```

## Configuration

AudioRAG uses pydantic-settings with environment variable support. All settings use the `AUDIORAG_` prefix.

```bash
# Example: Using OpenAI for STT, embeddings, and generation
export AUDIORAG_OPENAI_API_KEY="sk-..."
export AUDIORAG_STT_PROVIDER="openai"
export AUDIORAG_EMBEDDING_PROVIDER="openai"
export AUDIORAG_VECTOR_STORE_PROVIDER="chromadb"
export AUDIORAG_GENERATION_PROVIDER="openai"

# Example: Using different providers
export AUDIORAG_DEEPGRAM_API_KEY="..."
export AUDIORAG_STT_PROVIDER="deepgram"
export AUDIORAG_VOYAGE_API_KEY="..."
export AUDIORAG_EMBEDDING_PROVIDER="voyage"

# Processing settings
export AUDIORAG_CHUNK_DURATION_SECONDS="30"
export AUDIORAG_RETRIEVAL_TOP_K="10"
export AUDIORAG_RERANK_TOP_N="3"

# Optional YouTube 2026 advanced configuration (for stability)
export AUDIORAG_YOUTUBE_PO_TOKEN="..."           # PO token for bot detection bypass
export AUDIORAG_YOUTUBE_VISITOR_DATA="..."       # Visitor session (bound to PO token)
export AUDIORAG_JS_RUNTIME="deno"                # JS runtime (deno/node/bun)

# Optional budget governor
export AUDIORAG_BUDGET_ENABLED="true"
export AUDIORAG_BUDGET_RPM="60"
export AUDIORAG_BUDGET_TPM="120000"
export AUDIORAG_BUDGET_AUDIO_SECONDS_PER_HOUR="7200"

# Optional vector write verification
export AUDIORAG_VECTOR_STORE_VERIFY_MODE="best_effort"  # off | best_effort | strict
export AUDIORAG_VECTOR_STORE_VERIFY_MAX_ATTEMPTS="5"
export AUDIORAG_VECTOR_STORE_VERIFY_WAIT_SECONDS="0.5"

# Optional vector ID strategy
export AUDIORAG_VECTOR_ID_FORMAT="auto"  # auto | sha256 | uuid5
export AUDIORAG_VECTOR_ID_UUID5_NAMESPACE="6ba7b810-9dad-11d1-80b4-00c04fd430c8"  # optional
```

See [Configuration Guide](docs/configuration.md) for all options.

## Documentation

- [Quick Start Guide](docs/quickstart.md) - Get up and running
- [Configuration](docs/configuration.md) - All configuration options
- [Providers](docs/providers.md) - Available providers and setup
- [Architecture](docs/architecture.md) - Pipeline stages and data flow
- [API Reference](docs/api-reference.md) - Complete API documentation

## Development

```bash
# Clone and setup
git clone <repository-url>
cd audiorag
uv sync

# Run tests
uv run pytest

# Run checks
uv run ruff check . --fix
uv run ty check

# Install pre-commit hooks
uv run prek install
```

## Pipeline Stages

1. **Download**: Fetch audio from URL (YouTube supported)
2. **Split**: Divide large files into processable chunks
3. **Transcribe**: Convert audio to text using STT provider
4. **Chunk**: Group transcription into time-based segments
5. **Embed**: Generate vector embeddings for each chunk
6. **Store**: Persist embeddings in vector database

## Reliability Controls

- **Budget governor** (`AUDIORAG_BUDGET_ENABLED=true`): reserves budget before expensive calls and fails fast with `BudgetExceededError` when limits would be exceeded.
- **Pre-download budget checks**: for YouTube URLs, metadata is extracted before download to reserve budget upfront, preventing wasted bandwidth on files exceeding budget limits (~73% cost reduction on free-tier services).
- **Duration reconciliation**: actual audio duration is compared to estimated duration after download, with automatic budget adjustment.
- **Preflight transcription reservation**: when audio duration is known, indexing reserves full audio-seconds budget before STT starts.
- **Persistent budget accounting**: budget usage is persisted in SQLite for cross-process and restart safety.
- **Vector write verification**: after `add()`, providers that support `verify(ids)` are checked.
- **Verification modes**: `off` disables checks, `best_effort` warns on failure, `strict` fails indexing when verification fails.
- **Provider-aware vector IDs**: state IDs stay SHA-256; vector-store IDs can be auto-resolved to UUID5 for UUID-oriented providers.
- **Safe strategy changes**: if vector ID strategy changes for an existing source, reindex with `force=True` to avoid mixed IDs.

## License

MIT License
