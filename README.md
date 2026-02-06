# AudioRAG

Provider-agnostic RAG pipeline for audio content. Download, transcribe, chunk, embed, and search audio from YouTube and other sources.

## Features

- **Multi-provider support**: OpenAI, Deepgram, AssemblyAI, Groq (STT); OpenAI, Voyage, Cohere (embeddings); OpenAI, Anthropic, Gemini (generation); ChromaDB, Pinecone, Weaviate, Supabase (vector stores)
- **Resumable processing**: SQLite state tracking with hash-based IDs
- **Automatic chunking**: Time-based segmentation with configurable duration
- **Audio splitting**: Handles large files by splitting before transcription
- **Structured logging**: Context-aware logging with operation timing
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
    
    # Initialize pipeline
    pipeline = AudioRAGPipeline(config)
    
    # Index audio from YouTube
    await pipeline.index("https://youtube.com/watch?v=...")
    
    # Query the indexed content
    result = await pipeline.query("What are the main points discussed?")
    print(result.answer)
    
    # Access sources with timestamps
    for source in result.sources:
        print(f"{source.video_title} at {source.start_time}s")
        print(f"URL: {source.youtube_timestamp_url}")

asyncio.run(main())
```

## Installation

```bash
# Install with uv
uv add audiorag

# Or with pip
pip install audiorag
```

### Optional Dependencies

```bash
# Audio scraping utilities (yt-dlp, pydub)
uv add audiorag[defaults]

# All providers and utilities
uv add audiorag[all]

# Specific providers only
uv add audiorag[openai,chromadb,scraping,cohere]
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

## License

MIT License
