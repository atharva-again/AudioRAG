# Quick Start Guide

This guide gets you up and running with AudioRAG.

## Prerequisites

- Python 3.12 or higher
- API keys from providers you want to use

## Installation

```bash
# Install with uv (recommended)
uv pip install audiorag

# Or with pip
pip install audiorag
```

## Basic Usage

### 1. Set Up Environment Variables

Choose your providers and set the corresponding API keys:

```bash
# Example with OpenAI
export AUDIORAG_OPENAI_API_KEY="sk-your-key-here"
export AUDIORAG_STT_PROVIDER="openai"
export AUDIORAG_EMBEDDING_PROVIDER="openai"
export AUDIORAG_GENERATION_PROVIDER="openai"
export AUDIORAG_VECTOR_STORE_PROVIDER="chromadb"
```

Or create a `.env` file:

```
AUDIORAG_OPENAI_API_KEY=sk-your-key-here
AUDIORAG_STT_PROVIDER=openai
AUDIORAG_EMBEDDING_PROVIDER=openai
AUDIORAG_GENERATION_PROVIDER=openai
AUDIORAG_VECTOR_STORE_PROVIDER=chromadb
```

### 2. Index Your First Audio

```python
import asyncio
from audiorag import AudioRAGPipeline, AudioRAGConfig

async def index_content():
    # Configuration loads from environment variables
    config = AudioRAGConfig()

    # Use async context manager for automatic resource cleanup
    async with AudioRAGPipeline(config) as pipeline:
        # Index a local audio file
        await pipeline.index("./my_podcast.mp3")
        print("Indexing complete!")

asyncio.run(index_content())
```

### Batch Indexing Multiple Sources

Index multiple files and directories:

```python
import asyncio
from audiorag import AudioRAGPipeline, AudioRAGConfig

async def batch_index():
    config = AudioRAGConfig()

    # Async context manager ensures resources are cleaned up
    async with AudioRAGPipeline(config) as pipeline:
        # Define your inputs - files and directories
        inputs = [
            "./podcasts/",                      # Local directory
            "./interviews/special.mp3",        # Single file
            "./lectures/intro.wav",           # Another file
        ]

        result = await pipeline.index_many(inputs, raise_on_error=False)

        print(f"Indexed: {len(result.indexed_sources)}")
        print(f"Skipped: {len(result.skipped_sources)}")
        print(f"Failed: {len(result.failures)}")

        print("Batch indexing complete!")

asyncio.run(batch_index())
```

`index_many()` automatically handles source discovery for:
- **Local directories**: Recursively finds audio files (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`)
- **Files**: Processes individual audio files
- **Deduplication**: Combines all sources into a unique list
- **Per-source resumability**: Each file is tracked independently in state

### Resource Cleanup

AudioRAG ensures proper cleanup of resources (database connections, etc.) in all scenarios:

**Using async context manager (recommended):**
```python
async with AudioRAGPipeline(config) as pipeline:
    await pipeline.index_many(paths)
    result = await pipeline.query("question")
# Resources automatically cleaned up here
```

**Manual cleanup:**
```python
pipeline = AudioRAGPipeline(config)
await pipeline.index_many(paths)
await pipeline.close()  # Safe to call multiple times
```

**Cleanup occurs automatically on:**
- Successful completion
- Exceptions during processing
- Signal interruptions (SIGINT/SIGTERM) in CLI

### 3. Query the Indexed Content

```python
async def query_content():
    config = AudioRAGConfig()

    async with AudioRAGPipeline(config) as pipeline:
        result = await pipeline.query("What is the main topic of this audio?")

        print("Answer:", result.answer)
        print("\nSources:")
        for source in result.sources:
            print(f"  - {source.title}")
            print(f"    Timestamp: {source.start_time}s")
            print(f"    Relevance: {source.relevance_score:.2f}")
            print(f"    Source: {source.source_url}")

asyncio.run(query_content())
```

## Complete Example

```python
import asyncio
from audiorag import AudioRAGPipeline, AudioRAGConfig

async def main():
    # Initialize
    config = AudioRAGConfig()

    # Use async context manager for automatic cleanup
    async with AudioRAGPipeline(config) as pipeline:
        # Index multiple sources
        inputs = [
            "./my_podcasts/",           # Directory
            "./interview.mp3",         # Single file
            "./lecture.wav",            # Another file
        ]

        batch_result = await pipeline.index_many(inputs, raise_on_error=False)

        print(
            f"\nIndexing complete: indexed={len(batch_result.indexed_sources)} "
            f"skipped={len(batch_result.skipped_sources)} "
            f"failed={len(batch_result.failures)}\n"
        )

        # Query across all indexed content
        questions = [
            "What are the key takeaways?",
            "Summarize the main arguments",
            "What examples are provided?",
        ]

        for question in questions:
            print(f"\nQ: {question}")
            result = await pipeline.query(question)
            print(f"A: {result.answer}")
            print(f"Sources: {len(result.sources)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Using Different Providers

### With Deepgram for STT

```python
from audiorag import AudioRAGPipeline, AudioRAGConfig

config = AudioRAGConfig(
    deepgram_api_key="your-deepgram-key",
    stt_provider="deepgram",
    stt_model="nova-2",
)

async with AudioRAGPipeline(config) as pipeline:
    await pipeline.index("./audio.mp3")
```

### With Anthropic for Generation

```python
config = AudioRAGConfig(
    voyage_api_key="...",  # Required for embeddings
    anthropic_api_key="sk-ant-...",
    embedding_provider="voyage",
    generation_provider="anthropic",
    generation_model="claude-3-7-sonnet-20250219",
)
```

### With Pinecone for Vector Store

```python
config = AudioRAGConfig(
    openai_api_key="sk-...",  # Required for embeddings/generation
    pinecone_api_key="pc-...",
    vector_store_provider="pinecone",
    pinecone_index_name="my-audio-index",
)
```

## Reindexing Content

By default, AudioRAG skips already-indexed files.
To force reindexing:

```python
# Force reindex a specific file
await pipeline.index(path, force=True)
```

## Budget and Verification (Optional)

```python
config = AudioRAGConfig(
    budget_enabled=True,
    budget_rpm=60,
    budget_tpm=120000,
    budget_audio_seconds_per_hour=7200,
    vector_store_verify_mode="strict",  # off | best_effort | strict
    vector_store_verify_max_attempts=5,
    vector_store_verify_wait_seconds=0.5,
    vector_id_format="auto",  # auto | sha256 | uuid5
    # vector_id_uuid5_namespace="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
)
```

Notes:
- Budget checks run before expensive operations and fail fast when limits are exceeded.
- If source duration is known, indexing reserves transcription audio budget before STT begins.
- Strict verification ensures vector writes are confirmed before source status moves to completed.

## Checking Index Status

```python
# Access state manager to check status
await pipeline._ensure_initialized()
status = await pipeline._state.get_source_status(path)
print(f"Status: {status['status']}")
```

## Next Steps

- Read the [Configuration Guide](configuration.md) for all options
- Explore [Available Providers](providers.md)
- Understand the [Pipeline Architecture](architecture.md)
- Check the [API Reference](api-reference.md) for detailed usage
