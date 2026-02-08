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
    pipeline = AudioRAGPipeline(config)
    
    # Index a YouTube video
    await pipeline.index("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print("Indexing complete!")

asyncio.run(index_content())
```

### 3. Query the Indexed Content

```python
async def query_content():
    config = AudioRAGConfig()
    pipeline = AudioRAGPipeline(config)
    
    result = await pipeline.query("What is the main topic of this video?")
    
    print("Answer:", result.answer)
    print("\nSources:")
    for source in result.sources:
        print(f"  - {source.video_title}")
        print(f"    Timestamp: {source.start_time}s")
        print(f"    Relevance: {source.relevance_score:.2f}")
        print(f"    URL: {source.youtube_timestamp_url}")

asyncio.run(query_content())
```

## Complete Example

```python
import asyncio
from audiorag import AudioRAGPipeline, AudioRAGConfig

async def main():
    # Initialize
    config = AudioRAGConfig()
    pipeline = AudioRAGPipeline(config)
    
    # Index multiple videos
    urls = [
        "https://www.youtube.com/watch?v=video1",
        "https://www.youtube.com/watch?v=video2",
        "https://www.youtube.com/watch?v=video3",
    ]
    
    for url in urls:
        print(f"Indexing: {url}")
        await pipeline.index(url)
    
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
pipeline = AudioRAGPipeline(config)
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

By default, AudioRAG skips already-indexed URLs. To force reindexing:

```python
# Force reindex a specific URL
await pipeline.index(url, force=True)
```

## Checking Index Status

```python
# Access state manager to check status
await pipeline._ensure_initialized()
status = await pipeline._state.get_source_status(url)
print(f"Status: {status['status']}")
```

## Next Steps

- Read the [Configuration Guide](configuration.md) for all options
- Explore [Available Providers](providers.md)
- Understand the [Pipeline Architecture](architecture.md)
- Check the [API Reference](api-reference.md) for detailed usage
