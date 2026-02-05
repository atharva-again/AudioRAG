# Providers Guide

AudioRAG uses a provider-agnostic architecture. Each stage of the pipeline can use different providers based on your needs.

## Overview

| Stage | Protocol | Available Providers |
|-------|----------|---------------------|
| Audio Source | `AudioSourceProvider` | YouTubeScraper |
| STT | `STTProvider` | OpenAI, Deepgram, AssemblyAI, Groq |
| Embeddings | `EmbeddingProvider` | OpenAI, Voyage, Cohere |
| Vector Store | `VectorStoreProvider` | ChromaDB, Pinecone, Weaviate, Supabase |
| Generation | `GenerationProvider` | OpenAI, Anthropic, Gemini |
| Reranker | `RerankerProvider` | Cohere, Passthrough |

## Audio Source Providers

### YouTubeScraper

Downloads audio from YouTube videos using yt-dlp.

**Installation:**
```bash
uv add yt-dlp pydub
```

**Configuration:**
```bash
export AUDIORAG_YOUTUBE_CONCURRENT_FRAGMENTS="3"
export AUDIORAG_YOUTUBE_SKIP_AFTER_ERRORS="3"
export AUDIORAG_YOUTUBE_DOWNLOAD_ARCHIVE="./archive.txt"
```

**Features:**
- Automatic audio format conversion
- Resumable downloads via archive file
- Concurrent fragment downloading
- Playlist and channel support

## STT Providers

### OpenAI (Default)

Uses OpenAI's Whisper API for transcription.

**Setup:**
```bash
export AUDIORAG_OPENAI_API_KEY="sk-..."
export AUDIORAG_STT_PROVIDER="openai"
export AUDIORAG_STT_MODEL="whisper-1"
```

**Models:**
- `whisper-1` (default)

**Best For:** General purpose, high accuracy

### Deepgram

Uses Deepgram's Nova-2 speech-to-text API.

**Setup:**
```bash
export AUDIORAG_DEEPGRAM_API_KEY="..."
export AUDIORAG_STT_PROVIDER="deepgram"
export AUDIORAG_STT_MODEL="nova-2"
```

**Models:**
- `nova-2` (default)
- `nova-2-general`, `nova-2-meeting`, `nova-2-phonecall`
- `nova-2-voicemail`, `nova-2-finance`, `nova-2-conversationalai`
- `nova-2-video`, `nova-2-medical`, `nova-1`, `enhanced`, `base`

**Best For:** Fast transcription, specialized domains (meetings, calls)

**Installation:**
```bash
# Deepgram SDK installed automatically
```

### AssemblyAI

Uses AssemblyAI's speech-to-text API with speaker diarization support.

**Setup:**
```bash
export AUDIORAG_ASSEMBLYAI_API_KEY="..."
export AUDIORAG_STT_PROVIDER="assemblyai"
export AUDIORAG_STT_MODEL="best"
```

**Models:**
- `best` (default) - Highest accuracy
- `nano` - Fast, cost-effective
- `universal` - Balanced

**Best For:** Speaker diarization, high accuracy

**Installation:**
```bash
uv add assemblyai
```

### Groq

Uses Groq's Whisper implementation on fast inference hardware.

**Setup:**
```bash
export AUDIORAG_GROQ_API_KEY="gsk_..."
export AUDIORAG_STT_PROVIDER="groq"
export AUDIORAG_STT_MODEL="whisper-large-v3"
```

**Models:**
- `whisper-large-v3` (default)

**Best For:** Fast, cost-effective transcription

**Installation:**
```bash
uv add groq
```

## Embedding Providers

### OpenAI (Default)

Uses OpenAI's text embedding models.

**Setup:**
```bash
export AUDIORAG_OPENAI_API_KEY="sk-..."
export AUDIORAG_EMBEDDING_PROVIDER="openai"
export AUDIORAG_EMBEDDING_MODEL="text-embedding-3-small"
```

**Models:**
- `text-embedding-3-small` (default, 1536 dims) - Best cost/performance
- `text-embedding-3-large` (3072 dims) - Highest quality
- `text-embedding-ada-002` (1536 dims) - Legacy

**Best For:** General purpose, good quality/cost ratio

### Voyage AI

Uses Voyage's specialized embedding models.

**Setup:**
```bash
export AUDIORAG_VOYAGE_API_KEY="..."
export AUDIORAG_EMBEDDING_PROVIDER="voyage"
export AUDIORAG_EMBEDDING_MODEL="voyage-3.5"
```

**Models:**
- `voyage-3.5` (default)
- `voyage-3.5-lite`, `voyage-4`, `voyage-4-lite`
- `voyage-4-large`, `voyage-4-nano`

**Best For:** High-quality embeddings, document retrieval

**Installation:**
```bash
uv add voyageai
```

### Cohere

Uses Cohere's embed models.

**Setup:**
```bash
export AUDIORAG_COHERE_API_KEY="..."
export AUDIORAG_EMBEDDING_PROVIDER="cohere"
export AUDIORAG_EMBEDDING_MODEL="embed-v4.0"
```

**Models:**
- `embed-v4.0` (default)
- `embed-v3.0`, `embed-english-v3.0`
- `embed-multilingual-v3.0`, `embed-english-light-v3.0`
- `embed-multilingual-light-v3.0`

**Best For:** Multilingual content

**Installation:**
```bash
uv add cohere
```

## Vector Store Providers

### ChromaDB (Default)

Local vector database with persistent storage.

**Setup:**
```bash
export AUDIORAG_VECTOR_STORE_PROVIDER="chromadb"
export AUDIORAG_CHROMADB_PERSIST_DIRECTORY="./chroma_db"
export AUDIORAG_CHROMADB_COLLECTION_NAME="audiorag"
```

**Best For:** Local development, small to medium datasets

**Installation:**
```bash
uv add chromadb
```

### Pinecone

Managed vector database service.

**Setup:**
```bash
export AUDIORAG_PINECONE_API_KEY="pc-..."
export AUDIORAG_VECTOR_STORE_PROVIDER="pinecone"
export AUDIORAG_PINECONE_INDEX_NAME="audiorag"
export AUDIORAG_PINECONE_NAMESPACE="default"
```

**Best For:** Production, large scale, serverless

**Installation:**
```bash
uv add pinecone-client
```

### Weaviate

Vector database with GraphQL interface.

**Setup:**
```bash
export AUDIORAG_WEAVIATE_URL="https://your-cluster.weaviate.network"
export AUDIORAG_WEAVIATE_API_KEY="..."
export AUDIORAG_VECTOR_STORE_PROVIDER="weaviate"
export AUDIORAG_WEAVIATE_COLLECTION_NAME="AudioRAG"
```

**Best For:** Hybrid search, complex queries

**Installation:**
```bash
uv add weaviate-client
```

### Supabase

PostgreSQL with pgvector extension.

**Setup:**
```bash
export AUDIORAG_SUPABASE_CONNECTION_STRING="postgresql://..."
export AUDIORAG_VECTOR_STORE_PROVIDER="supabase"
export AUDIORAG_SUPABASE_COLLECTION_NAME="audiorag"
export AUDIORAG_SUPABASE_VECTOR_DIMENSION="1536"
```

**Best For:** Existing Supabase projects, SQL compatibility

**Installation:**
```bash
uv add supabase
```

## Generation Providers

### OpenAI (Default)

Uses OpenAI's GPT models for answer generation.

**Setup:**
```bash
export AUDIORAG_OPENAI_API_KEY="sk-..."
export AUDIORAG_GENERATION_PROVIDER="openai"
export AUDIORAG_GENERATION_MODEL="gpt-4o-mini"
```

**Models:**
- `gpt-4o-mini` (default) - Fast, cost-effective
- `gpt-4o` - High quality
- `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`

**Best For:** General purpose, reliable

### Anthropic

Uses Claude models from Anthropic.

**Setup:**
```bash
export AUDIORAG_ANTHROPIC_API_KEY="sk-ant-..."
export AUDIORAG_GENERATION_PROVIDER="anthropic"
export AUDIORAG_GENERATION_MODEL="claude-3-7-sonnet-20250219"
```

**Models:**
- `claude-3-7-sonnet-20250219` (default)
- `claude-3-7-opus-20250219`, `claude-3-7-haiku-20250219`
- `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`

**Best For:** Long context, nuanced responses

**Installation:**
```bash
uv add anthropic
```

### Gemini

Uses Google's Gemini models.

**Setup:**
```bash
export AUDIORAG_GOOGLE_API_KEY="..."
export AUDIORAG_GENERATION_PROVIDER="gemini"
export AUDIORAG_GENERATION_MODEL="gemini-2.0-flash-001"
```

**Models:**
- `gemini-2.0-flash-001` (default)
- `gemini-2.0-pro-001`, `gemini-2.0-ultra-001`
- `gemini-2.0-flash-lite-001`

**Best For:** Fast responses, multimodal future

**Installation:**
```bash
uv add google-generativeai
```

## Reranker Providers

### Cohere (Default)

Uses Cohere's reranking models to improve retrieval quality.

**Setup:**
```bash
export AUDIORAG_COHERE_API_KEY="..."
export AUDIORAG_RERANKER_PROVIDER="cohere"
export AUDIORAG_RERANKER_MODEL="rerank-v3.5"
```

**Models:**
- `rerank-v3.5` (default)
- `rerank-v3`, `rerank-english-v2.0`, `rerank-multilingual-v2.0`

**Best For:** Improved retrieval accuracy

### Passthrough

Skips reranking (use retrieved documents as-is).

**Setup:**
```bash
export AUDIORAG_RERANKER_PROVIDER="passthrough"
```

**Best For:** Faster queries, lower latency

## Provider Comparison

### STT Providers

| Provider | Speed | Accuracy | Cost | Best Use Case |
|----------|-------|----------|------|---------------|
| OpenAI | Medium | High | Medium | General purpose |
| Deepgram | Fast | High | Low | Real-time, large volumes |
| AssemblyAI | Medium | Very High | Medium | Speaker diarization |
| Groq | Very Fast | High | Very Low | Cost-sensitive, speed |

### Embedding Providers

| Provider | Quality | Cost | Dimensions | Best Use Case |
|----------|---------|------|------------|---------------|
| OpenAI | High | Low | 1536-3072 | General purpose |
| Voyage | Very High | Medium | Variable | Document retrieval |
| Cohere | High | Low | Variable | Multilingual |

### Vector Store Providers

| Provider | Scalability | Latency | Management | Best Use Case |
|----------|-------------|---------|------------|---------------|
| ChromaDB | Small-Medium | Low | Self | Development, prototyping |
| Pinecone | Unlimited | Very Low | Managed | Production, scale |
| Weaviate | Large | Low | Hybrid | Complex queries |
| Supabase | Medium | Low | Managed | SQL integration |

## Custom Providers

You can implement custom providers by implementing the protocol interfaces:

```python
from pathlib import Path
from audiorag.protocols import STTProvider
from audiorag.models import TranscriptionSegment

class MyCustomSTT(STTProvider):
    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptionSegment]:
        # Your implementation
        return [
            TranscriptionSegment(
                start_time=0.0,
                end_time=10.0,
                text="Transcribed text"
            )
        ]

# Use custom provider
pipeline = AudioRAGPipeline(
    config=config,
    stt=MyCustomSTT()
)
```

See protocol definitions in `src/audiorag/protocols/` for interface details.
