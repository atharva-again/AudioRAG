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

### OpenAI

Uses OpenAI's Whisper API for transcription.

**Setup:**
```bash
export AUDIORAG_OPENAI_API_KEY="sk-..."
export AUDIORAG_STT_PROVIDER="openai"
export AUDIORAG_STT_MODEL="whisper-1"
```

**Models:**
- `whisper-1`

### Deepgram

Uses Deepgram's Nova-2 speech-to-text API.

**Setup:**
```bash
export AUDIORAG_DEEPGRAM_API_KEY="..."
export AUDIORAG_STT_PROVIDER="deepgram"
export AUDIORAG_STT_MODEL="nova-2"
```

**Models:**
- `nova-2`
- `nova-2-general`, `nova-2-meeting`, `nova-2-phonecall`
- `nova-2-voicemail`, `nova-2-finance`, `nova-2-conversationalai`
- `nova-2-video`, `nova-2-medical`, `nova-1`, `enhanced`, `base`

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
- `best`
- `nano`
- `universal`

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
- `whisper-large-v3`

**Installation:**
```bash
uv add groq
```

## Embedding Providers

### OpenAI

Uses OpenAI's text embedding models.

**Setup:**
```bash
export AUDIORAG_OPENAI_API_KEY="sk-..."
export AUDIORAG_EMBEDDING_PROVIDER="openai"
export AUDIORAG_EMBEDDING_MODEL="text-embedding-3-small"
```

**Models:**
- `text-embedding-3-small` (1536 dims)
- `text-embedding-3-large` (3072 dims)
- `text-embedding-ada-002` (1536 dims)

### Voyage AI

Uses Voyage's specialized embedding models.

**Setup:**
```bash
export AUDIORAG_VOYAGE_API_KEY="..."
export AUDIORAG_EMBEDDING_PROVIDER="voyage"
export AUDIORAG_EMBEDDING_MODEL="voyage-3.5"
```

**Models:**
- `voyage-3.5`
- `voyage-3.5-lite`, `voyage-4`, `voyage-4-lite`
- `voyage-4-large`, `voyage-4-nano`

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
- `embed-v4.0`
- `embed-v3.0`, `embed-english-v3.0`
- `embed-multilingual-v3.0`, `embed-english-light-v3.0`
- `embed-multilingual-light-v3.0`

**Installation:**
```bash
uv add cohere
```

## Vector Store Providers

### ChromaDB

Local vector database with persistent storage.

**Setup:**
```bash
export AUDIORAG_VECTOR_STORE_PROVIDER="chromadb"
export AUDIORAG_CHROMADB_PERSIST_DIRECTORY="./chroma_db"
export AUDIORAG_CHROMADB_COLLECTION_NAME="audiorag"
```

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

**Installation:**
```bash
uv add supabase
```

## Generation Providers

### OpenAI

Uses OpenAI's GPT models for answer generation.

**Setup:**
```bash
export AUDIORAG_OPENAI_API_KEY="sk-..."
export AUDIORAG_GENERATION_PROVIDER="openai"
export AUDIORAG_GENERATION_MODEL="gpt-4o-mini"
```

**Models:**
- `gpt-4o-mini`
- `gpt-4o`
- `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`

### Anthropic

Uses Claude models from Anthropic.

**Setup:**
```bash
export AUDIORAG_ANTHROPIC_API_KEY="sk-ant-..."
export AUDIORAG_GENERATION_PROVIDER="anthropic"
export AUDIORAG_GENERATION_MODEL="claude-3-7-sonnet-20250219"
```

**Models:**
- `claude-3-7-sonnet-20250219`
- `claude-3-7-opus-20250219`, `claude-3-7-haiku-20250219`
- `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`

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
- `gemini-2.0-flash-001`
- `gemini-2.0-pro-001`, `gemini-2.0-ultra-001`
- `gemini-2.0-flash-lite-001`

**Installation:**
```bash
uv add google-generativeai
```

## Reranker Providers

### Cohere

Uses Cohere's reranking models to improve retrieval quality.

**Setup:**
```bash
export AUDIORAG_COHERE_API_KEY="..."
export AUDIORAG_RERANKER_PROVIDER="cohere"
export AUDIORAG_RERANKER_MODEL="rerank-v3.5"
```

**Models:**
- `rerank-v3.5`
- `rerank-v3`, `rerank-english-v2.0`, `rerank-multilingual-v2.0`

### Passthrough

Skips reranking (use retrieved documents as-is).

**Setup:**
```bash
export AUDIORAG_RERANKER_PROVIDER="passthrough"
```

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
