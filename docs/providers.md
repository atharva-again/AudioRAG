# Providers Guide

AudioRAG uses a provider-agnostic architecture. Each stage of the pipeline can use different providers based on your needs.

## Overview

| Stage | Protocol | Available Providers |
|-------|----------|---------------------|
| Audio Source | `AudioSourceProvider` | YouTubeSource, LocalSource, URLSource |
| STT | `STTProvider` | OpenAI, Deepgram, AssemblyAI, Groq |
| Embeddings | `EmbeddingProvider` | OpenAI, Voyage, Cohere |
| Vector Store | `VectorStoreProvider` | ChromaDB, Pinecone, Weaviate, Supabase |
| Vector Verification | `VerifiableVectorStoreProvider` | ChromaDB, Pinecone, Weaviate, Supabase |
| Generation | `GenerationProvider` | OpenAI, Anthropic, Gemini |
| Reranker | `RerankerProvider` | Cohere, Passthrough |

## Audio Source Providers

### YouTubeSource

Downloads audio from YouTube videos using yt-dlp with 2026 bleeding-edge extraction patterns.

**Installation:**
```bash
# Requires ffmpeg to be installed on your system
# On macOS: brew install ffmpeg
# On Ubuntu/Debian: sudo apt install ffmpeg

# YouTube extraction
uv add yt-dlp
# For YouTube 2026 JS challenge solving (mandatory since 2025.11.12)
uv add yt-dlp-ejs curl-cffi
```

**Basic Configuration:**
```bash
export AUDIORAG_YOUTUBE_CONCURRENT_FRAGMENTS="3"
export AUDIORAG_YOUTUBE_SKIP_AFTER_ERRORS="3"
export AUDIORAG_YOUTUBE_DOWNLOAD_ARCHIVE="./archive.txt"
```

**Advanced Configuration (YouTube 2026 Stability):**

For reliable extraction from YouTube, configure visitor context and PO tokens:

```bash
# PO Token: Required for bypassing YouTube bot detection
# Get from: https://github.com/yt-dlp/yt-dlp/wiki/Extractors#po-token-guide
export AUDIORAG_YOUTUBE_PO_TOKEN="MnXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX..."

# Visitor Data: Session identifier (cryptographically bound to PO token)
# Extract from browser cookies or network inspector
export AUDIORAG_YOUTUBE_VISITOR_DATA="CgtZZXXXXXXXXXXXXXXXXX..."

# Data Sync ID: Account session identifier (optional, for authenticated access)
export AUDIORAG_YOUTUBE_DATA_SYNC_ID="XXXXXXXXXXXXXXXXXXXXXXXXXXXX..."

# JS Runtime: External JavaScript runtime for signature challenge solving
# Options: "deno" (recommended, fastest), "node", "bun"
export AUDIORAG_JS_RUNTIME="deno"
```

**Important Notes:**
- **PO Token Binding**: PO tokens are cryptographically bound to visitor_data. Tokens generated for one visitor_data value will fail with a different visitor_data.
- **JS Runtime**: YouTube requires external JS runtime since 2025.11.12 for signature challenges. Deno is recommended per yt-dlp documentation.
- **Interactive Setup**: Run `audiorag setup` for guided configuration with prompts for PO token and visitor context.

**Features:**
- Automatic audio format conversion
- Resumable downloads via archive file
- Concurrent fragment downloading
- Playlist and channel support
- Pre-flight metadata extraction for budget checks
- Browser impersonation (chrome-120) for evasion
- Optimized player skip patterns (25-33% latency reduction)
- Cryptographically-bound PO token support

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

AudioRAG maintains canonical SHA-256 IDs in SQLite state, then adapts vector
IDs per provider when writing embeddings.

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
uv add pinecone
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

**Vector ID behavior:**
- Default (`AUDIORAG_VECTOR_ID_FORMAT=auto`): deterministic UUID5 IDs.
- Override with `AUDIORAG_VECTOR_ID_FORMAT=sha256` if you want raw canonical IDs.

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
uv add vecs
```

**Vector ID behavior:**
- Default (`AUDIORAG_VECTOR_ID_FORMAT=auto`): canonical SHA-256 string IDs.
- This matches vecs collection records where the `id` field is a string identifier.
- Set `AUDIORAG_VECTOR_ID_FORMAT=uuid5` only if your surrounding schema/tooling expects UUID-shaped IDs.

## Vector ID Strategy

```bash
export AUDIORAG_VECTOR_ID_FORMAT="auto"  # auto | sha256 | uuid5
export AUDIORAG_VECTOR_ID_UUID5_NAMESPACE="6ba7b810-9dad-11d1-80b4-00c04fd430c8"  # optional
```

- `auto`: provider-aware defaults (currently Weaviate -> UUID5, others -> SHA-256).
- `sha256`: always use canonical chunk IDs generated by `StateManager`.
- `uuid5`: deterministically convert canonical IDs to UUID5 before vector store writes.

If you change vector ID strategy for sources that were previously indexed with a
different strategy, run reindex with `force=True`.

## Vector Store Verification

All built-in vector stores implement `verify(ids)`.

Recommended mode by deployment type:
- `best_effort`: default; safer DX when stores are eventually consistent.
- `strict`: strongest correctness; indexing fails unless IDs are verified.
- `off`: disables verification.

```bash
export AUDIORAG_VECTOR_STORE_VERIFY_MODE="strict"
export AUDIORAG_VECTOR_STORE_VERIFY_MAX_ATTEMPTS="5"
export AUDIORAG_VECTOR_STORE_VERIFY_WAIT_SECONDS="0.5"
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
uv add google-genai
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
from audiorag.core.protocols import STTProvider
from audiorag.core.models import TranscriptionSegment

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

See protocol definitions in `src/audiorag/core/protocols/` for interface details.
