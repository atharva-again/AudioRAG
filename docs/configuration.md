# Configuration Guide

AudioRAG uses pydantic-settings for configuration management. All settings support environment variables with the `AUDIORAG_` prefix.

## Configuration Methods

Configuration is resolved in this order (later overrides earlier):

1. Default values in code
2. Environment variables (with `AUDIORAG_` prefix)
3. `.env` file
4. Direct instantiation

```python
from audiorag import AudioRAGConfig

# Method 1: Environment variables
# export AUDIORAG_OPENAI_API_KEY="sk-..."
config = AudioRAGConfig()

# Method 2: Direct instantiation
config = AudioRAGConfig(
    openai_api_key="sk-...",
    stt_provider="deepgram",
)

# Method 3: Mixed (env vars + overrides)
# export AUDIORAG_OPENAI_API_KEY="sk-..."
config = AudioRAGConfig(stt_provider="deepgram")
```

## Provider Selection

### STT Provider

| Provider | Value | Models |
|----------|-------|--------|
| OpenAI | `openai` | whisper-1 |
| Deepgram | `deepgram` | nova-2, nova-2-general, nova-2-meeting, nova-2-phonecall, nova-2-voicemail, nova-2-finance, nova-2-conversationalai, nova-2-video, nova-2-medical, nova-1, enhanced, base |
| AssemblyAI | `assemblyai` | best, nano, universal |
| Groq | `groq` | whisper-large-v3 |

```bash
export AUDIORAG_STT_PROVIDER="deepgram"
export AUDIORAG_DEEPGRAM_API_KEY="..."
```

### Embedding Provider

| Provider | Value | Models |
|----------|-------|--------|
| OpenAI | `openai` | text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 |
| Voyage | `voyage` | voyage-3.5, voyage-3.5-lite, voyage-4, voyage-4-lite, voyage-4-large, voyage-4-nano |
| Cohere | `cohere` | embed-v4.0, embed-v3.0, embed-english-v3.0, embed-multilingual-v3.0, embed-english-light-v3.0, embed-multilingual-light-v3.0 |

```bash
export AUDIORAG_EMBEDDING_PROVIDER="voyage"
export AUDIORAG_VOYAGE_API_KEY="..."
```

### Vector Store Provider

| Provider | Value |
|----------|-------|
| ChromaDB | `chromadb` |
| Pinecone | `pinecone` |
| Weaviate | `weaviate` |
| Supabase | `supabase` |

```bash
export AUDIORAG_VECTOR_STORE_PROVIDER="pinecone"
export AUDIORAG_PINECONE_API_KEY="..."
```

### Generation Provider

| Provider | Value | Models |
|----------|-------|--------|
| OpenAI | `openai` | gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| Anthropic | `anthropic` | claude-3-7-sonnet-20250219, claude-3-7-opus-20250219, claude-3-7-haiku-20250219, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022 |
| Gemini | `gemini` | gemini-2.0-flash-001, gemini-2.0-pro-001, gemini-2.0-ultra-001, gemini-2.0-flash-lite-001 |

```bash
export AUDIORAG_GENERATION_PROVIDER="anthropic"
export AUDIORAG_ANTHROPIC_API_KEY="..."
```

### Reranker Provider

| Provider | Value | Models |
|----------|-------|--------|
| Cohere | `cohere` | rerank-v3.5, rerank-v3, rerank-english-v2.0, rerank-multilingual-v2.0 |
| Passthrough | `passthrough` | No reranking |

```bash
export AUDIORAG_RERANKER_PROVIDER="cohere"
export AUDIORAG_COHERE_API_KEY="..."
```

## Model Configuration

### STT Models

**OpenAI:**
- `whisper-1`

**Deepgram:**
- `nova-2`
- `nova-2-general`, `nova-2-meeting`, `nova-2-phonecall`
- `nova-2-voicemail`, `nova-2-finance`, `nova-2-conversationalai`
- `nova-2-video`, `nova-2-medical`, `nova-1`, `enhanced`, `base`

**AssemblyAI:**
- `best`
- `nano`, `universal`

**Groq:**
- `whisper-large-v3`

```bash
export AUDIORAG_STT_MODEL="nova-2-meeting"
export AUDIORAG_STT_LANGUAGE="en"  # Optional language hint
```

### Embedding Models

**OpenAI:**
- `text-embedding-3-small` (1536 dims)
- `text-embedding-3-large` (3072 dims)
- `text-embedding-ada-002` (1536 dims)

**Voyage:**
- `voyage-3.5`
- `voyage-3.5-lite`, `voyage-4`, `voyage-4-lite`
- `voyage-4-large`, `voyage-4-nano`

**Cohere:**
- `embed-v4.0`
- `embed-v3.0`, `embed-english-v3.0`
- `embed-multilingual-v3.0`, `embed-english-light-v3.0`
- `embed-multilingual-light-v3.0`

```bash
export AUDIORAG_EMBEDDING_MODEL="text-embedding-3-large"
```

### Generation Models

**OpenAI:**
- `gpt-4o-mini`
- `gpt-4o`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`

**Anthropic:**
- `claude-3-7-sonnet-20250219`
- `claude-3-7-opus-20250219`, `claude-3-7-haiku-20250219`
- `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`

**Gemini:**
- `gemini-2.0-flash-001`
- `gemini-2.0-pro-001`, `gemini-2.0-ultra-001`
- `gemini-2.0-flash-lite-001`

```bash
export AUDIORAG_GENERATION_MODEL="gpt-4o"
```

### Reranker Models

**Cohere:**
- `rerank-v3.5`
- `rerank-v3`, `rerank-english-v2.0`, `rerank-multilingual-v2.0`

```bash
export AUDIORAG_RERANKER_MODEL="rerank-v3.5"
```

## Vector Store Settings

### ChromaDB

```bash
export AUDIORAG_VECTOR_STORE_PROVIDER="chromadb"
export AUDIORAG_CHROMADB_PERSIST_DIRECTORY="./chroma_db"
export AUDIORAG_CHROMADB_COLLECTION_NAME="audiorag"
```

### Pinecone

```bash
export AUDIORAG_PINECONE_API_KEY="pc-..."
export AUDIORAG_PINECONE_INDEX_NAME="audiorag"
export AUDIORAG_PINECONE_NAMESPACE="default"
```

### Weaviate

```bash
export AUDIORAG_WEAVIATE_URL="https://your-cluster.weaviate.network"
export AUDIORAG_WEAVIATE_API_KEY="..."
export AUDIORAG_WEAVIATE_COLLECTION_NAME="AudioRAG"
```

### Supabase

```bash
export AUDIORAG_SUPABASE_CONNECTION_STRING="postgresql://..."
export AUDIORAG_SUPABASE_COLLECTION_NAME="audiorag"
export AUDIORAG_SUPABASE_VECTOR_DIMENSION="1536"
```

## Audio Processing

```bash
# Duration of each chunk in seconds
export AUDIORAG_CHUNK_DURATION_SECONDS="30"

# Audio format for downloads
export AUDIORAG_AUDIO_FORMAT="mp3"

# Maximum audio file size before splitting in MB
export AUDIORAG_AUDIO_SPLIT_MAX_SIZE_MB="24"
```

## YouTube Scraping

```bash
# Path to download archive file (for resumable scraping)
export AUDIORAG_YOUTUBE_DOWNLOAD_ARCHIVE="./download_archive.txt"

# Concurrent fragments per download (higher = faster, more bandwidth)
export AUDIORAG_YOUTUBE_CONCURRENT_FRAGMENTS="3"

# Skip playlist after N consecutive errors
export AUDIORAG_YOUTUBE_SKIP_AFTER_ERRORS="3"

# Batch size for channel scraping
export AUDIORAG_YOUTUBE_BATCH_SIZE="100"

# Max concurrent downloads within a batch
export AUDIORAG_YOUTUBE_MAX_CONCURRENT="3"
```

## Retrieval Settings

```bash
# Number of documents to retrieve from vector store
export AUDIORAG_RETRIEVAL_TOP_K="10"

# Number of documents to keep after reranking
export AUDIORAG_RERANK_TOP_N="3"

# Whether to cleanup downloaded audio after processing
export AUDIORAG_CLEANUP_AUDIO="true"
```

## Budget Governor

Use proactive, fail-fast limits to prevent avoidable 429s and runaway usage.

```bash
# Enable governor
export AUDIORAG_BUDGET_ENABLED="true"

# Global limits
export AUDIORAG_BUDGET_RPM="60"
export AUDIORAG_BUDGET_TPM="120000"
export AUDIORAG_BUDGET_AUDIO_SECONDS_PER_HOUR="7200"

# Token estimation ratio when exact token count is unknown
export AUDIORAG_BUDGET_TOKEN_CHARS_PER_TOKEN="4"

# Provider-specific overrides (JSON)
export AUDIORAG_BUDGET_PROVIDER_OVERRIDES='{"openai": {"rpm": 30, "tpm": 60000}, "deepgram": {"audio_seconds_per_hour": 3600}}'
```

Behavior:
- Budget is checked before API-heavy stages (transcribe/embed/query/generate/rerank).
- If audio duration is known, transcription budget is reserved before STT starts.
- Limits are persisted in SQLite for restart/process safety.

## Vector Write Verification

```bash
# Verification mode: off | best_effort | strict
export AUDIORAG_VECTOR_STORE_VERIFY_MODE="best_effort"

# Retry behavior for verification checks
export AUDIORAG_VECTOR_STORE_VERIFY_MAX_ATTEMPTS="5"
export AUDIORAG_VECTOR_STORE_VERIFY_WAIT_SECONDS="0.5"
```

Behavior:
- After embeddings are written, supported stores verify inserted IDs.
- `best_effort`: logs warning if verification fails.
- `strict`: indexing fails if verification does not pass after retries.

## Database and Storage

```bash
# SQLite database path
export AUDIORAG_DATABASE_PATH="audiorag.db"

# Working directory for audio files
export AUDIORAG_WORK_DIR="/tmp/audiorag"
```

## Logging

```bash
# Log level: DEBUG, INFO, WARNING, ERROR
export AUDIORAG_LOG_LEVEL="INFO"

# Log format: colored, plain
export AUDIORAG_LOG_FORMAT="colored"

# Include timestamps in logs
export AUDIORAG_LOG_TIMESTAMPS="true"
```

## Retry Configuration

All external API calls use tenacity for retry logic with exponential backoff.

```bash
# Maximum retry attempts
export AUDIORAG_RETRY_MAX_ATTEMPTS="3"

# Minimum wait between retries in seconds
export AUDIORAG_RETRY_MIN_WAIT_SECONDS="4.0"

# Maximum wait between retries in seconds
export AUDIORAG_RETRY_MAX_WAIT_SECONDS="60.0"

# Exponential multiplier
export AUDIORAG_RETRY_EXPONENTIAL_MULTIPLIER="1.0"
```

## Complete Example .env File

```bash
# API Keys
AUDIORAG_OPENAI_API_KEY=sk-...
AUDIORAG_DEEPGRAM_API_KEY=...
AUDIORAG_COHERE_API_KEY=...

# Provider Selection
AUDIORAG_STT_PROVIDER=deepgram
AUDIORAG_EMBEDDING_PROVIDER=openai
AUDIORAG_VECTOR_STORE_PROVIDER=chromadb
AUDIORAG_GENERATION_PROVIDER=openai
AUDIORAG_RERANKER_PROVIDER=cohere

# Models
AUDIORAG_STT_MODEL=nova-2
AUDIORAG_EMBEDDING_MODEL=text-embedding-3-small
AUDIORAG_GENERATION_MODEL=gpt-4o-mini
AUDIORAG_RERANKER_MODEL=rerank-v3.5

# Processing
AUDIORAG_CHUNK_DURATION_SECONDS=300
AUDIORAG_AUDIO_FORMAT=mp3
AUDIORAG_AUDIO_SPLIT_MAX_SIZE_MB=24

# Retrieval
AUDIORAG_RETRIEVAL_TOP_K=10
AUDIORAG_RERANK_TOP_N=3

# Budget governor
AUDIORAG_BUDGET_ENABLED=true
AUDIORAG_BUDGET_RPM=60
AUDIORAG_BUDGET_TPM=120000
AUDIORAG_BUDGET_AUDIO_SECONDS_PER_HOUR=7200

# Vector write verification
AUDIORAG_VECTOR_STORE_VERIFY_MODE=best_effort
AUDIORAG_VECTOR_STORE_VERIFY_MAX_ATTEMPTS=5
AUDIORAG_VECTOR_STORE_VERIFY_WAIT_SECONDS=0.5

# Storage
AUDIORAG_DATABASE_PATH=./audiorag.db
AUDIORAG_CHROMADB_PERSIST_DIRECTORY=./chroma_db

# Logging
AUDIORAG_LOG_LEVEL=INFO
AUDIORAG_LOG_FORMAT=colored
```
