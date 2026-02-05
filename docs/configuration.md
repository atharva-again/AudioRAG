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

| Provider | Value | Default Model |
|----------|-------|---------------|
| OpenAI | `openai` | whisper-1 |
| Deepgram | `deepgram` | nova-2 |
| AssemblyAI | `assemblyai` | best |
| Groq | `groq` | whisper-large-v3 |

```bash
export AUDIORAG_STT_PROVIDER="deepgram"
export AUDIORAG_DEEPGRAM_API_KEY="..."
```

### Embedding Provider

| Provider | Value | Default Model |
|----------|-------|---------------|
| OpenAI | `openai` | text-embedding-3-small |
| Voyage | `voyage` | voyage-3.5 |
| Cohere | `cohere` | embed-v4.0 |

```bash
export AUDIORAG_EMBEDDING_PROVIDER="voyage"
export AUDIORAG_VOYAGE_API_KEY="..."
```

### Vector Store Provider

| Provider | Value | Best For |
|----------|-------|----------|
| ChromaDB | `chromadb` | Local development, small datasets |
| Pinecone | `pinecone` | Production, large scale |
| Weaviate | `weaviate` | Hybrid search, GraphQL |
| Supabase | `supabase` | Existing Supabase projects |

```bash
export AUDIORAG_VECTOR_STORE_PROVIDER="pinecone"
export AUDIORAG_PINECONE_API_KEY="..."
```

### Generation Provider

| Provider | Value | Default Model |
|----------|-------|---------------|
| OpenAI | `openai` | gpt-4o-mini |
| Anthropic | `anthropic` | claude-3-7-sonnet-20250219 |
| Gemini | `gemini` | gemini-2.0-flash-001 |

```bash
export AUDIORAG_GENERATION_PROVIDER="anthropic"
export AUDIORAG_ANTHROPIC_API_KEY="..."
```

### Reranker Provider

| Provider | Value | Default Model |
|----------|-------|---------------|
| Cohere | `cohere` | rerank-v3.5 |
| Passthrough | `passthrough` | No reranking |

```bash
export AUDIORAG_RERANKER_PROVIDER="cohere"
export AUDIORAG_COHERE_API_KEY="..."
```

## Model Configuration

### STT Models

**OpenAI:**
- `whisper-1` (default)

**Deepgram:**
- `nova-2` (default)
- `nova-2-general`, `nova-2-meeting`, `nova-2-phonecall`
- `nova-2-voicemail`, `nova-2-finance`, `nova-2-conversationalai`
- `nova-2-video`, `nova-2-medical`, `nova-1`, `enhanced`, `base`

**AssemblyAI:**
- `best` (default)
- `nano`, `universal`

**Groq:**
- `whisper-large-v3` (default)

```bash
export AUDIORAG_STT_MODEL="nova-2-meeting"
export AUDIORAG_STT_LANGUAGE="en"  # Optional language hint
```

### Embedding Models

**OpenAI:**
- `text-embedding-3-small` (default, 1536 dims)
- `text-embedding-3-large` (3072 dims)
- `text-embedding-ada-002` (1536 dims)

**Voyage:**
- `voyage-3.5` (default)
- `voyage-3.5-lite`, `voyage-4`, `voyage-4-lite`
- `voyage-4-large`, `voyage-4-nano`

**Cohere:**
- `embed-v4.0` (default)
- `embed-v3.0`, `embed-english-v3.0`
- `embed-multilingual-v3.0`, `embed-english-light-v3.0`
- `embed-multilingual-light-v3.0`

```bash
export AUDIORAG_EMBEDDING_MODEL="text-embedding-3-large"
```

### Generation Models

**OpenAI:**
- `gpt-4o-mini` (default)
- `gpt-4o`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`

**Anthropic:**
- `claude-3-7-sonnet-20250219` (default)
- `claude-3-7-opus-20250219`, `claude-3-7-haiku-20250219`
- `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`

**Gemini:**
- `gemini-2.0-flash-001` (default)
- `gemini-2.0-pro-001`, `gemini-2.0-ultra-001`
- `gemini-2.0-flash-lite-001`

```bash
export AUDIORAG_GENERATION_MODEL="gpt-4o"
```

### Reranker Models

**Cohere:**
- `rerank-v3.5` (default)
- `rerank-v3`, `rerank-english-v2.0`, `rerank-multilingual-v2.0`

```bash
export AUDIORAG_RERANKER_MODEL="rerank-v3.5"
```

## Vector Store Settings

### ChromaDB (Default)

```bash
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
# Duration of each chunk in seconds (default: 300 = 5 minutes)
export AUDIORAG_CHUNK_DURATION_SECONDS="300"

# Audio format for downloads (default: mp3)
export AUDIORAG_AUDIO_FORMAT="mp3"

# Maximum audio file size before splitting in MB (default: 24)
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
# Number of documents to retrieve from vector store (default: 10)
export AUDIORAG_RETRIEVAL_TOP_K="10"

# Number of documents to keep after reranking (default: 3)
export AUDIORAG_RERANK_TOP_N="3"

# Whether to cleanup downloaded audio after processing (default: true)
export AUDIORAG_CLEANUP_AUDIO="true"
```

## Database and Storage

```bash
# SQLite database path (default: audiorag.db)
export AUDIORAG_DATABASE_PATH="audiorag.db"

# Working directory for audio files (default: temp directory)
export AUDIORAG_WORK_DIR="/tmp/audiorag"
```

## Logging

```bash
# Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
export AUDIORAG_LOG_LEVEL="INFO"

# Log format: colored, plain (default: colored)
export AUDIORAG_LOG_FORMAT="colored"

# Include timestamps in logs (default: true)
export AUDIORAG_LOG_TIMESTAMPS="true"
```

## Retry Configuration

All external API calls use tenacity for retry logic with exponential backoff.

```bash
# Maximum retry attempts (default: 3)
export AUDIORAG_RETRY_MAX_ATTEMPTS="3"

# Minimum wait between retries in seconds (default: 4.0)
export AUDIORAG_RETRY_MIN_WAIT_SECONDS="4.0"

# Maximum wait between retries in seconds (default: 60.0)
export AUDIORAG_RETRY_MAX_WAIT_SECONDS="60.0"

# Exponential multiplier (default: 1.0)
export AUDIORAG_RETRY_EXPONENTIAL_MULTIPLIER="1.0"
```

## Complete Example .env File

```bash
# API Keys
AUDIORAG_OPENAI_API_KEY=sk-...
AUDIORAG_DEEPGRAM_API_KEY=...
AUDITORAG_COHERE_API_KEY=...

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

# Storage
AUDIORAG_DATABASE_PATH=./audiorag.db
AUDIORAG_CHROMADB_PERSIST_DIRECTORY=./chroma_db

# Logging
AUDIORAG_LOG_LEVEL=INFO
AUDIORAG_LOG_FORMAT=colored
```
