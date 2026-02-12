# API Reference

Complete reference for AudioRAG classes, methods, and protocols.

## Core Classes

### AudioRAGPipeline

Main orchestrator for indexing and querying audio content.

**Location:** `audiorag.pipeline.AudioRAGPipeline`

```python
class AudioRAGPipeline:
    def __init__(
        self,
        config: AudioRAGConfig,
        *,
        audio_source: AudioSourceProvider | None = None,
        stt: STTProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: VectorStoreProvider | None = None,
        generator: GenerationProvider | None = None,
        reranker: RerankerProvider | None = None,
    ) -> None
```

**Parameters:**
- `config`: AudioRAGConfig instance with all settings
- `audio_source`: Custom audio source provider (default: YouTubeSource)
- `stt`: Custom STT provider (default: from config.stt_provider)
- `embedder`: Custom embedding provider (default: from config.embedding_provider)
- `vector_store`: Custom vector store (default: from config.vector_store_provider)
- `generator`: Custom generation provider (default: from config.generation_provider)
- `reranker`: Custom reranker (default: from config.reranker_provider)

#### index

```python
async def index(self, url: str, *, force: bool = False) -> None
```

Index audio from a URL/path through the full pipeline.

`index()` performs source discovery first. If `url` is a YouTube playlist/channel
or a local directory path, it is expanded into individual sources and each source
is tracked independently for resumability.

**Stages:**
1. Download audio from URL
2. Split large files if needed
3. Transcribe audio to text
4. Chunk transcription by duration
5. Generate embeddings
6. Store in vector database
7. Mark completed

**Parameters:**
- `url`: URL/path to index
- `force`: If True, re-index even if already completed

**Raises:**
- `PipelineError`: If any pipeline stage fails
- `ProviderError`: If external provider fails
- `StateError`: If database operations fail

**Example:**
```python
# Using async context manager (recommended)
async with AudioRAGPipeline(config) as pipeline:
    # First index
    await pipeline.index("https://youtube.com/watch?v=...")

    # Skip if already indexed
    await pipeline.index("https://youtube.com/watch?v=...")  # No-op

    # Force reindex
    await pipeline.index("https://youtube.com/watch?v=...", force=True)

    # Playlist URL (auto-expanded to per-video sources)
    await pipeline.index("https://youtube.com/playlist?list=...")
```

#### close

```python
async def close(self) -> None
```

Close all resources and connections.

Should be called when the pipeline is no longer needed to ensure proper cleanup of database connections and other resources. Safe to call multiple times (idempotent).

**When to use:**
- When not using the async context manager
- In long-running applications to release resources
- In signal handlers for graceful shutdown

**Example:**
```python
pipeline = AudioRAGPipeline(config)
try:
    await pipeline.index_many(urls)
    result = await pipeline.query("question")
finally:
    await pipeline.close()  # Always cleanup
```

#### Async Context Manager

`AudioRAGPipeline` supports async context manager protocol for automatic resource cleanup.

**Example:**
```python
async with AudioRAGPipeline(config) as pipeline:
    await pipeline.index_many(urls)
    result = await pipeline.query("question")
# Resources automatically cleaned up on exit
```

Resources are cleaned up automatically when:
- Exiting the context normally (success)
- An exception is raised inside the context
- The context is exited via `return`, `break`, or `continue`

#### index_many

```python
async def index_many(
    self,
    inputs: list[str],
    *,
    force: bool = False,
    raise_on_error: bool = True,
) -> BatchIndexResult
```

Index multiple inputs in one call with automatic source discovery.

**Parameters:**
- `inputs`: List of URLs and/or local paths
- `force`: If True, re-index even if already completed
- `raise_on_error`: If True, raise when any source fails. If False, return failures in result.

**Returns:**
- `BatchIndexResult` containing discovered sources, indexed/skipped sources, and failures.

**Raises:**
- `PipelineError`: If one or more sources fail indexing and `raise_on_error=True`

**Example:**
```python
result = await pipeline.index_many([
    "https://youtube.com/playlist?list=...",
    "./podcasts/",
    "https://youtube.com/watch?v=singleVideo",
], raise_on_error=False)

print(result.indexed_sources)
print(result.failures)
```

#### query

```python
async def query(self, query: str) -> QueryResult
```

Query the indexed audio content.

**Steps:**
1. Embed the query text
2. Retrieve top-k documents from vector store
3. Rerank by relevance
4. Generate answer using LLM
5. Return answer with sources

**Parameters:**
- `query`: The user's question as a string

**Returns:**
- `QueryResult` with answer and sources

**Raises:**
- `ProviderError`: If embedding or generation fails
- `StateError`: If database access fails

**Example:**
```python
result = await pipeline.query("What are the main points?")
print(result.answer)
for source in result.sources:
    print(f"{source.title} at {source.start_time}s")
```

### AudioRAGConfig

Configuration management with environment variable support.

**Location:** `audiorag.core.config.AudioRAGConfig`

**Environment Variable Prefix:** `AUDIORAG_`

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `stt_provider` | str | STT provider name (openai, deepgram, assemblyai, groq) |
| `embedding_provider` | str | Embedding provider name (openai, voyage, cohere) |
| `vector_store_provider` | str | Vector store provider name (chromadb, pinecone, weaviate, supabase) |
| `generation_provider` | str | Generation provider name (openai, anthropic, gemini) |
| `reranker_provider` | str | Reranker provider name (cohere, passthrough) |
| `openai_api_key` | str | OpenAI API key |
| `deepgram_api_key` | str | Deepgram API key |
| `assemblyai_api_key` | str | AssemblyAI API key |
| `groq_api_key` | str | Groq API key |
| `voyage_api_key` | str | Voyage AI API key |
| `cohere_api_key` | str | Cohere API key |
| `anthropic_api_key` | str | Anthropic API key |
| `google_api_key` | str | Google API key for Gemini |
| `chunk_duration_seconds` | int | Duration per chunk in seconds |
| `retrieval_top_k` | int | Documents to retrieve from vector store |
| `rerank_top_n` | int | Documents to keep after reranking |
| `database_path` | str | SQLite database path |
| `cleanup_audio` | bool | Whether to cleanup temp files after processing |
| `log_level` | str | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `budget_enabled` | bool | Enable proactive API budget enforcement |
| `budget_rpm` | int \| None | Global requests-per-minute limit |
| `budget_tpm` | int \| None | Global tokens-per-minute limit |
| `budget_audio_seconds_per_hour` | int \| None | Global STT audio-seconds/hour limit |
| `budget_provider_overrides` | dict | Provider-specific limit overrides |
| `vector_store_verify_mode` | str | Vector verify mode: off, best_effort, strict |
| `vector_store_verify_max_attempts` | int | Max verification attempts |
| `vector_store_verify_wait_seconds` | float | Delay between verification attempts |
| `vector_id_format` | str | Vector ID strategy: auto, sha256, uuid5 |
| `vector_id_uuid5_namespace` | str \| None | Optional UUID namespace for deterministic UUID5 conversion |

**Methods:**

```python
def get_stt_model(self) -> str
```

Get the appropriate STT model based on provider. Returns provider-specific default if using generic name.

```python
def get_embedding_model(self) -> str
```

Get the appropriate embedding model based on provider.

```python
def get_generation_model(self) -> str
```

Get the appropriate generation model based on provider.

### Vector ID behavior

- Canonical `source_id` and `chunk_id` in SQLite state are SHA-256 hashes.
- `vector_id_format="auto"` applies provider defaults for vector-store-facing IDs.
- Current built-in default mapping: `weaviate -> uuid5`, others -> `sha256`.
- `vector_id_uuid5_namespace` allows deterministic UUID5 namespace override.
- If a source has existing metadata with a different vector ID strategy, run
  `index(..., force=True)` before reindexing that source.

**Example:**
```python
from audiorag import AudioRAGConfig

# From environment variables
config = AudioRAGConfig()

# With overrides
config = AudioRAGConfig(
    stt_provider="deepgram",
    chunk_duration_seconds=600,
)
```

## Data Models

### QueryResult

Result of a query with generated answer and sources.

**Location:** `audiorag.core.models.QueryResult`

```python
class QueryResult(BaseModel):
    answer: str
    sources: list[Source]
```

**Attributes:**
- `answer`: Generated answer text
- `sources`: List of Source objects with timestamps and relevance

### Source

A source document with relevance score and timing information.

**Location:** `audiorag.core.models.Source`

```python
class Source(BaseModel):
    text: str
    start_time: float
    end_time: float
    source_url: str
    title: str
    relevance_score: float
```

**Attributes:**
- `text`: The source text content
- `start_time`: Start time in seconds
- `end_time`: End time in seconds
- `source_url`: Original source URL
- `title`: Title of the source audio
- `relevance_score`: Relevance score from reranker (0.0 to 1.0)

**Example:**
```python
result = await pipeline.query("...")
for source in result.sources:
    print(source.text)
    print(f"URL: {source.source_url}")
```

### ChunkMetadata

Metadata for a text chunk.

**Location:** `audiorag.core.models.ChunkMetadata`

```python
class ChunkMetadata(BaseModel):
    start_time: float
    end_time: float
    text: str
    source_url: str
    title: str
```

### TranscriptionSegment

A segment of transcribed audio.

**Location:** `audiorag.core.models.TranscriptionSegment`

```python
class TranscriptionSegment(BaseModel):
    start_time: float
    end_time: float
    text: str
```

### AudioFile

Audio file metadata.

**Location:** `audiorag.core.models.AudioFile`

```python
class AudioFile(BaseModel):
    path: Path
    source_url: str
    title: str
    duration: float | None = None
```

### IndexingStatus

Pipeline stages for audio indexing (StrEnum).

**Location:** `audiorag.core.models.IndexingStatus`

**Values:**
- `DOWNLOADING`
- `DOWNLOADED`
- `SPLITTING`
- `TRANSCRIBING`
- `TRANSCRIBED`
- `CHUNKING`
- `CHUNKED`
- `EMBEDDING`
- `EMBEDDED`
- `COMPLETED`
- `FAILED`

## Protocols

### STTProvider

Protocol for speech-to-text providers.

**Location:** `audiorag.core.protocols.STTProvider`

```python
@runtime_checkable
class STTProvider(Protocol):
    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptionSegment]: ...
```

**Parameters:**
- `audio_path`: Path to audio file
- `language`: Optional language code (e.g., "en", "es")

**Returns:**
- List of TranscriptionSegment objects

### EmbeddingProvider

Protocol for text embedding providers.

**Location:** `audiorag.core.protocols.EmbeddingProvider`

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

**Parameters:**
- `texts`: List of text strings to embed

**Returns:**
- List of embedding vectors (list of floats)

### VectorStoreProvider

Protocol for vector database providers.

**Location:** `audiorag.core.protocols.VectorStoreProvider`

```python
@runtime_checkable
class VectorStoreProvider(Protocol):
    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None: ...

    async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]: ...

    async def delete_by_source(self, source_url: str) -> None: ...
```

```python
@runtime_checkable
class VerifiableVectorStoreProvider(Protocol):
    async def verify(self, ids: list[str]) -> bool: ...
```

### GenerationProvider

Protocol for text generation providers.

**Location:** `audiorag.core.protocols.GenerationProvider`

```python
@runtime_checkable
class GenerationProvider(Protocol):
    async def generate(self, query: str, context: list[str]) -> str: ...
```

**Parameters:**
- `query`: The user's question
- `context`: List of context documents

**Returns:**
- Generated answer string

### RerankerProvider

Protocol for document reranking providers.

**Location:** `audiorag.core.protocols.RerankerProvider`

```python
@runtime_checkable
class RerankerProvider(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_n: int = 3
    ) -> list[tuple[int, float]]: ...
```

**Parameters:**
- `query`: Original query string
- `documents`: List of documents to rerank
- `top_n`: Number of top documents to return

**Returns:**
- List of tuples (original_index, relevance_score)

### AudioSourceProvider

Protocol for audio source providers.

**Location:** `audiorag.core.protocols.AudioSourceProvider`

```python
@runtime_checkable
class AudioSourceProvider(Protocol):
    async def download(
        self, url: str, output_dir: Path, audio_format: str = "mp3"
    ) -> AudioFile: ...
```

## State Management

### StateManager

Manages persistent state for audio sources and chunks using SQLite.

**Location:** `audiorag.core.state.StateManager`

```python
class StateManager:
    def __init__(self, db_path: str | Path) -> None
    
    async def initialize(self) -> None
    
    async def get_source_status(
        self, source_path: str
    ) -> dict[str, Any] | None
    
    async def upsert_source(
        self,
        source_path: str,
        status: str,
        metadata: dict[str, Any] | None = None
    ) -> str
    
    async def update_source_status(
        self,
        source_path: str,
        status: str,
        metadata: dict[str, Any] | None = None
    ) -> None
    
    async def store_chunks(
        self, source_path: str, chunks: list[dict[str, Any]]
    ) -> list[str]
    
    async def get_chunks_for_source(
        self, source_path: str
    ) -> list[dict[str, Any]]
    
    async def delete_source(self, source_path: str) -> bool
    
    async def close(self) -> None
```

**Context Manager Support:**
```python
async with StateManager("audiorag.db") as state:
    await state.upsert_source(url, "processing")
```

## Retry Configuration

### RetryConfig

Configuration for retry behavior.

**Location:** `audiorag.core.retry_config.RetryConfig`

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    min_wait_seconds: float = 4.0
    max_wait_seconds: float = 60.0
    exponential_multiplier: float = 1.0
```

## Logging

### get_logger

Get a structured logger instance.

**Location:** `audiorag.core.logging_config.get_logger`

```python
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger
```

**Example:**
```python
from audiorag import get_logger

logger = get_logger(__name__)
logger.info("operation_started")
logger.bind(user_id="123").info("user_action")
```

### configure_logging

Configure structured logging.

**Location:** `audiorag.core.logging_config.configure_logging`

```python
def configure_logging(
    log_level: str = "INFO",
    log_format: str = "colored",
    log_timestamps: bool = True,
) -> None
```

### Timer

Context manager for timing operations.

**Location:** `audiorag.core.logging_config.Timer`

```python
class Timer:
    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        operation: str,
        **context: Any
    ) -> None
    
    def complete(self, **extra_context: Any) -> None
```

**Example:**
```python
from audiorag.logging_config import Timer, get_logger

logger = get_logger(__name__)

with Timer(logger, "download", url=url) as timer:
    result = await download_audio(url)
    timer.complete(bytes_downloaded=len(result))
```

## Exceptions

### AudioRAGError

Base exception for all AudioRAG errors.

**Location:** `audiorag.core.exceptions.AudioRAGError`

### PipelineError

Exception raised when pipeline execution fails.

**Location:** `audiorag.core.exceptions.PipelineError`

```python
class PipelineError(AudioRAGError):
    def __init__(
        self, message: str, stage: str, source_url: str | None = None
    ) -> None
    
    stage: str
    source_url: str | None
```

### ProviderError

Exception raised when an external provider fails.

**Location:** `audiorag.core.exceptions.ProviderError`

```python
class ProviderError(AudioRAGError):
    def __init__(
        self, message: str, provider: str, retryable: bool = False
    ) -> None
    
    provider: str
    retryable: bool
```

### ConfigurationError

Exception raised when configuration validation fails.

**Location:** `audiorag.core.exceptions.ConfigurationError`

### StateError

Exception raised when database or state management fails.

**Location:** `audiorag.core.exceptions.StateError`

### BudgetExceededError

Raised when proactive budget checks fail.

**Location:** `audiorag.core.exceptions.BudgetExceededError`

### DiscoveryError

Raised when source discovery or expansion fails (e.g., YouTube playlist/channel returns no videos).

**Location:** `audiorag.core.exceptions.DiscoveryError`

```python
class DiscoveryError(AudioRAGError):
    def __init__(
        self, message: str, url: str | None = None
    ) -> None
    
    url: str | None
```

## Source Discovery

### discover_sources

Expand input URLs and paths into individual indexable sources.

**Location:** `audiorag.source.discovery.discover_sources`

```python
async def discover_sources(
    inputs: list[str],
    config: AudioRAGConfig | None = None
) -> list[str]
```

Automatically handles:
- **YouTube playlists/channels**: Expanded to individual video URLs
- **Local directories**: Recursively scanned for audio files
- **Local files**: Added directly
- **Direct URLs**: Passed through
- **Deduplication**: Removes duplicate sources

**Parameters:**
- `inputs`: List of URLs or file paths to expand
- `config`: Optional AudioRAGConfig for YouTubeSource configuration

**Returns:**
- List of expanded, unique source URLs/paths ready for indexing

**Supported Audio Formats:**
`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.wma`

**Example:**
```python
from audiorag.source import discover_sources
from audiorag import AudioRAGConfig

config = AudioRAGConfig()

inputs = [
    "https://youtube.com/playlist?list=...",  # Playlist
    "./podcasts/",                             # Directory
    "./single.mp3",                           # Single file
]

sources = await discover_sources(inputs, config)
# Returns: ["https://youtube.com/watch?v=video1", "https://youtube.com/watch?v=video2", "/abs/path/to/podcasts/ep1.mp3", ...]
```

**Error Handling:**
- Invalid paths are skipped
- YouTube expansion failures return the original URL
- Missing optional dependencies (yt-dlp) log warnings but don't fail

## Chunking

### chunk_transcription

Group transcription segments into time-based chunks.

**Location:** `audiorag.chunking.chunk_transcription`

```python
def chunk_transcription(
    segments: list[TranscriptionSegment],
    chunk_duration_seconds: int | float,
    source_url: str,
    title: str,
) -> list[ChunkMetadata]
```

**Parameters:**
- `segments`: List of transcription segments
- `chunk_duration_seconds`: Target duration for each chunk
- `source_url`: Source URL for all chunks
- `title`: Source title for all chunks

**Returns:**
- List of ChunkMetadata objects

## Provider Implementations

Providers are grouped by capability modules:

**STT Providers:**
- `audiorag.transcribe`
- `OpenAITranscriber`
- `DeepgramTranscriber`
- `AssemblyAITranscriber`
- `GroqTranscriber`

**Embedding Providers:**
- `audiorag.embed`
- `OpenAIEmbeddingProvider`
- `VoyageEmbeddingProvider`
- `CohereEmbeddingProvider`

**Vector Store Providers:**
- `audiorag.store`
- `ChromaDBVectorStore`
- `PineconeVectorStore`
- `WeaviateVectorStore`
- `SupabasePgVectorStore`

**Generation Providers:**
- `audiorag.generate`
- `OpenAIGenerator`
- `AnthropicGenerator`
- `GeminiGenerator`

**Reranker Providers:**
- `audiorag.rerank`
- `CohereReranker`
- `PassthroughReranker`

**Audio Source Providers:**
- `audiorag.source`
- `YouTubeSource`
- `LocalSource`
- `URLSource`

**Example:**
```python
from audiorag.source import YouTubeSource
from audiorag.transcribe import OpenAITranscriber

source = YouTubeSource()
audio_file = await source.download(url, output_dir)
```
