# Architecture Overview

AudioRAG uses a modular, provider-agnostic architecture built around protocols and pipelines.

## High-Level Architecture

```
User Query
    |
    v
+-------------------+     +-------------------+     +-------------------+
|   Query Pipeline  |---->|  Vector Search    |---->|    Reranking      |
|                   |     |                   |     |                   |
| - Embed query     |     | - Similarity      |     | - Relevance       |
| - Search index    |     |   search          |     |   scoring         |
| - Generate answer |     | - Retrieve top-k  |     | - Sort results    |
+-------------------+     +-------------------+     +-------------------+
                                                               |
                                                               v
+-------------------+     +-------------------+     +-------------------+
|  AudioRAGPipeline |<----|   LLM Generation  |<----|  Context Assembly |
|                   |     |                   |     |                   |
| Orchestrates      |     | - Answer synthesis|     | - Build prompt    |
| all stages        |     | - Source citation |     | - Format sources  |
+-------------------+     +-------------------+     +-------------------+
          |
          | Index Pipeline
          v
+-------------------+     +-------------------+     +-------------------+
|     Ingest        |---->|      Split        |---->|   Transcribe      |
|                   |     |                   |     |                   |
| - Local files     |     | - Large files     |     | - STT provider    |
| - Duration detect |     | - ffmpeg          |     | - Whisper/Deepgram|
+-------------------+     +-------------------+     +-------------------+
                                                               |
                                                               v
+-------------------+     +-------------------+     +-------------------+
|     Complete      |<----|      Embed        |<----|      Chunk        |
|                   |     |                   |     |                   |
| - Mark done       |     | - Embedding       |     | - Time-based      |
| - Cleanup         |     |   provider        |     | - Segment groups  |
|                   |     | - Vector store    |     |                   |
+-------------------+     +-------------------+     +-------------------+
```

## Core Components

### 1. AudioRAGPipeline

The main orchestrator that coordinates all pipeline stages. Located in `src/audiorag/pipeline.py`.

**Responsibilities:**
- Initialize providers based on configuration
- Orchestrate index and query flows
- Manage state persistence
- Handle errors and cleanup

**Key Methods:**
- `index(url, force=False)`: Index audio from URL
- `query(query)`: Query indexed content

### 2. Providers

Provider implementations abstract external services. Each provider implements a protocol interface.

**Location:** `src/audiorag/` (grouped by capability: `embed/`, `generate/`, `transcribe/`, `store/`, `source/`, `rerank/`)

**Design Pattern:**
- Protocol-based interfaces
- Lazy loading via `__getattr__`
- Retry decorators from `retry_config.py`
- Optional dependency imports inside methods

### 3. Protocols

Protocol definitions establish contracts that providers must implement.

**Location:** `src/audiorag/core/protocols/`

**Protocols:**
- `STTProvider`: `transcribe(audio_path, language) -> list[TranscriptionSegment]`
- `EmbeddingProvider`: `embed(texts) -> list[list[float]]`
- `VectorStoreProvider`: `add()`, `query()`, `delete_by_source()`
- `VerifiableVectorStoreProvider`: `verify(ids) -> bool`
- `GenerationProvider`: `generate(query, context) -> str`
- `RerankerProvider`: `rerank(query, documents, top_n) -> list[tuple[int, float]]`
- `AudioSourceProvider`: `download(url, output_dir, format) -> AudioFile`

### 4. State Management

SQLite-based state tracking with WAL mode for concurrency.

**Location:** `src/audiorag/core/state.py`

**Tables:**
- `sources`: URL, status, metadata, timestamps
- `chunks`: chunk_id, source_id, timing, text, embedding

**Features:**
- SHA-256 hash IDs
- ISO 8601 timestamps
- Foreign key constraints
- Async context manager support

State IDs are canonical and deterministic:
- `source_id`: SHA-256 of source URL
- `chunk_id`: SHA-256 of `source_id:chunk_index`

Before vector store writes, pipeline can adapt canonical `chunk_id` values using
provider-aware ID strategy (`auto`, `sha256`, `uuid5`).

### 5. Models

Pydantic models for data validation and serialization.

**Location:** `src/audiorag/models.py`

**Key Models:**
- `ChunkMetadata`: Time-based chunk with text
- `Source`: Search result with relevance and timing
- `QueryResult`: Answer with source list
- `AudioFile`: Downloaded audio metadata
- `TranscriptionSegment`: STT output segment
- `IndexingStatus`: Pipeline stage enum

### 6. Configuration

Pydantic-settings based configuration with environment variable support.

**Location:** `src/audiorag/core/config.py`

**Features:**
- `AUDIORAG_` prefix for env vars
- `.env` file support
- Model selection helpers
- Provider-specific settings

## Pipeline Stages

### Index Pipeline

1. **Ingest**
   - Load audio from local files
   - Extract duration using ffprobe
   - No external downloads required

 2. **Split**
   - Check file size against threshold
   - Split large files using ffmpeg
   - Maintain sequential ordering
   - Track cumulative time offsets

3. **Transcribe**
   - Convert audio to text segments
   - Adjust timestamps for split parts
   - Use configured STT provider
   - Aggregate all segments

4. **Chunk**
   - Group segments by duration
    - Configurable chunk size (default: 30s)
   - Filter empty chunks
   - Preserve metadata

5. **Embed**
   - Generate vector embeddings
   - Batch processing for efficiency
   - Store embeddings in vector DB
   - Verify vector writes (mode-dependent)

6. **Complete**
     - Mark source as completed
     - Cleanup temporary files (optional)
     - Log completion metrics

### Reliability Controls

- **Budget governor**: Optional fail-fast limits for `rpm`, `tpm`, and `audio_seconds_per_hour`.
- **Preflight STT reservation**: If audio duration is known, reserve full audio budget before transcription starts.
- **Persistent accounting**: Budget events are persisted in SQLite for restart/process safety.
- **Vector write verification**: post-`add()` verification with `off`, `best_effort`, or `strict` mode.

### Stage Execution Model

`AudioRAGPipeline.index()` executes an ordered list of stage objects. Each stage
receives a shared mutable context (`StageContext`) containing intermediate
artifacts (audio file, parts, segments, chunks) and is run by a stage runner
that wraps failures as `PipelineError` with stage context.

### Concurrency Guard

Indexing uses a per-URL `asyncio.Lock` to prevent concurrent indexing of the
same URL within a process. A DB-level guard also skips URLs with in-progress
statuses to reduce multi-process collisions unless `force=True` is used.

### Query Pipeline

1. **Embed Query**
   - Convert query to embedding
   - Use same embedding provider as indexing

2. **Retrieve**
   - Search vector store
   - Retrieve top-k matches
   - Return documents with metadata

3. **Rerank**
   - Score relevance of retrieved documents
   - Sort by relevance score
   - Keep top-n results

4. **Generate**
   - Build context from sources
   - Send to LLM for answer synthesis
   - Include source citations

5. **Return**
    - Answer text
    - Source list with timestamps
    - Relevance scores
    - Source URLs and metadata

## Data Flow

### Indexing Flow

```
File Path -> LocalSource -> AudioFile
                            |
                            v
                    AudioSplitter (if needed)
                            |
                            v
                    STTProvider.transcribe()
                            |
                            v
                    list[TranscriptionSegment]
                            |
                            v
                    chunk_transcription()
                            |
                            v
                    list[ChunkMetadata]
                            |
                            +------> StateManager.store_chunks()
                            |
                            v
                    EmbeddingProvider.embed()
                            |
                            v
                    list[embeddings]
                            |
                            +------> StateManager (canonical chunk IDs)
                            |
                            +------> Provider-aware ID strategy
                            |
                            v
                    VectorStoreProvider.add()
```

### Query Flow

```
Query String
     |
     v
EmbeddingProvider.embed([query])
     |
     v
Query Embedding
     |
     v
VectorStoreProvider.query(embedding, top_k)
     |
     v
Raw Results (documents + metadata)
     |
     v
RerankerProvider.rerank(query, documents, top_n)
     |
     v
Reranked Indices + Scores
     |
     v
GenerationProvider.generate(query, context)
     |
     v
Answer String
     |
     +------> Build QueryResult
                |
                v
         QueryResult(answer, sources)
```

## State Management

### SQLite Schema

```sql
-- Sources table
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,      -- SHA-256 hash of URL
    source_path TEXT NOT NULL UNIQUE,-- Original URL
    status TEXT NOT NULL,            -- IndexingStatus enum
    metadata TEXT,                   -- JSON metadata
    created_at TEXT NOT NULL,        -- ISO 8601 UTC
    updated_at TEXT NOT NULL         -- ISO 8601 UTC
);

-- Chunks table
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,       -- SHA-256 hash of source_id:index
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,                  -- Optional embedding storage
    metadata TEXT,                   -- JSON metadata
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
        ON DELETE CASCADE,
    UNIQUE(source_id, chunk_index)
);

-- Indexes
CREATE INDEX idx_chunks_source_id ON chunks(source_id);
CREATE INDEX idx_sources_status ON sources(status);
```

### Status Transitions

```
PENDING -> DOWNLOADING -> DOWNLOADED
                              |
                              v
                    SPLITTING -> TRANSCRIBING -> TRANSCRIBED
                                                            |
                                                            v
                                        CHUNKING -> CHUNKED -> EMBEDDING
                                                                            |
                                                                            v
                                                                EMBEDDED -> COMPLETED
                                                                            |
                                                                    (or FAILED)
```

## Error Handling

### Exception Hierarchy

```
AudioRAGError (base)
    |
    +-- PipelineError (stage, source_url)
    |
    +-- ProviderError (provider, retryable)
    |
    +-- ConfigurationError (configuration issues)
    |
    +-- StateError (database/state issues)
```

### Retry Logic

All external API calls use tenacity with:
- 3 attempts by default
- Exponential backoff: 4s to 60s
- Configurable via `RetryConfig`

```python
@retry(
    stop=stop_after_attempt(config.max_attempts),
    wait=wait_exponential(
        multiplier=config.exponential_multiplier,
        min=config.min_wait_seconds,
        max=config.max_wait_seconds,
    ),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
async def api_call():
    ...
```

## Logging

Structured logging with structlog:
- Context binding per operation
- Stage timing with Timer class
- JSON-compatible output
- Configurable levels and formats

Example log output:
```
2024-01-15T10:30:00Z [INFO] index_started url=https://... operation=index
2024-01-15T10:30:05Z [INFO] stage_download_completed duration_ms=5000.0 title=... duration_seconds=600.0
2024-01-15T10:30:10Z [INFO] index_completed url=https://... operation=index
```

## Design Principles

1. **Provider Agnostic**: All external services behind protocols
2. **Resumable**: State tracking enables restart at any stage
3. **Type Safe**: Full type annotations with modern Python syntax
4. **Async First**: All I/O operations are async
5. **Lazy Loading**: Optional dependencies imported only when needed
6. **Configurable**: Environment variables and .env file support
7. **Testable**: Protocol interfaces enable easy mocking
