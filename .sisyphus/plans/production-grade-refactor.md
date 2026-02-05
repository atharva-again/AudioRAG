# Production-Grade Refactoring Plan

## Overview

**Goal:** Transform AudioRAG codebase to production-grade quality (target: 8.8/10 from current 5.8/10)

**Total Tasks:** 15 high-impact refactoring tasks
**Estimated Effort:** 2-3 weeks for one senior engineer
**Risk Level:** Medium (comprehensive test suite provides safety net)

## Success Criteria

- All tests pass (current baseline + new tests)
- Zero new LSP errors
- Code coverage >= 80% for all modules
- All critical issues resolved
- Structured logging implemented
- Retry logic on all external APIs
- Comprehensive error handling

---

## Phase 1: Foundation (Error Handling & Logging)

### Task 1.1: Create Structured Exception Hierarchy
**Priority:** CRITICAL  
**Scope:** New file + pipeline modifications  
**Files:** `src/audiorag/exceptions.py`, `src/audiorag/pipeline.py`

**Current Problem:**
- Generic `RuntimeError` and `Exception` used everywhere
- No way to catch specific error types
- Error context is lost

**Implementation:**
```python
# New file: src/audiorag/exceptions.py
class AudioRAGError(Exception):
    """Base exception for all AudioRAG errors."""
    pass

class PipelineError(AudioRAGError):
    """Pipeline execution errors."""
    def __init__(self, message: str, stage: str, source_url: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.source_url = source_url

class ProviderError(AudioRAGError):
    """External provider errors."""
    def __init__(self, message: str, provider: str, retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable

class ConfigurationError(AudioRAGError):
    """Configuration validation errors."""
    pass

class StateError(AudioRAGError):
    """Database/state management errors."""
    pass
```

**Verification:**
- [ ] Import exceptions in pipeline.py
- [ ] Replace RuntimeError with specific exceptions
- [ ] Verify all exception types have proper attributes

---

### Task 1.2: Implement Structured Logging
**Priority:** CRITICAL  
**Scope:** New module + integration  
**Files:** `src/audiorag/logging_config.py`, all provider files

**Current Problem:**
- Basic logging without context
- No request tracing
- No structured output for log aggregation

**Implementation:**
```python
# New file: src/audiorag/logging_config.py
import logging
import sys
from typing import Any

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Implementation details...
        pass

def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure structured logging for the application."""
    # Implementation...
    pass

# Add context managers for operation tracking
import contextvars
operation_id = contextvars.ContextVar('operation_id', default=None)

class OperationContext:
    """Context manager for tracking operations."""
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
```

**Verification:**
- [ ] Logs include operation_id, timestamps, context
- [ ] All providers use structured logging
- [ ] JSON format option works

---

### Task 1.3: Add Tenacity Retry Decorators
**Priority:** CRITICAL  
**Scope:** External API providers  
**Files:** All `providers/openai_*.py`, `providers/cohere_reranker.py`

**Current Problem:**
- No retry logic for transient failures
- 429/500 errors cause immediate failure
- No exponential backoff

**Implementation:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, TimeoutError)),
    before_sleep=log_retry_attempt,
)
async def transcribe(self, audio_path: Path, ...) -> list[TranscriptionSegment]:
    # Existing implementation
```

**Verification:**
- [ ] Add tenacity to pyproject.toml dependencies
- [ ] Apply retry decorators to all external API calls
- [ ] Test retry behavior with mocked failures

---

## Phase 2: Pipeline Architecture

### Task 2.1: Refactor Monolithic index() Method
**Priority:** CRITICAL  
**Scope:** `pipeline.py` refactor  
**Files:** `src/audiorag/pipeline.py`

**Current Problem:**
- 233-line method handling 6 stages
- Hard to test individual stages
- No stage isolation for error handling
- Sequential when some stages could be optimized

**Implementation:**
```python
class AudioRAGPipeline:
    # ... existing code ...
    
    async def index(self, url: str, *, force: bool = False) -> None:
        """Index audio from a URL through the full pipeline."""
        await self._ensure_initialized()
        
        operation_id = generate_operation_id()
        with OperationContext("index", url=url, operation_id=operation_id):
            try:
                # Stage 1: Download
                audio_file = await self._stage_download(url)
                
                # Stage 2: Split
                audio_parts = await self._stage_split(audio_file)
                
                # Stage 3: Transcribe
                segments = await self._stage_transcribe(audio_parts)
                
                # Stage 4: Chunk
                chunks = await self._stage_chunk(segments, url, audio_file.video_title)
                
                # Stage 5: Embed
                await self._stage_embed(chunks, url)
                
                # Stage 6: Complete
                await self._stage_complete(url)
                
            except PipelineError:
                raise
            except Exception as e:
                logger.exception("Pipeline failed", url=url)
                await self._handle_pipeline_failure(url, e)
                raise PipelineError(str(e), stage=self._current_stage, source_url=url) from e
    
    async def _stage_download(self, url: str) -> AudioFile:
        """Stage 1: Download audio from source."""
        logger.info("stage.start", stage="download", url=url)
        await self._state.upsert_source(url, IndexingStatus.DOWNLOADING)
        
        try:
            audio_file = await self._audio_source.download(...)
            await self._state.upsert_source(url, IndexingStatus.DOWNLOADED, ...)
            logger.info("stage.complete", stage="download")
            return audio_file
        except Exception as e:
            logger.error("stage.failed", stage="download", error=str(e))
            raise
    
    # ... similar for _stage_split, _stage_transcribe, etc.
```

**Verification:**
- [ ] Each stage is a separate async method
- [ ] Each stage has proper error handling
- [ ] Stage transitions are logged
- [ ] All existing tests pass

---

### Task 2.2: Fix Temp Directory Race Condition
**Priority:** HIGH  
**Scope:** Pipeline cleanup  
**Files:** `src/audiorag/pipeline.py` (lines 312-318)

**Current Problem:**
- Cleanup happens after await operations
- Potential for directory to be removed while in use
- No guaranteed cleanup on exception

**Implementation:**
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def _managed_work_dir(self) -> AsyncGenerator[Path, None]:
    """Context manager for work directory with guaranteed cleanup."""
    work_dir: Path
    created_temp = False
    
    if self._config.work_dir:
        work_dir = self._config.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="audiorag_"))
        created_temp = True
    
    try:
        yield work_dir
    finally:
        if created_temp and self._config.cleanup_audio:
            try:
                await asyncio.to_thread(shutil.rmtree, work_dir, ignore_errors=True)
                logger.debug("work_dir.cleaned", path=str(work_dir))
            except Exception as e:
                logger.warning("work_dir.cleanup_failed", path=str(work_dir), error=str(e))

# Usage in index():
async with self._managed_work_dir() as work_dir:
    # Pipeline stages here
```

**Verification:**
- [ ] Cleanup runs even if exception occurs
- [ ] Uses asyncio.to_thread for blocking shutil.rmtree
- [ ] Logs cleanup success/failure

---

## Phase 3: Provider Improvements

### Task 3.1: Add Retry Logic to All External Providers
**Priority:** CRITICAL  
**Scope:** All provider implementations  
**Files:** `providers/openai_*.py`, `providers/cohere_reranker.py`

**Implementation:**
Add consistent retry logic to:
1. OpenAISTTProvider.transcribe()
2. OpenAIEmbeddingProvider.embed()
3. OpenAIGenerationProvider.generate()
4. CohereReranker.rerank()

Each with appropriate:
- Retry attempts: 3
- Exponential backoff: 4s min, 10s max
- Retry on: RateLimitError, TimeoutError, 5xx errors
- No retry on: Auth errors, 4xx client errors

**Verification:**
- [ ] All providers have retry decorators
- [ ] Retry behavior tested with mocks
- [ ] Logs show retry attempts

---

### Task 3.2: Improve ChromaDB Async Handling
**Priority:** MEDIUM  
**Scope:** ChromaDB vector store  
**Files:** `providers/chromadb_store.py`

**Current Problem:**
- Sync ChromaDB operations wrapped in asyncio.to_thread
- Could be improved with better batching

**Implementation:**
```python
# Add batching support for better performance
async def add_batch(
    self,
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    documents: list[str],
    batch_size: int = 100,
) -> None:
    """Add embeddings in batches for better performance."""
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]
        
        await self.add(batch_ids, batch_embeddings, batch_metadatas, batch_documents)
```

**Verification:**
- [ ] Batch operations work correctly
- [ ] Performance improvement verified
- [ ] All existing tests pass

---

### Task 3.3: Add Provider Health Checks
**Priority:** MEDIUM  
**Scope:** All providers  
**Files:** All provider files

**Implementation:**
Add health check method to each provider:
```python
async def health_check(self) -> dict[str, Any]:
    """Check provider health and connectivity.
    
    Returns:
        dict with status, latency, and error info
    """
    start_time = time.time()
    try:
        # Provider-specific health check
        await self._perform_health_check()
        return {
            "status": "healthy",
            "latency_ms": (time.time() - start_time) * 1000,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "latency_ms": (time.time() - start_time) * 1000,
        }
```

**Verification:**
- [ ] All providers implement health_check()
- [ ] Health checks test actual connectivity
- [ ] Returns structured response

---

## Phase 4: Configuration & Validation

### Task 4.1: Add Configuration Validation
**Priority:** HIGH  
**Scope:** Config model enhancements  
**Files:** `src/audiorag/config.py`

**Current Problem:**
- No validation of API keys (empty strings allowed)
- Negative values allowed for positive-only fields
- Invalid paths accepted

**Implementation:**
```python
from pydantic import Field, validator, ValidationError

class AudioRAGConfig(BaseSettings):
    # ... existing fields ...
    
    @validator('chunk_duration_seconds')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('chunk_duration_seconds must be positive')
        return v
    
    @validator('openai_api_key')
    def validate_api_key_format(cls, v):
        if v and not v.startswith('sk-'):
            raise ValueError('OpenAI API key must start with "sk-"')
        return v
    
    def validate_required_for_operation(self) -> None:
        """Validate configuration is ready for pipeline execution."""
        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required")
```

**Verification:**
- [ ] Invalid configs raise ConfigurationError
- [ ] Validation happens at startup
- [ ] Error messages are clear and actionable

---

### Task 4.2: Refactor Lazy Imports
**Priority:** MEDIUM  
**Scope:** Provider initialization  
**Files:** `src/audiorag/pipeline.py`, `providers/__init__.py`

**Current Problem:**
- __getattr__ prevents static analysis
- IDE can't autocomplete
- Type checkers can't verify

**Implementation:**
Replace dynamic __getattr__ with explicit conditional imports:
```python
# In pipeline.py __init__
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from cohere import AsyncClientV2

# Explicit imports with fallback
self._stt: STTProvider
if stt is not None:
    self._stt = stt
else:
    try:
        from audiorag.providers.openai_stt import OpenAISTTProvider
        self._stt = OpenAISTTProvider(api_key=config.openai_api_key)
    except ImportError as e:
        raise ConfigurationError(
            "OpenAI provider requested but 'openai' package not installed. "
            "Install with: uv sync --extra openai"
        ) from e
```

**Verification:**
- [ ] No dynamic __getattr__ usage
- [ ] TYPE_CHECKING imports for type hints
- [ ] Clear error messages for missing deps
- [ ] Static analysis works correctly

---

## Phase 5: Testing Improvements

### Task 5.1: Add Provider Unit Tests
**Priority:** CRITICAL  
**Scope:** New test files  
**Files:** `tests/providers/test_openai_stt.py`, `tests/providers/test_openai_embeddings.py`, etc.

**Current Problem:**
- All provider implementations untested
- No error scenario coverage
- No retry behavior testing

**Implementation:**
Create comprehensive tests for each provider:
```python
# tests/providers/test_openai_stt.py
class TestOpenAISTTProvider:
    """Unit tests for OpenAI STT provider."""
    
    @pytest.mark.asyncio
    async def test_transcribe_success(self, mocker):
        """Test successful transcription."""
        # Mock AsyncOpenAI
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.segments = [...]
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        
        provider = OpenAISTTProvider(api_key="test-key")
        provider.client = mock_client
        
        result = await provider.transcribe(Path("test.mp3"))
        
        assert len(result) == 2
        assert result[0].text == "Hello world"
    
    @pytest.mark.asyncio
    async def test_transcribe_retry_on_rate_limit(self, mocker):
        """Test retry behavior on rate limit."""
        # Test tenacity retry logic
        pass
    
    @pytest.mark.asyncio
    async def test_transcribe_handles_api_error(self, mocker):
        """Test error handling for API failures."""
        pass
```

**Verification:**
- [ ] Tests for all provider methods
- [ ] Error scenario coverage
- [ ] Retry behavior tested
- [ ] Mock usage is appropriate

---

### Task 5.2: Add Integration Tests for Error Paths
**Priority:** HIGH  
**Scope:** Pipeline error testing  
**Files:** `tests/test_pipeline_errors.py`

**Implementation:**
```python
# tests/test_pipeline_errors.py
class TestPipelineErrorHandling:
    """Integration tests for pipeline error scenarios."""
    
    @pytest.mark.asyncio
    async def test_download_failure_handling(self):
        """Test pipeline handles download failures gracefully."""
        pass
    
    @pytest.mark.asyncio
    async def test_transcription_failure_recovery(self):
        """Test state is updated on transcription failure."""
        pass
    
    @pytest.mark.asyncio
    async def test_partial_failure_cleanup(self):
        """Test cleanup after partial pipeline failure."""
        pass
```

**Verification:**
- [ ] Tests cover all pipeline stages
- [ ] State is correctly updated on failures
- [ ] Cleanup happens correctly

---

### Task 5.3: Add Property-Based Tests
**Priority:** MEDIUM  
**Scope:** Core algorithms  
**Files:** `tests/test_chunking_properties.py`

**Implementation:**
```python
# tests/test_chunking_properties.py
from hypothesis import given, strategies as st

class TestChunkingProperties:
    """Property-based tests for chunking logic."""
    
    @given(
        segments=st.lists(st.builds(TranscriptionSegment, ...), min_size=1),
        duration=st.integers(min_value=1, max_value=600),
    )
    def test_chunk_duration_respected(self, segments, duration):
        """No chunk should exceed target duration (with tolerance)."""
        chunks = chunk_transcription(segments, duration, "url", "title")
        for chunk in chunks:
            assert chunk.end_time - chunk.start_time <= duration * 1.1
    
    @given(segments=st.lists(...))
    def test_all_text_preserved(self, segments):
        """All input text appears in output chunks."""
        pass
```

**Verification:**
- [ ] Hypothesis tests pass
- [ ] Edge cases discovered and fixed

---

## Phase 6: State Management Improvements

### Task 6.1: Improve StateManager Connection Handling
**Priority:** HIGH  
**Scope:** State management  
**Files:** `src/audiorag/state.py`

**Current Problem:**
- No connection pooling
- Connection leak potential
- Generic error handling

**Implementation:**
```python
class StateManager:
    """Improved state management with better connection handling."""
    
    def __init__(self, db_path: str | Path, pool_size: int = 5):
        self.db_path = Path(db_path)
        self._pool_size = pool_size
        self._connection_pool: asyncio.Queue[aiosqlite.Connection] | None = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize with connection pooling."""
        # Create connection pool
        pass
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get connection from pool."""
        pass
    
    async def _release_connection(self, conn: aiosqlite.Connection) -> None:
        """Return connection to pool."""
        pass
```

**Verification:**
- [ ] Connection pooling works
- [ ] No connection leaks
- [ ] Concurrent operations safe

---

### Task 6.2: Add Database Migration System
**Priority:** MEDIUM  
**Scope:** Schema versioning  
**Files:** `src/audiorag/migrations.py`

**Implementation:**
```python
# New file: src/audiorag/migrations.py
class MigrationManager:
    """Manage database schema migrations."""
    
    MIGRATIONS: list[Migration] = [
        Migration(1, "Initial schema", initial_schema),
        Migration(2, "Add indices", add_indices),
    ]
    
    async def migrate(self) -> None:
        """Run pending migrations."""
        pass
```

**Verification:**
- [ ] Migrations run automatically
- [ ] Schema version tracked
- [ ] Rollback possible

---

## Execution Order

### Week 1: Critical Foundation
1. Task 1.1: Exception Hierarchy
2. Task 1.2: Structured Logging
3. Task 1.3: Tenacity Retry Decorators
4. Task 2.1: Refactor index() Method
5. Task 2.2: Fix Temp Directory Race

### Week 2: Provider & Config Improvements
6. Task 3.1: Provider Retry Logic
7. Task 3.2: ChromaDB Improvements
8. Task 3.3: Health Checks
9. Task 4.1: Config Validation
10. Task 4.2: Refactor Lazy Imports

### Week 3: Testing & State Management
11. Task 5.1: Provider Unit Tests
12. Task 5.2: Integration Error Tests
13. Task 5.3: Property-Based Tests
14. Task 6.1: Connection Handling
15. Task 6.2: Database Migrations

## Rollback Strategy

Each task:
- Is independent and reversible
- Has clear before/after states
- Includes comprehensive tests
- Can be feature-flagged if needed

**Emergency Rollback:**
```bash
git revert <commit-hash>
# Or for specific changes:
git checkout HEAD -- path/to/file
```

## Verification Checklist

Before marking complete:
- [ ] All existing tests pass
- [ ] New tests added and passing
- [ ] Coverage >= 80% for modified modules
- [ ] No new LSP errors
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

## Ready to Execute

This plan provides a comprehensive roadmap to production-grade quality. Each task is:
- **Specific** - Clear implementation details
- **Measurable** - Verification criteria defined
- **Achievable** - Within scope of existing architecture
- **Relevant** - Addresses production readiness gaps
- **Time-bound** - 3-week execution timeline

Proceed with Phase 5: Execute Refactoring
