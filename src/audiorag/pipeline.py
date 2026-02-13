"""Main AudioRAG pipeline orchestrator.

The index method is decomposed into discrete Stage classes executed by a
stage-runner loop.  A per-URL ``asyncio.Lock`` prevents the same URL from
being indexed concurrently within a single process, and an in-progress DB
status guard protects against multi-process races.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from audiorag.chunking import chunk_transcription
from audiorag.core import (
    AudioRAGConfig,
    AudioSourceProvider,
    BatchIndexFailure,
    BatchIndexResult,
    BudgetGovernor,
    EmbeddingProvider,
    GenerationProvider,
    IndexingStatus,
    PipelineError,
    QueryResult,
    RerankerProvider,
    RetryConfig,
    Source,
    StateManager,
    STTProvider,
    VectorStoreProvider,
    VerifiableVectorStoreProvider,
    configure_logging,
    get_logger,
)
from audiorag.core.exceptions import BudgetExceededError, StateError
from audiorag.core.id_strategy import (
    resolve_vector_id_format,
    resolve_vector_id_strategy,
    to_provider_vector_ids,
)
from audiorag.core.logging_config import Timer
from audiorag.core.provider_factory import (
    create_embedding_provider,
    create_generation_provider,
    create_reranker_provider,
    create_stt_provider,
    create_vector_store_provider,
)
from audiorag.source.discovery import discover_sources

if TYPE_CHECKING:
    import structlog

    from audiorag.core.models import AudioFile, ChunkMetadata, TranscriptionSegment

# YouTubeSource imported lazily to avoid yt_dlp dependency at module load time

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Statuses that indicate an indexing run is currently in-flight.
# Used by the DB-level guard to prevent multi-process collisions.
# ---------------------------------------------------------------------------
_IN_PROGRESS_STATUSES: frozenset[str] = frozenset(
    {
        IndexingStatus.DOWNLOADING,
        IndexingStatus.DOWNLOADED,
        IndexingStatus.SPLITTING,
        IndexingStatus.TRANSCRIBING,
        IndexingStatus.TRANSCRIBED,
        IndexingStatus.CHUNKING,
        IndexingStatus.CHUNKED,
        IndexingStatus.EMBEDDING,
    }
)


# ---------------------------------------------------------------------------
# Stage context - mutable bag of data passed through the stage pipeline
# ---------------------------------------------------------------------------
@dataclass
class StageContext:
    """Mutable context shared across all stages of a single indexing run."""

    url: str
    config: AudioRAGConfig
    logger: structlog.stdlib.BoundLogger

    # Populated during execution
    work_dir: Path = field(default_factory=lambda: Path("."))
    created_temp_dir: bool = False
    audio_file: AudioFile | None = None
    audio_parts: list[Path] = field(default_factory=list)
    all_segments: list[TranscriptionSegment] = field(default_factory=list)
    chunks: list[ChunkMetadata] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    reserved_audio_seconds: int = 0


# ---------------------------------------------------------------------------
# Stage base class
# ---------------------------------------------------------------------------
class Stage(ABC):
    """Abstract pipeline stage."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, logging-friendly stage name (e.g. ``'download'``)."""

    @abstractmethod
    async def execute(self, ctx: StageContext, pipeline: AudioRAGPipeline) -> None:
        """Run the stage, mutating *ctx* in place.

        Args:
            ctx: Shared mutable context for this indexing run.
            pipeline: The pipeline instance (provides providers & state).
        """


# ---------------------------------------------------------------------------
# Concrete stages
# ---------------------------------------------------------------------------
class DownloadStage(Stage):
    """Stage 1 - Download audio from the source URL."""

    @property
    def name(self) -> str:
        return "download"

    async def execute(self, ctx: StageContext, pipeline: AudioRAGPipeline) -> None:
        with Timer(ctx.logger, "stage_download") as timer:
            await pipeline._state.upsert_source(ctx.url, IndexingStatus.DOWNLOADING)

            # Pre-download budget check using metadata (duration)
            metadata = None
            if "youtube.com" in ctx.url or "youtu.be" in ctx.url:
                ctx.logger.debug("pre_download_metadata_extraction")
                try:
                    metadata = await pipeline._audio_source.get_metadata(ctx.url)
                    if metadata.duration and metadata.duration > 0:
                        seconds = int(metadata.duration)
                        ctx.logger.debug(
                            "pre_download_budget_reservation", duration_seconds=seconds
                        )
                        pipeline._budget_governor.reserve(
                            provider=ctx.config.stt_provider,
                            audio_seconds=seconds,
                        )
                        ctx.reserved_audio_seconds = seconds
                except Exception as e:
                    ctx.logger.warning("pre_download_metadata_failed", error=str(e))
                    # Fallback to standard download if metadata fetch fails

            try:
                audio_file = await pipeline._audio_source.download(
                    ctx.url, ctx.work_dir, ctx.config.audio_format, metadata=metadata
                )
            except Exception:
                # Release pre-flight reservation if download fails
                if ctx.reserved_audio_seconds > 0:
                    pipeline._budget_governor.release(
                        provider=ctx.config.stt_provider,
                        audio_seconds=ctx.reserved_audio_seconds,
                    )
                    ctx.reserved_audio_seconds = 0
                raise

            await pipeline._state.upsert_source(
                ctx.url,
                IndexingStatus.DOWNLOADED,
                metadata={
                    "title": audio_file.title,
                    "duration": audio_file.duration,
                },
            )

            # Reconcile actual duration vs reserved duration
            if audio_file.duration and audio_file.duration > 0:
                actual = int(audio_file.duration)
                if ctx.reserved_audio_seconds == 0:
                    # No pre-flight reservation, reserve full amount
                    pipeline._budget_governor.reserve(
                        provider=ctx.config.stt_provider,
                        audio_seconds=actual,
                    )
                    ctx.reserved_audio_seconds = actual
                elif actual != ctx.reserved_audio_seconds:
                    # Reconcile delta
                    delta = actual - ctx.reserved_audio_seconds
                    ctx.logger.debug("reconciling_duration_delta", delta_seconds=delta)
                    if delta > 0:
                        pipeline._budget_governor.reserve(
                            provider=ctx.config.stt_provider,
                            audio_seconds=delta,
                        )
                    else:
                        pipeline._budget_governor.release(
                            provider=ctx.config.stt_provider,
                            audio_seconds=abs(delta),
                        )
                    ctx.reserved_audio_seconds = actual

            ctx.audio_file = audio_file
            timer.complete(
                title=audio_file.title,
                duration_seconds=audio_file.duration,
                file_path=str(audio_file.path),
            )


class SplitStage(Stage):
    """Stage 2 - Split large audio files into manageable parts."""

    @property
    def name(self) -> str:
        return "split"

    async def execute(self, ctx: StageContext, pipeline: AudioRAGPipeline) -> None:
        assert ctx.audio_file is not None
        with Timer(ctx.logger, "stage_split") as timer:
            await pipeline._state.update_source_status(ctx.url, IndexingStatus.SPLITTING)
            ctx.audio_parts = await pipeline._splitter.split_if_needed(
                ctx.audio_file.path, ctx.work_dir
            )
            timer.complete(parts_count=len(ctx.audio_parts))


class TranscribeStage(Stage):
    """Stage 3 - Transcribe audio parts into text segments."""

    @property
    def name(self) -> str:
        return "transcribe"

    async def execute(self, ctx: StageContext, pipeline: AudioRAGPipeline) -> None:
        with Timer(ctx.logger, "stage_transcribe", parts=len(ctx.audio_parts)) as timer:
            await pipeline._state.update_source_status(ctx.url, IndexingStatus.TRANSCRIBING)
            all_segments: list[Any] = []
            cumulative_offset = 0.0
            estimated_part_seconds = 0
            if (
                ctx.reserved_audio_seconds == 0
                and ctx.audio_parts
                and ctx.audio_file is not None
                and ctx.audio_file.duration
            ):
                estimated_part_seconds = int(ctx.audio_file.duration / len(ctx.audio_parts))

            for part_idx, part_path in enumerate(ctx.audio_parts):
                part_logger = ctx.logger.bind(part_index=part_idx, part_path=str(part_path))
                part_logger.debug("transcribing_part")
                pipeline._budget_governor.reserve(
                    provider=ctx.config.stt_provider,
                    requests=1,
                    audio_seconds=estimated_part_seconds,
                )

                segments = await pipeline._stt.transcribe(part_path, ctx.config.stt_language)
                # Adjust timestamps for subsequent parts
                if cumulative_offset > 0:
                    from audiorag.core.models import TranscriptionSegment

                    segments = [
                        TranscriptionSegment(
                            start_time=s.start_time + cumulative_offset,
                            end_time=s.end_time + cumulative_offset,
                            text=s.text,
                        )
                        for s in segments
                    ]
                all_segments.extend(segments)

                # Update offset: use the last segment's end time from this part
                if segments:
                    cumulative_offset = segments[-1].end_time

            await pipeline._state.update_source_status(ctx.url, IndexingStatus.TRANSCRIBED)
            ctx.all_segments = all_segments
            timer.complete(segments_count=len(all_segments))


class ChunkStage(Stage):
    """Stage 4 - Chunk transcription segments into time-based groups."""

    @property
    def name(self) -> str:
        return "chunk"

    async def execute(self, ctx: StageContext, pipeline: AudioRAGPipeline) -> None:
        assert ctx.audio_file is not None
        with Timer(ctx.logger, "stage_chunk") as timer:
            await pipeline._state.update_source_status(ctx.url, IndexingStatus.CHUNKING)
            chunks = chunk_transcription(
                ctx.all_segments,
                ctx.config.chunk_duration_seconds,
                ctx.url,
                ctx.audio_file.title,
            )

            if not chunks:
                ctx.logger.warning("no_chunks_produced")
                await pipeline._state.update_source_status(ctx.url, IndexingStatus.COMPLETED)
                return

            # Store chunks in SQLite
            chunk_dicts = [
                {
                    "chunk_index": i,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "text": c.text,
                    "metadata": {
                        "source_url": c.source_url,
                        "title": c.title,
                    },
                }
                for i, c in enumerate(chunks)
            ]
            ctx.chunk_ids = await pipeline._state.store_chunks(ctx.url, chunk_dicts)
            ctx.chunks = chunks
            await pipeline._state.update_source_status(ctx.url, IndexingStatus.CHUNKED)
            timer.complete(chunks_count=len(chunks))


class EmbedStage(Stage):
    """Stage 5 - Embed chunk texts and store in vector store."""

    @property
    def name(self) -> str:
        return "embed"

    async def execute(self, ctx: StageContext, pipeline: AudioRAGPipeline) -> None:
        if not ctx.chunks:
            # Nothing to embed (e.g. ChunkStage short-circuited)
            return

        with Timer(ctx.logger, "stage_embed") as timer:
            await pipeline._state.update_source_status(ctx.url, IndexingStatus.EMBEDDING)
            texts = [c.text for c in ctx.chunks]
            pipeline._budget_governor.reserve(
                provider=ctx.config.embedding_provider,
                requests=1,
                text_chars=sum(len(text) for text in texts),
            )
            embeddings = await pipeline._embedder.embed(texts)

            # Store in vector store
            metadatas = [
                {
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "source_url": c.source_url,
                    "title": c.title,
                }
                for c in ctx.chunks
            ]
            documents = [c.text for c in ctx.chunks]
            effective_vector_id_format = resolve_vector_id_format(
                provider_name=ctx.config.vector_store_provider,
                configured_format=ctx.config.vector_id_format,
                vector_store=pipeline._vector_store,
            )
            vector_id_strategy = resolve_vector_id_strategy(
                provider_name=ctx.config.vector_store_provider,
                configured_format=ctx.config.vector_id_format,
                uuid5_namespace=ctx.config.vector_id_uuid5_namespace,
                vector_store=pipeline._vector_store,
            )
            vector_ids = to_provider_vector_ids(
                canonical_ids=ctx.chunk_ids,
                provider_name=ctx.config.vector_store_provider,
                configured_format=ctx.config.vector_id_format,
                uuid5_namespace=ctx.config.vector_id_uuid5_namespace,
                vector_store=pipeline._vector_store,
            )
            await pipeline._vector_store.add(
                ids=vector_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            await pipeline._verify_vector_store_write(ctx.url, vector_ids)
            await pipeline._state.update_source_status(
                ctx.url,
                IndexingStatus.EMBEDDED,
                metadata={
                    "vector_id_format_effective": effective_vector_id_format,
                    "vector_id_strategy": vector_id_strategy,
                },
            )
            timer.complete(chunks_count=len(ctx.chunks))


class CompleteStage(Stage):
    """Stage 6 - Mark indexing as completed."""

    @property
    def name(self) -> str:
        return "complete"

    async def execute(self, ctx: StageContext, pipeline: AudioRAGPipeline) -> None:
        await pipeline._state.update_source_status(ctx.url, IndexingStatus.COMPLETED)
        ctx.logger.info("index_completed")


# ---------------------------------------------------------------------------
# Default stage ordering
# ---------------------------------------------------------------------------
_DEFAULT_STAGES: tuple[Stage, ...] = (
    DownloadStage(),
    SplitStage(),
    TranscribeStage(),
    ChunkStage(),
    EmbedStage(),
    CompleteStage(),
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class AudioRAGPipeline:
    """Orchestrates the full RAG pipeline over audio.

    Handles download, split, transcribe, chunk, embed, and query stages
    with SQLite state tracking and provider-agnostic design.
    """

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
    ) -> None:
        """Initialize the pipeline with config and optional provider overrides.

        Args:
            config: AudioRAG configuration.
            audio_source: Custom audio source provider. Defaults to YouTubeSource.
            stt: Custom STT provider. Defaults based on config.stt_provider.
            embedder: Custom embedding provider. Defaults based on config.embedding_provider.
            vector_store: Custom vector store provider. Defaults based on
                config.vector_store_provider.
            generator: Custom generation provider. Defaults based on config.generation_provider.
            reranker: Custom reranker provider. Defaults based on config.reranker_provider.
        """
        self._config = config

        # Configure logging based on config
        configure_logging(
            log_level=config.log_level,
            log_format=config.log_format,
            log_timestamps=config.log_timestamps,
        )

        # Create retry configuration from settings
        retry_config = RetryConfig(
            max_attempts=config.retry_max_attempts,
            min_wait_seconds=config.retry_min_wait_seconds,
            max_wait_seconds=config.retry_max_wait_seconds,
            exponential_multiplier=config.retry_exponential_multiplier,
        )

        # Initialize default providers lazily using config values
        if audio_source is not None:
            self._audio_source = audio_source
        else:
            from audiorag.source.youtube import YouTubeSource

            archive_path = (
                Path(config.youtube_download_archive) if config.youtube_download_archive else None
            )
            self._audio_source = YouTubeSource(
                download_archive=archive_path,
                concurrent_fragments=config.youtube_concurrent_fragments,
                skip_playlist_after_errors=config.youtube_skip_after_errors,
                cookie_file=config.youtube_cookie_file,
                po_token=config.youtube_po_token,
                visitor_data=config.youtube_visitor_data,
                data_sync_id=config.youtube_data_sync_id,
                impersonate_client=config.youtube_impersonate,
                player_clients=config.youtube_player_clients,
                js_runtime=config.js_runtime,
            )

        # Initialize STT provider based on config
        if stt is not None:
            self._stt = stt
        else:
            self._stt = create_stt_provider(config, retry_config)

        # Initialize embedder based on config
        if embedder is not None:
            self._embedder = embedder
        else:
            self._embedder = create_embedding_provider(config, retry_config)

        # Initialize vector store based on config
        if vector_store is not None:
            self._vector_store = vector_store
        else:
            self._vector_store = create_vector_store_provider(config, retry_config)

        # Initialize generator based on config
        if generator is not None:
            self._generator = generator
        else:
            self._generator = create_generation_provider(config, retry_config)

        # Initialize reranker based on config
        if reranker is not None:
            self._reranker = reranker
        else:
            self._reranker = create_reranker_provider(config, retry_config)

        # Internal utilities
        from audiorag.source.splitter import AudioSplitter

        self._splitter = AudioSplitter(max_size_mb=config.audio_split_max_size_mb)
        self._state = StateManager(config.database_path)
        self._budget_governor = BudgetGovernor.from_config(config)
        self._initialized = False

        # Per-URL asyncio locks for single-process concurrency guard
        self._url_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        """Initialize state manager if not already done."""
        if not self._initialized:
            await self._state.initialize()
            self._initialized = True

    async def close(self) -> None:
        """Close all resources and connections.

        Should be called when the pipeline is no longer needed to ensure
        proper cleanup of database connections and other resources.
        Safe to call multiple times (idempotent).
        """
        if self._initialized:
            await self._state.close()
            self._initialized = False

    async def __aenter__(self) -> AudioRAGPipeline:
        """Async context manager entry.

        Returns:
            The pipeline instance for use within the context.
        """
        await self._ensure_initialized()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit.

        Ensures resources are cleaned up regardless of whether an
        exception occurred.
        """
        await self.close()

    def _get_url_lock(self, url: str) -> asyncio.Lock:
        """Return (or create) the per-URL asyncio lock."""
        if url not in self._url_locks:
            self._url_locks[url] = asyncio.Lock()
        return self._url_locks[url]

    async def _verify_vector_store_write(self, source_url: str, ids: list[str]) -> None:
        mode = str(getattr(self._config, "vector_store_verify_mode", "best_effort")).lower()
        if mode == "off":
            return

        if not isinstance(self._vector_store, VerifiableVectorStoreProvider):
            if mode == "strict":
                raise PipelineError(
                    "Vector store verification required but provider does not support verify()",
                    stage="embed",
                    source_url=source_url,
                )
            logger.warning(
                "vector_store_verification_unavailable", provider=self._config.vector_store_provider
            )
            return

        max_attempts = max(1, int(getattr(self._config, "vector_store_verify_max_attempts", 5)))
        wait_seconds = max(
            0.0, float(getattr(self._config, "vector_store_verify_wait_seconds", 0.5))
        )

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                verified = await self._vector_store.verify(ids)
                if verified:
                    return
            except Exception as exc:
                last_error = exc

            if attempt < max_attempts and wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

        if mode == "strict":
            if last_error is not None:
                raise PipelineError(
                    f"Vector store verification failed with provider error: {last_error}",
                    stage="embed",
                    source_url=source_url,
                ) from last_error
            raise PipelineError(
                "Vector store verification failed after add()",
                stage="embed",
                source_url=source_url,
            )

        if last_error is not None:
            logger.warning(
                "vector_store_verification_error",
                provider=self._config.vector_store_provider,
                error=str(last_error),
            )
            return

        logger.warning(
            "vector_store_verification_failed", provider=self._config.vector_store_provider
        )

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------

    async def _run_stages(
        self,
        stages: tuple[Stage, ...],
        ctx: StageContext,
    ) -> None:
        """Execute *stages* in order, recording failures via PipelineError.

        Args:
            stages: Ordered tuple of Stage instances.
            ctx: Shared mutable context for this indexing run.
        """
        for stage in stages:
            try:
                await stage.execute(ctx, self)
            except PipelineError:
                raise
            except Exception as exc:
                raise PipelineError(
                    f"Stage '{stage.name}' failed for {ctx.url}: {exc}",
                    stage=stage.name,
                    source_url=ctx.url,
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _index_single_source(
        self, url: str, *, force: bool = False
    ) -> Literal["indexed", "skipped"]:
        """Index one fully-resolved source URL/path through the full pipeline."""
        # Bind URL context for all logging in this operation
        operation_logger = logger.bind(url=url, operation="index")
        operation_logger.info("index_started", force=force)

        # ---- Resolve work directory ----
        created_temp_dir = False
        if self._config.work_dir is not None:
            work_dir = self._config.work_dir
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="audiorag_"))
            created_temp_dir = True

        # ---- Per-URL asyncio lock (single-process concurrency guard) ----
        url_lock = self._get_url_lock(url)
        async with url_lock:
            try:
                # Re-check status inside the lock to avoid race conditions
                source_info = await self._state.get_source_status(url)
                if source_info is not None:
                    status = source_info["status"]
                    current_vector_id_strategy = resolve_vector_id_strategy(
                        provider_name=self._config.vector_store_provider,
                        configured_format=self._config.vector_id_format,
                        uuid5_namespace=self._config.vector_id_uuid5_namespace,
                        vector_store=self._vector_store,
                    )
                    source_metadata = source_info.get("metadata") or {}
                    previous_vector_id_strategy = source_metadata.get("vector_id_strategy")

                    if status == IndexingStatus.COMPLETED and not force:
                        operation_logger.info("index_skipped", reason="already_indexed")
                        return "skipped"
                    if status in _IN_PROGRESS_STATUSES and not force:
                        operation_logger.info(
                            "index_skipped",
                            reason="in_progress",
                            current_status=status,
                        )
                        return "skipped"
                    if (
                        previous_vector_id_strategy is not None
                        and previous_vector_id_strategy != current_vector_id_strategy
                        and not force
                    ):
                        raise PipelineError(
                            "Vector ID strategy changed for existing source. "
                            "Use force=True to reindex and avoid mixed vector IDs.",
                            stage="index",
                            source_url=url,
                        )
                    if force:
                        operation_logger.info("index_force_reindex")
                        await self._state.delete_source(url)
                        await self._vector_store.delete_by_source(url)

                ctx = StageContext(
                    url=url,
                    config=self._config,
                    logger=operation_logger,
                    work_dir=work_dir,
                    created_temp_dir=created_temp_dir,
                )

                await self._run_stages(_DEFAULT_STAGES, ctx)
                return "indexed"
            except Exception as e:
                operation_logger.error("index_failed", error=str(e), error_type=type(e).__name__)
                try:
                    await self._state.update_source_status(
                        url,
                        IndexingStatus.FAILED,
                        metadata={"error_message": str(e)},
                    )
                except StateError as state_error:
                    operation_logger.error(
                        "state_update_failed",
                        error=str(state_error),
                        error_type=type(state_error).__name__,
                    )
                raise
            finally:
                # Cleanup
                if self._config.cleanup_audio and created_temp_dir:
                    try:
                        shutil.rmtree(work_dir, ignore_errors=True)
                        operation_logger.debug("work_directory_cleaned", work_dir=str(work_dir))
                    except Exception as e:
                        operation_logger.warning(
                            "cleanup_failed", work_dir=str(work_dir), error=str(e)
                        )

    async def index_many(
        self,
        inputs: list[str],
        *,
        force: bool = False,
        raise_on_error: bool = True,
    ) -> BatchIndexResult:
        """Index inputs after source discovery with per-source resumable state.

        Inputs are expanded via ``discover_sources`` before indexing. Playlist/
        channel URLs and local directories are processed as independent sources.

        Args:
            inputs: List of URLs and/or local paths to index.
            force: If True, re-index even if already completed.
            raise_on_error: Raise a PipelineError if any source fails.

        Raises:
            PipelineError: If one or more source indexing runs fail.
        """
        await self._ensure_initialized()

        operation_logger = logger.bind(operation="index_many")
        operation_logger.info(
            "index_many_started",
            input_count=len(inputs),
            force=force,
            raise_on_error=raise_on_error,
        )

        sources = await discover_sources(inputs, self._config)
        if not sources:
            operation_logger.warning("index_many_no_sources")
            return BatchIndexResult(inputs=inputs)

        result = BatchIndexResult(inputs=inputs, discovered_sources=sources)
        pipeline_failures: list[PipelineError] = []
        failure_causes: list[Exception | None] = []

        for source in sources:
            try:
                source_result = await self._index_single_source(source, force=force)
                if source_result == "indexed":
                    result.indexed_sources.append(source)
                else:
                    result.skipped_sources.append(source)
            except Exception as exc:
                pipeline_error, failure, cause = self._normalize_batch_failure(source, exc)
                pipeline_failures.append(pipeline_error)
                failure_causes.append(cause)
                result.failures.append(failure)
                operation_logger.error(
                    "index_many_source_failed",
                    source_url=source,
                    stage=failure.stage,
                    error=failure.error_message,
                )

        operation_logger.info(
            "index_many_completed",
            source_count=len(sources),
            indexed_count=len(result.indexed_sources),
            skipped_count=len(result.skipped_sources),
            failure_count=len(result.failures),
        )

        if result.failures and raise_on_error:
            if len(pipeline_failures) == 1:
                if failure_causes[0] is not None:
                    raise pipeline_failures[0] from failure_causes[0]
                raise pipeline_failures[0]

            failed_sources = [failure.source_url for failure in result.failures]
            failed_sources_str = ", ".join(failed_sources)
            raise PipelineError(
                (
                    f"Indexing failed for {len(result.failures)} of {len(sources)} sources. "
                    f"Failed sources: {failed_sources_str}"
                ),
                stage="index_many",
            )

        return result

    def _normalize_batch_failure(
        self, source: str, exc: Exception
    ) -> tuple[PipelineError, BatchIndexFailure, Exception | None]:
        if isinstance(exc, PipelineError):
            return (
                exc,
                BatchIndexFailure(
                    source_url=exc.source_url or source,
                    stage=exc.stage,
                    error_message=str(exc),
                ),
                None,
            )

        wrapped = PipelineError(
            f"Indexing failed for {source}: {type(exc).__name__}: {exc}",
            stage="index_many",
            source_url=source,
        )
        return (
            wrapped,
            BatchIndexFailure(
                source_url=source,
                stage="index_many",
                error_message=f"{type(exc).__name__}: {exc}",
            ),
            exc,
        )

    async def index(self, url: str, *, force: bool = False) -> None:
        """Index audio from a URL/path through the full pipeline.

        Stages: download -> split -> transcribe -> chunk -> embed -> complete.
        State is tracked in SQLite at each stage for resumability.

        The input is expanded through source discovery. Playlist/channel URLs
        are indexed as per-video sources with independent state.

        Args:
            url: URL/path to index.
            force: If True, re-index even if already completed.

        Raises:
            PipelineError: If any pipeline stage fails.
        """
        await self.index_many([url], force=force)

    async def query(self, query: str) -> QueryResult:
        """Query the indexed audio content.

        Embeds the query, retrieves from vector store, reranks, generates
        an answer, and returns a QueryResult with sources.

        Args:
            query: The user's question.

        Returns:
            QueryResult with generated answer and ranked sources.
        """
        await self._ensure_initialized()

        # Bind query context for logging
        operation_logger = logger.bind(query=query[:100], operation="query")
        operation_logger.info("query_started")

        try:
            # Step 1 - Embed query
            with Timer(operation_logger, "query_embed"):
                self._budget_governor.reserve(
                    provider=self._config.embedding_provider,
                    requests=1,
                    text_chars=len(query),
                )
                query_embedding = (await self._embedder.embed([query]))[0]

            # Step 2 - Retrieve
            with Timer(operation_logger, "query_retrieve") as timer:
                self._budget_governor.reserve(
                    provider=self._config.vector_store_provider,
                    requests=1,
                )
                raw_results = await self._vector_store.query(
                    query_embedding, top_k=self._config.retrieval_top_k
                )
                timer.complete(results_count=len(raw_results))

            if not raw_results:
                operation_logger.info("query_no_results")
                return QueryResult(answer="No relevant information found.", sources=[])

            # Step 3 - Rerank
            with Timer(operation_logger, "query_rerank") as timer:
                documents = [r["document"] for r in raw_results]
                self._budget_governor.reserve(
                    provider=self._config.reranker_provider,
                    requests=1,
                    text_chars=len(query) + sum(len(doc) for doc in documents),
                )
                reranked = await self._reranker.rerank(
                    query, documents, top_n=self._config.rerank_top_n
                )
                timer.complete(reranked_count=len(reranked))

            # Map reranked indices back to original results
            reranked_results = [(raw_results[idx], score) for idx, score in reranked]

            # Step 4 - Generate
            with Timer(operation_logger, "query_generate"):
                context_texts = [r["document"] for r, _ in reranked_results]
                self._budget_governor.reserve(
                    provider=self._config.generation_provider,
                    requests=1,
                    text_chars=len(query) + sum(len(text) for text in context_texts),
                )
                answer = await self._generator.generate(query, context_texts)

            # Step 5 - Build response
            sources = []
            for result, score in reranked_results:
                metadata = result.get("metadata", {})
                sources.append(
                    Source(
                        text=result["document"],
                        start_time=metadata.get("start_time", 0.0),
                        end_time=metadata.get("end_time", 0.0),
                        source_url=metadata.get("source_url", ""),
                        title=metadata.get("title", ""),
                        relevance_score=score,
                    )
                )

            operation_logger.info(
                "query_completed",
                sources_count=len(sources),
                answer_length=len(answer),
            )
            return QueryResult(answer=answer, sources=sources)

        except BudgetExceededError as e:
            operation_logger.error("query_budget_exceeded", error=str(e), provider=e.provider)
            raise
        except Exception as e:
            operation_logger.error("query_failed", error=str(e), error_type=type(e).__name__)
            raise
