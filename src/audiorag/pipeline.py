"""Main AudioRAG pipeline orchestrator."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from audiorag.chunking import chunk_transcription
from audiorag.config import AudioRAGConfig
from audiorag.logging_config import Timer, configure_logging, get_logger
from audiorag.models import (
    IndexingStatus,
    QueryResult,
    Source,
)
from audiorag.protocols import (
    AudioSourceProvider,
    EmbeddingProvider,
    GenerationProvider,
    RerankerProvider,
    STTProvider,
    VectorStoreProvider,
)
from audiorag.retry_config import RetryConfig
from audiorag.state import StateManager

logger = get_logger(__name__)


class AudioRAGPipeline:
    """Orchestrates the full RAG pipeline over audio.

    Handles download, split, transcribe, chunk, embed, and query stages
    with SQLite state tracking and provider-agnostic design.
    """

    def __init__(  # noqa: PLR0913
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
            audio_source: Custom audio source provider. Defaults to YouTubeScraper.
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
            from audiorag.providers.youtube_scraper import YouTubeScraper  # noqa: PLC0415

            archive_path = (
                Path(config.youtube_download_archive) if config.youtube_download_archive else None
            )
            self._audio_source = YouTubeScraper(
                retry_config=retry_config,
                download_archive=archive_path,
                concurrent_fragments=config.youtube_concurrent_fragments,
                skip_playlist_after_errors=config.youtube_skip_after_errors,
                cookie_file=config.youtube_cookie_file,
                po_token=config.youtube_po_token,
                impersonate_client=config.youtube_impersonate,
                player_clients=config.youtube_player_clients,
                js_runtime=config.js_runtime,
            )

        # Initialize STT provider based on config
        if stt is not None:
            self._stt = stt
        else:
            self._stt = self._create_stt_provider(config, retry_config)

        # Initialize embedder based on config
        if embedder is not None:
            self._embedder = embedder
        else:
            self._embedder = self._create_embedding_provider(config, retry_config)

        # Initialize vector store based on config
        if vector_store is not None:
            self._vector_store = vector_store
        else:
            self._vector_store = self._create_vector_store_provider(config, retry_config)

        # Initialize generator based on config
        if generator is not None:
            self._generator = generator
        else:
            self._generator = self._create_generation_provider(config, retry_config)

        # Initialize reranker based on config
        if reranker is not None:
            self._reranker = reranker
        else:
            self._reranker = self._create_reranker_provider(config, retry_config)

        # Internal utilities
        from audiorag.providers.audio_splitter import AudioSplitter  # noqa: PLC0415

        self._splitter = AudioSplitter(max_size_mb=config.audio_split_max_size_mb)
        self._state = StateManager(config.database_path)
        self._initialized = False

    def _create_stt_provider(
        self, config: AudioRAGConfig, retry_config: RetryConfig
    ) -> STTProvider:
        """Create STT provider based on config."""
        provider_name = config.stt_provider.lower()

        if provider_name == "groq":
            from audiorag.providers.groq_stt import GroqSTTProvider  # noqa: PLC0415

            return GroqSTTProvider(
                api_key=config.groq_api_key or None,
                model=config.get_stt_model(),
                retry_config=retry_config,
            )
        if provider_name == "deepgram":
            from audiorag.providers.deepgram_stt import DeepgramSTTProvider  # noqa: PLC0415

            return DeepgramSTTProvider(
                api_key=config.deepgram_api_key or None,
                model=config.get_stt_model(),
                retry_config=retry_config,
            )
        if provider_name == "assemblyai":
            from audiorag.providers.assemblyai_stt import AssemblyAISTTProvider  # noqa: PLC0415

            return AssemblyAISTTProvider(
                api_key=config.assemblyai_api_key or None,
                model=config.get_stt_model(),
                retry_config=retry_config,
            )
        # default to openai
        from audiorag.providers.openai_stt import OpenAISTTProvider  # noqa: PLC0415

        return OpenAISTTProvider(
            api_key=config.openai_api_key or None,
            model=config.stt_model,
            retry_config=retry_config,
        )

    def _create_embedding_provider(
        self, config: AudioRAGConfig, retry_config: RetryConfig
    ) -> EmbeddingProvider:
        """Create embedding provider based on config."""
        provider_name = config.embedding_provider.lower()

        if provider_name == "voyage":
            from audiorag.providers.voyage_embeddings import (  # noqa: PLC0415
                VoyageEmbeddingProvider,
            )

            return VoyageEmbeddingProvider(
                api_key=config.voyage_api_key or None,
                model=config.get_embedding_model(),
                retry_config=retry_config,
            )
        if provider_name == "cohere":
            from audiorag.providers.cohere_embeddings import (  # noqa: PLC0415
                CohereEmbeddingProvider,
            )

            return CohereEmbeddingProvider(
                api_key=config.cohere_api_key or None,
                model=config.get_embedding_model(),
                retry_config=retry_config,
            )
        # default to openai
        from openai import AsyncOpenAI  # noqa: PLC0415 # type: ignore

        from audiorag.providers.openai_embeddings import OpenAIEmbeddingProvider  # noqa: PLC0415

        openai_client = (
            AsyncOpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
        )
        return OpenAIEmbeddingProvider(
            client=openai_client,
            model=config.embedding_model,
            retry_config=retry_config,
        )

    def _create_vector_store_provider(
        self, config: AudioRAGConfig, retry_config: RetryConfig
    ) -> VectorStoreProvider:
        """Create vector store provider based on config."""
        provider_name = config.vector_store_provider.lower()

        if provider_name == "supabase":
            from audiorag.providers.supabase_pgvector import SupabasePgVectorStore  # noqa: PLC0415

            return SupabasePgVectorStore(
                connection_string=config.supabase_connection_string,
                collection_name=config.supabase_collection_name,
                dimension=config.supabase_vector_dimension,
                retry_config=retry_config,
            )
        if provider_name == "pinecone":
            from audiorag.providers.pinecone_store import PineconeVectorStore  # noqa: PLC0415

            return PineconeVectorStore(
                api_key=config.pinecone_api_key or None,
                index_name=config.pinecone_index_name,
                namespace=config.pinecone_namespace,
                retry_config=retry_config,
            )
        if provider_name == "weaviate":
            from audiorag.providers.weaviate_store import WeaviateVectorStore  # noqa: PLC0415

            return WeaviateVectorStore(
                url=config.weaviate_url or None,
                api_key=config.weaviate_api_key or None,
                collection_name=config.weaviate_collection_name,
                retry_config=retry_config,
            )
        # default to chromadb
        from audiorag.providers.chromadb_store import ChromaDBVectorStore  # noqa: PLC0415

        return ChromaDBVectorStore(
            persist_directory=config.chromadb_persist_directory,
            collection_name=config.chromadb_collection_name,
            retry_config=retry_config,
        )

    def _create_generation_provider(
        self, config: AudioRAGConfig, retry_config: RetryConfig
    ) -> GenerationProvider:
        """Create generation provider based on config."""
        provider_name = config.generation_provider.lower()

        if provider_name == "anthropic":
            from audiorag.providers.anthropic_generation import (  # noqa: PLC0415
                AnthropicGenerationProvider,
            )

            return AnthropicGenerationProvider(
                api_key=config.anthropic_api_key or None,
                model=config.get_generation_model(),
                retry_config=retry_config,
            )
        if provider_name == "gemini":
            from audiorag.providers.gemini_generation import (  # noqa: PLC0415
                GeminiGenerationProvider,
            )

            return GeminiGenerationProvider(
                api_key=config.google_api_key or None,
                model=config.get_generation_model(),
                retry_config=retry_config,
            )
        # default to openai
        from openai import AsyncOpenAI  # noqa: PLC0415 # type: ignore

        from audiorag.providers.openai_generation import OpenAIGenerationProvider  # noqa: PLC0415

        openai_client = (
            AsyncOpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
        )
        return OpenAIGenerationProvider(
            client=openai_client,
            model=config.generation_model,
            retry_config=retry_config,
        )

    def _create_reranker_provider(
        self, config: AudioRAGConfig, retry_config: RetryConfig
    ) -> RerankerProvider:
        """Create reranker provider based on config."""
        provider_name = config.reranker_provider.lower()

        if provider_name == "passthrough" or not config.cohere_api_key:
            from audiorag.providers.passthrough_reranker import PassthroughReranker  # noqa: PLC0415

            return PassthroughReranker()
        # default to cohere
        from cohere import AsyncClientV2  # noqa: PLC0415 # type: ignore

        from audiorag.providers.cohere_reranker import CohereReranker  # noqa: PLC0415

        cohere_client = AsyncClientV2(api_key=config.cohere_api_key)
        return CohereReranker(
            client=cohere_client,
            model=config.reranker_model,
            retry_config=retry_config,
        )

    async def _ensure_initialized(self) -> None:
        """Initialize state manager if not already done."""
        if not self._initialized:
            await self._state.initialize()
            self._initialized = True

    async def index(self, url: str, *, force: bool = False) -> None:  # noqa: PLR0912, PLR0915
        """Index audio from a URL through the full pipeline.

        Stages: download -> split -> transcribe -> chunk -> embed -> complete.
        State is tracked in SQLite at each stage for resumability.

        Args:
            url: URL of the audio/video to index.
            force: If True, re-index even if already completed.

        Raises:
            RuntimeError: If any pipeline stage fails.
        """
        await self._ensure_initialized()

        # Bind URL context for all logging in this operation
        operation_logger = logger.bind(url=url, operation="index")
        operation_logger.info("index_started", force=force)

        # Check idempotency
        source_info = await self._state.get_source_status(url)
        if source_info is not None:
            status = source_info["status"]
            if status == IndexingStatus.COMPLETED and not force:
                operation_logger.info("index_skipped", reason="already_indexed")
                return
            if force:
                operation_logger.info("index_force_reindex")
                await self._state.delete_source(url)
                await self._vector_store.delete_by_source(url)

        # Resolve work directory
        work_dir: Path
        created_temp_dir = False
        if self._config.work_dir is not None:
            work_dir = self._config.work_dir
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="audiorag_"))
            created_temp_dir = True

        try:
            # Stage 1 — Download
            with Timer(operation_logger, "stage_download") as timer:
                await self._state.upsert_source(url, IndexingStatus.DOWNLOADING)
                audio_file = await self._audio_source.download(
                    url, work_dir, self._config.audio_format
                )
                await self._state.upsert_source(
                    url,
                    IndexingStatus.DOWNLOADED,
                    metadata={
                        "video_title": audio_file.video_title,
                        "duration": audio_file.duration,
                    },
                )
                timer.complete(
                    video_title=audio_file.video_title,
                    duration_seconds=audio_file.duration,
                    file_path=str(audio_file.path),
                )

            # Stage 2 — Split
            with Timer(operation_logger, "stage_split") as timer:
                await self._state.update_source_status(url, IndexingStatus.SPLITTING)
                audio_parts = await self._splitter.split_if_needed(audio_file.path, work_dir)
                timer.complete(parts_count=len(audio_parts))

            # Stage 3 — Transcribe
            with Timer(operation_logger, "stage_transcribe", parts=len(audio_parts)) as timer:
                await self._state.update_source_status(url, IndexingStatus.TRANSCRIBING)
                all_segments = []
                cumulative_offset = 0.0

                for part_idx, part_path in enumerate(audio_parts):
                    part_logger = operation_logger.bind(
                        part_index=part_idx, part_path=str(part_path)
                    )
                    part_logger.debug("transcribing_part")

                    segments = await self._stt.transcribe(part_path, self._config.stt_language)
                    # Adjust timestamps for subsequent parts
                    if cumulative_offset > 0:
                        from audiorag.models import TranscriptionSegment  # noqa: PLC0415

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

                await self._state.update_source_status(url, IndexingStatus.TRANSCRIBED)
                timer.complete(segments_count=len(all_segments))

            # Stage 4 — Chunk
            with Timer(operation_logger, "stage_chunk") as timer:
                await self._state.update_source_status(url, IndexingStatus.CHUNKING)
                chunks = chunk_transcription(
                    all_segments,
                    self._config.chunk_duration_seconds,
                    url,
                    audio_file.video_title,
                )

                if not chunks:
                    operation_logger.warning("no_chunks_produced")
                    await self._state.update_source_status(url, IndexingStatus.COMPLETED)
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
                            "video_title": c.video_title,
                        },
                    }
                    for i, c in enumerate(chunks)
                ]
                chunk_ids = await self._state.store_chunks(url, chunk_dicts)
                await self._state.update_source_status(url, IndexingStatus.CHUNKED)
                timer.complete(chunks_count=len(chunks))

            # Stage 5 — Embed
            with Timer(operation_logger, "stage_embed") as timer:
                await self._state.update_source_status(url, IndexingStatus.EMBEDDING)
                texts = [c.text for c in chunks]
                embeddings = await self._embedder.embed(texts)

                # Store in vector store
                metadatas = [
                    {
                        "start_time": c.start_time,
                        "end_time": c.end_time,
                        "source_url": c.source_url,
                        "video_title": c.video_title,
                    }
                    for c in chunks
                ]
                documents = [c.text for c in chunks]
                await self._vector_store.add(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                await self._state.update_source_status(url, IndexingStatus.EMBEDDED)
                timer.complete(chunks_count=len(chunks))

            # Stage 6 — Complete
            await self._state.update_source_status(url, IndexingStatus.COMPLETED)
            operation_logger.info("index_completed")

        except Exception as e:
            operation_logger.error("index_failed", error=str(e), error_type=type(e).__name__)
            await self._state.update_source_status(
                url,
                IndexingStatus.FAILED,
                metadata={"error_message": str(e)},
            )
            raise
        finally:
            # Cleanup
            if self._config.cleanup_audio and created_temp_dir:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                    operation_logger.debug("work_directory_cleaned", work_dir=str(work_dir))
                except Exception as e:
                    operation_logger.warning("cleanup_failed", work_dir=str(work_dir), error=str(e))

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
            # Step 1 — Embed query
            with Timer(operation_logger, "query_embed"):
                query_embedding = (await self._embedder.embed([query]))[0]

            # Step 2 — Retrieve
            with Timer(operation_logger, "query_retrieve") as timer:
                raw_results = await self._vector_store.query(
                    query_embedding, top_k=self._config.retrieval_top_k
                )
                timer.complete(results_count=len(raw_results))

            if not raw_results:
                operation_logger.info("query_no_results")
                return QueryResult(answer="No relevant information found.", sources=[])

            # Step 3 — Rerank
            with Timer(operation_logger, "query_rerank") as timer:
                documents = [r["document"] for r in raw_results]
                reranked = await self._reranker.rerank(
                    query, documents, top_n=self._config.rerank_top_n
                )
                timer.complete(reranked_count=len(reranked))

            # Map reranked indices back to original results
            reranked_results = [(raw_results[idx], score) for idx, score in reranked]

            # Step 4 — Generate
            with Timer(operation_logger, "query_generate"):
                context_texts = [r["document"] for r, _ in reranked_results]
                answer = await self._generator.generate(query, context_texts)

            # Step 5 — Build response
            sources = []
            for result, score in reranked_results:
                metadata = result.get("metadata", {})
                sources.append(
                    Source(
                        text=result["document"],
                        start_time=metadata.get("start_time", 0.0),
                        end_time=metadata.get("end_time", 0.0),
                        source_url=metadata.get("source_url", ""),
                        video_title=metadata.get("video_title", ""),
                        relevance_score=score,
                    )
                )

            operation_logger.info(
                "query_completed",
                sources_count=len(sources),
                answer_length=len(answer),
            )
            return QueryResult(answer=answer, sources=sources)

        except Exception as e:
            operation_logger.error("query_failed", error=str(e), error_type=type(e).__name__)
            raise
