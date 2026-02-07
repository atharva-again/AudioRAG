# ARCHIVED — Superseded by plan
# See: .sisyphus/plans/audiorag-v2-redesign.md
- [x] LangChain/LlamaIndex pain points (instability, over-abstraction, breaking changes, framework tax)
- [x] Successful framework patterns (LiteLLM, Instructor, Haystack 2.0)
- [x] Audio/video RAG landscape (no dedicated library exists, Ragie commercial, multimodal threat)

## Current State Assessment

### What's Good
- Protocol-based provider pattern (clean, runtime_checkable, well-typed)
- Pydantic models (standard, familiar)
- Structured exceptions with context (PipelineError.stage, ProviderError.retryable)
- State management with resumability (SQLite, WAL mode)
- Retry config centralized
- Optional dependencies via pyproject.toml extras
- Good test foundation

### Critical Problems
1. **Identity crisis**: Half all-in-one pipeline, half composable toolkit. Modular packages (query, retrieve) are TODO stubs. Worse than either extreme.
2. **Rigid pipeline**: 6 hardcoded stages, no hooks/customization, can't skip/reorder stages
3. **God config**: 239 lines, every provider's config in one class. Overwhelming.
4. **Provider factory in pipeline**: _create_stt_provider(), _create_embedding_provider() etc. inside pipeline.py. Coupling.
5. **YouTube-centrism**: video_title, youtube_timestamp_url baked into core models
6. **Maintenance burden**: 14+ providers to maintain, each a potential breaking change surface
7. **Import inconsistency**: Old providers/ path vs new modular packages. Two ways to access same thing.

## Landscape Findings

### No dedicated audio/video RAG library exists
- People cobble together: yt-dlp + whisper + chromadb + langchain
- Ragie (commercial SaaS) just launched audio/video RAG
- NVIDIA published a tutorial but no library
- **This is a real opportunity** - AudioRAG could own this niche

### Multimodal threat is real but not fatal
- GPT-4o, Gemini can process audio directly
- But text-based RAG still wins for: cost, search precision, existing tooling, scale
- Smart play: support BOTH paths (transcribe-then-RAG AND native audio embedding when available)

### What killed LangChain's reputation
- Deep abstraction (can't debug, can't understand)
- Breaking changes every release
- "Framework tax" - forced into their patterns for simple things
- Dependency bloat
- Documentation always behind the code
- 45% of devs never use it in production

### What makes good frameworks loved
- **LiteLLM**: One function, swap providers by string. `completion(model="gpt-4", ...)` → `completion(model="claude-3", ...)`
- **Instructor**: Pydantic-first, works WITH your code not against it
- **Haystack 2.0**: Component protocol with `run()`, composable pipelines
- **Pipecat**: Frame-based data flow, everything is a FrameProcessor, composable pipelines, per-provider optional deps

### Common pattern in loved frameworks
1. Thin core, no magic
2. Composability over configuration
3. Work WITH the user's code, not wrap it
4. Stability over features
5. Plugin/extension model for providers

## User Decisions (from interview)
- **Mission**: "People doing RAG on audio shouldn't have to reinvent the wheel"
- **Breaking changes**: Pre-launch, break anything. Full design freedom.
- **Provider strategy**: Support everything in-house (user wants quality control)
- **Scope**: Audio-first, video-aware
- **Identity**: Not yet decided — user is open. Goal is utility, not framework ideology.

## My Recommendation Emerging
**Layered Library** — Simple high-level API ("index this audio, query it") built on composable primitives.

Key changes needed:
1. Kill the God Config → per-provider config objects
2. Decouple pipeline stages → standalone composable components
3. Abstract away YouTube → generic AudioSource, YouTube is one implementation
4. Make provider wrappers thinner → sustainable maintenance at scale
5. Fix or kill the half-baked modular packages
6. Remove YouTube-centrism from core models

## Additional User Decisions (round 2)
- **Team**: Solo now, hoping to grow community → architecture must invite contributions
- **Thin wrappers**: Approved. "Do whatever you deem fit, but easy to integrate"
- **Scope**: "You tell me" → User trusts my judgment
- **Implicit**: User cares about *outcomes* not *ideology*. Not married to any pattern.

## My Final Recommendation
**Full architectural redesign as a Layered Library.**

Rationale: Current issues are structural (identity crisis, god config, rigid pipeline, 
half-baked modules, YouTube-centrism). Band-aids won't fix these. Pre-launch = full freedom.

**Core principle**: AudioRAG's job is "audio → searchable knowledge base." 
Query/generation is a convenience layer, NOT the core identity.

### Architecture Vision
- Layer 1: Composable primitives (each usable standalone)
- Layer 2: Pipeline (the "don't reinvent the wheel" experience, built on Layer 1)
- Thin provider wrappers with shared base classes
- Per-domain config (not God Config)
- Generic source model (not YouTube-specific)
- Plugin-friendly protocols (easy for community to contribute providers)

## Scope Boundaries
- IN: Full codebase restructuring, new module layout, base classes, thinned providers, 
  hookable pipeline, config decomposition, model genericization, public API cleanup
- OUT: New features (semantic chunking, diarization, etc.) — only architecture
- OUT: Video-specific features (visual frame extraction, OCR) — future
- OUT: CLI / web UI / dashboard — not in scope

## Technical Decisions
- Layered library architecture (simple API + composable internals)
- Base classes per provider category for shared retry/logging/error handling
- Protocols remain (runtime_checkable, structural subtyping) — proven good
- State management stays as-is (solid foundation)
- Retry config stays centralized (good pattern)
- Config decomposed into per-component configs composed by pipeline
- All core models made source-agnostic (no video_title, no youtube_timestamp_url in base)
- Local file and URL audio sources added alongside YouTube

