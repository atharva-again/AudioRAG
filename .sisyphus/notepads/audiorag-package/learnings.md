
## T11: Pipeline Orchestrator

- Provider implementations diverge from plan in constructor signatures:
  - `OpenAIEmbeddingProvider` and `OpenAIGenerationProvider` take `client: AsyncOpenAI` not `api_key: str`
  - `CohereReranker` takes `client: AsyncClientV2` not `api_key: str`
  - `AudioSplitter` takes `max_size_mb` in constructor, `split_if_needed(audio_path, output_dir)` not `(audio_file, max_size_mb)`
  - `StateManager` uses `source_path` as the key, `store_chunks` takes `list[dict]` not `list[ChunkMetadata]`
- Must make `openai` import lazy (inside `__init__`) since it's an optional dependency — module-level import causes `ModuleNotFoundError`
- Shared `AsyncOpenAI` client between embedder and generator avoids duplicate connections
- `from __future__ import annotations` is essential for forward references in type hints
# Code Style Patterns Analysis

## Import Patterns
- **Organization**: Standard library imports first, then third-party, then local package imports
- **Grouping**: Imports are grouped with blank lines between categories
- **Style**: Absolute imports for local modules (e.g., from audiorag.models import ...)
- **Future imports**: Used in pipeline.py for annotations (from __future__ import annotations)

## Naming Conventions
- **Classes**: PascalCase (AudioRAGPipeline, ChunkMetadata, OpenAIEmbeddingProvider)
- **Functions/Methods**: snake_case (chunk_transcription, embed, get_source_status)
- **Variables**: snake_case (config, audio_file, source_id)
- **Constants/Enums**: UPPER_CASE (IndexingStatus.DOWNLOADING, FAILED)
- **Modules**: snake_case (pipeline.py, models.py)

## Type Annotations
- **Extensive usage**: All function parameters and return types annotated
- **Modern syntax**: Union types use | (float | None), list types use list[Type]
- **Path types**: Uses pathlib.Path consistently
- **Optional types**: Explicit None unions for optional parameters

## Error Handling Patterns
- **Try/except blocks**: Used in pipeline stages with specific error handling
- **Logging**: Uses logging.getLogger with module-specific names ("audiorag.pipeline")
- **State updates**: Failed operations update status to FAILED with error_message
- **Re-raising**: Exceptions are logged but re-raised for caller handling
- **Validation**: Runtime checks (e.g., FFmpeg availability) with clear error messages

## Documentation Patterns
- **Docstrings**: Triple-quoted strings on all classes and public methods
- **Format**: Google-style with Args and Returns sections
- **Content**: Descriptive explanations, parameter descriptions, return value details
- **No type hints in docstrings**: Relies on type annotations instead

## Configuration Files
- **pyproject.toml**: Contains build system, dependencies, optional groups, dev tools
- **Tool configs**: pytest, coverage settings included
- **No linting configs**: No .eslintrc, .prettierrc, setup.cfg (Python project)
- **Build backend**: uv_build for pure Python packaging

## Additional Patterns
- **Async-first**: All public APIs are async coroutines
- **Protocol-based**: Provider interfaces use @runtime_checkable Protocol classes
- **Pydantic models**: Data classes use BaseModel with computed fields
- **Enum usage**: StrEnum for status values
- **Path handling**: Consistent use of pathlib.Path over strings
- **Logging levels**: info for progress, debug for details, warning/error for issues
