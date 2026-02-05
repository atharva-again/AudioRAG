# AudioRAG

RAG over audio files with provider-agnostic pipeline.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd audiorag

# Install dependencies (including dev dependencies)
uv sync

# Install prek hooks
uv run prek install
uv run prek install --hook-type pre-push
```

### Git Hooks

This project uses **prek** (Rust-based, drop-in pre-commit replacement) with hooks from the Astral ecosystem:

**Pre-commit hooks** (run on every commit - fast feedback):
- **Astral Ruff** - Linting with auto-fix and formatting
- **Astral uv** - Lock file validation
- **Ty** - Type checking (100x faster than mypy)
- **validate-pyproject** - pyproject.toml validation
- Large file check (max 1MB)
- JSON/TOML/YAML validation
- Trailing whitespace removal
- Security scanning (gitleaks)
- Common file checks (merge conflicts, private keys, etc.)

**Pre-push hooks** (run before pushing):
- Fast test suite (`pytest --maxfail=3 -x`)

#### Why prek?

- **10-100x faster** than pre-commit (Rust-based)
- **Single binary** - no Python venv overhead
- **Parallel execution** of hooks
- **Native uv integration** for Python environments
- **Drop-in replacement** - same `.pre-commit-config.yaml` format
- Used by Astral's own projects (ruff, ty, uv)

You can run hooks manually:

```bash
# Run all hooks on all files
uv run prek run --all-files

# Run specific hook
uv run prek run ruff

# Run only pre-commit stage hooks
uv run prek run --hook-stage pre-commit

# Run only pre-push stage hooks
uv run prek run --hook-stage pre-push

# List all configured hooks
uv run prek list
```

To bypass hooks temporarily (not recommended):

```bash
git commit --no-verify
git push --no-verify
```

### Code Quality Tools

#### Ruff (Linting & Formatting)

```bash
# Check all files
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix

# Format all files
uv run ruff format .

# Check specific file
uv run ruff check src/audiorag/models.py
```

Configuration: `pyproject.toml` under `[tool.ruff]`

#### Ty (Type Checking)

```bash
# Type check entire project
uv run ty check

# Type check specific file
uv run ty check src/audiorag/models.py

# Watch mode for development
uv run ty check --watch
```

Configuration: `pyproject.toml` under `[tool.ty]`

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/audiorag --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_models.py

# Run specific test
uv run pytest tests/test_models.py::TestChunkMetadata
```

### Building

```bash
uv build
```

---

**Stack:** Python 3.12+ · uv · Ruff · Ty · prek · pytest
