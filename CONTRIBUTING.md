# Contributing to NeuralMem

Thanks for your interest in contributing! This guide will help you get started.

## Prerequisites

- Python 3.10+
- Git
- (Optional) Ollama for LLM extraction features

## Setup

```bash
# Clone the repository
git clone https://github.com/your-org/neuralmem.git
cd neuralmem

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run only fast unit tests (skip slow/integration/stub markers)
pytest tests/unit/ -m "not stub and not slow"

# Run with coverage
pytest --cov=neuralmem --cov-report=term-missing

# Run a single test file
pytest tests/unit/test_retrieval.py -v

# Run a single test by name
pytest tests/unit/test_retrieval.py::test_rrf_fusion -v
```

### Test Structure

```
tests/
├── unit/        # Fast, isolated — uses mock_embedder (4-dim, deterministic)
├── integration/ # Real SQLiteStorage with tmp_path fixtures
├── contract/    # Protocol conformance tests for Storage/Embedder/Extractor
└── smoke/       # End-to-end sanity checks
```

Pytest markers: `stub` (placeholder implementations), `integration` (real storage), `slow`.

## Linting

```bash
ruff check src/ tests/
```

Fix auto-fixable issues:

```bash
ruff check --fix src/ tests/
```

## Code Style

- **Formatter/Linter**: Ruff (configured in `pyproject.toml`)
- **Type hints**: Required on all public functions and method signatures
- **Docstrings**: Use Google-style docstrings on all public APIs
- **Imports**: Absolute imports from `neuralmem.*`; sorted by Ruff
- **Frozen models**: Use Pydantic `frozen=True` for data classes (see `core/types.py`)

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`

Examples:
```
feat(storage): add pgvector backend support
fix(retrieval): handle empty query in BM25 search
test(graph): add traversal depth tests
docs(readme): update provider selection table
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass: `pytest`
4. Ensure lint passes: `ruff check src/ tests/`
5. Open a PR with a clear description of what changed and why
6. Link any related issues

## Architecture Overview

See [CLAUDE.md](CLAUDE.md) for a detailed architecture guide. Key points:

- **Public surface**: `NeuralMem` class in `core/memory.py` — four main methods: `remember()`, `recall()`, `reflect()`, `forget()`
- **Protocol-driven**: All backends implement protocols defined in `core/protocols.py`
- **Config**: `NeuralMemConfig` with `NEURALMEM_*` environment variable overrides

## Adding a New Storage Backend

1. Create `src/neuralmem/storage/<backend>.py`
2. Implement the `StorageProtocol` from `core/protocols.py`:

```python
from neuralmem.core.protocols import StorageProtocol
from neuralmem.core.types import Memory

class MyStorage:
    """My custom storage backend."""

    def save_memory(self, memory: Memory) -> str:
        ...

    def get_memory(self, memory_id: str) -> Memory | None:
        ...

    def vector_search(self, vector, user_id=None, memory_types=None, limit=10):
        ...

    # ... implement all other StorageProtocol methods
```

3. Add contract tests in `tests/contract/test_storage_contract.py`
4. Register it in config if needed (add a `storage_backend` field to `NeuralMemConfig`)
5. Inject it via `NeuralMem.__init__(config=..., embedder=...)` or add auto-discovery

## Adding a New Embedding Provider

1. Create `src/neuralmem/embedding/<provider>.py`
2. Implement `EmbedderProtocol` from `core/protocols.py`:

```python
from collections.abc import Sequence

class MyEmbedder:
    """My custom embedding provider."""

    dimension: int = 768

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of texts into vectors."""
        ...

    def encode_one(self, text: str) -> list[float]:
        """Encode a single text into a vector."""
        return self.encode([text])[0]
```

3. Register in `src/neuralmem/embedding/registry.py` — map the provider name to your class
4. Add optional dependency group in `pyproject.toml` if external packages are needed
5. Add contract tests in `tests/contract/test_embedder_contract.py`

## Questions?

Open an issue or start a discussion. We're happy to help!
