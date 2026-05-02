# NeuralMem LangChain Integration Design

**Date:** 2026-04-30
**Status:** Approved
**Scope:** `integrations/langchain/` — `NeuralMemRetriever` as LangChain `BaseRetriever`

---

## 1. Problem Statement

NeuralMem v0.2.0 provides a powerful multi-provider memory engine, but teams using LangChain cannot plug it into RAG pipelines without writing custom glue code. There is no `BaseRetriever` adapter, so NeuralMem cannot participate in LCEL chains or be used with LangChain's standard retrieval interfaces.

---

## 2. Goals

- Implement `NeuralMemRetriever(BaseRetriever)` — sync + async — for use in LangChain RAG pipelines
- Package as `neuralmem-langchain` (wheel), published from the same monorepo
- Zero changes to the `neuralmem` core package
- All dependencies resolved at install time with no version conflicts
- ≥80% test coverage, no real network calls in CI

## 3. Non-Goals

- `NeuralMemChatMemory` (BaseChatMemory / BaseMemory adapter) — deferred to a follow-up
- LlamaIndex, CrewAI, AutoGen adapters — separate sub-projects
- Streaming retrieval or async generators
- LangChain < 0.1 compatibility

---

## 4. Architecture

### 4.1 Directory Structure

```
NeuralMem/
├── integrations/
│   └── langchain/
│       ├── pyproject.toml
│       ├── neuralmem_langchain/
│       │   ├── __init__.py
│       │   └── retriever.py
│       └── tests/
│           ├── conftest.py
│           ├── test_retriever.py
│           └── test_retriever_async.py
└── src/neuralmem/          # unchanged
```

### 4.2 Dependency Graph

```
neuralmem-langchain (0.2.0)
  ├── neuralmem >= 0.2.0
  └── langchain-core >= 0.1.0   # base classes only, NOT full langchain
```

`langchain-core` is used (not `langchain`) to avoid pulling in the full LangChain
dependency tree. All `BaseRetriever` base classes live in `langchain-core`.

### 4.3 Core Class

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from neuralmem import NeuralMem

class NeuralMemRetriever(BaseRetriever):
    """
    LangChain BaseRetriever backed by NeuralMem.

    Usage:
        mem = NeuralMem()
        retriever = NeuralMemRetriever(mem=mem, user_id="alice", k=5)
        docs = retriever.invoke("user preferences")
    """
    mem: NeuralMem
    user_id: str = "default"
    k: int = 5
    min_score: float = 0.3

    model_config = {"arbitrary_types_allowed": True}  # NeuralMem is not a Pydantic model

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = self.mem.recall(
            query,
            user_id=self.user_id,
            limit=self.k,
            min_score=self.min_score,
        )
        return [_to_document(r) for r in results]

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        import asyncio
        results = await asyncio.to_thread(
            self.mem.recall,
            query,
            user_id=self.user_id,
            limit=self.k,
            min_score=self.min_score,
        )
        return [_to_document(r) for r in results]


def _to_document(result: SearchResult) -> Document:
    return Document(
        page_content=result.memory.content,
        metadata={
            "memory_id": result.memory.id,
            "score": result.score,
            "retrieval_method": result.retrieval_method,
            "memory_type": result.memory.memory_type.value,
            "tags": list(result.memory.tags),
            "created_at": result.memory.created_at.isoformat(),
            "user_id": result.memory.user_id,
        },
    )
```

### 4.4 Data Flow

```
LangChain chain.invoke(query)
  → NeuralMemRetriever._get_relevant_documents(query)
  → NeuralMem.recall(query, user_id, limit=k, min_score)
  → [SearchResult(memory, score, retrieval_method)] 
  → [Document(page_content=content, metadata={memory_id, score, ...})]
  → returned to chain
```

For async (`ainvoke`):
```
async chain.ainvoke(query)
  → _aget_relevant_documents(query)
  → asyncio.to_thread(mem.recall, ...)   # runs sync recall in thread pool
  → same Document list
```

---

## 5. Package Configuration

### `integrations/langchain/pyproject.toml`

```toml
[project]
name = "neuralmem-langchain"
version = "0.2.0"
description = "LangChain integration for NeuralMem — BaseRetriever adapter"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
authors = [{ name = "NeuralMem Team" }]
dependencies = [
    "neuralmem>=0.2.0",
    "langchain-core>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["neuralmem_langchain"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### `neuralmem_langchain/__init__.py`

```python
from neuralmem_langchain.retriever import NeuralMemRetriever

__version__ = "0.2.0"
__all__ = ["NeuralMemRetriever"]
```

---

## 6. Testing Strategy

No real network calls. `NeuralMem` is mocked via `unittest.mock.MagicMock`.

### Test Files

```
tests/
├── conftest.py              # shared fixtures
├── test_retriever.py        # sync path
└── test_retriever_async.py  # async path
```

### `conftest.py` fixtures

```python
@pytest.fixture
def mock_mem():
    """NeuralMem instance with recall() mocked."""
    from unittest.mock import MagicMock
    from neuralmem.core.types import Memory, MemoryType, SearchResult
    mem = MagicMock()
    mem.recall.return_value = [
        SearchResult(
            memory=Memory(
                content="User prefers Python",
                memory_type=MemoryType.SEMANTIC,
                user_id="test-user",
            ),
            score=0.85,
            retrieval_method="semantic",
        )
    ]
    return mem
```

### Test Cases

| File | Test | Verifies |
|------|------|---------|
| `test_retriever.py` | `test_returns_documents` | `recall()` result → `Document` list |
| `test_retriever.py` | `test_document_page_content` | `page_content` == `memory.content` |
| `test_retriever.py` | `test_document_metadata_fields` | all metadata keys present and correct |
| `test_retriever.py` | `test_empty_results` | `recall()` returns `[]` → retriever returns `[]` |
| `test_retriever.py` | `test_k_param_forwarded` | `k=3` → `recall(limit=3)` |
| `test_retriever.py` | `test_user_id_forwarded` | `user_id="alice"` → `recall(user_id="alice")` |
| `test_retriever.py` | `test_min_score_forwarded` | `min_score=0.5` → `recall(min_score=0.5)` |
| `test_retriever.py` | `test_recall_error_propagates` | `NeuralMemError` raised → propagates to caller |
| `test_retriever_async.py` | `test_async_returns_documents` | async path returns same Documents |
| `test_retriever_async.py` | `test_async_uses_thread` | `asyncio.to_thread` is called (not blocking) |

---

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| `recall()` raises `NeuralMemError` | Propagates to LangChain caller — no swallowing |
| `recall()` returns `[]` | Returns `[]` — LangChain handles empty retrieval |
| `NeuralMem` not initialized | `ConfigError` at construction time (before any retrieval) |

No silent fallbacks. LangChain callers receive the actual exception.

---

## 8. Delivery Order

1. Create `integrations/langchain/` directory structure
2. Write `pyproject.toml` and `neuralmem_langchain/__init__.py`
3. Write tests first (TDD) — confirm they fail
4. Implement `retriever.py` — confirm tests pass
5. Run `ruff check` + `pytest`
6. Commit

---

## 9. Open Questions

- None. All decisions resolved during brainstorming.
