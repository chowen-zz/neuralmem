# LangChain Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `integrations/langchain/` 目录下实现 `NeuralMemRetriever(BaseRetriever)`，作为独立包 `neuralmem-langchain` 发布，支持同步和异步两种调用方式。

**Architecture:** `NeuralMemRetriever` 继承 `langchain-core` 的 `BaseRetriever`，通过 `mem.recall()` 检索记忆并转换为 `Document` 列表。异步路径使用 `asyncio.to_thread()` 包裹同步的 `recall()`，与核心包 MCP server 的模式保持一致。

**Tech Stack:** Python 3.10+, `langchain-core>=0.1.0`, `neuralmem>=0.2.0`, `pytest`, `pytest-asyncio`

---

## File Map

**新建文件：**
- `integrations/langchain/pyproject.toml` — 独立包配置
- `integrations/langchain/neuralmem_langchain/__init__.py` — 公开导出
- `integrations/langchain/neuralmem_langchain/retriever.py` — NeuralMemRetriever + _to_document
- `integrations/langchain/tests/conftest.py` — 共享 fixtures
- `integrations/langchain/tests/test_retriever.py` — 同步路径测试（8 个）
- `integrations/langchain/tests/test_retriever_async.py` — 异步路径测试（4 个）

---

## Task 1: 目录结构 + 包配置

**Files:**
- Create: `integrations/langchain/pyproject.toml`
- Create: `integrations/langchain/neuralmem_langchain/__init__.py`（空占位）

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain/neuralmem_langchain
mkdir -p /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain/tests
```

- [ ] **Step 2: 创建 pyproject.toml**

写入 `integrations/langchain/pyproject.toml`：

```toml
[project]
name = "neuralmem-langchain"
version = "0.2.0"
description = "LangChain integration for NeuralMem — BaseRetriever adapter"
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
addopts = "--tb=short -q"
```

- [ ] **Step 3: 创建空的 __init__.py 占位**

写入 `integrations/langchain/neuralmem_langchain/__init__.py`：

```python
# populated in Task 5
```

- [ ] **Step 4: 安装包（editable 模式）**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
pip install -e ".[dev]"
```

期望输出包含：`Successfully installed neuralmem-langchain`，同时安装 `langchain-core` 和 `neuralmem`。

- [ ] **Step 5: 验证安装**

```bash
python -c "import langchain_core; import neuralmem; print('OK')"
```

期望：`OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/langchain/pyproject.toml integrations/langchain/neuralmem_langchain/__init__.py
git commit -m "feat(integrations): 初始化 neuralmem-langchain 包结构"
```

---

## Task 2: conftest.py — 共享 Fixtures

**Files:**
- Create: `integrations/langchain/tests/conftest.py`

- [ ] **Step 1: 创建 conftest.py**

写入 `integrations/langchain/tests/conftest.py`：

```python
"""Shared fixtures for neuralmem-langchain tests."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import Memory, MemoryScope, MemoryType, SearchResult


@pytest.fixture
def sample_memory() -> Memory:
    return Memory(
        id="abc123def456",
        content="User prefers Python for backend development",
        memory_type=MemoryType.SEMANTIC,
        scope=MemoryScope.USER,
        user_id="test-user",
        tags=("preference", "technology"),
        importance=0.8,
        created_at=datetime(2026, 4, 30, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_result(sample_memory: Memory) -> SearchResult:
    return SearchResult(
        memory=sample_memory,
        score=0.85,
        retrieval_method="semantic",
    )


@pytest.fixture
def mock_mem(sample_result: SearchResult) -> MagicMock:
    """NeuralMem mock — recall() returns one SearchResult by default."""
    mem = MagicMock()
    mem.recall.return_value = [sample_result]
    return mem
```

- [ ] **Step 2: 验证 fixtures 可以导入**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
python -c "from tests.conftest import *; print('fixtures OK')"
```

期望：`fixtures OK`（或无错误输出）

- [ ] **Step 3: Commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/langchain/tests/conftest.py
git commit -m "test(integrations/langchain): 添加 conftest 共享 fixtures"
```

---

## Task 3: 同步 Retriever — TDD

**Files:**
- Create: `integrations/langchain/tests/test_retriever.py`
- Create: `integrations/langchain/neuralmem_langchain/retriever.py`

- [ ] **Step 1: 写同步测试**

写入 `integrations/langchain/tests/test_retriever.py`：

```python
"""Tests for NeuralMemRetriever — synchronous path."""
from __future__ import annotations

import pytest
from neuralmem.core.exceptions import NeuralMemError

from neuralmem_langchain import NeuralMemRetriever


def test_returns_documents(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = retriever.invoke("user preferences")
    assert len(docs) == 1
    assert docs[0].page_content == "User prefers Python for backend development"


def test_document_metadata_has_all_fields(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = retriever.invoke("user preferences")
    meta = docs[0].metadata
    assert meta["memory_id"] == "abc123def456"
    assert meta["score"] == pytest.approx(0.85)
    assert meta["retrieval_method"] == "semantic"
    assert meta["memory_type"] == "semantic"
    assert meta["tags"] == ["preference", "technology"]
    assert meta["created_at"] == "2026-04-30T00:00:00+00:00"
    assert meta["user_id"] == "test-user"


def test_empty_results_returns_empty_list(mock_mem):
    mock_mem.recall.return_value = []
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = retriever.invoke("nothing here")
    assert docs == []


def test_k_param_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, k=3)
    retriever.invoke("query")
    assert mock_mem.recall.call_args.kwargs["limit"] == 3


def test_user_id_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, user_id="alice")
    retriever.invoke("query")
    assert mock_mem.recall.call_args.kwargs["user_id"] == "alice"


def test_min_score_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, min_score=0.7)
    retriever.invoke("query")
    assert mock_mem.recall.call_args.kwargs["min_score"] == pytest.approx(0.7)


def test_default_user_id_is_default(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    retriever.invoke("query")
    assert mock_mem.recall.call_args.kwargs["user_id"] == "default"


def test_neuralmem_error_propagates(mock_mem):
    mock_mem.recall.side_effect = NeuralMemError("recall failed")
    retriever = NeuralMemRetriever(mem=mock_mem)
    with pytest.raises(NeuralMemError, match="recall failed"):
        retriever.invoke("query")
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
pytest tests/test_retriever.py -v
```

期望：`FAILED` — `ImportError: cannot import name 'NeuralMemRetriever' from 'neuralmem_langchain'`

- [ ] **Step 3: 实现 retriever.py**

写入 `integrations/langchain/neuralmem_langchain/retriever.py`：

```python
"""NeuralMem LangChain BaseRetriever adapter."""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from neuralmem.core.types import SearchResult


class NeuralMemRetriever(BaseRetriever):
    """
    LangChain BaseRetriever backed by NeuralMem.

    Usage:
        from neuralmem import NeuralMem
        from neuralmem_langchain import NeuralMemRetriever

        mem = NeuralMem()
        retriever = NeuralMemRetriever(mem=mem, user_id="alice", k=5)

        # Sync:
        docs = retriever.invoke("user preferences")

        # Async (in LCEL chain):
        docs = await retriever.ainvoke("user preferences")
    """

    mem: Any
    user_id: str = "default"
    k: int = 5
    min_score: float = 0.3

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = self.mem.recall(
            query,
            user_id=self.user_id,
            limit=self.k,
            min_score=self.min_score,
        )
        return [_to_document(r) for r in results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = await asyncio.to_thread(
            self.mem.recall,
            query,
            user_id=self.user_id,
            limit=self.k,
            min_score=self.min_score,
        )
        return [_to_document(r) for r in results]


def _to_document(result: SearchResult) -> Document:
    """Convert a NeuralMem SearchResult to a LangChain Document."""
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

- [ ] **Step 4: 运行同步测试确认通过**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
pytest tests/test_retriever.py -v
```

期望：`8 passed`

- [ ] **Step 5: ruff lint**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
ruff check neuralmem_langchain/retriever.py
```

期望：`All checks passed!`

- [ ] **Step 6: Commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/langchain/tests/test_retriever.py \
        integrations/langchain/neuralmem_langchain/retriever.py
git commit -m "feat(integrations/langchain): 实现 NeuralMemRetriever 同步路径"
```

---

## Task 4: 异步 Retriever — TDD

**Files:**
- Create: `integrations/langchain/tests/test_retriever_async.py`

注意：异步实现已在 Task 3 的 `retriever.py` 中一并完成（`_aget_relevant_documents`）。本任务只需写异步测试并验证。

- [ ] **Step 1: 写异步测试**

写入 `integrations/langchain/tests/test_retriever_async.py`：

```python
"""Tests for NeuralMemRetriever — asynchronous path."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from neuralmem_langchain import NeuralMemRetriever


@pytest.mark.asyncio
async def test_async_returns_documents(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = await retriever.ainvoke("user preferences")
    assert len(docs) == 1
    assert docs[0].page_content == "User prefers Python for backend development"


@pytest.mark.asyncio
async def test_async_uses_to_thread(mock_mem):
    """Verify asyncio.to_thread() is used so recall() doesn't block the event loop."""
    retriever = NeuralMemRetriever(mem=mock_mem)
    with patch(
        "neuralmem_langchain.retriever.asyncio.to_thread",
        wraps=asyncio.to_thread,
    ) as mock_thread:
        await retriever.ainvoke("query")
        mock_thread.assert_called_once()


@pytest.mark.asyncio
async def test_async_empty_results(mock_mem):
    mock_mem.recall.return_value = []
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = await retriever.ainvoke("nothing")
    assert docs == []


@pytest.mark.asyncio
async def test_async_user_id_forwarded(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, user_id="bob")
    await retriever.ainvoke("query")
    assert mock_mem.recall.call_args.kwargs["user_id"] == "bob"
```

- [ ] **Step 2: 运行确认测试通过（实现已在 Task 3 完成）**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
pytest tests/test_retriever_async.py -v
```

期望：`4 passed`

- [ ] **Step 3: 运行全部测试**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
pytest tests/ -v
```

期望：`12 passed`

- [ ] **Step 4: Commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/langchain/tests/test_retriever_async.py
git commit -m "test(integrations/langchain): 添加 NeuralMemRetriever 异步路径测试"
```

---

## Task 5: 完善 __init__.py + 最终验证

**Files:**
- Modify: `integrations/langchain/neuralmem_langchain/__init__.py`

- [ ] **Step 1: 更新 __init__.py 公开导出**

写入 `integrations/langchain/neuralmem_langchain/__init__.py`：

```python
"""neuralmem-langchain — LangChain integration for NeuralMem."""
from neuralmem_langchain.retriever import NeuralMemRetriever

__version__ = "0.2.0"
__all__ = ["NeuralMemRetriever"]
```

- [ ] **Step 2: 验证公开导入路径**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
python -c "from neuralmem_langchain import NeuralMemRetriever; print(NeuralMemRetriever.__doc__[:40])"
```

期望：`LangChain BaseRetriever backed by Neural`

- [ ] **Step 3: 全量测试 + lint**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
pytest tests/ -q
ruff check neuralmem_langchain/
```

期望：`12 passed` + `All checks passed!`

- [ ] **Step 4: 验证 NeuralMem 核心测试未受影响**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
pytest tests/ -m "not slow" -q
```

期望：`216 passed`

- [ ] **Step 5: Commit 最终版本**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/langchain/neuralmem_langchain/__init__.py
git commit -m "feat(integrations/langchain): 完成 neuralmem-langchain v0.2.0"
```

---

## Self-Review

**Spec coverage:**
- ✅ `NeuralMemRetriever(BaseRetriever)` — Task 3
- ✅ `_get_relevant_documents()` 同步 — Task 3
- ✅ `_aget_relevant_documents()` 异步 — Task 3 实现 + Task 4 测试
- ✅ `asyncio.to_thread()` 包裹 — Task 3 实现 + Task 4 `test_async_uses_to_thread`
- ✅ `Document.metadata` 包含所有字段 — Task 3 `test_document_metadata_has_all_fields`
- ✅ `pyproject.toml` 独立包配置 — Task 1
- ✅ 10 个测试用例（spec Section 6）— Tasks 3+4 合计 12 个（超出 spec 最低要求）
- ✅ `NeuralMemError` 向上传播 — Task 3 `test_neuralmem_error_propagates`
- ✅ 无真实网络调用 — 全部 mock `NeuralMem`

**Type consistency:**
- `NeuralMemRetriever.mem: Any` — 在 Task 3 和 Task 4 中一致
- `_to_document(result: SearchResult) -> Document` — 在 Task 3 定义，Task 4 通过同一个 retriever 间接调用
- `mock_mem.recall.call_args.kwargs["user_id"]` — 同步 Task 3 和异步 Task 4 使用相同断言模式
