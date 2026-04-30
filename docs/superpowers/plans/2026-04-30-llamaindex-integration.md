# LlamaIndex Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `integrations/llamaindex/` 下实现 `NeuralMemRetriever(BaseRetriever)` 和 `NeuralMemChatMemory(BaseMemory)`，作为独立包 `neuralmem-llamaindex` 发布，支持 LlamaIndex RAG pipeline 和对话 agent 记忆。

**Architecture:** `NeuralMemRetriever` 将 `mem.recall()` 结果映射为 `NodeWithScore` 列表；`NeuralMemChatMemory` 在有 query 时走语义检索，无 query 时直接从 storage 按时间排序返回最近消息，`put()` 将消息存为 EPISODIC 记忆并用 tags 记录 role。结构与 `integrations/langchain/` 完全对称。

**Tech Stack:** Python 3.10+, `llama-index-core>=0.10.0`, `neuralmem>=0.2.0`, `pytest`, `pytest-asyncio`

---

## File Map

**新建文件：**
- `integrations/llamaindex/pyproject.toml`
- `integrations/llamaindex/neuralmem_llamaindex/__init__.py`
- `integrations/llamaindex/neuralmem_llamaindex/retriever.py`
- `integrations/llamaindex/neuralmem_llamaindex/chat_memory.py`
- `integrations/llamaindex/tests/conftest.py`
- `integrations/llamaindex/tests/test_retriever.py`
- `integrations/llamaindex/tests/test_chat_memory.py`

---

## Task 1: 目录结构 + 包配置

**Files:**
- Create: `integrations/llamaindex/pyproject.toml`
- Create: `integrations/llamaindex/neuralmem_llamaindex/__init__.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex/neuralmem_llamaindex
mkdir -p /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex/tests
```

- [ ] **Step 2: 创建 pyproject.toml**

写入 `integrations/llamaindex/pyproject.toml`：

```toml
[project]
name = "neuralmem-llamaindex"
version = "0.2.0"
description = "LlamaIndex integration for NeuralMem — BaseRetriever and BaseMemory adapters"
requires-python = ">=3.10"
license = "Apache-2.0"
authors = [{ name = "NeuralMem Team" }]
dependencies = [
    "neuralmem>=0.2.0",
    "llama-index-core>=0.10.0",
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
packages = ["neuralmem_llamaindex"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--tb=short -q"
```

- [ ] **Step 3: 创建 __init__.py 占位**

写入 `integrations/llamaindex/neuralmem_llamaindex/__init__.py`：

```python
# populated in Task 5
```

- [ ] **Step 4: 安装包（editable 模式）**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
pip install -e ".[dev]"
```

期望：`Successfully installed neuralmem-llamaindex`，同时安装 `llama-index-core` 和 `neuralmem`。

- [ ] **Step 5: 验证安装**

```bash
python -c "import llama_index.core; import neuralmem; print('OK')"
```

期望：`OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/llamaindex/pyproject.toml integrations/llamaindex/neuralmem_llamaindex/__init__.py
git commit -m "feat(integrations): 初始化 neuralmem-llamaindex 包结构"
```

---

## Task 2: conftest.py — 共享 Fixtures

**Files:**
- Create: `integrations/llamaindex/tests/conftest.py`

- [ ] **Step 1: 创建 conftest.py**

写入 `integrations/llamaindex/tests/conftest.py`：

```python
"""Shared fixtures for neuralmem-llamaindex tests."""
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
def mock_mem(sample_memory: Memory, sample_result: SearchResult) -> MagicMock:
    """NeuralMem mock with recall(), remember(), forget(), storage.list_memories() pre-configured."""
    mem = MagicMock()
    mem.recall.return_value = [sample_result]
    mem.remember.return_value = [sample_memory]
    mem.storage.list_memories.return_value = [sample_memory]
    return mem
```

- [ ] **Step 2: 验证 fixtures 可以导入**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
python -c "from neuralmem.core.types import Memory, MemoryScope, MemoryType, SearchResult; print('types OK')"
```

期望：`types OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/llamaindex/tests/conftest.py
git commit -m "test(integrations/llamaindex): 添加 conftest 共享 fixtures"
```

---

## Task 3: NeuralMemRetriever — TDD

**Files:**
- Create: `integrations/llamaindex/tests/test_retriever.py`
- Create: `integrations/llamaindex/neuralmem_llamaindex/retriever.py`

- [ ] **Step 1: 写测试文件**

写入 `integrations/llamaindex/tests/test_retriever.py`：

```python
"""Tests for NeuralMemRetriever."""
from __future__ import annotations

import pytest
from neuralmem.core.exceptions import NeuralMemError

from neuralmem_llamaindex import NeuralMemRetriever


def test_returns_nodes(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("user preferences")
    assert len(nodes) == 1


def test_node_text_is_memory_content(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("user preferences")
    assert nodes[0].node.text == "User prefers Python for backend development"


def test_node_score(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("user preferences")
    assert nodes[0].score == pytest.approx(0.85)


def test_node_metadata_fields(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("user preferences")
    meta = nodes[0].node.metadata
    assert meta["memory_id"] == "abc123def456"
    assert meta["memory_type"] == "semantic"
    assert meta["tags"] == ["preference", "technology"]
    assert meta["created_at"] == "2026-04-30T00:00:00+00:00"
    assert meta["user_id"] == "test-user"


def test_empty_results(mock_mem):
    mock_mem.recall.return_value = []
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("nothing here")
    assert nodes == []


def test_k_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, k=3)
    retriever.retrieve("query")
    assert mock_mem.recall.call_args.kwargs["limit"] == 3


def test_user_id_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, user_id="alice")
    retriever.retrieve("query")
    assert mock_mem.recall.call_args.kwargs["user_id"] == "alice"


def test_error_propagates(mock_mem):
    mock_mem.recall.side_effect = NeuralMemError("recall failed")
    retriever = NeuralMemRetriever(mem=mock_mem)
    with pytest.raises(NeuralMemError, match="recall failed"):
        retriever.retrieve("query")
```

注意：LlamaIndex `BaseRetriever` 的公开方法是 `retrieve()`，它内部调用 `_retrieve()`。

- [ ] **Step 2: 运行确认失败**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
pytest tests/test_retriever.py -v
```

期望：`FAILED` — `ImportError: cannot import name 'NeuralMemRetriever'`

- [ ] **Step 3: 实现 retriever.py**

写入 `integrations/llamaindex/neuralmem_llamaindex/retriever.py`：

```python
"""NeuralMem LlamaIndex BaseRetriever adapter."""
from __future__ import annotations

from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from neuralmem.core.types import SearchResult


class NeuralMemRetriever(BaseRetriever):
    """
    LlamaIndex BaseRetriever backed by NeuralMem.

    Usage:
        from neuralmem import NeuralMem
        from neuralmem_llamaindex import NeuralMemRetriever

        mem = NeuralMem()
        retriever = NeuralMemRetriever(mem=mem, user_id="alice", k=5)
        nodes = retriever.retrieve("user preferences")
    """

    mem: Any
    user_id: str = "default"
    k: int = 5
    min_score: float = 0.3

    model_config = {"arbitrary_types_allowed": True}

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = self.mem.recall(
            query_bundle.query_str,
            user_id=self.user_id,
            limit=self.k,
            min_score=self.min_score,
        )
        return [_to_node_with_score(r) for r in results]


def _to_node_with_score(result: SearchResult) -> NodeWithScore:
    """Convert a NeuralMem SearchResult to a LlamaIndex NodeWithScore."""
    return NodeWithScore(
        node=TextNode(
            text=result.memory.content,
            metadata={
                "memory_id": result.memory.id,
                "memory_type": result.memory.memory_type.value,
                "tags": list(result.memory.tags),
                "created_at": result.memory.created_at.isoformat(),
                "user_id": result.memory.user_id,
            },
        ),
        score=result.score,
    )
```

- [ ] **Step 4: 同时更新 __init__.py（使测试可以 import）**

写入 `integrations/llamaindex/neuralmem_llamaindex/__init__.py`：

```python
from neuralmem_llamaindex.retriever import NeuralMemRetriever

__version__ = "0.2.0"
__all__ = ["NeuralMemRetriever"]
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
pytest tests/test_retriever.py -v
```

期望：`8 passed`

- [ ] **Step 6: ruff lint**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
ruff check neuralmem_llamaindex/retriever.py
```

期望：`All checks passed!`

- [ ] **Step 7: Commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/llamaindex/tests/test_retriever.py \
        integrations/llamaindex/neuralmem_llamaindex/retriever.py \
        integrations/llamaindex/neuralmem_llamaindex/__init__.py
git commit -m "feat(integrations/llamaindex): 实现 NeuralMemRetriever"
```

---

## Task 4: NeuralMemChatMemory — TDD

**Files:**
- Create: `integrations/llamaindex/tests/test_chat_memory.py`
- Create: `integrations/llamaindex/neuralmem_llamaindex/chat_memory.py`

- [ ] **Step 1: 写测试文件**

写入 `integrations/llamaindex/tests/test_chat_memory.py`：

```python
"""Tests for NeuralMemChatMemory."""
from __future__ import annotations

import pytest
from llama_index.core.llms import ChatMessage, MessageRole

from neuralmem_llamaindex import NeuralMemChatMemory


def test_put_calls_remember(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    msg = ChatMessage(role=MessageRole.USER, content="Hello, I like Python")
    memory.put(msg)
    mock_mem.remember.assert_called_once()
    call_kwargs = mock_mem.remember.call_args
    assert call_kwargs.args[0] == "Hello, I like Python"
    assert call_kwargs.kwargs["user_id"] == "default"


def test_put_tags_role(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    msg = ChatMessage(role=MessageRole.ASSISTANT, content="Got it")
    memory.put(msg)
    tags = mock_mem.remember.call_args.kwargs["tags"]
    assert "assistant" in tags


def test_get_with_input_calls_recall(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    result = memory.get(input="user preferences")
    mock_mem.recall.assert_called_once()
    assert mock_mem.recall.call_args.args[0] == "user preferences"
    assert isinstance(result, list)
    assert len(result) == 1


def test_get_without_input_uses_storage(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    result = memory.get(input=None)
    mock_mem.storage.list_memories.assert_called_once()
    assert isinstance(result, list)


def test_get_returns_chat_messages(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    result = memory.get(input="query")
    assert len(result) == 1
    assert isinstance(result[0], ChatMessage)
    assert result[0].content == "User prefers Python for backend development"


def test_reset_calls_forget(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem, user_id="alice")
    memory.reset()
    mock_mem.forget.assert_called_once_with(user_id="alice")


def test_set_resets_then_puts(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    messages = [
        ChatMessage(role=MessageRole.USER, content="msg1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="msg2"),
    ]
    memory.set(messages)
    mock_mem.forget.assert_called_once()
    assert mock_mem.remember.call_count == 2


def test_get_all_uses_storage(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    result = memory.get_all()
    mock_mem.storage.list_memories.assert_called_once()
    assert isinstance(result, list)
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
pytest tests/test_chat_memory.py -v
```

期望：`FAILED` — `ImportError: cannot import name 'NeuralMemChatMemory'`

- [ ] **Step 3: 实现 chat_memory.py**

写入 `integrations/llamaindex/neuralmem_llamaindex/chat_memory.py`：

```python
"""NeuralMem LlamaIndex BaseMemory adapter."""
from __future__ import annotations

from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory

from neuralmem.core.types import Memory, MemoryType, SearchResult


class NeuralMemChatMemory(BaseMemory):
    """
    LlamaIndex BaseMemory backed by NeuralMem.
    Stores chat messages as EPISODIC memories, with role stored in tags.

    Usage:
        from neuralmem import NeuralMem
        from neuralmem_llamaindex import NeuralMemChatMemory

        mem = NeuralMem()
        memory = NeuralMemChatMemory(mem=mem, user_id="alice")
        # In a ReActAgent or similar:
        agent = ReActAgent.from_tools([...], memory=memory)
    """

    mem: Any
    user_id: str = "default"
    window_size: int = 10

    model_config = {"arbitrary_types_allowed": True}

    def put(self, message: ChatMessage) -> None:
        self.mem.remember(
            message.content,
            user_id=self.user_id,
            memory_type=MemoryType.EPISODIC,
            tags=[message.role.value],
        )

    def get(self, input: str | None = None, **kwargs: Any) -> list[ChatMessage]:
        if input:
            results = self.mem.recall(input, user_id=self.user_id, limit=self.window_size)
            return [_result_to_chat_message(r) for r in results]
        memories = self.mem.storage.list_memories(user_id=self.user_id, limit=self.window_size)
        return [
            _memory_to_chat_message(m)
            for m in sorted(memories, key=lambda m: m.created_at)
        ]

    def get_all(self) -> list[ChatMessage]:
        memories = self.mem.storage.list_memories(user_id=self.user_id, limit=10_000)
        return [
            _memory_to_chat_message(m)
            for m in sorted(memories, key=lambda m: m.created_at)
        ]

    def reset(self) -> None:
        self.mem.forget(user_id=self.user_id)

    def set(self, messages: list[ChatMessage]) -> None:
        self.reset()
        for msg in messages:
            self.put(msg)

    def to_string(self) -> str:
        messages = self.get_all()
        return "\n".join(f"{m.role.value}: {m.content}" for m in messages)


def _result_to_chat_message(result: SearchResult) -> ChatMessage:
    """Convert a recall() SearchResult to a ChatMessage, recovering role from tags."""
    tags = list(result.memory.tags)
    role = _role_from_tags(tags)
    return ChatMessage(role=role, content=result.memory.content)


def _memory_to_chat_message(memory: Memory) -> ChatMessage:
    """Convert a Memory to a ChatMessage for the chronological fallback path."""
    tags = list(memory.tags)
    role = _role_from_tags(tags)
    return ChatMessage(role=role, content=memory.content)


def _role_from_tags(tags: list[str]) -> MessageRole:
    """Recover MessageRole from tags stored during put(). Defaults to USER."""
    for candidate in (MessageRole.ASSISTANT, MessageRole.SYSTEM, MessageRole.USER):
        if candidate.value in tags:
            return candidate
    return MessageRole.USER
```

- [ ] **Step 4: 更新 __init__.py 添加 NeuralMemChatMemory 导出**

写入 `integrations/llamaindex/neuralmem_llamaindex/__init__.py`：

```python
from neuralmem_llamaindex.chat_memory import NeuralMemChatMemory
from neuralmem_llamaindex.retriever import NeuralMemRetriever

__version__ = "0.2.0"
__all__ = ["NeuralMemRetriever", "NeuralMemChatMemory"]
```

- [ ] **Step 5: 运行 chat_memory 测试确认通过**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
pytest tests/test_chat_memory.py -v
```

期望：`8 passed`

- [ ] **Step 6: 运行全部测试**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
pytest tests/ -v
```

期望：`16 passed`

- [ ] **Step 7: ruff lint**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
ruff check neuralmem_llamaindex/
```

期望：`All checks passed!`

- [ ] **Step 8: Commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/llamaindex/tests/test_chat_memory.py \
        integrations/llamaindex/neuralmem_llamaindex/chat_memory.py \
        integrations/llamaindex/neuralmem_llamaindex/__init__.py
git commit -m "feat(integrations/llamaindex): 实现 NeuralMemChatMemory"
```

---

## Task 5: 最终验证

**Files:**
- Verify: `integrations/llamaindex/neuralmem_llamaindex/__init__.py`

- [ ] **Step 1: 验证公开导入路径**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
python -c "
from neuralmem_llamaindex import NeuralMemRetriever, NeuralMemChatMemory
print('NeuralMemRetriever:', NeuralMemRetriever.__doc__[:30])
print('NeuralMemChatMemory:', NeuralMemChatMemory.__doc__[:30])
"
```

期望：两行输出，均包含对应类的 docstring 开头。

- [ ] **Step 2: 全量测试 + lint**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/llamaindex
pytest tests/ -q
ruff check neuralmem_llamaindex/
```

期望：`16 passed` + `All checks passed!`

- [ ] **Step 3: 验证 NeuralMem 核心测试未受影响**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
pytest tests/ -m "not slow" -q
```

期望：`216 passed`

- [ ] **Step 4: 验证 LangChain 集成测试未受影响**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem/integrations/langchain
pytest tests/ -q
```

期望：`14 passed`

- [ ] **Step 5: 如 __init__.py 有任何遗漏则补全并 commit**

```bash
cd /Users/zhouwen/Desktop/open_source/ai-agents/NeuralMem
git add integrations/llamaindex/neuralmem_llamaindex/__init__.py
git commit -m "feat(integrations/llamaindex): 完成 neuralmem-llamaindex v0.2.0"
```

---

## Self-Review

**Spec coverage:**
- ✅ `NeuralMemRetriever(BaseRetriever)` — Task 3
- ✅ `_retrieve(QueryBundle)` → `list[NodeWithScore]` — Task 3
- ✅ `NodeWithScore` 包含所有 metadata 字段 — Task 3 `test_node_metadata_fields`
- ✅ `NeuralMemChatMemory(BaseMemory)` — Task 4
- ✅ `put()` 调用 `remember()` 并存储 role 到 tags — Task 4
- ✅ `get(input=...)` 触发 `recall()` — Task 4 `test_get_with_input_calls_recall`
- ✅ `get(input=None)` 使用 `storage.list_memories()` — Task 4 `test_get_without_input_uses_storage`
- ✅ `reset()` 调用 `forget()` — Task 4 `test_reset_calls_forget`
- ✅ `set()` 先 reset 再 put — Task 4 `test_set_resets_then_puts`
- ✅ `get_all()` — Task 4 `test_get_all_uses_storage`
- ✅ `model_config = {"arbitrary_types_allowed": True}` — 两个类都有
- ✅ 依赖 `llama-index-core>=0.10.0` 只 — Task 1 pyproject.toml
- ✅ 14 个测试（spec Section 6 要求 6+8=14）— Tasks 3+4

**Type consistency:**
- `NeuralMemRetriever.mem: Any` → Task 3 实现，Task 3 测试均一致
- `NeuralMemChatMemory.mem: Any` → Task 4 实现，Task 4 测试均一致
- `_to_node_with_score(result: SearchResult) -> NodeWithScore` — Task 3 定义，在 `_retrieve()` 中调用
- `_result_to_chat_message(result: SearchResult) -> ChatMessage` — Task 4 定义，在 `get(input=...)` 中调用
- `_memory_to_chat_message(memory: Memory) -> ChatMessage` — Task 4 定义，在 `get(input=None)` 和 `get_all()` 中调用
- `_role_from_tags(tags: list[str]) -> MessageRole` — Task 4 定义，被两个 helper 调用
