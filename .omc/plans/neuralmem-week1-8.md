# NeuralMem Week 1-8 实现计划（Ralplan 共识版）

> 版本: v2（已纳入 Architect + Critic 两轮反馈）  
> 状态: APPROVED  
> 生成时间: 2026-04-29

---

## RALPLAN-DR 摘要

### 核心原则
1. **本地优先**: 所有核心能力在零外部 API 依赖下可用
2. **接口先行**: Phase 1 冻结 `core/protocols.py` + `core/types.py` 契约，Phase 2 并行模块只 import protocols
3. **同步优先**: 核心 API 全同步，MCP 层统一 `asyncio.to_thread` 适配
4. **存储/检索分离**: 写路径（提取→存储）与读路径（四策略融合）解耦
5. **渐进式骨架**: Week 9-10 模块以 ABC + NotImplementedError stub 先行落地

### 决策驱动（Top 3）
1. 零冷启动门槛（FastEmbed + SQLite，pip install 即可运行）
2. 后向兼容性（stub 接口稳定，避免 Week 9 重写）
3. 测试可信度（pytest.mark.stub 排除假绿，contract tests 验证 Protocol）

---

## ADR 记录

### ADR-001: SQLite + sqlite-vec 统一存储
- **决策**: 使用 SQLite（主存储）+ sqlite-vec（向量扩展）
- **驱动**: 零部署/单文件/本地优先
- **替代**: ChromaDB（依赖重）、Qdrant（需独立服务）
- **后果**: 受限于 sqlite-vec 索引能力；10k 内可接受

### ADR-002: FastEmbed (all-MiniLM-L6-v2) 本地 Embedding
- **决策**: FastEmbed ONNX，无需 PyTorch，无需 API Key
- **驱动**: 零依赖运行，模型本地缓存
- **替代**: sentence-transformers（PyTorch 重依赖）
- **后果**: 模型选择受限于 FastEmbed 支持列表

### ADR-003: NetworkX 内存图谱 + SQLite 持久化
- **决策**: NetworkX 内存图 + 快照序列化到 SQLite
- **驱动**: 零依赖，够用到 10 万节点
- **后果**: 100k+ 节点时内存压力增大（Week 9+ 评估迁移）

### ADR-004: sqlite-vec 加载失败回退策略 [新增]
- **决策**: 启动时检测 sqlite-vec；失败时降级到 `numpy.dot` 暴力检索
- **性能边界**: 1k 记忆 < 5ms，10k < 50ms（float32，BLAS 加速）
- **实现**:
```python
# storage/sqlite.py — 启动检测
import logging
_logger = logging.getLogger(__name__)

def _load_vec_extension(conn):
    try:
        conn.load_extension("vec0")
        return True
    except Exception as e:
        _logger.warning(
            "sqlite-vec unavailable (%s). Falling back to numpy brute-force search. "
            "Performance: OK for <10k memories.", e
        )
        return False
```

### ADR-005: 同步/异步边界策略 [新增]
- **决策**: 核心 `NeuralMem` 类和所有存储/Embedding/检索模块全同步；MCP Server 层统一用 `asyncio.to_thread` 适配
- **原因**: sqlite-vec 和 FastEmbed 都是同步 C 扩展；避免"双色函数"污染核心层
- **实现**:
```python
# mcp/server.py — asyncio.to_thread 适配模式
from mcp.server.fastmcp import FastMCP
import asyncio
from neuralmem import NeuralMem

mcp = FastMCP("NeuralMem")
_engine: NeuralMem | None = None

def get_engine() -> NeuralMem:
    global _engine
    if _engine is None:
        _engine = NeuralMem()
    return _engine

@mcp.tool()
async def remember(content: str, user_id: str = "default") -> str:
    memories = await asyncio.to_thread(
        get_engine().remember_sync, content, user_id=user_id
    )
    return f"Stored {len(memories)} memories."
```

---

## 文件清单（完整）

```
neuralmem/
├── pyproject.toml                    # Phase 0
├── .python-version                   # Phase 0
├── .github/
│   └── workflows/ci.yml              # Phase 0
├── pytest.ini                        # Phase 0
├── src/neuralmem/
│   ├── __init__.py                   # Phase 0
│   ├── py.typed                      # Phase 0
│   ├── core/
│   │   ├── __init__.py
│   │   ├── protocols.py              # Phase 1 [新增]
│   │   ├── types.py                  # Phase 1 [修订: frozen=True]
│   │   ├── config.py                 # Phase 1
│   │   ├── exceptions.py             # Phase 1
│   │   └── memory.py                 # Phase 5
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py                   # Phase 2a
│   │   └── sqlite.py                 # Phase 2a [含 numpy 回退]
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── base.py                   # Phase 2b
│   │   └── local.py                  # Phase 2b
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── extractor.py              # Phase 2c
│   │   └── llm_extractor.py          # Phase 2c [Ollama 可选]
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── entity.py                 # Phase 2d
│   │   ├── relation.py               # Phase 2d
│   │   └── knowledge_graph.py        # Phase 2d
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── semantic.py               # Phase 3
│   │   ├── keyword.py                # Phase 3
│   │   ├── graph.py                  # Phase 3
│   │   ├── temporal.py               # Phase 3
│   │   ├── fusion.py                 # Phase 3 [新增: RRFMerger]
│   │   ├── reranker.py               # Phase 3 [stub]
│   │   └── engine.py                 # Phase 3
│   ├── lifecycle/
│   │   ├── __init__.py
│   │   ├── decay.py                  # Phase 4 [stub: ABC]
│   │   ├── consolidation.py          # Phase 4 [stub: ABC]
│   │   └── importance.py             # Phase 4 [stub: ABC]
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py                 # Phase 6
│   │   ├── tools.py                  # Phase 6
│   │   └── resources.py              # Phase 6
│   └── cli/
│       ├── __init__.py
│       └── main.py                   # Phase 7
└── tests/
    ├── conftest.py                   # Phase 0
    ├── contract/                     # Phase 2 [新增]
    │   ├── test_storage_protocol.py
    │   ├── test_embedder_protocol.py
    │   └── test_extractor_protocol.py
    ├── unit/
    │   ├── test_types.py             # Phase 1
    │   ├── test_storage.py           # Phase 2a
    │   ├── test_embedding.py         # Phase 2b
    │   ├── test_extractor.py         # Phase 2c
    │   ├── test_graph.py             # Phase 2d
    │   ├── test_fusion.py            # Phase 3
    │   ├── test_retrieval.py         # Phase 3
    │   └── test_lifecycle_stubs.py   # Phase 4
    └── integration/
        ├── test_memory_facade.py     # Phase 5
        ├── test_mcp_server.py        # Phase 6
        └── test_end_to_end.py        # Phase 7
```

---

## Phase 0: 基础设施（可直接复制的代码）

### pyproject.toml
```toml
[project]
name = "neuralmem"
version = "0.1.0"
description = "Memory as Infrastructure — Local-first, MCP-native agent memory"
readme = "README.md"
license = "AGPL-3.0-or-later"
requires-python = ">=3.10"
authors = [{ name = "NeuralMem Team" }]
keywords = ["agent", "memory", "mcp", "llm", "ai"]

dependencies = [
    "pydantic>=2.0",
    "sqlite-vec>=0.1",
    "fastembed>=0.3",
    "networkx>=3.0",
    "rank-bm25>=0.2",
    "mcp>=1.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
server = ["fastapi>=0.110", "uvicorn>=0.30"]
openai = ["openai>=1.0"]
ollama = ["httpx>=0.27"]
reranker = ["sentence-transformers>=3.0"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
neuralmem = "neuralmem.cli.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "stub: marks tests for stub/placeholder implementations (deselect with '-m not stub')",
    "integration: marks integration tests requiring real storage",
    "slow: marks slow tests",
]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true
```

### pytest.ini（独立配置文件）
```ini
[pytest]
asyncio_mode = auto
markers =
    stub: marks stub implementations (deselect with '-m not stub')
    integration: integration tests
    slow: slow tests
addopts = --tb=short -q
```

### .github/workflows/ci.yml
```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: ${{ matrix.python-version }}
      
      # FastEmbed 模型缓存（避免每次 CI 重新下载 ~80MB 模型）
      - uses: actions/cache@v4
        with:
          path: ~/.cache/fastembed
          key: fastembed-model-MiniLM-L6-v2-v1
          restore-keys: fastembed-model-
      
      - run: uv sync --extra dev
      - run: uv run ruff check .
      - run: uv run mypy src/neuralmem
      
      # 默认跳过 stub 测试（-m "not stub"）
      - run: uv run pytest -m "not stub" --cov=neuralmem --cov-fail-under=80
```

---

## Phase 1: 核心契约层（可直接复制的代码）

### core/protocols.py [新增]
```python
"""NeuralMem Protocol 接口定义 — 所有模块只依赖此文件，消除循环导入"""
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable
from .types import Memory, Entity, Relation, SearchResult


@runtime_checkable
class StorageProtocol(Protocol):
    def add(self, memory: Memory) -> str: ...
    def get(self, memory_id: str) -> Memory | None: ...
    def update(self, memory_id: str, **kwargs) -> None: ...
    def delete(self, memory_id: str) -> bool: ...
    def vector_search(
        self, vector: list[float], k: int, user_id: str | None = None
    ) -> list[tuple[str, float]]: ...
    def keyword_search(
        self, query: str, k: int, user_id: str | None = None
    ) -> list[tuple[str, float]]: ...
    def get_stats(self) -> dict: ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    dimension: int
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...
    def encode_one(self, text: str) -> list[float]: ...


@runtime_checkable
class ExtractorProtocol(Protocol):
    def extract(
        self, text: str
    ) -> tuple[list[Entity], list[Relation]]: ...


@runtime_checkable
class GraphStoreProtocol(Protocol):
    def add_entity(self, entity: Entity) -> None: ...
    def add_relation(self, relation: Relation) -> None: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def neighbors(self, entity_id: str, depth: int = 1) -> list[Entity]: ...
    def find_by_name(self, name: str) -> list[Entity]: ...
    def memory_ids_for_entities(self, entity_ids: list[str]) -> list[str]: ...


@runtime_checkable
class LifecycleProtocol(Protocol):
    def apply_decay(self, user_id: str | None = None) -> int: ...
    def remove_forgotten(self, user_id: str | None = None) -> int: ...
    def consolidate(self, user_id: str | None = None) -> int: ...
```

### core/types.py（关键片段）
```python
"""NeuralMem 核心数据模型 — frozen=True 确保跨模块类型契约稳定"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class MemoryScope(str, Enum):
    USER = "user"
    AGENT = "agent"
    SESSION = "session"
    SHARED = "shared"


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    entity_type: str = "unknown"
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    attributes: dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)


class Relation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    scope: MemoryScope = MemoryScope.USER
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = Field(default_factory=tuple)
    source: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    entity_ids: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, ge=0)
    # 向量不参与序列化，不包含在 frozen 契约中
    embedding: list[float] | None = Field(default=None, exclude=True, frozen=False)


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    memory: Memory
    score: float = Field(ge=0.0, le=1.0)
    retrieval_method: str
    explanation: str | None = None
```

### lifecycle/decay.py（stub 模式）
```python
"""DecayManager — stub 实现（Week 9 完整实现）"""
import pytest
from abc import ABC, abstractmethod
from neuralmem.core.protocols import LifecycleProtocol


class _LifecycleBase(ABC):
    """Week 9 将继承此基类实现真实逻辑"""

    @abstractmethod
    def apply_decay(self, user_id: str | None = None) -> int:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def remove_forgotten(self, user_id: str | None = None) -> int:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def consolidate(self, user_id: str | None = None) -> int:
        raise NotImplementedError  # pragma: no cover


class DecayManager(_LifecycleBase):
    """
    记忆衰减管理器（Week 1-8 stub）。
    
    # TODO(week-9): 实现谢宾斯基遗忘曲线 + 访问频次加权
    接口稳定，不会在 Week 9 变更。
    """

    def apply_decay(self, user_id: str | None = None) -> int:
        return 0  # stub: no decay applied

    def remove_forgotten(self, user_id: str | None = None) -> int:
        return 0  # stub: nothing removed

    def consolidate(self, user_id: str | None = None) -> int:
        return 0  # stub: nothing consolidated


# 确保 DecayManager 满足 LifecycleProtocol
assert isinstance(DecayManager(), LifecycleProtocol)
```

### tests/unit/test_lifecycle_stubs.py
```python
"""验证 stub 接口签名正确，标记为 stub 在 CI 中可被排除"""
import pytest
from neuralmem.lifecycle.decay import DecayManager
from neuralmem.lifecycle.consolidation import MemoryConsolidator
from neuralmem.core.protocols import LifecycleProtocol

pytestmark = pytest.mark.stub  # CI: pytest -m "not stub" 跳过此文件


def test_decay_manager_satisfies_protocol():
    dm = DecayManager()
    assert isinstance(dm, LifecycleProtocol)


def test_decay_returns_zero_stub():
    dm = DecayManager()
    assert dm.apply_decay() == 0
    assert dm.remove_forgotten() == 0
    assert dm.consolidate() == 0


def test_consolidator_satisfies_protocol():
    mc = MemoryConsolidator()
    assert isinstance(mc, LifecycleProtocol)
```

---

## Phase 2: 并行子模块（执行策略）

### 并行任务分配
Phase 2 四个子模块只依赖 `core/protocols.py` + `core/types.py`，可完全并行：

| 任务 | 文件 | 依赖 |
|------|------|------|
| Task-2a | storage/base.py, storage/sqlite.py | protocols.StorageProtocol |
| Task-2b | embedding/base.py, embedding/local.py | protocols.EmbedderProtocol |
| Task-2c | extraction/extractor.py, extraction/llm_extractor.py | protocols.ExtractorProtocol |
| Task-2d | graph/entity.py, graph/relation.py, graph/knowledge_graph.py | protocols.GraphStoreProtocol |

### 契约测试模板（tests/contract/）
```python
# tests/contract/test_storage_protocol.py
from neuralmem.core.protocols import StorageProtocol
from neuralmem.storage.sqlite import SQLiteStorage
from neuralmem.core.config import NeuralMemConfig

def test_sqlite_storage_satisfies_protocol(tmp_path):
    config = NeuralMemConfig(db_path=str(tmp_path / "test.db"))
    storage = SQLiteStorage(config)
    assert isinstance(storage, StorageProtocol), (
        "SQLiteStorage must satisfy StorageProtocol"
    )
```

---

## Phase 3: 检索引擎（RRF 融合独立模块）

### retrieval/fusion.py [新增]
```python
"""RRF (Reciprocal Rank Fusion) 融合器 — 独立可测试模块"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RankedItem:
    id: str
    score: float
    method: str


class RRFMerger:
    """
    倒数排名融合 (Reciprocal Rank Fusion)。
    RRF(d) = Σ 1/(k + rank_i(d))，k=60 为经典默认值。
    """

    def __init__(self, k: int = 60):
        self.k = k

    def merge(
        self,
        ranked_lists: dict[str, list[RankedItem]],
    ) -> list[tuple[str, float]]:
        """
        Args:
            ranked_lists: {strategy_name: [RankedItem sorted by score desc]}
        Returns:
            List of (id, normalized_rrf_score) sorted by score desc
        """
        rrf_scores: dict[str, float] = {}
        for _method, items in ranked_lists.items():
            for rank, item in enumerate(items, start=1):
                rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + 1.0 / (self.k + rank)

        if not rrf_scores:
            return []

        max_score = max(rrf_scores.values())
        if max_score > 0:
            rrf_scores = {k: v / max_score for k, v in rrf_scores.items()}

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

---

## Phase 各阶段验证命令

| Phase | 验证命令 | 通过标准 |
|-------|---------|---------|
| 0 基础设施 | `uv sync --extra dev && uv run pytest --collect-only` | 无导入错误 |
| 1 类型契约 | `uv run pytest tests/unit/test_types.py -v && uv run mypy src/neuralmem/core/` | 全通过，mypy 无错 |
| 2 子模块 | `uv run pytest tests/contract/ tests/unit/test_storage.py tests/unit/test_embedding.py tests/unit/test_extractor.py tests/unit/test_graph.py -v` | 全通过 |
| 3 检索引擎 | `uv run pytest tests/unit/test_fusion.py tests/unit/test_retrieval.py -v` | 全通过 |
| 4 生命周期stub | `uv run pytest -m stub -v` | 接口验证通过（仅开发时运行） |
| 5 主类 | `uv run pytest tests/integration/test_memory_facade.py -v` | 全通过 |
| 6 MCP Server | `uv run pytest tests/integration/test_mcp_server.py -v` | 全通过 |
| 7 CLI | `uv run neuralmem --help && uv run pytest tests/ -m "not stub" --cov=neuralmem --cov-fail-under=80` | 覆盖率 ≥ 80% |

### CI 默认命令
```bash
# CI 排除 stub，要求 80% 覆盖率
uv run pytest -m "not stub" --cov=neuralmem --cov-fail-under=80 -v
```

---

## 验收标准（Week 1-8 完成）

- [ ] `uv pip install -e .` 成功
- [ ] `python -c "from neuralmem import NeuralMem; m = NeuralMem(); print('OK')"` 可运行
- [ ] `neuralmem --help` 显示帮助
- [ ] 3行代码示例运行无误（remember + recall + print）
- [ ] `pytest -m "not stub" --cov-fail-under=80` 全通过
- [ ] MCP Server 可通过 `neuralmem mcp` 启动
- [ ] Claude Desktop MCP 配置可正常连接

---

## 执行指令

执行时使用 dispatching-parallel-agents 策略：
- **串行关键路径**: Phase 0 → Phase 1 → Phase 5 → Phase 6 → Phase 7
- **可并行**: Phase 2 的 Task-2a/2b/2c/2d（4个并行 executor）
- **依赖 Phase 2**: Phase 3 和 Phase 4（等 Phase 2 完成后并行）
