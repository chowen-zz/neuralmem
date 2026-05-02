# NeuralMem — Agent Memory 开源项目落地方案

> **项目代号**: NeuralMem（候选：NeuralMem / AgentVault / Mnemos / RecallDB）  
> **Slogan**: "Memory as Infrastructure — Local-first, MCP-native, Enterprise-ready"  
> **一句话**: Agent 记忆领域的 SQLite —— 零依赖安装，本地优先，企业可扩展

---

## 第一部分：项目定义与顶层设计

### 1.1 项目核心目标

做一个**真正零依赖、pip install 即用**的 Agent 长期记忆系统，满足：

| 用户 | 核心诉求 | 我们给的答案 |
|------|---------|------------|
| 个人开发者 | Agent 跨会话记住我说过什么 | `pip install neuralmem` → 3行代码接入 |
| 独立开发者 | 给我的 AI 产品加记忆 | Python/TS SDK + REST API，AGPL 免费 |
| 创业公司 | 不想被 Mem0 的 $249/月 Pro 绑架 | 图谱、时序、四策略检索全部开源 |
| 大企业 | 数据不能出我的网络 | 完全本地部署 + 商业许可 + HIPAA路径 |

### 1.2 技术选型决策表

| 决策点 | 选择 | 理由 | 备选方案 |
|--------|------|------|---------|
| **主语言** | Python 3.10+ | Agent 生态主力语言，与 MCP SDK 一致 | Rust（性能更优但生态门槛高） |
| **核心存储** | SQLite + sqlite-vec | 零依赖、嵌入式、性能够用（5ms 检索） | DuckDB（分析场景更强但向量支持弱） |
| **本地Embedding** | FastEmbed (ONNX) | 无需 GPU、无需 API Key、all-MiniLM-L6-v2 | sentence-transformers（更大但更准） |
| **图谱** | NetworkX（内存） | 零依赖、够用到 10万节点 | rustworkx（更快）→ Neo4j（企业版） |
| **MCP 框架** | FastMCP (官方 SDK) | 标准实现，支持 stdio/SSE/HTTP | 自己实现 JSON-RPC（不必要） |
| **API 框架** | FastAPI | 异步、自动文档、生态成熟 | Litestar（更快但社区小） |
| **包管理** | uv + pyproject.toml | 2026年 Python 标准，比 pip 快 10x | poetry（成熟但慢） |
| **测试** | pytest + pytest-asyncio | 标准选择 | — |
| **CI/CD** | GitHub Actions | 免费、社区标准 | — |
| **文档** | MkDocs Material | 好看、搜索好、Python 生态标配 | — |

---

## 第二部分：项目代码结构

### 2.1 Monorepo 目录结构

```
neuralmem/
├── LICENSE                          # AGPL-3.0
├── LICENSE-COMMERCIAL.md            # 商业许可说明
├── README.md                        # 项目首页（英文）
├── README.zh-CN.md                  # 中文 README
├── pyproject.toml                   # 包定义 + 依赖
├── Makefile                         # 常用命令快捷方式
├── Dockerfile                       # 生产镜像
├── docker-compose.yml               # 一键启动（API + Dashboard）
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                   # 测试 + lint
│   │   ├── release.yml              # PyPI 发布
│   │   └── benchmark.yml            # 每周自动跑 LongMemEval
│   └── ISSUE_TEMPLATE/
│
├── src/neuralmem/                   # 核心 Python 包
│   ├── __init__.py                  # 版本号 + 公开 API
│   ├── py.typed                     # PEP 561 类型标记
│   │
│   ├── core/                        # 核心引擎
│   │   ├── __init__.py
│   │   ├── memory.py                # MemoryEngine 主类
│   │   ├── types.py                 # 数据模型 (Pydantic)
│   │   ├── config.py                # 配置管理
│   │   └── exceptions.py            # 自定义异常
│   │
│   ├── storage/                     # 存储层（可插拔）
│   │   ├── __init__.py
│   │   ├── base.py                  # StorageBackend 抽象基类
│   │   ├── sqlite.py                # SQLite + sqlite-vec 实现
│   │   ├── postgres.py              # PostgreSQL 实现 [Team+]
│   │   └── redis.py                 # Redis 实现 [Team+]
│   │
│   ├── embedding/                   # Embedding 层（可插拔）
│   │   ├── __init__.py
│   │   ├── base.py                  # EmbeddingProvider 抽象基类
│   │   ├── local.py                 # FastEmbed/ONNX 本地模型
│   │   ├── openai.py                # OpenAI text-embedding-3-small
│   │   └── ollama.py                # Ollama 本地模型
│   │
│   ├── extraction/                  # 记忆提取层
│   │   ├── __init__.py
│   │   ├── extractor.py             # 实体/关系/事实提取
│   │   ├── entity_resolver.py       # 实体消歧 ("Alice" = "我同事Alice")
│   │   └── llm_extractor.py         # LLM 辅助提取（可选增强）
│   │
│   ├── retrieval/                   # 检索层（核心差异化）
│   │   ├── __init__.py
│   │   ├── engine.py                # 多策略检索引擎
│   │   ├── semantic.py              # 语义向量搜索
│   │   ├── keyword.py               # BM25 关键词搜索
│   │   ├── graph.py                 # 图谱遍历搜索
│   │   ├── temporal.py              # 时序过滤搜索
│   │   └── reranker.py              # Cross-Encoder 重排序
│   │
│   ├── graph/                       # 知识图谱
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py       # 图谱管理
│   │   ├── entity.py                # 实体节点
│   │   └── relation.py              # 关系边
│   │
│   ├── lifecycle/                   # 记忆生命周期
│   │   ├── __init__.py
│   │   ├── decay.py                 # 遗忘曲线 + 访问衰减
│   │   ├── consolidation.py         # 记忆合并/去重
│   │   └── importance.py            # 重要性评分
│   │
│   ├── mcp/                         # MCP Server 实现
│   │   ├── __init__.py
│   │   ├── server.py                # FastMCP Server 主入口
│   │   ├── tools.py                 # MCP Tool 定义
│   │   └── resources.py             # MCP Resource 定义
│   │
│   ├── api/                         # REST API（可选启动）
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI 应用
│   │   ├── routes/
│   │   │   ├── memories.py          # /memories CRUD
│   │   │   ├── search.py            # /search 检索
│   │   │   ├── graph.py             # /graph 图谱查询
│   │   │   └── health.py            # /health 健康检查
│   │   └── middleware/
│   │       ├── auth.py              # API Key / JWT 认证
│   │       └── rate_limit.py        # 速率限制
│   │
│   └── cli/                         # 命令行工具
│       ├── __init__.py
│       └── main.py                  # neuralmem serve / neuralmem mcp / ...
│
├── sdks/                            # 多语言 SDK
│   ├── typescript/                  # npm 包 @neuralmem/sdk
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── index.ts
│   │   │   └── client.ts
│   │   └── tsconfig.json
│   └── go/                          # Go module（Phase 3）
│
├── dashboard/                       # Web 管理面板（React）
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── MemoryBrowser.tsx    # 记忆浏览器
│   │   │   ├── GraphView.tsx        # 图谱可视化
│   │   │   ├── Analytics.tsx        # 使用分析
│   │   │   └── Settings.tsx         # 配置管理
│   │   └── components/
│   └── vite.config.ts
│
├── benchmarks/                      # 基准测试
│   ├── longmemeval/                 # LongMemEval 适配
│   ├── convomem/                    # ConvoMem 适配
│   └── run_benchmark.py
│
├── tests/                           # 测试套件
│   ├── unit/
│   │   ├── test_memory.py
│   │   ├── test_retrieval.py
│   │   ├── test_graph.py
│   │   └── test_lifecycle.py
│   ├── integration/
│   │   ├── test_mcp_server.py
│   │   ├── test_api.py
│   │   └── test_end_to_end.py
│   └── conftest.py
│
├── docs/                            # MkDocs 文档
│   ├── mkdocs.yml
│   ├── docs/
│   │   ├── index.md                 # 首页
│   │   ├── quickstart.md            # 5分钟快速开始
│   │   ├── concepts/
│   │   │   ├── memory-types.md
│   │   │   ├── retrieval.md
│   │   │   └── architecture.md
│   │   ├── guides/
│   │   │   ├── claude-code.md       # Claude Code 集成
│   │   │   ├── cursor.md            # Cursor 集成
│   │   │   ├── langchain.md         # LangChain 集成
│   │   │   └── crewai.md            # CrewAI 集成
│   │   ├── api-reference/
│   │   └── enterprise/
│   │       ├── deployment.md
│   │       ├── security.md
│   │       └── licensing.md
│
├── examples/                        # 示例代码
│   ├── 01_basic_memory.py           # 最简使用
│   ├── 02_mcp_claude_code.py        # Claude Code 集成
│   ├── 03_langchain_agent.py        # LangChain Agent
│   ├── 04_multi_agent.py            # 多 Agent 共享记忆
│   └── 05_custom_backend.py         # 自定义存储后端
│
└── scripts/                         # 运维脚本
    ├── migrate.py                   # 数据迁移
    ├── import_mem0.py               # 从 Mem0 导入
    └── export.py                    # 数据导出
```

---

## 第三部分：核心代码骨架

### 3.1 数据模型 (`core/types.py`)

```python
"""NeuralMem 核心数据模型"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """四类认知记忆模型"""
    EPISODIC = "episodic"        # 事件/交互记录
    SEMANTIC = "semantic"        # 事实/偏好/知识
    PROCEDURAL = "procedural"    # 流程/SOP/最佳实践
    WORKING = "working"          # 当前会话上下文


class MemoryScope(str, Enum):
    """记忆归属范围"""
    USER = "user"           # 归属特定用户
    AGENT = "agent"         # 归属特定 Agent
    SESSION = "session"     # 归属特定会话
    SHARED = "shared"       # 团队共享


class Entity(BaseModel):
    """知识图谱实体"""
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    entity_type: str = "unknown"    # person, project, tool, concept...
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)


class Relation(BaseModel):
    """知识图谱关系"""
    source_id: str
    target_id: str
    relation_type: str              # "prefers", "works_on", "uses"...
    weight: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """一条记忆"""
    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str                           # 记忆内容（自然语言）
    memory_type: MemoryType = MemoryType.SEMANTIC
    scope: MemoryScope = MemoryScope.USER
    
    # 归属
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    
    # 元数据
    tags: list[str] = Field(default_factory=list)
    source: str | None = None              # 来源（哪次对话）
    importance: float = 0.5                # 重要性 0-1
    
    # 关联实体
    entity_ids: list[str] = Field(default_factory=list)
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    
    # 向量（内部使用，不序列化到API）
    embedding: list[float] | None = Field(default=None, exclude=True)


class SearchResult(BaseModel):
    """搜索结果"""
    memory: Memory
    score: float                  # 综合相关性分数 0-1
    retrieval_method: str         # 哪种检索策略命中的
    explanation: str | None = None


class SearchQuery(BaseModel):
    """搜索请求"""
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    memory_types: list[MemoryType] | None = None
    tags: list[str] | None = None
    time_range: tuple[datetime, datetime] | None = None
    limit: int = 10
    min_score: float = 0.3
```

### 3.2 核心引擎 (`core/memory.py`)

```python
"""NeuralMem 核心记忆引擎 — 3行代码即用的设计"""
from __future__ import annotations
from neuralmem.core.types import (
    Memory, MemoryType, SearchQuery, SearchResult, MemoryScope
)
from neuralmem.core.config import NeuralMemConfig
from neuralmem.storage.base import StorageBackend
from neuralmem.storage.sqlite import SQLiteStorage
from neuralmem.embedding.base import EmbeddingProvider
from neuralmem.embedding.local import LocalEmbedding
from neuralmem.extraction.extractor import MemoryExtractor
from neuralmem.retrieval.engine import RetrievalEngine
from neuralmem.graph.knowledge_graph import KnowledgeGraph
from neuralmem.lifecycle.decay import DecayManager


class NeuralMem:
    """
    Agent 记忆引擎的统一入口。
    
    最简使用（3行代码）:
        >>> from neuralmem import NeuralMem
        >>> mem = NeuralMem()
        >>> mem.remember("用户偏好用 TypeScript 写前端")
        >>> results = mem.recall("用户的技术偏好是什么？")
    
    自定义配置:
        >>> mem = NeuralMem(
        ...     db_path="./my_memory.db",
        ...     embedding_provider="openai",
        ...     openai_api_key="sk-..."
        ... )
    """
    
    def __init__(
        self,
        db_path: str = "~/.neuralmem/memory.db",
        config: NeuralMemConfig | None = None,
        storage: StorageBackend | None = None,
        embedding: EmbeddingProvider | None = None,
    ):
        self.config = config or NeuralMemConfig(db_path=db_path)
        self.storage = storage or SQLiteStorage(self.config)
        self.embedding = embedding or LocalEmbedding(self.config)
        self.extractor = MemoryExtractor(self.config)
        self.graph = KnowledgeGraph(self.storage)
        self.retrieval = RetrievalEngine(
            storage=self.storage,
            embedding=self.embedding,
            graph=self.graph,
            config=self.config,
        )
        self.decay = DecayManager(self.storage, self.config)
    
    # ==================== 核心 API ====================
    
    async def remember(
        self,
        content: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> list[Memory]:
        """
        存储记忆。自动执行：
        1. 从内容中提取事实/实体/关系
        2. 生成 Embedding 向量
        3. 更新知识图谱
        4. 去重检查（与已有记忆比较）
        5. 持久化存储
        
        Args:
            content: 要记住的内容（对话、事实、任意文本）
            user_id: 用户标识
            memory_type: 记忆类型（为空则自动推断）
            tags: 自定义标签
            importance: 重要性 0-1（为空则自动评估）
        
        Returns:
            提取并存储的记忆列表（一段内容可提取多条记忆）
        """
        # 1. 提取结构化记忆
        extracted = await self.extractor.extract(
            content,
            memory_type=memory_type,
            existing_entities=await self.graph.get_entities(user_id=user_id),
        )
        
        memories = []
        for item in extracted:
            # 2. 生成 Embedding
            vector = await self.embedding.embed(item.content)
            
            # 3. 去重检查
            duplicates = await self.storage.find_similar(
                vector, user_id=user_id, threshold=0.95
            )
            if duplicates:
                # 更新已有记忆而非创建新的
                await self._merge_memory(duplicates[0], item)
                continue
            
            # 4. 构建 Memory 对象
            memory = Memory(
                content=item.content,
                memory_type=item.memory_type or memory_type or MemoryType.SEMANTIC,
                scope=MemoryScope.USER if user_id else MemoryScope.SESSION,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                tags=tags or item.tags,
                importance=importance or item.importance,
                entity_ids=item.entity_ids,
                embedding=vector,
            )
            
            # 5. 存储
            await self.storage.save_memory(memory)
            
            # 6. 更新图谱
            for entity in item.entities:
                await self.graph.upsert_entity(entity)
            for relation in item.relations:
                await self.graph.add_relation(relation)
            
            memories.append(memory)
        
        return memories
    
    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        time_range: tuple | None = None,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """
        检索相关记忆。自动使用四策略并行检索 + 重排序。
        
        Returns:
            按相关性排序的搜索结果列表
        """
        search_query = SearchQuery(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            memory_types=memory_types,
            tags=tags,
            time_range=time_range,
            limit=limit,
            min_score=min_score,
        )
        
        results = await self.retrieval.search(search_query)
        
        # 更新访问记录（用于衰减计算）
        for result in results:
            await self.storage.record_access(result.memory.id)
        
        return results
    
    async def reflect(
        self,
        topic: str,
        *,
        user_id: str | None = None,
        depth: int = 2,
    ) -> str:
        """
        对指定主题进行记忆推理和总结。
        通过多轮检索 + 图谱遍历，生成结构化的认知报告。
        """
        # 第一轮：直接检索
        direct = await self.recall(topic, user_id=user_id, limit=20)
        
        # 第二轮：基于图谱扩展
        entity_ids = set()
        for r in direct:
            entity_ids.update(r.memory.entity_ids)
        
        related_entities = await self.graph.get_neighbors(
            list(entity_ids), depth=depth
        )
        
        # 构建认知报告
        report = self._build_reflection(topic, direct, related_entities)
        return report
    
    async def forget(
        self,
        memory_id: str | None = None,
        *,
        user_id: str | None = None,
        before: datetime | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """
        删除指定记忆。支持 GDPR 合规的完全删除。
        
        Returns:
            删除的记忆数量
        """
        return await self.storage.delete_memories(
            memory_id=memory_id,
            user_id=user_id,
            before=before,
            tags=tags,
        )
    
    async def consolidate(
        self,
        user_id: str | None = None,
    ) -> dict:
        """
        后台记忆整理：衰减旧记忆、合并相似记忆、更新重要性。
        建议定期调用（如每天一次）。
        """
        stats = {
            "decayed": await self.decay.apply_decay(user_id=user_id),
            "merged": await self._merge_similar_memories(user_id=user_id),
            "removed": await self.decay.remove_forgotten(user_id=user_id),
        }
        return stats
    
    # ==================== 同步 API（便捷封装）====================
    
    def remember_sync(self, content: str, **kwargs) -> list[Memory]:
        """同步版 remember，适合简单脚本使用"""
        import asyncio
        return asyncio.run(self.remember(content, **kwargs))
    
    def recall_sync(self, query: str, **kwargs) -> list[SearchResult]:
        """同步版 recall"""
        import asyncio
        return asyncio.run(self.recall(query, **kwargs))
```

### 3.3 MCP Server (`mcp/server.py`)

```python
"""NeuralMem MCP Server — 连接任意 MCP 客户端"""
from mcp.server.fastmcp import FastMCP
from neuralmem import NeuralMem
from neuralmem.core.types import MemoryType

# 创建 MCP Server 实例
mcp = FastMCP(
    "NeuralMem",
    description="Persistent memory for AI agents. Remember, recall, reflect.",
)

# 全局记忆引擎（启动时初始化）
engine: NeuralMem | None = None


def get_engine() -> NeuralMem:
    global engine
    if engine is None:
        engine = NeuralMem()
    return engine


@mcp.tool()
async def remember(
    content: str,
    memory_type: str = "semantic",
    tags: str = "",
    user_id: str = "default",
) -> str:
    """
    Store a memory. The content will be automatically analyzed to extract
    entities, relationships, and key facts. Memories persist across sessions.
    
    Args:
        content: What to remember (natural language text, facts, preferences)
        memory_type: One of: semantic (facts/prefs), episodic (events),
                     procedural (workflows), working (current session)
        tags: Comma-separated tags for organization
        user_id: User identifier for memory scoping
    
    Returns:
        Confirmation with number of memories stored
    """
    mem = get_engine()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    mt = MemoryType(memory_type) if memory_type else None
    
    memories = await mem.remember(
        content, user_id=user_id, memory_type=mt, tags=tag_list
    )
    
    return f"Stored {len(memories)} memories. IDs: {[m.id[:8] for m in memories]}"


@mcp.tool()
async def recall(
    query: str,
    user_id: str = "default",
    limit: int = 5,
    memory_type: str = "",
) -> str:
    """
    Retrieve relevant memories based on a query. Uses 4-strategy hybrid
    retrieval: semantic search, keyword matching, graph traversal, and
    temporal filtering, with cross-encoder reranking.
    
    Args:
        query: What to search for (natural language)
        user_id: User identifier
        limit: Maximum number of results (default: 5)
        memory_type: Filter by type (empty = all types)
    
    Returns:
        Relevant memories with scores and retrieval methods
    """
    mem = get_engine()
    types = [MemoryType(memory_type)] if memory_type else None
    
    results = await mem.recall(
        query, user_id=user_id, limit=limit, memory_types=types
    )
    
    if not results:
        return "No relevant memories found."
    
    output = []
    for i, r in enumerate(results, 1):
        output.append(
            f"{i}. [{r.score:.2f}] ({r.retrieval_method}) "
            f"[{r.memory.memory_type.value}] {r.memory.content}"
        )
    
    return "\n".join(output)


@mcp.tool()
async def reflect(
    topic: str,
    user_id: str = "default",
) -> str:
    """
    Reason over stored memories about a topic. Performs multi-hop retrieval
    and graph traversal to build a comprehensive understanding.
    
    Args:
        topic: Subject to reflect on
        user_id: User identifier
    
    Returns:
        A structured reflection summarizing what is known about the topic
    """
    mem = get_engine()
    return await mem.reflect(topic, user_id=user_id)


@mcp.tool()
async def forget(
    memory_id: str = "",
    user_id: str = "default",
    tags: str = "",
) -> str:
    """
    Delete specific memories. Supports GDPR-compliant complete deletion.
    
    Args:
        memory_id: Specific memory ID to delete (empty = use other filters)
        user_id: Delete all memories for this user
        tags: Delete memories with these tags (comma-separated)
    """
    mem = get_engine()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    
    count = await mem.forget(
        memory_id=memory_id or None,
        user_id=user_id,
        tags=tag_list,
    )
    
    return f"Deleted {count} memories."


@mcp.tool()
async def consolidate(user_id: str = "default") -> str:
    """
    Run background memory maintenance: decay old memories, merge duplicates,
    update importance scores. Call periodically (e.g., daily).
    """
    mem = get_engine()
    stats = await mem.consolidate(user_id=user_id)
    return (
        f"Consolidation complete: "
        f"{stats['decayed']} decayed, "
        f"{stats['merged']} merged, "
        f"{stats['removed']} removed."
    )


# MCP Resource: 暴露记忆统计信息
@mcp.resource("neuralmem://stats/{user_id}")
async def memory_stats(user_id: str) -> str:
    """Get memory statistics for a user"""
    mem = get_engine()
    stats = await mem.storage.get_stats(user_id=user_id)
    return (
        f"Total memories: {stats['total']}\n"
        f"By type: {stats['by_type']}\n"
        f"Entities: {stats['entity_count']}\n"
        f"Relations: {stats['relation_count']}"
    )


def main():
    """CLI 入口点"""
    import sys
    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"
    elif "--sse" in sys.argv:
        transport = "sse"
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
```

### 3.4 四策略检索引擎 (`retrieval/engine.py`)

```python
"""多策略混合检索引擎 — NeuralMem 的核心差异化"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from neuralmem.core.types import SearchQuery, SearchResult, Memory
from neuralmem.storage.base import StorageBackend
from neuralmem.embedding.base import EmbeddingProvider
from neuralmem.graph.knowledge_graph import KnowledgeGraph
from neuralmem.core.config import NeuralMemConfig


@dataclass
class StrategyResult:
    """单策略检索结果"""
    memory_id: str
    score: float
    method: str


class RetrievalEngine:
    """
    四策略并行检索 + Cross-Encoder 重排序
    
    策略:
    1. Semantic Search  - 向量相似度（捕获语义接近的记忆）
    2. BM25 Keyword     - 关键词匹配（捕获精确术语）
    3. Graph Traversal  - 图谱遍历（捕获关系链上的记忆）
    4. Temporal Filter   - 时序加权（近期记忆权重更高）
    
    所有策略的结果经过倒数排名融合 (RRF) 后，
    由 Cross-Encoder 做最终重排序。
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        embedding: EmbeddingProvider,
        graph: KnowledgeGraph,
        config: NeuralMemConfig,
    ):
        self.storage = storage
        self.embedding = embedding
        self.graph = graph
        self.config = config
        self._reranker = None  # 延迟加载
    
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """执行四策略并行检索"""
        
        # 1. 并行执行四个检索策略
        query_vector = await self.embedding.embed(query.query)
        
        strategies = await asyncio.gather(
            self._semantic_search(query, query_vector),
            self._keyword_search(query),
            self._graph_search(query),
            self._temporal_search(query, query_vector),
            return_exceptions=True,
        )
        
        # 2. 收集所有候选（忽略失败的策略）
        all_candidates: dict[str, list[StrategyResult]] = {}
        for strategy_results in strategies:
            if isinstance(strategy_results, Exception):
                continue
            for result in strategy_results:
                if result.memory_id not in all_candidates:
                    all_candidates[result.memory_id] = []
                all_candidates[result.memory_id].append(result)
        
        if not all_candidates:
            return []
        
        # 3. 倒数排名融合 (Reciprocal Rank Fusion)
        rrf_scores = self._rrf_fusion(all_candidates)
        
        # 4. 取 Top-K 候选
        top_k = min(query.limit * 3, len(rrf_scores))  # 取3倍做重排序
        top_candidates = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        # 5. 加载完整 Memory 对象
        memories = {}
        for memory_id, _ in top_candidates:
            memory = await self.storage.get_memory(memory_id)
            if memory:
                memories[memory_id] = memory
        
        # 6. Cross-Encoder 重排序（如果可用）
        if self.config.enable_reranker and memories:
            reranked = await self._rerank(query.query, memories)
        else:
            reranked = [
                (mid, rrf_scores[mid]) for mid, _ in top_candidates
                if mid in memories
            ]
        
        # 7. 构建最终结果
        results = []
        for memory_id, score in reranked[:query.limit]:
            if score < query.min_score:
                continue
            
            # 确定主要检索方法
            methods = all_candidates.get(memory_id, [])
            primary_method = max(methods, key=lambda x: x.score).method if methods else "unknown"
            
            results.append(SearchResult(
                memory=memories[memory_id],
                score=score,
                retrieval_method=primary_method,
            ))
        
        return results
    
    async def _semantic_search(
        self, query: SearchQuery, vector: list[float]
    ) -> list[StrategyResult]:
        """策略1: 向量语义搜索"""
        raw = await self.storage.vector_search(
            vector=vector,
            user_id=query.user_id,
            memory_types=query.memory_types,
            limit=query.limit * 2,
        )
        return [
            StrategyResult(memory_id=mid, score=score, method="semantic")
            for mid, score in raw
        ]
    
    async def _keyword_search(
        self, query: SearchQuery
    ) -> list[StrategyResult]:
        """策略2: BM25 关键词搜索"""
        raw = await self.storage.keyword_search(
            query=query.query,
            user_id=query.user_id,
            memory_types=query.memory_types,
            limit=query.limit * 2,
        )
        return [
            StrategyResult(memory_id=mid, score=score, method="keyword")
            for mid, score in raw
        ]
    
    async def _graph_search(
        self, query: SearchQuery
    ) -> list[StrategyResult]:
        """策略3: 知识图谱遍历搜索"""
        # 从查询中提取实体
        entities = await self.graph.find_entities(query.query)
        if not entities:
            return []
        
        # 遍历图谱找到相关记忆
        related_memory_ids = await self.graph.traverse_for_memories(
            entity_ids=[e.id for e in entities],
            depth=2,
            user_id=query.user_id,
        )
        
        return [
            StrategyResult(memory_id=mid, score=score, method="graph")
            for mid, score in related_memory_ids
        ]
    
    async def _temporal_search(
        self, query: SearchQuery, vector: list[float]
    ) -> list[StrategyResult]:
        """策略4: 时序加权搜索（近期记忆加权 + 时间范围过滤）"""
        raw = await self.storage.temporal_search(
            vector=vector,
            user_id=query.user_id,
            time_range=query.time_range,
            recency_weight=self.config.recency_weight,
            limit=query.limit * 2,
        )
        return [
            StrategyResult(memory_id=mid, score=score, method="temporal")
            for mid, score in raw
        ]
    
    def _rrf_fusion(
        self, candidates: dict[str, list[StrategyResult]], k: int = 60
    ) -> dict[str, float]:
        """
        倒数排名融合 (Reciprocal Rank Fusion)
        RRF(d) = Σ 1/(k + rank_i(d))
        """
        # 按策略分组并排序
        strategy_rankings: dict[str, list[str]] = {}
        for mid, results in candidates.items():
            for r in results:
                if r.method not in strategy_rankings:
                    strategy_rankings[r.method] = []
                strategy_rankings[r.method].append((mid, r.score))
        
        # 对每个策略按分数排序
        for method in strategy_rankings:
            strategy_rankings[method].sort(key=lambda x: x[1], reverse=True)
        
        # 计算 RRF 分数
        rrf_scores: dict[str, float] = {}
        for method, ranked_list in strategy_rankings.items():
            for rank, (mid, _) in enumerate(ranked_list, 1):
                if mid not in rrf_scores:
                    rrf_scores[mid] = 0.0
                rrf_scores[mid] += 1.0 / (k + rank)
        
        # 归一化到 0-1
        if rrf_scores:
            max_score = max(rrf_scores.values())
            if max_score > 0:
                rrf_scores = {k: v / max_score for k, v in rrf_scores.items()}
        
        return rrf_scores
    
    async def _rerank(
        self, query: str, memories: dict[str, Memory]
    ) -> list[tuple[str, float]]:
        """Cross-Encoder 重排序"""
        if self._reranker is None:
            from neuralmem.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker()
        
        pairs = [(query, m.content) for m in memories.values()]
        scores = await self._reranker.score(pairs)
        
        ranked = list(zip(memories.keys(), scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
```

### 3.5 pyproject.toml

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
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: GNU Affero General Public License v3",
    "Programming Language :: Python :: 3",
]

# 核心依赖：尽量少，保持零外部服务依赖
dependencies = [
    "pydantic>=2.0",
    "sqlite-vec>=0.1",       # SQLite 向量扩展
    "fastembed>=0.3",        # ONNX 本地 Embedding
    "networkx>=3.0",         # 内存级图谱
    "rank-bm25>=0.2",        # BM25 关键词搜索
    "mcp>=1.0",              # MCP SDK
]

[project.optional-dependencies]
# MCP Server 模式
server = ["fastapi>=0.110", "uvicorn>=0.30", "httpx>=0.27"]
# 云端 Embedding Provider
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.30"]
ollama = ["ollama>=0.3"]
# 企业级存储后端
postgres = ["asyncpg>=0.29", "pgvector>=0.3"]
redis = ["redis>=5.0"]
neo4j = ["neo4j>=5.0"]
# Cross-Encoder 重排序
reranker = ["sentence-transformers>=3.0"]
# 开发依赖
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4", "mypy>=1.10"]
# 全部安装
all = ["neuralmem[server,openai,reranker]"]

[project.scripts]
neuralmem = "neuralmem.cli.main:main"

[project.entry-points."mcp.servers"]
neuralmem = "neuralmem.mcp.server:mcp"

[project.urls]
Homepage = "https://github.com/neuralmem/neuralmem"
Documentation = "https://docs.neuralmem.dev"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py310"
line-length = 100
```

---

## 第四部分：配置与部署

### 4.1 MCP 客户端配置示例

**Claude Code / Claude Desktop (`claude_desktop_config.json`)**:

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"],
      "env": {
        "NEURALMEM_DB_PATH": "~/.neuralmem/memory.db",
        "NEURALMEM_USER_ID": "my-user"
      }
    }
  }
}
```

**Cursor (`.cursor/mcp.json`)**:

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"],
      "env": {
        "NEURALMEM_DB_PATH": "./.neuralmem/project-memory.db"
      }
    }
  }
}
```

**远程 MCP (HTTP 模式，适用于团队共享)**:

```json
{
  "mcpServers": {
    "neuralmem-team": {
      "url": "https://memory.mycompany.com/mcp",
      "headers": {
        "Authorization": "Bearer <team-api-key>"
      }
    }
  }
}
```

### 4.2 Docker Compose（一键启动 API + Dashboard）

```yaml
version: '3.8'

services:
  neuralmem-api:
    build: .
    ports:
      - "8420:8420"    # REST API
      - "9420:9420"    # MCP SSE
    volumes:
      - neuralmem_data:/data
    environment:
      - NEURALMEM_DB_PATH=/data/memory.db
      - NEURALMEM_API_KEY=${NEURALMEM_API_KEY:-}
      - NEURALMEM_ENABLE_DASHBOARD=true
    command: neuralmem serve --host 0.0.0.0 --port 8420 --mcp-port 9420
    restart: unless-stopped

  neuralmem-dashboard:
    build:
      context: ./dashboard
    ports:
      - "3420:3420"
    environment:
      - NEURALMEM_API_URL=http://neuralmem-api:8420
    depends_on:
      - neuralmem-api

volumes:
  neuralmem_data:
```

### 4.3 Dockerfile（生产镜像）

```dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[server,reranker]"

# 复制源码
COPY src/ ./src/

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8420/health || exit 1

EXPOSE 8420 9420

ENTRYPOINT ["neuralmem"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8420"]
```

---

## 第五部分：开发执行计划

### 5.1 Sprint 计划（MVP 12周）

```
Week 1-2: 基础骨架
├── 项目初始化（monorepo, CI, 文档框架）
├── core/types.py — 数据模型
├── storage/sqlite.py — SQLite + sqlite-vec 存储层
└── embedding/local.py — FastEmbed 本地 Embedding

Week 3-4: 记忆存取
├── core/memory.py — NeuralMem 主类（remember + recall 基础版）
├── extraction/extractor.py — 基于规则的实体/事实提取
├── retrieval/semantic.py — 向量语义搜索
└── tests/ — 单元测试覆盖 >80%

Week 5-6: 多策略检索
├── retrieval/keyword.py — BM25 关键词搜索
├── retrieval/engine.py — 多策略融合 (RRF)
├── graph/knowledge_graph.py — NetworkX 图谱
├── retrieval/graph.py — 图谱遍历检索
└── retrieval/temporal.py — 时序加权

Week 7-8: MCP Server
├── mcp/server.py — FastMCP 实现（5个 Tool + 1个 Resource）
├── mcp/tools.py — Tool 定义优化
├── cli/main.py — CLI 入口（neuralmem mcp / serve / ...）
├── Claude Code + Cursor 集成测试
└── 编写 Quickstart 文档

Week 9-10: 生命周期 + 增强
├── lifecycle/decay.py — 遗忘曲线
├── lifecycle/consolidation.py — 记忆合并去重
├── extraction/entity_resolver.py — 实体消歧
├── retrieval/reranker.py — Cross-Encoder 重排序
└── LongMemEval 基准测试适配

Week 11-12: 发布准备
├── README.md — 精雕细琢（英文 + 中文）
├── examples/ — 5个示例代码
├── docs/ — MkDocs 完整文档站
├── benchmarks/ — 公开基准分数
├── PyPI 发布 + GitHub Release v0.1.0
└── 撰写发布博客 + Hacker News / Reddit 投稿
```

### 5.2 质量标准

| 指标 | 目标 |
|------|------|
| 安装时间 | `pip install neuralmem` < 30秒 |
| 首次使用 | 从安装到第一条记忆存取 < 3分钟 |
| 单条记忆存储延迟 | < 100ms（本地模式，含 Embedding） |
| 检索延迟（1万条记忆） | < 50ms（四策略并行） |
| LongMemEval 分数 | > 90%（目标92%+） |
| 测试覆盖率 | > 85% |
| 零依赖外部服务 | ✅ 默认模式无需任何 API Key 或 Docker |
| 包体积 | < 100MB（含 ONNX 模型） |

---

## 第六部分：GTM 与增长策略

### 6.1 发布策略（Launch Playbook）

```
D-14: 预热期
├── Twitter/X 发布技术预告 ("Building an open-source agent memory...")
├── 在 r/LocalLLaMA, r/MachineLearning 预告
└── 联系 5-10 个 AI 开发者 KOL 给 early access

D-Day: 正式发布
├── Hacker News: "Show HN: NeuralMem – Local-first agent memory with 4-strategy retrieval"
├── Reddit: r/MachineLearning, r/LocalLLaMA, r/artificial
├── Twitter/X: 线程式介绍 (问题→方案→基准→对比→quickstart)
├── Dev.to: 技术深度文章
└── GitHub: Star + Pin README

D+7: 跟进期
├── 回应所有 GitHub Issues（24h 内）
├── 发布 "NeuralMem vs Mem0: Honest Comparison" 博客
├── 录制 YouTube 教程: "5 Min Setup: Give Claude Code Perfect Memory"
└── 提交到 Awesome MCP Servers 列表

D+30: 巩固期
├── 发布 v0.2.0（社区反馈修复 + 新功能）
├── LangChain 集成 PR
├── 每周 benchmark 自动发布
└── 开始 Office Hours (Discord/GitHub Discussions)
```

### 6.2 社区运营矩阵

| 渠道 | 内容类型 | 频率 | 目标 |
|------|---------|------|------|
| **GitHub** | Issues/Discussions/Release Notes | 每日活跃 | 开发者主阵地 |
| **Discord** | 技术讨论、支持、社区 showcase | 持续 | 社区归属感 |
| **Twitter/X** | 功能更新、基准数据、行业洞察 | 2-3次/周 | 曝光 + 影响力 |
| **Dev.to/Medium** | 技术深度文章 | 2次/月 | SEO + 开发者教育 |
| **YouTube** | 教程视频、架构讲解 | 2次/月 | 长尾流量 |
| **Hacker News** | 重大版本发布 | 按版本 | 爆发式增长 |

### 6.3 AGPL 商业转化漏斗

```
开发者发现 NeuralMem (GitHub / HN / 搜索)
         ↓
pip install neuralmem → 个人项目使用（免费，AGPL）
         ↓
在公司内部 PoC 使用（内部不对外 = AGPL OK）
         ↓
需要嵌入公司产品/SaaS → 触发 AGPL 传染条款
         ↓
选择 A: 开源自己的产品代码（几乎不可能）
选择 B: 购买商业许可证 ← 收入来源
         ↓
Team $29/月 → Business $149/月 → Enterprise 联系销售
```

**关键转化触点**:
1. README 底部明确写 "Using NeuralMem in a commercial product? See our licensing options."
2. CLI 启动时打印一行: `NeuralMem v0.x | AGPL-3.0 | Commercial license: neuralmem.dev/pricing`
3. 文档站有专门的 `/enterprise` 页面
4. 提供 30天免费商业许可试用

---

## 第七部分：关键风险 & 应对预案

| 风险 | 概率 | 应对预案 |
|------|------|---------|
| **LongMemEval 分数不达预期** | 中 | 先发布，持续优化；基准透明比分数高更重要 |
| **MCP 协议重大变更** | 低 | 紧跟 MCP SDK 更新；抽象层隔离协议依赖 |
| **sqlite-vec 不稳定** | 低 | 备选 ChromaDB（仍零 API 依赖）|
| **无人 star / 关注** | 中 | 先做内容营销铺垫；确保 DX 极致好；找 KOL 推荐 |
| **大厂（Claude/OpenAI）内置记忆越来越强** | 高 | 差异化: 跨模型、数据主权、可迁移、企业级 |
| **竞品抄袭架构** | 中 | AGPL 保护；持续快速迭代；社区才是护城河 |

---

## 第八部分：立即行动清单

### 今天就可以做的 5 件事

- [ ] **注册 GitHub 组织** `neuralmem` + 创建 repo + 选 AGPL-3.0 许可
- [ ] **初始化项目骨架**: `uv init neuralmem && uv add pydantic sqlite-vec fastembed networkx rank-bm25 mcp`
- [ ] **实现最小闭环**: `NeuralMem.remember()` + `NeuralMem.recall()` + SQLite 存储 + FastEmbed 本地 Embedding → 跑通 3行代码示例
- [ ] **注册域名**: `neuralmem.dev` / `neuralmem.ai`
- [ ] **写 README 的 "Why NeuralMem" 段落**: 明确 vs Mem0 的差异，让人 10秒内理解价值

### 第一周目标

- [ ] 核心 API 可用（remember / recall / forget）
- [ ] MCP Server 可连接 Claude Desktop
- [ ] 20+ 单元测试
- [ ] README + Quickstart 文档就绪
- [ ] 内部 dogfooding（自己用来开发自己）
