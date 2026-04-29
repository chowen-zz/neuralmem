"""NeuralMem 核心记忆引擎 — 3行代码即用的设计"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Memory, MemoryType, MemoryScope, SearchResult, SearchQuery
from neuralmem.core.exceptions import NeuralMemError
from neuralmem.storage.sqlite import SQLiteStorage
from neuralmem.embedding.local import LocalEmbedding
from neuralmem.extraction.extractor import MemoryExtractor
from neuralmem.extraction.llm_extractor import LLMExtractor
from neuralmem.graph.knowledge_graph import KnowledgeGraph
from neuralmem.retrieval.engine import RetrievalEngine
from neuralmem.lifecycle.decay import DecayManager
from neuralmem.lifecycle.consolidation import MemoryConsolidator

_logger = logging.getLogger(__name__)


class NeuralMem:
    """
    Agent 记忆引擎的统一入口。

    最简使用（3行代码）:
        >>> from neuralmem import NeuralMem
        >>> mem = NeuralMem()
        >>> mem.remember("用户偏好用 TypeScript 写前端")
        >>> results = mem.recall("用户的技术偏好是什么？")

    自定义配置:
        >>> mem = NeuralMem(db_path="./my_memory.db")
    """

    def __init__(
        self,
        db_path: str = "~/.neuralmem/memory.db",
        config: NeuralMemConfig | None = None,
    ):
        self.config = config or NeuralMemConfig(db_path=db_path)
        # 确保数据库目录存在
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)

        # 初始化子系统
        self.storage = SQLiteStorage(self.config)
        self.embedding = LocalEmbedding(self.config)
        self.graph = KnowledgeGraph(self.storage)

        # 提取器：优先使用 LLM（如果配置了的话）
        if self.config.enable_llm_extraction:
            self.extractor: MemoryExtractor | LLMExtractor = LLMExtractor(self.config)
        else:
            self.extractor = MemoryExtractor(self.config)

        self.retrieval = RetrievalEngine(
            storage=self.storage,
            embedder=self.embedding,
            graph=self.graph,
            config=self.config,
        )
        self.decay = DecayManager()
        self.consolidator = MemoryConsolidator()

    # ==================== 核心 API ====================

    def remember(
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
        存储记忆。自动执行：提取实体 → 生成 Embedding → 去重检查 → 持久化 → 更新图谱

        Returns:
            提取并存储的记忆列表（一段内容可提取多条记忆）
        """
        try:
            extracted = self.extractor.extract(
                content,
                memory_type=memory_type,
                existing_entities=self.graph.get_entities(user_id=user_id),
            )

            memories: list[Memory] = []
            for item in extracted:
                vector = self.embedding.encode_one(item.content)

                # 去重检查（相似度 >= 0.95 则跳过）
                duplicates = self.storage.find_similar(vector, user_id=user_id, threshold=0.95)
                if duplicates:
                    _logger.debug("Skipping duplicate memory: %s", item.content[:50])
                    continue

                memory = Memory(
                    content=item.content,
                    memory_type=item.memory_type or memory_type or MemoryType.SEMANTIC,
                    scope=MemoryScope.USER if user_id else MemoryScope.SESSION,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    tags=tuple(tags or item.tags),
                    importance=importance or item.importance,
                    entity_ids=tuple(item.entity_ids),
                    embedding=vector,
                )

                self.storage.save_memory(memory)

                # 更新图谱
                for entity in item.entities:
                    self.graph.upsert_entity(entity)
                    self.graph.link_memory_to_entity(memory.id, entity.id)
                for relation in item.relations:
                    self.graph.add_relation(relation)

                memories.append(memory)

            return memories

        except NeuralMemError:
            raise
        except Exception as e:
            raise NeuralMemError(f"remember() failed: {e}") from e

    def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """
        检索相关记忆。使用四策略并行检索 + RRF 融合。

        Returns:
            按相关性排序的搜索结果列表
        """
        search_query = SearchQuery(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            memory_types=tuple(memory_types) if memory_types else None,
            tags=tuple(tags) if tags else None,
            time_range=time_range,
            limit=limit,
            min_score=min_score,
        )

        results = self.retrieval.search(search_query)

        # 记录访问（用于后续衰减计算）
        for result in results:
            self.storage.record_access(result.memory.id)

        return results

    def reflect(
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
        direct = self.recall(topic, user_id=user_id, limit=20)

        # 基于图谱扩展
        entity_ids: set[str] = set()
        for r in direct:
            entity_ids.update(r.memory.entity_ids)

        related_entities = self.graph.get_neighbors(list(entity_ids), depth=depth)

        # 构建报告
        lines = [f"# Reflection on: {topic}", ""]
        if direct:
            lines.append("## Direct Memories")
            for i, r in enumerate(direct[:5], 1):
                lines.append(f"{i}. [{r.score:.2f}] {r.memory.content}")
        if related_entities:
            lines.append("")
            lines.append("## Related Entities")
            for entity in related_entities[:10]:
                lines.append(f"- {entity.name} ({entity.entity_type})")

        return "\n".join(lines)

    def forget(
        self,
        memory_id: str | None = None,
        *,
        user_id: str | None = None,
        before: datetime | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """
        删除指定记忆（支持 GDPR 合规的完全删除）。

        Returns:
            删除的记忆数量
        """
        return self.storage.delete_memories(
            memory_id=memory_id,
            user_id=user_id,
            before=before,
            tags=tags,
        )

    def consolidate(self, user_id: str | None = None) -> dict[str, int]:
        """
        后台记忆整理：衰减旧记忆、合并相似记忆（当前为 stub）。
        建议定期调用（如每天一次）。
        """
        return {
            "decayed": self.decay.apply_decay(user_id=user_id),
            "forgotten": self.decay.remove_forgotten(user_id=user_id),
            "merged": self.consolidator.merge_similar(user_id=user_id),
        }

    def get_stats(self) -> dict[str, object]:
        """返回记忆库统计信息"""
        storage_stats = self.storage.get_stats()
        graph_stats = self.graph.get_stats()
        return {**storage_stats, **graph_stats}

    # ==================== 同步便捷 API ====================

    def remember_sync(self, content: str, **kwargs: object) -> list[Memory]:
        """remember 的同步别名（兼容接口）"""
        return self.remember(content, **kwargs)  # type: ignore[arg-type]

    def recall_sync(self, query: str, **kwargs: object) -> list[SearchResult]:
        """recall 的同步别名"""
        return self.recall(query, **kwargs)  # type: ignore[arg-type]

    # ==================== 上下文管理器 ====================

    def __enter__(self) -> "NeuralMem":
        return self

    def __exit__(self, *args: object) -> None:
        pass  # SQLite 连接在各方法中自动管理
