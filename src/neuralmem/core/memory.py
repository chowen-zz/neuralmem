"""NeuralMem 核心记忆引擎 — 3行代码即用的设计"""
from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import NeuralMemError
from neuralmem.core.types import (
    ExportFormat,
    Memory,
    MemoryScope,
    MemoryType,
    SearchQuery,
    SearchResult,
)
from neuralmem.embedding.registry import get_embedder
from neuralmem.extraction.entity_resolver import EntityResolver
from neuralmem.extraction.extractor_registry import get_extractor
from neuralmem.graph.knowledge_graph import KnowledgeGraph
from neuralmem.lifecycle.consolidation import MemoryConsolidator
from neuralmem.lifecycle.decay import DecayManager
from neuralmem.retrieval.engine import RetrievalEngine
from neuralmem.storage.sqlite import SQLiteStorage

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
        self.embedding = get_embedder(self.config)
        self.graph = KnowledgeGraph(self.storage)

        # 提取器：通过 registry 动态选择（config.llm_extractor 控制）
        self.extractor = get_extractor(self.config)

        self.retrieval = RetrievalEngine(
            storage=self.storage,
            embedder=self.embedding,
            graph=self.graph,
            config=self.config,
        )
        self.decay = DecayManager(self.storage)
        self.consolidator = MemoryConsolidator(storage=self.storage, embedder=self.embedding)
        self.resolver = EntityResolver(self.embedding)

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

                # 去重 + 冲突检测
                high_th = self.config.conflict_threshold_high
                low_th = self.config.conflict_threshold_low
                exact_dupes = self.storage.find_similar(vector, user_id=user_id, threshold=high_th)
                if exact_dupes:
                    _logger.debug("Skipping duplicate memory: %s", item.content[:50])
                    continue

                # 冲突检测: 相似度在 [low_th, high_th) 视为 "version update"
                conflicts = self.storage.find_similar(
                    vector, user_id=user_id, threshold=low_th
                )
                # 过滤掉已失效的和超出高阈值的（已在上面处理）
                active_conflicts = [
                    c for c in conflicts
                    if c.is_active and c.superseded_by is None
                ]

                # 实体消歧（在构建 Memory 之前）
                resolved_entities = self.resolver.resolve(
                    item.entities,
                    self.graph.get_entities(user_id=user_id),
                )

                memory = Memory(
                    content=item.content,
                    memory_type=item.memory_type or memory_type or MemoryType.SEMANTIC,
                    scope=MemoryScope.USER if user_id else MemoryScope.SESSION,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    tags=tuple(tags or item.tags),
                    importance=importance or item.importance,
                    entity_ids=tuple(e.id for e in resolved_entities),
                    embedding=vector,
                )

                # 标记被取代的旧记忆
                superseded_ids: list[str] = []
                for old in active_conflicts:
                    self.storage.update_memory(
                        old.id,
                        is_active=False,
                        superseded_by=memory.id,
                    )
                    superseded_ids.append(old.id)
                    _logger.debug("Memory %s superseded by %s", old.id[:8], memory.id[:8])

                if superseded_ids:
                    # 创建带 supersedes 字段的记忆（frozen model 需要重建）
                    memory = memory.model_copy(update={"supersedes": tuple(superseded_ids)})

                self.storage.save_memory(memory)

                for entity in resolved_entities:
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

        # Filter superseded memories
        results = [r for r in results if r.memory.is_active]

        # Record access and optionally reinforce importance
        for result in results:
            self.storage.record_access(result.memory.id)
            if self.config.enable_importance_reinforcement:
                boosted = min(1.0, result.memory.importance + self.config.reinforcement_boost)
                if boosted > result.memory.importance:
                    self.storage.update_memory(result.memory.id, importance=boosted)

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

    def remember_batch(
        self,
        contents: list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Memory]:
        """
        Batch remember - processes multiple items efficiently.

        Args:
            contents: List of content strings to remember.
            user_id: User identifier for memory scoping.
            agent_id: Agent identifier.
            memory_type: Override memory type for all items.
            tags: Tags to apply to all items.
            progress_callback: Optional callback(current, total, content_preview).

        Returns:
            List of all stored memories across all items.
        """
        all_memories: list[Memory] = []
        total = len(contents)

        for idx, content in enumerate(contents):
            if progress_callback:
                preview = content[:60] + ("..." if len(content) > 60 else "")
                progress_callback(idx, total, preview)

            try:
                memories = self.remember(
                    content,
                    user_id=user_id,
                    agent_id=agent_id,
                    memory_type=memory_type,
                    tags=tags,
                )
                all_memories.extend(memories)
            except NeuralMemError:
                _logger.warning("Failed to remember item %d/%d: %s", idx + 1, total, content[:80])
                raise
            except Exception as e:
                _logger.warning("Failed to remember item %d/%d: %s", idx + 1, total, e)
                continue

        if progress_callback:
            progress_callback(total, total, "done")

        return all_memories

    def export_memories(
        self,
        *,
        user_id: str | None = None,
        format: str = "json",
        include_embeddings: bool = False,
    ) -> str:
        """
        Export memories as JSON, markdown, or CSV.

        Args:
            user_id: Filter memories by user. None = all users.
            format: Output format - "json", "markdown", or "csv".
            include_embeddings: If False (default), omit embedding vectors.

        Returns:
            Formatted string of exported memories.
        """
        memories = self.storage.list_memories(user_id=user_id)
        fmt = format.lower()

        if fmt == ExportFormat.JSON:
            return self._export_json(memories, include_embeddings)
        elif fmt == ExportFormat.MARKDOWN:
            return self._export_markdown(memories)
        elif fmt == ExportFormat.CSV:
            return self._export_csv(memories)
        else:
            raise NeuralMemError(
                f"Unsupported export format: {format!r}. "
                "Use json, markdown, or csv."
            )

    def _export_json(self, memories: list[Memory], include_embeddings: bool) -> str:
        items = []
        for m in memories:
            d: dict[str, object] = {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type.value,
                "scope": m.scope.value,
                "user_id": m.user_id,
                "agent_id": m.agent_id,
                "tags": list(m.tags),
                "importance": m.importance,
                "is_active": m.is_active,
                "created_at": m.created_at.isoformat(),
                "updated_at": m.updated_at.isoformat(),
                "access_count": m.access_count,
            }
            if include_embeddings and m.embedding:
                d["embedding"] = m.embedding
            items.append(d)
        return json.dumps(items, indent=2, ensure_ascii=False)

    def _export_markdown(self, memories: list[Memory]) -> str:
        lines = ["# NeuralMem Export", ""]
        for m in memories:
            status = "active" if m.is_active else "superseded"
            lines.append(f"## [{m.memory_type.value}] {m.content[:80]}")
            lines.append("")
            lines.append(f"- **ID**: `{m.id}`")
            lines.append(f"- **Status**: {status}")
            lines.append(f"- **Importance**: {m.importance:.2f}")
            if m.user_id:
                lines.append(f"- **User**: {m.user_id}")
            if m.tags:
                lines.append(f"- **Tags**: {', '.join(m.tags)}")
            lines.append(f"- **Created**: {m.created_at.isoformat()}")
            lines.append(f"- **Access count**: {m.access_count}")
            lines.append("")
        return "\n".join(lines)

    def _export_csv(self, memories: list[Memory]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "content", "memory_type", "scope", "user_id",
            "agent_id", "tags", "importance", "is_active",
            "created_at", "updated_at", "access_count",
        ])
        for m in memories:
            writer.writerow([
                m.id, m.content, m.memory_type.value, m.scope.value,
                m.user_id or "", m.agent_id or "",
                ";".join(m.tags), m.importance,
                int(m.is_active), m.created_at.isoformat(),
                m.updated_at.isoformat(), m.access_count,
            ])
        return buf.getvalue()

    def forget_batch(
        self,
        memory_ids: list[str] | None = None,
        *,
        user_id: str | None = None,
        tags: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        """
        Batch delete with dry_run preview.

        Args:
            memory_ids: List of specific memory IDs to delete.
            user_id: Delete all memories for this user.
            tags: Delete memories with any of these tags.
            dry_run: If True, return what would be deleted without actually deleting.

        Returns:
            Dict with 'count' (int), 'memory_ids' (list[str]), and 'dry_run' (bool).
        """
        # Collect target IDs from all sources
        target_ids: set[str] = set()
        if memory_ids:
            target_ids.update(memory_ids)

        if user_id is not None:
            memories = self.storage.list_memories(user_id=user_id)
            for m in memories:
                target_ids.add(m.id)

        if tags:
            all_mems = self.storage.list_memories(user_id=user_id)
            for m in all_mems:
                if any(tag in m.tags for tag in tags):
                    target_ids.add(m.id)

        target_list = sorted(target_ids)

        if dry_run:
            return {
                "count": len(target_list),
                "memory_ids": target_list,
                "dry_run": True,
            }

        deleted = 0
        for mid in target_list:
            try:
                n = self.storage.delete_memories(memory_id=mid)
                deleted += n
            except Exception as e:
                _logger.warning("Failed to delete memory %s: %s", mid[:8], e)

        return {
            "count": deleted,
            "memory_ids": target_list,
            "dry_run": False,
        }

    def consolidate(self, user_id: str | None = None) -> dict[str, int]:
        """
        后台记忆整理：衰减旧记忆、合并相似记忆。
        建议定期调用（如每天一次）。
        """
        return {
            "decayed": self.decay.apply_decay(user_id=user_id),
            "forgotten": self.decay.remove_forgotten(user_id=user_id),
            "merged": self.consolidator.merge_similar(user_id=user_id),
        }

    def resolve_conflict(
        self,
        memory_id: str,
        *,
        action: str = "reactivate",
    ) -> bool:
        """
        Resolve memory conflict manually.

        Args:
            memory_id: Target memory ID
            action: "reactivate" (re-enable) or "delete" (permanent removal)

        Returns:
            Whether the operation succeeded
        """
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            return False

        if action == "reactivate":
            if memory.superseded_by:
                self.storage.update_memory(memory.superseded_by, supersedes=())
            self.storage.update_memory(memory_id, is_active=True, superseded_by=None)
            return True
        elif action == "delete":
            self.storage.delete_memories(memory_id=memory_id)
            return True
        return False

    def get_stats(self) -> dict[str, object]:
        """Return memory store statistics."""
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

    def __enter__(self) -> NeuralMem:
        return self

    def __exit__(self, *args: object) -> None:
        pass  # SQLite 连接在各方法中自动管理

    # ==================== Session API ====================

    def session(
        self,
        conversation_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        promote_threshold: float = 0.7,
    ):
        """Return a SessionContext for 3-layer session-aware memory.

        Usage::

            with mem.session(user_id="u1") as ctx:
                ctx.append_working("transient context")
                ctx.remember_to_session("important fact")
                results = ctx.recall("search across layers")
            # On exit: session memories compressed, important ones promoted

        Args:
            conversation_id: Unique ID for this conversation. Auto-generated if None.
            user_id: User scope for session memories.
            agent_id: Agent scope for session memories.
            promote_threshold: Minimum importance for promoting session → long-term.

        Returns:
            SessionContext (context manager)
        """
        # Lazy import to avoid circular dependency
        from neuralmem.session.context import SessionContext

        return SessionContext(
            mem=self,
            conversation_id=conversation_id,
            user_id=user_id,
            agent_id=agent_id,
            promote_threshold=promote_threshold,
        )

    def session_start(
        self,
        conversation_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        """Start a session manually (non-context-manager usage).

        Returns:
            The conversation_id (auto-generated if not provided).
        """
        from uuid import uuid4

        from neuralmem.session.context import SessionContext

        cid = conversation_id or uuid4().hex[:16]
        # Store active session reference for session_append/session_end
        if not hasattr(self, '_active_sessions'):
            self._active_sessions: dict[str, object] = {}

        ctx = SessionContext(
            mem=self,
            conversation_id=cid,
            user_id=user_id,
            agent_id=agent_id,
        )
        ctx.__enter__()
        self._active_sessions[cid] = ctx
        return cid

    def session_append(
        self,
        conversation_id: str,
        content: str,
        *,
        layer: str = "session",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> list[Memory] | None:
        """Append content to an active session.

        Args:
            conversation_id: The session's conversation ID (from session_start).
            content: Content to remember.
            layer: 'working' (ephemeral) or 'session' (persisted).
            importance: Importance score for session-layer memories.
            tags: Optional tags.

        Returns:
            List of Memory objects (for session layer) or None (for working).
        """
        ctx = self._get_active_session(conversation_id)
        if ctx is None:
            _logger.warning("No active session for conversation_id=%s", conversation_id)
            return None

        if layer == "working":
            ctx.append_working(content)
            return None
        else:
            return ctx.remember_to_session(content, importance=importance, tags=tags)

    def session_end(
        self,
        conversation_id: str,
    ) -> None:
        """End an active session (triggers compression and promotion).

        Args:
            conversation_id: The session's conversation ID.
        """
        ctx = self._get_active_session(conversation_id)
        if ctx is None:
            _logger.warning("No active session for conversation_id=%s", conversation_id)
            return

        ctx.__exit__(None, None, None)
        self._active_sessions.pop(conversation_id, None)  # type: ignore[union-attr]

    def _get_active_session(self, conversation_id: str):
        """Retrieve an active session context by conversation_id."""
        if not hasattr(self, '_active_sessions'):
            return None
        return self._active_sessions.get(conversation_id)  # type: ignore[union-attr]
