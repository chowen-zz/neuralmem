"""AsyncNeuralMem — 异步版本的 NeuralMem 记忆引擎。

保持与同步 NeuralMem 完全相同的方法签名，所有公共方法均为 async def。
内部通过 AsyncStorage、AsyncEmbedder、AsyncRetrievalEngine 实现异步 I/O，
使用 asyncio.gather 并行执行多个策略和批量操作。

最简使用（3行异步代码）:
    >>> from neuralmem.async_api import AsyncNeuralMem
    >>> mem = AsyncNeuralMem()
    >>> await mem.remember("用户偏好用 TypeScript 写前端")
    >>> results = await mem.recall("用户的技术偏好是什么？")
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from neuralmem.async_api.embedding import AsyncEmbedder
from neuralmem.async_api.retrieval import AsyncRetrievalEngine
from neuralmem.async_api.storage import AsyncStorage
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import NeuralMemError
from neuralmem.core.metrics import MetricsCollector
from neuralmem.core.types import (
    ExportFormat,
    Memory,
    MemoryHistoryEntry,
    MemoryScope,
    MemoryType,
    SearchQuery,
    SearchResult,
    _generate_ulid,
)
from neuralmem.embedding.registry import get_embedder
from neuralmem.extraction.entity_resolver import EntityResolver
from neuralmem.extraction.extractor_registry import get_extractor
from neuralmem.graph.knowledge_graph import KnowledgeGraph
from neuralmem.lifecycle.consolidation import MemoryConsolidator
from neuralmem.lifecycle.decay import DecayManager
from neuralmem.storage.sqlite import SQLiteStorage

_logger = logging.getLogger(__name__)


class AsyncNeuralMem:
    """Agent 记忆引擎的异步统一入口。

    所有公共方法均为 async def，保持与同步 NeuralMem 相同的方法签名。
    内部子系统通过异步包装器实现非阻塞 I/O。
    """

    def __init__(
        self,
        db_path: str = "~/.neuralmem/memory.db",
        config: NeuralMemConfig | None = None,
        embedder=None,  # Optional: any object matching EmbedderProtocol (encode method)
    ):
        self.config = config or NeuralMemConfig(db_path=db_path)
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)

        # 初始化同步子系统，再包装为异步接口
        sync_storage = SQLiteStorage(self.config)
        sync_embedder = embedder or get_embedder(self.config)

        self.storage = AsyncStorage(sync_storage)
        self.embedding = AsyncEmbedder(sync_embedder)
        self.graph = KnowledgeGraph(sync_storage)

        self.extractor = get_extractor(self.config)
        self.resolver = EntityResolver(sync_embedder)

        self.retrieval = AsyncRetrievalEngine(
            storage=self.storage,
            embedder=self.embedding,
            graph=self.graph,
            config=self.config,
        )
        self.decay = DecayManager(sync_storage)
        self.consolidator = MemoryConsolidator(storage=sync_storage, embedder=sync_embedder)

        self.metrics = MetricsCollector(
            enabled=getattr(self.config, "enable_metrics", False),
        )

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
        expires_at: datetime | None = None,
        expires_in: timedelta | None = None,
        infer: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> list[Memory]:
        """异步存储记忆。自动执行：提取实体 → 生成 Embedding → 去重检查 → 持久化 → 更新图谱。

        与同步版本保持相同参数签名，返回提取并存储的记忆列表。
        """
        self.metrics.record_counter("neuralmem.async.remember.calls")
        with self.metrics.timer("neuralmem.async.remember"):
            return await self._remember_impl(
                content,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type=memory_type,
                tags=tags,
                importance=importance,
                expires_at=expires_at,
                expires_in=expires_in,
                infer=infer,
                metadata=metadata,
            )

    async def _remember_impl(
        self,
        content: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        expires_at: datetime | None = None,
        expires_in: timedelta | None = None,
        infer: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> list[Memory]:
        if not content or not content.strip():
            return []

        # Resolve expiration
        resolved_expires_at = expires_at
        if resolved_expires_at is None and expires_in is not None:
            from datetime import timezone
            resolved_expires_at = datetime.now(timezone.utc) + expires_in

        if not infer:
            return await self._remember_verbatim(
                content,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type=memory_type,
                tags=tags,
                importance=importance,
                expires_at=resolved_expires_at,
                metadata=metadata,
            )

        try:
            # 异步获取现有实体
            existing_entities = await asyncio.to_thread(
                self.graph.get_entities, user_id=user_id
            )

            # 提取（同步操作，可能涉及 LLM 调用）
            extracted = await asyncio.to_thread(
                self.extractor.extract,
                content,
                memory_type=memory_type,
                existing_entities=existing_entities,
            )

            if not extracted:
                return []

            # 批量编码所有提取内容
            all_contents = [item.content for item in extracted]
            all_vectors = await self.embedding.encode(all_contents)

            memories: list[Memory] = []
            for item, vector in zip(extracted, all_vectors):
                # 去重检查
                high_th = self.config.conflict_threshold_high
                exact_dupes = await self.storage.find_similar(
                    vector, user_id=user_id, threshold=high_th
                )
                if exact_dupes:
                    _logger.debug("Skipping duplicate memory: %s", item.content[:50])
                    continue

                # 冲突检测
                low_th = self.config.conflict_threshold_low
                conflicts = await self.storage.find_similar(
                    vector, user_id=user_id, threshold=low_th
                )
                active_conflicts = [
                    c for c in conflicts
                    if c.is_active and c.superseded_by is None
                ]

                # 实体消歧
                resolved_entities = await asyncio.to_thread(
                    self.resolver.resolve,
                    item.entities,
                    existing_entities,
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
                    expires_at=resolved_expires_at,
                )

                # 标记被取代的旧记忆
                superseded_ids: list[str] = []
                for old in active_conflicts:
                    await self.storage.update_memory(
                        old.id,
                        is_active=False,
                        superseded_by=memory.id,
                    )
                    superseded_ids.append(old.id)
                    _logger.debug("Memory %s superseded by %s", old.id[:8], memory.id[:8])

                if superseded_ids:
                    memory = memory.model_copy(update={"supersedes": tuple(superseded_ids)})

                await self.storage.save_memory(memory)

                # 异步更新图谱（实体和关系可以并行）
                entity_tasks = [
                    asyncio.to_thread(self.graph.upsert_entity, entity)
                    for entity in resolved_entities
                ]
                link_tasks = [
                    asyncio.to_thread(self.graph.link_memory_to_entity, memory.id, entity.id)
                    for entity in resolved_entities
                ]
                relation_tasks = [
                    asyncio.to_thread(self.graph.add_relation, relation)
                    for relation in item.relations
                ]
                await asyncio.gather(*entity_tasks, *link_tasks, *relation_tasks, return_exceptions=True)

                memories.append(memory)

            return memories

        except NeuralMemError:
            raise
        except Exception as e:
            raise NeuralMemError(f"remember() failed: {e}") from e

    async def recall(
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
        """异步检索相关记忆。使用四策略并行检索 + RRF 融合。"""
        self.metrics.record_counter("neuralmem.async.recall.calls")
        with self.metrics.timer("neuralmem.async.recall"):
            return await self._recall_impl(
                query,
                user_id=user_id,
                agent_id=agent_id,
                memory_types=memory_types,
                tags=tags,
                time_range=time_range,
                limit=limit,
                min_score=min_score,
            )

    async def _recall_impl(
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

        results = await self.retrieval.search(search_query)

        # Filter superseded memories
        results = [r for r in results if r.memory.is_active]

        # Batch record access
        if results:
            await self.storage.batch_record_access([r.memory.id for r in results])

        # Optional importance reinforcement
        if self.config.enable_importance_reinforcement:
            reinforcement_tasks = []
            for result in results:
                boosted = min(1.0, result.memory.importance + self.config.reinforcement_boost)
                if boosted > result.memory.importance:
                    reinforcement_tasks.append(
                        self.storage.update_memory(result.memory.id, importance=boosted)
                    )
            if reinforcement_tasks:
                await asyncio.gather(*reinforcement_tasks, return_exceptions=True)

        return results

    async def reflect(
        self,
        topic: str,
        *,
        user_id: str | None = None,
        depth: int = 2,
    ) -> str:
        """异步对指定主题进行记忆推理和总结。"""
        direct = await self.recall(topic, user_id=user_id, limit=20)

        # 基于图谱扩展
        entity_ids: set[str] = set()
        for r in direct:
            entity_ids.update(r.memory.entity_ids)

        related_entities = await asyncio.to_thread(
            self.graph.get_neighbors, list(entity_ids), depth=depth
        )

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

    async def forget(
        self,
        memory_id: str | None = None,
        *,
        user_id: str | None = None,
        before: datetime | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """异步删除指定记忆（支持 GDPR 合规的完全删除）。"""
        return await self.storage.delete_memories(
            memory_id=memory_id,
            user_id=user_id,
            before=before,
            tags=tags,
        )

    async def remember_batch(
        self,
        contents: list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Memory]:
        """异步批量 remember — 使用 asyncio.gather 并发处理多个内容。

        注意：由于 remember 涉及写入操作（数据库/图谱更新），
        默认使用 semaphore 控制并发度为 4，避免数据库锁竞争。
        """
        all_memories: list[Memory] = []
        total = len(contents)
        semaphore = asyncio.Semaphore(4)

        async def _remember_one(idx: int, content: str) -> list[Memory]:
            if progress_callback:
                preview = content[:60] + ("..." if len(content) > 60 else "")
                progress_callback(idx, total, preview)

            async with semaphore:
                try:
                    return await self.remember(
                        content,
                        user_id=user_id,
                        agent_id=agent_id,
                        memory_type=memory_type,
                        tags=tags,
                    )
                except NeuralMemError:
                    _logger.warning(
                        "Failed to remember item %d/%d: %s", idx + 1, total, content[:80]
                    )
                    raise
                except Exception as e:
                    _logger.warning(
                        "Failed to remember item %d/%d: %s", idx + 1, total, e
                    )
                    return []

        tasks = [_remember_one(i, content) for i, content in enumerate(contents)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_memories.extend(result)
            elif isinstance(result, Exception):
                _logger.warning("Batch remember item failed: %s", result)

        if progress_callback:
            progress_callback(total, total, "done")

        return all_memories

    async def remember_conversation(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        auto_merge: bool = True,
        tags: list[str] | None = None,
    ) -> list[Memory]:
        """异步从多轮对话中提取并存储记忆。"""
        from neuralmem.extraction.conversation_extractor import ConversationExtractor
        from neuralmem.extraction.merger import MemoryMerger

        self.metrics.record_counter("neuralmem.async.remember_conversation.calls")

        if not messages:
            return []

        extractor = ConversationExtractor()
        extracted = await asyncio.to_thread(extractor.extract, messages)

        if not extracted:
            return []

        all_memories: list[Memory] = []
        semaphore = asyncio.Semaphore(4)

        async def _store_one(item) -> list[Memory]:
            combined_tags = tuple(
                dict.fromkeys(list(item.tags) + (tags or []))
            )
            async with semaphore:
                return await self.remember(
                    item.content,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    memory_type=item.memory_type,
                    tags=list(combined_tags),
                    importance=item.confidence,
                    infer=True,
                    metadata=item.metadata,
                )

        tasks = [_store_one(item) for item in extracted]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_memories.extend(result)
            elif isinstance(result, Exception):
                _logger.warning("Conversation memory extraction failed: %s", result)

        if auto_merge and len(all_memories) > 1:
            merger = MemoryMerger()
            merge_results = await asyncio.to_thread(
                merger.merge_batch, all_memories, existing_memories=[]
            )
            merged_memories: list[Memory] = []
            for result in merge_results:
                merged_memories.append(result.merged_memory)
            all_memories = merged_memories

        return all_memories

    # ==================== 导出 / 导入 ====================

    async def export_memories(
        self,
        *,
        user_id: str | None = None,
        format: str = "json",
        include_embeddings: bool = False,
    ) -> str:
        """异步导出记忆为 JSON、markdown 或 CSV。"""
        memories = await self.storage.list_memories(user_id=user_id)
        fmt = format.lower()

        if fmt == ExportFormat.JSON:
            return await asyncio.to_thread(
                self._export_json, memories, include_embeddings
            )
        elif fmt == ExportFormat.MARKDOWN:
            return await asyncio.to_thread(self._export_markdown, memories)
        elif fmt == ExportFormat.CSV:
            return await asyncio.to_thread(self._export_csv, memories)
        else:
            raise NeuralMemError(
                f"Unsupported export format: {format!r}. Use json, markdown, or csv."
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

    async def import_memories(
        self,
        data: str | list[dict[str, object]],
        *,
        format: str = "json",
        user_id: str | None = None,
        skip_duplicates: bool = True,
    ) -> int:
        """异步导入记忆从导出数据。"""
        if isinstance(data, list):
            items = data
        else:
            fmt = format.lower()
            if fmt == ExportFormat.JSON:
                items = await asyncio.to_thread(self._import_json, data)
            elif fmt == ExportFormat.MARKDOWN:
                items = await asyncio.to_thread(self._import_markdown, data)
            elif fmt == ExportFormat.CSV:
                items = await asyncio.to_thread(self._import_csv, data)
            else:
                raise NeuralMemError(
                    f"Unsupported import format: {format!r}. Use json, markdown, or csv."
                )

        imported = 0
        for raw in items:
            if user_id is not None:
                raw["user_id"] = user_id

            content = raw.get("content", "")
            if not content:
                continue

            memory_type = MemoryType(raw.get("memory_type", "semantic"))
            scope = MemoryScope(raw.get("scope", "user"))
            raw_user_id = raw.get("user_id")
            raw_agent_id = raw.get("agent_id")
            tags = tuple(raw.get("tags", []))
            importance = float(raw.get("importance", 0.5))

            if skip_duplicates:
                existing = await self.storage.list_memories(user_id=raw_user_id)
                is_dup = False
                for m in existing:
                    if m.content.strip().lower() == content.strip().lower():
                        _logger.debug("Skipping duplicate import: %s", content[:50])
                        is_dup = True
                        break
                if is_dup:
                    continue

            vectors = await self.embedding.encode([content])
            vector = vectors[0]

            memory = Memory(
                content=content,
                memory_type=memory_type,
                scope=scope,
                user_id=raw_user_id,
                agent_id=raw_agent_id,
                tags=tags,
                importance=importance,
                embedding=vector,
            )

            await self.storage.save_memory(memory)
            imported += 1

        return imported

    def _import_json(self, data: str) -> list[dict[str, object]]:
        try:
            items = json.loads(data)
        except (json.JSONDecodeError, TypeError) as e:
            raise NeuralMemError(f"Invalid JSON data: {e}") from e
        if not isinstance(items, list):
            raise NeuralMemError("JSON import expects an array of memory objects")
        return items

    def _import_markdown(self, data: str) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        for line in data.splitlines():
            line = line.strip()
            if line.startswith("## ["):
                if current is not None:
                    items.append(current)
                bracket_end = line.index("]")
                memory_type = line[4:bracket_end]
                content = line[bracket_end + 1:].strip()
                current = {"memory_type": memory_type, "content": content, "tags": []}
            elif current is not None and line.startswith("- **"):
                try:
                    colon_idx = line.index(":")
                    key = line[4:colon_idx].strip().strip("*").strip().lower()
                    value = line[colon_idx + 1:].strip().strip("`").strip()
                    if key == "importance":
                        current["importance"] = float(value)
                    elif key == "user":
                        current["user_id"] = value
                    elif key == "tags":
                        current["tags"] = [t.strip() for t in value.split(",")]
                    elif key == "id":
                        current["id"] = value
                except (ValueError, IndexError):
                    pass
        if current is not None:
            items.append(current)
        return items

    def _import_csv(self, data: str) -> list[dict[str, object]]:
        reader = csv.DictReader(io.StringIO(data))
        items: list[dict[str, object]] = []
        for row in reader:
            raw: dict[str, object] = {
                "content": row.get("content", ""),
                "memory_type": row.get("memory_type", "semantic"),
                "scope": row.get("scope", "user"),
                "user_id": row.get("user_id") or None,
                "agent_id": row.get("agent_id") or None,
                "importance": float(row.get("importance", 0.5)),
            }
            tags_str = row.get("tags", "")
            raw["tags"] = [t.strip() for t in tags_str.split(";") if t.strip()]
            items.append(raw)
        return items

    # ==================== 批量删除 / 整理 ====================

    async def forget_batch(
        self,
        memory_ids: list[str] | None = None,
        *,
        user_id: str | None = None,
        tags: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        """异步批量删除，支持 dry_run 预览。"""
        target_ids: set[str] = set()
        if memory_ids:
            target_ids.update(memory_ids)

        if user_id is not None:
            memories = await self.storage.list_memories(user_id=user_id)
            for m in memories:
                target_ids.add(m.id)

        if tags:
            all_mems = await self.storage.list_memories(user_id=user_id)
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

        # 并发删除，控制并发度
        semaphore = asyncio.Semaphore(8)
        deleted = 0

        async def _delete_one(mid: str) -> int:
            async with semaphore:
                try:
                    return await self.storage.delete_memories(memory_id=mid)
                except Exception as e:
                    _logger.warning("Failed to delete memory %s: %s", mid[:8], e)
                    return 0

        tasks = [_delete_one(mid) for mid in target_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, int):
                deleted += r

        return {
            "count": deleted,
            "memory_ids": target_list,
            "dry_run": False,
        }

    async def consolidate(self, user_id: str | None = None) -> dict[str, int]:
        """异步后台记忆整理：衰减旧记忆、合并相似记忆。"""
        decayed, forgotten, merged = await asyncio.gather(
            asyncio.to_thread(self.decay.apply_decay, user_id=user_id),
            asyncio.to_thread(self.decay.remove_forgotten, user_id=user_id),
            asyncio.to_thread(self.consolidator.merge_similar, user_id=user_id),
            return_exceptions=True,
        )
        return {
            "decayed": decayed if isinstance(decayed, int) else 0,
            "forgotten": forgotten if isinstance(forgotten, int) else 0,
            "merged": merged if isinstance(merged, int) else 0,
        }

    async def cleanup_expired(self) -> int:
        """异步清理已过期的记忆。"""
        return await self.storage.cleanup_expired()

    async def resolve_conflict(
        self,
        memory_id: str,
        *,
        action: str = "reactivate",
    ) -> bool:
        """异步解决记忆冲突。"""
        memory = await self.storage.get_memory(memory_id)
        if memory is None:
            return False

        if action == "reactivate":
            if memory.superseded_by:
                await self.storage.update_memory(memory.superseded_by, supersedes=())
            await self.storage.update_memory(memory_id, is_active=True, superseded_by=None)
            return True
        elif action == "delete":
            await self.storage.delete_memories(memory_id=memory_id)
            return True
        return False

    async def get_stats(self) -> dict[str, object]:
        """异步获取记忆存储统计。"""
        storage_stats, graph_stats = await asyncio.gather(
            self.storage.get_stats(),
            asyncio.to_thread(self.graph.get_stats),
            return_exceptions=True,
        )
        if isinstance(storage_stats, Exception):
            storage_stats = {}
        if isinstance(graph_stats, Exception):
            graph_stats = {}
        return {**storage_stats, **graph_stats}

    # ==================== 单条操作 ====================

    async def get(self, memory_id: str) -> Memory | None:
        """异步按 ID 获取单条记忆。"""
        self.metrics.record_counter("neuralmem.async.get.calls")
        return await self.storage.get_memory(memory_id)

    async def update(
        self,
        memory_id: str,
        content: str,
        *,
        importance: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Memory | None:
        """异步更新记忆内容，自动记录版本历史。"""
        self.metrics.record_counter("neuralmem.async.update.calls")
        existing = await self.storage.get_memory(memory_id)
        if existing is None:
            return None

        old_content = existing.content
        content_changed = old_content != content

        storage_kwargs: dict[str, object] = {}
        if content_changed:
            vectors = await self.embedding.encode([content])
            vector = vectors[0]
            storage_kwargs["content"] = content
            storage_kwargs["embedding"] = vector
        if importance is not None:
            storage_kwargs["importance"] = importance

        if not storage_kwargs:
            return existing

        await self.storage.update_memory(memory_id, **storage_kwargs)

        event = "UPDATE"
        history_new = content if content_changed else old_content
        await self.storage.save_history(
            memory_id, old_content, history_new, event=event, metadata=metadata,
        )

        if content_changed:
            try:
                items = await asyncio.to_thread(self.extractor.extract, content)
                for item in items:
                    all_entities = await asyncio.to_thread(self.graph.get_entities)
                    resolved = await asyncio.to_thread(
                        self.resolver.resolve,
                        item.entities,
                        existing_entities=all_entities,
                    )
                    for entity in resolved:
                        await asyncio.to_thread(self.graph.upsert_entity, entity)
                        await asyncio.to_thread(
                            self.graph.link_memory_to_entity, memory_id, entity.id
                        )
            except Exception:
                _logger.debug("Graph update skipped for memory %s", memory_id[:8])

        updated = await self.storage.get_memory(memory_id)
        _logger.debug("Updated memory %s", memory_id[:8])
        return updated

    async def history(self, memory_id: str) -> list[MemoryHistoryEntry]:
        """异步获取记忆的版本历史。"""
        self.metrics.record_counter("neuralmem.async.history.calls")
        raw = await self.storage.get_history(memory_id)
        entries: list[MemoryHistoryEntry] = []
        for row in raw:
            try:
                changed_at = row["changed_at"]
                if isinstance(changed_at, str):
                    changed_at = datetime.fromisoformat(changed_at)
                entries.append(MemoryHistoryEntry(
                    id=row["id"],
                    memory_id=row["memory_id"],
                    old_content=row.get("old_content"),
                    new_content=row["new_content"],
                    event=row["event"],
                    changed_at=changed_at,
                    metadata=row.get("metadata", {}),
                ))
            except Exception as e:
                _logger.warning("Skipping malformed history entry: %s", e)
        return entries

    # ==================== 内部辅助 ====================

    async def _remember_verbatim(
        self,
        content: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, object] | None = None,
    ) -> list[Memory]:
        """异步 verbatim 存储（infer=False 模式）。"""
        vector = await self.embedding.encode_one(content)
        memory = Memory(
            id=_generate_ulid(),
            content=content,
            memory_type=memory_type or MemoryType.EPISODIC,
            scope=MemoryScope.USER if user_id else MemoryScope.SESSION,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            tags=tuple(tags or ()),
            importance=importance or 0.5,
            embedding=vector,
            expires_at=expires_at,
        )
        await self.storage.save_memory(memory)
        await self.storage.save_history(memory.id, None, content, event="CREATE", metadata=metadata)
        _logger.debug("Stored verbatim memory %s (%d chars)", memory.id[:8], len(content))
        return [memory]

    # ==================== 上下文管理器 ====================

    async def __aenter__(self) -> AsyncNeuralMem:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.retrieval.close()

    # ==================== Session API ====================

    async def session(
        self,
        conversation_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        promote_threshold: float = 0.7,
    ):
        """返回异步 SessionContext（上下文管理器）。"""
        from neuralmem.session.context import SessionContext

        return SessionContext(
            mem=self,
            conversation_id=conversation_id,
            user_id=user_id,
            agent_id=agent_id,
            promote_threshold=promote_threshold,
        )

    async def session_start(
        self,
        conversation_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        """异步手动启动 session（非上下文管理器用法）。"""
        from uuid import uuid4

        from neuralmem.session.context import SessionContext

        cid = conversation_id or uuid4().hex[:16]
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

    async def session_append(
        self,
        conversation_id: str,
        content: str,
        *,
        layer: str = "session",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> list[Memory] | None:
        """异步追加内容到活跃 session。"""
        ctx = self._get_active_session(conversation_id)
        if ctx is None:
            _logger.warning("No active session for conversation_id=%s", conversation_id)
            return None

        if layer == "working":
            ctx.append_working(content)
            return None
        else:
            return ctx.remember_to_session(content, importance=importance, tags=tags)

    async def session_end(self, conversation_id: str) -> None:
        """异步结束活跃 session。"""
        ctx = self._get_active_session(conversation_id)
        if ctx is None:
            _logger.warning("No active session for conversation_id=%s", conversation_id)
            return

        ctx.__exit__(None, None, None)
        self._active_sessions.pop(conversation_id, None)

    def _get_active_session(self, conversation_id: str):
        """获取活跃 session 上下文。"""
        if not hasattr(self, '_active_sessions'):
            return None
        return self._active_sessions.get(conversation_id)
