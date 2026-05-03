"""SummaryUpdater — 增量更新已有摘要。

当新记忆加入或已有记忆变更时，增量更新相关摘要内容，
避免全量重新生成，保持摘要的时效性。
"""
from __future__ import annotations

import logging
from typing import Any

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Memory
from neuralmem.rewrite.base import ContextRewriter, LLMCaller, RewriteResult
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

_DEFAULT_UPDATE_PROMPT = (
    "You are a summary updater. Given an existing summary and one or more new memories, "
    "produce an updated summary that incorporates the new information while keeping it concise.\n\n"
    "Existing summary: {existing_summary}\n\n"
    "New memories:\n{new_memories}\n\n"
    "Return JSON: {{\"updated_summary\": \"updated text\", "
    "\"key_themes\": [\"theme1\", \"theme2\"], "
    "\"importance\": 0.0-1.0, "
    "\"should_archive\": true|false}}"
)


class SummaryUpdater(ContextRewriter):
    """增量更新已有摘要记忆。

    工作流程：
    1. 从存储中拉取所有 summary 标签的记忆
    2. 对每个摘要，检查其 source_memory_ids 对应的源记忆是否有新增/变更
    3. 若有，则调用 LLM 增量更新摘要内容
    4. 若摘要已过于庞大或过时，可选择归档并生成新摘要
    """

    def __init__(
        self,
        config: NeuralMemConfig,
        storage: StorageBackend,
        llm_caller: LLMCaller | None = None,
        max_source_memories: int = 20,
        archive_threshold: int = 30,
    ) -> None:
        super().__init__(config, storage, llm_caller)
        self._max_source_memories = max_source_memories
        self._archive_threshold = archive_threshold

    # ------------------------------------------------------------------
    # 差异检测
    # ------------------------------------------------------------------

    def _find_new_source_memories(self, summary: Memory) -> list[Memory]:
        """找出 summary 的 source_memory_ids 之外、语义相关的新记忆。"""
        source_ids = set(summary.entity_ids)
        user_id = summary.user_id

        # 拉取该用户近期活跃记忆
        recent = self._storage.list_memories(user_id=user_id, limit=5_000)
        candidates = [
            m for m in recent
            if m.is_active and m.id not in source_ids and "summary" not in m.tags
        ]

        # 若摘要无向量，则返回空（无法判断语义相关）
        if not summary.embedding:
            return []

        # 按向量相似度筛选与摘要相关的新记忆
        import numpy as np

        def _cosine(a: list[float], b: list[float]) -> float:
            av = np.array(a, dtype=np.float32)
            bv = np.array(b, dtype=np.float32)
            return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-9))

        related: list[tuple[Memory, float]] = []
        for m in candidates:
            if m.embedding:
                sim = _cosine(summary.embedding, m.embedding)
                if sim >= 0.75:
                    related.append((m, sim))

        related.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in related[: self._max_source_memories]]

    # ------------------------------------------------------------------
    # 增量更新
    # ------------------------------------------------------------------

    def _update_summary(self, summary: Memory, new_memories: list[Memory]) -> tuple[Memory, bool] | None:
        """对摘要执行增量更新，返回 (updated_memory, should_archive)。"""
        if not new_memories:
            return None

        new_texts = [f"- {m.content}" for m in new_memories]
        combined = "\n".join(new_texts)

        if self._llm is not None:
            try:
                return self._llm_update(summary, combined, new_memories)
            except Exception as exc:
                _logger.warning("LLM summary update failed (%s), using rules.", exc)

        return self._rule_update(summary, combined, new_memories)

    def _llm_update(
        self, summary: Memory, combined: str, new_memories: list[Memory]
    ) -> tuple[Memory, bool] | None:
        prompt = _DEFAULT_UPDATE_PROMPT.format(
            existing_summary=summary.content[:2000],
            new_memories=combined[:2000],
        )
        raw = self._call_llm(prompt)
        data = self._parse_json(raw)
        updated_text = data.get("updated_summary", "")
        if not updated_text:
            return None
        importance = float(data.get("importance", summary.importance))
        importance = max(0.0, min(1.0, importance))
        should_archive = bool(data.get("should_archive", False))

        updated = Memory(
            content=updated_text,
            memory_type=summary.memory_type,
            scope=summary.scope,
            user_id=summary.user_id,
            agent_id=summary.agent_id,
            session_id=summary.session_id,
            tags=summary.tags,
            source=summary.source,
            importance=importance,
            entity_ids=summary.entity_ids + tuple(m.id for m in new_memories),
            is_active=True,
            created_at=summary.created_at,
            updated_at=self._now(),
            last_accessed=summary.last_accessed,
            access_count=summary.access_count,
            embedding=summary.embedding,
        )
        return updated, should_archive

    def _rule_update(
        self, summary: Memory, combined: str, new_memories: list[Memory]
    ) -> tuple[Memory, bool] | None:
        """规则回退：在原有摘要后追加新记忆要点。"""
        # 若源记忆数超过阈值，建议归档
        total_sources = len(summary.entity_ids) + len(new_memories)
        should_archive = total_sources > self._archive_threshold

        # 简单追加式更新
        additions = " | ".join(m.content[:200] for m in new_memories[:3])
        updated_text = f"{summary.content} [Updated: {additions}]"
        if len(updated_text) > 2000:
            updated_text = updated_text[:1997] + "..."
            should_archive = True

        updated = Memory(
            content=updated_text,
            memory_type=summary.memory_type,
            scope=summary.scope,
            user_id=summary.user_id,
            agent_id=summary.agent_id,
            session_id=summary.session_id,
            tags=summary.tags,
            source=summary.source,
            importance=summary.importance,
            entity_ids=summary.entity_ids + tuple(m.id for m in new_memories),
            is_active=True,
            created_at=summary.created_at,
            updated_at=self._now(),
            last_accessed=summary.last_accessed,
            access_count=summary.access_count,
            embedding=summary.embedding,
        )
        return updated, should_archive

    # ------------------------------------------------------------------
    # 重写入口
    # ------------------------------------------------------------------

    def rewrite(
        self,
        user_id: str | None = None,
        memory_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> RewriteResult:
        result = RewriteResult()

        try:
            # 1. 拉取所有 summary 记忆
            if memory_ids:
                summaries: list[Memory] = []
                for mid in memory_ids:
                    m = self._storage.get_memory(mid)
                    if m and "summary" in m.tags:
                        summaries.append(m)
            else:
                all_mems = self._storage.list_memories(user_id=user_id, limit=10_000)
                summaries = [m for m in all_mems if "summary" in m.tags and m.is_active]

            if not summaries:
                return result

            # 2. 对每个摘要检查增量
            for summary in summaries:
                new_memories = self._find_new_source_memories(summary)
                if not new_memories:
                    continue

                update_result = self._update_summary(summary, new_memories)
                if not update_result:
                    continue
                updated_memory, should_archive = update_result

                # 写入更新后的摘要
                saved_id = self._storage.save_memory(updated_memory)
                _logger.info(
                    "Updated summary %s (was %s) with %d new memories",
                    saved_id,
                    summary.id,
                    len(new_memories),
                )
                result.updated_summaries.append(updated_memory)

                # 若需要归档旧摘要
                if should_archive:
                    try:
                        self._storage.update_memory(
                            summary.id, is_active=False, superseded_by=saved_id
                        )
                        result.memories_archived.append(summary.id)
                        _logger.info("Archived old summary %s", summary.id)
                    except Exception as exc:
                        _logger.warning("Failed to archive old summary %s: %s", summary.id, exc)

        except Exception as exc:
            _logger.exception("SummaryUpdater.rewrite failed")
            result.errors.append(str(exc))

        return result
