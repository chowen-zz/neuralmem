"""MemorySummarizer — 持续压缩记忆为摘要。

定期扫描记忆库，将语义相近的记忆聚类并生成高层次摘要记忆。
支持基于向量相似度聚类 + LLM 摘要生成，以及纯规则回退。
"""
from __future__ import annotations

import logging
from typing import Any

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Memory, MemoryType
from neuralmem.rewrite.base import ContextRewriter, LLMCaller, RewriteResult
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

_DEFAULT_SUMMARY_PROMPT = (
    "You are a memory summarizer. Given the following memories, produce a concise, "
    "high-level summary that captures the key facts, themes, and relationships.\n\n"
    "Memories:\n{memories}\n\n"
    "Return JSON: {{\"summary\": \"concise summary text\", "
    "\"key_themes\": [\"theme1\", \"theme2\"], "
    "\"importance\": 0.0-1.0}}"
)


class MemorySummarizer(ContextRewriter):
    """将相关记忆持续压缩为摘要记忆。

    工作流程：
    1. 从存储中获取候选记忆（按 user_id 过滤，排除已有 summary 标签）
    2. 按向量相似度聚类（若存储支持）或按时间窗口分块
    3. 对每个聚类调用 LLM 生成摘要（或规则回退）
    4. 将摘要作为新记忆写入存储，标记源记忆为已归档
    """

    def __init__(
        self,
        config: NeuralMemConfig,
        storage: StorageBackend,
        llm_caller: LLMCaller | None = None,
        cluster_threshold: float = 0.85,
        max_cluster_size: int = 10,
        min_cluster_size: int = 2,
    ) -> None:
        super().__init__(config, storage, llm_caller)
        self._cluster_threshold = cluster_threshold
        self._max_cluster_size = max_cluster_size
        self._min_cluster_size = min_cluster_size

    # ------------------------------------------------------------------
    # 聚类
    # ------------------------------------------------------------------

    def _cluster_memories(self, memories: list[Memory]) -> list[list[Memory]]:
        """将记忆按向量相似度聚类。若向量不可用则按时间窗口分块。"""
        if not memories:
            return []

        # 优先使用向量聚类
        vecs = [m.embedding for m in memories if m.embedding]
        if len(vecs) >= self._min_cluster_size:
            return self._vector_cluster(memories)

        # 回退：按时间窗口分块
        return self._time_window_cluster(memories)

    def _vector_cluster(self, memories: list[Memory]) -> list[list[Memory]]:
        """贪心向量聚类：以第一条记忆为种子，逐条比较余弦相似度。"""
        import numpy as np

        def _cosine(a: list[float], b: list[float]) -> float:
            av = np.array(a, dtype=np.float32)
            bv = np.array(b, dtype=np.float32)
            return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-9))

        unassigned = list(memories)
        clusters: list[list[Memory]] = []
        while unassigned:
            seed = unassigned.pop(0)
            if not seed.embedding:
                continue
            cluster = [seed]
            i = 0
            while i < len(unassigned) and len(cluster) < self._max_cluster_size:
                m = unassigned[i]
                if m.embedding and _cosine(seed.embedding, m.embedding) >= self._cluster_threshold:
                    cluster.append(m)
                    unassigned.pop(i)
                else:
                    i += 1
            if len(cluster) >= self._min_cluster_size:
                clusters.append(cluster)
            else:
                # 太小的簇拆散回未分配（后续时间窗口聚类会收掉）
                unassigned.extend(cluster)
        # 剩余未分配的按时间窗口聚类兜底
        if unassigned:
            clusters.extend(self._time_window_cluster(unassigned))
        return clusters

    def _time_window_cluster(self, memories: list[Memory]) -> list[list[Memory]]:
        """按创建时间排序后每 max_cluster_size 条为一簇。"""
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        clusters: list[list[Memory]] = []
        for i in range(0, len(sorted_memories), self._max_cluster_size):
            chunk = sorted_memories[i : i + self._max_cluster_size]
            if len(chunk) >= self._min_cluster_size:
                clusters.append(chunk)
        return clusters

    # ------------------------------------------------------------------
    # 摘要生成
    # ------------------------------------------------------------------

    def _generate_summary(self, cluster: list[Memory]) -> tuple[str, float] | None:
        """对聚类生成摘要。优先 LLM，否则规则拼接。"""
        texts = [f"- {m.content}" for m in cluster]
        combined = "\n".join(texts)

        if self._llm is not None:
            try:
                return self._llm_summary(combined)
            except Exception as exc:
                _logger.warning("LLM summarization failed (%s), falling back to rules.", exc)

        return self._rule_summary(combined)

    def _llm_summary(self, combined: str) -> tuple[str, float] | None:
        prompt = _DEFAULT_SUMMARY_PROMPT.format(memories=combined[:4000])
        raw = self._call_llm(prompt)
        data = self._parse_json(raw)
        summary = data.get("summary", "")
        if not summary:
            return None
        importance = float(data.get("importance", 0.7))
        importance = max(0.0, min(1.0, importance))
        return summary, importance

    def _rule_summary(self, combined: str) -> tuple[str, float] | None:
        """规则回退：拼接前3条记忆内容作为摘要。"""
        lines = combined.strip().splitlines()
        selected = lines[:3]
        if not selected:
            return None
        summary = "Summary: " + " | ".join(line.lstrip("- ").strip() for line in selected)
        return summary, 0.5

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
            # 1. 拉取候选记忆
            if memory_ids:
                memories: list[Memory] = []
                for mid in memory_ids:
                    m = self._storage.get_memory(mid)
                    if m:
                        memories.append(m)
            else:
                memories = self._storage.list_memories(user_id=user_id, limit=10_000)

            # 排除已有 summary 标签且活跃的摘要记忆（避免重复压缩）
            candidates = [
                m for m in memories
                if m.is_active and "summary" not in m.tags
            ]

            if len(candidates) < self._min_cluster_size:
                return result

            # 2. 聚类
            clusters = self._cluster_memories(candidates)

            # 3. 逐簇生成摘要并写入
            for cluster in clusters:
                summary_tuple = self._generate_summary(cluster)
                if not summary_tuple:
                    continue
                summary_text, importance = summary_tuple
                source_ids = [m.id for m in cluster]

                summary_memory = self._make_summary_memory(
                    content=summary_text,
                    source_memory_ids=source_ids,
                    importance=importance,
                    tags=("summary",),
                    user_id=user_id or cluster[0].user_id,
                )

                # 写入存储
                saved_id = self._storage.save_memory(summary_memory)
                _logger.info(
                    "Created summary %s from %d memories (%s)",
                    saved_id,
                    len(cluster),
                    ", ".join(source_ids[:3]) + ("..." if len(source_ids) > 3 else ""),
                )

                result.new_summaries.append(summary_memory)
                result.memories_archived.extend(source_ids)

                # 可选：标记源记忆为已归档（置 is_active=False）
                for mid in source_ids:
                    try:
                        self._storage.update_memory(mid, is_active=False, superseded_by=saved_id)
                    except Exception as exc:
                        _logger.warning("Failed to archive memory %s: %s", mid, exc)

        except Exception as exc:
            _logger.exception("MemorySummarizer.rewrite failed")
            result.errors.append(str(exc))

        return result
