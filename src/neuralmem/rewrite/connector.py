"""ConnectionFinder — 发现跨领域隐性关联。

扫描记忆库，通过向量相似度 + LLM 推理发现不同主题/领域记忆之间的隐性连接，
并在图谱中记录这些连接关系。
"""
from __future__ import annotations

import logging
from typing import Any

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Memory
from neuralmem.rewrite.base import ContextRewriter, LLMCaller, RewriteResult
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

_DEFAULT_CONNECTION_PROMPT = (
    "You are a connection finder. Given two memories from potentially different domains, "
    "determine if there is a meaningful hidden connection between them.\n\n"
    "Memory A: {memory_a}\n\n"
    "Memory B: {memory_b}\n\n"
    "Return JSON: {{\"connected\": true|false, "
    "\"connection_type\": \"causal|analogical|compositional|temporal|thematic\", "
    "\"explanation\": \"brief explanation\", "
    "\"strength\": 0.0-1.0}}"
)


class ConnectionFinder(ContextRewriter):
    """发现记忆之间的隐性连接。

    工作流程：
    1. 从存储中获取候选记忆（排除 summary）
    2. 对每对记忆计算向量相似度，筛选高相似度但标签差异大的候选对
    3. 对候选对调用 LLM 判断是否存在隐性连接
    4. 返回连接结果（供上层写入图谱或关系存储）
    """

    def __init__(
        self,
        config: NeuralMemConfig,
        storage: StorageBackend,
        llm_caller: LLMCaller | None = None,
        similarity_threshold: float = 0.70,
        max_pairs_per_run: int = 50,
    ) -> None:
        super().__init__(config, storage, llm_caller)
        self._similarity_threshold = similarity_threshold
        self._max_pairs_per_run = max_pairs_per_run

    # ------------------------------------------------------------------
    # 候选对生成
    # ------------------------------------------------------------------

    def _generate_candidate_pairs(self, memories: list[Memory]) -> list[tuple[Memory, Memory, float]]:
        """基于向量相似度生成候选记忆对，优先跨标签/跨类型的组合。"""
        import numpy as np

        def _cosine(a: list[float], b: list[float]) -> float:
            av = np.array(a, dtype=np.float32)
            bv = np.array(b, dtype=np.float32)
            return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-9))

        # 只保留有向量的记忆
        mems_with_vec = [m for m in memories if m.embedding]
        if len(mems_with_vec) < 2:
            return []

        scored_pairs: list[tuple[Memory, Memory, float]] = []
        n = len(mems_with_vec)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = mems_with_vec[i], mems_with_vec[j]
                sim = _cosine(a.embedding, b.embedding)
                if sim >= self._similarity_threshold:
                    # 跨标签/跨类型加分
                    cross_tag = bool(set(a.tags) ^ set(b.tags))
                    cross_type = a.memory_type != b.memory_type
                    bonus = 0.05 * int(cross_tag) + 0.05 * int(cross_type)
                    scored_pairs.append((a, b, sim + bonus))

        # 按得分降序，取前 N
        scored_pairs.sort(key=lambda x: x[2], reverse=True)
        return scored_pairs[: self._max_pairs_per_run]

    # ------------------------------------------------------------------
    # 连接判定
    # ------------------------------------------------------------------

    def _evaluate_connection(
        self, a: Memory, b: Memory
    ) -> dict[str, Any] | None:
        """使用 LLM 或规则判定两记忆之间是否存在隐性连接。"""
        if self._llm is not None:
            try:
                return self._llm_evaluate(a, b)
            except Exception as exc:
                _logger.warning("LLM connection evaluation failed (%s), using rules.", exc)

        return self._rule_evaluate(a, b)

    def _llm_evaluate(self, a: Memory, b: Memory) -> dict[str, Any] | None:
        prompt = _DEFAULT_CONNECTION_PROMPT.format(
            memory_a=a.content[:1500],
            memory_b=b.content[:1500],
        )
        raw = self._call_llm(prompt)
        data = self._parse_json(raw)
        if not data.get("connected"):
            return None
        return {
            "source_id": a.id,
            "target_id": b.id,
            "connection_type": data.get("connection_type", "thematic"),
            "explanation": data.get("explanation", ""),
            "strength": max(0.0, min(1.0, float(data.get("strength", 0.5)))),
        }

    def _rule_evaluate(self, a: Memory, b: Memory) -> dict[str, Any] | None:
        """规则回退：若两记忆有共享实体或共享关键词则认为有关联。"""
        shared_entities = set(a.entity_ids) & set(b.entity_ids)
        if shared_entities:
            return {
                "source_id": a.id,
                "target_id": b.id,
                "connection_type": "compositional",
                "explanation": f"Shared entities: {', '.join(shared_entities)}",
                "strength": 0.6,
            }

        # 简单关键词重叠
        a_words = set(a.content.lower().split())
        b_words = set(b.content.lower().split())
        overlap = a_words & b_words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "and", "in", "on", "at", "for", "with", "as", "by", "it", "this", "that"}
        meaningful = overlap - stopwords
        if len(meaningful) >= 3:
            return {
                "source_id": a.id,
                "target_id": b.id,
                "connection_type": "thematic",
                "explanation": f"Shared keywords: {', '.join(list(meaningful)[:5])}",
                "strength": min(0.5 + 0.05 * len(meaningful), 0.9),
            }
        return None

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

            # 排除 summary 记忆（避免摘要之间互相连接）
            candidates = [m for m in memories if m.is_active and "summary" not in m.tags]
            if len(candidates) < 2:
                return result

            # 2. 生成候选对
            pairs = self._generate_candidate_pairs(candidates)
            if not pairs:
                return result

            # 3. 逐对评估
            for a, b, score in pairs:
                conn = self._evaluate_connection(a, b)
                if conn:
                    result.connections_found.append(conn)
                    _logger.info(
                        "Found %s connection between %s and %s (strength %.2f)",
                        conn["connection_type"],
                        a.id,
                        b.id,
                        conn["strength"],
                    )

        except Exception as exc:
            _logger.exception("ConnectionFinder.rewrite failed")
            result.errors.append(str(exc))

        return result
