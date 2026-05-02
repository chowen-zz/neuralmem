"""四策略并行检索引擎 — NeuralMem 核心差异化"""
from __future__ import annotations

import concurrent.futures
import logging
import math as _math

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.protocols import EmbedderProtocol, GraphStoreProtocol, StorageProtocol
from neuralmem.core.types import SearchQuery, SearchResult
from neuralmem.retrieval.fusion import RankedItem, RRFMerger
from neuralmem.retrieval.graph import GraphStrategy
from neuralmem.retrieval.keyword import KeywordStrategy
from neuralmem.retrieval.reranker import CrossEncoderReranker
from neuralmem.retrieval.semantic import SemanticStrategy
from neuralmem.retrieval.temporal import TemporalStrategy

_logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + _math.exp(-x))


class RetrievalEngine:
    """
    四策略并行检索 + RRF 融合 + 可选 Cross-Encoder 重排序。
    策略:
    1. Semantic  — 向量语义搜索
    2. Keyword   — BM25 关键词匹配
    3. Graph     — 图谱遍历
    4. Temporal  — 时序加权
    """

    def __init__(
        self,
        storage: StorageProtocol,
        embedder: EmbedderProtocol,
        graph: GraphStoreProtocol,
        config: NeuralMemConfig,
    ):
        self._storage = storage
        self._embedder = embedder
        self._graph = graph
        self._config = config
        self._merger = RRFMerger(k=60)
        self._reranker = CrossEncoderReranker()
        self._semantic = SemanticStrategy(storage, embedder)
        self._keyword = KeywordStrategy(storage)
        self._graph_strategy = GraphStrategy(graph)
        self._temporal = TemporalStrategy(storage, embedder)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def close(self) -> None:
        """Shut down the internal thread pool executor."""
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        self.close()

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """执行四策略并行检索，返回按相关性排序的结果列表"""
        limit_per_strategy = query.limit * 3

        # 并行执行四个策略（使用线程池，因为核心 API 全同步）
        strategy_results: dict[str, list[RankedItem]] = {}

        def run_semantic() -> list[RankedItem]:
            return self._semantic.retrieve(
                query.query,
                user_id=query.user_id,
                memory_types=list(query.memory_types) if query.memory_types else None,
                limit=limit_per_strategy,
            )

        def run_keyword() -> list[RankedItem]:
            return self._keyword.retrieve(
                query.query,
                user_id=query.user_id,
                memory_types=list(query.memory_types) if query.memory_types else None,
                limit=limit_per_strategy,
            )

        def run_graph() -> list[RankedItem]:
            return self._graph_strategy.retrieve(
                query.query, user_id=query.user_id, limit=limit_per_strategy
            )

        def run_temporal() -> list[RankedItem]:
            return self._temporal.retrieve(
                query.query,
                user_id=query.user_id,
                time_range=query.time_range,
                recency_weight=self._config.recency_weight,
                limit=limit_per_strategy,
            )

        futures = {
            "semantic": self._executor.submit(run_semantic),
            "keyword": self._executor.submit(run_keyword),
            "graph": self._executor.submit(run_graph),
            "temporal": self._executor.submit(run_temporal),
        }
        for name, future in futures.items():
            try:
                results = future.result(timeout=10.0)
                if results:
                    strategy_results[name] = results
            except Exception as e:
                _logger.warning("Strategy %s failed: %s", name, e)

        if not strategy_results:
            return []

        # RRF 融合
        merged = self._merger.merge(strategy_results)

        # 取 Top-K 候选（先取 limit * 2 做可选重排序）
        top_candidates = merged[: query.limit * 2]

        # 可选：Cross-Encoder 重排序
        memory_cache: dict[str, object] = {}
        use_reranker = self._config.enable_reranker and top_candidates
        if use_reranker:
            loaded_for_rerank = []
            for mid, s in top_candidates:
                m = self._storage.get_memory(mid)
                if m is not None:
                    memory_cache[mid] = m
                    loaded_for_rerank.append((m, s))
            top_candidates = self._reranker.rerank(query.query, loaded_for_rerank)

        # 加载完整 Memory 对象并构建结果
        results: list[SearchResult] = []
        for memory_id, score in top_candidates[: query.limit]:
            if use_reranker:
                final_score = float(_sigmoid(score))
            else:
                final_score = float(score)
            final_score = max(0.0, min(1.0, final_score))
            if final_score < query.min_score:
                continue
            memory = memory_cache.get(memory_id) or self._storage.get_memory(memory_id)
            if memory is None:
                continue
            primary_method, all_methods = self._get_methods(memory_id, strategy_results)
            explanation = self._build_explanation(
                primary_method=primary_method,
                all_methods=all_methods,
                score=final_score,
                memory=memory,
            )
            results.append(SearchResult(
                memory=memory,
                score=final_score,
                retrieval_method=primary_method,
                explanation=explanation,
            ))

        return results

    def _get_primary_method(
        self, memory_id: str, strategy_results: dict[str, list[RankedItem]]
    ) -> str:
        best_score = -1.0
        best_method = "rrf"
        for method, items in strategy_results.items():
            for item in items:
                if item.id == memory_id and item.score > best_score:
                    best_score = item.score
                    best_method = method
        return best_method

    def _get_methods(
        self, memory_id: str, strategy_results: dict[str, list[RankedItem]]
    ) -> tuple[str, list[str]]:
        """Return (primary_method, all_matching_methods) for a memory."""
        best_score = -1.0
        best_method = "rrf"
        all_methods: list[str] = []
        for method, items in strategy_results.items():
            for item in items:
                if item.id == memory_id:
                    all_methods.append(method)
                    if item.score > best_score:
                        best_score = item.score
                        best_method = method
        if not all_methods:
            all_methods = ["rrf"]
        return best_method, all_methods

    def _build_explanation(
        self,
        *,
        primary_method: str,
        all_methods: list[str],
        score: float,
        memory: object,
    ) -> str:
        """Build a human-readable explanation for why a memory was retrieved."""
        methods_str = "+".join(all_methods)
        parts = [
            f"{primary_method} match (score={score:.2f})",
            f"Found via: {methods_str}",
        ]
        # Add importance context if notable
        importance = getattr(memory, "importance", None)
        if importance is not None and importance >= 0.7:
            parts.append(f"High importance ({importance:.1f})")
        # Add access count context
        access_count = getattr(memory, "access_count", None)
        if access_count is not None and access_count > 0:
            parts.append(f"Access count: {access_count}")
        return ". ".join(parts) + "."
