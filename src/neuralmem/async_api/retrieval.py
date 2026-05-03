"""AsyncRetrievalEngine — 异步检索引擎，四策略并行检索 + RRF 融合。

将同步 RetrievalEngine 的核心逻辑改写为纯 asyncio 实现：
- 四个检索策略（Semantic/Keyword/Graph/Temporal）通过 asyncio.gather 并发执行
- 使用 AsyncStorage 和 AsyncEmbedder 进行异步 I/O
- 保留 RRF 融合、Cross-Encoder 重排序、解释生成等完整功能
"""
from __future__ import annotations

import asyncio
import logging
import math as _math
from collections.abc import Sequence

from neuralmem.async_api.embedding import AsyncEmbedder
from neuralmem.async_api.storage import AsyncStorage
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import SearchQuery, SearchResult
from neuralmem.graph.knowledge_graph import KnowledgeGraph
from neuralmem.retrieval.fusion import RankedItem, RRFMerger
from neuralmem.retrieval.reranker import CrossEncoderReranker

_logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + _math.exp(-x))


class _AsyncCachingEmbedder:
    """异步 LRU 缓存包装器，拦截重复查询的 encode_one 调用。

    使用 asyncio.Lock 保证并发安全，dict 维护 LRU 顺序。
    """

    def __init__(self, inner: AsyncEmbedder, maxsize: int = 128) -> None:
        self._inner = inner
        self._lock = asyncio.Lock()
        self._cache: dict[str, list[float]] = {}
        self._maxsize = maxsize

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码 — 不缓存（批次大小变化大）。"""
        return await self._inner.encode(texts)

    async def encode_one(self, text: str) -> list[float]:
        """缓存单条查询编码。"""
        async with self._lock:
            if text in self._cache:
                vec = self._cache.pop(text)
                self._cache[text] = vec
                return vec

        vec = await self._inner.encode_one(text)

        async with self._lock:
            self._cache[text] = vec
            while len(self._cache) > self._maxsize:
                self._cache.pop(next(iter(self._cache)))
        return vec

    def cache_clear(self) -> None:
        """清空缓存（同步操作，无需锁因为 asyncio.Lock 在同步上下文不安全）。"""
        self._cache.clear()

    @property
    def cache_info(self) -> tuple[int, int]:
        return len(self._cache), self._maxsize


class AsyncRetrievalEngine:
    """异步四策略检索引擎 — Semantic + Keyword + Graph + Temporal。

    所有策略通过 asyncio.gather 并发执行，充分利用异步 I/O 优势。
    支持可选的 Cross-Encoder 重排序和结果解释生成。
    """

    def __init__(
        self,
        storage: AsyncStorage,
        embedder: AsyncEmbedder,
        graph: KnowledgeGraph,
        config: NeuralMemConfig,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._graph = graph
        self._config = config
        self._merger = RRFMerger(k=60)
        self._reranker = CrossEncoderReranker()

        # 包装 embedder 为异步缓存版本
        cache_size = config.query_embedding_cache_size
        if cache_size > 0:
            self._cached_embedder: AsyncEmbedder = _AsyncCachingEmbedder(
                embedder, maxsize=cache_size
            )
        else:
            self._cached_embedder = embedder

    async def close(self) -> None:
        """关闭引擎，清理缓存。"""
        self.clear_embedding_cache()

    def clear_embedding_cache(self) -> None:
        """清空查询 embedding 缓存。"""
        if isinstance(self._cached_embedder, _AsyncCachingEmbedder):
            self._cached_embedder.cache_clear()

    @property
    def embedding_cache_info(self) -> tuple[int, int] | None:
        """返回缓存信息 (current_size, maxsize)，未启用缓存返回 None。"""
        if isinstance(self._cached_embedder, _AsyncCachingEmbedder):
            return self._cached_embedder.cache_info
        return None

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """执行四策略异步并行检索，返回按相关性排序的结果列表。"""
        limit_per_strategy = query.limit * 3

        # 并发执行四个策略
        async def run_semantic() -> list[RankedItem]:
            from neuralmem.retrieval.semantic import SemanticStrategy

            strategy = SemanticStrategy(self._storage.underlying, self._cached_embedder.underlying)
            return await asyncio.to_thread(
                strategy.retrieve,
                query.query,
                user_id=query.user_id,
                memory_types=list(query.memory_types) if query.memory_types else None,
                limit=limit_per_strategy,
            )

        async def run_keyword() -> list[RankedItem]:
            from neuralmem.retrieval.keyword import KeywordStrategy

            strategy = KeywordStrategy(self._storage.underlying)
            return await asyncio.to_thread(
                strategy.retrieve,
                query.query,
                user_id=query.user_id,
                memory_types=list(query.memory_types) if query.memory_types else None,
                limit=limit_per_strategy,
            )

        async def run_graph() -> list[RankedItem]:
            from neuralmem.retrieval.graph import GraphStrategy

            strategy = GraphStrategy(self._graph)
            return await asyncio.to_thread(
                strategy.retrieve,
                query.query,
                user_id=query.user_id,
                limit=limit_per_strategy,
            )

        async def run_temporal() -> list[RankedItem]:
            from neuralmem.retrieval.temporal import TemporalStrategy

            strategy = TemporalStrategy(self._storage.underlying, self._cached_embedder.underlying)
            return await asyncio.to_thread(
                strategy.retrieve,
                query.query,
                user_id=query.user_id,
                time_range=query.time_range,
                recency_weight=self._config.recency_weight,
                limit=limit_per_strategy,
            )

        tasks = {
            "semantic": asyncio.create_task(run_semantic()),
            "keyword": asyncio.create_task(run_keyword()),
            "graph": asyncio.create_task(run_graph()),
            "temporal": asyncio.create_task(run_temporal()),
        }

        strategy_results: dict[str, list[RankedItem]] = {}
        for name, task in tasks.items():
            try:
                results = await asyncio.wait_for(task, timeout=10.0)
                if results:
                    strategy_results[name] = results
            except asyncio.TimeoutError:
                _logger.warning("Strategy %s timed out after 10s", name)
            except Exception as e:
                _logger.warning("Strategy %s failed: %s", name, e)

        if not strategy_results:
            return []

        # RRF 融合（同步计算，无 I/O）
        merged = self._merger.merge(strategy_results)

        # 取 Top-K 候选
        top_candidates = merged[: query.limit * 2]

        # 可选：Cross-Encoder 重排序
        memory_cache: dict[str, object] = {}
        use_reranker = self._config.enable_reranker and top_candidates
        if use_reranker:
            loaded_for_rerank = []
            for mid, s in top_candidates:
                m = await self._storage.get_memory(mid)
                if m is not None:
                    memory_cache[mid] = m
                    loaded_for_rerank.append((m, s))
            top_candidates = await asyncio.to_thread(
                self._reranker.rerank, query.query, loaded_for_rerank
            )

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
            memory = memory_cache.get(memory_id) or await self._storage.get_memory(memory_id)
            if memory is None:
                continue
            primary_method, all_methods = self._get_methods(memory_id, strategy_results)
            explanation = self._build_explanation(
                primary_method=primary_method,
                all_methods=all_methods,
                score=final_score,
                memory=memory,
            )
            results.append(
                SearchResult(
                    memory=memory,
                    score=final_score,
                    retrieval_method=primary_method,
                    explanation=explanation,
                )
            )

        return results

    def _get_methods(
        self, memory_id: str, strategy_results: dict[str, list[RankedItem]]
    ) -> tuple[str, list[str]]:
        """返回 (primary_method, all_matching_methods)。"""
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
        """构建人类可读的结果解释。"""
        methods_str = "+".join(all_methods)
        parts = [
            f"{primary_method} match (score={score:.2f})",
            f"Found via: {methods_str}",
        ]
        importance = getattr(memory, "importance", None)
        if importance is not None and importance >= 0.7:
            parts.append(f"High importance ({importance:.1f})")
        access_count = getattr(memory, "access_count", None)
        if access_count is not None and access_count > 0:
            parts.append(f"Access count: {access_count}")
        return ". ".join(parts) + "."
