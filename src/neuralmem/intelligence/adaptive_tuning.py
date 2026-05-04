"""Adaptive parameter tuning based on query patterns."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
import time


class QueryPattern(Enum):
    SEQUENTIAL = auto()
    RANDOM = auto()
    BATCH = auto()
    OLTP = auto()
    OLAP = auto()


@dataclass
class ParameterSet:
    rrf_k: float = 60.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    vector_dim: int = 384
    cache_strategy: str = "lru"
    prefetch_depth: int = 3


@dataclass
class TuningResult:
    pattern: QueryPattern
    previous_params: ParameterSet
    new_params: ParameterSet
    confidence: float
    timestamp: float = field(default_factory=time.time)


class AdaptiveTuningEngine:
    """Auto-tune memory parameters based on observed query patterns."""

    def __init__(self, param_callback: Callable[[ParameterSet], None] | None = None) -> None:
        self._callback = param_callback
        self._history: list[TuningResult] = []
        self._current = ParameterSet()
        self._query_log: list[dict] = []

    def record_query(self, query_type: str, latency_ms: float, result_count: int, metadata: dict | None = None) -> None:
        self._query_log.append({
            "type": query_type,
            "latency_ms": latency_ms,
            "result_count": result_count,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

    def classify_pattern(self, window: int = 100) -> QueryPattern:
        recent = self._query_log[-window:]
        if not recent:
            return QueryPattern.RANDOM
        types = [q["type"] for q in recent]
        if len(set(types)) == 1 and types[0] == "batch":
            return QueryPattern.BATCH
        if all(q["latency_ms"] < 10 for q in recent):
            return QueryPattern.OLTP
        if any(q["result_count"] > 100 for q in recent):
            return QueryPattern.OLAP
        # Check sequential access pattern
        timestamps = [q["timestamp"] for q in recent]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if intervals and max(intervals) - min(intervals) < 1.0:
            return QueryPattern.SEQUENTIAL
        return QueryPattern.RANDOM

    def recommend_params(self, pattern: QueryPattern | None = None) -> ParameterSet:
        pattern = pattern or self.classify_pattern()
        params = ParameterSet()
        if pattern == QueryPattern.SEQUENTIAL:
            params.cache_strategy = "predictive"
            params.prefetch_depth = 5
            params.rrf_k = 40.0
        elif pattern == QueryPattern.BATCH:
            params.cache_strategy = "lfu"
            params.prefetch_depth = 1
            params.rrf_k = 80.0
        elif pattern == QueryPattern.OLTP:
            params.bm25_k1 = 1.2
            params.bm25_b = 0.6
            params.vector_dim = 256
        elif pattern == QueryPattern.OLAP:
            params.bm25_k1 = 2.0
            params.bm25_b = 0.9
            params.vector_dim = 512
            params.rrf_k = 100.0
        return params

    def apply_tuning(self, pattern: QueryPattern | None = None) -> TuningResult:
        pattern = pattern or self.classify_pattern()
        previous = ParameterSet(
            rrf_k=self._current.rrf_k,
            bm25_k1=self._current.bm25_k1,
            bm25_b=self._current.bm25_b,
            vector_dim=self._current.vector_dim,
            cache_strategy=self._current.cache_strategy,
            prefetch_depth=self._current.prefetch_depth,
        )
        new_params = self.recommend_params(pattern)
        self._current = new_params
        result = TuningResult(
            pattern=pattern,
            previous_params=previous,
            new_params=new_params,
            confidence=0.85,
        )
        self._history.append(result)
        if self._callback:
            self._callback(new_params)
        return result

    def get_current_params(self) -> ParameterSet:
        return ParameterSet(
            rrf_k=self._current.rrf_k,
            bm25_k1=self._current.bm25_k1,
            bm25_b=self._current.bm25_b,
            vector_dim=self._current.vector_dim,
            cache_strategy=self._current.cache_strategy,
            prefetch_depth=self._current.prefetch_depth,
        )

    def get_tuning_history(self) -> list[TuningResult]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._query_log.clear()
        self._current = ParameterSet()
