"""Auto index optimization based on query logs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class IndexRecommendation:
    columns: list[str]
    index_type: str = "composite"
    confidence: float = 0.0
    expected_speedup: float = 1.0
    reason: str = ""


@dataclass
class QueryLogEntry:
    query: str
    filters: dict[str, Any]
    sort_by: list[str]
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class IndexOptimizer:
    """Analyze query logs and recommend index improvements."""

    def __init__(self) -> None:
        self._query_logs: list[QueryLogEntry] = []
        self._indexes: list[list[str]] = []
        self._recommendations: list[IndexRecommendation] = []

    def record_query(self, filters: dict[str, Any], sort_by: list[str], latency_ms: float) -> None:
        self._query_logs.append(QueryLogEntry(
            query=str(filters),
            filters=filters,
            sort_by=sort_by,
            latency_ms=latency_ms,
        ))

    def analyze_filter_patterns(self, window: int = 1000) -> dict[str, int]:
        recent = self._query_logs[-window:]
        column_freq: dict[str, int] = {}
        for entry in recent:
            for col in entry.filters:
                column_freq[col] = column_freq.get(col, 0) + 1
            for col in entry.sort_by:
                column_freq[col] = column_freq.get(col, 0) + 1
        return column_freq

    def recommend_indexes(self, min_frequency: int = 10) -> list[IndexRecommendation]:
        freq = self.analyze_filter_patterns()
        # Sort by frequency
        sorted_cols = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        recommendations: list[IndexRecommendation] = []
        # Recommend composite index for top 2-3 columns
        top_cols = [c for c, f in sorted_cols if f >= min_frequency]
        if len(top_cols) >= 2:
            recommendations.append(IndexRecommendation(
                columns=top_cols[:3],
                index_type="composite",
                confidence=min(0.95, 0.5 + len(top_cols) * 0.1),
                expected_speedup=2.0,
                reason=f"High frequency filter columns: {top_cols[:3]}",
            ))
        self._recommendations = recommendations
        return recommendations

    def estimate_speedup(self, recommendation: IndexRecommendation) -> float:
        # Simple model: composite index on N columns gives Nx speedup for matching queries
        return len(recommendation.columns) * 1.5

    def get_query_stats(self) -> dict:
        if not self._query_logs:
            return {"total_queries": 0, "avg_latency_ms": 0.0}
        latencies = [q.latency_ms for q in self._query_logs]
        return {
            "total_queries": len(self._query_logs),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0] if latencies else 0.0,
        }

    def reset(self) -> None:
        self._query_logs.clear()
        self._indexes.clear()
        self._recommendations.clear()
