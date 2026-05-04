"""Root cause analysis for memory system failures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class CausalChain:
    root_cause: str
    symptoms: list[str]
    contributing_factors: list[str]
    confidence: float
    recommended_fix: str


@dataclass
class AttributionResult:
    factor: str
    impact_score: float  # 0-1
    evidence: list[str]


class RootCauseAnalyzer:
    """Analyze failures and generate causal chains."""

    def __init__(self) -> None:
        self._knowledge_base: dict[str, list[str]] = {
            "high_latency": ["cache_miss", "index_fragmentation", "network_slow"],
            "recall_failure": ["embedding_mismatch", "index_corrupt", "filter_error"],
            "memory_leak": ["unclosed_connections", "circular_references", "large_vectors"],
            "data_loss": ["disk_full", "write_error", "sync_failure"],
        }
        self._history: list[CausalChain] = []

    def analyze(self, symptom: str, metrics: dict[str, Any]) -> CausalChain:
        possible_causes = self._knowledge_base.get(symptom, ["unknown"])
        # Score causes based on metrics
        scored: list[tuple[str, float]] = []
        for cause in possible_causes:
            score = 0.5
            if cause == "cache_miss" and metrics.get("cache_hit_rate", 1.0) < 0.3:
                score = 0.9
            if cause == "index_fragmentation" and metrics.get("index_size_ratio", 1.0) > 2.0:
                score = 0.8
            if cause == "embedding_mismatch" and metrics.get("vector_dim_mismatch", False):
                score = 0.95
            scored.append((cause, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        root = scored[0][0] if scored else "unknown"
        chain = CausalChain(
            root_cause=root,
            symptoms=[symptom],
            contributing_factors=[c for c, _ in scored[1:3]],
            confidence=scored[0][1] if scored else 0.5,
            recommended_fix=self._get_fix(root),
        )
        self._history.append(chain)
        return chain

    def _get_fix(self, cause: str) -> str:
        fixes = {
            "cache_miss": "Increase cache size or switch to predictive prefetching",
            "index_fragmentation": "Run index optimization or rebuild",
            "embedding_mismatch": "Re-embed all memories with consistent model",
            "network_slow": "Check network latency or enable local mode",
            "unknown": "Investigate logs and metrics manually",
        }
        return fixes.get(cause, "No known fix - escalate to engineering")

    def multi_factor_attribution(self, outcome: str, factors: dict[str, float]) -> list[AttributionResult]:
        total = sum(factors.values())
        if total == 0:
            return []
        results = []
        for factor, score in factors.items():
            results.append(AttributionResult(
                factor=factor,
                impact_score=round(score / total, 2),
                evidence=[f"{factor} score: {score}"],
            ))
        return sorted(results, key=lambda x: x.impact_score, reverse=True)

    def get_history(self) -> list[CausalChain]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
