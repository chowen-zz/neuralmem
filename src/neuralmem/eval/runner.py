"""Evaluation runner for NeuralMem."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuralmem.eval.dataset import EvalDataset
from neuralmem.eval.metrics import (
    mrr,
    p95_latency,
    precision_at_k,
    recall_at_k,
    stale_hit_rate,
)


@dataclass
class EvalReport:
    """Complete evaluation report.

    Attributes:
        timestamp: ISO-8601 timestamp of the evaluation run.
        config_snapshot: Dict representation of NeuralMemConfig at eval time.
        k_values: The K values that were evaluated.
        recall_at_k: Dict mapping K -> mean Recall@K.
        precision_at_k: Dict mapping K -> mean Precision@K.
        mrr: Mean Reciprocal Rank across all queries.
        stale_hit_rate: Mean fraction of stale results.
        p95_latency_ms: 95th-percentile retrieval latency in ms.
        num_queries: Number of queries evaluated.
    """

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    stale_hit_rate: float = 0.0
    p95_latency_ms: float = 0.0
    num_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        # Ensure integer keys are preserved (JSON serialises them as strings)
        d["recall_at_k"] = {str(k): v for k, v in self.recall_at_k.items()}
        d["precision_at_k"] = {str(k): v for k, v in self.precision_at_k.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalReport:
        """Deserialize from a dict."""
        rak = {int(k): v for k, v in d.get("recall_at_k", {}).items()}
        pak = {int(k): v for k, v in d.get("precision_at_k", {}).items()}
        return cls(
            timestamp=d.get("timestamp", ""),
            config_snapshot=d.get("config_snapshot", {}),
            k_values=d.get("k_values", []),
            recall_at_k=rak,
            precision_at_k=pak,
            mrr=d.get("mrr", 0.0),
            stale_hit_rate=d.get("stale_hit_rate", 0.0),
            p95_latency_ms=d.get("p95_latency_ms", 0.0),
            num_queries=d.get("num_queries", 0),
        )


def save_report(report: EvalReport, path: str | Path) -> None:
    """Persist an EvalReport to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


def load_report(path: str | Path) -> EvalReport:
    """Load an EvalReport from a JSON file."""
    path = Path(path)
    with path.open() as f:
        return EvalReport.from_dict(json.load(f))


def check_regression(
    current: EvalReport,
    baseline: EvalReport,
    tolerance: float = 0.05,
) -> list[str]:
    """Check whether any metric regressed beyond *tolerance* compared to baseline.

    A metric is flagged as regressed if::

        baseline_value - current_value > tolerance

    Args:
        current: New evaluation report.
        baseline: Previous evaluation report to compare against.
        tolerance: Acceptable drop before flagging a regression.

    Returns:
        List of human-readable metric names that regressed.
    """
    regressed: list[str] = []

    for k, base_val in baseline.recall_at_k.items():
        cur_val = current.recall_at_k.get(k, 0.0)
        if base_val - cur_val > tolerance:
            regressed.append(f"recall@{k}")

    for k, base_val in baseline.precision_at_k.items():
        cur_val = current.precision_at_k.get(k, 0.0)
        if base_val - cur_val > tolerance:
            regressed.append(f"precision@{k}")

    if baseline.mrr - current.mrr > tolerance:
        regressed.append("mrr")

    # For stale_hit_rate, *higher* is worse — but we frame regression as
    # current exceeding baseline beyond tolerance.
    if current.stale_hit_rate - baseline.stale_hit_rate > tolerance:
        regressed.append("stale_hit_rate")

    # For latency, use relative comparison (higher is worse).
    if baseline.p95_latency_ms > 0:
        lat_rel = (current.p95_latency_ms - baseline.p95_latency_ms) / baseline.p95_latency_ms
    else:
        lat_rel = current.p95_latency_ms - baseline.p95_latency_ms
    if lat_rel > tolerance:
        regressed.append("p95_latency_ms")

    return regressed


class EvalRunner:
    """Runs evaluation datasets against a NeuralMem instance.

    Args:
        mem: A NeuralMem instance with a ``recall()`` method.
    """

    def __init__(self, mem: object) -> None:
        self._mem = mem

    def run(
        self,
        dataset: EvalDataset,
        k_values: list[int] | None = None,
        max_k: int | None = None,
    ) -> EvalReport:
        """Evaluate the memory system on the given dataset.

        Args:
            dataset: Evaluation dataset.
            k_values: List of K values for Recall/Precision@K.
            max_k: Maximum number of results to request from recall().

        Returns:
            Populated EvalReport.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        if max_k is None:
            max_k = max(k_values)

        # Accumulate per-query metrics
        recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
        precision_sums: dict[int, float] = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        stale_sum = 0.0
        latencies: list[float] = []

        for query, relevant in zip(dataset.queries, dataset.ground_truth):
            relevant_set = set(relevant)

            start = time.perf_counter()
            results = self._mem.recall(query, limit=max_k)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            retrieved_ids = [r.memory.id for r in results]

            for k in k_values:
                recall_sums[k] += recall_at_k(retrieved_ids, relevant_set, k)
                precision_sums[k] += precision_at_k(retrieved_ids, relevant_set, k)

            mrr_sum += mrr(retrieved_ids, relevant_set)
            stale_sum += stale_hit_rate(results)

        n = len(dataset)
        if n == 0:
            n = 1  # avoid division by zero; all sums are 0 anyway

        # Build config snapshot (best-effort)
        config_snapshot: dict[str, Any] = {}
        if hasattr(self._mem, "config"):
            cfg = self._mem.config
            if hasattr(cfg, "model_dump"):
                config_snapshot = cfg.model_dump()
            elif hasattr(cfg, "__dict__"):
                config_snapshot = dict(cfg.__dict__)

        return EvalReport(
            config_snapshot=config_snapshot,
            k_values=k_values,
            recall_at_k={k: recall_sums[k] / n for k in k_values},
            precision_at_k={k: precision_sums[k] / n for k in k_values},
            mrr=mrr_sum / n,
            stale_hit_rate=stale_sum / n,
            p95_latency_ms=p95_latency(latencies),
            num_queries=len(dataset),
        )
