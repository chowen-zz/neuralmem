"""Extended LoCoMo benchmark for NeuralMem — multi-strategy timing & report generation."""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuralmem.eval.metrics import (
    mrr,
    p95_latency,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QueryTiming:
    """Per-query latency breakdown by strategy."""
    query_index: int
    total_ms: float
    semantic_ms: float = 0.0
    keyword_ms: float = 0.0
    graph_ms: float = 0.0
    temporal_ms: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report with metrics and timing breakdown."""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    num_queries: int = 0
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])

    # Retrieval quality metrics (averaged across queries)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0

    # Latency percentiles (ms)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0

    # Per-strategy timing breakdown (mean ms)
    strategy_timing: dict[str, float] = field(default_factory=dict)

    # Raw query timings (not serialized by default)
    _query_timings: list[QueryTiming] = field(
        default_factory=list, repr=False
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        d.pop("_query_timings", None)
        d["recall_at_k"] = {str(k): v for k, v in self.recall_at_k.items()}
        d["precision_at_k"] = {str(k): v for k, v in self.precision_at_k.items()}
        return d

    def to_markdown(self) -> str:
        """Generate a human-readable markdown report."""
        lines: list[str] = []
        lines.append("# NeuralMem LoCoMo Benchmark Report")
        lines.append("")
        lines.append(f"**Generated:** {self.timestamp}")
        lines.append(f"**Queries evaluated:** {self.num_queries}")
        lines.append("")

        # Quality metrics
        lines.append("## Retrieval Quality")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| MRR | {self.mrr:.4f} |")
        for k in self.k_values:
            r = self.recall_at_k.get(k, 0.0)
            p = self.precision_at_k.get(k, 0.0)
            lines.append(f"| Recall@{k} | {r:.4f} |")
            lines.append(f"| Precision@{k} | {p:.4f} |")
        lines.append("")

        # Latency
        lines.append("## Latency (ms)")
        lines.append("")
        lines.append("| Percentile | Value |")
        lines.append("|------------|-------|")
        lines.append(f"| P50 | {self.p50_latency_ms:.2f} |")
        lines.append(f"| P95 | {self.p95_latency_ms:.2f} |")
        lines.append(f"| P99 | {self.p99_latency_ms:.2f} |")
        lines.append(f"| Mean | {self.mean_latency_ms:.2f} |")
        lines.append("")

        # Strategy breakdown
        if self.strategy_timing:
            lines.append("## Strategy Timing Breakdown (mean ms)")
            lines.append("")
            lines.append("| Strategy | Mean Latency (ms) |")
            lines.append("|----------|-------------------|")
            for strat, ms in sorted(self.strategy_timing.items()):
                lines.append(f"| {strat} | {ms:.2f} |")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    num_queries: int = 20,
    num_ground_truth: int = 3,
) -> tuple[list[str], list[list[str]]]:
    """Generate synthetic query/ground-truth pairs for benchmarking.

    Returns:
        Tuple of (queries, ground_truth_ids).
    """
    topics = [
        ("What is the user's favorite programming language?", "lang-pref"),
        ("When did the user start their current job?", "job-start"),
        ("What dietary restrictions does the user have?", "diet-rest"),
        ("What are the user's hobbies?", "hobbies"),
        ("Where does the user live?", "location"),
        ("What is the user's preferred meeting schedule?", "schedule"),
        ("What frameworks does the user prefer?", "frameworks"),
        ("What is the user's timezone?", "timezone"),
        ("What books has the user recommended?", "books"),
        ("What is the user's name?", "name"),
    ]
    queries: list[str] = []
    ground_truth: list[list[str]] = []

    for i in range(num_queries):
        topic_q, topic_id = topics[i % len(topics)]
        queries.append(topic_q)
        ids = [f"{topic_id}-{j}" for j in range(num_ground_truth)]
        ground_truth.append(ids)

    return queries, ground_truth


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class ExtendedLoCoMoBenchmark:
    """Extended LoCoMo benchmark with per-strategy timing and report generation.

    This benchmark evaluates NeuralMem's retrieval quality (Recall@K,
    Precision@K, MRR) while measuring latency percentiles (P50/P95/P99)
    and per-strategy timing breakdown (semantic/keyword/graph/temporal).

    Args:
        mem: A NeuralMem instance with a ``recall()`` method.
        k_values: K values for Recall@K and Precision@K.
    """

    def __init__(
        self,
        mem: object,
        k_values: list[int] | None = None,
    ) -> None:
        self._mem = mem
        self._k_values = k_values or [1, 3, 5, 10]

    def load_dataset(
        self,
        dataset_path: str | Path | None = None,
    ) -> tuple[list[str], list[list[str]]]:
        """Load or generate a LoCoMo-compatible dataset.

        Args:
            dataset_path: Path to a JSON dataset file. If None, generates
                synthetic data.

        Returns:
            Tuple of (queries, ground_truth_ids).
        """
        if dataset_path is not None:
            path = Path(dataset_path)
            import json

            with path.open() as f:
                raw = json.load(f)

            queries: list[str] = []
            ground_truth: list[list[str]] = []
            for entry in raw["queries"]:
                queries.append(entry["query"])
                ground_truth.append(list(entry.get("relevant_ids", [])))
            return queries, ground_truth

        return generate_synthetic_dataset()

    def run(
        self,
        queries: list[str],
        ground_truth: list[list[str]],
        max_k: int | None = None,
    ) -> BenchmarkReport:
        """Run the benchmark over all queries.

        Args:
            queries: List of query strings.
            ground_truth: Per-query list of relevant memory IDs.
            max_k: Max results to request from recall(). Defaults to max(k_values).

        Returns:
            Populated BenchmarkReport.
        """
        if len(queries) != len(ground_truth):
            raise ValueError(
                f"queries ({len(queries)}) and ground_truth "
                f"({len(ground_truth)}) must have the same length"
            )

        if max_k is None:
            max_k = max(self._k_values)

        recall_sums: dict[int, float] = {k: 0.0 for k in self._k_values}
        precision_sums: dict[int, float] = {k: 0.0 for k in self._k_values}
        mrr_sum = 0.0
        all_latencies: list[float] = []
        query_timings: list[QueryTiming] = []

        # Per-strategy timing accumulators
        strategy_totals: dict[str, float] = {
            "semantic": 0.0,
            "keyword": 0.0,
            "graph": 0.0,
            "temporal": 0.0,
        }

        n = len(queries) if queries else 1

        for idx, (query, relevant) in enumerate(
            zip(queries, ground_truth)
        ):
            relevant_set = set(relevant)

            # Time the full recall call
            start = time.perf_counter()
            results = self._mem.recall(query, limit=max_k)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            all_latencies.append(elapsed_ms)

            retrieved_ids = [
                r.memory.id if hasattr(r, "memory") else str(r)
                for r in results
            ]

            for k in self._k_values:
                recall_sums[k] += recall_at_k(
                    retrieved_ids, relevant_set, k
                )
                precision_sums[k] += precision_at_k(
                    retrieved_ids, relevant_set, k
                )

            mrr_sum += mrr(retrieved_ids, relevant_set)

            # Strategy timing: try to extract from retrieval_method
            strat_ms = self._estimate_strategy_times(results, elapsed_ms)
            for strat_name, ms_val in strat_ms.items():
                strategy_totals[strat_name] += ms_val

            query_timings.append(QueryTiming(
                query_index=idx,
                total_ms=elapsed_ms,
                semantic_ms=strat_ms.get("semantic", 0.0),
                keyword_ms=strat_ms.get("keyword", 0.0),
                graph_ms=strat_ms.get("graph", 0.0),
                temporal_ms=strat_ms.get("temporal", 0.0),
            ))

        # Compute strategy means
        strategy_means: dict[str, float] = {}
        for strat_name, total in strategy_totals.items():
            strategy_means[strat_name] = total / n

        # Build report
        report = BenchmarkReport(
            num_queries=len(queries),
            k_values=list(self._k_values),
            recall_at_k={k: recall_sums[k] / n for k in self._k_values},
            precision_at_k={
                k: precision_sums[k] / n for k in self._k_values
            },
            mrr=mrr_sum / n,
            p50_latency_ms=self._percentile(all_latencies, 50),
            p95_latency_ms=p95_latency(all_latencies),
            p99_latency_ms=self._percentile(all_latencies, 99),
            mean_latency_ms=(
                statistics.mean(all_latencies) if all_latencies else 0.0
            ),
            strategy_timing=strategy_means,
            _query_timings=query_timings,
        )

        return report

    def run_from_path(
        self,
        dataset_path: str | Path | None = None,
        max_k: int | None = None,
    ) -> BenchmarkReport:
        """Convenience: load dataset and run in one call.

        Args:
            dataset_path: Path to dataset JSON, or None for synthetic.
            max_k: Max results to request from recall().

        Returns:
            Populated BenchmarkReport.
        """
        queries, ground_truth = self.load_dataset(dataset_path)
        return self.run(queries, ground_truth, max_k=max_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_strategy_times(
        results: list[object], total_ms: float
    ) -> dict[str, float]:
        """Estimate per-strategy timing from retrieval_method metadata.

        When per-strategy timing is not available from the engine, we
        distribute the total time equally among strategies that contributed
        results.
        """
        seen_methods: set[str] = set()
        for r in results:
            method = getattr(r, "retrieval_method", "rrf")
            if method and method != "rrf":
                seen_methods.add(method)

        if not seen_methods:
            # Distribute equally across all four strategies
            share = total_ms / 4.0
            return {
                "semantic": share,
                "keyword": share,
                "graph": share,
                "temporal": share,
            }

        # Distribute equally among contributing strategies
        share = total_ms / len(seen_methods)
        result: dict[str, float] = {
            "semantic": 0.0,
            "keyword": 0.0,
            "graph": 0.0,
            "temporal": 0.0,
        }
        for m in seen_methods:
            if m in result:
                result[m] = share
        return result

    @staticmethod
    def _percentile(
        values: list[float], percentile: float
    ) -> float:
        """Compute percentile using nearest-rank method."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = math.ceil(percentile / 100.0 * len(sorted_vals)) - 1
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return sorted_vals[idx]
