"""Tests for ExtendedLoCoMoBenchmark — fully mock-based."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neuralmem.eval.metrics import mrr, p95_latency, precision_at_k, recall_at_k
from neuralmem.perf.locomo_extended import (
    BenchmarkReport,
    ExtendedLoCoMoBenchmark,
    QueryTiming,
    generate_synthetic_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_search_result(memory_id: str, score: float = 0.8) -> SimpleNamespace:
    """Create a mock SearchResult-like object."""
    mem = SimpleNamespace(id=memory_id)
    return SimpleNamespace(memory=mem, score=score, retrieval_method="semantic")


def _make_mock_mem(num_results: int = 3) -> MagicMock:
    """Create a mock NeuralMem instance that returns predictable results."""
    mem = MagicMock()

    def fake_recall(query: str, limit: int = 10) -> list[SimpleNamespace]:
        return [
            _make_search_result(f"result-{i}", 0.9 - i * 0.1)
            for i in range(min(num_results, limit))
        ]

    mem.recall.side_effect = fake_recall
    return mem


# ---------------------------------------------------------------------------
# Tests for generate_synthetic_dataset
# ---------------------------------------------------------------------------

class TestGenerateSyntheticDataset:

    def test_returns_correct_lengths(self) -> None:
        queries, gt = generate_synthetic_dataset(num_queries=10)
        assert len(queries) == 10
        assert len(gt) == 10

    def test_ground_truth_per_query(self) -> None:
        queries, gt = generate_synthetic_dataset(
            num_queries=5, num_ground_truth=4
        )
        for ids in gt:
            assert len(ids) == 4

    def test_default_num_queries(self) -> None:
        queries, gt = generate_synthetic_dataset()
        assert len(queries) == 20

    def test_queries_are_strings(self) -> None:
        queries, _ = generate_synthetic_dataset(num_queries=3)
        for q in queries:
            assert isinstance(q, str)
            assert len(q) > 0


# ---------------------------------------------------------------------------
# Tests for BenchmarkReport
# ---------------------------------------------------------------------------

class TestBenchmarkReport:

    def test_default_k_values(self) -> None:
        report = BenchmarkReport()
        assert report.k_values == [1, 3, 5, 10]

    def test_to_dict_has_required_keys(self) -> None:
        report = BenchmarkReport(num_queries=5, mrr=0.75)
        d = report.to_dict()
        assert "recall_at_k" in d
        assert "precision_at_k" in d
        assert "mrr" in d
        assert "p50_latency_ms" in d
        assert "p95_latency_ms" in d
        assert "p99_latency_ms" in d
        assert "strategy_timing" in d

    def test_to_dict_excludes_query_timings(self) -> None:
        report = BenchmarkReport(
            _query_timings=[QueryTiming(query_index=0, total_ms=10.0)]
        )
        d = report.to_dict()
        assert "_query_timings" not in d

    def test_to_dict_integer_keys(self) -> None:
        report = BenchmarkReport(
            recall_at_k={1: 0.5, 5: 0.8},
            precision_at_k={1: 0.5, 5: 0.3},
        )
        d = report.to_dict()
        assert "1" in d["recall_at_k"]
        assert "5" in d["recall_at_k"]

    def test_to_markdown_contains_header(self) -> None:
        report = BenchmarkReport(num_queries=10, mrr=0.5)
        md = report.to_markdown()
        assert "# NeuralMem LoCoMo Benchmark Report" in md

    def test_to_markdown_contains_metrics(self) -> None:
        report = BenchmarkReport(
            num_queries=10,
            mrr=0.65,
            recall_at_k={1: 0.5, 5: 0.8},
            precision_at_k={1: 0.5, 5: 0.3},
            p50_latency_ms=12.5,
            p95_latency_ms=45.0,
            p99_latency_ms=90.0,
        )
        md = report.to_markdown()
        assert "MRR" in md
        assert "Recall@1" in md
        assert "Precision@5" in md
        assert "P50" in md
        assert "P95" in md
        assert "P99" in md

    def test_to_markdown_strategy_breakdown(self) -> None:
        report = BenchmarkReport(
            strategy_timing={
                "semantic": 5.2,
                "keyword": 3.1,
                "graph": 1.0,
                "temporal": 0.8,
            }
        )
        md = report.to_markdown()
        assert "Strategy Timing" in md
        assert "semantic" in md
        assert "keyword" in md
        assert "graph" in md
        assert "temporal" in md

    def test_to_markdown_empty_report(self) -> None:
        report = BenchmarkReport()
        md = report.to_markdown()
        assert "**Queries evaluated:** 0" in md


# ---------------------------------------------------------------------------
# Tests for ExtendedLoCoMoBenchmark
# ---------------------------------------------------------------------------

class TestExtendedLoCoMoBenchmark:

    def test_init_defaults(self) -> None:
        mem = MagicMock()
        bench = ExtendedLoCoMoBenchmark(mem)
        assert bench._k_values == [1, 3, 5, 10]

    def test_init_custom_k_values(self) -> None:
        mem = MagicMock()
        bench = ExtendedLoCoMoBenchmark(mem, k_values=[1, 5])
        assert bench._k_values == [1, 5]

    def test_run_returns_report(self) -> None:
        mem = _make_mock_mem()
        bench = ExtendedLoCoMoBenchmark(mem, k_values=[1, 3])
        queries = ["q1", "q2", "q3"]
        gt = [["r1"], ["r2"], ["r3"]]
        report = bench.run(queries, gt)
        assert isinstance(report, BenchmarkReport)
        assert report.num_queries == 3

    def test_run_computes_recall_at_k(self) -> None:
        mem = _make_mock_mem()
        bench = ExtendedLoCoMoBenchmark(mem, k_values=[1, 3])
        queries = ["q1", "q2"]
        gt = [["result-0"], ["result-0", "result-1"]]
        report = bench.run(queries, gt)
        assert 1 in report.recall_at_k
        assert 3 in report.recall_at_k
        assert report.recall_at_k[1] >= 0.0

    def test_run_computes_precision_at_k(self) -> None:
        mem = _make_mock_mem()
        bench = ExtendedLoCoMoBenchmark(mem, k_values=[1, 3])
        queries = ["q1"]
        gt = [["result-0"]]
        report = bench.run(queries, gt)
        assert 1 in report.precision_at_k
        assert 3 in report.precision_at_k

    def test_run_computes_mrr(self) -> None:
        mem = _make_mock_mem()
        bench = ExtendedLoCoMoBenchmark(mem)
        queries = ["q1", "q2"]
        gt = [["result-0"], ["result-1"]]
        report = bench.run(queries, gt)
        assert report.mrr >= 0.0

    def test_run_measures_latency(self) -> None:
        mem = _make_mock_mem()
        bench = ExtendedLoCoMoBenchmark(mem)
        queries = ["q1", "q2"]
        gt = [["r1"], ["r2"]]
        report = bench.run(queries, gt)
        assert report.p50_latency_ms >= 0.0
        assert report.p95_latency_ms >= 0.0
        assert report.p99_latency_ms >= 0.0
        assert report.mean_latency_ms >= 0.0

    def test_run_strategy_timing(self) -> None:
        mem = _make_mock_mem()
        bench = ExtendedLoCoMoBenchmark(mem)
        queries = ["q1"]
        gt = [["result-0"]]
        report = bench.run(queries, gt)
        assert "semantic" in report.strategy_timing
        assert "keyword" in report.strategy_timing
        assert "graph" in report.strategy_timing
        assert "temporal" in report.strategy_timing

    def test_run_mismatched_lengths_raises(self) -> None:
        mem = MagicMock()
        bench = ExtendedLoCoMoBenchmark(mem)
        with pytest.raises(ValueError, match="same length"):
            bench.run(["q1", "q2"], [["r1"]])

    def test_run_empty_queries(self) -> None:
        mem = MagicMock()
        bench = ExtendedLoCoMoBenchmark(mem, k_values=[1])
        report = bench.run([], [])
        assert report.num_queries == 0
        assert report.mrr == 0.0

    def test_load_dataset_synthetic(self) -> None:
        mem = MagicMock()
        bench = ExtendedLoCoMoBenchmark(mem)
        queries, gt = bench.load_dataset()
        assert len(queries) == 20
        assert len(gt) == 20

    def test_load_dataset_from_file(self, tmp_path: Path) -> None:
        mem = MagicMock()
        bench = ExtendedLoCoMoBenchmark(mem)
        dataset = {
            "queries": [
                {"query": "test query", "relevant_ids": ["id1", "id2"]},
                {"query": "another query", "relevant_ids": ["id3"]},
            ],
            "metadata": {},
        }
        fpath = tmp_path / "test_dataset.json"
        fpath.write_text(json.dumps(dataset))
        queries, gt = bench.load_dataset(fpath)
        assert len(queries) == 2
        assert gt[0] == ["id1", "id2"]

    def test_run_from_path_synthetic(self) -> None:
        mem = _make_mock_mem()
        bench = ExtendedLoCoMoBenchmark(mem, k_values=[1, 3])
        report = bench.run_from_path()
        assert report.num_queries == 20
        assert report.mrr >= 0.0

    def test_run_from_path_file(self, tmp_path: Path) -> None:
        mem = _make_mock_mem()
        bench = ExtendedLoCoMoBenchmark(mem, k_values=[1])
        dataset = {
            "queries": [
                {"query": "q1", "relevant_ids": ["result-0"]},
            ],
            "metadata": {},
        }
        fpath = tmp_path / "ds.json"
        fpath.write_text(json.dumps(dataset))
        report = bench.run_from_path(fpath)
        assert report.num_queries == 1

    def test_percentile_empty(self) -> None:
        result = ExtendedLoCoMoBenchmark._percentile([], 95)
        assert result == 0.0

    def test_percentile_single(self) -> None:
        result = ExtendedLoCoMoBenchmark._percentile([42.0], 50)
        assert result == 42.0

    def test_percentile_multiple(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        p50 = ExtendedLoCoMoBenchmark._percentile(vals, 50)
        assert p50 >= 1.0
        assert p50 <= 5.0


# ---------------------------------------------------------------------------
# Tests for metric functions used by benchmark
# ---------------------------------------------------------------------------

class TestMetricFunctions:

    def test_recall_at_k(self) -> None:
        ids = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(ids, relevant, 2) == 1.0

    def test_precision_at_k(self) -> None:
        ids = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert precision_at_k(ids, relevant, 2) == 1.0

    def test_mrr_first_hit(self) -> None:
        assert mrr(["a", "b"], {"a"}) == 1.0

    def test_mrr_second_hit(self) -> None:
        assert mrr(["a", "b"], {"b"}) == 0.5

    def test_mrr_no_hit(self) -> None:
        assert mrr(["a", "b"], {"c"}) == 0.0

    def test_p95_latency(self) -> None:
        vals = [float(i) for i in range(1, 101)]
        result = p95_latency(vals)
        assert result > 0
