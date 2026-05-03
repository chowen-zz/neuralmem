"""Tests for LatencyBenchmark."""
from __future__ import annotations

from unittest.mock import MagicMock

from neuralmem.perf.benchmark import (
    BenchmarkReport,
    LatencyBenchmark,
    _percentile,
)


class TestPercentile:
    def test_basic(self):
        # int(5*50/100)-1 = 1, so index 1 -> 2
        assert _percentile([1, 2, 3, 4, 5], 50) == 2

    def test_empty(self):
        assert _percentile([], 90) == 0.0

    def test_single(self):
        assert _percentile([10.0], 99) == 10.0

    def test_p99(self):
        data = list(range(1, 101))
        assert _percentile(data, 99) == 99


class TestBenchmarkReport:
    def test_mean(self):
        r = BenchmarkReport(samples=[0.1, 0.2, 0.3], label="test")
        assert abs(r.mean - 0.2) < 1e-9

    def test_min_max(self):
        r = BenchmarkReport(samples=[0.5, 0.1, 0.9])
        assert r.min == 0.1
        assert r.max == 0.9

    def test_count(self):
        r = BenchmarkReport(samples=[1, 2, 3, 4])
        assert r.count == 4

    def test_percentiles(self):
        r = BenchmarkReport(samples=list(range(1, 101)))
        assert r.p50 == 50
        assert r.p95 == 95
        assert r.p99 == 99

    def test_empty_report(self):
        r = BenchmarkReport()
        assert r.count == 0
        assert r.mean == 0.0
        assert r.max == 0.0
        assert r.min == 0.0
        assert r.p50 == 0.0

    def test_summary_string(self):
        r = BenchmarkReport(samples=[0.01, 0.02, 0.03], label="recall")
        s = r.summary()
        assert "recall" in s
        assert "n=3" in s
        assert "mean=" in s
        assert "p50=" in s
        assert "ms" in s


class TestLatencyBenchmark:
    def test_benchmark_recall_collects_samples(self):
        mem = MagicMock()
        mem.recall.return_value = ["r1"]
        bench = LatencyBenchmark(mem)

        report = bench.benchmark_recall("hello", iterations=5, user_id="u1")
        assert report.count == 5
        assert mem.recall.call_count == 5
        assert all(s > 0 for s in report.samples)

    def test_benchmark_recall_timing_list_populated(self):
        mem = MagicMock()
        mem.recall.return_value = []
        bench = LatencyBenchmark(mem)

        bench.benchmark_recall("q", iterations=3)
        assert len(bench.timings) == 3

    def test_benchmark_recall_label(self):
        mem = MagicMock()
        mem.recall.return_value = []
        report = LatencyBenchmark(mem).benchmark_recall("q", iterations=1)
        assert report.label == "recall"

    def test_benchmark_bulk_write_collects_samples(self):
        mem = MagicMock()
        mem.remember.return_value = ["m"]
        bench = LatencyBenchmark(mem)

        report = bench.benchmark_bulk_write(count=5, user_id="u1")
        assert report.count == 5
        assert mem.remember.call_count == 5
        assert report.label == "bulk_write"

    def test_benchmark_bulk_write_prefix(self):
        mem = MagicMock()
        mem.remember.return_value = ["m"]
        LatencyBenchmark(mem).benchmark_bulk_write(
            count=3, prefix="test"
        )
        calls = mem.remember.call_args_list
        contents = [c.args[0] for c in calls]
        assert contents == ["test item 0", "test item 1", "test item 2"]

    def test_benchmark_bulk_write_timing(self):
        mem = MagicMock()
        mem.remember.return_value = []
        report = LatencyBenchmark(mem).benchmark_bulk_write(count=2)
        assert all(s > 0 for s in report.samples)

    def test_recall_forwards_kwargs(self):
        mem = MagicMock()
        mem.recall.return_value = []
        LatencyBenchmark(mem).benchmark_recall(
            "q", iterations=1, user_id="u1", limit=5
        )
        mem.recall.assert_called_with("q", user_id="u1", limit=5)

    def test_bulk_write_forwards_user_id(self):
        mem = MagicMock()
        mem.remember.return_value = []
        LatencyBenchmark(mem).benchmark_bulk_write(
            count=1, user_id="u42"
        )
        mem.remember.assert_called_once_with(
            "benchmark item 0", user_id="u42"
        )

    def test_timings_cleared_between_runs(self):
        mem = MagicMock()
        mem.recall.return_value = []
        bench = LatencyBenchmark(mem)

        bench.benchmark_recall("q", iterations=3)
        assert len(bench.timings) == 3

        bench.benchmark_recall("q", iterations=5)
        assert len(bench.timings) == 5  # not 8
