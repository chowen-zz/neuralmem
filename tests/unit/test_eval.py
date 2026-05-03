"""Unit tests for the NeuralMem eval framework."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neuralmem.eval.dataset import EvalDataset, load_dataset
from neuralmem.eval.metrics import (
    mrr,
    p95_latency,
    precision_at_k,
    recall_at_k,
    stale_hit_rate,
)
from neuralmem.eval.runner import EvalReport, EvalRunner, check_regression, load_report, save_report

# ---------------------------------------------------------------------------
# Fixtures specific to eval tests
# ---------------------------------------------------------------------------

@dataclass
class _FakeMemory:
    id: str
    is_active: bool = True
    expires_at: datetime | None = None


@dataclass
class _FakeSearchResult:
    memory: _FakeMemory
    score: float = 0.9
    retrieval_method: str = "test"
    explanation: str | None = None


@pytest.fixture
def eval_dataset_json(tmp_path: Path) -> Path:
    """Write a small eval dataset to disk and return its path."""
    data = {
        "queries": [
            {"query": "What is Python?", "relevant_ids": ["m1", "m2"]},
            {"query": "TypeScript frontend", "relevant_ids": ["m3"]},
            {"query": "No matches query", "relevant_ids": []},
        ],
        "metadata": {"version": "1.0", "description": "test dataset"},
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(data))
    return p


# ===================================================================
# Metric tests (pure math — no NeuralMem dependency)
# ===================================================================

class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_no_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_k_zero(self) -> None:
        assert recall_at_k(["a"], {"a"}, 0) == 0.0

    def test_k_larger_than_results(self) -> None:
        assert recall_at_k(["a"], {"a", "b"}, 5) == 0.5

    def test_empty_results(self) -> None:
        assert recall_at_k([], {"a"}, 3) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_half_relevant(self) -> None:
        assert precision_at_k(["a", "x"], {"a", "b"}, 2) == 0.5

    def test_k_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_results(self) -> None:
        assert precision_at_k([], {"a"}, 3) == 0.0

    def test_no_relevant_set(self) -> None:
        # Precision doesn't depend on len(relevant); it's hits/k
        assert precision_at_k(["a", "b"], set(), 2) == 0.0


class TestMRR:
    def test_first_result_relevant(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_result_relevant(self) -> None:
        assert mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)

    def test_third_result_relevant(self) -> None:
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant(self) -> None:
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_relevant(self) -> None:
        assert mrr(["a"], set()) == 0.0

    def test_multiple_relevant_returns_first(self) -> None:
        # "b" is at rank 1, "a" at rank 3; should return 1/1
        assert mrr(["b", "x", "a"], {"a", "b"}) == 1.0


class TestStaleHitRate:
    def test_all_active(self) -> None:
        results = [
            _FakeSearchResult(memory=_FakeMemory(id="1")),
            _FakeSearchResult(memory=_FakeMemory(id="2")),
        ]
        assert stale_hit_rate(results) == 0.0

    def test_one_inactive(self) -> None:
        results = [
            _FakeSearchResult(memory=_FakeMemory(id="1", is_active=False)),
            _FakeSearchResult(memory=_FakeMemory(id="2")),
        ]
        assert stale_hit_rate(results) == 0.5

    def test_expired(self) -> None:
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        results = [
            _FakeSearchResult(memory=_FakeMemory(id="1", expires_at=past)),
        ]
        assert stale_hit_rate(results) == 1.0

    def test_not_yet_expired(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        results = [
            _FakeSearchResult(memory=_FakeMemory(id="1", expires_at=future)),
        ]
        assert stale_hit_rate(results) == 0.0

    def test_empty(self) -> None:
        assert stale_hit_rate([]) == 0.0

    def test_inactive_overrides_expiry(self) -> None:
        """is_active=False counts as stale regardless of expires_at."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        results = [
            _FakeSearchResult(memory=_FakeMemory(id="1", is_active=False, expires_at=future)),
        ]
        assert stale_hit_rate(results) == 1.0


class TestP95Latency:
    def test_single_value(self) -> None:
        assert p95_latency([10.0]) == 10.0

    def test_empty(self) -> None:
        assert p95_latency([]) == 0.0

    def test_hundred_values(self) -> None:
        vals = [float(i) for i in range(1, 101)]
        # 95th percentile of 1..100 → ceil(0.95*100) = 95 → value 95.0
        assert p95_latency(vals) == 95.0

    def test_two_values(self) -> None:
        # ceil(0.95*2) = 2 → index 1 → second value
        assert p95_latency([10.0, 20.0]) == 20.0


# ===================================================================
# Dataset tests
# ===================================================================

class TestEvalDataset:
    def test_load_dataset(self, eval_dataset_json: Path) -> None:
        ds = load_dataset(eval_dataset_json)
        assert len(ds) == 3
        assert ds.queries[0] == "What is Python?"
        assert ds.ground_truth[0] == ["m1", "m2"]
        assert ds.metadata["version"] == "1.0"

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            EvalDataset(queries=["q1", "q2"], ground_truth=[["a"]])

    def test_empty_dataset(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"queries": [], "metadata": {}}))
        ds = load_dataset(p)
        assert len(ds) == 0


# ===================================================================
# EvalReport serialization tests
# ===================================================================

class TestEvalReport:
    def test_round_trip(self) -> None:
        report = EvalReport(
            timestamp="2025-01-01T00:00:00+00:00",
            config_snapshot={"db_path": "/tmp/test.db"},
            k_values=[1, 5],
            recall_at_k={1: 0.8, 5: 0.95},
            precision_at_k={1: 0.8, 5: 0.3},
            mrr=0.75,
            stale_hit_rate=0.02,
            p95_latency_ms=42.5,
            num_queries=100,
        )
        d = report.to_dict()
        restored = EvalReport.from_dict(d)
        assert restored.timestamp == report.timestamp
        assert restored.recall_at_k == report.recall_at_k
        assert restored.precision_at_k == report.precision_at_k
        assert restored.mrr == report.mrr
        assert restored.stale_hit_rate == report.stale_hit_rate
        assert restored.p95_latency_ms == report.p95_latency_ms

    def test_save_load_report(self, tmp_path: Path) -> None:
        report = EvalReport(mrr=0.5, num_queries=10)
        path = tmp_path / "report.json"
        save_report(report, path)
        loaded = load_report(path)
        assert loaded.mrr == 0.5
        assert loaded.num_queries == 10


class TestCheckRegression:
    def test_no_regression(self) -> None:
        baseline = EvalReport(
            recall_at_k={5: 0.9},
            precision_at_k={5: 0.5},
            mrr=0.8,
            stale_hit_rate=0.01,
            p95_latency_ms=50.0,
        )
        current = EvalReport(
            recall_at_k={5: 0.88},
            precision_at_k={5: 0.48},
            mrr=0.78,
            stale_hit_rate=0.03,
            p95_latency_ms=52.0,
        )
        assert check_regression(current, baseline, tolerance=0.05) == []

    def test_recall_regressed(self) -> None:
        baseline = EvalReport(recall_at_k={5: 0.9})
        current = EvalReport(recall_at_k={5: 0.7})
        regressed = check_regression(current, baseline, tolerance=0.05)
        assert "recall@5" in regressed

    def test_mrr_regressed(self) -> None:
        baseline = EvalReport(mrr=0.9)
        current = EvalReport(mrr=0.6)
        regressed = check_regression(current, baseline, tolerance=0.05)
        assert "mrr" in regressed

    def test_stale_rate_increased(self) -> None:
        baseline = EvalReport(stale_hit_rate=0.01)
        current = EvalReport(stale_hit_rate=0.15)
        regressed = check_regression(current, baseline, tolerance=0.05)
        assert "stale_hit_rate" in regressed

    def test_latency_increased(self) -> None:
        baseline = EvalReport(p95_latency_ms=50.0)
        current = EvalReport(p95_latency_ms=100.0)
        regressed = check_regression(current, baseline, tolerance=0.05)
        assert "p95_latency_ms" in regressed


# ===================================================================
# EvalRunner integration test (uses mem_with_mock fixture)
# ===================================================================

class TestEvalRunner:
    def test_runner_with_mock(self, mem_with_mock, tmp_path: Path) -> None:
        """Smoke test: runner should complete without errors on a mock system."""
        # Store a few memories
        mem_with_mock.remember("Python is a programming language", user_id="test")
        mem_with_mock.remember("TypeScript is used for frontend", user_id="test")

        # Build a tiny eval dataset from stored memory IDs
        all_mems = mem_with_mock.storage.list_memories(user_id="test")
        if not all_mems:
            pytest.skip("No memories stored — extractor may not have fired")

        mem_id = all_mems[0].id
        ds = EvalDataset(
            queries=["programming language"],
            ground_truth=[[mem_id]],
            metadata={"test": True},
        )

        runner = EvalRunner(mem_with_mock)
        report = runner.run(ds, k_values=[1, 3, 5])

        assert report.num_queries == 1
        assert 1 in report.recall_at_k
        assert 1 in report.precision_at_k
        assert report.p95_latency_ms >= 0.0
        assert report.config_snapshot  # should have captured config

        # Round-trip through save/load
        report_path = tmp_path / "report.json"
        save_report(report, report_path)
        loaded = load_report(report_path)
        assert loaded.num_queries == 1
        assert loaded.recall_at_k == report.recall_at_k
