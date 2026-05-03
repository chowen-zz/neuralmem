"""Tests for BatchProcessor."""
from __future__ import annotations

from unittest.mock import MagicMock

from neuralmem.perf.batch import (
    BatchProcessor,
    BatchResult,
    ItemResult,
    _percentile,
)


class TestPercentileHelper:
    def test_p50(self):
        # int(5*50/100)-1 = 1, so index 1 -> 2.0
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 2.0

    def test_p95(self):
        data = list(range(1, 101))
        result = _percentile(data, 95)
        assert result == 95

    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 50) == 42.0


class TestBatchResult:
    def test_success_and_error_counts(self):
        items = [
            ItemResult(index=0, success=True, elapsed=0.1),
            ItemResult(index=1, success=False, error="err", elapsed=0.05),
            ItemResult(index=2, success=True, elapsed=0.2),
        ]
        br = BatchResult(items=items, total_time=0.35)
        assert br.success_count == 2
        assert br.error_count == 1

    def test_avg_time(self):
        items = [
            ItemResult(index=0, success=True, elapsed=0.1),
            ItemResult(index=1, success=True, elapsed=0.3),
        ]
        br = BatchResult(items=items, total_time=0.4)
        assert abs(br.avg_time - 0.2) < 1e-9

    def test_percentile_properties(self):
        items = [
            ItemResult(index=i, success=True, elapsed=float(i) / 10)
            for i in range(10)
        ]
        br = BatchResult(items=items, total_time=4.5)
        assert br.p50 >= 0
        assert br.p95 >= 0
        assert br.p99 >= 0

    def test_empty_batch_result(self):
        br = BatchResult()
        assert br.success_count == 0
        assert br.avg_time == 0.0
        assert br.p50 == 0.0


class TestBatchProcessor:
    def _mock_mem(self, remember_side_effect=None, recall_side_effect=None):
        mem = MagicMock()
        if remember_side_effect:
            mem.remember.side_effect = remember_side_effect
        else:
            mem.remember.return_value = ["mem1"]
        if recall_side_effect:
            mem.recall.side_effect = recall_side_effect
        else:
            mem.recall.return_value = ["r1"]
        return mem

    def test_batch_add_success(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_add(["a", "b", "c"], user_id="u1")
        assert result.success_count == 3
        assert result.error_count == 0
        assert mem.remember.call_count == 3

    def test_batch_add_partial_failure(self):
        def side_effect(content, **kw):
            if content == "bad":
                raise ValueError("fail")
            return ["ok"]

        mem = self._mock_mem(remember_side_effect=side_effect)
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_add(["good", "bad", "ok"], user_id="u1")
        assert result.success_count == 2
        assert result.error_count == 1
        errors = [r for r in result.items if not r.success]
        assert "fail" in errors[0].error

    def test_batch_add_all_fail(self):
        mem = self._mock_mem(remember_side_effect=RuntimeError("boom"))
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_add(["a", "b"], user_id="u1")
        assert result.error_count == 2

    def test_batch_search_success(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_search(["q1", "q2"], user_id="u1")
        assert result.success_count == 2
        assert mem.recall.call_count == 2

    def test_batch_search_partial_failure(self):
        def side_effect(query, **kw):
            if query == "bad":
                raise RuntimeError("oops")
            return ["ok"]

        mem = self._mock_mem(recall_side_effect=side_effect)
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_search(["good", "bad", "ok"], user_id="u1")
        assert result.success_count == 2
        assert result.error_count == 1

    def test_batch_add_preserves_order(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=4)
        result = bp.batch_add(["a", "b", "c"], user_id="u1")
        indices = [r.index for r in result.items]
        assert indices == [0, 1, 2]

    def test_batch_search_preserves_order(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=4)
        result = bp.batch_search(["q1", "q2", "q3"], user_id="u1")
        indices = [r.index for r in result.items]
        assert indices == [0, 1, 2]

    def test_batch_add_timing_recorded(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_add(["a"], user_id="u1")
        assert result.items[0].elapsed > 0
        assert result.total_time > 0

    def test_batch_search_timing_recorded(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_search(["q1"], user_id="u1")
        assert result.items[0].elapsed > 0
        assert result.total_time > 0

    def test_batch_add_empty_list(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_add([], user_id="u1")
        assert result.success_count == 0
        assert mem.remember.call_count == 0

    def test_batch_search_empty_list(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=2)
        result = bp.batch_search([], user_id="u1")
        assert result.success_count == 0
        assert mem.recall.call_count == 0

    def test_max_workers_property(self):
        mem = MagicMock()
        bp = BatchProcessor(mem, max_workers=8)
        assert bp.max_workers == 8

    def test_batch_add_forwards_kwargs(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=2)
        bp.batch_add(["a"], user_id="u1", agent_id="agent1")
        mem.remember.assert_called_once_with("a", user_id="u1", agent_id="agent1")

    def test_batch_search_forwards_kwargs(self):
        mem = self._mock_mem()
        bp = BatchProcessor(mem, max_workers=2)
        bp.batch_search(["q1"], user_id="u1", limit=5)
        mem.recall.assert_called_once_with("q1", user_id="u1", limit=5)

    def test_batch_result_elapsed_times_only_successes(self):
        items = [
            ItemResult(index=0, success=True, elapsed=0.1),
            ItemResult(index=1, success=False, error="x", elapsed=0.0),
            ItemResult(index=2, success=True, elapsed=0.3),
        ]
        br = BatchResult(items=items, total_time=0.4)
        assert br.elapsed_times == [0.1, 0.3]
