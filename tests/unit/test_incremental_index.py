"""Tests for IncrementalIndex."""
from __future__ import annotations

from unittest.mock import MagicMock

from neuralmem.perf.incremental_index import IncrementalIndex, IndexStats


class TestIncrementalIndex:
    def test_mark_dirty(self):
        idx = IncrementalIndex()
        idx.mark_dirty("mem1")
        assert "mem1" in idx.get_dirty()

    def test_mark_clean(self):
        idx = IncrementalIndex()
        idx.mark_dirty("mem1")
        idx.mark_clean("mem1")
        assert "mem1" not in idx.get_dirty()

    def test_mark_dirty_removes_from_clean(self):
        idx = IncrementalIndex()
        idx.mark_clean("mem1")
        idx.mark_dirty("mem1")
        assert "mem1" in idx.get_dirty()

    def test_get_dirty_returns_copy(self):
        idx = IncrementalIndex()
        idx.mark_dirty("mem1")
        dirty = idx.get_dirty()
        dirty.add("mem2")
        assert "mem2" not in idx.get_dirty()

    def test_get_dirty_empty(self):
        idx = IncrementalIndex()
        assert idx.get_dirty() == set()

    def test_stats_initial(self):
        idx = IncrementalIndex()
        s = idx.stats()
        assert s.dirty_count == 0
        assert s.total_indexed == 0
        assert s.last_reindex_time == 0.0
        assert s.total_reindexes == 0

    def test_stats_after_operations(self):
        idx = IncrementalIndex()
        idx.mark_dirty("m1")
        idx.mark_dirty("m2")
        idx.mark_clean("m3")
        s = idx.stats()
        assert s.dirty_count == 2
        assert s.total_indexed == 1

    def test_update_content_hash_changed(self):
        idx = IncrementalIndex()
        changed = idx.update_content_hash("m1", "hello")
        assert changed is True

    def test_update_content_hash_unchanged(self):
        idx = IncrementalIndex()
        idx.update_content_hash("m1", "hello")
        changed = idx.update_content_hash("m1", "hello")
        assert changed is False

    def test_update_content_hash_different(self):
        idx = IncrementalIndex()
        idx.update_content_hash("m1", "hello")
        changed = idx.update_content_hash("m1", "world")
        assert changed is True

    def test_reindex_dirty_empty(self):
        idx = IncrementalIndex()
        embedder = MagicMock()
        storage = MagicMock()
        count = idx.reindex_dirty(embedder, storage)
        assert count == 0

    def test_reindex_dirty_success(self):
        idx = IncrementalIndex()
        embedder = MagicMock()
        embedder.encode_one.return_value = [0.1] * 8
        storage = MagicMock()
        mem = MagicMock()
        mem.content = "some content"
        storage.get_memory.return_value = mem

        idx.mark_dirty("m1")
        idx.mark_dirty("m2")
        count = idx.reindex_dirty(embedder, storage)
        assert count == 2
        assert storage.update_memory.call_count == 2
        s = idx.stats()
        assert s.dirty_count == 0
        assert s.total_indexed == 2

    def test_reindex_dirty_with_missing_memory(self):
        idx = IncrementalIndex()
        embedder = MagicMock()
        storage = MagicMock()
        storage.get_memory.return_value = None

        idx.mark_dirty("m1")
        count = idx.reindex_dirty(embedder, storage)
        assert count == 0
        assert "m1" not in idx.get_dirty()

    def test_reindex_dirty_with_exception(self):
        idx = IncrementalIndex()
        embedder = MagicMock()
        embedder.encode_one.side_effect = RuntimeError("fail")
        storage = MagicMock()
        mem = MagicMock()
        mem.content = "content"
        storage.get_memory.return_value = mem

        idx.mark_dirty("m1")
        count = idx.reindex_dirty(embedder, storage)
        assert count == 0
        # m1 should still be dirty after failure
        assert "m1" in idx.get_dirty()

    def test_reindex_with_content_getter(self):
        idx = IncrementalIndex()
        embedder = MagicMock()
        embedder.encode_one.return_value = [0.1] * 8
        storage = MagicMock()

        def getter(mid, stor):
            return f"content for {mid}"

        idx.mark_dirty("m1")
        count = idx.reindex_dirty(
            embedder, storage, content_getter=getter
        )
        assert count == 1
        embedder.encode_one.assert_called_once_with("content for m1")

    def test_stats_dataclass(self):
        s = IndexStats(
            dirty_count=5,
            total_indexed=10,
            last_reindex_time=1.5,
            total_reindexes=3,
        )
        assert s.dirty_count == 5
        assert s.total_indexed == 10
        assert s.last_reindex_time == 1.5
        assert s.total_reindexes == 3
