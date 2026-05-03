"""Tests for neuralmem.ops.expiry — MemoryExpiry."""
from __future__ import annotations

from datetime import datetime, timezone

from neuralmem.core.types import Memory
from neuralmem.ops.expiry import ExpiryPolicy, ExpiryResult, MemoryExpiry

# ---------------------------------------------------------------------------
# Mock storage
# ---------------------------------------------------------------------------


class MockStorage:
    """In-memory storage mock for expiry tests."""

    def __init__(self):
        self._memories: dict[str, Memory] = {}
        self._deleted: list[str] = []

    def save_memory(self, memory: Memory):
        self._memories[memory.id] = memory
        return memory.id

    def list_memories(self, user_id=None, limit=10_000):
        mems = list(self._memories.values())
        if user_id is not None:
            mems = [m for m in mems if m.user_id == user_id]
        return mems[:limit]

    def delete_memories(
        self,
        memory_id=None,
        user_id=None,
        before=None,
        tags=None,
        max_importance=None,
    ):
        to_delete = []
        for mid, mem in self._memories.items():
            if memory_id is not None and mid != memory_id:
                continue
            if user_id is not None and mem.user_id != user_id:
                continue
            if before is not None and mem.created_at >= before:
                continue
            if max_importance is not None and mem.importance >= max_importance:
                continue
            to_delete.append(mid)

        for mid in to_delete:
            del self._memories[mid]
            self._deleted.append(mid)
        return len(to_delete)

    def get_memory(self, memory_id):
        return self._memories.get(memory_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    content: str = "test",
    user_id: str = "u1",
    importance: float = 0.5,
    source: str | None = None,
    created_at: datetime | None = None,
) -> Memory:
    mem = Memory(
        content=content,
        user_id=user_id,
        importance=importance,
        source=source,
    )
    # For frozen models we need model_copy to set created_at
    if created_at is not None:
        mem = mem.model_copy(update={"created_at": created_at})
    return mem


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryExpiry:
    def test_no_policies_no_deletion(self):
        storage = MockStorage()
        storage.save_memory(_make_memory("hello"))
        expiry = MemoryExpiry(storage=storage)
        result = expiry.apply_policies()
        assert result.expired_ids == []
        assert len(storage._memories) == 1

    def test_ttl_deletes_old_memories(self):
        storage = MockStorage()
        old = _make_memory("old", created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
        new = _make_memory("new")
        storage.save_memory(old)
        storage.save_memory(new)
        policy = ExpiryPolicy(ttl_days=30)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        assert old.id in result.expired_ids
        assert new.id not in result.expired_ids
        assert len(storage._memories) == 1

    def test_ttl_keeps_recent_memories(self):
        storage = MockStorage()
        recent = _make_memory("recent")
        storage.save_memory(recent)
        policy = ExpiryPolicy(ttl_days=30)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        assert result.expired_ids == []

    def test_importance_threshold_deletes_low(self):
        storage = MockStorage()
        low = _make_memory("low", importance=0.1)
        high = _make_memory("high", importance=0.9)
        storage.save_memory(low)
        storage.save_memory(high)
        policy = ExpiryPolicy(importance_threshold=0.3)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        assert low.id in result.expired_ids
        assert high.id not in result.expired_ids

    def test_importance_threshold_exact_boundary(self):
        storage = MockStorage()
        exact = _make_memory("exact", importance=0.3)
        storage.save_memory(exact)
        policy = ExpiryPolicy(importance_threshold=0.3)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        # importance 0.3 is NOT < 0.3, so should NOT be deleted
        assert exact.id not in result.expired_ids

    def test_max_count_keeps_top_n(self):
        storage = MockStorage()
        for i, imp in enumerate([0.9, 0.7, 0.5, 0.3, 0.1]):
            storage.save_memory(_make_memory(f"m{i}", importance=imp))
        policy = ExpiryPolicy(max_count=2)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        assert len(result.expired_ids) == 3
        # The top 2 should remain
        remaining = list(storage._memories.values())
        importances = sorted(
            [m.importance for m in remaining], reverse=True
        )
        assert importances == [0.9, 0.7]

    def test_max_count_no_deletion_if_under_limit(self):
        storage = MockStorage()
        storage.save_memory(_make_memory("a", importance=0.9))
        storage.save_memory(_make_memory("b", importance=0.7))
        policy = ExpiryPolicy(max_count=5)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        assert result.expired_ids == []

    def test_combined_policies(self):
        storage = MockStorage()
        old_low = _make_memory(
            "old low",
            importance=0.1,
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        new_low = _make_memory("new low", importance=0.1)
        new_high = _make_memory("new high", importance=0.9)
        storage.save_memory(old_low)
        storage.save_memory(new_low)
        storage.save_memory(new_high)
        policy = ExpiryPolicy(ttl_days=30, importance_threshold=0.3)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        assert old_low.id in result.expired_ids
        assert new_low.id in result.expired_ids
        assert new_high.id not in result.expired_ids

    def test_user_scoped(self):
        storage = MockStorage()
        m1 = _make_memory("u1 mem", user_id="u1", importance=0.1)
        m2 = _make_memory("u2 mem", user_id="u2", importance=0.1)
        storage.save_memory(m1)
        storage.save_memory(m2)
        policy = ExpiryPolicy(importance_threshold=0.3)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies(user_id="u1")
        assert m1.id in result.expired_ids
        assert m2.id not in result.expired_ids

    def test_result_has_details(self):
        storage = MockStorage()
        storage.save_memory(_make_memory("m1"))
        policy = ExpiryPolicy(ttl_days=365)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        assert "ttl" in result.details
        assert "importance_threshold" in result.details
        assert "max_count" in result.details

    def test_no_deduplication_needed(self):
        storage = MockStorage()
        m = _make_memory(
            "old",
            importance=0.1,
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        storage.save_memory(m)
        # Both TTL and importance would delete the same memory
        policy = ExpiryPolicy(ttl_days=30, importance_threshold=0.3)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        # Should appear only once (deduped)
        assert result.expired_ids.count(m.id) == 1

    def test_override_policy_per_call(self):
        storage = MockStorage()
        m = _make_memory("m1", importance=0.1)
        storage.save_memory(m)
        default_policy = ExpiryPolicy(importance_threshold=0.05)
        call_policy = ExpiryPolicy(importance_threshold=0.5)
        expiry = MemoryExpiry(storage=storage, policy=default_policy)
        result = expiry.apply_policies(policy=call_policy)
        assert m.id in result.expired_ids

    def test_empty_storage(self):
        storage = MockStorage()
        policy = ExpiryPolicy(ttl_days=30, max_count=5)
        expiry = MemoryExpiry(storage=storage, policy=policy)
        result = expiry.apply_policies()
        assert result.expired_ids == []

    def test_expiry_result_dataclass(self):
        result = ExpiryResult()
        assert result.expired_ids == []
        assert result.details == {}
