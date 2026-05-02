"""Tests for AsyncNeuralMem async wrapper."""
from __future__ import annotations

import asyncio
import json

from neuralmem.core.async_memory import AsyncNeuralMem
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.memory import NeuralMem
from neuralmem.core.types import Memory, SearchResult

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_neural_mem(tmp_path, mock_embedder):
    """Create a NeuralMem with mock embedder for testing."""
    db_path = str(tmp_path / "async_test.db")
    cfg = NeuralMemConfig(
        db_path=db_path,
        embedding_dim=4,
        conflict_threshold_high=0.95,
        conflict_threshold_low=0.75,
        enable_importance_reinforcement=True,
        reinforcement_boost=0.05,
    )
    return NeuralMem(config=cfg, embedder=mock_embedder)


# ── Tests ────────────────────────────────────────────────────────────────


class TestAsyncNeuralMemConstruction:
    """Test AsyncNeuralMem creation and context manager protocol."""

    def test_init(self, tmp_path, mock_embedder):
        """AsyncNeuralMem wraps a NeuralMem instance."""
        mem = _make_neural_mem(tmp_path, mock_embedder)
        amem = AsyncNeuralMem(mem)
        assert amem._mem is mem
        assert amem._executor is not None

    def test_init_custom_workers(self, tmp_path, mock_embedder):
        """max_workers controls thread pool size."""
        mem = _make_neural_mem(tmp_path, mock_embedder)
        amem = AsyncNeuralMem(mem, max_workers=8)
        assert amem._executor._max_workers == 8

    def test_context_manager(self, tmp_path, mock_embedder):
        """Async context manager yields self and cleans up."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                assert isinstance(amem, AsyncNeuralMem)
            # After exit, executor should be shut down
            assert amem._executor._shutdown

        asyncio.run(_run())


class TestAsyncRemember:
    """Test async remember method."""

    def test_remember_returns_list(self, tmp_path, mock_embedder):
        """remember() returns a list of Memory objects."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                results = await amem.remember("User likes Python")
                assert isinstance(results, list)
                for r in results:
                    assert isinstance(r, Memory)

        asyncio.run(_run())

    def test_remember_with_params(self, tmp_path, mock_embedder):
        """remember() passes all keyword arguments through."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                results = await amem.remember(
                    "User works at Acme Corp",
                    user_id="u1",
                    agent_id="a1",
                    importance=0.9,
                    tags=["work", "identity"],
                )
                # At least one memory stored
                assert len(results) >= 1
                m = results[0]
                assert m.user_id == "u1"
                assert m.agent_id == "a1"
                assert m.importance == 0.9
                assert "work" in m.tags

        asyncio.run(_run())


class TestAsyncRecall:
    """Test async recall method."""

    def test_recall_returns_search_results(self, tmp_path, mock_embedder):
        """recall() returns list of SearchResult."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                await amem.remember("User likes dark mode")
                results = await amem.recall("What mode does user prefer?")
                assert isinstance(results, list)
                for r in results:
                    assert isinstance(r, SearchResult)
                    assert isinstance(r.memory, Memory)
                    assert 0.0 <= r.score <= 1.0

        asyncio.run(_run())

    def test_recall_with_limit(self, tmp_path, mock_embedder):
        """recall() respects limit parameter."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                for i in range(5):
                    await amem.remember(f"Fact number {i} about the user")
                results = await amem.recall("fact", limit=3)
                assert len(results) <= 3

        asyncio.run(_run())


class TestAsyncReflect:
    """Test async reflect method."""

    def test_reflect_returns_string(self, tmp_path, mock_embedder):
        """reflect() returns a string report."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                await amem.remember("User prefers TypeScript for frontend")
                report = await amem.reflect("frontend preferences")
                assert isinstance(report, str)
                assert len(report) > 0

        asyncio.run(_run())


class TestAsyncForget:
    """Test async forget method."""

    def test_forget_returns_count(self, tmp_path, mock_embedder):
        """forget() returns the number of deleted memories."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                memories = await amem.remember("User is allergic to peanuts")
                assert len(memories) >= 1
                mid = memories[0].id
                count = await amem.forget(memory_id=mid)
                assert isinstance(count, int)
                assert count >= 1

        asyncio.run(_run())

    def test_forget_by_user(self, tmp_path, mock_embedder):
        """forget() with user_id deletes all memories for that user."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                await amem.remember("Fact A", user_id="u1")
                await amem.remember("Fact B", user_id="u1")
                await amem.remember("Fact C", user_id="u2")
                count = await amem.forget(user_id="u1")
                assert count >= 2

        asyncio.run(_run())


class TestAsyncRememberBatch:
    """Test async remember_batch method."""

    def test_batch_returns_all(self, tmp_path, mock_embedder):
        """remember_batch() processes multiple items."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                contents = ["Fact one", "Fact two", "Fact three"]
                results = await amem.remember_batch(contents, user_id="u1")
                assert isinstance(results, list)
                assert len(results) >= 1  # at least some stored

        asyncio.run(_run())


class TestAsyncExportImport:
    """Test async export and import methods."""

    def test_export_json(self, tmp_path, mock_embedder):
        """export_memories() returns valid JSON by default."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                await amem.remember("Exportable fact", user_id="u1")
                data = await amem.export_memories(user_id="u1")
                parsed = json.loads(data)
                assert isinstance(parsed, list)
                assert len(parsed) >= 1
                assert "content" in parsed[0]

        asyncio.run(_run())

    def test_export_csv(self, tmp_path, mock_embedder):
        """export_memories() supports csv format."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                await amem.remember("CSV fact", user_id="u1")
                data = await amem.export_memories(user_id="u1", format="csv")
                assert "content" in data
                assert "CSV fact" in data

        asyncio.run(_run())

    def test_import_json(self, tmp_path, mock_embedder):
        """import_memories() imports from JSON data."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                export_data = json.dumps([
                    {
                        "content": "Imported memory",
                        "memory_type": "semantic",
                        "scope": "user",
                        "user_id": "u1",
                        "importance": 0.7,
                        "tags": ["imported"],
                    }
                ])
                count = await amem.import_memories(export_data, format="json")
                assert count >= 1

        asyncio.run(_run())


class TestAsyncConsolidateAndCleanup:
    """Test async consolidate, cleanup_expired, and get_stats."""

    def test_consolidate(self, tmp_path, mock_embedder):
        """consolidate() returns dict with expected keys."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                result = await amem.consolidate()
                assert isinstance(result, dict)
                assert "decayed" in result
                assert "forgotten" in result
                assert "merged" in result

        asyncio.run(_run())

    def test_cleanup_expired(self, tmp_path, mock_embedder):
        """cleanup_expired() returns an integer count."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                count = await amem.cleanup_expired()
                assert isinstance(count, int)

        asyncio.run(_run())

    def test_get_stats(self, tmp_path, mock_embedder):
        """get_stats() returns a dict."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                stats = await amem.get_stats()
                assert isinstance(stats, dict)
                assert len(stats) > 0

        asyncio.run(_run())


class TestAsyncResolveConflict:
    """Test async resolve_conflict method."""

    def test_resolve_nonexistent(self, tmp_path, mock_embedder):
        """resolve_conflict() returns False for nonexistent memory."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                result = await amem.resolve_conflict("nonexistent-id")
                assert result is False

        asyncio.run(_run())


class TestAsyncForgetBatch:
    """Test async forget_batch method."""

    def test_forget_batch_dry_run(self, tmp_path, mock_embedder):
        """forget_batch() with dry_run returns preview without deleting."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                await amem.remember("Fact A", user_id="u1")
                await amem.remember("Fact B", user_id="u1")
                result = await amem.forget_batch(user_id="u1", dry_run=True)
                assert isinstance(result, dict)
                assert result["dry_run"] is True
                assert result["count"] >= 2

        asyncio.run(_run())

    def test_forget_batch_real(self, tmp_path, mock_embedder):
        """forget_batch() actually deletes when dry_run=False."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem) as amem:
                await amem.remember("Deletable fact", user_id="u1")
                result = await amem.forget_batch(user_id="u1", dry_run=False)
                assert result["dry_run"] is False
                assert result["count"] >= 1

        asyncio.run(_run())


class TestAsyncConcurrency:
    """Test that multiple async operations can run concurrently."""

    def test_concurrent_remembers(self, tmp_path, mock_embedder):
        """Multiple remember() calls can be awaited concurrently."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem, max_workers=4) as amem:
                tasks = [
                    amem.remember(f"Concurrent fact {i}", user_id="u1")
                    for i in range(5)
                ]
                results = await asyncio.gather(*tasks)
                assert len(results) == 5
                for r in results:
                    assert isinstance(r, list)

        asyncio.run(_run())

    def test_concurrent_mixed_operations(self, tmp_path, mock_embedder):
        """Multiple async operations can be awaited in sequence."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            async with AsyncNeuralMem(mem, max_workers=4) as amem:
                # Concurrent remembers work fine (each writes independently)
                tasks = [
                    amem.remember("User likes cats", user_id="u1"),
                    amem.remember("User likes dogs", user_id="u1"),
                ]
                await asyncio.gather(*tasks)

                # Sequential recall (SQLite connections are not thread-safe
                # for concurrent cross-thread reads from the same connection)
                r1 = await amem.recall("cats", user_id="u1")
                r2 = await amem.recall("dogs", user_id="u1")
                assert isinstance(r1, list)
                assert isinstance(r2, list)

        asyncio.run(_run())


class TestAsyncCloseIdempotent:
    """Test that close() is safe to call multiple times."""

    def test_double_close(self, tmp_path, mock_embedder):
        """Calling close() twice does not raise."""
        mem = _make_neural_mem(tmp_path, mock_embedder)

        async def _run():
            amem = AsyncNeuralMem(mem)
            await amem.close()
            await amem.close()  # Should not raise

        asyncio.run(_run())
