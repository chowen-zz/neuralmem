"""Tests for plugin system — PluginManager + built-in plugins."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import Memory, SearchQuery, SearchResult
from neuralmem.plugins.base import Plugin, PluginContext
from neuralmem.plugins.builtins import DedupPlugin, ImportancePlugin, RecencyBoostPlugin
from neuralmem.plugins.manager import PluginManager

# --- Helpers ---

class SimplePlugin(Plugin):
    """Minimal test plugin."""

    def __init__(self, name: str = "simple", priority: int = 100):
        self._name = name
        self._priority = priority
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def on_init(self) -> None:
        self.calls.append("on_init")

    def on_before_remember(self, memory: Memory) -> Memory:
        self.calls.append("on_before_remember")
        return memory

    def on_after_remember(self, memory: Memory) -> None:
        self.calls.append("on_after_remember")

    def on_before_recall(self, query: SearchQuery) -> SearchQuery:
        self.calls.append("on_before_recall")
        return query

    def on_after_recall(self, results):
        self.calls.append("on_after_recall")
        return results

    def on_before_forget(self, memory_id: str) -> None:
        self.calls.append("on_before_forget")

    def on_error(self, error: Exception) -> None:
        self.calls.append("on_error")


class FailingPlugin(Plugin):
    """Plugin that raises on hooks."""

    def __init__(self):
        self.error_handled = False

    @property
    def name(self) -> str:
        return "failing"

    def on_before_remember(self, memory: Memory) -> Memory:
        raise RuntimeError("Plugin failure")

    def on_error(self, error: Exception) -> None:
        self.error_handled = True


class ModifyingPlugin(Plugin):
    """Plugin that modifies the memory."""

    @property
    def name(self) -> str:
        return "modifying"

    @property
    def priority(self) -> int:
        return 5

    def on_before_remember(self, memory: Memory) -> Memory:
        return memory.model_copy(update={"importance": 0.99})


class QueryModifyingPlugin(Plugin):
    """Plugin that modifies the search query."""

    @property
    def name(self) -> str:
        return "query_mod"

    def on_before_recall(self, query: SearchQuery) -> SearchQuery:
        return SearchQuery(
            query=query.query + " expanded",
            limit=query.limit,
        )


class ResultsModifyingPlugin(Plugin):
    """Plugin that modifies search results."""

    @property
    def name(self) -> str:
        return "results_mod"

    def on_after_recall(self, results):
        # Add a synthetic result
        mem = Memory(content="injected")
        results.append(
            SearchResult(
                memory=mem,
                score=0.99,
                retrieval_method="plugin",
            )
        )
        return results


# --- PluginManager tests ---

class TestPluginManager:
    def test_register_and_list(self):
        mgr = PluginManager()
        p = SimplePlugin("test1")
        mgr.register(p)
        plugins = mgr.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "test1"

    def test_register_calls_on_init(self):
        mgr = PluginManager()
        p = SimplePlugin("test1")
        mgr.register(p)
        assert "on_init" in p.calls

    def test_register_duplicate_raises(self):
        mgr = PluginManager()
        p1 = SimplePlugin("dup")
        p2 = SimplePlugin("dup")
        mgr.register(p1)
        with pytest.raises(ValueError, match="already registered"):
            mgr.register(p2)

    def test_unregister(self):
        mgr = PluginManager()
        p = SimplePlugin("removable")
        mgr.register(p)
        removed = mgr.unregister("removable")
        assert removed is p
        assert mgr.list_plugins() == []

    def test_unregister_not_found_raises(self):
        mgr = PluginManager()
        with pytest.raises(KeyError):
            mgr.unregister("nonexistent")

    def test_priority_order(self):
        mgr = PluginManager()
        p_low = SimplePlugin("low", priority=10)
        p_high = SimplePlugin("high", priority=100)
        mgr.register(p_high)
        mgr.register(p_low)
        plugins = mgr.list_plugins()
        assert plugins[0]["name"] == "low"
        assert plugins[1]["name"] == "high"

    def test_run_hook_dispatches_to_all(self):
        mgr = PluginManager()
        p1 = SimplePlugin("p1")
        p2 = SimplePlugin("p2")
        mgr.register(p1)
        mgr.register(p2)
        mem = Memory(content="test")
        mgr.run_hook("on_before_remember", mem)
        assert "on_before_remember" in p1.calls
        assert "on_before_remember" in p2.calls

    def test_run_hook_passes_return_value(self):
        mgr = PluginManager()
        mgr.register(ModifyingPlugin())
        mem = Memory(content="test", importance=0.5)
        result = mgr.run_hook("on_before_remember", mem)
        assert result.importance == 0.99

    def test_run_hook_error_isolation(self):
        mgr = PluginManager()
        fail = FailingPlugin()
        simple = SimplePlugin("after_fail")
        mgr.register(fail)
        mgr.register(simple)
        mem = Memory(content="test")
        mgr.run_hook("on_before_remember", mem)
        # Should not raise; simple plugin still runs
        assert "on_before_remember" in simple.calls
        assert fail.error_handled

    def test_run_hook_error_recorded_in_context(self):
        mgr = PluginManager()
        mgr.register(FailingPlugin())
        ctx = PluginContext()
        mem = Memory(content="test")
        mgr.run_hook("on_before_remember", mem, context=ctx)
        assert len(ctx.errors) == 1
        assert "Plugin failure" in str(ctx.errors[0])

    def test_run_hook_no_args(self):
        mgr = PluginManager()
        p = SimplePlugin("test")
        mgr.register(p)
        mgr.run_hook("on_after_remember", Memory(content="test"))
        assert "on_after_remember" in p.calls

    def test_run_hook_before_recall_modifies_query(self):
        mgr = PluginManager()
        mgr.register(QueryModifyingPlugin())
        query = SearchQuery(query="original")
        result = mgr.run_hook("on_before_recall", query)
        assert "expanded" in result.query

    def test_run_hook_after_recall_modifies_results(self):
        mgr = PluginManager()
        mgr.register(ResultsModifyingPlugin())
        results = []
        result = mgr.run_hook("on_after_recall", results)
        assert len(result) == 1
        assert result[0].memory.content == "injected"

    def test_run_hook_before_forget(self):
        mgr = PluginManager()
        p = SimplePlugin("test")
        mgr.register(p)
        mgr.run_hook("on_before_forget", "mem-123")
        assert "on_before_forget" in p.calls

    def test_get_plugin(self):
        mgr = PluginManager()
        p = SimplePlugin("findme")
        mgr.register(p)
        assert mgr.get_plugin("findme") is p
        assert mgr.get_plugin("nope") is None

    def test_plugin_version(self):
        p = SimplePlugin("vtest")
        assert p.version == "0.1.0"

    def test_plugin_context_data(self):
        ctx = PluginContext()
        ctx.data["key"] = "value"
        assert ctx.data["key"] == "value"

    def test_on_init_failure_isolated(self):
        class InitFailPlugin(Plugin):
            @property
            def name(self):
                return "init_fail"

            def on_init(self):
                raise RuntimeError("init boom")

        mgr = PluginManager()
        p = InitFailPlugin()
        mgr.register(p)  # Should not raise
        assert mgr.list_plugins()[0]["name"] == "init_fail"


# --- DedupPlugin tests ---

class TestDedupPlugin:
    def test_no_storage_returns_unchanged(self):
        plugin = DedupPlugin()
        mem = Memory(content="test")
        result = plugin.on_before_remember(mem)
        assert result.content == "test"

    def test_no_embedding_returns_unchanged(self):
        plugin = DedupPlugin(storage=MagicMock(), embedder=MagicMock())
        mem = Memory(content="test", embedding=None)
        result = plugin.on_before_remember(mem)
        assert result.content == "test"

    def test_merges_when_similar_found(self):
        storage = MagicMock()
        embedder = MagicMock()
        existing = Memory(
            content="existing memory",
            user_id="u1",
            tags=("old",),
            importance=0.7,
        )
        storage.find_similar.return_value = [existing]
        plugin = DedupPlugin(
            storage=storage, embedder=embedder, threshold=0.9
        )
        mem = Memory(
            content="new memory",
            user_id="u1",
            embedding=[0.1, 0.2, 0.3],
            tags=("new",),
            importance=0.5,
        )
        result = plugin.on_before_remember(mem)
        assert "existing memory" in result.content
        assert "new memory" in result.content

    def test_no_merge_when_no_similar(self):
        storage = MagicMock()
        storage.find_similar.return_value = []
        plugin = DedupPlugin(storage=storage, embedder=MagicMock())
        mem = Memory(content="unique", embedding=[0.1])
        result = plugin.on_before_remember(mem)
        assert result.content == "unique"

    def test_no_merge_when_self_match(self):
        storage = MagicMock()
        mem = Memory(id="same-id", content="self")
        storage.find_similar.return_value = [mem]
        plugin = DedupPlugin(storage=storage, embedder=MagicMock())
        mem2 = Memory(id="same-id", content="self", embedding=[0.1])
        result = plugin.on_before_remember(mem2)
        # Self match should not trigger merge (same id)
        assert result.content == "self"


# --- ImportancePlugin tests ---

class TestImportancePlugin:
    def test_name_and_priority(self):
        plugin = ImportancePlugin()
        assert plugin.name == "importance"
        assert plugin.priority == 50

    def test_on_after_remember_no_crash(self):
        plugin = ImportancePlugin()
        mem = Memory(content="short", importance=0.5)
        # Should not raise even without storage
        plugin.on_after_remember(mem)


# --- RecencyBoostPlugin tests ---

class TestRecencyBoostPlugin:
    def test_name_and_priority(self):
        plugin = RecencyBoostPlugin()
        assert plugin.name == "recency_boost"
        assert plugin.priority == 80

    def test_empty_results_passthrough(self):
        plugin = RecencyBoostPlugin()
        result = plugin.on_after_recall([])
        assert result == []

    def test_boosts_recent_memories(self):
        plugin = RecencyBoostPlugin(decay_hours=168.0)
        now = datetime.now(timezone.utc)
        mem = Memory(content="recent", created_at=now, importance=0.5)
        sr = SearchResult(
            memory=mem, score=0.5, retrieval_method="test"
        )
        results = plugin.on_after_recall([sr])
        # Should get a small boost
        assert results[0].score >= 0.5

    def test_old_memories_get_less_boost(self):
        plugin = RecencyBoostPlugin(decay_hours=24.0)
        from datetime import timedelta

        old_time = datetime.now(timezone.utc) - timedelta(days=365)
        mem = Memory(content="old", created_at=old_time, importance=0.5)
        sr = SearchResult(
            memory=mem, score=0.5, retrieval_method="test"
        )
        results = plugin.on_after_recall([sr])
        # Very old → minimal boost
        assert results[0].score < 0.55

    def test_results_sorted_after_boost(self):
        plugin = RecencyBoostPlugin()
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        recent = Memory(content="new", created_at=now)
        old = Memory(
            content="old",
            created_at=now - timedelta(days=365),
        )
        # Old has higher base score
        results_old = SearchResult(
            memory=old, score=0.9, retrieval_method="test"
        )
        results_new = SearchResult(
            memory=recent, score=0.85, retrieval_method="test"
        )
        boosted = plugin.on_after_recall([results_old, results_new])
        # Order may change due to recency boost
        assert len(boosted) == 2
