"""Tests for NeuralMem V1.3 plugin system — PluginRegistry + built-in plugins.

All tests use mock / monkeypatch; no external API dependencies.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.types import Memory, SearchQuery, SearchResult
from neuralmem.plugins.base import Plugin, PluginContext
from neuralmem.plugins.builtin import LoggingPlugin, MetricsPlugin, ValidationPlugin
from neuralmem.plugins.manager import PluginManager
from neuralmem.plugins.registry import PluginRegistry


# --------------------------------------------------------------------------- #
# 最小插件实现 (用于 registry 测试)
# --------------------------------------------------------------------------- #

class MockPlugin(Plugin):
    """用于测试的最小插件实现."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    @property
    def name(self) -> str:
        return "mock"

    def on_remember(self, memory: Memory) -> Memory:
        return memory.model_copy(update={"content": f"mock:{memory.content}"})


class AnotherMockPlugin(Plugin):
    """另一个用于测试的插件实现."""

    @property
    def name(self) -> str:
        return "another_mock"

    def on_recall(self, results: list[SearchResult]) -> list[SearchResult]:
        return results


# --------------------------------------------------------------------------- #
# setup / teardown — 备份并恢复内置注册
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _preserve_registry():
    """每个测试前备份注册表, 测试后恢复."""
    original = dict(PluginRegistry._registry)
    # 移除内置注册, 让测试在干净状态运行
    for name in list(PluginRegistry._registry):
        if name in ("logging", "metrics", "validation"):
            del PluginRegistry._registry[name]
    yield
    PluginRegistry._registry.clear()
    PluginRegistry._registry.update(original)


# --------------------------------------------------------------------------- #
# PluginRegistry — register
# --------------------------------------------------------------------------- #

class TestPluginRegistryRegister:
    def test_register_plugin(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        assert "mock" in PluginRegistry.list_plugins()

    def test_register_duplicate_overwrites(self):
        """重复注册应覆盖(不报错)."""
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        PluginRegistry.register("mock", "tests.unit.test_plugins", "AnotherMockPlugin")
        assert PluginRegistry._registry["mock"][1] == "AnotherMockPlugin"


# --------------------------------------------------------------------------- #
# PluginRegistry — unregister
# --------------------------------------------------------------------------- #

class TestPluginRegistryUnregister:
    def test_unregister_removes_entry(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        entry = PluginRegistry.unregister("mock")
        assert entry == ("tests.unit.test_plugins", "MockPlugin")
        assert "mock" not in PluginRegistry.list_plugins()

    def test_unregister_missing_raises(self):
        with pytest.raises(KeyError):
            PluginRegistry.unregister("nonexistent")


# --------------------------------------------------------------------------- #
# PluginRegistry — list / introspection
# --------------------------------------------------------------------------- #

class TestPluginRegistryIntrospection:
    def test_list_plugins_empty(self):
        assert PluginRegistry.list_plugins() == []

    def test_list_plugins_multiple(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        PluginRegistry.register("another", "tests.unit.test_plugins", "AnotherMockPlugin")
        names = PluginRegistry.list_plugins()
        assert sorted(names) == ["another", "mock"]

    def test_is_registered(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        assert PluginRegistry.is_registered("mock") is True
        assert PluginRegistry.is_registered("nope") is False

    def test_get_info(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        info = PluginRegistry.get_info("mock")
        assert info["module"] == "tests.unit.test_plugins"
        assert info["class"] == "MockPlugin"

    def test_registry_info(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        info = PluginRegistry.registry_info()
        assert "mock" in info
        assert info["mock"]["class"] == "MockPlugin"


# --------------------------------------------------------------------------- #
# PluginRegistry — load
# --------------------------------------------------------------------------- #

class TestPluginRegistryLoad:
    def test_load_plugin(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        instance = PluginRegistry.load("mock")
        assert isinstance(instance, MockPlugin)

    def test_load_plugin_with_kwargs(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        instance = PluginRegistry.load("mock", value="hello")
        assert instance.value == "hello"

    def test_load_missing_raises(self):
        with pytest.raises(ValueError, match="Unknown plugin"):
            PluginRegistry.load("missing")

    def test_load_all(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        PluginRegistry.register("another", "tests.unit.test_plugins", "AnotherMockPlugin")
        loaded = PluginRegistry.load_all()
        assert "mock" in loaded
        assert "another" in loaded
        assert isinstance(loaded["mock"], MockPlugin)
        assert isinstance(loaded["another"], AnotherMockPlugin)

    def test_load_all_skips_broken(self):
        """load_all 应跳过加载失败的插件并继续."""
        PluginRegistry.register("bad", "nonexistent.module.path", "BadPlugin")
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        loaded = PluginRegistry.load_all()
        assert "mock" in loaded
        assert "bad" not in loaded


# --------------------------------------------------------------------------- #
# PluginRegistry — discover
# --------------------------------------------------------------------------- #

class TestPluginRegistryDiscover:
    def test_discover_finds_plugins(self):
        discovered = PluginRegistry.discover("tests.unit.test_plugins")
        assert "mock_plugin" in discovered
        assert "another_mock_plugin" in discovered

    def test_discover_missing_module_returns_empty(self):
        discovered = PluginRegistry.discover("nonexistent.module.path")
        assert discovered == []

    def test_discover_ignores_abstract_base(self):
        discovered = PluginRegistry.discover("neuralmem.plugins.base")
        # Plugin (abstract) should NOT be registered
        assert "plugin" not in discovered


# --------------------------------------------------------------------------- #
# Built-in plugins — LoggingPlugin
# --------------------------------------------------------------------------- #

class TestLoggingPlugin:
    def test_name_and_priority(self):
        plugin = LoggingPlugin()
        assert plugin.name == "logging"
        assert plugin.priority == 5

    def test_on_remember_logs(self):
        mock_logger = MagicMock()
        plugin = LoggingPlugin(logger=mock_logger)
        mem = Memory(content="hello")
        result = plugin.on_remember(mem)
        assert result.content == "hello"
        mock_logger.info.assert_called_once()
        assert "remember" in mock_logger.info.call_args[0][0]

    def test_on_recall_logs(self):
        mock_logger = MagicMock()
        plugin = LoggingPlugin(logger=mock_logger)
        results: list[SearchResult] = []
        plugin.on_recall(results)
        mock_logger.info.assert_called_once()
        assert "recall" in mock_logger.info.call_args[0][0]

    def test_on_reflect_logs(self):
        mock_logger = MagicMock()
        plugin = LoggingPlugin(logger=mock_logger)
        mem = Memory(content="hello")
        plugin.on_reflect(mem)
        mock_logger.info.assert_called_once()
        assert "reflect" in mock_logger.info.call_args[0][0]


# --------------------------------------------------------------------------- #
# Built-in plugins — MetricsPlugin
# --------------------------------------------------------------------------- #

class TestMetricsPlugin:
    def test_name_and_priority(self):
        plugin = MetricsPlugin()
        assert plugin.name == "metrics"
        assert plugin.priority == 10

    def test_on_remember_increments_count(self):
        plugin = MetricsPlugin()
        mem = Memory(content="hello")
        plugin.on_remember(mem)
        assert plugin._metrics["remember_count"] == 1

    def test_on_recall_increments_count(self):
        plugin = MetricsPlugin()
        plugin.on_recall([])
        assert plugin._metrics["recall_count"] == 1

    def test_on_reflect_increments_count(self):
        plugin = MetricsPlugin()
        mem = Memory(content="hello")
        plugin.on_reflect(mem)
        assert plugin._metrics["reflect_count"] == 1

    def test_flush_returns_snapshot(self):
        plugin = MetricsPlugin()
        plugin.on_remember(Memory(content="a"))
        plugin.on_recall([])
        snapshot = plugin.flush()
        assert snapshot["remember_count"] == 1
        assert snapshot["recall_count"] == 1
        # Latencies should be reset
        assert plugin._metrics["remember_latency_ms"] == []
        assert plugin._metrics["recall_latency_ms"] == []
        # Counts remain monotonic
        assert plugin._metrics["remember_count"] == 1


# --------------------------------------------------------------------------- #
# Built-in plugins — ValidationPlugin
# --------------------------------------------------------------------------- #

class TestValidationPlugin:
    def test_name_and_priority(self):
        plugin = ValidationPlugin()
        assert plugin.name == "validation"
        assert plugin.priority == 1

    def test_on_remember_empty_content_raises(self):
        plugin = ValidationPlugin()
        mem = Memory(content="")
        with pytest.raises(ValueError, match="empty"):
            plugin.on_remember(mem)

    def test_on_remember_whitespace_content_raises(self):
        plugin = ValidationPlugin()
        mem = Memory(content="   ")
        with pytest.raises(ValueError, match="empty"):
            plugin.on_remember(mem)

    def test_on_remember_too_long_raises(self):
        plugin = ValidationPlugin(max_content_length=10)
        mem = Memory(content="a" * 11)
        with pytest.raises(ValueError, match="exceeds"):
            plugin.on_remember(mem)

    def test_on_remember_valid_passes_through(self):
        plugin = ValidationPlugin()
        mem = Memory(content="valid content")
        result = plugin.on_remember(mem)
        assert result.content == "valid content"

    def test_on_recall_filters_invalid(self):
        plugin = ValidationPlugin(max_content_length=10)
        good = SearchResult(memory=Memory(content="ok"), score=0.5, retrieval_method="test")
        bad = SearchResult(memory=Memory(content="a" * 11), score=0.5, retrieval_method="test")
        results = plugin.on_recall([good, bad])
        assert len(results) == 1
        assert results[0].memory.content == "ok"

    def test_on_reflect_uses_same_rules(self):
        plugin = ValidationPlugin()
        mem = Memory(content="")
        with pytest.raises(ValueError, match="empty"):
            plugin.on_reflect(mem)


# --------------------------------------------------------------------------- #
# V1.3 hooks on Plugin base class
# --------------------------------------------------------------------------- #

class TestPluginV13Hooks:
    def test_on_remember_default_passes_through(self):
        class MinimalPlugin(Plugin):
            @property
            def name(self) -> str:
                return "minimal"

        plugin = MinimalPlugin()
        mem = Memory(content="hello")
        result = plugin.on_remember(mem)
        assert result.content == "hello"

    def test_on_recall_default_passes_through(self):
        class MinimalPlugin(Plugin):
            @property
            def name(self) -> str:
                return "minimal"

        plugin = MinimalPlugin()
        results: list[SearchResult] = []
        assert plugin.on_recall(results) == results

    def test_on_reflect_default_passes_through(self):
        class MinimalPlugin(Plugin):
            @property
            def name(self) -> str:
                return "minimal"

        plugin = MinimalPlugin()
        mem = Memory(content="hello")
        result = plugin.on_reflect(mem)
        assert result.content == "hello"


# --------------------------------------------------------------------------- #
# Integration: PluginRegistry + PluginManager
# --------------------------------------------------------------------------- #

class TestRegistryManagerIntegration:
    def test_load_and_register_with_manager(self):
        PluginRegistry.register("mock", "tests.unit.test_plugins", "MockPlugin")
        plugin = PluginRegistry.load("mock")
        mgr = PluginManager()
        mgr.register(plugin)
        assert mgr.get_plugin("mock") is plugin

    def test_builtin_plugins_loadable(self):
        """内置插件应能通过 registry 加载并在 manager 中运行."""
        # Re-register builtins (fixture stripped them)
        PluginRegistry.register("logging", "neuralmem.plugins.builtin", "LoggingPlugin")
        PluginRegistry.register("metrics", "neuralmem.plugins.builtin", "MetricsPlugin")
        PluginRegistry.register("validation", "neuralmem.plugins.builtin", "ValidationPlugin")

        mgr = PluginManager()
        for name in PluginRegistry.list_plugins():
            plugin = PluginRegistry.load(name)
            mgr.register(plugin)

        mem = Memory(content="integration test")
        mgr.run_hook("on_remember", mem)
        # All three should have executed without error
        assert len(mgr.list_plugins()) == 3
