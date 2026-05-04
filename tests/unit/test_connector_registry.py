"""ConnectorRegistry 单元测试 — 全部使用 mock, 不依赖外部 API."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem
from neuralmem.connectors.registry import ConnectorRegistry


# --------------------------------------------------------------------------- #
# 测试用的最小连接器实现
# --------------------------------------------------------------------------- #

class MockConnector(ConnectorProtocol):
    """用于测试的最小连接器实现."""

    def authenticate(self) -> bool:
        self.state = ConnectorState.CONNECTED
        return True

    def sync(self, **kwargs) -> list[SyncItem]:
        return [SyncItem(id="1", content="hello", source="mock")]

    def disconnect(self) -> None:
        self.state = ConnectorState.DISCONNECTED


class AnotherMockConnector(ConnectorProtocol):
    """另一个用于测试的连接器实现."""

    def authenticate(self) -> bool:
        return True

    def sync(self, **kwargs) -> list[SyncItem]:
        return [SyncItem(id="2", content="world", source="another")]

    def disconnect(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# setup / teardown — 备份并恢复内置注册
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _preserve_registry():
    """每个测试前备份注册表, 测试后恢复."""
    original = dict(ConnectorRegistry._registry)
    # 移除内置注册, 让测试在干净状态运行
    builtin = {"notion", "slack", "github", "filesystem", "gdrive", "s3"}
    for name in list(ConnectorRegistry._registry):
        if name in builtin:
            del ConnectorRegistry._registry[name]
    yield
    ConnectorRegistry._registry.clear()
    ConnectorRegistry._registry.update(original)


# --------------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------------- #

def test_register_connector():
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "MockConnector", "mock")
    assert "mock" in ConnectorRegistry.list_connectors()


def test_register_duplicate_overwrites():
    """重复注册应覆盖(不报错)."""
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "MockConnector", "mock")
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "AnotherMockConnector", "mock")
    assert ConnectorRegistry._registry["mock"][1] == "AnotherMockConnector"


# --------------------------------------------------------------------------- #
# list_connectors
# --------------------------------------------------------------------------- #

def test_list_connectors_empty():
    assert ConnectorRegistry.list_connectors() == []


def test_list_connectors_multiple():
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "MockConnector", "mock")
    ConnectorRegistry.register("another", "tests.unit.test_connector_registry", "AnotherMockConnector", "mock")
    names = ConnectorRegistry.list_connectors()
    assert sorted(names) == ["another", "mock"]


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #

def test_create_connector():
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "MockConnector", "mock")
    instance = ConnectorRegistry.create("mock")
    assert isinstance(instance, MockConnector)
    assert instance.name == "mock"
    assert instance.config == {}


def test_create_connector_with_config():
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "MockConnector", "mock")
    cfg = {"token": "***"}
    instance = ConnectorRegistry.create("mock", cfg)
    assert instance.config == cfg


def test_create_missing_raises():
    with pytest.raises(ValueError, match="Unknown connector"):
        ConnectorRegistry.create("missing")


# --------------------------------------------------------------------------- #
# check_dependency
# --------------------------------------------------------------------------- #

def test_check_dependency_true():
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "MockConnector", "mock")
    assert ConnectorRegistry.check_dependency("mock") is True


def test_check_dependency_false():
    ConnectorRegistry.register("bad", "nonexistent.module.path", "BadConnector", "bad")
    assert ConnectorRegistry.check_dependency("bad") is False


# --------------------------------------------------------------------------- #
# registry_info
# --------------------------------------------------------------------------- #

def test_registry_info():
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "MockConnector", "mock")
    info = ConnectorRegistry.registry_info()
    assert "mock" in info
    assert info["mock"]["class"] == "MockConnector"
    assert info["mock"]["available"] == "True"


# --------------------------------------------------------------------------- #
# 完整工作流
# --------------------------------------------------------------------------- #

def test_full_workflow():
    """注册 -> 实例化 -> authenticate -> sync -> disconnect."""
    ConnectorRegistry.register("mock", "tests.unit.test_connector_registry", "MockConnector", "mock")
    conn = ConnectorRegistry.create("mock")
    assert conn.authenticate() is True
    items = conn.sync()
    assert len(items) == 1
    assert items[0].content == "hello"
    conn.disconnect()
    assert conn.state == ConnectorState.DISCONNECTED
