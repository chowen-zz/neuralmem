"""ConnectorProtocol (连接器基类) 单元测试 — 全部使用 mock, 不依赖外部 API."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem


# --------------------------------------------------------------------------- #
# 测试用的最小连接器实现
# --------------------------------------------------------------------------- #

class DummyConnector(ConnectorProtocol):
    """用于测试的最小连接器实现."""

    def authenticate(self) -> bool:
        self.state = ConnectorState.CONNECTED
        return True

    def sync(self, **kwargs) -> list[SyncItem]:
        self.state = ConnectorState.SYNCING
        return [
            SyncItem(
                id="1",
                content="hello",
                source="dummy",
                title="Test",
                author="Alice",
            )
        ]

    def disconnect(self) -> None:
        self.state = ConnectorState.DISCONNECTED


class FailingConnector(ConnectorProtocol):
    """用于测试错误处理的最小连接器实现."""

    def authenticate(self) -> bool:
        self._set_error("auth failed")
        return False

    def sync(self, **kwargs) -> list[SyncItem]:
        self._set_error("sync failed")
        return []

    def disconnect(self) -> None:
        self.state = ConnectorState.DISCONNECTED


# --------------------------------------------------------------------------- #
# 初始化测试
# --------------------------------------------------------------------------- #

def test_connector_init_defaults():
    conn = DummyConnector("test")
    assert conn.name == "test"
    assert conn.config == {}
    assert conn.state == ConnectorState.DISCONNECTED
    assert conn.is_available() is True
    assert conn.last_error() is None


def test_connector_init_with_config():
    cfg = {"token": "abc123"}
    conn = DummyConnector("test", cfg)
    assert conn.config == cfg


# --------------------------------------------------------------------------- #
# 生命周期测试
# --------------------------------------------------------------------------- #

def test_authenticate_success():
    conn = DummyConnector("test")
    assert conn.authenticate() is True
    assert conn.state == ConnectorState.CONNECTED


def test_authenticate_failure():
    conn = FailingConnector("test")
    assert conn.authenticate() is False
    assert conn.state == ConnectorState.ERROR
    assert conn.last_error() == "auth failed"


def test_sync_returns_items():
    conn = DummyConnector("test")
    conn.authenticate()
    items = conn.sync()
    assert conn.state == ConnectorState.SYNCING
    assert len(items) == 1
    assert items[0].id == "1"
    assert items[0].content == "hello"
    assert items[0].source == "dummy"
    assert items[0].title == "Test"
    assert items[0].author == "Alice"


def test_sync_failure():
    conn = FailingConnector("test")
    items = conn.sync()
    assert items == []
    assert conn.state == ConnectorState.ERROR
    assert conn.last_error() == "sync failed"


def test_disconnect():
    conn = DummyConnector("test")
    conn.authenticate()
    conn.disconnect()
    assert conn.state == ConnectorState.DISCONNECTED


# --------------------------------------------------------------------------- #
# 上下文管理器测试
# --------------------------------------------------------------------------- #

def test_context_manager():
    conn = DummyConnector("test")
    with conn as c:
        assert c.state == ConnectorState.CONNECTED
    assert conn.state == ConnectorState.DISCONNECTED


def test_context_manager_exception():
    """上下文管理器应在异常时仍调用 disconnect."""
    conn = DummyConnector("test")
    try:
        with conn:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert conn.state == ConnectorState.DISCONNECTED


# --------------------------------------------------------------------------- #
# SyncItem 数据类测试
# --------------------------------------------------------------------------- #

def test_sync_item_defaults():
    item = SyncItem(id="1", content="hello", source="dummy")
    assert item.source_url is None
    assert item.title is None
    assert item.author is None
    assert isinstance(item.created_at, datetime)
    assert isinstance(item.updated_at, datetime)
    assert item.metadata == {}
    assert item.tags == []


def test_sync_item_custom_fields():
    now = datetime.now(timezone.utc)
    item = SyncItem(
        id="2",
        content="world",
        source="slack",
        source_url="https://example.com",
        title="My Title",
        author="Bob",
        created_at=now,
        updated_at=now,
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
    )
    assert item.source_url == "https://example.com"
    assert item.title == "My Title"
    assert item.author == "Bob"
    assert item.created_at == now
    assert item.updated_at == now
    assert item.metadata == {"key": "value"}
    assert item.tags == ["tag1", "tag2"]


# --------------------------------------------------------------------------- #
# ConnectorState 枚举测试
# --------------------------------------------------------------------------- #

def test_connector_state_values():
    assert ConnectorState.DISCONNECTED == "disconnected"
    assert ConnectorState.AUTHENTICATING == "authenticating"
    assert ConnectorState.CONNECTED == "connected"
    assert ConnectorState.SYNCING == "syncing"
    assert ConnectorState.ERROR == "error"


def test_connector_state_is_str():
    assert isinstance(ConnectorState.CONNECTED, str)


# --------------------------------------------------------------------------- #
# _set_error 测试
# --------------------------------------------------------------------------- #

def test_set_error_updates_state_and_message():
    conn = DummyConnector("test")
    conn._set_error("something went wrong")
    assert conn.state == ConnectorState.ERROR
    assert conn.last_error() == "something went wrong"


# --------------------------------------------------------------------------- #
# is_available / _available 测试
# --------------------------------------------------------------------------- #

def test_is_available_true_by_default():
    conn = DummyConnector("test")
    assert conn.is_available() is True


def test_is_available_false_when_set():
    conn = DummyConnector("test")
    conn._available = False
    assert conn.is_available() is False


# --------------------------------------------------------------------------- #
# 抽象类不能直接实例化
# --------------------------------------------------------------------------- #

def test_abstract_class_cannot_instantiate():
    with pytest.raises(TypeError):
        ConnectorProtocol("test")
