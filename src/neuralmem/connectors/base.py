"""连接器抽象基类 — 所有外部数据源连接器必须实现此协议."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ConnectorState(str, Enum):
    """连接器生命周期状态."""

    DISCONNECTED = "disconnected"
    AUTHENTICATING = "authenticating"
    CONNECTED = "connected"
    SYNCING = "syncing"
    ERROR = "error"


@dataclass
class SyncItem:
    """单个同步提取的内容项.

    Attributes
    ----------
    id : str
        源系统中的唯一标识.
    content : str
        提取的文本内容.
    source : str
        来源标识, 如 'notion', 'slack', 'github'.
    source_url : str | None
        原始链接.
    title : str | None
        内容标题.
    author : str | None
        作者/创建者.
    created_at : datetime
        创建时间.
    updated_at : datetime
        更新时间.
    metadata : dict
        附加元数据.
    tags : list[str]
        标签列表.
    """

    id: str
    content: str
    source: str
    source_url: str | None = None
    title: str | None = None
    author: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class ConnectorProtocol(ABC):
    """NeuralMem V1.0 连接器协议.

    所有外部数据源连接器必须继承此类并实现三个核心方法:
    - authenticate(): 建立认证连接
    - sync(): 提取内容返回 SyncItem 列表
    - disconnect(): 清理资源

    缺少可选依赖时应在 __init__ 中捕获 ImportError 并设置
    ``_available = False``, 后续方法返回空列表或抛出清晰错误.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self.state = ConnectorState.DISCONNECTED
        self._available = True
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def authenticate(self) -> bool:
        """建立与远程服务的认证连接.

        Returns
        -------
        bool
            True 表示认证成功, False 表示失败.
        """
        ...

    @abstractmethod
    def sync(self, **kwargs: Any) -> list[SyncItem]:
        """同步提取内容.

        Parameters
        ----------
        **kwargs
            连接器特定的同步参数, 如 ``since``, ``limit``.

        Returns
        -------
        list[SyncItem]
            提取的内容列表. 无内容或出错时返回空列表.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接并清理资源."""
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """返回 *True* 如果连接器依赖已安装且可用."""
        return self._available

    def last_error(self) -> str | None:
        """返回最后一次错误信息."""
        return self._last_error

    def _set_error(self, message: str) -> None:
        """记录错误并切换状态."""
        self._last_error = message
        self.state = ConnectorState.ERROR

    def __enter__(self) -> ConnectorProtocol:
        """上下文管理器入口: authenticate()."""
        self.authenticate()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器出口: disconnect()."""
        self.disconnect()
