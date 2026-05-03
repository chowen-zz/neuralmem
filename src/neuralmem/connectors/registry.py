"""连接器注册表 — 参考 storage/factory.py 的注册表模式."""
from __future__ import annotations

import importlib
import logging
from typing import Any

from neuralmem.connectors.base import ConnectorProtocol

_logger = logging.getLogger(__name__)

# Maps connector name -> (module_path, class_name, pip_extra)
_CONNECTOR_REGISTRY: dict[str, tuple[str, str, str]] = {}


class ConnectorRegistry:
    """连接器注册表与工厂.

    用法::

        ConnectorRegistry.register(
            'notion',
            'neuralmem.connectors.notion',
            'NotionConnector',
            'notion',
        )
        connector = ConnectorRegistry.create('notion', config={'token': '...'})
        items = connector.sync()
    """

    _registry: dict[str, tuple[str, str, str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        module_path: str,
        class_name: str,
        pip_extra: str,
    ) -> None:
        """注册连接器.

        Parameters
        ----------
        name:
            短标识符, 用于 ``create()``.
        module_path:
            完全限定模块路径, 如 ``'neuralmem.connectors.notion'``.
        class_name:
            模块中的类名.
        pip_extra:
            依赖缺失时提示的安装 extras, 如 ``'notion'``.
        """
        cls._registry[name] = (module_path, class_name, pip_extra)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls, name: str, config: dict[str, Any] | None = None
    ) -> ConnectorProtocol:
        """实例化并返回指定连接器.

        Raises ``ValueError`` 如果连接器未知.
        Raises ``ImportError`` 如果依赖未安装.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise ValueError(
                f"Unknown connector '{name}'. Available: {available}"
            )

        module_path, class_name, pip_extra = cls._registry[name]

        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Connector '{name}' requires extra dependencies. "
                f"Install with: pip install neuralmem[{pip_extra}]"
            ) from exc

        cls_: type[ConnectorProtocol] = getattr(mod, class_name)
        return cls_(name=name, config=config or {})

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def list_connectors(cls) -> list[str]:
        """返回已注册的连接器名称列表."""
        return sorted(cls._registry)

    @classmethod
    def available_connectors(cls) -> list[str]:
        """返回依赖已安装且可用的连接器名称列表."""
        result: list[str] = []
        for name, (module_path, _cls, _extra) in cls._registry.items():
            try:
                mod = importlib.import_module(module_path)
                connector_cls = getattr(mod, _cls)
                instance = connector_cls(name=name, config={})
                if instance.is_available():
                    result.append(name)
            except Exception:
                pass
        return sorted(result)

    @classmethod
    def check_dependency(cls, name: str) -> bool:
        """返回 *True* 如果连接器的依赖已安装."""
        if name not in cls._registry:
            return False
        module_path, _cls, _extra = cls._registry[name]
        try:
            importlib.import_module(module_path)
            return True
        except ImportError:
            return False

    @classmethod
    def registry_info(cls) -> dict[str, dict[str, str]]:
        """返回 ``{name: {module, class, extra, available}}`` 字典."""
        info: dict[str, dict[str, str]] = {}
        for name, (mod, cls_name, extra) in cls._registry.items():
            info[name] = {
                "module": mod,
                "class": cls_name,
                "extra": extra,
                "available": str(cls.check_dependency(name)),
            }
        return info


# ------------------------------------------------------------------
# Auto-register all built-in connectors
# ------------------------------------------------------------------

_BUILTIN_CONNECTORS: list[tuple[str, str, str, str]] = [
    ("notion", "neuralmem.connectors.notion", "NotionConnector", "notion"),
    ("slack", "neuralmem.connectors.slack", "SlackConnector", "slack"),
    ("github", "neuralmem.connectors.github", "GitHubConnector", "github"),
    ("filesystem", "neuralmem.connectors.filesystem", "FilesystemConnector", "filesystem"),
]

for _name, _mod, _cls, _extra in _BUILTIN_CONNECTORS:
    ConnectorRegistry.register(_name, _mod, _cls, _extra)
