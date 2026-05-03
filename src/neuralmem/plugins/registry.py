"""Plugin registry — register/discover/load plugins by name.

Follows the same pattern as connectors/registry.py and profiles/engine.py.
"""
from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any

from neuralmem.plugins.base import Plugin

_logger = logging.getLogger(__name__)


class PluginRegistry:
    """Plugin registry with registration, discovery, and lazy loading.

    Usage::

        PluginRegistry.register("logging", "neuralmem.plugins.builtin", "LoggingPlugin")
        plugin = PluginRegistry.load("logging")
        plugin.on_remember(memory)

        # Auto-discovery
        PluginRegistry.discover("neuralmem.plugins.builtin")
    """

    _registry: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a plugin class by name.

        Parameters
        ----------
        name:
            Short identifier used by ``load()``.
        module_path:
            Fully-qualified module path, e.g. ``'neuralmem.plugins.builtin'``.
        class_name:
            Class name inside the module.
        """
        cls._registry[name] = (module_path, class_name)

    @classmethod
    def unregister(cls, name: str) -> tuple[str, str]:
        """Remove a registered plugin and return its (module_path, class_name).

        Raises ``KeyError`` if the plugin is not registered.
        """
        entry = cls._registry.pop(name)
        if entry is None:
            raise KeyError(f"Plugin '{name}' not found in registry")
        return entry

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls, module_path: str) -> list[str]:
        """Auto-discover Plugin subclasses in a module and register them.

        Returns the list of discovered plugin names.
        """
        discovered: list[str] = []
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            _logger.warning("Cannot discover plugins in %s: %s", module_path, exc)
            return discovered

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, Plugin)
                and obj is not Plugin
                and not getattr(obj, "__abstractmethods__", None)
            ):
                # Use a snake_case version of the class name as the key
                name = cls._to_snake(attr_name)
                cls.register(name, module_path, attr_name)
                discovered.append(name)
                _logger.debug("Discovered plugin '%s' from %s", name, module_path)

        return discovered

    # ------------------------------------------------------------------
    # Loading / Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, name: str, **kwargs: Any) -> Plugin:
        """Instantiate and return a registered plugin.

        Raises ``ValueError`` if the plugin name is unknown.
        Raises ``ImportError`` if the module cannot be imported.
        Raises ``AttributeError`` if the class is missing.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise ValueError(
                f"Unknown plugin '{name}'. Available: {available}"
            )

        module_path, class_name = cls._registry[name]
        mod = importlib.import_module(module_path)
        plugin_cls: type[Plugin] = getattr(mod, class_name)
        return plugin_cls(**kwargs)

    @classmethod
    def load_all(cls, **kwargs: Any) -> dict[str, Plugin]:
        """Load every registered plugin and return ``{name: instance}``.

        Errors for individual plugins are logged and skipped.
        """
        result: dict[str, Plugin] = {}
        for name in sorted(cls._registry):
            try:
                result[name] = cls.load(name, **kwargs)
            except Exception as exc:
                _logger.warning("Failed to load plugin '%s': %s", name, exc)
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def list_plugins(cls) -> list[str]:
        """Return all registered plugin names."""
        return sorted(cls._registry)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return *True* if the plugin name is registered."""
        return name in cls._registry

    @classmethod
    def get_info(cls, name: str) -> dict[str, str]:
        """Return ``{module, class}`` for a registered plugin.

        Raises ``KeyError`` if the plugin is not registered.
        """
        module_path, class_name = cls._registry[name]
        return {"module": module_path, "class": class_name}

    @classmethod
    def registry_info(cls) -> dict[str, dict[str, str]]:
        """Return ``{name: {module, class}}`` for all registered plugins."""
        return {
            name: {"module": mod, "class": cls_name}
            for name, (mod, cls_name) in cls._registry.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_snake(name: str) -> str:
        """Convert CamelCase to snake_case."""
        result: list[str] = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0:
                result.append("_")
            result.append(ch.lower())
        return "".join(result)


# ------------------------------------------------------------------
# Auto-register built-in plugins
# ------------------------------------------------------------------

_BUILTIN_PLUGINS: list[tuple[str, str, str]] = [
    ("logging", "neuralmem.plugins.builtin", "LoggingPlugin"),
    ("metrics", "neuralmem.plugins.builtin", "MetricsPlugin"),
    ("validation", "neuralmem.plugins.builtin", "ValidationPlugin"),
]

for _name, _mod, _cls in _BUILTIN_PLUGINS:
    PluginRegistry.register(_name, _mod, _cls)
