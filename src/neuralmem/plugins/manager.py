"""Plugin manager — registration, lifecycle, and hook dispatch."""
from __future__ import annotations

import logging
from typing import Any

from neuralmem.plugins.base import Plugin, PluginContext

_logger = logging.getLogger(__name__)


class PluginManager:
    """Manages plugin registration and hook execution.

    Plugins are executed in priority order (lower priority = first).
    Each hook call isolates errors so one failing plugin doesn't
    prevent others from running.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._sorted: list[Plugin] = []
        self._dirty = True

    # --- Registration ---

    def register(self, plugin: Plugin) -> None:
        """Register a plugin. Calls on_init() after registration."""
        if plugin.name in self._plugins:
            raise ValueError(
                f"Plugin '{plugin.name}' is already registered"
            )
        self._plugins[plugin.name] = plugin
        self._dirty = True
        try:
            plugin.on_init()
        except Exception as exc:
            _logger.warning(
                "Plugin '%s' on_init failed: %s", plugin.name, exc
            )

    def unregister(self, name: str) -> Plugin:
        """Unregister and return a plugin by name."""
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            raise KeyError(f"Plugin '{name}' not found")
        self._dirty = True
        return plugin

    def list_plugins(self) -> list[dict[str, Any]]:
        """Return info about all registered plugins."""
        self._ensure_sorted()
        return [
            {
                "name": p.name,
                "priority": p.priority,
                "version": p.version,
            }
            for p in self._sorted
        ]

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    # --- Hook dispatch ---

    def run_hook(
        self,
        hook_name: str,
        *args: Any,
        context: PluginContext | None = None,
    ) -> Any:
        """Execute a hook on all plugins in priority order.

        For hooks that return a value (on_before_remember,
        on_before_recall, on_after_recall), the return value of
        each plugin is passed as input to the next.  The final
        result is returned to the caller.

        Error isolation: if a plugin raises, the error is logged
        and recorded in context.errors, but execution continues.
        """
        self._ensure_sorted()
        if context is None:
            context = PluginContext()

        result = args[0] if args else None
        remaining = args[1:] if len(args) > 1 else ()

        for plugin in self._sorted:
            hook = getattr(plugin, hook_name, None)
            if hook is None:
                continue
            try:
                if result is not None or remaining:
                    call_args = (result, *remaining) if remaining else (result,)
                    ret = hook(*call_args)
                    if ret is not None:
                        result = ret
                else:
                    hook()
            except Exception as exc:
                _logger.warning(
                    "Plugin '%s' hook '%s' failed: %s",
                    plugin.name,
                    hook_name,
                    exc,
                )
                context.errors.append(exc)
                # Notify the plugin itself
                try:
                    plugin.on_error(exc)
                except Exception:
                    pass  # on_error itself failed — swallow

        return result

    # --- Internal ---

    def _ensure_sorted(self) -> None:
        if self._dirty:
            self._sorted = sorted(
                self._plugins.values(), key=lambda p: p.priority
            )
            self._dirty = False
