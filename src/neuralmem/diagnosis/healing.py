"""Auto-healing system for self-repair."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
import time


class HealingAction(Enum):
    REBUILD_INDEX = auto()
    CLEAR_CACHE = auto()
    RESTART_SERVICE = auto()
    ADJUST_PARAMS = auto()
    NOTIFY_ADMIN = auto()


@dataclass
class HealingResult:
    action: HealingAction
    success: bool
    description: str
    timestamp: float = field(default_factory=time.time)


class AutoHealingSystem:
    """Automatically detect and repair common issues."""

    def __init__(
        self,
        rebuild_fn: Callable | None = None,
        clear_cache_fn: Callable | None = None,
        restart_fn: Callable | None = None,
        adjust_fn: Callable | None = None,
    ) -> None:
        self._rebuild = rebuild_fn
        self._clear_cache = clear_cache_fn
        self._restart = restart_fn
        self._adjust = adjust_fn
        self._history: list[HealingResult] = []
        self._rules: list[dict] = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        self._rules = [
            {"condition": "index_corrupt", "action": HealingAction.REBUILD_INDEX},
            {"condition": "cache_stale", "action": HealingAction.CLEAR_CACHE},
            {"condition": "service_unresponsive", "action": HealingAction.RESTART_SERVICE},
            {"condition": "high_latency", "action": HealingAction.ADJUST_PARAMS},
        ]

    def diagnose_and_heal(self, symptoms: dict[str, Any]) -> list[HealingResult]:
        results = []
        for rule in self._rules:
            if rule["condition"] in symptoms and symptoms[rule["condition"]]:
                result = self._execute_action(rule["action"], symptoms)
                results.append(result)
        return results

    def _execute_action(self, action: HealingAction, context: dict) -> HealingResult:
        success = False
        description = ""
        if action == HealingAction.REBUILD_INDEX and self._rebuild:
            try:
                self._rebuild()
                success = True
                description = "Index rebuilt successfully"
            except Exception as e:
                description = f"Index rebuild failed: {e}"
        elif action == HealingAction.CLEAR_CACHE and self._clear_cache:
            try:
                self._clear_cache()
                success = True
                description = "Cache cleared successfully"
            except Exception as e:
                description = f"Cache clear failed: {e}"
        elif action == HealingAction.RESTART_SERVICE and self._restart:
            try:
                self._restart()
                success = True
                description = "Service restarted"
            except Exception as e:
                description = f"Restart failed: {e}"
        elif action == HealingAction.ADJUST_PARAMS and self._adjust:
            try:
                self._adjust(context.get("params", {}))
                success = True
                description = "Parameters adjusted"
            except Exception as e:
                description = f"Parameter adjustment failed: {e}"
        else:
            description = f"No handler for action {action.name}"
        result = HealingResult(action=action, success=success, description=description)
        self._history.append(result)
        return result

    def add_rule(self, condition: str, action: HealingAction) -> None:
        self._rules.append({"condition": condition, "action": action})

    def get_history(self) -> list[HealingResult]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._setup_default_rules()
