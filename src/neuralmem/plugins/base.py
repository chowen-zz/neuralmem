"""Plugin base classes and context for NeuralMem plugin ecosystem."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from neuralmem.core.types import Memory, SearchQuery, SearchResult


@dataclass
class PluginContext:
    """Shared context passed to plugins during hook execution.

    Plugins can read/write arbitrary data here to communicate
    with each other without direct coupling.
    """

    data: dict[str, Any] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Plugin(ABC):
    """Abstract base class for NeuralMem plugins.

    Subclass and override any hook methods you need.
    Plugins are ordered by priority (lower = executed first).
    """

    # --- Identity ---
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name."""

    @property
    def priority(self) -> int:
        """Execution priority (lower = earlier). Default 100."""
        return 100

    @property
    def version(self) -> str:
        """Plugin version string."""
        return "0.1.0"

    # --- Lifecycle hooks ---

    def on_init(self) -> None:
        """Called when plugin is registered with PluginManager."""

    def on_before_remember(self, memory: Memory) -> Memory:
        """Called before a memory is stored.

        Return the (possibly modified) memory.
        """
        return memory

    def on_after_remember(self, memory: Memory) -> None:
        """Called after a memory has been stored."""

    def on_before_recall(self, query: SearchQuery) -> SearchQuery:
        """Called before recall/search.

        Return the (possibly modified) query.
        """
        return query

    def on_after_recall(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Called after recall/search returns results.

        Return the (possibly modified) results.
        """
        return results

    def on_before_forget(self, memory_id: str) -> None:
        """Called before a memory is deleted."""

    def on_error(self, error: Exception) -> None:
        """Called when any error occurs in the plugin pipeline."""
