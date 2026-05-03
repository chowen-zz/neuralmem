"""Configuration hot-reload via mtime polling."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

ConfigCallback = Callable[[dict[str, Any], dict[str, Any]], None]


class ConfigHotReload:
    """Watch a JSON config file and notify callbacks on change.

    Uses ``os.stat`` mtime polling — no external dependencies.

    Parameters
    ----------
    path : str
        Path to the JSON config file.
    poll_interval : float
        Seconds between stat checks.
    """

    def __init__(
        self,
        path: str,
        *,
        poll_interval: float = 1.0,
    ) -> None:
        self._path = path
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._current: dict[str, Any] = {}
        self._mtime: float = 0.0
        self._callbacks: list[ConfigCallback] = []
        self._running = False
        self._thread: threading.Thread | None = None

        # Initial load
        self._load()

    # -- internal ----------------------------------------------------------

    def _load(self) -> None:
        """Read and parse the config file."""
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            with self._lock:
                self._current = data
                self._mtime = os.path.getmtime(self._path)
        except FileNotFoundError:
            logger.warning(
                "Config file not found: %s", self._path
            )
        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in %s: %s", self._path, exc
            )

    def _poll(self) -> None:
        """Background polling loop."""
        while self._running:
            time.sleep(self._poll_interval)
            try:
                mtime = os.path.getmtime(self._path)
            except OSError:
                continue
            if mtime <= self._mtime:
                continue

            old = self.get_current()
            self._load()
            new = self.get_current()

            if old != new:
                logger.info(
                    "Config changed: %s", self._path
                )
                for cb in self._callbacks:
                    try:
                        cb(old, new)
                    except Exception as exc:
                        logger.error(
                            "Config callback error: %s", exc
                        )

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Start the background file watcher."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background file watcher."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval * 2)
            self._thread = None

    def on_change(self, callback: ConfigCallback) -> None:
        """Register a callback ``(old_config, new_config)``."""
        self._callbacks.append(callback)

    def get_current(self) -> dict[str, Any]:
        """Return a snapshot of the current config."""
        with self._lock:
            return dict(self._current)

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience: get a single config key."""
        with self._lock:
            return self._current.get(key, default)
