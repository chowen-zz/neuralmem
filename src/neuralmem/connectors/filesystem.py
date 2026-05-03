"""本地文件系统监控连接器 — 基于 watchdog 的文件变更监听."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem

_logger = logging.getLogger(__name__)

# Graceful degradation for watchdog
try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False
    FileSystemEventHandler = object  # type: ignore[misc,assignment]


class _NeuralMemEventHandler(FileSystemEventHandler):  # type: ignore[misc]
    """watchdog 事件处理器 — 将文件变更转为 SyncItem."""

    def __init__(self, connector: "FilesystemConnector") -> None:
        self.connector = connector
        self._pending: list[SyncItem] = []

    def on_modified(self, event: Any) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self.connector._should_ignore(path):
            return
        item = self.connector._path_to_item(path)
        if item:
            self._pending.append(item)
            _logger.debug("File modified: %s", path)

    def on_created(self, event: Any) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self.connector._should_ignore(path):
            return
        item = self.connector._path_to_item(path)
        if item:
            self._pending.append(item)
            _logger.debug("File created: %s", path)

    def flush(self) -> list[SyncItem]:
        """返回并清空待处理队列."""
        items = self._pending[:]
        self._pending.clear()
        return items


class FilesystemConnector(ConnectorProtocol):
    """本地文件系统监控连接器.

    支持两种模式:
    1. 实时监听 (watchdog 可用时): 启动 observer 监控目录变更.
    2. 轮询扫描 (watchdog 不可用时): sync() 时扫描目录内容.

    Config keys
    -----------
    paths : list[str]
        监控的目录或文件路径列表.
    extensions : list[str] | None
        只同步指定扩展名, 如 ['.md', '.txt']; None 则同步所有.
    exclude_patterns : list[str]
        排除的文件名模式, 默认 ['*.tmp', '*.swp', '.git'].
    max_file_size : int
        最大文件大小(字节), 默认 10MB.
    encoding : str
        文件编码, 默认 'utf-8'.
    """

    def __init__(
        self, name: str = "filesystem", config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(name, config)
        self.paths: list[str] = self.config.get("paths", ["."])
        self.extensions: list[str] | None = self.config.get("extensions")
        self.exclude_patterns: list[str] = self.config.get(
            "exclude_patterns", ["*.tmp", "*.swp", ".git", "__pycache__", ".DS_Store"]
        )
        self.max_file_size: int = self.config.get("max_file_size", 10 * 1024 * 1024)
        self.encoding: str = self.config.get("encoding", "utf-8")

        self._observer: Any | None = None
        self._handlers: list[_NeuralMemEventHandler] = []

    def authenticate(self) -> bool:
        """验证监控路径存在且可读."""
        valid = False
        for p in self.paths:
            path = Path(p)
            if path.exists() and os.access(path, os.R_OK):
                valid = True
            else:
                _logger.warning("Filesystem path not accessible: %s", p)
        if valid:
            self.state = ConnectorState.CONNECTED
            return True
        self._set_error("No valid filesystem paths in config")
        return False

    def sync(self, **kwargs: Any) -> list[SyncItem]:
        """同步文件系统内容.

        如果 watchdog 可用且 observer 已启动, 返回事件队列中的变更.
        否则执行一次性目录扫描.

        Parameters
        ----------
        full_scan : bool
            强制全量扫描, 默认 False.
        """
        if self.state != ConnectorState.CONNECTED:
            _logger.warning("FilesystemConnector not connected")
            return []

        full_scan: bool = kwargs.get("full_scan", False)

        # 实时模式: 返回事件队列
        if _HAS_WATCHDOG and self._observer and not full_scan:
            items: list[SyncItem] = []
            for handler in self._handlers:
                items.extend(handler.flush())
            _logger.info("Filesystem sync (event mode): %d items", len(items))
            return items

        # 轮询/全量模式: 扫描目录
        return self._scan_directories()

    def disconnect(self) -> None:
        """停止 observer 并清理资源."""
        if _HAS_WATCHDOG and self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._handlers.clear()
        self.state = ConnectorState.DISCONNECTED

    # ------------------------------------------------------------------
    # Watchdog helpers
    # ------------------------------------------------------------------

    def start_watching(self) -> bool:
        """启动 watchdog observer (实时监听模式).

        Returns
        -------
        bool
            True if observer started successfully.
        """
        if not _HAS_WATCHDOG:
            _logger.warning("watchdog not installed; falling back to polling mode")
            return False
        if self._observer:
            return True
        try:
            self._observer = Observer()
            for p in self.paths:
                path = Path(p)
                if not path.exists():
                    continue
                handler = _NeuralMemEventHandler(self)
                self._handlers.append(handler)
                watch_path = str(path.resolve())
                self._observer.schedule(handler, watch_path, recursive=True)
            self._observer.start()
            _logger.info("Filesystem watching started for %d paths", len(self.paths))
            return True
        except Exception as exc:
            _logger.error("Failed to start filesystem watcher: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_directories(self) -> list[SyncItem]:
        items: list[SyncItem] = []
        for p in self.paths:
            path = Path(p)
            if not path.exists():
                continue
            if path.is_file():
                item = self._path_to_item(path)
                if item:
                    items.append(item)
            else:
                for file_path in path.rglob("*"):
                    if file_path.is_file() and not self._should_ignore(file_path):
                        item = self._path_to_item(file_path)
                        if item:
                            items.append(item)
        _logger.info("Filesystem sync (scan mode): %d items", len(items))
        return items

    def _should_ignore(self, path: Path) -> bool:
        name = path.name
        # Exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.startswith("*") and name.endswith(pattern[1:]):
                return True
            if pattern in name:
                return True
        # Extension filter
        if self.extensions and path.suffix.lower() not in self.extensions:
            return True
        # Size filter
        try:
            if path.stat().st_size > self.max_file_size:
                return True
        except OSError:
            return True
        return False

    def _path_to_item(self, path: Path) -> SyncItem | None:
        try:
            text = path.read_text(encoding=self.encoding, errors="replace")
        except (OSError, UnicodeDecodeError) as exc:
            _logger.debug("Cannot read %s: %s", path, exc)
            return None

        stat = path.stat()
        return SyncItem(
            id=str(path.resolve()),
            content=text,
            source="filesystem",
            source_url=f"file://{path.resolve()}",
            title=path.name,
            author=None,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            metadata={
                "path": str(path.resolve()),
                "size": stat.st_size,
                "extension": path.suffix,
            },
            tags=["filesystem", path.suffix.lstrip(".") or "unknown"],
        )
