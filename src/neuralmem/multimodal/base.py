"""Abstract base class for all multimodal extractors.

Defines the unified ``extract(path_or_bytes) -> list[str]`` interface and
shared helpers for dependency checking, file I/O, and graceful degradation.
"""
from __future__ import annotations

import abc
import io
import logging
import os
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

ExtractionResult = list[str]


class BaseMultimodalExtractor(abc.ABC):
    """Base class for all NeuralMem multimodal extractors.

    Subclasses must implement ``extract(self, path_or_bytes) -> list[str]``.
    The constructor accepts an optional ``max_chunk_size`` parameter that
    controls the maximum length of each returned string chunk (default 4096).
    """

    def __init__(self, max_chunk_size: int = 4096) -> None:
        self.max_chunk_size = max_chunk_size

    @abc.abstractmethod
    def extract(self, path_or_bytes: str | os.PathLike[str] | bytes) -> ExtractionResult:
        """Extract text content from a file path or raw bytes.

        Args:
            path_or_bytes: File path (``str`` or ``pathlib.Path``) or raw
                ``bytes`` buffer.

        Returns:
            A list of text chunks extracted from the input. If extraction
            fails or the input is empty, an empty list is returned.
        """
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def _read_bytes(self, path_or_bytes: str | os.PathLike[str] | bytes) -> bytes:
        """Normalise *path_or_bytes* to a ``bytes`` buffer.

        Args:
            path_or_bytes: File path or raw bytes.

        Returns:
            Raw bytes.

        Raises:
            FileNotFoundError: If a path is given but does not exist.
            TypeError: If the argument is neither a path nor bytes.
        """
        if isinstance(path_or_bytes, bytes):
            return path_or_bytes
        if isinstance(path_or_bytes, (str, os.PathLike)):
            path = pathlib.Path(path_or_bytes)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return path.read_bytes()
        raise TypeError(
            f"Expected str, PathLike, or bytes, got {type(path_or_bytes).__name__}"
        )

    def _read_text(self, path_or_bytes: str | os.PathLike[str] | bytes, encoding: str = "utf-8") -> str:
        """Normalise *path_or_bytes* to a ``str``.

        Args:
            path_or_bytes: File path or raw bytes.
            encoding: Text encoding to use when decoding bytes.

        Returns:
            Decoded text string.
        """
        raw = self._read_bytes(path_or_bytes)
        return raw.decode(encoding, errors="replace")

    def _chunk_text(self, text: str) -> ExtractionResult:
        """Split *text* into chunks no larger than ``self.max_chunk_size``.

        Splits on paragraph boundaries when possible to preserve semantic
        coherence.

        Args:
            text: The full text to chunk.

        Returns:
            List of text chunks.
        """
        if not text:
            return []
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks: list[str] = []
        paragraphs = text.split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > self.max_chunk_size:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(current.strip())

        # Fallback: hard split any chunk that is still too large
        final: list[str] = []
        for chunk in chunks:
            while len(chunk) > self.max_chunk_size:
                split_at = chunk.rfind(" ", 0, self.max_chunk_size)
                if split_at == -1:
                    split_at = self.max_chunk_size
                final.append(chunk[:split_at].strip())
                chunk = chunk[split_at:].strip()
            if chunk:
                final.append(chunk)
        return final

    @staticmethod
    def _check_dependency(name: str) -> bool:
        """Return ``True`` if *name* can be imported.

        Logs a warning when a dependency is missing so that callers can
        fall back to simpler implementations.
        """
        try:
            __import__(name)
            return True
        except ImportError:
            _logger.warning("Optional dependency %r is not installed.", name)
            return False

    @staticmethod
    def _optional_import(module: str, names: Sequence[str]) -> dict[str, object] | None:
        """Import *names* from *module*, returning ``None`` on failure.

        Args:
            module: Fully-qualified module name.
            names: Names to import.

        Returns:
            Dict mapping name to imported object, or ``None`` if the module
            or any name is missing.
        """
        try:
            mod = __import__(module, fromlist=list(names))
            return {n: getattr(mod, n) for n in names}
        except (ImportError, AttributeError) as exc:
            _logger.warning("Failed to import %s from %r: %s", names, module, exc)
            return None
