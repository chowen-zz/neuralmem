"""PDF text extractor — PyMuPDF (fitz) with pure-text fallback.

If ``PyMuPDF`` (``fitz``) is available the extractor parses the PDF and
returns text per page.  When the library is missing it falls back to reading
the raw bytes as UTF-8 (useful for text-based PDFs or when the caller has
already extracted text externally).
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from neuralmem.multimodal.base import BaseMultimodalExtractor, ExtractionResult

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

# Graceful dependency check -------------------------------------------------
try:
    import fitz  # type: ignore[import-untyped]  # PyMuPDF

    _HAS_FITZ = True
except ImportError:
    fitz = None  # type: ignore[misc]
    _HAS_FITZ = False


class PDFExtractor(BaseMultimodalExtractor):
    """Extract text from PDF files.

    Args:
        max_chunk_size: Maximum characters per returned chunk.
    """

    def __init__(self, max_chunk_size: int = 4096) -> None:
        super().__init__(max_chunk_size)
        if not _HAS_FITZ:
            _logger.warning(
                "PyMuPDF (fitz) is not installed. "
                "PDFExtractor will fall back to raw text decoding."
            )

    def extract(self, path_or_bytes: str | os.PathLike[str] | bytes) -> ExtractionResult:
        """Extract text from a PDF file or bytes buffer.

        Args:
            path_or_bytes: Path to a ``.pdf`` file or raw PDF bytes.

        Returns:
            List of text chunks, one per page (when PyMuPDF is available) or
            a single fallback chunk.
        """
        raw = self._read_bytes(path_or_bytes)
        if not raw:
            return []

        if _HAS_FITZ and fitz is not None:
            return self._extract_with_fitz(raw)
        return self._fallback(raw)

    # ------------------------------------------------------------------ #
    # PyMuPDF implementation
    # ------------------------------------------------------------------ #

    def _extract_with_fitz(self, data: bytes) -> ExtractionResult:
        """Use PyMuPDF to extract text page-by-page."""
        chunks: list[str] = []
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            _logger.warning("fitz failed to open PDF: %s", exc)
            return self._fallback(data)

        try:
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text:
                        chunks.extend(self._chunk_text(text))
                except Exception as exc:
                    _logger.warning("fitz failed on page %d: %s", page_num, exc)
        finally:
            doc.close()

        return chunks if chunks else self._fallback(data)

    # ------------------------------------------------------------------ #
    # Fallback implementation
    # ------------------------------------------------------------------ #

    def _fallback(self, data: bytes) -> ExtractionResult:
        """Decode raw PDF bytes as UTF-8 with replacement characters."""
        text = data.decode("utf-8", errors="replace")
        return self._chunk_text(text)
