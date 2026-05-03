"""Web page extractor — HTML content extraction.

Uses ``requests`` + ``beautifulsoup4`` to fetch and parse web pages.  If
neither is available the extractor falls back to returning an empty list.

The extractor attempts to extract the main article text by looking for
common semantic tags (``article``, ``main``) and falling back to all
paragraphs.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from neuralmem.multimodal.base import BaseMultimodalExtractor, ExtractionResult

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

# Graceful dependency checks ------------------------------------------------
try:
    import requests  # type: ignore[import-untyped]

    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore[misc]
    _HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]

    _HAS_BS4 = True
except ImportError:
    BeautifulSoup = None  # type: ignore[misc]
    _HAS_BS4 = False


class WebExtractor(BaseMultimodalExtractor):
    """Extract readable text from web pages.

    Args:
        max_chunk_size: Maximum characters per returned chunk.
        timeout: HTTP request timeout in seconds. Defaults to ``30``.
        user_agent: User-Agent string for requests.
    """

    def __init__(
        self,
        max_chunk_size: int = 4096,
        *,
        timeout: float = 30.0,
        user_agent: str = "NeuralMemBot/1.0",
    ) -> None:
        super().__init__(max_chunk_size)
        self.timeout = timeout
        self.user_agent = user_agent

        if not _HAS_REQUESTS:
            _logger.warning(
                "requests is not installed. "
                "WebExtractor will be unable to fetch remote URLs."
            )
        if not _HAS_BS4:
            _logger.warning(
                "beautifulsoup4 is not installed. "
                "WebExtractor will be unable to parse HTML."
            )

    def extract(self, path_or_bytes: str | os.PathLike[str] | bytes) -> ExtractionResult:
        """Extract text from a URL or raw HTML bytes.

        Args:
            path_or_bytes: A fully-qualified URL (``https://...``) or raw
                HTML bytes.

        Returns:
            List of text chunks extracted from the page.
        """
        if isinstance(path_or_bytes, bytes):
            html = path_or_bytes.decode("utf-8", errors="replace")
        else:
            url = str(path_or_bytes)
            if url.startswith("http://") or url.startswith("https://"):
                html = self._fetch(url)
            else:
                # Treat as local file path containing HTML
                html = self._read_text(url)

        if not html:
            return []

        text = self._parse_html(html)
        return self._chunk_text(text) if text else []

    # ------------------------------------------------------------------ #
    # HTTP fetch
    # ------------------------------------------------------------------ #

    def _fetch(self, url: str) -> str:
        """Fetch HTML from a remote URL."""
        if not _HAS_REQUESTS or requests is None:
            _logger.warning("requests is not installed; cannot fetch %s", url)
            return ""

        try:
            headers = {"User-Agent": self.user_agent}
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            _logger.warning("Failed to fetch %s: %s", url, exc)
            return ""

    # ------------------------------------------------------------------ #
    # HTML parsing
    # ------------------------------------------------------------------ #

    def _parse_html(self, html: str) -> str:
        """Parse HTML and return clean article text."""
        if not _HAS_BS4 or BeautifulSoup is None:
            # Very naive fallback: strip tags with regex
            import re

            text = re.sub(r"<[^>]+>", " ", html)
            return " ".join(text.split())

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script / style / nav / footer / aside elements
            for tag_name in ("script", "style", "nav", "footer", "aside", "header"):
                for tag in soup.find_all(tag_name):
                    tag.decompose()

            # Try semantic containers first
            for selector in ("article", "main", "[role='main']"):
                container = soup.find(selector)
                if container:
                    text = container.get_text(separator="\n", strip=True)
                    if len(text) > 200:
                        return text

            # Fallback: all paragraphs
            paragraphs = soup.find_all("p")
            texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
            return "\n\n".join(texts)
        except Exception as exc:
            _logger.warning("BeautifulSoup parsing failed: %s", exc)
            return ""
