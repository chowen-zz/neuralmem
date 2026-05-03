"""Multi-source context composer for NeuralMem.

Assembles context from multiple sources (memory, web, repo, tool traces,
custom) into a single context string that fits within a token budget,
respecting source priority and confidence-weighted allocation.
"""

from __future__ import annotations

import re

from neuralmem.context.types import ComposedContext, ContextSource

# Priority: lower index = higher priority (preserved first)
_PRIORITY_ORDER: list[ContextSource] = [
    ContextSource.memory,
    ContextSource.repo,
    ContextSource.tool_trace,
    ContextSource.web,
    ContextSource.custom,
]

# Map priority index -> source for quick lookups
_PRIORITY_RANK: dict[ContextSource, int] = {
    src: idx for idx, src in enumerate(_PRIORITY_ORDER)
}


class ContextComposer:
    """Compose context from multiple sources within a token budget.

    Parameters
    ----------
    token_budget : int
        Maximum estimated tokens for the composed output.
    tokenizer : str
        Token estimation strategy: ``"chars"`` (len/4) or ``"words"`` (len/5).
    """

    def __init__(self, token_budget: int = 4000, tokenizer: str = "chars") -> None:
        self.token_budget = token_budget
        self.tokenizer = tokenizer
        # source -> list of (content, confidence)
        self._entries: dict[ContextSource, list[tuple[str, float]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_source(
        self, source: ContextSource, content: str, confidence: float = 1.0
    ) -> None:
        """Add content from a source type."""
        self._entries.setdefault(source, []).append((content, confidence))

    def compose(self, query: str = "") -> ComposedContext:
        """Assemble all sources into a single context string within *token_budget*.

        Priority order: memory > repo > tool_trace > web > custom.
        Each source receives a proportional budget based on its aggregate
        confidence.  Lower-priority sources are truncated first.
        """
        if not self._entries:
            return ComposedContext(
                query=query, sources={}, composed="",
                token_count=0, confidence=0.0, metadata={},
            )

        # Build per-source aggregated content and average confidence
        source_info: list[tuple[ContextSource, str, float]] = []
        for src, entries in self._entries.items():
            combined = "\n".join(content for content, _ in entries)
            avg_conf = sum(c for _, c in entries) / len(entries)
            source_info.append((src, combined, avg_conf))

        # Sort by priority (highest first)
        source_info.sort(key=lambda item: _PRIORITY_RANK[item[0]])

        # Compute confidence-weighted budget allocation
        total_confidence = sum(conf for _, _, conf in source_info)
        budget_allocations: list[tuple[ContextSource, str, float, int]] = []

        for src, content, conf in source_info:
            if total_confidence > 0:
                proportion = conf / total_confidence
            else:
                proportion = 1.0 / len(source_info)
            allocated = int(self.token_budget * proportion)
            budget_allocations.append((src, content, conf, allocated))

        # Allocate budget: higher priority sources get their share first,
        # leftover rolls down to lower priority.
        allocated_tokens: list[tuple[ContextSource, str, float, int]] = []
        used = 0
        for src, content, conf, alloc in budget_allocations:
            available = max(0, self.token_budget - used)
            actual = min(alloc, available)
            allocated_tokens.append((src, content, conf, actual))
            used += actual

        # If there's leftover budget (due to rounding), distribute to
        # lower-priority sources that haven't been fully allocated yet.
        leftover = self.token_budget - used
        if leftover > 0:
            for i in range(len(allocated_tokens) - 1, -1, -1):
                src, content, conf, alloc = allocated_tokens[i]
                est = self.estimate_tokens(content)
                room = max(0, est - alloc)
                extra = min(leftover, room)
                if extra > 0:
                    allocated_tokens[i] = (src, content, conf, alloc + extra)
                    leftover -= extra
                if leftover <= 0:
                    break

        # Truncate each source to its budget and compose final string
        composed_parts: list[str] = []
        source_map: dict[ContextSource, list[str]] = {}
        confidences: list[float] = []
        total_tokens = 0

        for src, content, conf, max_tok in allocated_tokens:
            truncated = self._truncate_to_budget(content, max_tok)
            tok = self.estimate_tokens(truncated)
            if tok > 0:
                composed_parts.append(f"[{src.value}]\n{truncated}")
                source_map.setdefault(src, []).append(truncated)
                confidences.append(conf)
                total_tokens += tok

        final_composed = "\n\n".join(composed_parts)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return ComposedContext(
            query=query,
            sources=source_map,
            composed=final_composed,
            token_count=total_tokens,
            confidence=avg_confidence,
            metadata={
                "tokenizer": self.tokenizer,
                "budget": self.token_budget,
                "source_count": len(source_info),
            },
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using the configured tokenizer."""
        if not text:
            return 0
        if self.tokenizer == "words":
            return len(text.split())
        # Default: chars-based (len / 4)
        return len(text) // 4

    def clear(self) -> None:
        """Remove all sources."""
        self._entries.clear()

    def get_sources(self) -> dict[ContextSource, list[str]]:
        """Return current sources as ``{source_type: [content, ...]}``."""
        return {
            src: [content for content, _ in entries]
            for src, entries in self._entries.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Smart truncation at sentence boundaries.

        If the full text fits within *max_tokens* it is returned as-is.
        Otherwise we progressively shorten, preferring to cut at sentence
        boundaries (``.``, ``!``, ``?``).
        """
        if self.estimate_tokens(text) <= max_tokens:
            return text

        # Binary search for the longest prefix that fits within budget
        lo, hi = 0, len(text)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid]
            if self.estimate_tokens(candidate) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best >= len(text):
            return text

        truncated = text[:best]

        # Try to snap to the last sentence boundary within the truncated text
        # Look for . ! ? followed by whitespace or end-of-string
        match = None
        for m in re.finditer(r'[.!?](?:\s|$)', truncated):
            match = m
        if match is not None:
            # Include the punctuation
            end = match.end()
            # Strip trailing whitespace after punctuation
            truncated = truncated[:end].rstrip()

        return truncated
