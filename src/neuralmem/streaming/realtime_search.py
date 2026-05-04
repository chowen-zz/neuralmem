"""Near-real-time search on streaming data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time


@dataclass
class WindowAggregation:
    window_start: float
    window_end: float
    count: int = 0
    values: list[Any] = field(default_factory=list)


class RealtimeSearchEngine:
    """NRT search with sliding window aggregations."""

    def __init__(
        self,
        window_size_sec: float = 60.0,
        slide_interval_sec: float = 10.0,
        index_fn: Callable[[dict], list[str]] | None = None,
    ) -> None:
        self.window_size_sec = window_size_sec
        self.slide_interval_sec = slide_interval_sec
        self._index_fn = index_fn
        self._documents: list[dict] = []
        self._index: dict[str, list[int]] = {}
        self._last_slide = time.time()

    def index_document(self, doc: dict) -> None:
        doc_id = len(self._documents)
        self._documents.append({**doc, "_indexed_at": time.time()})
        if self._index_fn:
            terms = self._index_fn(doc)
        else:
            terms = []
            for v in doc.values():
                if isinstance(v, str):
                    terms.extend(v.lower().split())
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, str):
                            terms.extend(item.lower().split())
        for term in terms:
            if term not in self._index:
                self._index[term] = []
            self._index[term].append(doc_id)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        query_terms = query.lower().split()
        candidates: set[int] = set()
        for term in query_terms:
            if term in self._index:
                candidates.update(self._index[term])
        # Filter by window
        now = time.time()
        cutoff = now - self.window_size_sec
        results = []
        for doc_id in candidates:
            doc = self._documents[doc_id]
            if doc.get("_indexed_at", 0) >= cutoff:
                results.append(doc)
            if len(results) >= limit:
                break
        return results

    def aggregate_window(self, field: str, agg_fn: Callable | None = None) -> WindowAggregation:
        now = time.time()
        cutoff = now - self.window_size_sec
        window = WindowAggregation(window_start=cutoff, window_end=now)
        for doc in self._documents:
            if doc.get("_indexed_at", 0) >= cutoff:
                window.count += 1
                if field in doc:
                    window.values.append(doc[field])
        return window

    def slide_window(self) -> None:
        now = time.time()
        if now - self._last_slide >= self.slide_interval_sec:
            cutoff = now - self.window_size_sec
            self._documents = [d for d in self._documents if d.get("_indexed_at", 0) >= cutoff]
            # Rebuild index
            self._index.clear()
            for doc_id, doc in enumerate(self._documents):
                if self._index_fn:
                    terms = self._index_fn(doc)
                else:
                    terms = []
                    for v in doc.values():
                        if isinstance(v, str):
                            terms.extend(v.lower().split())
                        elif isinstance(v, (list, tuple)):
                            for item in v:
                                if isinstance(item, str):
                                    terms.extend(item.lower().split())
                for term in terms:
                    if term not in self._index:
                        self._index[term] = []
                    self._index[term].append(doc_id)
            self._last_slide = now

    def get_stats(self) -> dict:
        return {
            "total_docs": len(self._documents),
            "index_terms": len(self._index),
            "window_size_sec": self.window_size_sec,
        }

    def reset(self) -> None:
        self._documents.clear()
        self._index.clear()
        self._last_slide = time.time()
