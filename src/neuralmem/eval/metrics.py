"""Pure evaluation metric functions for NeuralMem retrieval quality."""
from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant items found in top-K results.

    Args:
        retrieved_ids: Ordered list of retrieved result IDs (best first).
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cut-off rank.

    Returns:
        Score in [0.0, 1.0].  Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-K results that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved result IDs (best first).
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cut-off rank.

    Returns:
        Score in [0.0, 1.0].  Returns 0.0 when k <= 0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k) if top_k else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank for a single query.

    Returns 1/rank of the first relevant result, or 0.0 if none found.
    """
    if not relevant_ids:
        return 0.0
    for idx, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / idx
    return 0.0


def stale_hit_rate(search_results: list[object]) -> float:
    """Fraction of results that are superseded (is_active=False) or expired.

    Args:
        search_results: List of SearchResult objects.  Each must have a
            ``.memory`` attribute with ``.is_active`` (bool) and
            ``.expires_at`` (datetime | None).

    Returns:
        Fraction in [0.0, 1.0].  Returns 0.0 for empty input.
    """
    if not search_results:
        return 0.0

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    stale = 0
    for sr in search_results:
        mem = sr.memory
        if not mem.is_active:
            stale += 1
        elif mem.expires_at is not None and mem.expires_at < now:
            stale += 1
    return stale / len(search_results)


def p95_latency(latencies_ms: list[float]) -> float:
    """95th-percentile retrieval latency in milliseconds.

    Uses the nearest-rank method.  Returns 0.0 for empty input.
    """
    if not latencies_ms:
        return 0.0
    sorted_lat = sorted(latencies_ms)
    idx = math.ceil(0.95 * len(sorted_lat)) - 1
    idx = max(0, min(idx, len(sorted_lat) - 1))
    return sorted_lat[idx]
