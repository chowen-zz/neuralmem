"""PredictiveRetrievalEngine — NeuralMem V1.5 predictive retrieval.

Pre-fetches memories based on user profile and context patterns to warm
HotStore caches before the user actually asks. Integrates with:
    - profiles/engine.py (ProfileEngine) for user-profile-driven predictions
    - perf/prefetch.py (PrefetchEngine) for sequence-based query prediction
    - tiered/hot_store.py (HotStore) for proactive cache warming
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from neuralmem.core.types import Memory, SearchQuery, SearchResult
from neuralmem.perf.prefetch import PrefetchEngine, PrefetchStats
from neuralmem.profiles.engine import ProfileEngine
from neuralmem.retrieval.engine import RetrievalEngine

_logger = logging.getLogger(__name__)


@dataclass
class PredictiveStats:
    """Statistics for predictive retrieval operations."""

    profile_prefetches: int = 0
    context_prefetches: int = 0
    hot_warms: int = 0
    memory_hits: int = 0
    memory_misses: int = 0
    total_latency_ms: float = 0.0
    prefetch_stats: PrefetchStats = field(default_factory=PrefetchStats)


@dataclass
class _UserContext:
    """Internal per-user context window for pattern detection."""

    user_id: str
    recent_queries: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    recent_memory_ids: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    last_activity: float = field(default_factory=time.monotonic)


class PredictiveRetrievalEngine(RetrievalEngine):
    """RetrievalEngine extension with predictive pre-fetching capabilities.

    PredictiveRetrievalEngine adds three predictive layers on top of the
    standard four-strategy retrieval:

    1. **Profile-based pre-fetching** — Uses ProfileEngine to infer a user's
       likely interests from their historical memories, then pre-fetches
       related memories into HotStore.

    2. **Context pattern prediction** — Maintains a sliding window of recent
       queries per user, detects repeating sequences, and predicts what the
       user will ask next (via PrefetchEngine).

    3. **Proactive HotStore warming** — After every search, automatically
       warms HotStore with top-k results from DeepStore so subsequent
       retrievals are sub-100ms.

    Args:
        storage: Storage backend (StorageProtocol).
        embedder: Embedding backend (EmbedderProtocol).
        graph: Graph store backend (GraphStoreProtocol).
        config: NeuralMem configuration.
        hot_store: Optional HotStore instance for proactive warming.
        profile_engine: Optional ProfileEngine for user-profile analysis.
        prefetch_engine: Optional PrefetchEngine for sequence prediction.
        max_context_window: Max number of recent queries to keep per user.
        prediction_threshold: Minimum confidence to trigger a pre-fetch.
    """

    def __init__(
        self,
        storage: Any,
        embedder: Any,
        graph: Any,
        config: Any,
        *,
        hot_store: Any | None = None,
        profile_engine: ProfileEngine | None = None,
        prefetch_engine: PrefetchEngine | None = None,
        max_context_window: int = 20,
        prediction_threshold: float = 0.5,
    ) -> None:
        super().__init__(storage, embedder, graph, config)
        self._hot_store = hot_store
        self._profile_engine = profile_engine
        self._prefetch_engine = prefetch_engine or PrefetchEngine()
        self._max_context_window = max_context_window
        self._prediction_threshold = prediction_threshold

        self._lock = threading.Lock()
        self._user_contexts: dict[str, _UserContext] = {}
        self._stats = PredictiveStats()
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._profile_cache_ttl: float = 300.0  # 5 minutes
        self._profile_cache_ts: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API overrides
    # ------------------------------------------------------------------ #

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute predictive search: standard search + post-warm HotStore."""
        t0 = time.monotonic()
        results = super().search(query)

        # Record query for pattern learning
        if query.user_id:
            self._record_user_query(query.user_id, query.query, len(results))

        # Warm HotStore with retrieved memories
        if self._hot_store is not None and results:
            self._warm_hot_store(results)

        elapsed = (time.monotonic() - t0) * 1000
        with self._lock:
            self._stats.total_latency_ms += elapsed

        return results

    def predictive_search(
        self,
        query: SearchQuery,
        *,
        warm_profile: bool = True,
        warm_context: bool = True,
    ) -> list[SearchResult]:
        """Search with pre-warming based on profile and context predictions.

        Before executing the actual search, this method:
        1. Runs profile-based pre-fetching (if warm_profile=True).
        2. Runs context-pattern pre-fetching (if warm_context=True).
        3. Executes the standard search.
        4. Warms HotStore with results.

        Returns:
            Standard SearchResult list.
        """
        t0 = time.monotonic()

        if query.user_id:
            if warm_profile:
                self._prefetch_from_profile(query.user_id)
            if warm_context:
                self._prefetch_from_context(query.user_id)

        results = super().search(query)

        if query.user_id:
            self._record_user_query(query.user_id, query.query, len(results))

        if self._hot_store is not None and results:
            self._warm_hot_store(results)

        elapsed = (time.monotonic() - t0) * 1000
        with self._lock:
            self._stats.total_latency_ms += elapsed

        return results

    # ------------------------------------------------------------------ #
    # Profile-based pre-fetching
    # ------------------------------------------------------------------ #

    def prefetch_user_profile(
        self,
        user_id: str,
        *,
        limit: int = 10,
        min_confidence: float | None = None,
    ) -> list[Memory]:
        """Pre-fetch memories likely relevant to a user based on their profile.

        Uses ProfileEngine to infer user interests, then searches for
        memories matching those interests and loads them into HotStore.

        Args:
            user_id: The user to pre-fetch for.
            limit: Max number of memories to pre-fetch.
            min_confidence: Minimum profile attribute confidence (defaults to
                self._prediction_threshold).

        Returns:
            List of Memory objects that were pre-fetched.
        """
        threshold = min_confidence if min_confidence is not None else self._prediction_threshold
        profile = self._get_cached_profile(user_id)
        if profile is None:
            return []

        predicted_queries = self._extract_predicted_queries(profile, threshold)
        if not predicted_queries:
            return []

        prefetched: list[Memory] = []
        for pred_query in predicted_queries[:3]:
            try:
                sq = SearchQuery(
                    query=pred_query,
                    user_id=user_id,
                    limit=limit,
                    min_score=0.3,
                )
                results = super().search(sq)
                for sr in results:
                    prefetched.append(sr.memory)
                if self._hot_store is not None and results:
                    self._warm_hot_store(results)
            except Exception as exc:
                _logger.warning("Profile prefetch failed for %r: %s", pred_query, exc)

        with self._lock:
            self._stats.profile_prefetches += len(prefetched)

        return prefetched[:limit]

    def _prefetch_from_profile(self, user_id: str) -> int:
        """Internal helper: pre-fetch based on profile without returning results."""
        memories = self.prefetch_user_profile(user_id)
        return len(memories)

    def _get_cached_profile(self, user_id: str) -> dict[str, Any] | None:
        """Fetch and cache user profile from ProfileEngine."""
        now = time.monotonic()
        with self._lock:
            ts = self._profile_cache_ts.get(user_id, 0)
            if now - ts < self._profile_cache_ttl and user_id in self._profile_cache:
                return self._profile_cache[user_id]

        if self._profile_engine is None or self._profile_engine.storage is None:
            return None

        try:
            memories = self._profile_engine.storage.list_memories(
                user_id=user_id, limit=500
            )
            if not memories:
                return None
            profile = self._profile_engine.build_profile(user_id, memories)
            with self._lock:
                self._profile_cache[user_id] = {
                    attr_name: {
                        "name": attr.name,
                        "value": attr.value,
                        "confidence": attr.confidence,
                        "evidence": list(attr.evidence),
                    }
                    for attr_name, attr in profile.items()
                }
                self._profile_cache_ts[user_id] = now
            return self._profile_cache[user_id]
        except Exception as exc:
            _logger.warning("Failed to build profile for %s: %s", user_id, exc)
            return None

    def _extract_predicted_queries(
        self, profile: dict[str, Any], threshold: float
    ) -> list[str]:
        """Turn profile attributes into predicted query strings."""
        queries: list[str] = []
        for attr in profile.values():
            conf = attr.get("confidence", 0.0)
            if conf < threshold:
                continue
            value = attr.get("value", {})
            if isinstance(value, dict):
                # Intent category -> query
                cat = value.get("category")
                if cat:
                    queries.append(f"{cat} related topics")
                # Knowledge domain -> query
                domain = value.get("domain")
                if domain:
                    queries.append(f"{domain} best practices")
                # Preference value -> query
                pref_val = value.get("value")
                if pref_val and isinstance(pref_val, str):
                    queries.append(f"{pref_val} tutorial")
                # Technology preference
                ptype = value.get("type")
                if ptype and pref_val:
                    queries.append(f"{pref_val} {ptype} guide")
            elif isinstance(value, str):
                queries.append(value)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            q_norm = q.lower().strip()
            if q_norm not in seen:
                seen.add(q_norm)
                unique.append(q)
        return unique

    # ------------------------------------------------------------------ #
    # Context pattern prediction
    # ------------------------------------------------------------------ #

    def predict_next_queries(self, user_id: str) -> list[str]:
        """Predict what a user will ask next based on their recent query history.

        Uses the internal PrefetchEngine to detect repeating query sequences.

        Args:
            user_id: The user to predict for.

        Returns:
            Ranked list of predicted query strings.
        """
        ctx = self._get_user_context(user_id)
        recent = list(ctx.recent_queries)
        if len(recent) < self._prefetch_engine._min_sequence_length:
            return []
        return self._prefetch_engine.predict_next(recent)

    def prefetch_from_context(
        self,
        user_id: str,
        *,
        limit: int = 10,
    ) -> list[Memory]:
        """Pre-fetch memories based on predicted next queries.

        Args:
            user_id: The user to pre-fetch for.
            limit: Max memories to pre-fetch per predicted query.

        Returns:
            List of pre-fetched Memory objects.
        """
        predictions = self.predict_next_queries(user_id)
        if not predictions:
            return []

        prefetched: list[Memory] = []
        for pred in predictions:
            try:
                sq = SearchQuery(
                    query=pred,
                    user_id=user_id,
                    limit=limit,
                    min_score=0.3,
                )
                results = super().search(sq)
                for sr in results:
                    prefetched.append(sr.memory)
                if self._hot_store is not None and results:
                    self._warm_hot_store(results)
            except Exception as exc:
                _logger.warning("Context prefetch failed for %r: %s", pred, exc)

        with self._lock:
            self._stats.context_prefetches += len(prefetched)

        return prefetched

    def _prefetch_from_context(self, user_id: str) -> int:
        """Internal helper: context-based pre-fetch without returning results."""
        memories = self.prefetch_from_context(user_id)
        return len(memories)

    # ------------------------------------------------------------------ #
    # HotStore warming
    # ------------------------------------------------------------------ #

    def warm_hot_store(
        self,
        memories: Sequence[Memory],
        *,
        force: bool = False,
    ) -> int:
        """Proactively load memories into HotStore.

        Args:
            memories: Memories to warm into HotStore.
            force: If True, warm even if memory already in HotStore.

        Returns:
            Number of memories actually warmed.
        """
        if self._hot_store is None:
            return 0

        warmed = 0
        for mem in memories:
            try:
                existing = self._hot_store.get_memory(mem.id)
                if existing is not None and not force:
                    with self._lock:
                        self._stats.memory_hits += 1
                    continue
                self._hot_store.save_memory(mem)
                warmed += 1
                with self._lock:
                    self._stats.memory_misses += 1
            except Exception as exc:
                _logger.warning("HotStore warm failed for %s: %s", mem.id, exc)

        with self._lock:
            self._stats.hot_warms += warmed

        return warmed

    def _warm_hot_store(self, results: Sequence[SearchResult]) -> int:
        """Warm HotStore from search results."""
        memories = [r.memory for r in results]
        return self.warm_hot_store(memories)

    # ------------------------------------------------------------------ #
    # User context tracking
    # ------------------------------------------------------------------ #

    def record_query(
        self,
        user_id: str,
        query: str,
        result_count: int = 0,
    ) -> None:
        """Record a user query for pattern learning.

        Also forwards to the internal PrefetchEngine for sequence detection.
        """
        self._record_user_query(user_id, query, result_count)

    def _record_user_query(
        self, user_id: str, query: str, result_count: int = 0
    ) -> None:
        """Internal: update per-user context and PrefetchEngine."""
        ctx = self._get_user_context(user_id)
        ctx.recent_queries.append(query)
        ctx.last_activity = time.monotonic()

        self._prefetch_engine.record_query(
            query, user_id=user_id, result_count=result_count
        )

    def _get_user_context(self, user_id: str) -> _UserContext:
        """Get or create per-user context window."""
        with self._lock:
            if user_id not in self._user_contexts:
                self._user_contexts[user_id] = _UserContext(
                    user_id=user_id,
                    recent_queries=deque(maxlen=self._max_context_window),
                    recent_memory_ids=deque(maxlen=50),
                )
            return self._user_contexts[user_id]

    # ------------------------------------------------------------------ #
    # Statistics & management
    # ------------------------------------------------------------------ #

    def stats(self) -> PredictiveStats:
        """Return predictive retrieval statistics."""
        with self._lock:
            prefetch_stats = self._prefetch_engine.stats()
            return PredictiveStats(
                profile_prefetches=self._stats.profile_prefetches,
                context_prefetches=self._stats.context_prefetches,
                hot_warms=self._stats.hot_warms,
                memory_hits=self._stats.memory_hits,
                memory_misses=self._stats.memory_misses,
                total_latency_ms=self._stats.total_latency_ms,
                prefetch_stats=prefetch_stats,
            )

    def reset_stats(self) -> None:
        """Reset all predictive statistics."""
        with self._lock:
            self._stats = PredictiveStats()
            self._prefetch_engine.reset()

    def clear_user_context(self, user_id: str | None = None) -> None:
        """Clear user context windows.

        Args:
            user_id: If provided, clear only that user's context.
                     If None, clear all user contexts.
        """
        with self._lock:
            if user_id is not None:
                self._user_contexts.pop(user_id, None)
                self._profile_cache.pop(user_id, None)
                self._profile_cache_ts.pop(user_id, None)
            else:
                self._user_contexts.clear()
                self._profile_cache.clear()
                self._profile_cache_ts.clear()

    def invalidate_profile_cache(self, user_id: str) -> None:
        """Invalidate cached profile for a user (e.g. after new memories)."""
        with self._lock:
            self._profile_cache.pop(user_id, None)
            self._profile_cache_ts.pop(user_id, None)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Shut down the engine and clear all predictive state."""
        try:
            super().close()
        except AttributeError:
            # super().__init__ may not have run (e.g. in mocked tests)
            pass
        self.clear_user_context()
        self.reset_stats()
