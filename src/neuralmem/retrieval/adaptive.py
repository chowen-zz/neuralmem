"""Adaptive retrieval — dynamically adjusts strategy weights via feedback."""
from __future__ import annotations

from neuralmem.core.types import SearchQuery, SearchResult
from neuralmem.retrieval.fusion import RankedItem


class AdaptiveRetriever:
    """Wraps a RetrievalEngine and adapts strategy weights over time.

    Uses exponential moving average (EMA) on per-strategy success
    rates.  Strategies that historically produce useful results get
    higher weights in the RRF fusion step.

    Usage::

        adaptive = AdaptiveRetriever(retrieval_engine)
        results = adaptive.search(query)
        adaptive.feedback("semantic", was_useful=True)
    """

    STRATEGIES = ("semantic", "keyword", "graph", "temporal")

    def __init__(
        self,
        engine: object,
        alpha: float = 0.1,
    ) -> None:
        """Initialize adaptive retriever.

        Args:
            engine: A RetrievalEngine instance with .search(SearchQuery).
            alpha: EMA smoothing factor (0 < alpha <= 1).
        """
        self._engine = engine
        self._alpha = alpha

        # EMA weights per strategy (start equal)
        self._weights: dict[str, float] = {
            s: 1.0 for s in self.STRATEGIES
        }

        # Success EMA per strategy
        self._success_ema: dict[str, float] = {
            s: 0.5 for s in self.STRATEGIES
        }

        # Raw counters for reporting
        self._success_count: dict[str, int] = {s: 0 for s in self.STRATEGIES}
        self._failure_count: dict[str, int] = {s: 0 for s in self.STRATEGIES}

        # Track last search strategy results for feedback attribution
        self._last_strategy_results: dict[str, list[RankedItem]] = {}

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Run search with weighted RRF fusion.

        Delegates to the underlying engine's search but records
        which strategies contributed for later feedback.
        """
        # Run the engine search normally — it uses equal RRF
        results: list[SearchResult] = self._engine.search(query)

        # Capture strategy attribution from results
        self._last_strategy_results = {}
        for r in results:
            method = r.retrieval_method
            if method not in self._last_strategy_results:
                self._last_strategy_results[method] = []
            self._last_strategy_results[method].append(
                RankedItem(
                    id=r.memory.id,
                    score=r.score,
                    method=method,
                )
            )

        return results

    def feedback(self, strategy_name: str, was_useful: bool) -> None:
        """Record feedback for a strategy's results.

        Updates the EMA success rate and recomputes weights.

        Args:
            strategy_name: Name of the strategy (semantic, keyword, etc.)
            was_useful: Whether the results were useful.
        """
        if strategy_name not in self._weights:
            return  # Unknown strategy — ignore

        # Update raw counters
        if was_useful:
            self._success_count[strategy_name] += 1
        else:
            self._failure_count[strategy_name] += 1

        # Update EMA
        signal = 1.0 if was_useful else 0.0
        old_ema = self._success_ema[strategy_name]
        self._success_ema[strategy_name] = (
            self._alpha * signal + (1 - self._alpha) * old_ema
        )

        # Recompute weights from EMA values
        self._update_weights()

    def get_weights(self) -> dict[str, float]:
        """Return current strategy weights as a dict."""
        return dict(self._weights)

    def get_success_rates(self) -> dict[str, float]:
        """Return current EMA success rates per strategy."""
        return dict(self._success_ema)

    def get_counters(self) -> dict[str, dict[str, int]]:
        """Return raw success/failure counters per strategy."""
        return {
            s: {
                "success": self._success_count[s],
                "failure": self._failure_count[s],
            }
            for s in self.STRATEGIES
        }

    def _update_weights(self) -> None:
        """Recompute weights from EMA success rates.

        Uses softmax-like normalization so weights sum to
        num_strategies (so equal success rates => equal weights).
        """
        ema_values = list(self._success_ema.values())
        if not ema_values:
            return

        n = len(self.STRATEGIES)
        total = sum(ema_values)
        if total == 0:
            # All zero — use equal weights
            for s in self.STRATEGIES:
                self._weights[s] = 1.0
            return

        # Linear normalization scaled by n so equal rates → weight 1.0
        for s in self.STRATEGIES:
            self._weights[s] = (self._success_ema[s] / total) * n

    def weighted_rrf_fusion(
        self,
        strategy_results: dict[str, list[RankedItem]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Perform weighted RRF fusion.

        Like standard RRF but each strategy's contribution is
        multiplied by its adaptive weight:
            score(d) = Σ weight_i / (k + rank_i(d))

        Returns list of (memory_id, normalized_score) sorted desc.
        """
        rrf_scores: dict[str, float] = {}

        for strategy, items in strategy_results.items():
            weight = self._weights.get(strategy, 1.0)
            sorted_items = sorted(
                items, key=lambda x: x.score, reverse=True
            )
            for rank, item in enumerate(sorted_items, start=1):
                contribution = weight / (k + rank)
                rrf_scores[item.id] = (
                    rrf_scores.get(item.id, 0.0) + contribution
                )

        if not rrf_scores:
            return []

        # Normalize to 0-1
        max_score = max(rrf_scores.values())
        if max_score > 0:
            rrf_scores = {
                mid: score / max_score
                for mid, score in rrf_scores.items()
            }

        return sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

    def reset(self) -> None:
        """Reset all weights and counters to defaults."""
        for s in self.STRATEGIES:
            self._weights[s] = 1.0
            self._success_ema[s] = 0.5
            self._success_count[s] = 0
            self._failure_count[s] = 0
