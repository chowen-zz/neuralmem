"""Reasoning chain — multi-step memory retrieval with confidence scoring."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from neuralmem.core.types import SearchResult

_logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    name: str
    description: str
    input_count: int
    output_count: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result of a reasoning chain execution."""

    results: list[SearchResult]
    reasoning_trace: list[ReasoningStep]
    confidence_scores: dict[str, float]
    total_steps: int
    execution_time_ms: float


class ReasoningChain:
    """Enhances recall with multi-step reasoning.

    Pipeline:
      1. Initial recall — get candidate memories
      2. Entity expansion — find related entities via graph, recall again
      3. Deduplication and ranking — merge, dedup, re-rank
      4. Confidence scoring — score each result on source diversity,
         graph distance, recency

    Usage::

        chain = ReasoningChain(neuralmem_instance)
        result = chain.reason("What is the user's tech stack?")
    """

    def __init__(
        self,
        neuralmem: object,
        graph: object | None = None,
        max_expansion_depth: int = 2,
        recency_decay_hours: float = 720.0,  # 30 days
    ) -> None:
        """Initialize reasoning chain.

        Args:
            neuralmem: NeuralMem instance with .recall() method.
            graph: KnowledgeGraph instance. If None, uses
                neuralmem.graph.
            max_expansion_depth: BFS depth for entity expansion.
            recency_decay_hours: Half-life for recency scoring.
        """
        self._mem = neuralmem
        self._graph = graph or getattr(neuralmem, "graph", None)
        self._max_depth = max_expansion_depth
        self._recency_decay = recency_decay_hours

    def reason(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 10,
        min_score: float = 0.1,
    ) -> ReasoningResult:
        """Execute the full reasoning chain.

        Returns a ReasoningResult with scored results and trace.
        """
        start = time.monotonic()
        trace: list[ReasoningStep] = []

        # ---- Step 1: Initial recall ----
        initial_results = self._mem.recall(
            query, user_id=user_id, limit=limit
        )
        trace.append(
            ReasoningStep(
                name="initial_recall",
                description="Direct recall for query",
                input_count=0,
                output_count=len(initial_results),
                details={"query": query},
            )
        )

        # ---- Step 2: Entity expansion ----
        expanded_results = self._entity_expansion(
            query, initial_results, user_id=user_id, limit=limit
        )
        all_candidates = initial_results + [
            r for r in expanded_results
            if r.memory.id not in {x.memory.id for x in initial_results}
        ]
        trace.append(
            ReasoningStep(
                name="entity_expansion",
                description="Expand via related entities in knowledge graph",
                input_count=len(initial_results),
                output_count=len(all_candidates),
                details={
                    "new_from_expansion": len(all_candidates)
                    - len(initial_results)
                },
            )
        )

        # ---- Step 3: Dedup and rank ----
        deduped = self._dedup_and_rank(all_candidates, limit=limit)
        trace.append(
            ReasoningStep(
                name="dedup_and_rank",
                description="Remove duplicates and re-rank by score",
                input_count=len(all_candidates),
                output_count=len(deduped),
            )
        )

        # ---- Step 4: Confidence scoring ----
        confidence_scores = self._compute_confidence(
            deduped, initial_results
        )
        trace.append(
            ReasoningStep(
                name="confidence_scoring",
                description="Score results by source diversity, graph distance, recency",
                input_count=len(deduped),
                output_count=len(deduped),
            )
        )

        # Apply confidence to scores and filter
        final_results: list[SearchResult] = []
        for result in deduped:
            conf = confidence_scores.get(result.memory.id, 0.5)
            adjusted_score = result.score * conf
            if adjusted_score >= min_score:
                final_results.append(
                    SearchResult(
                        memory=result.memory,
                        score=min(1.0, adjusted_score),
                        retrieval_method=result.retrieval_method,
                        explanation=(
                            f"Reasoning chain (confidence={conf:.2f}). "
                            f"{result.explanation or ''}"
                        ),
                    )
                )
        final_results.sort(key=lambda r: r.score, reverse=True)

        elapsed_ms = (time.monotonic() - start) * 1000

        return ReasoningResult(
            results=final_results[:limit],
            reasoning_trace=trace,
            confidence_scores=confidence_scores,
            total_steps=len(trace),
            execution_time_ms=elapsed_ms,
        )

    # ---- Internal steps ----

    def _entity_expansion(
        self,
        query: str,
        initial: list[SearchResult],
        *,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Find related entities from initial results, then recall again."""
        if self._graph is None:
            return []

        # Collect entity IDs from initial results
        entity_ids: set[str] = set()
        for r in initial:
            entity_ids.update(r.memory.entity_ids)

        if not entity_ids:
            return []

        # Get neighbors
        try:
            neighbors = self._graph.get_neighbors(
                list(entity_ids), depth=self._max_depth
            )
        except Exception as exc:
            _logger.debug("Entity expansion failed: %s", exc)
            return []

        if not neighbors:
            return []

        # Build expansion query from neighbor names
        neighbor_names = [e.name for e in neighbors[:10]]
        expanded_query = f"{query} {' '.join(neighbor_names)}"

        try:
            return self._mem.recall(
                expanded_query, user_id=user_id, limit=limit
            )
        except Exception as exc:
            _logger.debug("Expanded recall failed: %s", exc)
            return []

    def _dedup_and_rank(
        self,
        results: list[SearchResult],
        limit: int = 10,
    ) -> list[SearchResult]:
        """Deduplicate by memory ID, keep highest score, re-sort."""
        seen: dict[str, SearchResult] = {}
        for r in results:
            mid = r.memory.id
            if mid not in seen or r.score > seen[mid].score:
                seen[mid] = r
        deduped = sorted(
            seen.values(), key=lambda r: r.score, reverse=True
        )
        return deduped[:limit]

    def _compute_confidence(
        self,
        results: list[SearchResult],
        initial_results: list[SearchResult],
    ) -> dict[str, float]:
        """Compute confidence scores for each result.

        Factors:
          - Source diversity: appeared in multiple retrieval methods
          - Initial vs expansion: boost for initial hits
          - Recency: newer memories score higher
          - Importance: higher importance → higher confidence
        """
        initial_ids = {r.memory.id for r in initial_results}
        now = time.time()

        scores: dict[str, float] = {}
        for r in results:
            conf = 0.5  # baseline

            # Boost for initial recall hit
            if r.memory.id in initial_ids:
                conf += 0.15

            # Importance contribution
            conf += 0.1 * r.memory.importance

            # Recency contribution
            try:
                created_ts = r.memory.created_at.timestamp()
                age_hours = (now - created_ts) / 3600.0
                import math

                decay = math.exp(
                    -age_hours / self._recency_decay * math.log(2)
                )
                conf += 0.15 * decay
            except Exception:
                pass

            # Entity richness: more entities = higher confidence
            entity_count = len(r.memory.entity_ids)
            conf += min(0.1, 0.02 * entity_count)

            scores[r.memory.id] = min(1.0, max(0.0, conf))

        return scores
