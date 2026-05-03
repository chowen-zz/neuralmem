"""Query planner that analyzes queries and selects optimal strategies."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

# Regex patterns for query analysis
_ENTITY_PATTERN = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
)
_TEMPORAL_KEYWORDS = re.compile(
    r"\b(?:when|yesterday|today|tomorrow|last|next|recent|"
    r"ago|before|after|since|during|earlier|later|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|"
    r"morning|afternoon|evening|night|week|month|year)\b",
    re.IGNORECASE,
)
_FACTUAL_KEYWORDS = re.compile(
    r"\b(?:what|who|when|where|which|how|why|"
    r"define|explain|describe|list|name|identify)\b",
    re.IGNORECASE,
)


@dataclass
class QueryProfile:
    """Analysis profile of a query."""
    query: str
    length: int = 0
    size_category: str = "short"
    has_entities: bool = False
    entity_count: int = 0
    is_temporal: bool = False
    is_factual: bool = False
    entities: list[str] = field(default_factory=list)


@dataclass
class StrategyWeights:
    """Weights for retrieval strategies."""
    semantic: float = 0.0
    keyword: float = 0.0
    graph: float = 0.0
    temporal: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dict, excluding zero weights."""
        d: dict[str, float] = {}
        if self.semantic > 0:
            d["semantic"] = self.semantic
        if self.keyword > 0:
            d["keyword"] = self.keyword
        if self.graph > 0:
            d["graph"] = self.graph
        if self.temporal > 0:
            d["temporal"] = self.temporal
        return d


class QueryPlanner:
    """Analyze queries and select optimal retrieval strategies.

    Examines query characteristics (length, entities, temporal
    markers, factual keywords) to choose the best combination
    of retrieval strategies with appropriate weights.
    """

    def __init__(
        self,
        *,
        default_weights: dict[str, float] | None = None,
    ) -> None:
        self._default_weights = StrategyWeights(
            **(default_weights or {
                "semantic": 0.4,
                "keyword": 0.3,
                "graph": 0.2,
                "temporal": 0.1,
            })
        )

    @property
    def default_weights(self) -> StrategyWeights:
        """Return a copy of the default strategy weights."""
        return StrategyWeights(
            semantic=self._default_weights.semantic,
            keyword=self._default_weights.keyword,
            graph=self._default_weights.graph,
            temporal=self._default_weights.temporal,
        )

    def analyze(self, query: str) -> QueryProfile:
        """Analyze a query and produce a profile.

        Args:
            query: The query text to analyze.

        Returns:
            QueryProfile with detected features.
        """
        length = len(query)
        if length < 50:
            size_category = "short"
        elif length <= 200:
            size_category = "medium"
        else:
            size_category = "long"

        entities = _ENTITY_PATTERN.findall(query)
        entity_count = len(entities)
        has_entities = entity_count > 0

        is_temporal = bool(_TEMPORAL_KEYWORDS.search(query))
        is_factual = bool(_FACTUAL_KEYWORDS.search(query))

        return QueryProfile(
            query=query,
            length=length,
            size_category=size_category,
            has_entities=has_entities,
            entity_count=entity_count,
            is_temporal=is_temporal,
            is_factual=is_factual,
            entities=entities,
        )

    def select_strategies(
        self, profile: QueryProfile
    ) -> dict[str, float]:
        """Select strategy weights based on query profile.

        Args:
            profile: QueryProfile from analyze().

        Returns:
            Dict mapping strategy name to weight.
        """
        weights = StrategyWeights()

        # Base: always include semantic
        weights.semantic = 0.4

        if profile.size_category == "short" and profile.is_factual:
            # Short factual: semantic + keyword
            weights.semantic = 0.6
            weights.keyword = 0.4
        elif profile.size_category == "long":
            # Long narrative: semantic + temporal + graph
            weights.semantic = 0.4
            weights.temporal = 0.3
            weights.graph = 0.3
        elif profile.has_entities and profile.entity_count >= 2:
            # Entity-heavy: graph + semantic
            weights.graph = 0.5
            weights.semantic = 0.3
            weights.keyword = 0.2
        elif profile.is_temporal:
            # Temporal: temporal + semantic
            weights.temporal = 0.5
            weights.semantic = 0.4
            weights.keyword = 0.1
        elif profile.is_factual:
            # General factual: semantic + keyword
            weights.semantic = 0.5
            weights.keyword = 0.3
            weights.graph = 0.2
        else:
            # Default: use configured defaults
            weights = StrategyWeights(
                semantic=self._default_weights.semantic,
                keyword=self._default_weights.keyword,
                graph=self._default_weights.graph,
                temporal=self._default_weights.temporal,
            )

        return weights.to_dict()

    def explain(self, query: str) -> str:
        """Provide human-readable explanation of strategy selection.

        Args:
            query: The query text to analyze and explain.

        Returns:
            Multi-line explanation string.
        """
        profile = self.analyze(query)
        strategies = self.select_strategies(profile)

        lines = [f"Query: {query!r}"]
        lines.append(f"Length: {profile.length} chars ({profile.size_category})")
        if profile.has_entities:
            lines.append(
                f"Entities: {', '.join(profile.entities)}"
            )
        lines.append(f"Temporal: {profile.is_temporal}")
        lines.append(f"Factual: {profile.is_factual}")
        lines.append("Strategy weights:")
        for name, weight in strategies.items():
            lines.append(f"  {name}: {weight:.2f}")

        return "\n".join(lines)
