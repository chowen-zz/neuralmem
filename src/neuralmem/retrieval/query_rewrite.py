"""Query rewrite engine — expands/rewrites user queries for better retrieval.

NeuralMem V1.9 查询重写模块。支持三种策略：
1. SynonymExpander    — 同义词扩展
2. ContextEnricher    — 上下文富化（基于用户画像/历史）
3. QueryDecomposer    — 复杂查询分解为子查询

All strategies are mockable for testing (accept an optional LLM callable).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

_logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """Result of a query rewrite operation."""

    original_query: str
    rewritten_queries: list[str]  # expanded / enriched / decomposed queries
    strategy_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RewriteStrategy(ABC):
    """Abstract base for a single rewrite strategy."""

    @abstractmethod
    def rewrite(
        self,
        query: str,
        *,
        llm: Callable[[str], str] | None = None,
        max_expansion: int = 5,
        **kwargs: Any,
    ) -> RewriteResult:
        """Rewrite a query and return a RewriteResult."""
        ...


class SynonymExpander(RewriteStrategy):
    """Expand query keywords with synonyms.

    Uses a built-in synonym map plus an optional LLM for
    domain-specific expansions.
    """

    # Built-in lightweight synonym map (English + common tech terms)
    _SYNONYMS: dict[str, list[str]] = {
        "python": ["py", "python3", "cpython"],
        "javascript": ["js", "node", "nodejs", "ecmascript"],
        "database": ["db", "datastore", "storage", "sql"],
        "api": ["endpoint", "interface", "rest", "graphql"],
        "error": ["exception", "bug", "fault", "issue", "failure"],
        "fix": ["repair", "resolve", "patch", "workaround", "solution"],
        "deploy": ["release", "ship", "publish", "launch"],
        "test": ["unit test", "integration test", "qa", "validation"],
        "performance": ["speed", "latency", "throughput", "optimization"],
        "security": ["auth", "authentication", "encryption", "vulnerability"],
        "config": ["configuration", "settings", "env", "parameters"],
        "log": ["logging", "trace", "audit", "record"],
        "user": ["customer", "client", "account", "person"],
        "memory": ["recall", "retention", "storage", "cache"],
        "search": ["query", "retrieve", "find", "lookup"],
        "update": ["modify", "change", "edit", "refresh"],
        "delete": ["remove", "drop", "purge", "clear"],
        "create": ["add", "insert", "new", "generate"],
        "list": ["enumerate", "show", "display", "all"],
        "help": ["assist", "support", "guide", "documentation"],
    }

    def __init__(
        self,
        synonym_map: dict[str, list[str]] | None = None,
        use_llm: bool = False,
    ) -> None:
        self._synonyms = {**self._SYNONYMS, **(synonym_map or {})}
        self._use_llm = use_llm

    def rewrite(
        self,
        query: str,
        *,
        llm: Callable[[str], str] | None = None,
        max_expansion: int = 5,
        **kwargs: Any,
    ) -> RewriteResult:
        """Expand query with synonyms.

        Args:
            query: Original user query.
            llm: Optional LLM callable for additional expansions.
            max_expansion: Max number of expanded queries to return.
        """
        original_lower = query.lower()
        expansions: set[str] = {query}

        # 1. Built-in synonym expansion
        for keyword, syns in self._synonyms.items():
            if keyword in original_lower:
                for syn in syns:
                    expanded = query.lower().replace(keyword, syn)
                    if expanded != query.lower():
                        expansions.add(expanded)

        # 2. Optional LLM-based expansion
        if self._use_llm and llm is not None:
            try:
                llm_output = llm(
                    f"Generate up to {max_expansion} synonym variations "
                    f"for the search query: '{query}'. "
                    f"Return one per line, no numbering."
                )
                for line in llm_output.strip().splitlines():
                    line = line.strip()
                    if line and line.lower() != query.lower():
                        expansions.add(line)
            except Exception as exc:
                _logger.warning("LLM synonym expansion failed: %s", exc)

        rewritten = list(expansions)[:max_expansion]
        return RewriteResult(
            original_query=query,
            rewritten_queries=rewritten,
            strategy_used="synonym_expansion",
            metadata={
                "expansion_count": len(rewritten),
                "llm_used": self._use_llm and llm is not None,
            },
        )


class ContextEnricher(RewriteStrategy):
    """Enrich query with user-specific context (profile, recent history).

    Appends inferred user interests or recent topics to the query
    to improve retrieval relevance.
    """

    def __init__(
        self,
        context_provider: Callable[[str | None], dict[str, Any]] | None = None,
        use_llm: bool = False,
    ) -> None:
        self._context_provider = context_provider
        self._use_llm = use_llm

    def rewrite(
        self,
        query: str,
        *,
        llm: Callable[[str], str] | None = None,
        max_expansion: int = 5,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> RewriteResult:
        """Enrich query with user context.

        Args:
            query: Original user query.
            llm: Optional LLM callable for context-aware rewriting.
            max_expansion: Max number of enriched queries to return.
            user_id: User identifier for context lookup.
        """
        context = {}
        if self._context_provider is not None:
            try:
                context = self._context_provider(user_id)
            except Exception as exc:
                _logger.warning("Context provider failed: %s", exc)

        enriched: set[str] = {query}

        # 1. Append high-confidence context topics
        topics = context.get("topics", [])
        if isinstance(topics, list):
            for topic in topics[:max_expansion - 1]:
                if isinstance(topic, str):
                    enriched.add(f"{query} {topic}")

        # 2. Append recent query patterns
        recent_queries = context.get("recent_queries", [])
        if isinstance(recent_queries, list) and recent_queries:
            # Use the most recent query as context if semantically related
            recent = recent_queries[-1]
            if isinstance(recent, str) and recent.lower() != query.lower():
                enriched.add(f"{query} (context: {recent})")

        # 3. Optional LLM enrichment
        if self._use_llm and llm is not None:
            try:
                ctx_str = ", ".join(str(v) for v in topics[:3])
                prompt = (
                    f"Rewrite the search query '{query}' to be more specific "
                    f"given the user context: {ctx_str}. "
                    f"Return up to {max_expansion} variations, one per line."
                )
                llm_output = llm(prompt)
                for line in llm_output.strip().splitlines():
                    line = line.strip()
                    if line and line.lower() != query.lower():
                        enriched.add(line)
            except Exception as exc:
                _logger.warning("LLM context enrichment failed: %s", exc)

        rewritten = list(enriched)[:max_expansion]
        return RewriteResult(
            original_query=query,
            rewritten_queries=rewritten,
            strategy_used="context_enrichment",
            metadata={
                "expansion_count": len(rewritten),
                "user_id": user_id,
                "context_topics": topics[:3],
                "llm_used": self._use_llm and llm is not None,
            },
        )


class QueryDecomposer(RewriteStrategy):
    """Decompose complex multi-part queries into sub-queries.

    Uses rule-based heuristics plus optional LLM to split a
    compound query into simpler, retrievable parts.
    """

    # Conjunctions that typically indicate multiple sub-queries
    _SPLIT_TOKENS = [
        " and ", " vs ", " versus ", " compared to ", " difference between ",
        " pros and cons of ", " how to ", " what is ", " explain ",
    ]

    def __init__(self, use_llm: bool = False) -> None:
        self._use_llm = use_llm

    def rewrite(
        self,
        query: str,
        *,
        llm: Callable[[str], str] | None = None,
        max_expansion: int = 5,
        **kwargs: Any,
    ) -> RewriteResult:
        """Decompose a complex query into sub-queries.

        Args:
            query: Original user query.
            llm: Optional LLM callable for decomposition.
            max_expansion: Max number of sub-queries to return.
        """
        sub_queries: list[str] = [query]

        # 1. Rule-based splitting on conjunctions
        lower_q = query.lower()
        for token in self._SPLIT_TOKENS:
            if token in lower_q:
                parts = query.lower().split(token)
                for part in parts:
                    part = part.strip()
                    if part and part not in sub_queries:
                        sub_queries.append(part)
                break  # Only split on the first match

        # 2. Question-type decomposition
        if lower_q.startswith("what is the difference between "):
            remainder = query[len("what is the difference between "):]
            items = [i.strip() for i in remainder.split(" and ")]
            for item in items:
                if item:
                    sub_queries.append(f"what is {item}")

        elif lower_q.startswith("pros and cons of "):
            topic = query[len("pros and cons of "):].strip()
            sub_queries.extend([f"advantages of {topic}", f"disadvantages of {topic}"])

        elif lower_q.startswith("how to ") and " and " in lower_q:
            steps = [s.strip() for s in query[len("how to "):].split(" and ")]
            for i, step in enumerate(steps[:max_expansion], start=1):
                sub_queries.append(f"how to {step}")

        # 3. Optional LLM decomposition
        if self._use_llm and llm is not None:
            try:
                prompt = (
                    f"Decompose the following complex query into up to "
                    f"{max_expansion} simpler sub-queries that can be searched "
                    f"independently. Return one per line, no numbering:\n\n"
                    f"Query: {query}"
                )
                llm_output = llm(prompt)
                for line in llm_output.strip().splitlines():
                    line = line.strip("- ").strip()
                    if line and line.lower() != query.lower() and line not in sub_queries:
                        sub_queries.append(line)
            except Exception as exc:
                _logger.warning("LLM query decomposition failed: %s", exc)

        # Deduplicate and limit
        seen: set[str] = set()
        deduped: list[str] = []
        for q in sub_queries:
            q_norm = q.lower().strip()
            if q_norm not in seen:
                seen.add(q_norm)
                deduped.append(q)

        rewritten = deduped[:max_expansion]
        return RewriteResult(
            original_query=query,
            rewritten_queries=rewritten,
            strategy_used="query_decomposition",
            metadata={
                "sub_query_count": len(rewritten),
                "llm_used": self._use_llm and llm is not None,
            },
        )


class QueryRewriteEngine:
    """Orchestrates multiple query rewrite strategies.

    Usage::

        engine = QueryRewriteEngine(
            strategies=["synonym", "context", "decompose"],
            llm=mock_llm_callable,
        )
        result = engine.rewrite("python async patterns", user_id="u1")
        # result.rewritten_queries -> list of expanded/enriched/decomposed queries
    """

    _STRATEGY_MAP: dict[str, type[RewriteStrategy]] = {
        "synonym": SynonymExpander,
        "context": ContextEnricher,
        "decompose": QueryDecomposer,
    }

    def __init__(
        self,
        strategies: list[str] | None = None,
        *,
        llm: Callable[[str], str] | None = None,
        max_expansion: int = 5,
        synonym_map: dict[str, list[str]] | None = None,
        context_provider: Callable[[str | None], dict[str, Any]] | None = None,
        use_llm: bool = False,
    ) -> None:
        """Initialize the rewrite engine.

        Args:
            strategies: List of strategy names to enable.
                Default: ["synonym", "context", "decompose"].
            llm: Optional LLM callable for LLM-based rewriting.
            max_expansion: Max queries per strategy.
            synonym_map: Custom synonym dictionary.
            context_provider: Callable(user_id) -> dict with "topics" and
                "recent_queries" keys.
            use_llm: Whether to use LLM augmentation in strategies.
        """
        self._llm = llm
        self._max_expansion = max_expansion
        self._use_llm = use_llm

        if strategies is not None:
            strategy_names = list(strategies)
        else:
            strategy_names = list(self._STRATEGY_MAP.keys())
        self._strategies: list[RewriteStrategy] = []
        for name in strategy_names:
            cls = self._STRATEGY_MAP.get(name)
            if cls is None:
                _logger.warning("Unknown rewrite strategy: %s", name)
                continue
            if name == "synonym":
                self._strategies.append(
                    cls(synonym_map=synonym_map, use_llm=use_llm)
                )
            elif name == "context":
                self._strategies.append(
                    cls(context_provider=context_provider, use_llm=use_llm)
                )
            else:
                self._strategies.append(cls(use_llm=use_llm))

        self._enabled: set[str] = {s for s in strategy_names if s in self._STRATEGY_MAP}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def rewrite(
        self,
        query: str,
        *,
        user_id: str | None = None,
        strategy: str | None = None,
    ) -> RewriteResult:
        """Rewrite a query using enabled strategies.

        Args:
            query: Original user query.
            user_id: Optional user ID for context enrichment.
            strategy: If provided, use only this strategy. Otherwise
                runs all enabled strategies and merges results.

        Returns:
            RewriteResult with rewritten queries and metadata.
        """
        if not query or not query.strip():
            return RewriteResult(
                original_query=query,
                rewritten_queries=[],
                strategy_used="none",
                metadata={"error": "empty_query"},
            )

        query = query.strip()

        # Single-strategy mode
        if strategy is not None:
            for strat in self._strategies:
                cls_name = strat.__class__.__name__.lower()
                # Map strategy short names to class names
                mapping = {
                    "synonym": "synonymexpander",
                    "context": "contextenricher",
                    "decompose": "querydecomposer",
                }
                target = mapping.get(strategy, strategy.replace("_", ""))
                if cls_name == target or target in cls_name:
                    return strat.rewrite(
                        query,
                        llm=self._llm,
                        max_expansion=self._max_expansion,
                        user_id=user_id,
                    )
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy_used="none",
                metadata={"error": f"unknown_strategy: {strategy}"},
            )

        # Multi-strategy mode: run all enabled strategies and merge
        all_results: list[RewriteResult] = []
        for strat in self._strategies:
            try:
                result = strat.rewrite(
                    query,
                    llm=self._llm,
                    max_expansion=self._max_expansion,
                    user_id=user_id,
                )
                all_results.append(result)
            except Exception as exc:
                _logger.warning("Strategy %s failed: %s", strat.__class__.__name__, exc)

        if not self._strategies:
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy_used="none",
                metadata={"error": "no_strategies_enabled"},
            )

        if not all_results:
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy_used="none",
                metadata={"error": "all_strategies_failed"},
            )

        # Merge: deduplicate across strategies, preserve order
        seen: set[str] = set()
        merged: list[str] = []
        for result in all_results:
            for q in result.rewritten_queries:
                q_norm = q.lower().strip()
                if q_norm not in seen:
                    seen.add(q_norm)
                    merged.append(q)

        # Always include original query first
        if query.lower() not in seen:
            merged.insert(0, query)
        else:
            # Move original to front if it exists elsewhere
            merged = [q for q in merged if q.lower().strip() != query.lower().strip()]
            merged.insert(0, query)

        strategies_used = "+".join(r.strategy_used for r in all_results)
        return RewriteResult(
            original_query=query,
            rewritten_queries=merged[: self._max_expansion],
            strategy_used=strategies_used,
            metadata={
                "strategies_run": [r.strategy_used for r in all_results],
                "total_expansions": len(merged),
                "user_id": user_id,
            },
        )

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    def enable(self, strategy: str) -> bool:
        """Enable a strategy by name.

        Returns True if the strategy was newly enabled.
        """
        if strategy not in self._STRATEGY_MAP:
            return False
        if strategy in self._enabled:
            return False
        cls = self._STRATEGY_MAP[strategy]
        if strategy == "synonym":
            self._strategies.append(cls(use_llm=self._use_llm))
        elif strategy == "context":
            self._strategies.append(cls(use_llm=self._use_llm))
        else:
            self._strategies.append(cls(use_llm=self._use_llm))
        self._enabled.add(strategy)
        return True

    def disable(self, strategy: str) -> bool:
        """Disable a strategy by name.

        Returns True if the strategy was removed.
        """
        if strategy not in self._enabled:
            return False
        self._strategies = [
            s for s in self._strategies
            if strategy not in s.__class__.__name__.lower()
        ]
        self._enabled.discard(strategy)
        return True

    def list_strategies(self) -> list[str]:
        """Return list of currently enabled strategy names."""
        return sorted(self._enabled)

    def set_max_expansion(self, max_expansion: int) -> None:
        """Update max expansion limit."""
        self._max_expansion = max(max_expansion, 1)

    def get_max_expansion(self) -> int:
        """Return current max expansion limit."""
        return self._max_expansion
