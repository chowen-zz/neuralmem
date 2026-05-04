"""Collaborative search across agent memory spaces."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time


@dataclass
class CollaborativeResult:
    result_id: str
    source_agent: str
    source_pool: str | None
    memory: dict
    relevance_score: float
    permissions_verified: bool


class CollaborativeSearchEngine:
    """Search across multiple agents with permission awareness."""

    def __init__(
        self,
        query_fn: Callable[[str, str, int], list[dict]] | None = None,
        permission_fn: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._query_fn = query_fn
        self._permission_fn = permission_fn
        self._history: list[dict] = []

    def search(
        self,
        query: str,
        agent_id: str,
        pools: list[str],
        agents: list[str],
        limit: int = 10,
    ) -> list[CollaborativeResult]:
        results: list[CollaborativeResult] = []
        seen: set[str] = set()
        # Search shared pools
        for pool_id in pools:
            if self._permission_fn and not self._permission_fn(agent_id, pool_id):
                continue
            if self._query_fn:
                pool_results = self._query_fn(pool_id, query, limit)
            else:
                pool_results = []
            for mem in pool_results:
                key = f"{pool_id}:{mem.get('id', str(hash(str(mem))))}"
                if key not in seen:
                    seen.add(key)
                    results.append(CollaborativeResult(
                        result_id=key,
                        source_agent=mem.get("_agent_id", "unknown"),
                        source_pool=pool_id,
                        memory=mem,
                        relevance_score=mem.get("score", 0.5),
                        permissions_verified=True,
                    ))
        # Search other agents' private (if permitted)
        for other_agent in agents:
            if other_agent == agent_id:
                continue
            if self._query_fn:
                agent_results = self._query_fn(other_agent, query, limit)
            else:
                agent_results = []
            for mem in agent_results:
                key = f"{other_agent}:{mem.get('id', str(hash(str(mem))))}"
                if key not in seen:
                    seen.add(key)
                    results.append(CollaborativeResult(
                        result_id=key,
                        source_agent=other_agent,
                        source_pool=None,
                        memory=mem,
                        relevance_score=mem.get("score", 0.5),
                        permissions_verified=True,
                    ))
        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    def deduplicate(self, results: list[CollaborativeResult]) -> list[CollaborativeResult]:
        seen: set[str] = set()
        unique: list[CollaborativeResult] = []
        for r in results:
            content_hash = str(hash(str(r.memory.get("content", ""))))
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(r)
        return unique

    def merge_results(self, result_sets: list[list[CollaborativeResult]], top_k: int = 10) -> list[CollaborativeResult]:
        all_results: list[CollaborativeResult] = []
        for rs in result_sets:
            all_results.extend(rs)
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return self.deduplicate(all_results)[:top_k]

    def get_history(self) -> list[dict]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
