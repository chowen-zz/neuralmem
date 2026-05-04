"""Workload-aware cache with automatic strategy switching."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
import time


class CacheStrategy(Enum):
    LRU = auto()
    LFU = auto()
    PREDICTIVE = auto()


@dataclass
class CacheEntry:
    key: str
    value: Any
    access_count: int = 1
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkloadProfile:
    pattern: str = "mixed"
    hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    access_pattern: str = "random"


class WorkloadAwareCache:
    """Cache that adapts strategy based on workload patterns."""

    def __init__(self, capacity: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU) -> None:
        self.capacity = capacity
        self._strategy = strategy
        self._data: dict[str, CacheEntry] = {}
        self._access_log: list[dict] = []
        self._hit_count = 0
        self._miss_count = 0
        self._strategy_switch_callback: Callable | None = None

    def set_strategy_switch_callback(self, cb: Callable) -> None:
        self._strategy_switch_callback = cb

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry:
            entry.access_count += 1
            entry.last_access = time.time()
            self._hit_count += 1
            self._access_log.append({"key": key, "hit": True, "time": time.time()})
            return entry.value
        self._miss_count += 1
        self._access_log.append({"key": key, "hit": False, "time": time.time()})
        return None

    def put(self, key: str, value: Any) -> None:
        if len(self._data) >= self.capacity and key not in self._data:
            self._evict()
        self._data[key] = CacheEntry(key=key, value=value)

    def _evict(self) -> None:
        if not self._data:
            return
        if self._strategy == CacheStrategy.LRU:
            oldest = min(self._data.values(), key=lambda e: e.last_access)
        elif self._strategy == CacheStrategy.LFU:
            oldest = min(self._data.values(), key=lambda e: e.access_count)
        else:
            oldest = min(self._data.values(), key=lambda e: e.created_at)
        del self._data[oldest.key]

    def analyze_workload(self, window: int = 1000) -> WorkloadProfile:
        recent = self._access_log[-window:]
        if not recent:
            return WorkloadProfile()
        hits = sum(1 for r in recent if r["hit"])
        total = len(recent)
        hit_rate = hits / total if total > 0 else 0.0
        # Check access pattern
        keys = [r["key"] for r in recent]
        unique_ratio = len(set(keys)) / len(keys) if keys else 1.0
        # Check if keys follow sequential pattern (k0, k1, k2...)
        sequential = False
        if len(keys) >= 3:
            try:
                # Extract numeric suffixes
                nums = []
                for k in keys:
                    # Try to find trailing number
                    num_str = ""
                    for c in reversed(k):
                        if c.isdigit():
                            num_str = c + num_str
                        else:
                            break
                    if num_str:
                        nums.append(int(num_str))
                if len(nums) >= 3:
                    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
                    if all(d == 1 for d in diffs):
                        sequential = True
            except (ValueError, IndexError):
                pass
        if sequential or unique_ratio < 0.3:
            pattern = "sequential"
        elif unique_ratio > 0.8:
            pattern = "random"
        else:
            pattern = "mixed"
        return WorkloadProfile(
            pattern=pattern,
            hit_rate=hit_rate,
            access_pattern=pattern,
        )

    def recommend_strategy(self) -> CacheStrategy:
        profile = self.analyze_workload()
        if profile.hit_rate < 0.3:
            return CacheStrategy.PREDICTIVE
        if profile.access_pattern == "sequential":
            return CacheStrategy.LRU
        if profile.access_pattern == "random":
            return CacheStrategy.LFU
        return CacheStrategy.LRU

    def auto_switch_strategy(self) -> CacheStrategy:
        recommended = self.recommend_strategy()
        if recommended != self._strategy:
            old = self._strategy
            self._strategy = recommended
            if self._strategy_switch_callback:
                self._strategy_switch_callback(old, recommended)
        return self._strategy

    def get_stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "size": len(self._data),
            "capacity": self.capacity,
            "strategy": self._strategy.name,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / total if total > 0 else 0.0,
        }

    def reset(self) -> None:
        self._data.clear()
        self._access_log.clear()
        self._hit_count = 0
        self._miss_count = 0
