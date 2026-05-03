"""Per-key token bucket rate limiter for NeuralMem."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for a rate limiter.

    Attributes
    ----------
    requests_per_minute : int
        Sustained request rate per key per minute.
    burst_size : int
        Maximum number of tokens a bucket can hold (burst capacity).
    """

    requests_per_minute: int = 60
    burst_size: int = 10


@dataclass
class _Bucket:
    """Internal token bucket state."""

    tokens: float
    last_refill: float


class RateLimiter:
    """Per-key token bucket rate limiter.

    Each key gets an independent bucket with *burst_size* tokens.
    Tokens refill at ``requests_per_minute / 60`` per second.
    Buckets idle for more than 5 minutes are automatically removed.
    """

    _IDLE_TIMEOUT: float = 300.0  # 5 minutes

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self._config = config or RateLimitConfig()
        self._refill_rate: float = self._config.requests_per_minute / 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()  # protects _buckets / _locks dicts
        self._last_cleanup: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, key: str) -> tuple[bool, dict]:
        """Check whether a request for *key* would be allowed **without**
        consuming a token.

        Returns ``(allowed, info)`` where *info* contains ``limit``,
        ``remaining``, and ``reset_at``.
        """
        bucket = self._get_or_create_bucket(key)
        lock = self._get_lock(key)
        with lock:
            self._refill(bucket)
            remaining = int(bucket.tokens)
            info = self._build_info(remaining, bucket)
            return (remaining >= 1), info

    def consume(self, key: str) -> tuple[bool, dict]:
        """Consume one token for *key* if available.

        Returns ``(True, info)`` on success, ``(False, info)`` when the
        bucket is empty.  *info* includes ``retry_after`` when denied.
        """
        bucket = self._get_or_create_bucket(key)
        lock = self._get_lock(key)
        with lock:
            self._refill(bucket)
            remaining = int(bucket.tokens)
            info = self._build_info(remaining, bucket)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                info["remaining"] = int(bucket.tokens)
                return True, info
            else:
                # How long until the next token?
                deficit = 1.0 - bucket.tokens
                info["retry_after"] = deficit / self._refill_rate if self._refill_rate > 0 else None
                return False, info

    def get_bucket(self, key: str) -> dict:
        """Return current bucket state for *key*.

        Returns a dict with ``tokens``, ``last_refill``, and ``config``.
        """
        bucket = self._get_or_create_bucket(key)
        lock = self._get_lock(key)
        with lock:
            self._refill(bucket)
            return {
                "tokens": bucket.tokens,
                "last_refill": bucket.last_refill,
                "config": {
                    "requests_per_minute": self._config.requests_per_minute,
                    "burst_size": self._config.burst_size,
                },
            }

    def reset(self, key: str) -> None:
        """Reset the bucket for *key* back to full capacity."""
        bucket = self._get_or_create_bucket(key)
        lock = self._get_lock(key)
        with lock:
            bucket.tokens = float(self._config.burst_size)
            bucket.last_refill = time.monotonic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_bucket(self, key: str) -> _Bucket:
        """Return existing bucket or create a new one (thread-safe)."""
        now = time.monotonic()
        with self._global_lock:
            if key not in self._buckets:
                self._buckets[key] = _Bucket(
                    tokens=float(self._config.burst_size),
                    last_refill=now,
                )
                self._locks[key] = threading.Lock()
            # Opportunistic cleanup
            if now - self._last_cleanup > 60.0:
                self._cleanup(now)
            return self._buckets[key]

    def _get_lock(self, key: str) -> threading.Lock:
        with self._global_lock:
            return self._locks[key]

    def _refill(self, bucket: _Bucket) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        if elapsed > 0:
            bucket.tokens = min(
                float(self._config.burst_size),
                bucket.tokens + elapsed * self._refill_rate,
            )
            bucket.last_refill = now

    def _build_info(self, remaining: int, bucket: _Bucket) -> dict:
        """Build the info dict returned to callers."""
        # reset_at: timestamp when bucket will be fully refilled (wall-clock)
        deficit = self._config.burst_size - bucket.tokens
        if deficit <= 0:
            reset_at = time.time()
        else:
            if self._refill_rate > 0:
                reset_at = time.time() + deficit / self._refill_rate
            else:
                reset_at = time.time()
        return {
            "limit": self._config.burst_size,
            "remaining": remaining,
            "reset_at": reset_at,
        }

    def _cleanup(self, now: float) -> None:
        """Remove buckets idle for more than _IDLE_TIMEOUT seconds.

        Must be called while holding ``_global_lock``.
        """
        stale_keys = [
            k
            for k, b in self._buckets.items()
            if (now - b.last_refill) > self._IDLE_TIMEOUT
        ]
        for k in stale_keys:
            del self._buckets[k]
            del self._locks[k]
        self._last_cleanup = now
