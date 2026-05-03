"""Tests for the per-key token bucket rate limiter."""

from __future__ import annotations

import threading
import time

import pytest

from neuralmem.auth.ratelimit import RateLimitConfig, RateLimiter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def limiter() -> RateLimiter:
    """Default limiter: 60 rpm, burst 10."""
    return RateLimiter()


@pytest.fixture()
def small_limiter() -> RateLimiter:
    """Limiter with tiny burst for easy testing."""
    return RateLimiter(RateLimitConfig(requests_per_minute=60, burst_size=3))


# ---------------------------------------------------------------------------
# Basic consume / check
# ---------------------------------------------------------------------------

class TestConsumeAndCheck:
    def test_consume_within_limit(self, small_limiter: RateLimiter) -> None:
        """Consuming up to burst_size should succeed."""
        for _ in range(3):
            ok, info = small_limiter.consume("k")
            assert ok is True
            assert info["limit"] == 3

    def test_consume_exceeds_burst(self, small_limiter: RateLimiter) -> None:
        """Exceeding burst_size should fail."""
        for _ in range(3):
            small_limiter.consume("k")
        ok, info = small_limiter.consume("k")
        assert ok is False
        assert info["remaining"] == 0
        assert "retry_after" in info

    def test_check_does_not_consume(self, limiter: RateLimiter) -> None:
        """check() should not reduce the token count."""
        _, info_before = limiter.check("k")
        limiter.check("k")
        limiter.check("k")
        _, info_after = limiter.check("k")
        assert info_before["remaining"] == info_after["remaining"]

    def test_consume_reduces_tokens(self, limiter: RateLimiter) -> None:
        """Each consume() should reduce remaining by 1."""
        _, info0 = limiter.consume("k")
        assert info0["remaining"] == 9
        _, info1 = limiter.consume("k")
        assert info1["remaining"] == 8

    def test_remaining_decreases(self, limiter: RateLimiter) -> None:
        """remaining in info dict should decrease after each consume."""
        for i in range(5):
            _, info = limiter.consume("k")
            assert info["remaining"] == 10 - i - 1

    def test_reset_at_in_future(self, limiter: RateLimiter) -> None:
        """reset_at should be >= now when bucket is not full."""
        limiter.consume("k")
        _, info = limiter.check("k")
        assert info["reset_at"] >= time.time()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self, limiter: RateLimiter) -> None:
        """Default config should be 60 rpm, burst 10."""
        state = limiter.get_bucket("k")
        cfg = state["config"]
        assert cfg["requests_per_minute"] == 60
        assert cfg["burst_size"] == 10

    def test_custom_config(self) -> None:
        """Custom requests_per_minute and burst_size should be honoured."""
        rl = RateLimiter(RateLimitConfig(requests_per_minute=120, burst_size=20))
        state = rl.get_bucket("k")
        cfg = state["config"]
        assert cfg["requests_per_minute"] == 120
        assert cfg["burst_size"] == 20
        # Should be able to burst 20
        for _ in range(20):
            ok, _ = rl.consume("k")
            assert ok is True
        ok, _ = rl.consume("k")
        assert ok is False


# ---------------------------------------------------------------------------
# Bucket state
# ---------------------------------------------------------------------------

class TestBucketState:
    def test_get_bucket_state(self, limiter: RateLimiter) -> None:
        """get_bucket returns tokens, last_refill, config."""
        state = limiter.get_bucket("k")
        assert "tokens" in state
        assert "last_refill" in state
        assert "config" in state
        assert state["tokens"] == 10.0

    def test_reset(self, limiter: RateLimiter) -> None:
        """reset() restores full tokens."""
        for _ in range(10):
            limiter.consume("k")
        ok, _ = limiter.consume("k")
        assert ok is False

        limiter.reset("k")
        state = limiter.get_bucket("k")
        assert state["tokens"] == 10.0

        ok, _ = limiter.consume("k")
        assert ok is True


# ---------------------------------------------------------------------------
# Separate keys
# ---------------------------------------------------------------------------

class TestSeparateKeys:
    def test_separate_keys(self, small_limiter: RateLimiter) -> None:
        """Different keys should have independent buckets."""
        for _ in range(3):
            small_limiter.consume("a")
        ok_a, _ = small_limiter.consume("a")
        assert ok_a is False

        ok_b, _ = small_limiter.consume("b")
        assert ok_b is True

    def test_multiple_keys_independent(self, limiter: RateLimiter) -> None:
        """Heavy use of one key should not affect another."""
        for _ in range(10):
            limiter.consume("heavy")
        ok, _ = limiter.consume("heavy")
        assert ok is False

        ok2, info2 = limiter.consume("light")
        assert ok2 is True
        assert info2["remaining"] == 9


# ---------------------------------------------------------------------------
# Token refill
# ---------------------------------------------------------------------------

class TestRefill:
    def test_token_refill(self, small_limiter: RateLimiter) -> None:
        """Tokens should refill over time."""
        for _ in range(3):
            small_limiter.consume("k")
        ok, _ = small_limiter.consume("k")
        assert ok is False

        # At 60 rpm = 1 token/sec, wait 1.1 sec to get 1 token back
        time.sleep(1.1)
        ok, info = small_limiter.consume("k")
        assert ok is True
        assert info["remaining"] == 0


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_consume(self) -> None:
        """Multiple threads consuming the same key should not over-spend."""
        rl = RateLimiter(RateLimitConfig(requests_per_minute=60, burst_size=50))
        results: list[bool] = []
        barrier = threading.Barrier(10)

        def worker() -> None:
            barrier.wait()
            for _ in range(10):
                ok, _ = rl.consume("shared")
                results.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 50 should succeed out of 100 attempts (burst=50, no refill time)
        assert sum(results) == 50


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_idle_cleanup(self) -> None:
        """Buckets idle for >5 minutes should be removed."""
        rl = RateLimiter()
        rl.consume("stale")
        assert "stale" in rl._buckets

        # Fast-forward the stale bucket's last_refill to 6 minutes ago
        # and also reset _last_cleanup so the next _get_or_create triggers cleanup
        with rl._global_lock:
            rl._buckets["stale"].last_refill = time.monotonic() - 360.0
            rl._last_cleanup = time.monotonic() - 120.0  # >60s ago to trigger

        # Trigger cleanup by accessing any key
        rl.consume("fresh")

        # stale should be gone after cleanup (cleanup runs on _get_or_create)
        with rl._global_lock:
            assert "stale" not in rl._buckets
