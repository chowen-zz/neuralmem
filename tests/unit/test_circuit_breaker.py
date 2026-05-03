"""Tests for circuit_breaker module."""
from __future__ import annotations

import threading
import time

import pytest

from neuralmem.production.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state is CircuitState.CLOSED

    def test_success_stays_closed(self):
        cb = CircuitBreaker("test")
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state is CircuitState.CLOSED

    def test_failure_count_increments(self):
        cb = CircuitBreaker(
            "test", failure_threshold=3
        )
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(self._raise)
        assert cb.failure_count == 2

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(
            "test", failure_threshold=3
        )
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(self._raise)
        assert cb.state is CircuitState.OPEN

    def test_open_raises_circuit_breaker_error(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=60,
        )
        with pytest.raises(ValueError):
            cb.call(self._raise)
        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: None)

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.1,
        )
        with pytest.raises(ValueError):
            cb.call(self._raise)
        assert cb.state is CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state is CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.05,
            half_open_max_calls=1,
        )
        with pytest.raises(ValueError):
            cb.call(self._raise)
        time.sleep(0.1)
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state is CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.05,
            half_open_max_calls=1,
        )
        with pytest.raises(ValueError):
            cb.call(self._raise)
        time.sleep(0.1)
        with pytest.raises(ValueError):
            cb.call(self._raise)
        assert cb.state is CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        with pytest.raises(ValueError):
            cb.call(self._raise)
        assert cb.state is CircuitState.OPEN
        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_decorator_pattern(self):
        cb = CircuitBreaker(
            "test", failure_threshold=2
        )

        @cb
        def my_func(x):
            return x * 2

        assert my_func(3) == 6

    def test_on_state_change_callback(self):
        transitions: list[tuple] = []

        def on_change(old, new):
            transitions.append((old, new))

        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            on_state_change=on_change,
        )
        with pytest.raises(ValueError):
            cb.call(self._raise)
        assert len(transitions) == 1
        assert transitions[0] == (
            CircuitState.CLOSED,
            CircuitState.OPEN,
        )

    def test_thread_safety(self):
        cb = CircuitBreaker(
            "test", failure_threshold=100
        )
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    cb.call(lambda: 1)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert cb.state is CircuitState.CLOSED

    @staticmethod
    def _raise():
        raise ValueError("boom")
