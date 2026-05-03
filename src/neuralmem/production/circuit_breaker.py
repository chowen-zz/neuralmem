"""Circuit breaker pattern implementation."""
from __future__ import annotations

import enum
import functools
import threading
import time
from collections.abc import Callable
from typing import Any


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit breaker is open."""

    def __init__(self, name: str, remaining: float) -> None:
        self.name = name
        self.remaining = remaining
        super().__init__(
            f"Circuit '{name}' is open. "
            f"Retry in {remaining:.1f}s."
        )


class CircuitBreaker:
    """Thread-safe circuit breaker with three states.

    Parameters
    ----------
    name : str
        Identifier for this breaker.
    failure_threshold : int
        Consecutive failures before opening.
    recovery_timeout : float
        Seconds to wait before half-opening.
    half_open_max_calls : int
        Max trial calls allowed in half-open state.
    on_state_change : callable or None
        Callback(old_state, new_state) on transition.
    """

    def __init__(
        self,
        name: str = "default",
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        on_state_change: (
            Callable[[CircuitState, CircuitState], None] | None
        ) = None,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._on_state_change = on_state_change
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def _set_state(self, new: CircuitState) -> None:
        old = self._state
        if old is new:
            return
        self._state = new
        if self._on_state_change is not None:
            self._on_state_change(old, new)

    def _maybe_half_open(self) -> None:
        """Transition OPEN -> HALF_OPEN if timeout elapsed."""
        if self._state is not CircuitState.OPEN:
            return
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self._recovery_timeout:
            self._set_state(CircuitState.HALF_OPEN)
            self._half_open_calls = 0
            self._success_count = 0

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the circuit breaker."""
        with self._lock:
            self._maybe_half_open()
            state = self._state

            if state is CircuitState.OPEN:
                remaining = (
                    self._recovery_timeout
                    - (time.monotonic() - self._last_failure_time)
                )
                raise CircuitBreakerError(
                    self._name, max(0.0, remaining)
                )

            if state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    remaining = (
                        self._recovery_timeout
                        - (
                            time.monotonic()
                            - self._last_failure_time
                        )
                    )
                    raise CircuitBreakerError(
                        self._name, max(0.0, remaining)
                    )
                self._half_open_calls += 1

        # Execute outside the lock
        try:
            result = func(*args, **kwargs)
        except Exception:
            with self._lock:
                self._on_failure()
            raise
        else:
            with self._lock:
                self._on_success()
            return result

    def _on_success(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._half_open_max_calls:
                self._set_state(CircuitState.CLOSED)
                self._failure_count = 0
        else:
            self._failure_count = 0

    def _on_failure(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._set_state(CircuitState.OPEN)
            self._last_failure_time = time.monotonic()
            return
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._set_state(CircuitState.OPEN)

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        with self._lock:
            self._set_state(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator usage: ``@circuit_breaker(name='db')``."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper
