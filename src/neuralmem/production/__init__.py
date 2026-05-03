"""Production hardening utilities for NeuralMem."""
from __future__ import annotations

from neuralmem.production.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)
from neuralmem.production.config_hot_reload import ConfigHotReload
from neuralmem.production.connection_pool import (
    ConnectionPool,
    SQLiteConnectionPool,
)
from neuralmem.production.graceful_degradation import (
    FallbackChain,
    GracefulDegradation,
    LLMFallback,
)
from neuralmem.production.structured_logging import StructuredLogger

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "ConfigHotReload",
    "ConnectionPool",
    "SQLiteConnectionPool",
    "FallbackChain",
    "GracefulDegradation",
    "LLMFallback",
    "StructuredLogger",
]
