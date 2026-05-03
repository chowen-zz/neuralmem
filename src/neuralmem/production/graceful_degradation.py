"""Graceful degradation with fallback chains."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class FallbackChain:
    """Execute a chain of async callables, falling back on failure.

    Each callable is invoked with *args and **kwargs.
    The first successful result wins.

    Parameters
    ----------
    name : str
        Identifier for logging.
    default : any
        Value returned when every fallback fails.
    """

    def __init__(
        self,
        name: str = "chain",
        *,
        default: Any = None,
    ) -> None:
        self._name = name
        self._default = default
        self._steps: list[
            Callable[..., Awaitable[Any]]
        ] = []

    @property
    def name(self) -> str:
        return self._name

    def add_fallback(
        self, fn: Callable[..., Awaitable[Any]]
    ) -> None:
        """Append a fallback step."""
        self._steps.append(fn)

    async def execute(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Run the chain; return the first successful result."""
        last_exc: Exception | None = None
        for idx, step in enumerate(self._steps):
            try:
                result = await step(*args, **kwargs)
                logger.debug(
                    "Chain '%s' step %d succeeded",
                    self._name,
                    idx,
                )
                return result
            except Exception as exc:
                last_exc = exc  # noqa: F841
                logger.warning(
                    "Chain '%s' step %d failed: %s",
                    self._name,
                    idx,
                    exc,
                )
        logger.warning(
            "Chain '%s' exhausted, returning default", self._name
        )
        return self._default


class LLMFallback:
    """LLM call with automatic fallback to rule-based extractor.

    Parameters
    ----------
    llm_fn : async callable
        Primary LLM invocation.
    rule_fn : async callable
        Rule-based fallback extractor.
    """

    def __init__(
        self,
        llm_fn: Callable[..., Awaitable[Any]],
        rule_fn: Callable[..., Awaitable[Any]],
    ) -> None:
        self._llm_fn = llm_fn
        self._rule_fn = rule_fn

    async def execute(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Try LLM first; fall back to rule-based."""
        try:
            return await self._llm_fn(*args, **kwargs)
        except Exception as exc:
            logger.warning(
                "LLM call failed (%s), using rule-based",
                exc,
            )
            return await self._rule_fn(*args, **kwargs)


async def with_timeout(
    coro: Awaitable[Any],
    timeout: float,
    default: Any = None,
) -> Any:
    """Await *coro* with a timeout; return *default* on expiry."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Coroutine timed out after %.1fs", timeout
        )
        return default


class GracefulDegradation:
    """High-level wrapper combining fallback chains with timeouts.

    Parameters
    ----------
    timeout : float
        Per-step timeout in seconds.  0 disables.
    default : any
        Global default when everything fails.
    """

    def __init__(
        self,
        *,
        timeout: float = 0.0,
        default: Any = None,
    ) -> None:
        self._timeout = timeout
        self._default = default
        self._chains: dict[str, FallbackChain] = {}

    def add_fallback(
        self,
        name: str,
        fn: Callable[..., Awaitable[Any]],
    ) -> None:
        """Add a fallback step to the named chain."""
        if name not in self._chains:
            self._chains[name] = FallbackChain(
                name, default=self._default
            )
        self._chains[name].add_fallback(fn)

    async def execute(
        self, name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the named chain with optional timeout."""
        chain = self._chains.get(name)
        if chain is None:
            return self._default
        if self._timeout > 0:
            return await with_timeout(
                chain.execute(*args, **kwargs),
                self._timeout,
                default=self._default,
            )
        return await chain.execute(*args, **kwargs)
