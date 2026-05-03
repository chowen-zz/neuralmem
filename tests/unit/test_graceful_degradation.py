"""Tests for graceful_degradation module."""
from __future__ import annotations

import asyncio

import pytest

from neuralmem.production.graceful_degradation import (
    FallbackChain,
    GracefulDegradation,
    LLMFallback,
    with_timeout,
)


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# -- helpers ---------------------------------------------------------------
async def _ok(val):
    return val


async def _fail():
    raise RuntimeError("fail")


async def _slow():
    await asyncio.sleep(5)
    return "slow"


# -- FallbackChain tests ---------------------------------------------------
class TestFallbackChain:
    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        chain = FallbackChain("t")
        chain.add_fallback(lambda: _ok("primary"))
        result = await chain.execute()
        assert result == "primary"

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        chain = FallbackChain("t")
        chain.add_fallback(lambda: _fail())
        chain.add_fallback(lambda: _ok("fallback"))
        result = await chain.execute()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_default_when_all_fail(self):
        chain = FallbackChain("t", default="default")
        chain.add_fallback(lambda: _fail())
        chain.add_fallback(lambda: _fail())
        result = await chain.execute()
        assert result == "default"

    @pytest.mark.asyncio
    async def test_default_is_none(self):
        chain = FallbackChain("t")
        chain.add_fallback(lambda: _fail())
        result = await chain.execute()
        assert result is None

    @pytest.mark.asyncio
    async def test_args_forwarded(self):
        async def add(a, b):
            return a + b

        chain = FallbackChain("t")
        chain.add_fallback(add)
        result = await chain.execute(3, 4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_name_property(self):
        chain = FallbackChain("my_chain")
        assert chain.name == "my_chain"


# -- LLMFallback tests -----------------------------------------------------
class TestLLMFallback:
    @pytest.mark.asyncio
    async def test_llm_success(self):
        async def llm(x):
            return f"llm:{x}"

        async def rule(x):
            return f"rule:{x}"

        fb = LLMFallback(llm, rule)
        result = await fb.execute("hi")
        assert result == "llm:hi"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        async def llm(x):
            raise RuntimeError("no api key")

        async def rule(x):
            return f"rule:{x}"

        fb = LLMFallback(llm, rule)
        result = await fb.execute("hi")
        assert result == "rule:hi"


# -- with_timeout tests ----------------------------------------------------
class TestWithTimeout:
    @pytest.mark.asyncio
    async def test_completes_before_timeout(self):
        result = await with_timeout(_ok("fast"), timeout=1.0)
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_returns_default_on_timeout(self):
        result = await with_timeout(
            _slow(), timeout=0.05, default="timed_out"
        )
        assert result == "timed_out"


# -- GracefulDegradation tests ---------------------------------------------
class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_execute_basic(self):
        gd = GracefulDegradation()
        gd.add_fallback("x", _ok)
        result = await gd.execute("x", "val")
        assert result == "val"

    @pytest.mark.asyncio
    async def test_execute_unknown_chain(self):
        gd = GracefulDegradation(default="fallback")
        result = await gd.execute("nonexistent")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        gd = GracefulDegradation(timeout=0.05, default="t")
        gd.add_fallback("slow", _slow)
        result = await gd.execute("slow")
        assert result == "t"

    @pytest.mark.asyncio
    async def test_multiple_fallbacks(self):
        gd = GracefulDegradation(default="d")
        gd.add_fallback("m", _fail)
        gd.add_fallback("m", _ok)
        result = await gd.execute("m", "win")
        assert result == "win"
