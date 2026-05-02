"""Shared fixtures for neuralmem-langchain tests."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import Memory, MemoryScope, MemoryType, SearchResult


@pytest.fixture
def sample_memory() -> Memory:
    return Memory(
        id="abc123def456",
        content="User prefers Python for backend development",
        memory_type=MemoryType.SEMANTIC,
        scope=MemoryScope.USER,
        user_id="test-user",
        tags=("preference", "technology"),
        importance=0.8,
        created_at=datetime(2026, 4, 30, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_result(sample_memory: Memory) -> SearchResult:
    return SearchResult(
        memory=sample_memory,
        score=0.85,
        retrieval_method="semantic",
    )


@pytest.fixture
def mock_mem(sample_result: SearchResult) -> MagicMock:
    """NeuralMem mock — recall() returns one SearchResult by default."""
    mem = MagicMock()
    mem.recall.return_value = [sample_result]
    return mem
