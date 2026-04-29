"""生命周期 stub 测试"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from neuralmem.lifecycle.decay import DecayManager
from neuralmem.lifecycle.consolidation import MemoryConsolidator
from neuralmem.lifecycle.importance import ImportanceScorer
from neuralmem.core.protocols import LifecycleProtocol
from neuralmem.core.types import Memory

pytestmark = pytest.mark.stub


def _mock_storage():
    s = MagicMock()
    s.list_memories.return_value = []
    s.delete_memories.return_value = 0
    return s


def test_decay_manager_protocol():
    dm = DecayManager(_mock_storage())
    assert isinstance(dm, LifecycleProtocol)


def test_decay_apply_returns_zero():
    assert DecayManager(_mock_storage()).apply_decay() == 0
    assert DecayManager(_mock_storage()).apply_decay(user_id="test") == 0


def test_decay_remove_returns_zero():
    assert DecayManager(_mock_storage()).remove_forgotten() == 0


def test_consolidator_returns_zero():
    assert MemoryConsolidator().merge_similar() == 0


def test_importance_scorer_passthrough():
    scorer = ImportanceScorer()
    m = Memory(content="test", importance=0.75)
    assert scorer.score(m) == pytest.approx(0.75)


def test_importance_scorer_boundary_values():
    scorer = ImportanceScorer()
    assert scorer.score(Memory(content="x", importance=0.0)) == pytest.approx(0.0)
    assert scorer.score(Memory(content="x", importance=1.0)) == pytest.approx(1.0)
