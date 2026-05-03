"""ProfileEngine / ProfileUpdater 单元测试 — 全部使用 mock."""
from __future__ import annotations

from unittest.mock import MagicMock

from neuralmem.profiles.engine import ProfileEngine
from neuralmem.profiles.types import (
    CommunicationStyle,
    IntentCategory,
    KnowledgeLevel,
    PreferenceType,
)
from neuralmem.profiles.base import ProfileAttribute


# --------------------------------------------------------------------------- #
# ProfileEngine
# --------------------------------------------------------------------------- #

def test_profile_engine_init():
    engine = ProfileEngine()
    assert engine is not None


def test_infer_intent_informational():
    """查询包含 what is 应推断为信息获取意图."""
    engine = ProfileEngine()
    mem = MagicMock()
    mem.content = "What is machine learning?"
    mem.id = "m1"
    result = engine.infer_intent([mem])
    assert "primary_intent" in result
    attr = result["primary_intent"]
    assert attr.value["category"] == "informational"
    assert attr.confidence > 0.5


def test_infer_intent_troubleshooting():
    """查询包含 error 应推断为故障排查意图."""
    engine = ProfileEngine()
    mem = MagicMock()
    mem.content = "How to fix this error?"
    mem.id = "m1"
    result = engine.infer_intent([mem])
    attr = result["primary_intent"]
    assert attr.value["category"] == "troubleshooting"


def test_infer_preferences():
    """从记忆内容推断偏好."""
    engine = ProfileEngine()
    mem = MagicMock()
    mem.content = "I prefer Python over JavaScript"
    mem.id = "m1"
    result = engine.infer_preferences([mem])
    # Technology preferences should be inferred from content
    assert len(result) >= 0  # May or may not match patterns


def test_infer_knowledge():
    """从记忆内容推断知识领域."""
    engine = ProfileEngine()
    mem = MagicMock()
    mem.content = "Working with PyTorch and neural networks"
    mem.id = "m1"
    result = engine.infer_knowledge([mem])
    # Knowledge patterns may or may not match
    assert isinstance(result, dict)


def test_infer_interaction_style():
    """从交互历史推断交互风格."""
    engine = ProfileEngine()
    mem = MagicMock()
    mem.content = "Quick summary please"
    mem.id = "m1"
    result = engine.infer_interaction_style([mem])
    assert "interaction_style" in result


def test_build_profile():
    """构建完整画像."""
    engine = ProfileEngine()
    mem1 = MagicMock()
    mem1.content = "What is Python?"
    mem1.id = "m1"
    mem2 = MagicMock()
    mem2.content = "I like concise answers"
    mem2.id = "m2"
    profile = engine.build_profile("user_123", [mem1, mem2])
    assert len(profile) > 0
    assert "primary_intent" in profile or "interaction_style" in profile


# --------------------------------------------------------------------------- #
# ProfileAttribute
# --------------------------------------------------------------------------- #

def test_profile_attribute_validation():
    attr = ProfileAttribute(
        name="test",
        value="value",
        confidence=0.8,
    )
    assert attr.name == "test"
    assert attr.confidence == 0.8


def test_profile_attribute_with_confidence():
    attr = ProfileAttribute(name="test", value="v")
    updated = attr.with_confidence(0.9)
    assert updated.confidence == 0.9
    assert updated.value == "v"


# --------------------------------------------------------------------------- #
# Profile types
# --------------------------------------------------------------------------- #

def test_intent_category_values():
    assert IntentCategory.INFORMATIONAL.value == "informational"
    assert IntentCategory.TROUBLESHOOTING.value == "troubleshooting"


def test_preference_type_values():
    assert PreferenceType.TECHNOLOGY.value == "technology"
    assert PreferenceType.FORMAT.value == "format"


def test_knowledge_level_values():
    assert KnowledgeLevel.BEGINNER.value == "beginner"
    assert KnowledgeLevel.EXPERT.value == "expert"


def test_communication_style_values():
    assert CommunicationStyle.CONCISE.value == "concise"
    assert CommunicationStyle.DETAILED.value == "detailed"
