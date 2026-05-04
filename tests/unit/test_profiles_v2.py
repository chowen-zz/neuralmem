"""DeepProfileEngine V2 单元测试 — 全部使用 mock.

Tests all key features:
  - DeepProfileEngine extends ProfileEngine
  - Behavior pattern extraction from memory access logs
  - Preference inference with confidence scoring
  - Temporal preference drift detection
  - Cross-domain preference transfer
  - LLM-based deep inference with fallback
  - Continuous learning loop
  - Integration with existing Profile and Preference types
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.types import Memory, MemoryType
from neuralmem.profiles.base import ProfileAttribute
from neuralmem.profiles.engine import ProfileEngine
from neuralmem.profiles.types import (
    CommunicationStyle,
    IntentCategory,
    KnowledgeLevel,
    PreferenceType,
    ProfileDimension,
)
from neuralmem.profiles.v2_engine import AccessLogEntry, DeepProfileEngine


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def engine():
    return DeepProfileEngine()


@pytest.fixture
def mock_memory():
    """Return a mock Memory-like object."""
    mem = MagicMock()
    mem.content = "What is Python?"
    mem.id = "m1"
    mem.created_at = datetime.now(timezone.utc)
    return mem


@pytest.fixture
def mock_memories():
    """Return a list of mock Memory objects."""
    mems = []
    contents = [
        "What is Python?",
        "How to write Python code?",
        "Python vs JavaScript comparison",
        "Quick summary please",
        "Deep dive into Python internals",
    ]
    for i, content in enumerate(contents):
        mem = MagicMock()
        mem.content = content
        mem.id = f"m{i+1}"
        mem.created_at = datetime.now(timezone.utc) - timedelta(hours=i)
        mems.append(mem)
    return mems


@pytest.fixture
def mock_llm_client():
    """Return a mock LLM client that returns JSON responses."""
    def client(prompt: str, system: str | None = None) -> str:
        return json.dumps({
            "category": "informational",
            "subcategory": "code",
            "keywords": ["python", "code"],
            "confidence": 0.85,
        })
    return client


# --------------------------------------------------------------------------- #
# Inheritance & initialization
# --------------------------------------------------------------------------- #

def test_deep_profile_engine_extends_profile_engine():
    """DeepProfileEngine must be a subclass of ProfileEngine."""
    assert issubclass(DeepProfileEngine, ProfileEngine)


def test_init_default():
    engine = DeepProfileEngine()
    assert engine.learning_rate == 0.3
    assert engine.drift_window_days == 7.0
    assert engine.min_samples_for_drift == 5
    assert engine.llm_client is None
    assert engine.storage is None


def test_init_custom_params():
    engine = DeepProfileEngine(
        learning_rate=0.5,
        drift_window_days=14.0,
        min_samples_for_drift=10,
    )
    assert engine.learning_rate == 0.5
    assert engine.drift_window_days == 14.0
    assert engine.min_samples_for_drift == 10


def test_init_bounds_params():
    engine = DeepProfileEngine(learning_rate=2.0, drift_window_days=0.5, min_samples_for_drift=1)
    assert engine.learning_rate == 1.0
    assert engine.drift_window_days == 1.0
    assert engine.min_samples_for_drift == 3


# --------------------------------------------------------------------------- #
# Behavior pattern extraction from memory access logs
# --------------------------------------------------------------------------- #

def test_log_access(engine):
    entry = AccessLogEntry(
        memory_id="mem1",
        user_id="u1",
        action="search",
        query_text="Python tutorial for beginners",
        dwell_time_ms=5000,
    )
    engine.log_access(entry)
    assert len(engine._access_logs) == 1
    assert engine._access_logs[0].memory_id == "mem1"


def test_extract_behavior_patterns_empty(engine):
    """No logs should return empty dict."""
    result = engine.extract_behavior_patterns("u1")
    assert result == {}


def test_extract_behavior_patterns_query_depth(engine):
    logs = [
        AccessLogEntry("m1", "u1", "search", query_text="deep dive into python internals", dwell_time_ms=15000),
        AccessLogEntry("m2", "u1", "search", query_text="python source code analysis", dwell_time_ms=20000),
        AccessLogEntry("m3", "u1", "read", query_text="brief python overview", dwell_time_ms=3000),
    ]
    result = engine.extract_behavior_patterns("u1", logs)
    assert "behavior_query_depth" in result
    attr = result["behavior_query_depth"]
    assert attr.value["primary_depth"] == "deep"
    assert attr.confidence > 0.5


def test_extract_behavior_patterns_engagement(engine):
    logs = [
        AccessLogEntry("m1", "u1", "write", query_text="implement python parser", dwell_time_ms=30000),
        AccessLogEntry("m2", "u1", "update", query_text="build python web app", dwell_time_ms=25000),
        AccessLogEntry("m3", "u1", "search", query_text="look at python examples", dwell_time_ms=5000),
    ]
    result = engine.extract_behavior_patterns("u1", logs)
    assert "behavior_engagement" in result
    attr = result["behavior_engagement"]
    assert attr.value["primary_style"] == "active"
    assert attr.value["action_distribution"]["write"] == 1
    assert attr.value["action_distribution"]["update"] == 1


def test_extract_behavior_patterns_learning(engine):
    logs = [
        AccessLogEntry("m1", "u1", "search", query_text="python getting started tutorial", dwell_time_ms=10000),
        AccessLogEntry("m2", "u1", "read", query_text="python beginner hello world", dwell_time_ms=8000),
        AccessLogEntry("m3", "u1", "search", query_text="python documentation reference", dwell_time_ms=5000),
    ]
    result = engine.extract_behavior_patterns("u1", logs)
    assert "behavior_learning" in result
    attr = result["behavior_learning"]
    assert attr.value["primary_pattern"] == "tutorial_first"


def test_extract_behavior_patterns_uses_internal_logs(engine):
    """When no access_logs arg provided, should filter internal logs by user_id."""
    engine.log_access(AccessLogEntry("m1", "u1", "search", query_text="quick summary"))
    engine.log_access(AccessLogEntry("m2", "u2", "search", query_text="deep dive"))
    result = engine.extract_behavior_patterns("u1")
    assert "behavior_query_depth" in result


# --------------------------------------------------------------------------- #
# Preference inference with confidence scoring
# --------------------------------------------------------------------------- #

def test_infer_preferences_with_confidence_empty_raises(engine):
    with pytest.raises(ValueError, match="memories must not be empty"):
        engine.infer_preferences_with_confidence([])


def test_infer_preferences_with_confidence_basic(engine, mock_memory):
    mock_memory.content = "I prefer Python over JavaScript"
    result = engine.infer_preferences_with_confidence([mock_memory])
    assert isinstance(result, dict)
    # Should have at least one preference inferred
    assert len(result) >= 1


def test_infer_preferences_with_confidence_bayesian_update(engine, mock_memory):
    mock_memory.content = "I love Python"
    prior = MagicMock()
    prior.strength = 0.6
    prior.confidence = 0.7
    prior.context = "general"
    prior.value = "python"

    result = engine.infer_preferences_with_confidence(
        [mock_memory],
        prior_preferences={"technology": prior},
    )
    assert isinstance(result, dict)
    # Check that preference history was recorded
    assert len(engine._preference_history["technology"]) >= 1


def test_infer_preferences_with_confidence_evidence_boost(engine):
    """More evidence memories should boost confidence."""
    mems = []
    for i in range(5):
        mem = MagicMock()
        mem.content = "I use Python for everything"
        mem.id = f"m{i}"
        mems.append(mem)

    result = engine.infer_preferences_with_confidence(mems)
    for attr in result.values():
        assert attr.confidence > 0.0
        assert attr.value.get("evidence_count", 0) >= 1


def test_preference_history_tracking(engine, mock_memory):
    mock_memory.content = "I prefer markdown format"
    engine.infer_preferences_with_confidence([mock_memory])
    history = engine.get_preference_history("format")
    assert len(history) >= 1
    assert history[0].preference_type == PreferenceType.FORMAT


# --------------------------------------------------------------------------- #
# Temporal preference drift detection
# --------------------------------------------------------------------------- #

def test_detect_drift_insufficient_samples(engine):
    mems = [MagicMock()]
    for m in mems:
        m.created_at = datetime.now(timezone.utc)
    result = engine.detect_preference_drift("u1", mems)
    assert result["drift_detected"] is False
    assert result["reason"] == "insufficient_samples"


def test_detect_drift_insufficient_window_samples(engine):
    now = datetime.now(timezone.utc)
    mems = []
    for i in range(6):
        mem = MagicMock()
        mem.content = "Python question"
        mem.created_at = now - timedelta(hours=i)
        mem.id = f"m{i}"
        mems.append(mem)

    result = engine.detect_preference_drift("u1", mems)
    # All in recent window, no historical
    assert result["drift_detected"] is False
    assert result["reason"] == "insufficient_window_samples"


def test_detect_drift_value_shift(engine):
    now = datetime.now(timezone.utc)
    # Historical: prefers Python
    hist_mems = []
    for i in range(5):
        mem = MagicMock()
        mem.content = "I love Python programming"
        mem.created_at = now - timedelta(days=10 + i)
        mem.id = f"hist{i}"
        hist_mems.append(mem)

    # Recent: prefers Rust
    recent_mems = []
    for i in range(5):
        mem = MagicMock()
        mem.content = "I prefer Rust over everything"
        mem.created_at = now - timedelta(hours=i)
        mem.id = f"recent{i}"
        recent_mems.append(mem)

    all_mems = hist_mems + recent_mems
    result = engine.detect_preference_drift("u1", all_mems)
    assert result["drift_detected"] is True
    assert result["shift_count"] > 0
    assert len(engine.get_drift_alerts("u1")) == 1


def test_detect_drift_no_shift_same_value(engine):
    now = datetime.now(timezone.utc)
    mems = []
    for i in range(10):
        mem = MagicMock()
        mem.content = "I love Python"
        mem.created_at = now - timedelta(days=i)
        mem.id = f"m{i}"
        mems.append(mem)

    result = engine.detect_preference_drift("u1", mems)
    # Same preference across windows, no drift
    assert result["drift_detected"] is False
    assert result["shift_count"] == 0


def test_detect_drift_custom_window(engine):
    now = datetime.now(timezone.utc)
    mems = []
    for i in range(10):
        mem = MagicMock()
        mem.content = "Python"
        mem.created_at = now - timedelta(days=i)
        mem.id = f"m{i}"
        mems.append(mem)
    result = engine.detect_preference_drift("u1", mems, window_days=3.0)
    assert result["window_days"] == 3.0


# --------------------------------------------------------------------------- #
# Cross-domain preference transfer
# --------------------------------------------------------------------------- #

def test_transfer_preferences_no_targets(engine):
    """Unknown source domain should return empty."""
    result = engine.transfer_preferences("u1", "unknown_domain")
    assert result == {}


def test_transfer_preferences_with_history(engine):
    # Seed preference history
    from neuralmem.profiles.types import Preference
    pref = Preference(
        preference_type=PreferenceType.TECHNOLOGY,
        value="python",
        strength=0.8,
        context="machine_learning",
        confidence=0.75,
    )
    engine._preference_history["technology"].append(pref)

    result = engine.transfer_preferences("u1", "machine_learning")
    assert len(result) > 0
    # Should have transferred attributes for related domains
    assert any("data_science" in k for k in result.keys())
    assert any("web_development" in k for k in result.keys())


def test_transfer_preferences_explicit_targets(engine):
    from neuralmem.profiles.types import Preference
    pref = Preference(
        preference_type=PreferenceType.FORMAT,
        value="markdown",
        strength=0.7,
        context="web_development",
        confidence=0.6,
    )
    engine._preference_history["format"].append(pref)

    result = engine.transfer_preferences(
        "u1", "web_development", target_domains=["mobile", "security"]
    )
    assert any("mobile" in k for k in result.keys())
    assert any("security" in k for k in result.keys())


def test_transfer_confidence_capped(engine):
    from neuralmem.profiles.types import Preference
    pref = Preference(
        preference_type=PreferenceType.TECHNOLOGY,
        value="python",
        strength=0.9,
        context="machine_learning",
        confidence=0.95,
    )
    engine._preference_history["technology"].append(pref)

    result = engine.transfer_preferences("u1", "machine_learning")
    for attr in result.values():
        assert attr.confidence <= 0.7  # Transfer confidence is capped


# --------------------------------------------------------------------------- #
# LLM-based deep inference
# --------------------------------------------------------------------------- #

def test_llm_deep_infer_no_llm_fallback(engine, mock_memories):
    """Without LLM client, should fall back to keyword inference."""
    result = engine.llm_deep_infer(mock_memories, ProfileDimension.INTENT)
    assert "primary_intent" in result
    assert result["primary_intent"].confidence > 0


def test_llm_deep_infer_empty_raises(engine):
    with pytest.raises(ValueError, match="memories must not be empty"):
        engine.llm_deep_infer([])


def test_llm_deep_infer_with_mock_llm(mock_llm_client, mock_memories):
    engine = DeepProfileEngine(llm_client=mock_llm_client)
    result = engine.llm_deep_infer(mock_memories, ProfileDimension.INTENT)
    assert "primary_intent" in result
    assert result["primary_intent"].value["category"] == "informational"
    assert result["primary_intent"].confidence == 0.85


def test_llm_deep_infer_llm_failure_fallback(mock_memories):
    def failing_client(prompt, system=None):
        raise RuntimeError("LLM error")

    engine = DeepProfileEngine(llm_client=failing_client)
    result = engine.llm_deep_infer(mock_memories, ProfileDimension.INTENT)
    assert "primary_intent" in result  # Fallback worked


def test_parse_llm_response_intent():
    engine = DeepProfileEngine()
    response = json.dumps({
        "category": "learning",
        "subcategory": "python",
        "keywords": ["python", "tutorial"],
        "confidence": 0.9,
    })
    result = engine._parse_llm_response(response, ProfileDimension.INTENT)
    assert "primary_intent" in result
    assert result["primary_intent"].value["category"] == "learning"


def test_parse_llm_response_preference():
    engine = DeepProfileEngine()
    response = json.dumps([
        {"type": "technology", "value": "python", "strength": 0.8, "confidence": 0.75},
        {"type": "format", "value": "markdown", "strength": 0.6, "confidence": 0.7},
    ])
    result = engine._parse_llm_response(response, ProfileDimension.PREFERENCE)
    assert len(result) == 2
    assert "pref_technology" in result
    assert "pref_format" in result


def test_parse_llm_response_knowledge():
    engine = DeepProfileEngine()
    response = json.dumps([
        {"domain": "machine_learning", "level": "advanced", "confidence": 0.8},
    ])
    result = engine._parse_llm_response(response, ProfileDimension.KNOWLEDGE)
    assert "knowledge_machine_learning" in result
    assert result["knowledge_machine_learning"].value["level"] == "advanced"


def test_parse_llm_response_interaction_style():
    engine = DeepProfileEngine()
    response = json.dumps({
        "communication": "technical",
        "detail_level": 0.8,
        "initiative": 0.7,
        "follow_up": True,
        "confidence": 0.75,
    })
    result = engine._parse_llm_response(response, ProfileDimension.INTERACTION_STYLE)
    assert "interaction_style" in result
    assert result["interaction_style"].value["communication"] == "technical"


def test_parse_llm_response_markdown_block():
    engine = DeepProfileEngine()
    response = "```json\n{\"category\": \"creative\", \"confidence\": 0.8}\n```"
    result = engine._parse_llm_response(response, ProfileDimension.INTENT)
    assert "primary_intent" in result
    assert result["primary_intent"].value["category"] == "creative"


def test_parse_llm_response_invalid_json():
    engine = DeepProfileEngine()
    result = engine._parse_llm_response("not json", ProfileDimension.INTENT)
    assert result == {}


def test_parse_llm_response_unknown_category():
    engine = DeepProfileEngine()
    response = json.dumps({"category": "unknown_category", "confidence": 0.5})
    result = engine._parse_llm_response(response, ProfileDimension.INTENT)
    assert result["primary_intent"].value["category"] == "informational"  # Fallback


# --------------------------------------------------------------------------- #
# Continuous learning loop
# --------------------------------------------------------------------------- #

def test_learn_from_feedback_no_attribute(engine):
    result = engine.learn_from_feedback("u1", "nonexistent", 1.0)
    assert result is None


def test_learn_from_feedback_positive(engine):
    attr = ProfileAttribute(name="test_attr", value="v", confidence=0.5)
    engine._behavior_models["u1"] = {"test_attr": attr}

    result = engine.learn_from_feedback("u1", "test_attr", 1.0)
    assert result is not None
    assert result.confidence > 0.5  # Should increase
    assert result.confidence <= 1.0


def test_learn_from_feedback_negative(engine):
    attr = ProfileAttribute(name="test_attr", value="v", confidence=0.5)
    engine._behavior_models["u1"] = {"test_attr": attr}

    result = engine.learn_from_feedback("u1", "test_attr", -1.0)
    assert result is not None
    assert result.confidence < 0.5  # Should decrease
    assert result.confidence >= 0.0


def test_learn_from_feedback_with_memory_ids(engine):
    attr = ProfileAttribute(name="test_attr", value="v", confidence=0.6, evidence=("m1",))
    engine._behavior_models["u1"] = {"test_attr": attr}

    result = engine.learn_from_feedback("u1", "test_attr", 0.5, memory_ids=("m2", "m3"))
    assert result is not None
    assert "m1" in result.evidence
    assert "m2" in result.evidence
    assert "m3" in result.evidence


def test_learn_from_feedback_bounds(engine):
    attr = ProfileAttribute(name="test_attr", value="v", confidence=0.95)
    engine._behavior_models["u1"] = {"test_attr": attr}

    # Positive feedback should not exceed 1.0
    result = engine.learn_from_feedback("u1", "test_attr", 1.0)
    assert result.confidence == 1.0

    attr2 = ProfileAttribute(name="test_attr2", value="v", confidence=0.05)
    engine._behavior_models["u1"]["test_attr2"] = attr2

    # Negative feedback should not go below 0.0
    result2 = engine.learn_from_feedback("u1", "test_attr2", -1.0)
    assert result2.confidence == 0.0


# --------------------------------------------------------------------------- #
# Build deep profile (integration)
# --------------------------------------------------------------------------- #

def test_build_deep_profile_basic(engine, mock_memories):
    result = engine.build_deep_profile("u1", mock_memories)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert "primary_intent" in result or "interaction_style" in result


def test_build_deep_profile_with_access_logs(engine, mock_memories):
    logs = [
        AccessLogEntry("m1", "u1", "search", query_text="deep python internals", dwell_time_ms=15000),
    ]
    result = engine.build_deep_profile("u1", mock_memories, access_logs=logs)
    assert "behavior_query_depth" in result


def test_build_deep_profile_empty_raises(engine):
    with pytest.raises(ValueError, match="memories must not be empty"):
        engine.build_deep_profile("u1", [])


def test_build_deep_profile_with_llm(mock_llm_client, mock_memories):
    engine = DeepProfileEngine(llm_client=mock_llm_client)
    result = engine.build_deep_profile("u1", mock_memories, use_llm=True)
    assert isinstance(result, dict)
    # LLM intent should be present
    assert "primary_intent" in result


def test_build_deep_profile_drift_detected(engine):
    now = datetime.now(timezone.utc)
    mems = []
    for i in range(5):
        mem = MagicMock()
        mem.content = "I love Python"
        mem.created_at = now - timedelta(days=10 + i)
        mem.id = f"hist{i}"
        mems.append(mem)
    for i in range(5):
        mem = MagicMock()
        mem.content = "I prefer Rust"
        mem.created_at = now - timedelta(hours=i)
        mem.id = f"recent{i}"
        mems.append(mem)

    result = engine.build_deep_profile("u1", mems)
    assert "drift_report" in result
    assert result["drift_report"].value["drift_detected"] is True


def test_build_deep_profile_stores_behavior_model(engine, mock_memories):
    engine.build_deep_profile("u1", mock_memories)
    model = engine.get_behavior_model("u1")
    assert model["user_id"] == "u1"
    assert model["attribute_count"] > 0


# --------------------------------------------------------------------------- #
# Query / introspection helpers
# --------------------------------------------------------------------------- #

def test_get_preference_history_empty(engine):
    assert engine.get_preference_history("nonexistent") == []


def test_get_drift_alerts_empty(engine):
    assert engine.get_drift_alerts("u1") == []


def test_get_behavior_model_empty(engine):
    model = engine.get_behavior_model("unknown")
    assert model["attribute_count"] == 0
    assert model["user_id"] == "unknown"


def test_clear_user_data(engine):
    engine._behavior_models["u1"] = {"attr": ProfileAttribute(name="a", value="v")}
    engine._drift_alerts["u1"] = [{"shift": True}]
    engine.log_access(AccessLogEntry("m1", "u1", "read"))

    result = engine.clear_user_data("u1")
    assert result is True
    assert "u1" not in engine._behavior_models
    assert "u1" not in engine._drift_alerts


def test_clear_user_data_no_data(engine):
    result = engine.clear_user_data("nonexistent")
    assert result is False


# --------------------------------------------------------------------------- #
# Integration with existing types
# --------------------------------------------------------------------------- #

def test_integration_with_profile_attribute():
    attr = ProfileAttribute(
        name="deep_test",
        value={"key": "val"},
        confidence=0.8,
        source="deep_inference",
    )
    assert attr.name == "deep_test"
    assert attr.confidence == 0.8


def test_integration_preference_to_attribute():
    from neuralmem.profiles.types import Preference
    pref = Preference(
        preference_type=PreferenceType.TECHNOLOGY,
        value="python",
        strength=0.8,
        confidence=0.75,
    )
    attr = pref.to_attribute()
    assert attr.name == "pref_technology"
    assert attr.value["value"] == "python"


def test_deep_engine_uses_parent_infer_intent(engine, mock_memory):
    """DeepProfileEngine should still work with parent's infer_intent."""
    result = engine.infer_intent([mock_memory])
    assert "primary_intent" in result


def test_deep_engine_uses_parent_infer_preferences(engine, mock_memory):
    """DeepProfileEngine should still work with parent's infer_preferences."""
    result = engine.infer_preferences([mock_memory])
    assert isinstance(result, dict)


def test_deep_engine_uses_parent_build_profile(engine, mock_memories):
    """DeepProfileEngine should still work with parent's build_profile."""
    result = engine.build_profile("u1", mock_memories)
    assert isinstance(result, dict)
    assert len(result) > 0


# --------------------------------------------------------------------------- #
# AccessLogEntry
# --------------------------------------------------------------------------- #

def test_access_log_entry_defaults():
    entry = AccessLogEntry("m1", "u1", "read")
    assert entry.memory_id == "m1"
    assert entry.user_id == "u1"
    assert entry.action == "read"
    assert entry.query_text == ""
    assert entry.result_count == 0
    assert entry.dwell_time_ms == 0
    assert isinstance(entry.timestamp, datetime)


def test_access_log_entry_to_dict():
    ts = datetime.now(timezone.utc)
    entry = AccessLogEntry(
        memory_id="m1",
        user_id="u1",
        action="search",
        timestamp=ts,
        query_text="python",
        result_count=10,
        dwell_time_ms=5000,
    )
    d = entry.to_dict()
    assert d["memory_id"] == "m1"
    assert d["action"] == "search"
    assert d["query_text"] == "python"
    assert d["result_count"] == 10
    assert d["dwell_time_ms"] == 5000
    assert d["timestamp"] == ts.isoformat()
