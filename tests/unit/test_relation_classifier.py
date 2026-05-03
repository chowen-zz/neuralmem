"""Tests for RelationClassifier — 10+ tests."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from neuralmem.management.relation_classifier import (
    ClassifiedRelation,
    RelationClassifier,
    RelationType,
)

# ==================== RelationType Enum ====================


class TestRelationType:

    def test_relation_type_enum(self):
        assert RelationType.SEMANTIC.value == "semantic"
        assert RelationType.SPATIAL.value == "spatial"
        assert RelationType.TEMPORAL.value == "temporal"
        assert RelationType.CAUSAL.value == "causal"

    def test_relation_type_all_members(self):
        members = list(RelationType)
        assert len(members) == 4

    def test_relation_type_is_string(self):
        assert isinstance(RelationType.SEMANTIC, str)


# ==================== ClassifiedRelation Dataclass ====================


class TestClassifiedRelation:

    def test_classified_relation_dataclass(self):
        cr = ClassifiedRelation(
            memory_content="User works at Google",
            relation_type=RelationType.SEMANTIC,
            related_entity="Google",
            confidence=0.9,
        )
        assert cr.memory_content == "User works at Google"
        assert cr.relation_type is RelationType.SEMANTIC
        assert cr.related_entity == "Google"
        assert cr.confidence == 0.9

    def test_classified_relation_defaults(self):
        cr = ClassifiedRelation()
        assert cr.memory_content == ""
        assert cr.relation_type is RelationType.SEMANTIC
        assert cr.related_entity == ""
        assert cr.confidence == 0.5


# ==================== Init ====================


class TestClassifierInit:

    def test_classifier_init(self):
        with patch.object(
            RelationClassifier, "_init_client", return_value=None
        ):
            cls = RelationClassifier(llm_backend="ollama")
            assert cls._backend == "ollama"

    def test_classifier_init_unsupported_backend(self):
        with pytest.raises(ValueError, match="Unsupported"):
            RelationClassifier(llm_backend="nonexistent")


# ==================== Classification ====================


class TestClassify:

    @pytest.mark.asyncio
    async def test_classify_semantic_relation(self):
        response = json.dumps([{
            "memory_content": "User likes Python",
            "relation_type": "semantic",
            "related_entity": "Python",
            "confidence": 0.9,
        }])
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value=response)
        result = await cls.classify(["User likes Python"])
        assert len(result) == 1
        assert result[0].relation_type is RelationType.SEMANTIC
        assert result[0].related_entity == "Python"

    @pytest.mark.asyncio
    async def test_classify_temporal_relation(self):
        response = json.dumps([{
            "memory_content": "Met yesterday",
            "relation_type": "temporal",
            "related_entity": "meeting",
            "confidence": 0.8,
        }])
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value=response)
        result = await cls.classify(["Met yesterday"])
        assert result[0].relation_type is RelationType.TEMPORAL

    @pytest.mark.asyncio
    async def test_classify_causal_relation(self):
        response = json.dumps([{
            "memory_content": "Deploy broke prod",
            "relation_type": "causal",
            "related_entity": "deployment",
            "confidence": 0.85,
        }])
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value=response)
        result = await cls.classify(["Deploy broke prod"])
        assert result[0].relation_type is RelationType.CAUSAL

    @pytest.mark.asyncio
    async def test_classify_spatial_relation(self):
        response = json.dumps([{
            "memory_content": "Office in NYC",
            "relation_type": "spatial",
            "related_entity": "NYC",
            "confidence": 0.75,
        }])
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value=response)
        result = await cls.classify(["Office in NYC"])
        assert result[0].relation_type is RelationType.SPATIAL

    @pytest.mark.asyncio
    async def test_classify_batch(self):
        response = json.dumps([
            {
                "memory_content": "Fact A",
                "relation_type": "semantic",
                "related_entity": "A",
                "confidence": 0.7,
            },
            {
                "memory_content": "Fact B",
                "relation_type": "causal",
                "related_entity": "B",
                "confidence": 0.8,
            },
        ])
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value=response)
        result = await cls.classify(["Fact A", "Fact B"])
        assert len(result) == 2
        assert result[0].relation_type is RelationType.SEMANTIC
        assert result[1].relation_type is RelationType.CAUSAL

    @pytest.mark.asyncio
    async def test_classify_empty_input(self):
        cls = RelationClassifier.__new__(RelationClassifier)
        result = await cls.classify([])
        assert result == []


# ==================== LLM Call Mock ====================


class TestLLMCallMock:

    @pytest.mark.asyncio
    async def test_llm_call_mock(self):
        """Verify _call_llm is invoked during classify."""
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value="[]")
        await cls.classify(["test memory"])
        cls._call_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self):
        """Invalid LLM response returns empty list."""
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value="not json")
        result = await cls.classify(["test"])
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_relation_type_defaults_semantic(self):
        """Unknown relation_type defaults to SEMANTIC."""
        response = json.dumps([{
            "memory_content": "test",
            "relation_type": "unknown_type",
            "related_entity": "entity",
            "confidence": 0.5,
        }])
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value=response)
        result = await cls.classify(["test"])
        assert result[0].relation_type is RelationType.SEMANTIC

    @pytest.mark.asyncio
    async def test_confidence_clamped(self):
        """Confidence is clamped to [0, 1]."""
        response = json.dumps([{
            "memory_content": "test",
            "relation_type": "semantic",
            "related_entity": "e",
            "confidence": 2.0,
        }])
        cls = RelationClassifier.__new__(RelationClassifier)
        cls._call_llm = AsyncMock(return_value=response)
        result = await cls.classify(["test"])
        assert result[0].confidence <= 1.0
