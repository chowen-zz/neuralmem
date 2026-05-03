"""EntityExtractor and entity types unit tests — 14 tests."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from neuralmem.extraction.entity_types import (
    EntityExtractor,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    _clamp_confidence,
)


def _run(coro):
    """Helper to run async in sync tests."""
    return asyncio.run(coro)


SAMPLE_RESPONSE = json.dumps(
    {
        "entities": [
            {
                "name": "Alice",
                "type": "person",
                "confidence": 0.95,
                "context": "Alice is a developer",
            },
            {
                "name": "Acme Corp",
                "type": "organization",
                "confidence": 0.9,
                "context": "works at Acme Corp",
            },
        ],
        "relations": [
            {
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme Corp",
                "confidence": 0.85,
            }
        ],
    }
)


# ==================== EntityType enum ====================


class TestEntityTypeEnum:

    def test_entity_type_enum_values(self):
        assert EntityType.PERSON.value == "person"
        assert EntityType.ORGANIZATION.value == "organization"
        assert EntityType.LOCATION.value == "location"
        assert EntityType.EVENT.value == "event"
        assert EntityType.PREFERENCE.value == "preference"
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.OBJECT.value == "object"
        assert EntityType.TIME.value == "time"
        assert len(EntityType) == 8


# ==================== Dataclasses ====================


class TestDataclasses:

    def test_extracted_entity_dataclass(self):
        entity = ExtractedEntity(
            name="Alice",
            entity_type=EntityType.PERSON,
            confidence=0.9,
            context="a developer",
        )
        assert entity.name == "Alice"
        assert entity.entity_type is EntityType.PERSON
        assert entity.confidence == 0.9
        assert entity.context == "a developer"

    def test_extracted_relation_dataclass(self):
        rel = ExtractedRelation(
            subject="Alice",
            predicate="works_at",
            object_="Acme Corp",
            confidence=0.85,
        )
        assert rel.subject == "Alice"
        assert rel.predicate == "works_at"
        assert rel.object_ == "Acme Corp"
        assert rel.confidence == 0.85


# ==================== EntityExtractor ====================


class TestEntityExtractor:

    def test_entity_extractor_init(self):
        ext = EntityExtractor(llm_backend="ollama")
        assert ext._backend == "ollama"

    def test_extract_entities_returns_list(self):
        ext = EntityExtractor(llm_backend="ollama")
        ext._call_llm = AsyncMock(return_value=SAMPLE_RESPONSE)
        result = _run(ext.extract_entities("Alice works at Acme Corp"))
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].name == "Alice"
        assert result[0].entity_type is EntityType.PERSON

    def test_extract_entities_empty_text(self):
        ext = EntityExtractor(llm_backend="ollama")
        result = _run(ext.extract_entities(""))
        assert result == []

    def test_extract_relations_returns_list(self):
        ext = EntityExtractor(llm_backend="ollama")
        ext._call_llm = AsyncMock(return_value=SAMPLE_RESPONSE)
        result = _run(ext.extract_relations("Alice works at Acme Corp"))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].subject == "Alice"
        assert result[0].object_ == "Acme Corp"

    def test_extract_all_returns_dict(self):
        ext = EntityExtractor(llm_backend="ollama")
        ext._call_llm = AsyncMock(return_value=SAMPLE_RESPONSE)
        result = _run(ext.extract_all("Alice works at Acme Corp"))
        assert isinstance(result, dict)

    def test_extract_all_has_entities_and_relations(self):
        ext = EntityExtractor(llm_backend="ollama")
        ext._call_llm = AsyncMock(return_value=SAMPLE_RESPONSE)
        result = _run(ext.extract_all("Alice works at Acme Corp"))
        assert "entities" in result
        assert "relations" in result
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1

    def test_call_llm_not_implemented(self):
        ext = EntityExtractor(llm_backend="ollama")
        with pytest.raises(NotImplementedError):
            _run(ext._call_llm("test"))

    def test_parse_json_response_valid(self):
        ext = EntityExtractor(llm_backend="ollama")
        result = _run(ext._parse_response(SAMPLE_RESPONSE))
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1

    def test_parse_json_response_invalid(self):
        ext = EntityExtractor(llm_backend="ollama")
        result = _run(ext._parse_response("not json"))
        assert result == {"entities": [], "relations": []}

    def test_parse_json_response_markdown(self):
        ext = EntityExtractor(llm_backend="ollama")
        wrapped = "```json\n" + SAMPLE_RESPONSE + "\n```"
        result = _run(ext._parse_response(wrapped))
        assert len(result["entities"]) == 2

    def test_entity_confidence_range(self):
        ext = EntityExtractor(llm_backend="ollama")
        ext._call_llm = AsyncMock(return_value=SAMPLE_RESPONSE)
        result = _run(ext.extract_entities("test"))
        for entity in result:
            assert 0.0 <= entity.confidence <= 1.0

    def test_clamp_confidence(self):
        assert _clamp_confidence(0.5) == 0.5
        assert _clamp_confidence(-0.1) == 0.0
        assert _clamp_confidence(1.5) == 1.0

    def test_unknown_entity_type_defaults_to_concept(self):
        ext = EntityExtractor(llm_backend="ollama")
        response = json.dumps(
            {
                "entities": [
                    {
                        "name": "Thing",
                        "type": "unknown_garbage",
                        "confidence": 0.5,
                    }
                ],
                "relations": [],
            }
        )
        ext._call_llm = AsyncMock(return_value=response)
        result = _run(ext.extract_entities("test"))
        assert result[0].entity_type is EntityType.CONCEPT
