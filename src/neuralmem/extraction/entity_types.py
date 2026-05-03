"""Entity and relation extraction types and LLM-based extractor."""
from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


class EntityType(enum.Enum):
    """Types of entities that can be extracted."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    PREFERENCE = "preference"
    CONCEPT = "concept"
    OBJECT = "object"
    TIME = "time"


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""

    name: str
    entity_type: EntityType
    confidence: float = 0.8
    context: str = ""


@dataclass
class ExtractedRelation:
    """A subject-predicate-object relation extracted from text."""

    subject: str
    predicate: str
    object_: str
    confidence: float = 0.8


_ENTITY_TYPE_NAMES = ", ".join(t.value for t in EntityType)

ENTITY_EXTRACTION_PROMPT = (
    "Extract entities and relations from the following text.\n"
    "Entity types: {_entity_types}\n"
    "Relations: subject-predicate-object triples.\n\n"
    "Text: {text}\n\n"
    'Output JSON: {{"entities": [{{"name": "...", "type": "...", '
    '"confidence": 0.0-1.0, "context": "..."}}], '
    '"relations": [{{"subject": "...", "predicate": "...", '
    '"object": "...", "confidence": 0.0-1.0}}]}}'
)


class EntityExtractor:
    """LLM-based entity and relation extractor."""

    def __init__(
        self,
        llm_backend: str = "ollama",
        **kwargs: object,
    ) -> None:
        self._backend = llm_backend
        self._kwargs = kwargs

    async def extract_entities(
        self, text: str
    ) -> list[ExtractedEntity]:
        """Extract entities from text."""
        result = await self.extract_all(text)
        return result.get("entities", [])

    async def extract_relations(
        self, text: str
    ) -> list[ExtractedRelation]:
        """Extract relations from text."""
        result = await self.extract_all(text)
        return result.get("relations", [])

    async def extract_all(self, text: str) -> dict:
        """Extract both entities and relations from text.

        Returns dict with 'entities' and 'relations' lists.
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            _entity_types=_ENTITY_TYPE_NAMES,
            text=text[:2000],
        )
        raw = await self._call_llm(prompt)
        return await self._parse_response(raw)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM backend. Subclasses should override."""
        raise NotImplementedError(
            "Subclass must implement _call_llm for backend: "
            f"{self._backend}"
        )

    async def _parse_response(self, text: str) -> dict:
        """Parse JSON response into entities and relations."""
        cleaned = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            _logger.warning(
                "Failed to parse entity extraction response as JSON"
            )
            return {"entities": [], "relations": []}

        entities = []
        for e in data.get("entities", []):
            try:
                etype = EntityType(e.get("type", "concept"))
            except ValueError:
                etype = EntityType.CONCEPT
            entities.append(
                ExtractedEntity(
                    name=str(e.get("name", "")),
                    entity_type=etype,
                    confidence=_clamp_confidence(
                        float(e.get("confidence", 0.8))
                    ),
                    context=str(e.get("context", "")),
                )
            )

        relations = []
        for r in data.get("relations", []):
            relations.append(
                ExtractedRelation(
                    subject=str(r.get("subject", "")),
                    predicate=str(r.get("predicate", "")),
                    object_=str(r.get("object", "")),
                    confidence=_clamp_confidence(
                        float(r.get("confidence", 0.8))
                    ),
                )
            )

        return {"entities": entities, "relations": relations}


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))
