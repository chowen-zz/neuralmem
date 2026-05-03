"""Shared base for all LLM-backed extractors — subclasses only implement _call_llm()."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Entity, MemoryType
from neuralmem.extraction.extractor import ExtractedItem, MemoryExtractor

_logger = logging.getLogger(__name__)

_DEFAULT_EXTRACT_PROMPT = (
    "Analyze the following text and extract key facts and entities.\n"
    "For each fact, determine the event type:\n"
    "- ADD: new information not likely already known\n"
    "- UPDATE: information that modifies or supersedes a previous fact\n"
    "- DELETE: information that invalidates or contradicts a previous fact\n"
    "- NONE: trivial or redundant information (skip these)\n\n"
    'Return JSON: {"facts": [{"content": "fact text", "event": "ADD|UPDATE|DELETE|NONE", '
    '"importance": 0.0-1.0}], '
    '"entities": [{"name": "X", "type": "person|project|technology|concept"}]}\n\n'
    "Text: "
)

_DEFAULT_EVENT_PROMPT = (
    "Given the user's existing memories and a new message, decide what to do with each fact.\n"
    "For each extracted fact, return one of:\n"
    "  ADD    — this is genuinely new information\n"
    "  UPDATE — this supersedes or modifies an existing memory\n"
    "  DELETE — this invalidates a previous memory\n"
    "  NONE   — this is trivial or already known\n\n"
    "Existing memories: {existing}\n\n"
    "New message: {message}\n\n"
    'Return JSON: {"facts": [{"content": "...", "event": "ADD|UPDATE|DELETE|NONE", '
    '"old_memory": "...", "importance": 0.0-1.0}], '
    '"entities": [{"name": "...", "type": "..."}]}\n'
)


class BaseLLMExtractor(ABC):
    def __init__(self, config: NeuralMemConfig) -> None:
        self._config = config
        self._rule_extractor = MemoryExtractor(config)

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return raw response string."""
        ...

    def _get_prompt(self, content: str) -> str:
        """Build the extraction prompt, respecting custom prompt config."""
        if self._config.custom_extraction_prompt:
            return self._config.custom_extraction_prompt + "\n\nText: " + content[:2000] + "\nJSON:"
        return _DEFAULT_EXTRACT_PROMPT + content[:2000] + "\nJSON:"

    def extract(self, content: str, **kwargs: object) -> list[ExtractedItem]:
        try:
            prompt = self._get_prompt(content)
            raw = self._call_llm(prompt)
            raw = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            data = json.loads(raw)
            items = self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]

            # Extract LLM-detected entities
            extra_entities = [
                Entity(name=str(e.get("name", ""))[:200], entity_type=e.get("type", "concept"))
                for e in data.get("entities", [])
                if e.get("name")
            ]

            # Process LLM facts with event types
            llm_facts = data.get("facts", [])
            if llm_facts and isinstance(llm_facts[0], dict):
                # New format: facts with event + importance
                enhanced_items: list[ExtractedItem] = []
                for fact in llm_facts:
                    event = fact.get("event", "ADD").upper()
                    if event == "NONE":
                        continue
                    importance = float(fact.get("importance", 0.5))
                    importance = max(0.0, min(1.0, importance))
                    enhanced_items.append(ExtractedItem(
                        content=str(fact.get("content", ""))[:2000],
                        memory_type=items[0].memory_type if items else None,  # type: ignore[arg-type]
                        entities=extra_entities if not items else [],
                        relations=[],
                        tags=[],
                        importance=importance,
                        event=event,
                    ))
                if enhanced_items:
                    return enhanced_items

            # Merge entities into existing items (backward compatible)
            if items and extra_entities:
                first = items[0]
                merged = list({e.name: e for e in first.entities + extra_entities}.values())
                items[0] = ExtractedItem(
                    content=first.content,
                    memory_type=first.memory_type,
                    entities=merged,
                    relations=first.relations,
                    tags=first.tags,
                    importance=first.importance,
                    event=first.event,
                )
            return items
        except Exception as exc:
            _logger.warning("LLM extraction failed (%s), falling back to rules.", exc)
            return self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]

    def extract_with_context(
        self,
        content: str,
        existing_memories: list[str],
        **kwargs: object,
    ) -> list[ExtractedItem]:
        """Extract facts with awareness of existing memories (for UPDATE/DELETE decisions)."""
        try:
            existing_str = (
                "\n".join(f"- {m}" for m in existing_memories[:20])
                if existing_memories
                else "(none)"
            )
            prompt = _DEFAULT_EVENT_PROMPT.format(
                existing=existing_str,
                message=content[:2000],
            )
            if self._config.custom_extraction_prompt:
                prompt = self._config.custom_extraction_prompt + "\n\n" + prompt
            raw = self._call_llm(prompt)
            raw = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            data = json.loads(raw)
            items: list[ExtractedItem] = []
            for fact in data.get("facts", []):
                event = fact.get("event", "ADD").upper()
                if event == "NONE":
                    continue
                importance = float(fact.get("importance", 0.5))
                importance = max(0.0, min(1.0, importance))
                entities = [
                    Entity(name=str(e.get("name", ""))[:200], entity_type=e.get("type", "concept"))
                    for e in data.get("entities", [])
                    if e.get("name")
                ]
                items.append(ExtractedItem(
                    content=str(fact.get("content", ""))[:2000],
                    memory_type=MemoryType.SEMANTIC,
                    entities=entities,
                    relations=[],
                    tags=[],
                    importance=importance,
                    event=event,
                ))
            return items if items else self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            _logger.warning("Context-aware extraction failed (%s), falling back.", exc)
            return self.extract(content, **kwargs)
