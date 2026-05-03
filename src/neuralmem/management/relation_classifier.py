"""Relation classifier for memory relationships."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from neuralmem.management.prompts import RELATION_PROMPT

_logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """Types of relationships between memories."""

    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


@dataclass
class ClassifiedRelation:
    """A classified relationship for a memory."""

    memory_content: str = ""
    relation_type: RelationType = RelationType.SEMANTIC
    related_entity: str = ""
    confidence: float = 0.5


class RelationClassifier:
    """Classifies relationships between memories using LLM.

    Identifies whether memories are related by semantics, space,
    time, or causation.
    """

    def __init__(
        self,
        llm_backend: str = "ollama",
        model: str | None = None,
        **kwargs: object,
    ) -> None:
        self._backend = llm_backend
        self._model = model
        self._kwargs = kwargs
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize LLM client based on backend."""
        if self._backend == "ollama":
            self._model = self._model or "llama3.2:3b"
            self._base_url = self._kwargs.get(
                "base_url", "http://localhost:11434"
            )
        elif self._backend == "openai":
            try:
                import openai

                api_key = self._kwargs.get("api_key")
                self._client = openai.OpenAI(api_key=api_key)
                self._model = self._model or "gpt-4o-mini"
            except ImportError as exc:
                raise ImportError(
                    "Install openai: pip install 'neuralmem[openai]'"
                ) from exc
        elif self._backend == "anthropic":
            try:
                import anthropic

                api_key = self._kwargs.get("api_key")
                self._client = anthropic.Anthropic(api_key=api_key)
                self._model = self._model or "claude-haiku-4-5-20251001"
            except ImportError as exc:
                raise ImportError(
                    "Install anthropic: pip install 'neuralmem[anthropic]'"
                ) from exc
        else:
            raise ValueError(f"Unsupported LLM backend: {self._backend}")

    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM backend."""
        if self._backend == "ollama":
            import httpx

            url = f"{self._base_url}/api/generate"
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json().get("response", "{}")
        elif self._backend == "openai":
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content or "{}"
        elif self._backend == "anthropic":
            message = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            if message.content:
                return message.content[0].text
            return "{}"
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    async def _parse_json_response(
        self, text: str
    ) -> list[Any] | dict[str, Any]:
        """Parse JSON from LLM response, handling markdown wrappers."""
        cleaned = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start = cleaned.find(start_char)
                end = cleaned.rfind(end_char)
                if start != -1 and end > start:
                    try:
                        return json.loads(cleaned[start: end + 1])
                    except json.JSONDecodeError:
                        continue
            _logger.warning(
                "Failed to parse JSON from LLM response: %s",
                text[:200],
            )
            return []

    async def classify(
        self,
        memories: list[str],
    ) -> list[ClassifiedRelation]:
        """Classify relationships for a list of memories.

        Args:
            memories: List of memory content strings.

        Returns:
            List of ClassifiedRelation objects.
        """
        if not memories:
            return []

        memories_str = json.dumps(memories, indent=2)
        prompt = RELATION_PROMPT.format(memories=memories_str)

        raw = await self._call_llm(prompt)
        result = await self._parse_json_response(raw)

        if not isinstance(result, list):
            return []

        relations: list[ClassifiedRelation] = []
        for item in result:
            if not isinstance(item, dict):
                continue

            rel_type_str = item.get("relation_type", "semantic").lower()
            try:
                rel_type = RelationType(rel_type_str)
            except ValueError:
                rel_type = RelationType.SEMANTIC

            confidence = float(item.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            relations.append(
                ClassifiedRelation(
                    memory_content=item.get("memory_content", ""),
                    relation_type=rel_type,
                    related_entity=item.get("related_entity", ""),
                    confidence=confidence,
                )
            )

        return relations
