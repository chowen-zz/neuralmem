"""LLM-powered multi-turn conversation memory extractor.

Unlike the rule-based ConversationExtractor, this uses an LLM to
understand conversation context and extract nuanced memories.
"""
from __future__ import annotations

import json
import logging

_logger = logging.getLogger(__name__)

CONVERSATION_EXTRACTION_PROMPT = (
    "Given the following conversation, extract all user-related memories.\n"
    "For each memory, provide:\n"
    "- content: the standalone fact/preference/event\n"
    "- type: fact | preference | episodic | procedural\n"
    "- role: which role said it (user/assistant)\n"
    "- confidence: 0.0-1.0\n"
    "- entities: list of mentioned entities\n\n"
    "Conversation:\n{conversation}\n\n"
    "Output JSON array:"
)


class LLMConversationExtractor:
    """LLM-powered multi-turn conversation memory extractor.

    Supports multiple LLM backends via llm_backend parameter.
    The _call_llm method is intended to be overridden in subclasses
    or monkey-patched in tests.
    """

    def __init__(
        self,
        llm_backend: str = "ollama",
        **kwargs: object,
    ) -> None:
        self._backend = llm_backend
        self._kwargs = kwargs

    async def extract(
        self,
        messages: list[dict[str, str]],
        instructions: str | None = None,
    ) -> list[dict]:
        """Extract memories from a multi-turn conversation.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
            instructions: Optional instruction string to prepend to prompt.

        Returns:
            List of dicts with keys: content, type, role, confidence, entities.
        """
        if not messages:
            return []

        conversation = self._format_conversation(messages)
        prompt = CONVERSATION_EXTRACTION_PROMPT.format(
            conversation=conversation
        )
        if instructions:
            prompt = instructions + "\n\n" + prompt

        raw = await self._call_llm(prompt)
        return await self._parse_json_response(raw)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM backend and return raw response text.

        Subclasses should override this with actual API calls.
        """
        raise NotImplementedError(
            "Subclass must implement _call_llm for backend: "
            f"{self._backend}"
        )

    async def _parse_json_response(self, text: str) -> list:
        """Parse JSON array from LLM response, handling markdown wrappers."""
        cleaned = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "memories" in data:
                return data["memories"]
            return []
        except (json.JSONDecodeError, TypeError):
            _logger.warning(
                "Failed to parse LLM conversation response as JSON"
            )
            return []

    def _format_conversation(
        self, messages: list[dict]
    ) -> str:
        """Format messages as 'role: content' lines."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)
