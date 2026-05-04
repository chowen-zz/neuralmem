"""SuggestionEngine — autocomplete, rewrite, expand, summarize suggestions."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from neuralmem.core.types import Memory

_logger = logging.getLogger(__name__)


class SuggestionType(str, Enum):
    """Types of writing suggestions."""

    AUTOCOMPLETE = "autocomplete"
    REWRITE = "rewrite"
    EXPAND = "expand"
    SUMMARIZE = "summarize"
    STYLE_TRANSFER = "style_transfer"


@dataclass
class Suggestion:
    """A single writing suggestion."""

    suggestion_type: SuggestionType
    text: str
    confidence: float = 0.5
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMCaller(Protocol):
    """Protocol: anything that takes a prompt and returns text."""

    def __call__(self, prompt: str) -> str: ...


class SuggestionEngine:
    """AI-powered suggestion engine for writing assistance.

    Provides autocomplete, rewrite, expand, summarize, and style-transfer
    suggestions.  All LLM interactions go through the ``llm_caller`` protocol
    so tests can use deterministic mocks.
    """

    def __init__(
        self,
        llm_caller: LLMCaller | None = None,
        max_suggestions: int = 3,
        min_confidence: float = 0.3,
    ) -> None:
        self._llm = llm_caller
        self._max_suggestions = max_suggestions
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def autocomplete(
        self,
        partial_text: str,
        context: str = "",
        max_tokens: int = 50,
    ) -> list[Suggestion]:
        """Suggest completions for partial text."""
        prompt = self._build_autocomplete_prompt(partial_text, context, max_tokens)
        raw = self._call_llm(prompt)
        suggestions = self._parse_suggestions(raw, SuggestionType.AUTOCOMPLETE)
        return self._filter_and_rank(suggestions)

    def rewrite_suggestions(
        self,
        text: str,
        context: str = "",
        count: int | None = None,
    ) -> list[Suggestion]:
        """Suggest alternative phrasings / rewrites."""
        prompt = self._build_rewrite_suggestions_prompt(text, context, count or self._max_suggestions)
        raw = self._call_llm(prompt)
        suggestions = self._parse_suggestions(raw, SuggestionType.REWRITE)
        return self._filter_and_rank(suggestions)

    def expand_suggestions(
        self,
        text: str,
        context: str = "",
        count: int | None = None,
    ) -> list[Suggestion]:
        """Suggest ways to expand the text."""
        prompt = self._build_expand_suggestions_prompt(text, context, count or self._max_suggestions)
        raw = self._call_llm(prompt)
        suggestions = self._parse_suggestions(raw, SuggestionType.EXPAND)
        return self._filter_and_rank(suggestions)

    def summarize_suggestions(
        self,
        text: str,
        context: str = "",
        count: int | None = None,
    ) -> list[Suggestion]:
        """Suggest summary versions of the text."""
        prompt = self._build_summarize_suggestions_prompt(text, context, count or self._max_suggestions)
        raw = self._call_llm(prompt)
        suggestions = self._parse_suggestions(raw, SuggestionType.SUMMARIZE)
        return self._filter_and_rank(suggestions)

    def style_transfer(
        self,
        text: str,
        target_style: str,
        context: str = "",
    ) -> list[Suggestion]:
        """Suggest the text rewritten in a target style."""
        prompt = self._build_style_transfer_prompt(text, target_style, context)
        raw = self._call_llm(prompt)
        suggestions = self._parse_suggestions(raw, SuggestionType.STYLE_TRANSFER)
        return self._filter_and_rank(suggestions)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_autocomplete_prompt(
        partial_text: str, context: str, max_tokens: int
    ) -> str:
        parts = [
            "Complete the following text naturally.",
            f"Continue for approximately {max_tokens} tokens.",
        ]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Partial text:\n{partial_text}")
        parts.append("Completion:")
        return "\n\n".join(parts)

    @staticmethod
    def _build_rewrite_suggestions_prompt(
        text: str, context: str, count: int
    ) -> str:
        parts = [
            f"Provide {count} alternative ways to rewrite the following text.",
            "Return each suggestion on a new line prefixed with a number.",
        ]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Text:\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_expand_suggestions_prompt(
        text: str, context: str, count: int
    ) -> str:
        parts = [
            f"Provide {count} ways to expand the following text with more detail.",
            "Return each suggestion on a new line prefixed with a number.",
        ]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Text:\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_summarize_suggestions_prompt(
        text: str, context: str, count: int
    ) -> str:
        parts = [
            f"Provide {count} concise summaries of the following text.",
            "Return each summary on a new line prefixed with a number.",
        ]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Text:\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_style_transfer_prompt(
        text: str, target_style: str, context: str
    ) -> str:
        parts = [
            f"Rewrite the following text in the style of: {target_style}.",
            "Return the rewritten text only.",
        ]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Text:\n{text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM dispatch
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        if self._llm is None:
            raise RuntimeError("No LLM caller configured for SuggestionEngine")
        return self._llm(prompt).strip()

    # ------------------------------------------------------------------
    # Parsing & ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_suggestions(raw: str, suggestion_type: SuggestionType) -> list[Suggestion]:
        """Parse numbered or line-separated suggestions from LLM output."""
        suggestions: list[Suggestion] = []
        lines = raw.splitlines()
        current_text = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Detect numbered items: "1. ...", "1) ...", "- ..."
            if stripped[0].isdigit() and len(stripped) > 2 and stripped[1] in ".)":
                if current_text:
                    suggestions.append(
                        Suggestion(
                            suggestion_type=suggestion_type,
                            text=current_text.strip(),
                            confidence=0.7,
                        )
                    )
                current_text = stripped[2:].strip()
            elif stripped.startswith("-") or stripped.startswith("*"):
                if current_text:
                    suggestions.append(
                        Suggestion(
                            suggestion_type=suggestion_type,
                            text=current_text.strip(),
                            confidence=0.7,
                        )
                    )
                current_text = stripped[1:].strip()
            else:
                current_text += " " + stripped
        if current_text:
            suggestions.append(
                Suggestion(
                    suggestion_type=suggestion_type,
                    text=current_text.strip(),
                    confidence=0.7,
                )
            )
        # If no structured parsing worked, treat whole response as one suggestion
        if not suggestions and raw.strip():
            suggestions.append(
                Suggestion(
                    suggestion_type=suggestion_type,
                    text=raw.strip(),
                    confidence=0.6,
                )
            )
        return suggestions

    def _filter_and_rank(self, suggestions: list[Suggestion]) -> list[Suggestion]:
        """Filter by confidence and limit count."""
        filtered = [s for s in suggestions if s.confidence >= self._min_confidence]
        # Deduplicate by text content
        seen: set[str] = set()
        deduped: list[Suggestion] = []
        for s in filtered:
            key = s.text.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        return deduped[: self._max_suggestions]
