"""AI Writing Assistant — core writing operations with memory context."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from neuralmem.core.types import Memory, SearchResult

from .context import ContextInjector
from .suggestions import SuggestionEngine
from .templates import TemplateManager

_logger = logging.getLogger(__name__)


class LLMCaller(Protocol):
    """Protocol: anything that takes a prompt and returns text."""

    def __call__(self, prompt: str) -> str: ...


@dataclass
class WriteResult:
    """Result of a writing operation."""

    text: str
    operation: str  # write | rewrite | expand | summarize
    context_memories_used: list[Memory] = field(default_factory=list)
    template_used: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WritingAssistant:
    """AI-powered writing helper backed by NeuralMem memory context.

    Similar to Supermemory's Nova AI assistant, but fully mock-testable
    via the ``llm_caller`` protocol.
    """

    def __init__(
        self,
        context_injector: ContextInjector | None = None,
        suggestion_engine: SuggestionEngine | None = None,
        template_manager: TemplateManager | None = None,
        llm_caller: LLMCaller | None = None,
        max_context_memories: int = 5,
    ) -> None:
        self._context = context_injector
        self._suggestions = suggestion_engine
        self._templates = template_manager or TemplateManager()
        self._llm = llm_caller
        self._max_context = max_context_memories

    # ------------------------------------------------------------------
    # Core writing operations
    # ------------------------------------------------------------------

    def write(
        self,
        prompt: str,
        user_id: str | None = None,
        template_name: str | None = None,
        style: str | None = None,
        **kwargs: Any,
    ) -> WriteResult:
        """Generate fresh text from a prompt, optionally using a template."""
        enriched = self._enrich_prompt(prompt, user_id, style)
        if template_name:
            enriched = self._templates.apply(template_name, enriched, **kwargs)

        text = self._call_llm(enriched)
        return WriteResult(
            text=text,
            operation="write",
            context_memories_used=self._last_context_memories,
            template_used=template_name,
        )

    def rewrite(
        self,
        text: str,
        instruction: str = "improve clarity and conciseness",
        user_id: str | None = None,
        style: str | None = None,
        **kwargs: Any,
    ) -> WriteResult:
        """Rewrite existing text following an instruction."""
        context = self._fetch_context(user_id, text)
        prompt = self._build_rewrite_prompt(text, instruction, context, style)
        result_text = self._call_llm(prompt)
        return WriteResult(
            text=result_text,
            operation="rewrite",
            context_memories_used=self._last_context_memories,
        )

    def expand(
        self,
        text: str,
        target_length: int | None = None,
        user_id: str | None = None,
        style: str | None = None,
        **kwargs: Any,
    ) -> WriteResult:
        """Expand text with more detail, examples, or depth."""
        context = self._fetch_context(user_id, text)
        prompt = self._build_expand_prompt(text, target_length, context, style)
        result_text = self._call_llm(prompt)
        return WriteResult(
            text=result_text,
            operation="expand",
            context_memories_used=self._last_context_memories,
        )

    def summarize(
        self,
        text: str,
        max_words: int | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> WriteResult:
        """Summarize text to a shorter form."""
        context = self._fetch_context(user_id, text)
        prompt = self._build_summarize_prompt(text, max_words, context)
        result_text = self._call_llm(prompt)
        return WriteResult(
            text=result_text,
            operation="summarize",
            context_memories_used=self._last_context_memories,
        )

    # ------------------------------------------------------------------
    # Prompt enrichment & context
    # ------------------------------------------------------------------

    def _enrich_prompt(
        self, prompt: str, user_id: str | None, style: str | None
    ) -> str:
        parts: list[str] = []
        context = self._fetch_context(user_id, prompt)
        if context:
            parts.append("Relevant context from memory:\n" + context)
        if style:
            parts.append(f"Write in the following style: {style}")
        parts.append(f"User request:\n{prompt}")
        return "\n\n".join(parts)

    def _fetch_context(
        self, user_id: str | None, query: str
    ) -> str:
        """Retrieve relevant memories and format as context string."""
        self._last_context_memories: list[Memory] = []
        if self._context is None or user_id is None:
            return ""
        try:
            results = self._context.retrieve(query, user_id=user_id, limit=self._max_context)
            self._last_context_memories = [r.memory for r in results]
            lines = []
            for r in results:
                m = r.memory
                lines.append(f"- [{m.memory_type.value}] {m.content} (score={r.score:.2f})")
            return "\n".join(lines)
        except Exception as exc:
            _logger.warning("Context retrieval failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rewrite_prompt(
        text: str, instruction: str, context: str, style: str | None
    ) -> str:
        parts = [
            "Rewrite the following text.",
            f"Instruction: {instruction}",
        ]
        if style:
            parts.append(f"Target style: {style}")
        if context:
            parts.append(f"Relevant context:\n{context}")
        parts.append(f"Text to rewrite:\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_expand_prompt(
        text: str, target_length: int | None, context: str, style: str | None
    ) -> str:
        parts = ["Expand the following text with more detail and depth."]
        if target_length:
            parts.append(f"Target length: ~{target_length} words.")
        if style:
            parts.append(f"Target style: {style}")
        if context:
            parts.append(f"Relevant context:\n{context}")
        parts.append(f"Text to expand:\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_summarize_prompt(
        text: str, max_words: int | None, context: str
    ) -> str:
        parts = ["Summarize the following text concisely."]
        if max_words:
            parts.append(f"Maximum words: {max_words}")
        if context:
            parts.append(f"Relevant context:\n{context}")
        parts.append(f"Text to summarize:\n{text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM dispatch
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        if self._llm is None:
            raise RuntimeError("No LLM caller configured for WritingAssistant")
        return self._llm(prompt).strip()
