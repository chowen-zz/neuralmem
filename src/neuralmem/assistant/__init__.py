"""NeuralMem AI Writing Assistant — V2.2

Memory-aware writing helper with context injection, suggestions,
templates, and LLM-driven generation.  No real LLM calls in tests.
"""
from __future__ import annotations

from .assistant import WritingAssistant
from .context import ContextInjector
from .suggestions import SuggestionEngine
from .templates import TemplateManager

__all__ = [
    "WritingAssistant",
    "ContextInjector",
    "SuggestionEngine",
    "TemplateManager",
]
