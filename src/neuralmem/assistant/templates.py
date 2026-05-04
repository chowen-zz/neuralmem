"""TemplateManager — predefined writing templates for common formats."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class WritingTemplate:
    """A reusable writing template."""

    name: str
    description: str
    system_prompt: str
    placeholders: list[str] = field(default_factory=list)
    default_values: dict[str, str] = field(default_factory=dict)


class TemplateManager:
    """Manages predefined and custom writing templates.

    Templates include: email, blog_post, code_doc, meeting_notes,
    social_media, technical_spec, release_notes, and more.
    """

    def __init__(self, templates: dict[str, WritingTemplate] | None = None) -> None:
        self._templates: dict[str, WritingTemplate] = {}
        if templates:
            self._templates.update(templates)
        self._register_defaults()

    # ------------------------------------------------------------------
    # Default templates
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        defaults = [
            WritingTemplate(
                name="email",
                description="Professional email",
                system_prompt=(
                    "Write a professional email. Include a clear subject line, "
                    "greeting, body paragraphs, and a polite closing. "
                    "Tone: {tone}. Length: {length}."
                ),
                placeholders=["tone", "length"],
                default_values={"tone": "professional", "length": "medium"},
            ),
            WritingTemplate(
                name="blog_post",
                description="Engaging blog post",
                system_prompt=(
                    "Write an engaging blog post with a catchy headline, "
                    "introduction, structured body sections, and a conclusion. "
                    "Tone: {tone}. Target audience: {audience}."
                ),
                placeholders=["tone", "audience"],
                default_values={"tone": "conversational", "audience": "technical"},
            ),
            WritingTemplate(
                name="code_doc",
                description="Code documentation / docstring",
                system_prompt=(
                    "Write clear code documentation. Include a summary, "
                    "parameter descriptions, return value, and usage example. "
                    "Style: {style}. Language: {language}."
                ),
                placeholders=["style", "language"],
                default_values={"style": "Google", "language": "Python"},
            ),
            WritingTemplate(
                name="meeting_notes",
                description="Structured meeting notes",
                system_prompt=(
                    "Write structured meeting notes. Include attendees, agenda, "
                    "key discussion points, decisions, and action items with owners. "
                    "Format: {format}."
                ),
                placeholders=["format"],
                default_values={"format": "markdown"},
            ),
            WritingTemplate(
                name="social_media",
                description="Social media post",
                system_prompt=(
                    "Write a concise social media post. Include a hook, "
                    "main message, and call-to-action. Platform: {platform}. "
                    "Tone: {tone}. Max length: {max_length} characters."
                ),
                placeholders=["platform", "tone", "max_length"],
                default_values={"platform": "Twitter", "tone": "casual", "max_length": "280"},
            ),
            WritingTemplate(
                name="technical_spec",
                description="Technical specification document",
                system_prompt=(
                    "Write a technical specification. Include overview, goals, "
                    "non-goals, design, API surface, trade-offs, and rollout plan. "
                    "Audience: {audience}."
                ),
                placeholders=["audience"],
                default_values={"audience": "engineering"},
            ),
            WritingTemplate(
                name="release_notes",
                description="Software release notes",
                system_prompt=(
                    "Write software release notes. Include version, summary, "
                    "new features, improvements, bug fixes, breaking changes, "
                    "and migration guide if needed. Tone: {tone}."
                ),
                placeholders=["tone"],
                default_values={"tone": "professional"},
            ),
            WritingTemplate(
                name="pr_description",
                description="Pull request description",
                system_prompt=(
                    "Write a clear pull request description. Include what changed, "
                    "why it changed, how to test, and any risks or follow-ups. "
                    "Template style: {style}."
                ),
                placeholders=["style"],
                default_values={"style": "GitHub"},
            ),
            WritingTemplate(
                name="user_story",
                description="Agile user story",
                system_prompt=(
                    "Write a user story in the format: 'As a [role], I want [goal] "
                    "so that [benefit]'. Include acceptance criteria and notes. "
                    "Role: {role}."
                ),
                placeholders=["role"],
                default_values={"role": "user"},
            ),
        ]
        for t in defaults:
            self._templates[t.name] = t

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_templates(self) -> list[str]:
        """Return names of all registered templates."""
        return list(self._templates.keys())

    def get_template(self, name: str) -> WritingTemplate | None:
        """Get a template by name."""
        return self._templates.get(name)

    def add_template(self, template: WritingTemplate) -> None:
        """Register a custom template."""
        self._templates[template.name] = template

    def remove_template(self, name: str) -> bool:
        """Remove a template. Returns True if removed."""
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        template_name: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        """Apply a template to a user prompt, filling placeholders.

        Returns the enriched prompt ready for an LLM.
        """
        template = self._templates.get(template_name)
        if template is None:
            _logger.warning("Template '%s' not found, returning raw prompt", template_name)
            return user_prompt

        # Merge defaults with user overrides
        values = dict(template.default_values)
        values.update(kwargs)

        # Fill system prompt placeholders
        system = template.system_prompt
        for key in template.placeholders:
            val = values.get(key, f"{{{key}}}")
            system = system.replace(f"{{{key}}}", str(val))

        return f"{system}\n\nUser request:\n{user_prompt}"

    def get_template_info(self, name: str) -> dict[str, Any] | None:
        """Return human-readable template metadata."""
        t = self._templates.get(name)
        if t is None:
            return None
        return {
            "name": t.name,
            "description": t.description,
            "placeholders": t.placeholders,
            "defaults": t.default_values,
        }
