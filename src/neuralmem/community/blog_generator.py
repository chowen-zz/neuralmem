"""NeuralMem V1.7 community growth engine — blog generator.

TechBlogGenerator: auto-generate technical blogs from memory content with
SEO optimization, code examples, and multi-platform publishing support.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BlogPost:
    """Represents a generated technical blog post."""

    title: str
    slug: str
    content: str
    summary: str
    tags: list[str] = field(default_factory=list)
    seo_title: str = ""
    seo_description: str = ""
    canonical_url: str = ""
    reading_time_minutes: int = 0
    code_blocks: list[dict[str, Any]] = field(default_factory=list)
    platform_metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PublishResult:
    """Result of publishing a blog post to a platform."""

    platform: str
    success: bool
    post_url: str = ""
    post_id: str = ""
    error_message: str = ""
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SEOOptimizer:
    """SEO optimization engine for technical blog posts."""

    # Popular technical keywords for tag recommendations
    TECH_KEYWORDS: dict[str, list[str]] = {
        "python": ["python", "py", "cpython", "asyncio", "typing"],
        "memory": ["memory", "cache", "storage", "persistence", "vector"],
        "ai": ["ai", "llm", "agent", "machine-learning", "nlp", "embedding"],
        "infra": ["infrastructure", "docker", "kubernetes", "deployment", "devops"],
        "database": ["database", "sql", "sqlite", "vector-db", "indexing"],
        "api": ["api", "rest", "graphql", "mcp", "sdk"],
    }

    PLATFORM_TAG_LIMITS: dict[str, int] = {
        "devto": 4,
        "medium": 5,
        "hashnode": 5,
    }

    def __init__(self) -> None:
        self._keyword_index: dict[str, str] = {}
        for category, keywords in self.TECH_KEYWORDS.items():
            for kw in keywords:
                self._keyword_index[kw.lower()] = category

    def optimize(self, post: BlogPost) -> BlogPost:
        """Return an SEO-optimized copy of the given post."""
        optimized = BlogPost(
            title=post.title,
            slug=post.slug,
            content=post.content,
            summary=post.summary,
            tags=post.tags.copy(),
            code_blocks=[cb.copy() for cb in post.code_blocks],
            platform_metadata=dict(post.platform_metadata),
            generated_at=post.generated_at,
        )
        optimized.seo_title = self._generate_seo_title(post.title)
        optimized.seo_description = self._generate_seo_description(post.summary)
        optimized.tags = self._recommend_tags(post.title, post.content, post.tags)
        optimized.reading_time_minutes = self._estimate_reading_time(post.content)
        return optimized

    def recommend_platform_tags(self, tags: list[str], platform: str) -> list[str]:
        """Recommend tags trimmed to platform limits."""
        limit = self.PLATFORM_TAG_LIMITS.get(platform, 5)
        return tags[:limit]

    def _generate_seo_title(self, title: str, max_length: int = 60) -> str:
        """Generate an SEO-friendly title."""
        clean = re.sub(r"[^\w\s-]", "", title).strip()
        if len(clean) > max_length:
            clean = clean[: max_length - 3].rsplit(" ", 1)[0] + "..."
        return clean

    def _generate_seo_description(self, summary: str, max_length: int = 160) -> str:
        """Generate an SEO-friendly meta description."""
        clean = re.sub(r"\s+", " ", summary.strip())
        if len(clean) > max_length:
            clean = clean[: max_length - 3].rsplit(" ", 1)[0] + "..."
        return clean

    def _recommend_tags(self, title: str, content: str, existing: list[str]) -> list[str]:
        """Recommend tags based on content analysis."""
        text = f"{title} {content}".lower()
        discovered: set[str] = set()
        for kw, category in self._keyword_index.items():
            if kw in text:
                discovered.add(category)
        merged = list(dict.fromkeys(existing + sorted(discovered)))
        # Deduplicate case-insensitively while preserving first-seen case
        seen: set[str] = set()
        result: list[str] = []
        for tag in merged:
            lower = tag.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(tag)
        return result

    def _estimate_reading_time(self, content: str, wpm: int = 200) -> int:
        """Estimate reading time in minutes."""
        words = len(content.split())
        return max(1, round(words / wpm))


class PlatformPublisher:
    """Multi-platform blog publisher (Dev.to / Medium / Hashnode)."""

    SUPPORTED_PLATFORMS: set[str] = {"devto", "medium", "hashnode"}

    def __init__(self, api_tokens: dict[str, str] | None = None) -> None:
        self._tokens = api_tokens or {}
        self._published: list[PublishResult] = []

    def publish(self, post: BlogPost, platform: str) -> PublishResult:
        """Publish a blog post to the given platform."""
        if platform not in self.SUPPORTED_PLATFORMS:
            return PublishResult(
                platform=platform,
                success=False,
                error_message=f"Unsupported platform: {platform}",
            )
        token = self._tokens.get(platform)
        if not token:
            return PublishResult(
                platform=platform,
                success=False,
                error_message=f"No API token configured for {platform}",
            )
        # Platform-specific payload generation (mockable in tests)
        payload = self._build_payload(post, platform)
        result = self._send_publish_request(platform, payload, token)
        self._published.append(result)
        return result

    def publish_all(self, post: BlogPost) -> list[PublishResult]:
        """Publish to all configured platforms."""
        results: list[PublishResult] = []
        for platform in self.SUPPORTED_PLATFORMS:
            if platform in self._tokens:
                results.append(self.publish(post, platform))
        return results

    def get_published_posts(self) -> list[PublishResult]:
        """Return all published post results."""
        return self._published.copy()

    def _build_payload(self, post: BlogPost, platform: str) -> dict[str, Any]:
        """Build platform-specific publish payload."""
        base = {
            "title": post.seo_title or post.title,
            "body_markdown": post.content,
            "tags": post.tags,
            "canonical_url": post.canonical_url,
        }
        if platform == "devto":
            return {"article": base}
        if platform == "medium":
            return {
                "title": base["title"],
                "contentFormat": "markdown",
                "content": base["body_markdown"],
                "tags": base["tags"],
                "canonicalUrl": base["canonical_url"],
            }
        if platform == "hashnode":
            return {
                "title": base["title"],
                "contentMarkdown": base["body_markdown"],
                "tags": [{"name": t} for t in base["tags"]],
                "slug": post.slug,
            }
        return base

    def _send_publish_request(
        self, platform: str, payload: dict[str, Any], token: str
    ) -> PublishResult:
        """Send publish request — override in tests or subclass."""
        # Default no-op: subclasses / test mocks override this
        return PublishResult(
            platform=platform,
            success=False,
            error_message="_send_publish_request not implemented",
        )


class TechBlogGenerator:
    """Auto-generate technical blogs from memory content."""

    TEMPLATE_BLOG = """# {title}

> {summary}

## Introduction

{intro}

## Key Concepts

{concepts}

## Code Example

```python
{code_example}
```

## Use Cases

{use_cases}

## Conclusion

{conclusion}

---
*Generated by NeuralMem TechBlogGenerator*
"""

    def __init__(
        self,
        seo_optimizer: SEOOptimizer | None = None,
        publisher: PlatformPublisher | None = None,
    ) -> None:
        self._seo = seo_optimizer or SEOOptimizer()
        self._publisher = publisher or PlatformPublisher()
        self._history: list[BlogPost] = []

    def generate_from_memory(
        self,
        memory_content: str,
        title: str | None = None,
        tags: list[str] | None = None,
        code_example: str | None = None,
    ) -> BlogPost:
        """Generate a technical blog post from memory content."""
        generated_title = title or self._infer_title(memory_content)
        slug = self._slugify(generated_title)
        summary = self._generate_summary(memory_content)
        intro = self._generate_intro(memory_content)
        concepts = self._generate_concepts(memory_content)
        use_cases = self._generate_use_cases(memory_content)
        conclusion = self._generate_conclusion(memory_content)
        code = code_example or self._extract_code_example(memory_content) or self._default_code()

        content = self.TEMPLATE_BLOG.format(
            title=generated_title,
            summary=summary,
            intro=intro,
            concepts=concepts,
            code_example=code,
            use_cases=use_cases,
            conclusion=conclusion,
        )

        code_blocks = self._extract_code_blocks(content)

        post = BlogPost(
            title=generated_title,
            slug=slug,
            content=content,
            summary=summary,
            tags=tags or [],
            code_blocks=code_blocks,
        )
        optimized = self._seo.optimize(post)
        self._history.append(optimized)
        return optimized

    def generate_series(
        self, memory_contents: list[str], series_title: str
    ) -> list[BlogPost]:
        """Generate a series of blog posts from multiple memory contents."""
        posts: list[BlogPost] = []
        for idx, content in enumerate(memory_contents, start=1):
            title = f"{series_title} — Part {idx}"
            post = self.generate_from_memory(content, title=title)
            posts.append(post)
        return posts

    def publish(self, post: BlogPost, platform: str) -> PublishResult:
        """Publish a post to a specific platform."""
        return self._publisher.publish(post, platform)

    def publish_all(self, post: BlogPost) -> list[PublishResult]:
        """Publish a post to all configured platforms."""
        return self._publisher.publish_all(post)

    def get_history(self) -> list[BlogPost]:
        """Return all generated blog posts."""
        return self._history.copy()

    def export_to_markdown(self, post: BlogPost, output_path: Path | str) -> Path:
        """Export a blog post to a markdown file."""
        path = Path(output_path)
        path.write_text(post.content, encoding="utf-8")
        return path

    def export_to_json(self, post: BlogPost, output_path: Path | str) -> Path:
        """Export a blog post to a JSON file."""
        path = Path(output_path)
        data = {
            "title": post.title,
            "slug": post.slug,
            "summary": post.summary,
            "tags": post.tags,
            "seo_title": post.seo_title,
            "seo_description": post.seo_description,
            "reading_time_minutes": post.reading_time_minutes,
            "content": post.content,
            "generated_at": post.generated_at.isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _infer_title(self, content: str) -> str:
        """Infer a blog title from memory content."""
        lines = content.strip().splitlines()
        for line in lines[:5]:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                words = stripped.split()
                if len(words) >= 3:
                    return stripped[:80]
        return "Deep Dive: NeuralMem Technical Guide"

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[-\s]+", "-", slug).strip("-")
        return slug[:100]

    def _generate_summary(self, content: str) -> str:
        """Generate a one-paragraph summary."""
        sentences = content.split(".")
        summary = ".".join(s.strip() for s in sentences[:2] if s.strip())
        return summary + "." if summary else "A technical deep dive into NeuralMem capabilities."

    def _generate_intro(self, content: str) -> str:
        """Generate an introduction paragraph."""
        return (
            "This article explores how NeuralMem enables powerful "
            "memory infrastructure for AI agents. "
            f"{self._generate_summary(content)}"
        )

    def _generate_concepts(self, content: str) -> str:
        """Generate key concepts section."""
        return (
            "- **Memory Tiering**: Hot, warm, and cold storage layers\n"
            "- **Vector Search**: Semantic retrieval with embeddings\n"
            "- **MCP Integration**: Model Context Protocol native support\n"
            f"- **Context Awareness**: {self._generate_summary(content)[:60]}"
        )

    def _generate_use_cases(self, content: str) -> str:
        """Generate use cases section."""
        return (
            "1. **Agent Memory Persistence** — Long-running agents retain context\n"
            "2. **Multi-session Recall** — Cross-session memory retrieval\n"
            "3. **Collaborative Spaces** — Shared memory across team agents\n"
            f"4. **Custom Workflows** — {self._generate_summary(content)[:50]}"
        )

    def _generate_conclusion(self, _content: str) -> str:
        """Generate a conclusion paragraph."""
        return (
            "NeuralMem provides a robust, local-first memory infrastructure "
            "that scales from personal agents to enterprise deployments. "
            "Get started today and unlock the full potential of persistent agent memory."
        )

    def _extract_code_example(self, content: str) -> str | None:
        """Extract a code block from memory content."""
        match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _default_code(self) -> str:
        """Return a default code example."""
        return (
            "from neuralmem import NeuralMem\n\n"
            "mem = NeuralMem()\n"
            "mem.store('User prefers dark mode')\n"
            "context = mem.retrieve('What are user preferences?')\n"
            "print(context)"
        )

    def _extract_code_blocks(self, content: str) -> list[dict[str, Any]]:
        """Extract all code blocks from content."""
        blocks: list[dict[str, Any]] = []
        pattern = r"```(\w+)?\n(.*?)\n```"
        for match in re.finditer(pattern, content, re.DOTALL):
            blocks.append(
                {
                    "language": match.group(1) or "text",
                    "code": match.group(2),
                }
            )
        return blocks
