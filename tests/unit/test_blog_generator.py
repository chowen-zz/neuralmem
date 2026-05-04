"""Unit tests for NeuralMem V1.7 TechBlogGenerator — all mock-based.

Covers:
  • BlogPost / PublishResult dataclasses
  • SEOOptimizer (title, description, tags, reading time)
  • PlatformPublisher (payload building, publish flow)
  • TechBlogGenerator (generate, series, export, publish)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neuralmem.community.blog_generator import (
    BlogPost,
    PublishResult,
    SEOOptimizer,
    PlatformPublisher,
    TechBlogGenerator,
)


# =============================================================================
# BlogPost / PublishResult
# =============================================================================

class TestBlogPost:
    def test_blog_post_defaults(self):
        post = BlogPost(title="T", slug="t", content="C", summary="S")
        assert post.title == "T"
        assert post.tags == []
        assert post.reading_time_minutes == 0
        assert isinstance(post.generated_at, datetime)

    def test_blog_post_with_code_blocks(self):
        post = BlogPost(
            title="Code Blog",
            slug="code-blog",
            content="```python\nprint(1)\n```",
            summary="S",
            code_blocks=[{"language": "python", "code": "print(1)"}],
        )
        assert len(post.code_blocks) == 1
        assert post.code_blocks[0]["language"] == "python"


class TestPublishResult:
    def test_publish_result_success(self):
        result = PublishResult(platform="devto", success=True, post_url="https://dev.to/x")
        assert result.platform == "devto"
        assert result.success is True
        assert result.error_message == ""

    def test_publish_result_failure(self):
        result = PublishResult(platform="medium", success=False, error_message="Auth failed")
        assert result.success is False
        assert result.error_message == "Auth failed"


# =============================================================================
# SEOOptimizer
# =============================================================================

class TestSEOOptimizer:
    def test_generate_seo_title_trims_long(self):
        seo = SEOOptimizer()
        title = "A " * 40
        optimized = seo._generate_seo_title(title)
        assert len(optimized) <= 60

    def test_generate_seo_description_trims_long(self):
        seo = SEOOptimizer()
        summary = "word " * 100
        desc = seo._generate_seo_description(summary)
        assert len(desc) <= 160

    def test_recommend_tags_discovers_keywords(self):
        seo = SEOOptimizer()
        tags = seo._recommend_tags("Python Memory", "Using vector search with python memory", [])
        assert "python" in [t.lower() for t in tags]
        assert "memory" in [t.lower() for t in tags]

    def test_recommend_tags_deduplicates(self):
        seo = SEOOptimizer()
        tags = seo._recommend_tags("X", "Y", ["python", "Python", "ai"])
        lowered = [t.lower() for t in tags]
        assert lowered.count("python") == 1

    def test_estimate_reading_time(self):
        seo = SEOOptimizer()
        content = "word " * 400
        assert seo._estimate_reading_time(content) == 2

    def test_optimize_populates_all_fields(self):
        seo = SEOOptimizer()
        post = BlogPost(title="My Title", slug="my-title", content="word " * 600, summary="Summary.")
        optimized = seo.optimize(post)
        assert optimized.seo_title
        assert optimized.seo_description
        assert optimized.reading_time_minutes == 3
        # Tags may be empty if no keywords match; just ensure it's a list
        assert isinstance(optimized.tags, list)

    def test_recommend_platform_tags_respects_limits(self):
        seo = SEOOptimizer()
        tags = ["a", "b", "c", "d", "e", "f"]
        devto_tags = seo.recommend_platform_tags(tags, "devto")
        assert len(devto_tags) <= 4
        medium_tags = seo.recommend_platform_tags(tags, "medium")
        assert len(medium_tags) <= 5


# =============================================================================
# PlatformPublisher
# =============================================================================

class TestPlatformPublisher:
    def test_publish_unsupported_platform(self):
        pub = PlatformPublisher(api_tokens={"devto": "tok"})
        post = BlogPost(title="T", slug="t", content="C", summary="S")
        result = pub.publish(post, "unknown")
        assert result.success is False
        assert "Unsupported" in result.error_message

    def test_publish_missing_token(self):
        pub = PlatformPublisher()
        post = BlogPost(title="T", slug="t", content="C", summary="S")
        result = pub.publish(post, "devto")
        assert result.success is False
        assert "No API token" in result.error_message

    def test_publish_all_only_configured(self):
        pub = PlatformPublisher(api_tokens={"devto": "tok1", "medium": "tok2"})
        post = BlogPost(title="T", slug="t", content="C", summary="S")
        results = pub.publish_all(post)
        assert len(results) == 2
        platforms = {r.platform for r in results}
        assert platforms == {"devto", "medium"}

    def test_build_payload_devto(self):
        pub = PlatformPublisher()
        post = BlogPost(title="T", slug="t", content="C", summary="S", tags=["a"])
        payload = pub._build_payload(post, "devto")
        assert "article" in payload
        assert payload["article"]["title"] == "T"

    def test_build_payload_medium(self):
        pub = PlatformPublisher()
        post = BlogPost(title="T", slug="t", content="C", summary="S", tags=["a"])
        payload = pub._build_payload(post, "medium")
        assert payload["contentFormat"] == "markdown"
        assert payload["canonicalUrl"] == ""

    def test_build_payload_hashnode(self):
        pub = PlatformPublisher()
        post = BlogPost(title="T", slug="t", content="C", summary="S", tags=["a"])
        payload = pub._build_payload(post, "hashnode")
        assert payload["slug"] == "t"
        assert payload["tags"] == [{"name": "a"}]

    def test_get_published_posts_returns_copy(self):
        pub = PlatformPublisher()
        pub._published.append(PublishResult(platform="devto", success=True))
        assert pub.get_published_posts() == pub._published
        pub.get_published_posts().clear()
        assert len(pub._published) == 1


class TestPlatformPublisherMockHTTP:
    def test_publish_with_mock_http(self):
        mock_http = MagicMock(return_value={"url": "https://dev.to/x", "id": "123"})
        pub = PlatformPublisher(api_tokens={"devto": "tok"})
        pub._send_publish_request = lambda platform, payload, token: PublishResult(
            platform=platform,
            success=True,
            post_url="https://dev.to/x",
            post_id="123",
        )
        post = BlogPost(title="T", slug="t", content="C", summary="S")
        result = pub.publish(post, "devto")
        assert result.success is True
        assert result.post_url == "https://dev.to/x"


# =============================================================================
# TechBlogGenerator
# =============================================================================

class TestTechBlogGeneratorGenerate:
    def test_generate_from_memory_basic(self):
        gen = TechBlogGenerator()
        post = gen.generate_from_memory("NeuralMem vector search is powerful.")
        assert post.title
        assert post.slug
        assert "NeuralMem" in post.content
        assert post.reading_time_minutes > 0
        assert post.tags

    def test_generate_from_memory_with_custom_title(self):
        gen = TechBlogGenerator()
        post = gen.generate_from_memory("Content here.", title="Custom Title")
        assert post.title == "Custom Title"

    def test_generate_from_memory_with_tags(self):
        gen = TechBlogGenerator()
        post = gen.generate_from_memory("X", tags=["python", "ai"])
        assert "python" in [t.lower() for t in post.tags]
        assert "ai" in [t.lower() for t in post.tags]

    def test_generate_from_memory_extracts_code(self):
        gen = TechBlogGenerator()
        content = "Some text.\n```python\nprint('hello')\n```\nMore text."
        post = gen.generate_from_memory(content)
        assert any(cb["language"] == "python" for cb in post.code_blocks)

    def test_generate_from_memory_uses_provided_code_example(self):
        gen = TechBlogGenerator()
        post = gen.generate_from_memory("X", code_example="x = 1")
        assert "x = 1" in post.content

    def test_generate_series(self):
        gen = TechBlogGenerator()
        contents = ["Part one.", "Part two.", "Part three."]
        posts = gen.generate_series(contents, "My Series")
        assert len(posts) == 3
        assert posts[0].title == "My Series — Part 1"
        assert posts[2].title == "My Series — Part 3"


class TestTechBlogGeneratorHistory:
    def test_get_history_returns_copy(self):
        gen = TechBlogGenerator()
        gen.generate_from_memory("X")
        history = gen.get_history()
        assert len(history) == 1
        history.clear()
        assert len(gen.get_history()) == 1


class TestTechBlogGeneratorExport:
    def test_export_to_markdown(self, tmp_path: Path):
        gen = TechBlogGenerator()
        post = gen.generate_from_memory("Y")
        path = gen.export_to_markdown(post, tmp_path / "post.md")
        assert path.exists()
        assert post.content in path.read_text()

    def test_export_to_json(self, tmp_path: Path):
        gen = TechBlogGenerator()
        post = gen.generate_from_memory("Z")
        path = gen.export_to_json(post, tmp_path / "post.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["title"] == post.title
        assert "seo_title" in data


class TestTechBlogGeneratorPublish:
    def test_publish_delegates_to_publisher(self):
        mock_publisher = MagicMock(spec=PlatformPublisher)
        mock_publisher.publish.return_value = PublishResult(
            platform="devto", success=True, post_url="https://dev.to/x"
        )
        gen = TechBlogGenerator(publisher=mock_publisher)
        post = gen.generate_from_memory("X")
        result = gen.publish(post, "devto")
        mock_publisher.publish.assert_called_once_with(post, "devto")
        assert result.success is True

    def test_publish_all_delegates_to_publisher(self):
        mock_publisher = MagicMock(spec=PlatformPublisher)
        mock_publisher.publish_all.return_value = [
            PublishResult(platform="devto", success=True),
            PublishResult(platform="medium", success=True),
        ]
        gen = TechBlogGenerator(publisher=mock_publisher)
        post = gen.generate_from_memory("X")
        results = gen.publish_all(post)
        mock_publisher.publish_all.assert_called_once_with(post)
        assert len(results) == 2


class TestTechBlogGeneratorHelpers:
    def test_slugify(self):
        gen = TechBlogGenerator()
        assert gen._slugify("Hello World!") == "hello-world"
        assert gen._slugify("  a---b  ") == "a-b"

    def test_infer_title_from_content(self):
        gen = TechBlogGenerator()
        assert gen._infer_title("Short.") == "Deep Dive: NeuralMem Technical Guide"
        assert gen._infer_title("This is a longer meaningful line here.") == "This is a longer meaningful line here."

    def test_default_code(self):
        gen = TechBlogGenerator()
        code = gen._default_code()
        assert "from neuralmem import NeuralMem" in code

    def test_extract_code_blocks(self):
        gen = TechBlogGenerator()
        content = "```python\nx=1\n```\n```js\ny=2\n```"
        blocks = gen._extract_code_blocks(content)
        assert len(blocks) == 2
        assert blocks[0]["language"] == "python"
        assert blocks[1]["language"] == "js"
