"""规则提取器单元测试"""
from __future__ import annotations

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import MemoryType
from neuralmem.extraction.extractor import MemoryExtractor


@pytest.fixture
def extractor():
    return MemoryExtractor(NeuralMemConfig())


def test_extract_basic(extractor):
    items = extractor.extract("User prefers Python for data science")
    assert len(items) > 0
    assert all(item.content for item in items)


def test_extract_preference_tag(extractor):
    items = extractor.extract("I prefer TypeScript for frontend")
    tags = [tag for item in items for tag in item.tags]
    assert "preference" in tags


def test_extract_procedural_type(extractor):
    items = extractor.extract("First run tests, then build the image, then deploy")
    types = [item.memory_type for item in items]
    assert any(t == MemoryType.PROCEDURAL for t in types)


def test_extract_tech_entities(extractor):
    items = extractor.extract("I use python and react")
    entities = [e for item in items for e in item.entities]
    entity_names = {e.name.lower() for e in entities}
    assert "python" in entity_names or "react" in entity_names


def test_extract_empty_input(extractor):
    # 空输入应该返回列表（不报错）
    items = extractor.extract("")
    assert isinstance(items, list)


def test_extract_importance_score(extractor):
    items = extractor.extract("This is critical for production deployment")
    assert all(0.0 <= item.importance <= 1.0 for item in items)


def test_extract_short_text_no_crash(extractor):
    items = extractor.extract("ok")
    assert isinstance(items, list)


def test_extract_returns_extracted_items(extractor):
    from neuralmem.extraction.extractor import ExtractedItem
    items = extractor.extract("Alice uses Python and React for her project")
    assert all(isinstance(item, ExtractedItem) for item in items)


def test_extract_entity_ids_match_entities(extractor):
    """entity_ids 应与 entities 列表中的 id 一致"""
    items = extractor.extract("User works with Python")
    for item in items:
        assert set(item.entity_ids) == {e.id for e in item.entities}


def test_extract_importance_high_for_critical(extractor):
    """含 'critical' 关键词的文本重要性应高于普通文本"""
    normal = extractor.extract("User likes coffee")
    critical = extractor.extract("This is critical for the project")
    avg_normal = sum(i.importance for i in normal) / len(normal)
    avg_critical = sum(i.importance for i in critical) / len(critical)
    assert avg_critical >= avg_normal
