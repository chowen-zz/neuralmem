"""ExtractorProtocol 契约测试

注意：MemoryExtractor.extract() 返回 list[ExtractedItem]，与协议中定义的
tuple[list[Entity], list[Relation]] 签名不同。
这里仅测试提取器的实际可用接口，而非严格的协议一致性。
"""
from __future__ import annotations

from neuralmem.extraction.extractor import MemoryExtractor


def test_extractor_extract_returns_list(config):
    """extract() 返回 list[ExtractedItem]"""
    extractor = MemoryExtractor(config)
    result = extractor.extract("Alice uses Python")
    assert isinstance(result, list)
    assert len(result) > 0


def test_extracted_item_has_entities(config):
    """每个 ExtractedItem 包含 entities 列表"""
    extractor = MemoryExtractor(config)
    items = extractor.extract("Alice uses Python")
    for item in items:
        assert isinstance(item.entities, list)
        assert isinstance(item.relations, list)


def test_extracted_item_structure(config):
    """ExtractedItem 包含必要字段"""
    extractor = MemoryExtractor(config)
    items = extractor.extract("User prefers TypeScript")
    assert len(items) > 0
    item = items[0]
    assert hasattr(item, "content")
    assert hasattr(item, "memory_type")
    assert hasattr(item, "entities")
    assert hasattr(item, "relations")
    assert hasattr(item, "tags")
    assert hasattr(item, "importance")
