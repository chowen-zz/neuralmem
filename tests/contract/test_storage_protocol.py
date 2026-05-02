"""StorageProtocol 契约测试 — 验证 SQLiteStorage 满足协议"""
from __future__ import annotations

from neuralmem.core.protocols import StorageProtocol


def test_sqlite_satisfies_protocol(storage):
    assert isinstance(storage, StorageProtocol), \
        "SQLiteStorage must satisfy StorageProtocol"


def test_protocol_methods_exist(storage):
    required = [
        "save_memory", "get_memory", "update_memory", "delete_memories",
        "vector_search", "keyword_search", "temporal_search",
        "find_similar", "record_access", "get_stats",
    ]
    for method in required:
        assert hasattr(storage, method), f"Missing method: {method}"
