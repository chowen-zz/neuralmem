"""Unit tests for CloudflareWorkersAdapter.

All tests are mock-based; no external API or Workers runtime required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from neuralmem.edge.adapter import CloudflareWorkersAdapter
from neuralmem.edge.config import EdgeConfig
from neuralmem.edge.storage import EdgeStorage


@pytest.fixture
def edge_config():
    return EdgeConfig(
        kv_memories="TEST_MEMORIES",
        require_auth=False,
        default_search_limit=5,
    )


@pytest.fixture
def mock_kv():
    return {}


@pytest.fixture
def adapter(edge_config, mock_kv):
    adapter = CloudflareWorkersAdapter(config=edge_config)
    # Pre-seed storage with mock KV
    adapter._storage = EdgeStorage(config=edge_config, kv=mock_kv)
    return adapter


@pytest.fixture
def mock_request():
    def _make(method="GET", url="https://example.com/memories", body=None):
        req = MagicMock()
        req.method = method
        req.url = url
        # Simulate async json() coroutine
        async def _json():
            return body
        req.json = _json
        req.headers = {}
        return req
    return _make


# --------------------------------------------------------------------------- #
# fetch handler
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_fetch_remember(adapter, mock_request):
    req = mock_request(
        method="POST",
        url="https://example.com/memories",
        body={"content": "Edge test memory", "memory_type": "fact"},
    )
    env = {"TEST_MEMORIES": {}}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 201
    data = json.loads(resp["body"])
    assert data["status"] == "created"
    assert "id" in data


@pytest.mark.asyncio
async def test_fetch_list_memories(adapter, mock_request, mock_kv):
    # Seed a memory
    from neuralmem.core.types import Memory, MemoryType
    mem = Memory(content="hello", memory_type=MemoryType.FACT)
    adapter._storage.save_memory(mem)

    req = mock_request(method="GET", url="https://example.com/memories")
    env = {"TEST_MEMORIES": mock_kv}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 200
    data = json.loads(resp["body"])
    assert data["count"] == 1
    assert data["memories"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_fetch_get_memory(adapter, mock_request, mock_kv):
    from neuralmem.core.types import Memory, MemoryType
    mem = Memory(content="specific", memory_type=MemoryType.EPISODIC)
    adapter._storage.save_memory(mem)

    req = mock_request(method="GET", url=f"https://example.com/memories/{mem.id}")
    env = {"TEST_MEMORIES": mock_kv}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 200
    data = json.loads(resp["body"])
    assert data["memory"]["content"] == "specific"


@pytest.mark.asyncio
async def test_fetch_delete_memory(adapter, mock_request, mock_kv):
    from neuralmem.core.types import Memory, MemoryType
    mem = Memory(content="to delete", memory_type=MemoryType.FACT)
    adapter._storage.save_memory(mem)

    req = mock_request(method="DELETE", url=f"https://example.com/memories/{mem.id}")
    env = {"TEST_MEMORIES": mock_kv}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 200
    data = json.loads(resp["body"])
    assert data["deleted"] == 1


@pytest.mark.asyncio
async def test_fetch_search_keyword(adapter, mock_request, mock_kv):
    from neuralmem.core.types import Memory, MemoryType
    mem = Memory(content="searchable content", memory_type=MemoryType.FACT)
    adapter._storage.save_memory(mem)

    req = mock_request(
        method="POST",
        url="https://example.com/search",
        body={"query": "searchable", "type": "keyword"},
    )
    env = {"TEST_MEMORIES": mock_kv}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 200
    data = json.loads(resp["body"])
    assert data["count"] >= 1


@pytest.mark.asyncio
async def test_fetch_stats(adapter, mock_request, mock_kv):
    req = mock_request(method="GET", url="https://example.com/stats")
    env = {"TEST_MEMORIES": mock_kv}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 200
    data = json.loads(resp["body"])
    assert "stats" in data


@pytest.mark.asyncio
async def test_fetch_not_found(adapter, mock_request):
    req = mock_request(method="GET", url="https://example.com/unknown")
    env = {}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 404


@pytest.mark.asyncio
async def test_fetch_cors_preflight(adapter, mock_request):
    req = mock_request(method="OPTIONS", url="https://example.com/memories")
    env = {}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 204
    assert "Access-Control-Allow-Origin" in resp["headers"]


@pytest.mark.asyncio
async def test_fetch_auth_required(adapter, mock_request, edge_config):
    cfg = EdgeConfig.from_env()
    cfg.require_auth = True
    cfg.api_key = "secret-key"
    auth_adapter = CloudflareWorkersAdapter(config=cfg)

    req = mock_request(method="GET", url="https://example.com/memories")
    req.headers = {"Authorization": "Bearer secret-key"}
    env = {}
    resp = await auth_adapter.fetch(req, env)
    # Should not 401 because key is correct
    assert resp["status"] != 401

    req2 = mock_request(method="GET", url="https://example.com/memories")
    req2.headers = {"Authorization": "Bearer wrong-key"}
    resp2 = await auth_adapter.fetch(req2, env)
    assert resp2["status"] == 403


@pytest.mark.asyncio
async def test_fetch_bad_request(adapter, mock_request):
    req = mock_request(
        method="POST",
        url="https://example.com/memories",
        body={},  # missing content
    )
    env = {}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 400


@pytest.mark.asyncio
async def test_fetch_vector_search_requires_embedding(adapter, mock_request):
    req = mock_request(
        method="POST",
        url="https://example.com/search",
        body={"query": "test", "type": "vector"},
    )
    env = {}
    resp = await adapter.fetch(req, env)
    assert resp["status"] == 400
    data = json.loads(resp["body"])
    assert "embedding" in data["detail"]


# --------------------------------------------------------------------------- #
# convenience methods
# --------------------------------------------------------------------------- #

def test_remember_convenience(adapter, mock_kv):
    env = {"TEST_MEMORIES": mock_kv}
    # This path uses NeuralMem internally; we just verify no crash
    # Since NeuralMem requires embedder, we mock it out
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("neuralmem.core.memory.NeuralMem.__init__", lambda self, **kw: None)
        mp.setattr("neuralmem.core.memory.NeuralMem.remember", lambda self, content, **kw: [])
        result = adapter.remember("hello", env)
        assert result == []


def test_recall_convenience(adapter, mock_kv):
    env = {"TEST_MEMORIES": mock_kv}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("neuralmem.core.memory.NeuralMem.__init__", lambda self, **kw: None)
        mp.setattr("neuralmem.core.memory.NeuralMem.recall", lambda self, query, **kw: [])
        result = adapter.recall("hello", env)
        assert result == []


# --------------------------------------------------------------------------- #
# lazy init
# --------------------------------------------------------------------------- #

def test_lazy_handler_creation(adapter, mock_kv):
    assert adapter._handler is None
    env = {"TEST_MEMORIES": mock_kv}
    handler = adapter._get_handler(env)
    assert handler is not None
    assert adapter._handler is handler


def test_lazy_storage_creation(adapter, mock_kv):
    assert adapter._storage is not None  # pre-seeded in fixture
    env = {"TEST_MEMORIES": mock_kv}
    storage = adapter._get_storage(env)
    assert storage is not None
