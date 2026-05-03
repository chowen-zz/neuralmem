"""Async API 单元测试 — 全部使用 mock, 不依赖外部 API."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neuralmem.async_api.memory import AsyncNeuralMem
from neuralmem.async_api.storage import AsyncStorage
from neuralmem.async_api.embedding import AsyncEmbedder
from neuralmem.async_api.retrieval import AsyncRetrievalEngine


# --------------------------------------------------------------------------- #
# AsyncStorage
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_storage_get_memory():
    """测试异步 get_memory."""
    backend = MagicMock()
    backend.get_memory.return_value = MagicMock(id="m1")
    storage = AsyncStorage(backend)
    result = await storage.get_memory("m1")
    assert result is not None
    assert result.id == "m1"


@pytest.mark.asyncio
async def test_async_storage_save_memory():
    """测试异步 save_memory."""
    backend = MagicMock()
    backend.save_memory.return_value = "m1"
    storage = AsyncStorage(backend)
    mem = MagicMock(id="m1")
    result = await storage.save_memory(mem)
    assert result == "m1"


@pytest.mark.asyncio
async def test_async_storage_batch_get():
    """测试批量并发获取."""
    backend = MagicMock()
    backend.get_memory.side_effect = lambda mid: MagicMock(id=mid)
    storage = AsyncStorage(backend)
    results = await storage.batch_get_memories(["m1", "m2", "m3"])
    assert len(results) == 3


# --------------------------------------------------------------------------- #
# AsyncEmbedder
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_embedder_encode():
    """测试异步编码."""
    backend = MagicMock()
    backend.encode.return_value = [[0.1, 0.2, 0.3]]
    embedder = AsyncEmbedder(backend)
    result = await embedder.encode(["hello"])
    assert len(result) == 1


@pytest.mark.asyncio
async def test_async_embedder_encode_batch():
    """测试批量编码."""
    backend = MagicMock()
    backend.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    embedder = AsyncEmbedder(backend)
    result = await embedder.encode_batch(["a", "b"], batch_size=2)
    assert len(result) == 2


# --------------------------------------------------------------------------- #
# AsyncRetrievalEngine
# --------------------------------------------------------------------------- #

from neuralmem.core.types import SearchQuery, SearchResult

@pytest.mark.asyncio
async def test_async_retrieval_search():
    """测试异步检索."""
    storage = MagicMock()
    storage.vector_search.return_value = [("m1", 0.9)]
    storage.keyword_search.return_value = [("m1", 0.8)]
    storage.temporal_search.return_value = [("m1", 0.7)]
    
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1, 0.2]]
    graph = MagicMock()
    config = MagicMock()
    config.enable_reranker = False
    config.cache_query_embeddings = False
    config.query_embedding_cache_size = 0
    
    engine = AsyncRetrievalEngine(storage, embedder, graph, config)
    query = SearchQuery(query="query", limit=5)
    result = await engine.search(query)
    assert isinstance(result, list)


# --------------------------------------------------------------------------- #
# AsyncNeuralMem
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_neural_mem_remember():
    """测试异步 remember."""
    with patch("neuralmem.async_api.memory.SQLiteStorage") as mock_storage_cls, \
         patch("neuralmem.async_api.memory.get_embedder") as mock_get_embedder, \
         patch("neuralmem.async_api.memory.get_extractor") as mock_get_extractor:
        
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1, 0.2]]
        mock_get_embedder.return_value = mock_embedder
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []
        mock_get_extractor.return_value = mock_extractor
        
        mem = AsyncNeuralMem(db_path=":memory:")
        mem._embedder = AsyncMock()
        mem._embedder.encode.return_value = [[0.1, 0.2]]
        mem._storage = AsyncMock()
        mem._storage.save_memory.return_value = "m1"
        mem._extractor = MagicMock()
        mem._extractor.extract.return_value = []
        
        result = await mem.remember("test content")
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_async_neural_mem_recall():
    """测试异步 recall."""
    with patch("neuralmem.async_api.memory.SQLiteStorage") as mock_storage_cls, \
         patch("neuralmem.async_api.memory.get_embedder") as mock_get_embedder, \
         patch("neuralmem.async_api.memory.get_extractor") as mock_get_extractor:
        
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_get_embedder.return_value = MagicMock()
        mock_get_extractor.return_value = MagicMock()
        
        mem = AsyncNeuralMem(db_path=":memory:")
        
        # Mock retrieval engine
        mock_result = MagicMock()
        mock_result.memory = MagicMock(id="m1", content="test")
        mock_result.score = 0.9
        
        mem._retrieval = AsyncMock()
        mem._retrieval.search.return_value = [mock_result]
        mem._storage = AsyncMock()
        mem._storage.get_memory.return_value = mock_result.memory
        
        result = await mem.recall("query")
        assert isinstance(result, list)
