"""pytest fixtures 全局配置"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# 确保 src 在 Python 路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Memory, MemoryType
from neuralmem.storage.sqlite import SQLiteStorage


@pytest.fixture
def tmp_db_path(tmp_path):
    """每个测试独立的 SQLite 数据库路径"""
    return str(tmp_path / "test.db")


@pytest.fixture
def config(tmp_db_path):
    """测试配置（小维度加快测试）"""
    return NeuralMemConfig(db_path=tmp_db_path, embedding_dim=4)


@pytest.fixture
def storage(config):
    """测试用存储实例"""
    return SQLiteStorage(config)


@pytest.fixture
def sample_vector():
    """固定测试向量（4维，归一化）"""
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return v.tolist()


@pytest.fixture
def sample_memory(sample_vector):
    """带 embedding 的测试记忆"""
    return Memory(
        content="User prefers TypeScript for frontend development",
        memory_type=MemoryType.SEMANTIC,
        user_id="test-user",
        tags=("preference", "technology"),
        importance=0.8,
        embedding=sample_vector,
    )


@pytest.fixture
def sample_memories():
    """10 条测试记忆"""
    import random
    random.seed(42)
    contents = [
        "User likes Python",
        "User works on NeuralMem project",
        "User prefers TypeScript for frontend",
        "User uses React for UI",
        "User deploys to AWS",
        "User wrote a blog post yesterday",
        "User met Alice at conference",
        "User needs to fix authentication bug",
        "Deploy process: test then build then push",
        "User's favorite tool is Claude Code",
    ]
    memories = []
    for i, content in enumerate(contents):
        v = [random.random() for _ in range(4)]
        norm = sum(x ** 2 for x in v) ** 0.5
        v = [x / norm for x in v]
        memories.append(Memory(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.5 + i * 0.05,
            embedding=v,
        ))
    return memories


@pytest.fixture
def mock_embedder():
    """确定性 mock Embedder（不下载模型）"""

    class MockEmbedder:
        dimension = 4

        def encode(self, texts):
            import hashlib
            results = []
            for text in texts:
                h = int(hashlib.md5(text.encode()).hexdigest(), 16)
                v = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(4)]
                norm = sum(x ** 2 for x in v) ** 0.5 or 1.0
                results.append([x / norm for x in v])
            return results

        def encode_one(self, text):
            return self.encode([text])[0]

    return MockEmbedder()


@pytest.fixture
def mem_with_mock(tmp_db_path, mock_embedder):
    """NeuralMem 实例，使用 mock embedder 和规则提取器（不下载模型）"""
    from neuralmem.core.memory import NeuralMem

    config = NeuralMemConfig(db_path=tmp_db_path, embedding_dim=4)
    mem = NeuralMem(config=config)
    mem._embedding_provider = mock_embedder
    return mem
