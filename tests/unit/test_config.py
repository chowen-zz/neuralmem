"""配置单元测试"""
from __future__ import annotations
import pytest
from neuralmem.core.config import NeuralMemConfig


def test_default_config():
    cfg = NeuralMemConfig()
    assert "neuralmem" in cfg.db_path
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.enable_reranker is False


def test_db_path_expanded():
    cfg = NeuralMemConfig(db_path="~/.neuralmem/memory.db")
    assert not cfg.db_path.startswith("~")


def test_custom_db_path(tmp_path):
    path = str(tmp_path / "custom.db")
    cfg = NeuralMemConfig(db_path=path)
    assert cfg.db_path == path


def test_from_env(monkeypatch):
    monkeypatch.setenv("NEURALMEM_DB_PATH", "/tmp/test.db")
    cfg = NeuralMemConfig.from_env()
    assert cfg.db_path == "/tmp/test.db"


def test_embedding_dim_default():
    cfg = NeuralMemConfig()
    assert cfg.embedding_dim == 384


def test_min_score_default():
    cfg = NeuralMemConfig()
    assert cfg.min_score == 0.3


def test_default_search_limit():
    cfg = NeuralMemConfig()
    assert cfg.default_search_limit == 10
