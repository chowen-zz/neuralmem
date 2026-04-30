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


def test_new_provider_fields_have_defaults():
    cfg = NeuralMemConfig(db_path=":memory:")
    assert cfg.cohere_api_key is None
    assert cfg.cohere_embedding_model == "embed-multilingual-v3.0"
    assert cfg.gemini_api_key is None
    assert cfg.gemini_embedding_model == "text-embedding-004"
    assert cfg.hf_api_key is None
    assert cfg.hf_model == "BAAI/bge-m3"
    assert cfg.hf_inference_url == "https://api-inference.huggingface.co"
    assert cfg.azure_endpoint is None
    assert cfg.azure_api_key is None
    assert cfg.azure_deployment == "text-embedding-3-small"
    assert cfg.azure_api_version == "2024-02-01"
    assert cfg.llm_extractor == "none"
    assert cfg.openai_extractor_model == "gpt-4o-mini"
    assert cfg.anthropic_api_key is None
    assert cfg.anthropic_model == "claude-haiku-4-5-20251001"


def test_backward_compat_enable_llm_extraction():
    """enable_llm_extraction=True should silently map to llm_extractor='ollama'."""
    cfg = NeuralMemConfig(db_path=":memory:", enable_llm_extraction=True)
    assert cfg.llm_extractor == "ollama"


def test_explicit_llm_extractor_takes_precedence():
    cfg = NeuralMemConfig(db_path=":memory:", llm_extractor="openai")
    assert cfg.llm_extractor == "openai"
