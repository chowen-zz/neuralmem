"""LLMExtractor 单元测试（mock Ollama）"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.extraction.llm_extractor import LLMExtractor


@pytest.fixture
def llm_extractor():
    cfg = NeuralMemConfig(enable_llm_extraction=True)
    return LLMExtractor(cfg)


# ── 禁用 LLM 时直接走规则提取 ────────────────────────────────────────────────


def test_llm_extractor_fallback_when_disabled():
    cfg = NeuralMemConfig(enable_llm_extraction=False)
    extractor = LLMExtractor(cfg)
    items = extractor.extract("User likes Python")
    assert len(items) > 0


# ── Ollama 不可达时走规则提取，并缓存 _available=False ────────────────────────


def test_llm_extractor_fallback_when_ollama_unavailable(llm_extractor):
    with patch("httpx.get", side_effect=Exception("Connection refused")):
        llm_extractor._available = None  # 重置，让 _check_available 重新探测
        items = llm_extractor.extract("User prefers TypeScript")
        assert len(items) > 0
        assert llm_extractor._available is False


# ── 缓存 _available=False 时直接跳过检测 ─────────────────────────────────────


def test_llm_extractor_uses_cached_false(llm_extractor):
    llm_extractor._available = False
    items = llm_extractor.extract("test content")
    assert len(items) > 0


# ── Ollama 可达时 _check_available 返回 True 并缓存 ──────────────────────────


def test_llm_extractor_available_set_on_check():
    cfg = NeuralMemConfig(enable_llm_extraction=True)
    extractor = LLMExtractor(cfg)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("httpx.get", return_value=mock_resp):
        result = extractor._check_available()
        assert result is True
        assert extractor._available is True


# ── Ollama 返回非 200 时标记不可用 ───────────────────────────────────────────


def test_llm_extractor_unavailable_when_non_200():
    cfg = NeuralMemConfig(enable_llm_extraction=True)
    extractor = LLMExtractor(cfg)
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    with patch("httpx.get", return_value=mock_resp):
        result = extractor._check_available()
        assert result is False


# ── LLM 调用失败时降级为规则提取 ─────────────────────────────────────────────


def test_llm_extractor_fallback_on_llm_error(llm_extractor):
    llm_extractor._available = True
    with patch("httpx.post", side_effect=Exception("LLM error")):
        items = llm_extractor.extract("test content")
        assert isinstance(items, list)
        assert len(items) > 0


# ── LLM 正常返回时能合并实体 ──────────────────────────────────────────────────


def test_llm_extractor_merges_entities(llm_extractor):
    llm_extractor._available = True
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": '{"facts": ["fact1"], "entities": [{"name": "Python", "type": "technology"}]}'
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mock_resp):
        items = llm_extractor.extract("User uses Python")
        assert isinstance(items, list)
        assert len(items) > 0
        all_entity_names = {e.name for item in items for e in item.entities}
        assert "Python" in all_entity_names


# ── 已缓存 _available=True 时跳过 check ──────────────────────────────────────


def test_check_available_returns_cached_true(llm_extractor):
    llm_extractor._available = True
    result = llm_extractor._check_available()
    assert result is True


def test_base_llm_extractor_fallback_on_error():
    """BaseLLMExtractor falls back to rule extractor when _call_llm raises."""
    from neuralmem.core.config import NeuralMemConfig
    from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor

    class BrokenExtractor(BaseLLMExtractor):
        def _call_llm(self, prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

    cfg = NeuralMemConfig(db_path=":memory:")
    extractor = BrokenExtractor(cfg)
    items = extractor.extract("Alice works at OpenAI")
    assert isinstance(items, list)
    assert len(items) >= 1


def test_base_llm_extractor_merges_entities():
    """BaseLLMExtractor merges LLM-extracted entities with rule-extracted ones."""
    from neuralmem.core.config import NeuralMemConfig
    from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor

    class FakeExtractor(BaseLLMExtractor):
        def _call_llm(self, prompt: str) -> str:
            return '{"facts": [], "entities": [{"name": "DeepMind", "type": "project"}]}'

    cfg = NeuralMemConfig(db_path=":memory:")
    extractor = FakeExtractor(cfg)
    items = extractor.extract("Alice works at DeepMind")
    all_entity_names = [e.name for e in items[0].entities] if items else []
    assert "DeepMind" in all_entity_names
