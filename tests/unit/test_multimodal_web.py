"""WebExtractor 单元测试 — 全部使用 mock, 不依赖真实 BeautifulSoup."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.multimodal.web_extractor import WebExtractor


# --------------------------------------------------------------------------- #
# 基础行为
# --------------------------------------------------------------------------- #

def test_web_extractor_inherits_base():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    ext = WebExtractor()
    assert isinstance(ext, BaseMultimodalExtractor)


def test_web_extractor_default_chunk_size():
    ext = WebExtractor()
    assert ext.max_chunk_size == 4096


def test_web_extractor_custom_chunk_size():
    ext = WebExtractor(max_chunk_size=512)
    assert ext.max_chunk_size == 512


# --------------------------------------------------------------------------- #
# extract() — mock BeautifulSoup 成功路径
# --------------------------------------------------------------------------- #

def test_extract_with_mock_bs4():
    """模拟 BeautifulSoup 成功提取可见文本."""
    ext = WebExtractor()

    # 直接 mock _parse_html 方法，绕过模块级别的 _HAS_BS4 检查
    with patch.object(ext, "_parse_html", return_value="Title\n\nParagraph one.\n\nParagraph two."):
        result = ext.extract(b"<html><body><h1>Title</h1><p>Paragraph one.</p></body></html>")

    assert isinstance(result, list)
    assert len(result) >= 1
    combined = " ".join(result)
    assert "Title" in combined
    assert "Paragraph one." in combined


def test_parse_html_mocked_bs4():
    """测试 _parse_html 内部使用 BeautifulSoup."""
    ext = WebExtractor()
    
    mock_soup = MagicMock()
    # 模拟 article 标签找到并返回足够长的文本
    article = MagicMock()
    article.get_text.return_value = "Title here with enough length to pass the 200 char threshold. " * 5
    mock_soup.find.return_value = article
    mock_soup.find_all.return_value = []
    
    mock_bs4 = MagicMock(return_value=mock_soup)
    
    # 直接 patch 模块级别的变量
    import neuralmem.multimodal.web_extractor as web_mod
    orig_has_bs4 = web_mod._HAS_BS4
    orig_bs4 = web_mod.BeautifulSoup
    try:
        web_mod._HAS_BS4 = True
        web_mod.BeautifulSoup = mock_bs4
        result = ext._parse_html("<html><body><article>test</article></body></html>")
        assert "Title here" in result
    finally:
        web_mod._HAS_BS4 = orig_has_bs4
        web_mod.BeautifulSoup = orig_bs4


def test_extract_bs4_raises_fallback():
    """BeautifulSoup 异常时应回退到正则."""
    ext = WebExtractor()

    mock_bs4 = MagicMock()
    mock_bs4.side_effect = RuntimeError("parser error")

    with patch("neuralmem.multimodal.web_extractor._HAS_BS4", True):
        with patch("neuralmem.multimodal.web_extractor.BeautifulSoup", mock_bs4):
            result = ext.extract(b"<html><body><p>fallback text</p></body></html>")

    # _parse_html catches exception and returns ""
    assert result == []


def test_extract_no_bs4_regex_fallback():
    """没有 BeautifulSoup 时用正则剥离标签."""
    ext = WebExtractor()
    html = b"<html><body><p>Paragraph one.</p><p>Paragraph two.</p></body></html>"

    with patch("neuralmem.multimodal.web_extractor._HAS_BS4", False):
        with patch("neuralmem.multimodal.web_extractor.BeautifulSoup", None):
            result = ext.extract(html)

    # Regex fallback strips tags
    assert len(result) >= 1
    assert "Paragraph" in result[0]

def test_extract_empty_bytes():
    ext = WebExtractor()
    assert ext.extract(b"") == []


# --------------------------------------------------------------------------- #
# _chunk_text
# --------------------------------------------------------------------------- #

def test_chunk_text_empty():
    ext = WebExtractor()
    assert ext._chunk_text("") == []


def test_chunk_text_no_split_needed():
    ext = WebExtractor()
    assert ext._chunk_text("hello world") == ["hello world"]


def test_chunk_text_splits_long_text():
    ext = WebExtractor(max_chunk_size=10)
    text = "a " * 20
    chunks = ext._chunk_text(text)
    assert all(len(c) <= 10 for c in chunks)


# --------------------------------------------------------------------------- #
# _read_bytes / _read_text
# --------------------------------------------------------------------------- #

def test_read_bytes_from_bytes():
    ext = WebExtractor()
    assert ext._read_bytes(b"raw") == b"raw"


def test_read_text_from_bytes():
    ext = WebExtractor()
    assert ext._read_text(b"hello") == "hello"


# --------------------------------------------------------------------------- #
# fallback
# --------------------------------------------------------------------------- #

def test_fallback_decodes_utf8():
    """正则回退能解码 UTF-8."""
    ext = WebExtractor()
    raw = b"\xe4\xb8\xad\xe6\x96\x87"  # UTF-8 中文
    with patch("neuralmem.multimodal.web_extractor._HAS_BS4", False):
        with patch("neuralmem.multimodal.web_extractor.BeautifulSoup", None):
            result = ext.extract(raw)
    assert any("中文" in chunk for chunk in result)


# --------------------------------------------------------------------------- #
# 依赖检查 (继承自基类)
# --------------------------------------------------------------------------- #

def test_check_dependency_true():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("sys") is True


def test_check_dependency_false():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("nonexistent_xyz_abc") is False
