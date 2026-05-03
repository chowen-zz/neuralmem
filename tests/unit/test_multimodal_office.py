"""OfficeExtractor 单元测试 — 全部使用 mock, 不依赖真实 python-docx."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.multimodal.office_extractor import OfficeExtractor


# --------------------------------------------------------------------------- #
# 基础行为
# --------------------------------------------------------------------------- #

def test_office_extractor_inherits_base():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    ext = OfficeExtractor()
    assert isinstance(ext, BaseMultimodalExtractor)


def test_office_extractor_default_chunk_size():
    ext = OfficeExtractor()
    assert ext.max_chunk_size == 4096


def test_office_extractor_custom_chunk_size():
    ext = OfficeExtractor(max_chunk_size=512)
    assert ext.max_chunk_size == 512


# --------------------------------------------------------------------------- #
# extract() — mock python-docx 成功路径
# --------------------------------------------------------------------------- #

def test_extract_with_mock_docx():
    """模拟 python-docx 成功提取段落文本."""
    ext = OfficeExtractor()

    mock_para1 = MagicMock()
    mock_para1.text = "First paragraph."
    mock_para2 = MagicMock()
    mock_para2.text = "Second paragraph."

    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para1, mock_para2]

    # 直接 patch 模块级别的 _HAS_DOCX 和 docx 变量
    import neuralmem.multimodal.office_extractor as office_mod
    mock_docx_mod = MagicMock()
    mock_docx_mod.Document.return_value = mock_doc

    orig_has_docx = office_mod._HAS_DOCX
    orig_docx = office_mod.docx
    try:
        office_mod._HAS_DOCX = True
        office_mod.docx = mock_docx_mod
        result = ext.extract(b"fake docx bytes")
    finally:
        office_mod._HAS_DOCX = orig_has_docx
        office_mod.docx = orig_docx

    assert isinstance(result, list)
    assert any("First paragraph." in chunk for chunk in result)
    assert any("Second paragraph." in chunk for chunk in result)


# --------------------------------------------------------------------------- #
# extract() — python-docx 异常回退
# --------------------------------------------------------------------------- #

def test_extract_docx_raises_fallback():
    """python-docx 异常时应回退到 fallback."""
    ext = OfficeExtractor()

    import neuralmem.multimodal.office_extractor as office_mod
    mock_docx_mod = MagicMock()
    mock_docx_mod.Document.side_effect = RuntimeError("corrupt docx")

    orig_has_docx = office_mod._HAS_DOCX
    orig_docx = office_mod.docx
    try:
        office_mod._HAS_DOCX = True
        office_mod.docx = mock_docx_mod
        with patch.object(ext, "_fallback", return_value=["fallback"]) as mock_fallback:
            result = ext.extract(b"bad docx")
            assert result == ["fallback"]
            mock_fallback.assert_called_once()
    finally:
        office_mod._HAS_DOCX = orig_has_docx
        office_mod.docx = orig_docx


# --------------------------------------------------------------------------- #
# extract() — 空输入
# --------------------------------------------------------------------------- #

def test_extract_empty_bytes():
    ext = OfficeExtractor()
    assert ext.extract(b"") == []


# --------------------------------------------------------------------------- #
# _chunk_text
# --------------------------------------------------------------------------- #

def test_chunk_text_empty():
    ext = OfficeExtractor()
    assert ext._chunk_text("") == []


def test_chunk_text_no_split_needed():
    ext = OfficeExtractor()
    assert ext._chunk_text("hello world") == ["hello world"]


def test_chunk_text_splits_long_text():
    ext = OfficeExtractor(max_chunk_size=10)
    text = "a " * 20
    chunks = ext._chunk_text(text)
    assert all(len(c) <= 10 for c in chunks)


# --------------------------------------------------------------------------- #
# _read_bytes / _read_text
# --------------------------------------------------------------------------- #

def test_read_bytes_from_bytes():
    ext = OfficeExtractor()
    assert ext._read_bytes(b"raw") == b"raw"


def test_read_text_from_bytes():
    ext = OfficeExtractor()
    assert ext._read_text(b"hello") == "hello"


# --------------------------------------------------------------------------- #
# fallback
# --------------------------------------------------------------------------- #

def test_fallback_decodes_utf8():
    ext = OfficeExtractor()
    raw = b"\xe4\xb8\xad\xe6\x96\x87"  # UTF-8 中文
    result = ext._fallback(raw)
    assert any("中文" in chunk for chunk in result)


# --------------------------------------------------------------------------- #
# 依赖检查 (继承自基类)
# --------------------------------------------------------------------------- #

def test_check_dependency_true():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("os") is True


def test_check_dependency_false():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("nonexistent_xyz_def") is False
