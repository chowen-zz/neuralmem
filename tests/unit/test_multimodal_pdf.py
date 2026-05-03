"""PDFExtractor 单元测试 — 全部使用 mock, 不依赖真实 PyMuPDF."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.multimodal.pdf_extractor import PDFExtractor


# --------------------------------------------------------------------------- #
# 基础行为
# --------------------------------------------------------------------------- #

def test_pdf_extractor_inherits_base():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    ext = PDFExtractor()
    assert isinstance(ext, BaseMultimodalExtractor)


def test_pdf_extractor_default_chunk_size():
    ext = PDFExtractor()
    assert ext.max_chunk_size == 4096


def test_pdf_extractor_custom_chunk_size():
    ext = PDFExtractor(max_chunk_size=1024)
    assert ext.max_chunk_size == 1024


# --------------------------------------------------------------------------- #
# extract() 路径 — 使用 mock fitz
# --------------------------------------------------------------------------- #

def test_extract_with_mock_fitz():
    """模拟 fitz 成功提取两页文本."""
    ext = PDFExtractor()

    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page one content."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page two content."

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=2)
    mock_doc.load_page.side_effect = [mock_page1, mock_page2]

    # 模拟 fitz 模块级别变量
    mock_fitz_module = MagicMock()
    mock_fitz_module.open.return_value = mock_doc

    with patch("neuralmem.multimodal.pdf_extractor._HAS_FITZ", True):
        with patch("neuralmem.multimodal.pdf_extractor.fitz", mock_fitz_module):
            result = ext.extract(b"fake pdf bytes")

    assert isinstance(result, list)
    assert len(result) >= 1
    assert "Page one content." in " ".join(result)


def test_extract_empty_bytes():
    ext = PDFExtractor()
    assert ext.extract(b"") == []


# --------------------------------------------------------------------------- #
# _chunk_text 边界
# --------------------------------------------------------------------------- #

def test_chunk_text_empty():
    ext = PDFExtractor()
    assert ext._chunk_text("") == []


def test_chunk_text_small():
    ext = PDFExtractor()
    assert ext._chunk_text("hello") == ["hello"]


def test_chunk_text_splits_at_boundary():
    ext = PDFExtractor(max_chunk_size=10)
    text = "a" * 5 + "\n\n" + "b" * 5 + "\n\n" + "c" * 5
    chunks = ext._chunk_text(text)
    assert all(len(c) <= 10 for c in chunks)


# --------------------------------------------------------------------------- #
# _read_bytes / _read_text
# --------------------------------------------------------------------------- #

def test_read_bytes_from_bytes():
    ext = PDFExtractor()
    assert ext._read_bytes(b"raw") == b"raw"


def test_read_text_from_bytes():
    ext = PDFExtractor()
    assert ext._read_text(b"hello") == "hello"


# --------------------------------------------------------------------------- #
# fallback 路径
# --------------------------------------------------------------------------- #

def test_fallback_decodes_bytes():
    ext = PDFExtractor()
    raw = b"\xe4\xb8\xad\xe6\x96\x87"  # UTF-8 中文
    result = ext._fallback(raw)
    assert any("中文" in chunk for chunk in result)


# --------------------------------------------------------------------------- #
# fitz 异常处理
# --------------------------------------------------------------------------- #

def test_extract_fitz_open_raises():
    """fitz.open 抛出异常时应回退到 fallback."""
    ext = PDFExtractor()

    mock_fitz_module = MagicMock()
    mock_fitz_module.open.side_effect = RuntimeError("corrupted pdf")

    with patch("neuralmem.multimodal.pdf_extractor._HAS_FITZ", True):
        with patch("neuralmem.multimodal.pdf_extractor.fitz", mock_fitz_module):
            with patch.object(ext, "_fallback", return_value=["fallback"]) as mock_fallback:
                result = ext.extract(b"bad pdf")
                assert result == ["fallback"]
                mock_fallback.assert_called_once()


def test_extract_fitz_page_raises():
    """fitz 单页异常不应导致整体失败."""
    ext = PDFExtractor()

    mock_page = MagicMock()
    mock_page.get_text.side_effect = RuntimeError("page error")

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=1)
    mock_doc.load_page.return_value = mock_page

    mock_fitz_module = MagicMock()
    mock_fitz_module.open.return_value = mock_doc

    with patch("neuralmem.multimodal.pdf_extractor._HAS_FITZ", True):
        with patch("neuralmem.multimodal.pdf_extractor.fitz", mock_fitz_module):
            result = ext.extract(b"pdf with bad page")
            # 页异常被捕获后 fallback 触发
            assert isinstance(result, list)


# --------------------------------------------------------------------------- #
# 依赖检查
# --------------------------------------------------------------------------- #

def test_check_dependency_true():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("os") is True


def test_check_dependency_false():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("nonexistent_xyz_123") is False


# --------------------------------------------------------------------------- #
# _optional_import
# --------------------------------------------------------------------------- #

def test_optional_import_success():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    result = BaseMultimodalExtractor._optional_import("os", ["path"])
    assert result is not None
    assert "path" in result


def test_optional_import_failure():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    result = BaseMultimodalExtractor._optional_import("nonexistent_xyz_123", ["foo"])
    assert result is None
