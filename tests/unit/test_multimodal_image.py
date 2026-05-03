"""ImageExtractor 单元测试 — 全部使用 mock, 不依赖真实 Pillow/pytesseract."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.multimodal.image_extractor import ImageExtractor


# --------------------------------------------------------------------------- #
# 基础行为
# --------------------------------------------------------------------------- #

def test_image_extractor_inherits_base():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    ext = ImageExtractor()
    assert isinstance(ext, BaseMultimodalExtractor)


def test_image_extractor_default_chunk_size():
    ext = ImageExtractor()
    assert ext.max_chunk_size == 4096


def test_image_extractor_custom_chunk_size():
    ext = ImageExtractor(max_chunk_size=512)
    assert ext.max_chunk_size == 512


# --------------------------------------------------------------------------- #
# extract() — mock easyocr 成功路径
# --------------------------------------------------------------------------- #

def test_extract_with_mock_easyocr():
    """模拟 easyocr 成功提取文本."""
    ext = ImageExtractor()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["OCR", "extracted", "text"]

    mock_easyocr = MagicMock()
    mock_easyocr.Reader.return_value = mock_reader

    mock_pil = MagicMock()
    mock_img = MagicMock()
    mock_pil.open.return_value = mock_img

    with patch("neuralmem.multimodal.image_extractor._HAS_EASYOCR", True):
        with patch("neuralmem.multimodal.image_extractor.easyocr", mock_easyocr):
            with patch("neuralmem.multimodal.image_extractor.PILImage", mock_pil):
                with patch("neuralmem.multimodal.image_extractor._HAS_TESSERACT", False):
                    result = ext.extract(b"fake image bytes")

    assert isinstance(result, list)
    assert any("OCR" in chunk for chunk in result)
    mock_easyocr.Reader.assert_called_once()


# --------------------------------------------------------------------------- #
# extract() — mock pytesseract 成功路径
# --------------------------------------------------------------------------- #

def test_extract_with_mock_pytesseract():
    """模拟 pytesseract 成功提取文本."""
    ext = ImageExtractor()

    mock_pil = MagicMock()
    mock_img = MagicMock()
    mock_pil.open.return_value = mock_img

    mock_tesseract = MagicMock()
    mock_tesseract.image_to_string.return_value = "Tesseract extracted text."

    with patch("neuralmem.multimodal.image_extractor._HAS_EASYOCR", False):
        with patch("neuralmem.multimodal.image_extractor._HAS_TESSERACT", True):
            with patch("neuralmem.multimodal.image_extractor.PILImage", mock_pil):
                with patch("neuralmem.multimodal.image_extractor.pytesseract", mock_tesseract):
                    result = ext.extract(b"fake image bytes")

    assert isinstance(result, list)
    assert any("Tesseract" in chunk for chunk in result)
    mock_tesseract.image_to_string.assert_called_once()


# --------------------------------------------------------------------------- #
# extract() — OCR 异常回退
# --------------------------------------------------------------------------- #

def test_extract_ocr_raises_fallback():
    """OCR 异常时应回退到空结果 (无 OCR 库时返回空)."""
    ext = ImageExtractor()

    with patch("neuralmem.multimodal.image_extractor._HAS_EASYOCR", False):
        with patch("neuralmem.multimodal.image_extractor._HAS_TESSERACT", False):
            result = ext.extract(b"bad image")
            assert result == []


# --------------------------------------------------------------------------- #
# extract() — 空输入
# --------------------------------------------------------------------------- #

def test_extract_empty_bytes():
    ext = ImageExtractor()
    assert ext.extract(b"") == []


# --------------------------------------------------------------------------- #
# _chunk_text
# --------------------------------------------------------------------------- #

def test_chunk_text_empty():
    ext = ImageExtractor()
    assert ext._chunk_text("") == []


def test_chunk_text_no_split_needed():
    ext = ImageExtractor()
    assert ext._chunk_text("hello world") == ["hello world"]


def test_chunk_text_splits_long_text():
    ext = ImageExtractor(max_chunk_size=10)
    text = "a " * 20
    chunks = ext._chunk_text(text)
    assert all(len(c) <= 10 for c in chunks)
    # 所有字符应被保留
    assert " ".join(chunks).replace("  ", " ").strip() == text.strip()


# --------------------------------------------------------------------------- #
# _read_bytes / _read_text
# --------------------------------------------------------------------------- #

def test_read_bytes_from_bytes():
    ext = ImageExtractor()
    assert ext._read_bytes(b"raw") == b"raw"


def test_read_text_from_bytes():
    ext = ImageExtractor()
    assert ext._read_text(b"hello") == "hello"


# --------------------------------------------------------------------------- #
# 依赖检查 (继承自基类)
# --------------------------------------------------------------------------- #

def test_check_dependency_true():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("sys") is True


def test_check_dependency_false():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("nonexistent_xyz_456") is False
