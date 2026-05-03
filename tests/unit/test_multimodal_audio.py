"""AudioExtractor 单元测试 — 全部使用 mock, 不依赖真实 speech_recognition / whisper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.multimodal.audio_extractor import AudioExtractor


# --------------------------------------------------------------------------- #
# 基础行为
# --------------------------------------------------------------------------- #

def test_audio_extractor_inherits_base():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    ext = AudioExtractor()
    assert isinstance(ext, BaseMultimodalExtractor)


def test_audio_extractor_default_chunk_size():
    ext = AudioExtractor()
    assert ext.max_chunk_size == 4096


def test_audio_extractor_custom_chunk_size():
    ext = AudioExtractor(max_chunk_size=256)
    assert ext.max_chunk_size == 256


# --------------------------------------------------------------------------- #
# extract() — mock openai 成功路径
# --------------------------------------------------------------------------- #

def test_extract_with_mock_openai():
    """模拟 openai 成功提取文本."""
    ext = AudioExtractor(api_key="fake-key")

    mock_transcript = MagicMock()
    mock_transcript.text = "Hello world this is audio."

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_transcript

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client

    with patch("neuralmem.multimodal.audio_extractor._HAS_OPENAI", True):
        with patch("neuralmem.multimodal.audio_extractor.openai", mock_openai):
            with patch("builtins.open", MagicMock()):
                result = ext.extract(b"fake audio bytes")

    assert isinstance(result, list)
    assert any("Hello world" in chunk for chunk in result)


# --------------------------------------------------------------------------- #
# extract() — mock faster-whisper 成功路径
# --------------------------------------------------------------------------- #

def test_extract_with_mock_faster_whisper():
    """模拟 faster-whisper 成功提取文本."""
    ext = AudioExtractor()

    mock_segment = MagicMock()
    mock_segment.text = "faster whisper text"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)

    mock_faster = MagicMock()
    mock_faster.WhisperModel.return_value = mock_model

    with patch("neuralmem.multimodal.audio_extractor._HAS_OPENAI", False):
        with patch("neuralmem.multimodal.audio_extractor._HAS_FASTER", True):
            with patch("neuralmem.multimodal.audio_extractor.faster_whisper", mock_faster):
                result = ext.extract(b"fake audio bytes")

    assert isinstance(result, list)
    assert any("faster whisper" in chunk for chunk in result)


# --------------------------------------------------------------------------- #
# extract() — mock whisper 成功路径
# --------------------------------------------------------------------------- #

def test_extract_with_mock_whisper():
    """模拟 whisper 成功提取文本."""
    ext = AudioExtractor()

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "whisper text result"}

    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    with patch("neuralmem.multimodal.audio_extractor._HAS_OPENAI", False):
        with patch("neuralmem.multimodal.audio_extractor._HAS_FASTER", False):
            with patch("neuralmem.multimodal.audio_extractor._HAS_WHISPER", True):
                with patch("neuralmem.multimodal.audio_extractor.whisper", mock_whisper):
                    result = ext.extract(b"fake audio bytes")

    assert isinstance(result, list)
    assert any("whisper text" in chunk for chunk in result)


# --------------------------------------------------------------------------- #
# extract() — 无后端时返回空
# --------------------------------------------------------------------------- #

def test_extract_no_backend():
    """无可用后端时应返回空列表."""
    ext = AudioExtractor()

    with patch("neuralmem.multimodal.audio_extractor._HAS_OPENAI", False):
        with patch("neuralmem.multimodal.audio_extractor._HAS_FASTER", False):
            with patch("neuralmem.multimodal.audio_extractor._HAS_WHISPER", False):
                result = ext.extract(b"fake audio bytes")
                assert result == []


# --------------------------------------------------------------------------- #
# extract() — 空输入
# --------------------------------------------------------------------------- #

def test_extract_empty_bytes():
    ext = AudioExtractor()
    assert ext.extract(b"") == []


# --------------------------------------------------------------------------- #
# _chunk_text
# --------------------------------------------------------------------------- #

def test_chunk_text_empty():
    ext = AudioExtractor()
    assert ext._chunk_text("") == []


def test_chunk_text_no_split_needed():
    ext = AudioExtractor()
    assert ext._chunk_text("short audio transcript") == ["short audio transcript"]


def test_chunk_text_splits_long_text():
    ext = AudioExtractor(max_chunk_size=10)
    text = "word " * 20
    chunks = ext._chunk_text(text)
    assert all(len(c) <= 10 for c in chunks)


# --------------------------------------------------------------------------- #
# _read_bytes / _read_text
# --------------------------------------------------------------------------- #

def test_read_bytes_from_bytes():
    ext = AudioExtractor()
    assert ext._read_bytes(b"raw") == b"raw"


def test_read_text_from_bytes():
    ext = AudioExtractor()
    assert ext._read_text(b"hello") == "hello"


# --------------------------------------------------------------------------- #
# 依赖检查 (继承自基类)
# --------------------------------------------------------------------------- #

def test_check_dependency_true():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("os") is True


def test_check_dependency_false():
    from neuralmem.multimodal.base import BaseMultimodalExtractor
    assert BaseMultimodalExtractor._check_dependency("nonexistent_xyz_789") is False
