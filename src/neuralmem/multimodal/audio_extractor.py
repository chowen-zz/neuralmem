"""Audio extractor — Whisper transcription (OpenAI / faster-whisper / whisper).

The extractor attempts transcription in the following priority order:

1. ``openai`` client (requires API key).
2. ``faster-whisper`` (local, fast).
3. Original ``whisper`` (local, OpenAI reference implementation).

If none are available the extractor returns an empty list.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import TYPE_CHECKING

from neuralmem.multimodal.base import BaseMultimodalExtractor, ExtractionResult

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

# Graceful dependency checks ------------------------------------------------
try:
    import openai  # type: ignore[import-untyped]

    _HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore[misc]
    _HAS_OPENAI = False

try:
    import faster_whisper  # type: ignore[import-untyped]

    _HAS_FASTER = True
except ImportError:
    faster_whisper = None  # type: ignore[misc]
    _HAS_FASTER = False

try:
    import whisper  # type: ignore[import-untyped]

    _HAS_WHISPER = True
except ImportError:
    whisper = None  # type: ignore[misc]
    _HAS_WHISPER = False


class AudioExtractor(BaseMultimodalExtractor):
    """Transcribe audio files to text using Whisper.

    Args:
        max_chunk_size: Maximum characters per returned chunk.
        model_size: Whisper model size (``tiny``, ``base``, ``small``,
            ``medium``, ``large``). Used by local backends.
        api_key: OpenAI API key for the cloud Whisper API. If ``None`` the
            ``OPENAI_API_KEY`` environment variable is consulted.
        language: ISO-639-1 language code (e.g. ``"en"``). ``None`` means
            auto-detect.
    """

    def __init__(
        self,
        max_chunk_size: int = 4096,
        *,
        model_size: str = "base",
        api_key: str | None = None,
        language: str | None = None,
    ) -> None:
        super().__init__(max_chunk_size)
        self.model_size = model_size
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.language = language

        if not _HAS_OPENAI and not _HAS_FASTER and not _HAS_WHISPER:
            _logger.warning(
                "No Whisper backend is installed. "
                "AudioExtractor will return empty results."
            )

        # Lazy-loaded local model instances
        self._faster_model: object | None = None
        self._whisper_model: object | None = None

    def extract(self, path_or_bytes: str | os.PathLike[str] | bytes) -> ExtractionResult:
        """Transcribe an audio file or bytes buffer to text.

        Args:
            path_or_bytes: Path to an audio file or raw audio bytes.

        Returns:
            List of text chunks (usually a single transcript).
        """
        raw = self._read_bytes(path_or_bytes)
        if not raw:
            return []

        # Write bytes to a temp file because most whisper libs expect a path
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            text = self._transcribe(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return self._chunk_text(text) if text else []

    # ------------------------------------------------------------------ #
    # Transcription dispatch
    # ------------------------------------------------------------------ #

    def _transcribe(self, audio_path: str) -> str:
        """Try each backend in priority order."""
        if _HAS_OPENAI and openai is not None and self.api_key:
            result = self._transcribe_openai(audio_path)
            if result:
                return result

        if _HAS_FASTER and faster_whisper is not None:
            result = self._transcribe_faster(audio_path)
            if result:
                return result

        if _HAS_WHISPER and whisper is not None:
            result = self._transcribe_whisper(audio_path)
            if result:
                return result

        _logger.warning("All Whisper backends failed or are unavailable.")
        return ""

    # ------------------------------------------------------------------ #
    # OpenAI API
    # ------------------------------------------------------------------ #

    def _transcribe_openai(self, audio_path: str) -> str:
        """Transcribe via OpenAI Whisper API."""
        try:
            client = openai.OpenAI(api_key=self.api_key)
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.language,
                )
            return transcript.text or ""
        except Exception as exc:
            _logger.warning("OpenAI Whisper API failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    # faster-whisper (local)
    # ------------------------------------------------------------------ #

    def _transcribe_faster(self, audio_path: str) -> str:
        """Transcribe via faster-whisper (local CTranslate2)."""
        try:
            model = self._load_faster_model()
            segments, _info = model.transcribe(
                audio_path,
                language=self.language,
                beam_size=5,
            )
            texts = [segment.text for segment in segments]
            return " ".join(texts)
        except Exception as exc:
            _logger.warning("faster-whisper failed: %s", exc)
            return ""

    def _load_faster_model(self) -> object:
        """Lazy-load the faster-whisper model."""
        if self._faster_model is None:
            _logger.info("Loading faster-whisper model %r ...", self.model_size)
            self._faster_model = faster_whisper.WhisperModel(self.model_size)
        return self._faster_model

    # ------------------------------------------------------------------ #
    # Original whisper (local)
    # ------------------------------------------------------------------ #

    def _transcribe_whisper(self, audio_path: str) -> str:
        """Transcribe via original OpenAI whisper."""
        try:
            model = self._load_whisper_model()
            result = model.transcribe(audio_path, language=self.language)
            return result.get("text", "")
        except Exception as exc:
            _logger.warning("whisper failed: %s", exc)
            return ""

    def _load_whisper_model(self) -> object:
        """Lazy-load the original whisper model."""
        if self._whisper_model is None:
            _logger.info("Loading whisper model %r ...", self.model_size)
            self._whisper_model = whisper.load_model(self.model_size)
        return self._whisper_model
