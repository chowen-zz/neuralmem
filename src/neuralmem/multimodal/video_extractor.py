"""Video extractor — key-frame extraction + audio transcription.

The pipeline is:

1. **Keyframes** — sample frames at a configurable interval, run OCR on each
   frame (via ``ImageExtractor``), and collect text.
2. **Audio track** — extract audio with ``ffmpeg`` (or ``moviepy``), then
   transcribe with ``AudioExtractor``.

Either step may be disabled or may fail gracefully; the extractor returns
whatever text could be recovered.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING

from neuralmem.multimodal.base import BaseMultimodalExtractor, ExtractionResult
from neuralmem.multimodal.image_extractor import ImageExtractor
from neuralmem.multimodal.audio_extractor import AudioExtractor

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

# Graceful dependency checks ------------------------------------------------
try:
    import cv2  # type: ignore[import-untyped]

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[misc]
    _HAS_CV2 = False

try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[misc]
    _HAS_NUMPY = False

try:
    from PIL import Image as PILImage  # type: ignore[import-untyped]

    _HAS_PIL = True
except ImportError:
    PILImage = None  # type: ignore[misc]
    _HAS_PIL = False


class VideoExtractor(BaseMultimodalExtractor):
    """Extract text from video files via key-frame OCR and audio transcription.

    Args:
        max_chunk_size: Maximum characters per returned chunk.
        frame_interval_sec: Sample a frame every N seconds. Defaults to ``5``.
        enable_ocr: Whether to run OCR on sampled frames. Defaults to ``True``.
        enable_audio: Whether to extract and transcribe the audio track.
            Defaults to ``True``.
        image_extractor: Optional ``ImageExtractor`` instance. If ``None`` a
            default instance is created.
        audio_extractor: Optional ``AudioExtractor`` instance. If ``None`` a
            default instance is created.
    """

    def __init__(
        self,
        max_chunk_size: int = 4096,
        *,
        frame_interval_sec: float = 5.0,
        enable_ocr: bool = True,
        enable_audio: bool = True,
        image_extractor: ImageExtractor | None = None,
        audio_extractor: AudioExtractor | None = None,
    ) -> None:
        super().__init__(max_chunk_size)
        self.frame_interval_sec = frame_interval_sec
        self.enable_ocr = enable_ocr
        self.enable_audio = enable_audio
        self._image_extractor = image_extractor or ImageExtractor()
        self._audio_extractor = audio_extractor or AudioExtractor()

        if not _HAS_CV2:
            _logger.warning(
                "opencv-python (cv2) is not installed. "
                "VideoExtractor key-frame OCR will be disabled."
            )
        if not _HAS_PIL:
            _logger.warning(
                "Pillow is not installed. "
                "VideoExtractor key-frame OCR may be degraded."
            )

    def extract(self, path_or_bytes: str | os.PathLike[str] | bytes) -> ExtractionResult:
        """Extract text from a video file or bytes buffer.

        Args:
            path_or_bytes: Path to a video file or raw video bytes.

        Returns:
            List of text chunks from OCR + audio transcription.
        """
        raw = self._read_bytes(path_or_bytes)
        if not raw:
            return []

        # Write bytes to a temp file because cv2/ffmpeg need a path
        with tempfile.NamedTemporaryFile(suffix=".video", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        chunks: list[str] = []
        try:
            if self.enable_ocr:
                ocr_text = self._extract_keyframes(tmp_path)
                if ocr_text:
                    chunks.extend(self._chunk_text(ocr_text))

            if self.enable_audio:
                audio_text = self._extract_audio(tmp_path)
                if audio_text:
                    chunks.extend(self._chunk_text(audio_text))
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return chunks

    # ------------------------------------------------------------------ #
    # Key-frame OCR
    # ------------------------------------------------------------------ #

    def _extract_keyframes(self, video_path: str) -> str:
        """Sample frames and run OCR. Returns concatenated text."""
        if not _HAS_CV2 or cv2 is None:
            return ""

        texts: list[str] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _logger.warning("cv2 could not open video: %s", video_path)
            return ""

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval_frames = max(1, int(self.frame_interval_sec * fps))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval_frames == 0:
                    text = self._ocr_frame(frame)
                    if text:
                        texts.append(text)
                frame_idx += 1
        except Exception as exc:
            _logger.warning("Keyframe extraction failed: %s", exc)
        finally:
            cap.release()

        return "\n".join(texts)

    def _ocr_frame(self, frame: object) -> str:
        """Run OCR on a single OpenCV frame (numpy array)."""
        if not _HAS_PIL or PILImage is None:
            return ""
        try:
            if _HAS_NUMPY and np is not None:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PILImage.fromarray(rgb)
            else:
                return ""
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result = self._image_extractor.extract(buf.getvalue())
            return "\n".join(result)
        except Exception as exc:
            _logger.warning("Frame OCR failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    # Audio extraction + transcription
    # ------------------------------------------------------------------ #

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio track to a temp file and transcribe it."""
        audio_path = self._extract_audio_track(video_path)
        if not audio_path:
            return ""
        try:
            result = self._audio_extractor.extract(audio_path)
            return "\n".join(result)
        except Exception as exc:
            _logger.warning("Audio transcription failed: %s", exc)
            return ""
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    def _extract_audio_track(self, video_path: str) -> str | None:
        """Extract audio from video using ffmpeg. Returns path to audio file."""
        # Prefer ffmpeg subprocess (no extra Python deps)
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_path = tmp.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return audio_path
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            _logger.warning("ffmpeg audio extraction failed: %s", exc)
            try:
                os.remove(audio_path)
            except OSError:
                pass
            return None
