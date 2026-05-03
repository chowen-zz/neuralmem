"""Image extractor — OCR (easyocr / pytesseract) and optional CLIP encoding.

The extractor attempts the following pipeline:

1. **OCR** — ``easyocr`` is preferred; falls back to ``pytesseract``.
2. **CLIP** — if ``transformers`` + ``torch`` are available, the image is
   also encoded with a CLIP vision model and a textual description of the
   embedding is appended (this is useful for semantic retrieval).

If no OCR library is installed the extractor falls back to returning an
empty list (the caller may supply external text).
"""
from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING

from neuralmem.multimodal.base import BaseMultimodalExtractor, ExtractionResult

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

# Graceful dependency checks ------------------------------------------------
try:
    import easyocr  # type: ignore[import-untyped]

    _HAS_EASYOCR = True
except ImportError:
    easyocr = None  # type: ignore[misc]
    _HAS_EASYOCR = False

try:
    import pytesseract  # type: ignore[import-untyped]
    from PIL import Image as PILImage  # type: ignore[import-untyped]

    _HAS_TESSERACT = True
except ImportError:
    pytesseract = None  # type: ignore[misc]
    PILImage = None  # type: ignore[misc]
    _HAS_TESSERACT = False

try:
    import torch  # type: ignore[import-untyped]
    from transformers import (  # type: ignore[import-untyped]
        CLIPImageProcessor,
        CLIPModel,
        CLIPTokenizer,
    )

    _HAS_CLIP = True
except ImportError:
    torch = None  # type: ignore[misc]
    CLIPModel = None  # type: ignore[misc]
    CLIPImageProcessor = None  # type: ignore[misc]
    CLIPTokenizer = None  # type: ignore[misc]
    _HAS_CLIP = False


class ImageExtractor(BaseMultimodalExtractor):
    """Extract text from images via OCR and optional CLIP semantic encoding.

    Args:
        max_chunk_size: Maximum characters per returned chunk.
        ocr_languages: Language list passed to the OCR engine. Defaults to
            ``["en"]``.
        clip_model_name: HuggingFace model id for CLIP. Defaults to
            ``"openai/clip-vit-base-patch32"``.
        use_clip: Whether to append a CLIP-derived description. Defaults to
            ``True`` when the dependencies are available.
    """

    def __init__(
        self,
        max_chunk_size: int = 4096,
        *,
        ocr_languages: Sequence[str] | None = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        use_clip: bool = True,
    ) -> None:
        super().__init__(max_chunk_size)
        self.ocr_languages = list(ocr_languages) if ocr_languages else ["en"]
        self.clip_model_name = clip_model_name
        self._use_clip = use_clip and _HAS_CLIP

        if not _HAS_EASYOCR and not _HAS_TESSERACT:
            _logger.warning(
                "Neither easyocr nor pytesseract is installed. "
                "ImageExtractor will return empty results."
            )

        # Lazy-loaded CLIP resources
        self._clip_model: object | None = None
        self._clip_processor: object | None = None
        self._clip_tokenizer: object | None = None

    def extract(self, path_or_bytes: str | os.PathLike[str] | bytes) -> ExtractionResult:
        """Extract text from an image file or bytes buffer.

        Args:
            path_or_bytes: Path to an image file or raw image bytes.

        Returns:
            List of text chunks. Contains OCR text and, when CLIP is enabled,
            a synthetic description of the image embedding.
        """
        raw = self._read_bytes(path_or_bytes)
        if not raw:
            return []

        chunks: list[str] = []

        # 1. OCR ---------------------------------------------------------
        ocr_text = self._run_ocr(raw)
        if ocr_text:
            chunks.extend(self._chunk_text(ocr_text))

        # 2. CLIP semantic description -----------------------------------
        if self._use_clip:
            clip_desc = self._run_clip(raw)
            if clip_desc:
                chunks.extend(self._chunk_text(clip_desc))

        return chunks

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #

    def _run_ocr(self, data: bytes) -> str:
        """Run OCR and return concatenated text."""
        if _HAS_EASYOCR and easyocr is not None:
            return self._ocr_easyocr(data)
        if _HAS_TESSERACT and pytesseract is not None and PILImage is not None:
            return self._ocr_pytesseract(data)
        return ""

    def _ocr_easyocr(self, data: bytes) -> str:
        """Use EasyOCR to read text from image bytes."""
        try:
            reader = easyocr.Reader(self.ocr_languages, gpu=False)
            img = PILImage.open(io.BytesIO(data)) if PILImage else None
            # easyocr can accept numpy array or path; we use the bytes path
            # workaround: write to a temp file when PIL is unavailable
            if img is None:
                return ""
            import numpy as np  # type: ignore[import-untyped]

            result = reader.readtext(np.array(img), detail=0)
            return "\n".join(result)
        except Exception as exc:
            _logger.warning("easyocr failed: %s", exc)
            return ""

    def _ocr_pytesseract(self, data: bytes) -> str:
        """Use pytesseract to read text from image bytes."""
        try:
            if PILImage is None:
                return ""
            img = PILImage.open(io.BytesIO(data))
            text = pytesseract.image_to_string(img, lang="+".join(self.ocr_languages))
            return text
        except Exception as exc:
            _logger.warning("pytesseract failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    # CLIP
    # ------------------------------------------------------------------ #

    def _run_clip(self, data: bytes) -> str:
        """Encode the image with CLIP and return a synthetic description."""
        if not _HAS_CLIP or torch is None or CLIPModel is None:
            return ""

        try:
            model, processor, tokenizer = self._load_clip()
            if model is None or processor is None:
                return ""

            if PILImage is None:
                return ""
            img = PILImage.open(io.BytesIO(data))
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            # Synthetic description: report embedding shape / norm
            vec = image_features[0].cpu().numpy()
            desc = (
                f"[CLIP image embedding] shape={vec.shape}, "
                f"norm={float(torch.norm(image_features[0]).item()):.4f}"
            )
            return desc
        except Exception as exc:
            _logger.warning("CLIP encoding failed: %s", exc)
            return ""

    def _load_clip(self) -> tuple[object, object, object]:
        """Lazy-load CLIP model, processor, and tokenizer."""
        if self._clip_model is not None:
            return (self._clip_model, self._clip_processor, self._clip_tokenizer)

        if CLIPModel is None or CLIPImageProcessor is None or CLIPTokenizer is None:
            raise RuntimeError("CLIP dependencies are not available")

        _logger.info("Loading CLIP model %r ...", self.clip_model_name)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self._clip_processor = CLIPImageProcessor.from_pretrained(self.clip_model_name)
        self._clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
        return (self._clip_model, self._clip_processor, self._clip_tokenizer)
