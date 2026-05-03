"""NeuralMem multimodal extraction module.

Provides extractors for PDF, image, audio, video, web, and office documents.
All extractors inherit from BaseMultimodalExtractor and return a list of text
chunks via the unified ``extract(path_or_bytes) -> list[str]`` interface.
"""
from __future__ import annotations

from neuralmem.multimodal.base import BaseMultimodalExtractor, ExtractionResult
from neuralmem.multimodal.pdf_extractor import PDFExtractor
from neuralmem.multimodal.image_extractor import ImageExtractor
from neuralmem.multimodal.audio_extractor import AudioExtractor
from neuralmem.multimodal.video_extractor import VideoExtractor
from neuralmem.multimodal.web_extractor import WebExtractor
from neuralmem.multimodal.office_extractor import OfficeExtractor

__all__ = [
    "BaseMultimodalExtractor",
    "ExtractionResult",
    "PDFExtractor",
    "ImageExtractor",
    "AudioExtractor",
    "VideoExtractor",
    "WebExtractor",
    "OfficeExtractor",
]
