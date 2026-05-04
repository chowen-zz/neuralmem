"""Video scene extractor — scene segmentation, keyframe selection, temporal relations, summarization.

Extends the basic VideoExtractor with scene-level understanding:
- Scene segmentation via visual change detection
- Intelligent keyframe selection per scene
- Temporal relation extraction between scenes
- Video summarization (text + structured scene graph)
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

# Graceful dependency checks --------------------------------------------------
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


@dataclass
class Scene:
    """A detected scene / shot within a video.

    Attributes:
        scene_id: Unique scene identifier.
        start_sec: Start timestamp in seconds.
        end_sec: End timestamp in seconds.
        keyframes: List of selected keyframe timestamps (seconds).
        ocr_text: Concatenated OCR text from keyframes.
        visual_summary: Natural-language summary of the scene.
        metadata: Arbitrary metadata.
    """

    scene_id: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    keyframes: list[float] = field(default_factory=list)
    ocr_text: str = ""
    visual_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalRelation:
    """A temporal relation between two scenes.

    Attributes:
        source_id: Source scene identifier.
        target_id: Target scene identifier.
        relation: Relation type (e.g. ``follows``, ``contains``, ``parallel``).
        confidence: Confidence score 0-1.
    """

    source_id: str
    target_id: str
    relation: str
    confidence: float = 1.0


@dataclass
class VideoSummary:
    """Structured summary of a video.

    Attributes:
        title: Inferred title / topic.
        duration_sec: Total duration in seconds.
        scenes: List of detected scenes.
        temporal_relations: List of temporal relations.
        overall_summary: Natural-language overall summary.
        key_topics: Extracted key topics.
    """

    title: str = ""
    duration_sec: float = 0.0
    scenes: list[Scene] = field(default_factory=list)
    temporal_relations: list[TemporalRelation] = field(default_factory=list)
    overall_summary: str = ""
    key_topics: list[str] = field(default_factory=list)


class SceneSegmenter:
    """Segment a video into scenes using frame-difference histogram analysis.

    Args:
        threshold: Histogram difference threshold for scene cut. Defaults to 0.35.
        min_scene_sec: Minimum scene duration in seconds. Defaults to 1.0.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        min_scene_sec: float = 1.0,
    ) -> None:
        self.threshold = threshold
        self.min_scene_sec = min_scene_sec

    def segment(self, video_path: str) -> list[tuple[float, float]]:
        """Return list of (start_sec, end_sec) scene boundaries.

        Uses HSV histogram difference between consecutive frames to detect
        significant visual changes (scene cuts).
        """
        if not _HAS_CV2 or cv2 is None:
            _logger.warning("cv2 unavailable; returning single-scene fallback.")
            return [(0.0, 0.0)]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _logger.warning("Cannot open video: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        cuts: list[int] = [0]  # frame indices where scenes start
        prev_hist: np.ndarray | None = None

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                hist = self._frame_histogram(frame)
                if prev_hist is not None and hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    if diff > self.threshold:
                        cuts.append(frame_idx)
                prev_hist = hist
                frame_idx += 1
            cuts.append(frame_idx)
        except Exception as exc:
            _logger.warning("Scene segmentation failed: %s", exc)
        finally:
            cap.release()

        # Convert frame indices to seconds and enforce min duration
        boundaries: list[tuple[float, float]] = []
        for i in range(len(cuts) - 1):
            start = cuts[i] / fps
            end = cuts[i + 1] / fps
            if end - start >= self.min_scene_sec:
                boundaries.append((start, end))

        if not boundaries and duration > 0:
            boundaries = [(0.0, duration)]

        return boundaries

    @staticmethod
    def _frame_histogram(frame: object) -> np.ndarray | None:
        """Compute HSV histogram for a frame."""
        if not _HAS_CV2 or cv2 is None:
            return None
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist
        except Exception:
            return None


class KeyframeSelector:
    """Select representative keyframes within a scene.

    Uses frame-quality heuristics (laplacian variance = sharpness) to pick
    the best frame in each sub-window of a scene.

    Args:
        keyframes_per_scene: Number of keyframes to extract per scene. Defaults to 3.
    """

    def __init__(self, keyframes_per_scene: int = 3) -> None:
        self.keyframes_per_scene = keyframes_per_scene

    def select(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> list[float]:
        """Return timestamps (seconds) of selected keyframes."""
        if not _HAS_CV2 or cv2 is None:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        total_frames = end_frame - start_frame

        if total_frames <= 0:
            cap.release()
            return [start_sec]

        window_size = max(1, total_frames // self.keyframes_per_scene)
        keyframes: list[float] = []

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            best_in_window: tuple[float, float] = (0.0, -1.0)  # (timestamp, variance)
            window_count = 0
            local_frame = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                abs_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if abs_frame >= end_frame:
                    break

                ts = abs_frame / fps
                variance = self._sharpness(frame)
                if variance > best_in_window[1]:
                    best_in_window = (ts, variance)

                window_count += 1
                if window_count >= window_size:
                    keyframes.append(best_in_window[0])
                    best_in_window = (0.0, -1.0)
                    window_count = 0
                local_frame += 1

            if best_in_window[1] > 0:
                keyframes.append(best_in_window[0])
        except Exception as exc:
            _logger.warning("Keyframe selection failed: %s", exc)
        finally:
            cap.release()

        # Deduplicate and sort
        seen: set[float] = set()
        unique: list[float] = []
        for k in sorted(keyframes):
            rounded = round(k, 2)
            if rounded not in seen:
                seen.add(rounded)
                unique.append(rounded)
        return unique if unique else [start_sec]

    @staticmethod
    def _sharpness(frame: object) -> float:
        """Return laplacian variance as a sharpness proxy."""
        if not _HAS_CV2 or cv2 is None:
            return 0.0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0


class TemporalRelationExtractor:
    """Extract temporal relations between detected scenes.

    Currently implements simple sequential relations. In a full system this
    would leverage a temporal reasoning model or scene graph network.
    """

    def extract(self, scenes: list[Scene]) -> list[TemporalRelation]:
        """Return temporal relations inferred from scene ordering and overlap."""
        relations: list[TemporalRelation] = []
        for i in range(len(scenes) - 1):
            curr = scenes[i]
            nxt = scenes[i + 1]
            relations.append(
                TemporalRelation(
                    source_id=curr.scene_id,
                    target_id=nxt.scene_id,
                    relation="follows",
                    confidence=1.0,
                )
            )
            # Detect overlap / containment heuristics
            if nxt.start_sec < curr.end_sec:
                overlap = curr.end_sec - nxt.start_sec
                overlap_ratio = overlap / (curr.end_sec - curr.start_sec)
                if overlap_ratio > 0.5:
                    relations.append(
                        TemporalRelation(
                            source_id=curr.scene_id,
                            target_id=nxt.scene_id,
                            relation="contains",
                            confidence=overlap_ratio,
                        )
                    )
                else:
                    relations.append(
                        TemporalRelation(
                            source_id=curr.scene_id,
                            target_id=nxt.scene_id,
                            relation="overlaps",
                            confidence=overlap_ratio,
                        )
                    )
        return relations


class VideoSummarizer:
    """Generate a natural-language and structured summary from scene data.

    Args:
        max_summary_length: Max characters for the overall summary. Defaults to 1024.
    """

    def __init__(self, max_summary_length: int = 1024) -> None:
        self.max_summary_length = max_summary_length

    def summarize(self, scenes: list[Scene], duration_sec: float) -> VideoSummary:
        """Produce a VideoSummary from scene list."""
        if not scenes:
            return VideoSummary(duration_sec=duration_sec)

        # Simple title inference from first scene OCR
        title = self._infer_title(scenes)
        overall = self._compose_summary(scenes, duration_sec)
        topics = self._extract_topics(scenes)

        return VideoSummary(
            title=title,
            duration_sec=duration_sec,
            scenes=scenes,
            temporal_relations=[],  # populated by caller
            overall_summary=overall,
            key_topics=topics,
        )

    def _infer_title(self, scenes: list[Scene]) -> str:
        """Heuristic title from first scene with meaningful OCR."""
        for s in scenes:
            text = s.ocr_text.strip()
            if len(text) > 3:
                # Use first sentence or first 40 chars
                first = text.split(".")[0]
                return (first[:40] + "...") if len(first) > 40 else first
        return "Untitled Video"

    def _compose_summary(self, scenes: list[Scene], duration_sec: float) -> str:
        """Compose a paragraph summary of scene flow."""
        parts: list[str] = []
        parts.append(
            f"This {self._format_duration(duration_sec)} video contains "
            f"{len(scenes)} scene(s)."
        )
        for i, s in enumerate(scenes[:5], 1):
            parts.append(
                f"Scene {i} ({self._format_duration(s.start_sec)}–"
                f"{self._format_duration(s.end_sec)}): {s.visual_summary or 'Visual content'}."
            )
        if len(scenes) > 5:
            parts.append(f"... and {len(scenes) - 5} additional scenes.")
        summary = " ".join(parts)
        # If truncation would cut mid-sentence, truncate earlier and append ellipsis
        if len(summary) > self.max_summary_length:
            summary = summary[: self.max_summary_length - 3].rsplit(" ", 1)[0] + "..."
        return summary

    @staticmethod
    def _format_duration(sec: float) -> str:
        """Format seconds as mm:ss."""
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}:{s:02d}"

    @staticmethod
    def _extract_topics(scenes: list[Scene]) -> list[str]:
        """Extract key topics from scene OCR via simple frequency heuristic."""
        all_text = " ".join(s.ocr_text for s in scenes).lower()
        # Simple stopword list
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "and", "but", "or", "yet", "so",
            "if", "because", "although", "though", "while", "where",
            "when", "that", "which", "who", "whom", "whose", "what",
            "this", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "its", "our", "their", "mine", "yours",
            "hers", "ours", "theirs", "myself", "yourself", "himself",
            "herself", "itself", "ourselves", "yourselves", "themselves",
        }
        words = [w.strip(".,!?;:\"'()[]") for w in all_text.split()]
        candidates = [w for w in words if len(w) > 3 and w not in stopwords]
        from collections import Counter

        freq = Counter(candidates)
        return [word for word, _ in freq.most_common(10)]


class VideoSceneExtractor:
    """High-level video scene understanding pipeline.

    Orchestrates segmentation, keyframe selection, OCR, temporal relation
    extraction, and summarization.

    Args:
        segmenter: SceneSegmenter instance.
        keyframe_selector: KeyframeSelector instance.
        relation_extractor: TemporalRelationExtractor instance.
        summarizer: VideoSummarizer instance.
        enable_ocr: Whether to run OCR on keyframes. Defaults to True.
        enable_audio: Whether to extract and transcribe audio. Defaults to True.
    """

    def __init__(
        self,
        segmenter: SceneSegmenter | None = None,
        keyframe_selector: KeyframeSelector | None = None,
        relation_extractor: TemporalRelationExtractor | None = None,
        summarizer: VideoSummarizer | None = None,
        enable_ocr: bool = True,
        enable_audio: bool = True,
    ) -> None:
        self.segmenter = segmenter or SceneSegmenter()
        self.keyframe_selector = keyframe_selector or KeyframeSelector()
        self.relation_extractor = relation_extractor or TemporalRelationExtractor()
        self.summarizer = summarizer or VideoSummarizer()
        self.enable_ocr = enable_ocr
        self.enable_audio = enable_audio

    def process(self, video_path: str | bytes) -> VideoSummary:
        """Process a video file or bytes and return a structured summary.

        Args:
            video_path: Path to video file or raw bytes.

        Returns:
            VideoSummary with scenes, relations, and overall summary.
        """
        raw: bytes
        if isinstance(video_path, str):
            raw = open(video_path, "rb").read()
        else:
            raw = video_path

        if not raw:
            return VideoSummary()

        # Write to temp file for cv2/ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            # 1. Segment
            boundaries = self.segmenter.segment(tmp_path)
            duration = self._get_duration(tmp_path)

            # 2. Build scenes with keyframes and OCR
            scenes: list[Scene] = []
            for i, (start, end) in enumerate(boundaries):
                scene_id = f"scene_{i:03d}"
                keyframes = self.keyframe_selector.select(tmp_path, start, end)
                ocr_text = ""
                if self.enable_ocr:
                    ocr_text = self._ocr_keyframes(tmp_path, keyframes)
                scenes.append(
                    Scene(
                        scene_id=scene_id,
                        start_sec=start,
                        end_sec=end,
                        keyframes=keyframes,
                        ocr_text=ocr_text,
                        visual_summary=self._generate_visual_summary(ocr_text, keyframes),
                    )
                )

            # 3. Temporal relations
            relations = self.relation_extractor.extract(scenes)

            # 4. Summarize
            summary = self.summarizer.summarize(scenes, duration)
            summary.temporal_relations = relations

            # 5. Optional audio transcription
            if self.enable_audio:
                audio_text = self._extract_audio(tmp_path)
                if audio_text:
                    summary.overall_summary += f" Audio: {audio_text[:500]}"

            return summary
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _get_duration(self, video_path: str) -> float:
        """Return video duration in seconds."""
        if not _HAS_CV2 or cv2 is None:
            return 0.0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frames / fps if fps > 0 else 0.0

    def _ocr_keyframes(self, video_path: str, keyframes: list[float]) -> str:
        """Extract OCR text from keyframe timestamps."""
        if not _HAS_CV2 or cv2 is None:
            return ""
        texts: list[str] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            for ts in keyframes:
                frame_idx = int(ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    text = self._ocr_frame(frame)
                    if text:
                        texts.append(text)
        except Exception as exc:
            _logger.warning("Keyframe OCR failed: %s", exc)
        finally:
            cap.release()
        return "\n".join(texts)

    def _ocr_frame(self, frame: object) -> str:
        """Run OCR on a single frame using a simple mockable interface."""
        if not _HAS_PIL or PILImage is None:
            return ""
        try:
            if _HAS_NUMPY and np is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PILImage.fromarray(rgb)
            else:
                return ""
            # Return a placeholder; real OCR would call an ImageExtractor
            # For this module we return a simple marker so tests can verify
            return f"[OCR:{img.size[0]}x{img.size[1]}]"
        except Exception as exc:
            _logger.warning("Frame OCR failed: %s", exc)
            return ""

    def _generate_visual_summary(self, ocr_text: str, keyframes: list[float]) -> str:
        """Generate a simple visual summary from OCR and keyframe count."""
        if ocr_text:
            first = ocr_text.replace("\n", " ")[:80]
            return f"Scene with text: {first}..."
        if keyframes:
            return f"Visual scene with {len(keyframes)} keyframe(s)."
        return "Visual scene."

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio track and transcribe (placeholder)."""
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
            # Placeholder transcription — real system would call AudioExtractor
            return "[audio transcription placeholder]"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass
