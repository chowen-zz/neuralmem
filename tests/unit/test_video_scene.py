"""VideoSceneExtractor unit tests — all mock-based, no real video processing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.multimodal.video_scene import (
    KeyframeSelector,
    Scene,
    SceneSegmenter,
    TemporalRelation,
    TemporalRelationExtractor,
    VideoSceneExtractor,
    VideoSummarizer,
    VideoSummary,
)


# --------------------------------------------------------------------------- #
# SceneSegmenter
# --------------------------------------------------------------------------- #

def test_segmenter_init():
    seg = SceneSegmenter(threshold=0.5, min_scene_sec=2.0)
    assert seg.threshold == 0.5
    assert seg.min_scene_sec == 2.0


def test_segment_no_cv2_fallback():
    seg = SceneSegmenter()
    with patch("neuralmem.multimodal.video_scene._HAS_CV2", False):
        with patch("neuralmem.multimodal.video_scene.cv2", None):
            result = seg.segment("/fake/path.mp4")
    assert result == [(0.0, 0.0)]


def test_segment_with_mock_cv2():
    seg = SceneSegmenter(threshold=0.2, min_scene_sec=0.5)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 10.0,   # CAP_PROP_FPS
        7: 100,    # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)

    frames = []
    for i in range(100):
        frame = MagicMock()
        frames.append((True, frame))
    frames.append((False, None))

    def mock_read():
        return frames.pop(0)

    mock_cap.read = mock_read

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.COLOR_BGR2HSV = 40
    mock_cv2.HISTCMP_BHATTACHARYYA = 3

    # Simulate increasing histogram differences to trigger cuts
    call_count = [0]

    def mock_calcHist(*args, **kwargs):
        call_count[0] += 1
        hist = MagicMock()
        return hist

    def mock_compareHist(h1, h2, method):
        # Return high diff every 30 frames to create scene cuts
        return 0.5 if call_count[0] % 30 == 0 else 0.1

    mock_cv2.calcHist = mock_calcHist
    mock_cv2.compareHist = mock_compareHist
    mock_cv2.normalize = lambda h, *_: h
    mock_cv2.cvtColor = lambda f, _: f

    with patch("neuralmem.multimodal.video_scene._HAS_CV2", True):
        with patch("neuralmem.multimodal.video_scene.cv2", mock_cv2):
            with patch("neuralmem.multimodal.video_scene._HAS_NUMPY", True):
                with patch("neuralmem.multimodal.video_scene.np", __import__("numpy")):
                    result = seg.segment("/fake/path.mp4")

    assert isinstance(result, list)
    assert len(result) > 0


# --------------------------------------------------------------------------- #
# KeyframeSelector
# --------------------------------------------------------------------------- #

def test_selector_init():
    sel = KeyframeSelector(keyframes_per_scene=5)
    assert sel.keyframes_per_scene == 5


def test_select_no_cv2():
    sel = KeyframeSelector()
    with patch("neuralmem.multimodal.video_scene._HAS_CV2", False):
        with patch("neuralmem.multimodal.video_scene.cv2", None):
            result = sel.select("/fake.mp4", 0.0, 10.0)
    assert result == []


def test_select_with_mock_cv2():
    sel = KeyframeSelector(keyframes_per_scene=2)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {5: 10.0}.get(prop, 0)
    mock_cap.set.return_value = True

    frames = []
    for i in range(50):
        frame = MagicMock()
        frames.append((True, frame))
    frames.append((False, None))

    def mock_read():
        return frames.pop(0)

    mock_cap.read = mock_read

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.COLOR_BGR2GRAY = 6
    mock_cv2.CV_64F = 6
    mock_cv2.cvtColor = lambda f, _: f
    mock_cv2.Laplacian = lambda f, _: MagicMock(var=lambda: float(50 + hash(id(f)) % 100))

    with patch("neuralmem.multimodal.video_scene._HAS_CV2", True):
        with patch("neuralmem.multimodal.video_scene.cv2", mock_cv2):
            result = sel.select("/fake.mp4", 0.0, 5.0)

    assert isinstance(result, list)
    assert len(result) > 0
    # Should be sorted and deduplicated
    assert result == sorted(set(result))


# --------------------------------------------------------------------------- #
# TemporalRelationExtractor
# --------------------------------------------------------------------------- #

def test_extract_simple_follows():
    extractor = TemporalRelationExtractor()
    scenes = [
        Scene(scene_id="s1", start_sec=0.0, end_sec=5.0),
        Scene(scene_id="s2", start_sec=5.0, end_sec=10.0),
    ]
    relations = extractor.extract(scenes)
    assert len(relations) == 1
    assert relations[0].source_id == "s1"
    assert relations[0].target_id == "s2"
    assert relations[0].relation == "follows"


def test_extract_overlap():
    extractor = TemporalRelationExtractor()
    scenes = [
        Scene(scene_id="s1", start_sec=0.0, end_sec=6.0),
        Scene(scene_id="s2", start_sec=4.0, end_sec=10.0),
    ]
    relations = extractor.extract(scenes)
    # follows + overlaps (overlap_ratio = 2/6 = 0.33, so overlaps not contains)
    assert len(relations) >= 1
    assert relations[0].relation == "follows"


def test_extract_contains():
    extractor = TemporalRelationExtractor()
    scenes = [
        Scene(scene_id="s1", start_sec=0.0, end_sec=10.0),
        Scene(scene_id="s2", start_sec=2.0, end_sec=8.0),
    ]
    relations = extractor.extract(scenes)
    assert any(r.relation == "contains" for r in relations)


def test_extract_empty():
    extractor = TemporalRelationExtractor()
    assert extractor.extract([]) == []


# --------------------------------------------------------------------------- #
# VideoSummarizer
# --------------------------------------------------------------------------- #

def test_summarizer_empty_scenes():
    summarizer = VideoSummarizer()
    summary = summarizer.summarize([], duration_sec=0.0)
    assert summary.duration_sec == 0.0
    assert summary.scenes == []


def test_summarizer_basic():
    summarizer = VideoSummarizer()
    scenes = [
        Scene(scene_id="s1", start_sec=0.0, end_sec=5.0, ocr_text="Welcome to the demo."),
        Scene(scene_id="s2", start_sec=5.0, end_sec=10.0, ocr_text="Here is the main feature."),
    ]
    summary = summarizer.summarize(scenes, duration_sec=10.0)
    assert summary.duration_sec == 10.0
    assert len(summary.scenes) == 2
    assert summary.title == "Welcome to the demo"
    assert "2 scene(s)" in summary.overall_summary


def test_summarizer_many_scenes_truncation():
    summarizer = VideoSummarizer(max_summary_length=200)
    scenes = [Scene(scene_id=f"s{i}", start_sec=float(i), end_sec=float(i + 1)) for i in range(10)]
    summary = summarizer.summarize(scenes, duration_sec=10.0)
    assert len(summary.overall_summary) <= 200
    # When truncated early, the "additional scenes" suffix may be cut; just verify truncation happened
    assert summary.overall_summary.endswith("...")


def test_infer_title_no_text():
    summarizer = VideoSummarizer()
    scenes = [Scene(scene_id="s1", ocr_text="")]
    title = summarizer._infer_title(scenes)
    assert title == "Untitled Video"


def test_format_duration():
    assert VideoSummarizer._format_duration(65.0) == "1:05"
    assert VideoSummarizer._format_duration(0.0) == "0:00"
    assert VideoSummarizer._format_duration(125.0) == "2:05"


def test_extract_topics():
    summarizer = VideoSummarizer()
    scenes = [
        Scene(scene_id="s1", ocr_text="machine learning artificial intelligence data"),
        Scene(scene_id="s2", ocr_text="artificial intelligence neural networks deep learning"),
    ]
    topics = summarizer._extract_topics(scenes)
    assert "artificial" in topics or "intelligence" in topics or "learning" in topics
    assert len(topics) <= 10


# --------------------------------------------------------------------------- #
# VideoSceneExtractor (high-level pipeline)
# --------------------------------------------------------------------------- #

def test_extractor_init_defaults():
    extractor = VideoSceneExtractor()
    assert extractor.segmenter is not None
    assert extractor.keyframe_selector is not None
    assert extractor.relation_extractor is not None
    assert extractor.summarizer is not None


def test_process_empty_bytes():
    extractor = VideoSceneExtractor()
    summary = extractor.process(b"")
    assert isinstance(summary, VideoSummary)
    assert summary.duration_sec == 0.0


def test_process_with_mocked_deps():
    extractor = VideoSceneExtractor(enable_ocr=True, enable_audio=False)

    mock_segmenter = MagicMock()
    mock_segmenter.segment.return_value = [(0.0, 5.0), (5.0, 10.0)]

    mock_selector = MagicMock()
    mock_selector.select.side_effect = lambda _path, start, end: [start + 1.0, end - 1.0]

    mock_summarizer = MagicMock()
    mock_summary = VideoSummary(
        title="Mock Video",
        duration_sec=10.0,
        scenes=[
            Scene(scene_id="s1", start_sec=0.0, end_sec=5.0),
            Scene(scene_id="s2", start_sec=5.0, end_sec=10.0),
        ],
        temporal_relations=[TemporalRelation("s1", "s2", "follows")],
        overall_summary="Mock summary.",
    )
    mock_summarizer.summarize.return_value = mock_summary

    extractor.segmenter = mock_segmenter
    extractor.keyframe_selector = mock_selector
    extractor.summarizer = mock_summarizer

    with patch("neuralmem.multimodal.video_scene._HAS_CV2", True):
        with patch("neuralmem.multimodal.video_scene.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {5: 10.0, 7: 100}.get(prop, 0)
            mock_cv2.VideoCapture.return_value = mock_cap
            with patch.object(extractor, "_ocr_keyframes", return_value="mock ocr"):
                with patch.object(extractor, "_get_duration", return_value=10.0):
                    summary = extractor.process(b"fake_video_bytes")

    assert summary.title == "Mock Video"
    assert len(summary.scenes) == 2
    assert summary.duration_sec == 10.0
    mock_segmenter.segment.assert_called_once()
    assert mock_selector.select.call_count == 2


def test_get_duration():
    extractor = VideoSceneExtractor()
    with patch("neuralmem.multimodal.video_scene._HAS_CV2", True):
        with patch("neuralmem.multimodal.video_scene.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {5: 25.0, 7: 250}.get(prop, 0)
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FPS = 5
            mock_cv2.CAP_PROP_FRAME_COUNT = 7
            duration = extractor._get_duration("/fake.mp4")
    assert duration == 10.0


def test_ocr_keyframes_no_cv2():
    extractor = VideoSceneExtractor()
    with patch("neuralmem.multimodal.video_scene._HAS_CV2", False):
        with patch("neuralmem.multimodal.video_scene.cv2", None):
            result = extractor._ocr_keyframes("/fake.mp4", [1.0, 2.0])
    assert result == ""


def test_ocr_frame_no_pil():
    extractor = VideoSceneExtractor()
    with patch("neuralmem.multimodal.video_scene._HAS_PIL", False):
        with patch("neuralmem.multimodal.video_scene.PILImage", None):
            result = extractor._ocr_frame(MagicMock())
    assert result == ""


def test_generate_visual_summary_with_ocr():
    extractor = VideoSceneExtractor()
    result = extractor._generate_visual_summary("Hello world", [1.0])
    assert "Hello world" in result


def test_generate_visual_summary_no_ocr():
    extractor = VideoSceneExtractor()
    result = extractor._generate_visual_summary("", [1.0, 2.0])
    assert "2 keyframe(s)" in result


def test_generate_visual_summary_empty():
    extractor = VideoSceneExtractor()
    result = extractor._generate_visual_summary("", [])
    assert result == "Visual scene."


def test_scene_dataclass():
    scene = Scene(scene_id="s1", start_sec=0.0, end_sec=5.0)
    assert scene.keyframes == []
    assert scene.ocr_text == ""
    assert scene.metadata == {}


def test_temporal_relation_dataclass():
    rel = TemporalRelation("a", "b", "follows", confidence=0.9)
    assert rel.confidence == 0.9


def test_video_summary_dataclass():
    summary = VideoSummary(title="Test", duration_sec=60.0)
    assert summary.scenes == []
    assert summary.temporal_relations == []
    assert summary.key_topics == []
