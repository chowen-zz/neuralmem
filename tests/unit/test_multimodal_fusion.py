"""MultimodalFusionEngine unit tests — all mock-based, no real embeddings."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neuralmem.retrieval.multimodal_fusion import (
    CrossModalAligner,
    FusionResult,
    ModalityResult,
    MultimodalFusionEngine,
    UnifiedEmbeddingSpace,
)


# --------------------------------------------------------------------------- #
# CrossModalAligner
# --------------------------------------------------------------------------- #

def test_aligner_init_defaults():
    aligner = CrossModalAligner()
    assert aligner.target_dim == 768
    assert aligner.modality_dims == {}


def test_aligner_register_modality():
    aligner = CrossModalAligner(target_dim=128)
    aligner.register_modality("text", 512)
    assert "text" in aligner.modality_dims
    assert aligner.modality_dims["text"] == 512
    assert "text" in aligner._projections
    assert aligner._projections["text"].shape == (512, 128)


def test_align_vector_known_modality():
    aligner = CrossModalAligner(target_dim=64, modality_dims={"text": 128})
    vec = np.ones(128, dtype=np.float32)
    aligned = aligner.align(vec, "text")
    assert aligned.shape == (64,)
    assert aligned.dtype == np.float32


def test_align_vector_unknown_modality_fallback():
    aligner = CrossModalAligner(target_dim=64)
    vec = np.ones(100, dtype=np.float32)
    aligned = aligner.align(vec, "unknown")
    assert aligned.shape == (64,)


def test_align_batch():
    aligner = CrossModalAligner(target_dim=32, modality_dims={"image": 64})
    batch = np.ones((5, 64), dtype=np.float32)
    aligned = aligner.batch_align(batch, "image")
    assert aligned.shape == (5, 32)


def test_truncate_or_pad_exact():
    aligner = CrossModalAligner(target_dim=64)
    vec = np.ones(64, dtype=np.float32)
    out = aligner._truncate_or_pad(vec)
    assert np.array_equal(out, vec)


def test_truncate_or_pad_long():
    aligner = CrossModalAligner(target_dim=32)
    vec = np.ones(64, dtype=np.float32)
    out = aligner._truncate_or_pad(vec)
    assert out.shape == (32,)
    assert np.array_equal(out, np.ones(32, dtype=np.float32))


def test_truncate_or_pad_short():
    aligner = CrossModalAligner(target_dim=64)
    vec = np.ones(32, dtype=np.float32)
    out = aligner._truncate_or_pad(vec)
    assert out.shape == (64,)
    assert np.array_equal(out[:32], vec)
    assert np.array_equal(out[32:], np.zeros(32, dtype=np.float32))


def test_truncate_or_pad_batch():
    aligner = CrossModalAligner(target_dim=16)
    batch = np.ones((3, 32), dtype=np.float32)
    out = aligner._truncate_or_pad(batch)
    assert out.shape == (3, 16)


def test_cosine_similarity_identical():
    aligner = CrossModalAligner(target_dim=64)
    vec = np.array([1.0, 0.0, 0.0] + [0.0] * 61, dtype=np.float32)
    sim = aligner.cosine_similarity(vec, vec)
    assert sim == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    aligner = CrossModalAligner(target_dim=64)
    a = np.array([1.0, 0.0] + [0.0] * 62, dtype=np.float32)
    b = np.array([0.0, 1.0] + [0.0] * 62, dtype=np.float32)
    sim = aligner.cosine_similarity(a, b)
    assert sim == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    aligner = CrossModalAligner(target_dim=8)
    a = np.zeros(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    assert aligner.cosine_similarity(a, b) == 0.0


def test_batch_cosine_similarity():
    aligner = CrossModalAligner(target_dim=8)
    queries = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    candidates = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32
    )
    sims = aligner.batch_cosine_similarity(queries, candidates)
    assert sims.shape == (1, 2)
    assert sims[0, 0] == pytest.approx(1.0)
    assert sims[0, 1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# UnifiedEmbeddingSpace
# --------------------------------------------------------------------------- #

def test_add_and_remove():
    aligner = CrossModalAligner(target_dim=16, modality_dims={"text": 32})
    space = UnifiedEmbeddingSpace(aligner)
    vec = np.ones(32, dtype=np.float32)
    space.add("doc1", vec, "text", {"title": "hello"})
    assert "doc1" in space._vectors
    assert space._modality_map["doc1"] == "text"
    space.remove("doc1")
    assert "doc1" not in space._vectors


def test_search_empty():
    aligner = CrossModalAligner(target_dim=8, modality_dims={"text": 16})
    space = UnifiedEmbeddingSpace(aligner)
    q = np.ones(16, dtype=np.float32)
    assert space.search(q, "text", top_k=5) == []


def test_search_basic():
    aligner = CrossModalAligner(target_dim=4, modality_dims={"text": 8, "image": 8})
    space = UnifiedEmbeddingSpace(aligner)
    # Add two docs
    v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    space.add("doc1", v1, "text")
    space.add("doc2", v2, "image")
    # Query close to doc1
    q = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    results = space.search(q, "text", top_k=2)
    assert len(results) == 2
    assert results[0].id == "doc1"
    assert results[0].score > results[1].score


def test_search_top_k_limit():
    aligner = CrossModalAligner(target_dim=4, modality_dims={"text": 8})
    space = UnifiedEmbeddingSpace(aligner)
    for i in range(10):
        vec = np.zeros(8, dtype=np.float32)
        vec[i % 8] = 1.0
        space.add(f"doc{i}", vec, "text")
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    results = space.search(q, "text", top_k=3)
    assert len(results) == 3


# --------------------------------------------------------------------------- #
# MultimodalFusionEngine
# --------------------------------------------------------------------------- #

def test_engine_index_and_retrieve():
    engine = MultimodalFusionEngine(
        aligner=CrossModalAligner(target_dim=8, modality_dims={"text": 16})
    )
    vec = np.ones(16, dtype=np.float32)
    engine.index("doc1", vec, "text", {"source": "test"})
    results = engine.retrieve(vec, "text", top_k=1)
    assert len(results) == 1
    assert results[0].id == "doc1"


def test_engine_fuse_two_modalities():
    engine = MultimodalFusionEngine(rrf_k=60)
    text_results = [
        ModalityResult(id="a", modality="text", score=0.9),
        ModalityResult(id="b", modality="text", score=0.7),
    ]
    image_results = [
        ModalityResult(id="a", modality="image", score=0.85),
        ModalityResult(id="c", modality="image", score=0.6),
    ]
    fused = engine.fuse({"text": text_results, "image": image_results})
    ids = [f.id for f in fused]
    # "a" appears in both, should rank highest
    assert ids[0] == "a"
    # No duplicates
    assert len(ids) == len(set(ids))


def test_engine_fuse_single_modality():
    engine = MultimodalFusionEngine(rrf_k=60)
    results = [
        ModalityResult(id="x", modality="text", score=1.0),
        ModalityResult(id="y", modality="text", score=0.5),
    ]
    fused = engine.fuse({"text": results})
    assert len(fused) == 2
    assert fused[0].id == "x"
    assert fused[0].fused_score == pytest.approx(1.0)


def test_engine_fuse_empty():
    engine = MultimodalFusionEngine()
    assert engine.fuse({}) == []


def test_cross_modal_search():
    aligner = CrossModalAligner(target_dim=4, modality_dims={"text": 8, "image": 8})
    engine = MultimodalFusionEngine(aligner=aligner)
    v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    engine.index("doc1", v1, "text")
    engine.index("doc2", v2, "image")
    q = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    fused = engine.cross_modal_search(q, "text", top_k=2)
    assert len(fused) > 0
    assert all(0.0 <= f.fused_score <= 1.0 for f in fused)


def test_cross_modal_search_with_target_modalities():
    aligner = CrossModalAligner(target_dim=4, modality_dims={"text": 8, "image": 8, "audio": 8})
    engine = MultimodalFusionEngine(aligner=aligner)
    for i, mod in enumerate(["text", "image", "audio"]):
        vec = np.zeros(8, dtype=np.float32)
        vec[i] = 1.0
        engine.index(f"doc_{mod}", vec, mod)
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    fused = engine.cross_modal_search(q, "text", target_modalities=["text", "image"], top_k=5)
    # Should only return text and image docs
    for f in fused:
        assert f.modality_scores  # at least one modality contributed


def test_compute_alignment_quality():
    aligner = CrossModalAligner(target_dim=8, modality_dims={"text": 16, "image": 16})
    engine = MultimodalFusionEngine(aligner=aligner)
    v_text = np.ones(16, dtype=np.float32)
    v_image = np.ones(16, dtype=np.float32)
    pairs = [(v_text, v_image, "text", "image")]
    quality = engine.compute_alignment_quality(pairs)
    assert "mean_similarity" in quality
    assert "min_similarity" in quality
    assert "max_similarity" in quality
    assert 0.0 <= quality["mean_similarity"] <= 1.0


def test_compute_alignment_quality_empty():
    engine = MultimodalFusionEngine()
    quality = engine.compute_alignment_quality([])
    assert quality["mean_similarity"] == 0.0


# --------------------------------------------------------------------------- #
# FusionResult / ModalityResult dataclasses
# --------------------------------------------------------------------------- #

def test_modality_result_defaults():
    r = ModalityResult(id="x", modality="text")
    assert r.score == 0.0
    assert r.metadata == {}
    assert r.embedding is None


def test_fusion_result_defaults():
    f = FusionResult(id="x")
    assert f.fused_score == 0.0
    assert f.modality_scores == {}
    assert f.metadata == {}
