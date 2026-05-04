"""Multimodal fusion engine — cross-modal vector alignment and unified retrieval.

Implements:
- Cross-modal vector alignment (text ↔ image ↔ audio)
- Unified embedding space projection
- Multimodal RRF fusion for ranking results from multiple modalities
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neuralmem.retrieval.fusion import RankedItem, RRFMerger

_logger = logging.getLogger(__name__)


@dataclass
class ModalityResult:
    """A single retrieval result from one modality.

    Attributes:
        id: Unique document / chunk identifier.
        modality: Source modality (e.g. ``text``, ``image``, ``audio``).
        embedding: Raw embedding vector (optional).
        score: Similarity or relevance score.
        metadata: Arbitrary metadata dict.
    """

    id: str
    modality: str
    embedding: np.ndarray | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Fused result after cross-modal alignment and RRF scoring.

    Attributes:
        id: Unique identifier.
        fused_score: Final multimodal score (0-1).
        modality_scores: Per-modality raw scores.
        metadata: Merged metadata from all contributing modalities.
    """

    id: str
    fused_score: float = 0.0
    modality_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossModalAligner:
    """Align vectors from different modalities into a shared latent space.

    Uses a learnable linear projection (or identity fallback) per modality.
    In production this would be replaced by a trained contrastive model;
    here we provide a configurable projection matrix interface.

    Args:
        target_dim: Dimensionality of the unified space. Defaults to 768.
        modality_dims: Mapping ``modality -> input_dim``.
        random_seed: Seed for reproducible initialisation.
    """

    def __init__(
        self,
        target_dim: int = 768,
        modality_dims: dict[str, int] | None = None,
        *,
        random_seed: int = 42,
    ) -> None:
        self.target_dim = target_dim
        self.modality_dims = modality_dims or {}
        self._projections: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(random_seed)
        self._build_projections()

    def _build_projections(self) -> None:
        """Initialise random orthonormal projection matrices per modality."""
        for modality, in_dim in self.modality_dims.items():
            # Xavier-like init scaled for orthonormal projection
            w = self._rng.standard_normal((in_dim, self.target_dim))
            # QR decomposition to get orthonormal columns
            q, _ = np.linalg.qr(w)
            self._projections[modality] = q.astype(np.float32)

    def register_modality(self, modality: str, in_dim: int) -> None:
        """Register a new modality and create its projection matrix."""
        self.modality_dims[modality] = in_dim
        w = self._rng.standard_normal((in_dim, self.target_dim))
        q, _ = np.linalg.qr(w)
        self._projections[modality] = q.astype(np.float32)
        _logger.info("Registered modality %r with projection %dx%d", modality, in_dim, self.target_dim)

    def align(self, vector: np.ndarray, modality: str) -> np.ndarray:
        """Project a vector from *modality* space into the unified space.

        Args:
            vector: Input vector of shape ``(in_dim,)`` or ``(batch, in_dim)``.
            modality: Source modality name.

        Returns:
            Projected vector of shape ``(target_dim,)`` or ``(batch, target_dim)``.
        """
        if modality not in self._projections:
            # Fallback: truncate / pad to target_dim
            return self._truncate_or_pad(vector)
        proj = self._projections[modality]
        return np.dot(vector, proj)

    def batch_align(
        self,
        vectors: np.ndarray,
        modality: str,
    ) -> np.ndarray:
        """Align a batch of vectors."""
        return self.align(vectors, modality)

    def _truncate_or_pad(self, vector: np.ndarray) -> np.ndarray:
        """Fallback alignment by truncating or zero-padding to target_dim."""
        ndim = vector.ndim
        if ndim == 1:
            dim = vector.shape[0]
            if dim == self.target_dim:
                return vector.astype(np.float32)
            if dim > self.target_dim:
                return vector[: self.target_dim].astype(np.float32)
            padded = np.zeros(self.target_dim, dtype=np.float32)
            padded[:dim] = vector
            return padded
        if ndim == 2:
            batch, dim = vector.shape
            if dim == self.target_dim:
                return vector.astype(np.float32)
            if dim > self.target_dim:
                return vector[:, : self.target_dim].astype(np.float32)
            padded = np.zeros((batch, self.target_dim), dtype=np.float32)
            padded[:, :dim] = vector
            return padded
        raise ValueError(f"Unsupported vector ndim {ndim}")

    def cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """Cosine similarity between two unified-space vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def batch_cosine_similarity(
        self,
        queries: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Cosine similarity between query batch and candidate batch.

        Args:
            queries: ``(nq, target_dim)``
            candidates: ``(nc, target_dim)``

        Returns:
            ``(nq, nc)`` similarity matrix.
        """
        q_norm = np.linalg.norm(queries, axis=1, keepdims=True)
        c_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        c_norm[c_norm == 0] = 1.0
        qn = queries / q_norm
        cn = candidates / c_norm
        return np.dot(qn, cn.T)


class UnifiedEmbeddingSpace:
    """Manager for the unified embedding space.

    Holds all aligned vectors and provides nearest-neighbour search.

    Args:
        aligner: CrossModalAligner instance.
    """

    def __init__(self, aligner: CrossModalAligner) -> None:
        self.aligner = aligner
        self._vectors: dict[str, np.ndarray] = {}
        self._modality_map: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add(
        self,
        id: str,
        vector: np.ndarray,
        modality: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the unified space after alignment."""
        aligned = self.aligner.align(vector, modality)
        self._vectors[id] = aligned
        self._modality_map[id] = modality
        self._metadata[id] = metadata or {}

    def remove(self, id: str) -> None:
        """Remove a vector from the unified space."""
        self._vectors.pop(id, None)
        self._modality_map.pop(id, None)
        self._metadata.pop(id, None)

    def search(
        self,
        query_vector: np.ndarray,
        query_modality: str,
        top_k: int = 10,
    ) -> list[ModalityResult]:
        """Search the unified space with a query vector.

        Args:
            query_vector: Raw query embedding.
            query_modality: Modality of the query.
            top_k: Number of results to return.

        Returns:
            List of ModalityResult sorted by cosine similarity descending.
        """
        if not self._vectors:
            return []
        aligned_query = self.aligner.align(query_vector, query_modality)
        ids = list(self._vectors.keys())
        matrix = np.stack([self._vectors[i] for i in ids])
        sims = self.aligner.batch_cosine_similarity(
            aligned_query.reshape(1, -1), matrix
        )[0]
        # argsort descending
        order = np.argsort(-sims)
        results: list[ModalityResult] = []
        for idx in order[:top_k]:
            doc_id = ids[idx]
            results.append(
                ModalityResult(
                    id=doc_id,
                    modality=self._modality_map[doc_id],
                    embedding=self._vectors[doc_id],
                    score=float(sims[idx]),
                    metadata=self._metadata[doc_id],
                )
            )
        return results


class MultimodalFusionEngine:
    """End-to-end multimodal retrieval fusion engine.

    Combines cross-modal alignment, unified embedding search, and RRF fusion
    across multiple modality-specific retrievers.

    Args:
        aligner: CrossModalAligner instance.
        unified_space: UnifiedEmbeddingSpace instance.
        rrf_k: RRF hyperparameter. Defaults to 60.
    """

    def __init__(
        self,
        aligner: CrossModalAligner | None = None,
        unified_space: UnifiedEmbeddingSpace | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.aligner = aligner or CrossModalAligner()
        self.unified_space = unified_space or UnifiedEmbeddingSpace(self.aligner)
        self.rrf_merger = RRFMerger(k=rrf_k)

    def index(
        self,
        id: str,
        vector: np.ndarray,
        modality: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index a multimodal vector into the unified space."""
        self.unified_space.add(id, vector, modality, metadata)

    def retrieve(
        self,
        query_vector: np.ndarray,
        query_modality: str,
        top_k: int = 10,
    ) -> list[ModalityResult]:
        """Retrieve from the unified embedding space."""
        return self.unified_space.search(query_vector, query_modality, top_k)

    def fuse(
        self,
        ranked_lists: dict[str, list[ModalityResult]],
    ) -> list[FusionResult]:
        """Fuse multiple modality-specific ranked lists via RRF.

        Args:
            ranked_lists: Mapping ``modality -> list[ModalityResult]`` sorted by score desc.

        Returns:
            List of FusionResult sorted by fused score descending.
        """
        # Convert ModalityResult lists to RankedItem lists for RRFMerger
        rrf_input: dict[str, list[RankedItem]] = {}
        for modality, results in ranked_lists.items():
            rrf_input[modality] = [
                RankedItem(id=r.id, score=r.score, method=modality) for r in results
            ]

        merged = self.rrf_merger.merge(rrf_input)

        # Aggregate per-id modality scores and metadata
        id_modality_scores: dict[str, dict[str, float]] = {}
        id_metadata: dict[str, dict[str, Any]] = {}
        for modality, results in ranked_lists.items():
            for r in results:
                id_modality_scores.setdefault(r.id, {})[modality] = r.score
                id_metadata.setdefault(r.id, {}).update(r.metadata)

        fused: list[FusionResult] = []
        for doc_id, rrf_score in merged:
            fused.append(
                FusionResult(
                    id=doc_id,
                    fused_score=rrf_score,
                    modality_scores=id_modality_scores.get(doc_id, {}),
                    metadata=id_metadata.get(doc_id, {}),
                )
            )
        return fused

    def cross_modal_search(
        self,
        query_vector: np.ndarray,
        query_modality: str,
        target_modalities: list[str] | None = None,
        top_k: int = 10,
    ) -> list[FusionResult]:
        """Search across modalities and fuse results.

        Args:
            query_vector: Query embedding.
            query_modality: Modality of the query.
            target_modalities: Modalities to search. Defaults to all indexed.
            top_k: Results per modality.

        Returns:
            Fused and ranked FusionResult list.
        """
        # Retrieve from unified space (already cross-modal)
        unified_results = self.unified_space.search(query_vector, query_modality, top_k)

        # Group by modality for RRF fusion
        by_modality: dict[str, list[ModalityResult]] = {}
        for r in unified_results:
            by_modality.setdefault(r.modality, []).append(r)

        if target_modalities:
            by_modality = {
                k: v for k, v in by_modality.items() if k in target_modalities
            }

        return self.fuse(by_modality)

    def compute_alignment_quality(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray, str, str]],
    ) -> dict[str, float]:
        """Compute average cosine similarity for aligned cross-modal pairs.

        Args:
            pairs: List of (vec_a, vec_b, modality_a, modality_b) tuples.

        Returns:
            Dict with ``mean_similarity``, ``min_similarity``, ``max_similarity``.
        """
        if not pairs:
            return {"mean_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0}
        sims: list[float] = []
        for va, vb, ma, mb in pairs:
            a_aligned = self.aligner.align(va, ma)
            b_aligned = self.aligner.align(vb, mb)
            sims.append(self.aligner.cosine_similarity(a_aligned, b_aligned))
        return {
            "mean_similarity": float(np.mean(sims)),
            "min_similarity": float(np.min(sims)),
            "max_similarity": float(np.max(sims)),
        }
