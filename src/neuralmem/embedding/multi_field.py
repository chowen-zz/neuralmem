"""Multi-field embedding strategy for NeuralMem.

Supports multiple embedding fields per memory (summary, content, metadata)
similar to Supermemory's multi-embedding approach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class EmbedderProtocol(Protocol):
    """Protocol for embedding backends."""
    dimension: int
    def encode(self, texts: list[str]) -> list[list[float]]: ...
    def encode_one(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class FieldEmbedding:
    """Embedding for a specific field."""
    field_name: str
    vector: list[float]
    model_name: str = "default"


@dataclass
class MultiFieldEmbeddings:
    """Collection of embeddings for different fields of a memory."""
    memory_id: str
    fields: dict[str, FieldEmbedding] = field(default_factory=dict)

    def add_field(self, name: str, vector: list[float], model_name: str = "default") -> None:
        self.fields[name] = FieldEmbedding(name, vector, model_name)

    def get_field(self, name: str) -> FieldEmbedding | None:
        return self.fields.get(name)

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "fields": {
                name: {"field_name": fe.field_name, "vector": fe.vector, "model_name": fe.model_name}
                for name, fe in self.fields.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> MultiFieldEmbeddings:
        mfe = cls(memory_id=data["memory_id"])
        for name, fe_data in data.get("fields", {}).items():
            mfe.add_field(name, fe_data["vector"], fe_data.get("model_name", "default"))
        return mfe


class MultiFieldEmbedder:
    """Generates multiple embeddings for different fields of a memory.
    
    Similar to Supermemory's approach with summaryEmbedding, embedding,
    memoryEmbedding, matryokshaEmbedding fields.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        fields: list[str] | None = None,
        field_templates: dict[str, str] | None = None,
    ):
        self.embedder = embedder
        self.fields = fields or ["content", "summary"]
        self.field_templates = field_templates or {}

    def embed_memory(
        self,
        memory_id: str,
        content: str,
        summary: str | None = None,
        metadata: dict | None = None,
    ) -> MultiFieldEmbeddings:
        result = MultiFieldEmbeddings(memory_id=memory_id)
        texts_to_embed: dict[str, str] = {}

        for field_name in self.fields:
            if field_name == "content":
                texts_to_embed[field_name] = content
            elif field_name == "summary" and summary:
                texts_to_embed[field_name] = summary
            elif field_name == "metadata" and metadata:
                texts_to_embed[field_name] = " ".join(f"{k}: {v}" for k, v in metadata.items())
            elif field_name in self.field_templates:
                template = self.field_templates[field_name]
                texts_to_embed[field_name] = template.format(
                    content=content, summary=summary or "", metadata=metadata or {}
                )

        if texts_to_embed:
            vectors = self.embedder.encode(list(texts_to_embed.values()))
            for (field_name, _), vector in zip(texts_to_embed.items(), vectors):
                result.add_field(field_name, vector, model_name="default")

        return result

    def search_by_field(
        self,
        query_vector: list[float],
        candidates: list[MultiFieldEmbeddings],
        field_name: str = "content",
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search candidates by a specific field."""
        scored: list[tuple[str, float]] = []
        for candidate in candidates:
            fe = candidate.get_field(field_name)
            if fe is None:
                continue
            sim = cosine_similarity(query_vector, fe.vector)
            scored.append((candidate.memory_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def hybrid_search(
        self,
        query_vector: list[float],
        candidates: list[MultiFieldEmbeddings],
        field_weights: dict[str, float] | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search across all fields with weights."""
        weights = field_weights or {f: 1.0 for f in self.fields}
        scored: dict[str, float] = {}

        for candidate in candidates:
            total_score = 0.0
            total_weight = 0.0
            for field_name, weight in weights.items():
                fe = candidate.get_field(field_name)
                if fe is None:
                    continue
                sim = cosine_similarity(query_vector, fe.vector)
                total_score += sim * weight
                total_weight += weight
            if total_weight > 0:
                scored[candidate.memory_id] = total_score / total_weight

        results = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)
