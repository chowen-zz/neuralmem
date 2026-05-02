"""NeuralMem Protocol 接口定义 — 所有模块只依赖此文件，消除循环导入"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from neuralmem.core.types import Entity, Memory, Relation


@runtime_checkable
class StorageProtocol(Protocol):
    """存储后端协议"""

    def save_memory(self, memory: Memory) -> str: ...
    def get_memory(self, memory_id: str) -> Memory | None: ...
    def update_memory(self, memory_id: str, **kwargs: object) -> None: ...
    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int: ...
    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...
    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...
    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...
    def find_similar(
        self, vector: list[float], user_id: str | None = None, threshold: float = 0.95
    ) -> list[Memory]: ...
    def record_access(self, memory_id: str) -> None: ...
    def batch_record_access(self, memory_ids: list[str]) -> None: ...
    def get_stats(self, user_id: str | None = None) -> dict[str, object]: ...
    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]: ...
    def load_graph_snapshot(self) -> dict | None: ...
    def save_graph_snapshot(self, data: dict) -> None: ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Embedding 提供商协议"""

    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...
    def encode_one(self, text: str) -> list[float]: ...


@runtime_checkable
class ExtractorProtocol(Protocol):
    """记忆提取器协议"""

    def extract(self, text: str) -> tuple[list[Entity], list[Relation]]: ...


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """知识图谱协议"""

    def upsert_entity(self, entity: Entity) -> None: ...
    def add_relation(self, relation: Relation) -> None: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def get_entities(self, user_id: str | None = None) -> list[Entity]: ...
    def get_neighbors(
        self, entity_ids: list[str], depth: int = 1
    ) -> list[Entity]: ...
    def find_entities(self, query: str) -> list[Entity]: ...
    def traverse_for_memories(
        self,
        entity_ids: list[str],
        depth: int = 2,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]: ...


@runtime_checkable
class LifecycleProtocol(Protocol):
    """记忆生命周期协议"""

    def apply_decay(self, user_id: str | None = None) -> int: ...
    def remove_forgotten(self, user_id: str | None = None) -> int: ...
