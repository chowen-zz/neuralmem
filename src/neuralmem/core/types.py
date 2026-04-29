"""NeuralMem 核心数据模型 — frozen=True 确保跨模块类型契约稳定"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field


class MemoryType(str, Enum):
    """四类认知记忆模型"""
    EPISODIC = "episodic"        # 事件/交互记录
    SEMANTIC = "semantic"        # 事实/偏好/知识
    PROCEDURAL = "procedural"    # 流程/SOP/最佳实践
    WORKING = "working"          # 当前会话上下文


class MemoryScope(str, Enum):
    """记忆归属范围"""
    USER = "user"
    AGENT = "agent"
    SESSION = "session"
    SHARED = "shared"


class Entity(BaseModel):
    """知识图谱实体"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    entity_type: str = "unknown"
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    attributes: dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)


class Relation(BaseModel):
    """知识图谱关系"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    target_id: str
    relation_type: str
    weight: float = Field(default=1.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """一条记忆"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    scope: MemoryScope = MemoryScope.USER

    # 归属
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None

    # 元数据
    tags: tuple[str, ...] = Field(default_factory=tuple)
    source: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # 关联实体
    entity_ids: tuple[str, ...] = Field(default_factory=tuple)

    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, ge=0)

    # 向量（内部使用，不序列化到 API）
    embedding: list[float] | None = Field(default=None, exclude=True, frozen=False)


class SearchResult(BaseModel):
    """搜索结果"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory: Memory
    score: float = Field(ge=0.0, le=1.0)
    retrieval_method: str
    explanation: str | None = None


class SearchQuery(BaseModel):
    """搜索请求"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    user_id: str | None = None
    agent_id: str | None = None
    memory_types: tuple[MemoryType, ...] | None = None
    tags: tuple[str, ...] | None = None
    time_range: tuple[datetime, datetime] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
