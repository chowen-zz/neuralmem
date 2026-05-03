"""NeuralMem 核心数据模型 — frozen=True 确保跨模块类型契约稳定"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --- ULID generator (time-ordered, no external dependency) ---
_ULID_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _generate_ulid() -> str:
    """Generate a ULID: 48-bit timestamp + 80-bit randomness, Crockford base32.

    26 characters, lexicographically sortable by creation time.
    """
    # 48-bit timestamp in milliseconds
    ts = int(time.time() * 1000)
    # 80-bit randomness (10 bytes)
    rand_int = int.from_bytes(os.urandom(10), "big")

    # Encode timestamp (10 chars)
    result = ""
    t = ts
    for _ in range(10):
        result = _ULID_ENCODING[t & 0x1F] + result
        t >>= 5

    # Encode randomness (16 chars)
    r = rand_int
    for _ in range(16):
        result += _ULID_ENCODING[r & 0x1F]
        r >>= 5

    return result


def _generate_short_ulid() -> str:
    """Generate a 12-char prefix of a ULID (time-ordered, for entity IDs)."""
    return _generate_ulid()[:12]


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


class SessionLayer(str, Enum):
    """会话记忆的三层架构"""
    WORKING = "working"      # 当前轮次的工作记忆（不持久化）
    SESSION = "session"      # 当前会话的记忆（会话结束时压缩）
    LONG_TERM = "long_term"  # 长期持久记忆


class ExportFormat(str, Enum):
    """记忆导出格式"""
    JSON = "json"
    MARKDOWN = "markdown"
    CSV = "csv"


class Entity(BaseModel):
    """知识图谱实体"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=_generate_short_ulid)
    name: str
    entity_type: str = "unknown"
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    attributes: dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Relation(BaseModel):
    """知识图谱关系"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    target_id: str
    relation_type: str
    weight: float = Field(default=1.0, ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """一条记忆"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=_generate_ulid)
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

    # 冲突解决
    is_active: bool = Field(
        default=True, description="是否为当前有效记忆（被新记忆取代时置False）"
    )
    superseded_by: str | None = Field(default=None, description="取代此记忆的新记忆 ID")
    supersedes: tuple[str, ...] = Field(
        default_factory=tuple, description="此记忆取代的旧记忆ID列表"
    )

    # 时间戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = Field(default=0, ge=0)

    # 向量（内部使用，不序列化到 API）
    embedding: list[float] | None = Field(default=None, exclude=True, frozen=False)

    # TTL / 过期时间
    expires_at: datetime | None = Field(
        default=None, description="记忆过期时间（UTC），None 表示永不过期"
    )


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


class MemoryHistoryEntry(BaseModel):
    """A single version history entry for a memory."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    memory_id: str
    old_content: str | None = None
    new_content: str
    event: str  # 'CREATE', 'UPDATE', 'DELETE'
    changed_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
