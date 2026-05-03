"""用户画像类型定义 — Intent, Preference, Knowledge, InteractionStyle.

定义四种核心画像维度及其具体数据结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from neuralmem.profiles.base import ProfileAttribute, UserProfile


class ProfileDimension(str, Enum):
    """画像维度枚举."""

    INTENT = "intent"
    PREFERENCE = "preference"
    KNOWLEDGE = "knowledge"
    INTERACTION_STYLE = "interaction_style"


# ------------------------------------------------------------------ #
# Intent — 用户意图画像
# ------------------------------------------------------------------ #

class IntentCategory(str, Enum):
    """意图类别."""

    INFORMATIONAL = "informational"      # 获取信息
    TRANSACTIONAL = "transactional"        # 执行操作/交易
    NAVIGATIONAL = "navigational"          # 导航/查找
    SOCIAL = "social"                      # 社交互动
    CREATIVE = "creative"                  # 创作/生成
    TROUBLESHOOTING = "troubleshooting"    # 故障排查
    LEARNING = "learning"                  # 学习/教育
    ENTERTAINMENT = "entertainment"        # 娱乐


@dataclass(frozen=True, slots=True)
class Intent:
    """用户意图实例.

    Attributes:
        category: 意图类别.
        subcategory: 子类别（如 "code_generation"）。
        keywords: 触发此意图的关键词.
        confidence: 置信度.
        timestamp: 推断时间.
    """

    category: IntentCategory
    subcategory: str = ""
    keywords: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def to_attribute(self, name: str = "primary_intent") -> ProfileAttribute:
        """Convert to a ProfileAttribute."""
        return ProfileAttribute(
            name=name,
            value={
                "category": self.category.value,
                "subcategory": self.subcategory,
                "keywords": list(self.keywords),
            },
            confidence=self.confidence,
            source="intent_analysis",
            timestamp=self.timestamp,
        )


# ------------------------------------------------------------------ #
# Preference — 用户偏好画像
# ------------------------------------------------------------------ #

class PreferenceType(str, Enum):
    """偏好类型."""

    TECHNOLOGY = "technology"          # 技术栈偏好
    FORMAT = "format"                  # 输出格式偏好
    TONE = "tone"                      # 语气/风格偏好
    DEPTH = "depth"                    # 回答深度偏好
    LANGUAGE = "language"              # 语言偏好
    TIMING = "timing"                  # 时间/节奏偏好
    TOPIC = "topic"                    # 主题/领域偏好
    TOOL = "tool"                      # 工具偏好


@dataclass(frozen=True, slots=True)
class Preference:
    """用户偏好实例.

    Attributes:
        preference_type: 偏好类型.
        value: 偏好值.
        strength: 偏好强度 0.0~1.0.
        context: 适用上下文（如 "coding", "general"）.
        confidence: 置信度.
    """

    preference_type: PreferenceType
    value: Any
    strength: float = 0.5
    context: str = "general"
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {self.strength}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def to_attribute(self, name: str | None = None) -> ProfileAttribute:
        """Convert to a ProfileAttribute."""
        attr_name = name or f"pref_{self.preference_type.value}"
        return ProfileAttribute(
            name=attr_name,
            value={
                "type": self.preference_type.value,
                "value": self.value,
                "strength": self.strength,
                "context": self.context,
            },
            confidence=self.confidence,
            source="preference_analysis",
            timestamp=self.timestamp,
        )


# ------------------------------------------------------------------ #
# Knowledge — 用户知识画像
# ------------------------------------------------------------------ #

class KnowledgeLevel(str, Enum):
    """知识水平."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass(frozen=True, slots=True)
class Knowledge:
    """用户知识领域画像.

    Attributes:
        domain: 知识领域（如 "machine_learning", "web_development"）.
        level: 知识水平.
        topics: 具体主题列表.
        gaps: 知识缺口/薄弱环节.
        confidence: 置信度.
    """

    domain: str
    level: KnowledgeLevel
    topics: tuple[str, ...] = field(default_factory=tuple)
    gaps: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def to_attribute(self, name: str | None = None) -> ProfileAttribute:
        """Convert to a ProfileAttribute."""
        attr_name = name or f"knowledge_{self.domain}"
        return ProfileAttribute(
            name=attr_name,
            value={
                "domain": self.domain,
                "level": self.level.value,
                "topics": list(self.topics),
                "gaps": list(self.gaps),
            },
            confidence=self.confidence,
            source="knowledge_analysis",
            timestamp=self.timestamp,
        )


# ------------------------------------------------------------------ #
# InteractionStyle — 交互风格画像
# ------------------------------------------------------------------ #

class CommunicationStyle(str, Enum):
    """沟通风格."""

    CONCISE = "concise"           # 简洁
    DETAILED = "detailed"         # 详细
    TECHNICAL = "technical"       # 技术化
    CONVERSATIONAL = "conversational"  # 对话式
    FORMAL = "formal"             # 正式
    CASUAL = "casual"             # 随意
    VISUAL = "visual"             # 视觉导向
    STRUCTURED = "structured"     # 结构化


@dataclass(frozen=True, slots=True)
class InteractionStyle:
    """用户交互风格画像.

    Attributes:
        communication: 主要沟通风格.
        response_speed: 期望响应速度（"immediate", "thoughtful", "async"）.
        initiative: 主动性 0.0~1.0（高=用户喜欢主动引导）.
        detail_level: 细节偏好 0.0~1.0.
        follow_up: 是否喜欢跟进/追问.
        confidence: 置信度.
    """

    communication: CommunicationStyle
    response_speed: str = "thoughtful"
    initiative: float = 0.5
    detail_level: float = 0.5
    follow_up: bool = True
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.initiative <= 1.0:
            raise ValueError(
                f"initiative must be in [0.0, 1.0], got {self.initiative}"
            )
        if not 0.0 <= self.detail_level <= 1.0:
            raise ValueError(
                f"detail_level must be in [0.0, 1.0], got {self.detail_level}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def to_attribute(self, name: str = "interaction_style") -> ProfileAttribute:
        """Convert to a ProfileAttribute."""
        return ProfileAttribute(
            name=name,
            value={
                "communication": self.communication.value,
                "response_speed": self.response_speed,
                "initiative": self.initiative,
                "detail_level": self.detail_level,
                "follow_up": self.follow_up,
            },
            confidence=self.confidence,
            source="interaction_analysis",
            timestamp=self.timestamp,
        )
