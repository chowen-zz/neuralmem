"""画像引擎 — ProfileEngine: 从记忆行为推断用户画像.

分析用户的查询模式、记忆内容和交互历史来推断画像维度。
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from neuralmem.core.types import Memory, MemoryType
from neuralmem.profiles.base import ProfileAttribute, UserProfile
from neuralmem.profiles.types import (
    CommunicationStyle,
    Intent,
    IntentCategory,
    InteractionStyle,
    Knowledge,
    KnowledgeLevel,
    Preference,
    PreferenceType,
)

if TYPE_CHECKING:
    from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Keyword patterns for intent classification
# ------------------------------------------------------------------ #

_INTENT_PATTERNS: dict[IntentCategory, list[str]] = {
    IntentCategory.INFORMATIONAL: [
        "what", "how", "why", "explain", "tell me", "information",
        "了解", "什么是", "怎么", "为什么", "解释",
    ],
    IntentCategory.TRANSACTIONAL: [
        "create", "make", "build", "generate", "write", "do",
        "创建", "生成", "写", "构建", "制作",
    ],
    IntentCategory.TROUBLESHOOTING: [
        "fix", "error", "bug", "issue", "problem", "not working",
        "修复", "错误", "bug", "问题", "故障",
    ],
    IntentCategory.LEARNING: [
        "learn", "tutorial", "guide", "course", "study",
        "学习", "教程", "指南", "课程",
    ],
    IntentCategory.CREATIVE: [
        "design", "idea", "brainstorm", "creative", "story",
        "设计", "创意", "头脑风暴", "故事",
    ],
    IntentCategory.SOCIAL: [
        "chat", "talk", "discuss", "opinion", "think",
        "聊天", "讨论", "看法", "观点",
    ],
}

# Preference patterns
_PREFERENCE_PATTERNS: dict[PreferenceType, dict[str, list[str]]] = {
    PreferenceType.TECHNOLOGY: {
        "python": ["python", "py"],
        "javascript": ["javascript", "js", "node"],
        "typescript": ["typescript", "ts"],
        "rust": ["rust", "rs"],
        "go": ["go", "golang"],
        "java": ["java"],
        "cpp": ["c++", "cpp"],
    },
    PreferenceType.FORMAT: {
        "json": ["json"],
        "markdown": ["markdown", "md"],
        "code": ["code", "snippet"],
        "table": ["table", "表格"],
    },
    PreferenceType.TONE: {
        "formal": ["formal", "professional", "正式"],
        "casual": ["casual", "friendly", "随意"],
        "technical": ["technical", "技术"],
        "humorous": ["humor", "funny", "幽默"],
    },
    PreferenceType.DEPTH: {
        "brief": ["brief", "short", "简洁", "简单"],
        "detailed": ["detailed", "thorough", "详细", "深入"],
    },
    PreferenceType.LANGUAGE: {
        "chinese": ["chinese", "中文", "汉语"],
        "english": ["english", "英文", "英语"],
        "japanese": ["japanese", "日文", "日语"],
    },
}

# Knowledge domain patterns
_KNOWLEDGE_PATTERNS: dict[str, list[str]] = {
    "machine_learning": ["ml", "machine learning", "深度学习", "神经网络", "模型训练"],
    "web_development": ["web", "frontend", "backend", "html", "css", "react", "vue"],
    "data_science": ["data", "pandas", "numpy", "数据分析", "可视化"],
    "devops": ["docker", "kubernetes", "k8s", "ci/cd", "部署", "运维"],
    "mobile": ["android", "ios", "flutter", "react native", "移动端"],
    "security": ["security", "安全", "加密", "漏洞", "渗透"],
}

# Communication style indicators
_STYLE_INDICATORS: dict[CommunicationStyle, list[str]] = {
    CommunicationStyle.CONCISE: ["brief", "short", "quick", "简单", "简洁"],
    CommunicationStyle.DETAILED: ["detailed", "thorough", "comprehensive", "详细", "全面"],
    CommunicationStyle.TECHNICAL: ["technical", "implementation", "algorithm", "技术", "实现"],
    CommunicationStyle.CONVERSATIONAL: ["chat", "discuss", "talk", "聊聊", "讨论"],
    CommunicationStyle.FORMAL: ["formal", "professional", "official", "正式", "官方"],
    CommunicationStyle.CASUAL: ["casual", "relaxed", "随意", "轻松"],
    CommunicationStyle.VISUAL: ["diagram", "chart", "image", "visual", "图", "图表"],
    CommunicationStyle.STRUCTURED: ["step", "structure", "outline", "步骤", "结构"],
}


class ProfileEngine:
    """画像引擎：从记忆行为推断用户画像.

    分析用户的查询模式、记忆内容和交互历史，推断出意图、偏好、
    知识水平和交互风格等画像维度。

    Args:
        storage: 可选的存储后端，用于获取用户历史记忆.
    """

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self.storage = storage
        self._profiles: dict[str, UserProfile] = {}

    # ------------------------------------------------------------------ #
    # Core inference methods
    # ------------------------------------------------------------------ #

    def infer_intent(self, memories: Sequence[Memory]) -> dict[str, ProfileAttribute]:
        """Infer user intent from query memories.

        Args:
            memories: Sequence of memories (typically EPISODIC or WORKING type).

        Returns:
            Dict mapping attribute names to ProfileAttribute instances.

        Raises:
            ValueError: If memories is empty.
        """
        if not memories:
            raise ValueError("memories must not be empty for intent inference")

        category_scores: Counter[IntentCategory] = Counter()
        all_keywords: list[str] = []
        evidence: list[str] = []

        for mem in memories:
            content = mem.content.lower()
            evidence.append(mem.id)

            for category, patterns in _INTENT_PATTERNS.items():
                score = sum(1 for p in patterns if p in content)
                if score > 0:
                    category_scores[category] += score
                    all_keywords.extend(p for p in patterns if p in content)

        if not category_scores:
            # Default to informational if no strong signals
            primary = IntentCategory.INFORMATIONAL
            confidence = 0.3
        else:
            primary, count = category_scores.most_common(1)[0]
            total = sum(category_scores.values())
            confidence = min(0.95, count / max(total * 0.5, 1))

        # Deduplicate keywords
        unique_keywords = tuple(sorted(set(all_keywords)))[:10]

        intent = Intent(
            category=primary,
            subcategory="",
            keywords=unique_keywords,
            confidence=round(confidence, 3),
        )

        return {"primary_intent": intent.to_attribute()}

    def infer_preferences(self, memories: Sequence[Memory]) -> dict[str, ProfileAttribute]:
        """Infer user preferences from memory content.

        Args:
            memories: Sequence of memories to analyze.

        Returns:
            Dict mapping preference attribute names to ProfileAttribute instances.

        Raises:
            ValueError: If memories is empty.
        """
        if not memories:
            raise ValueError("memories must not be empty for preference inference")

        results: dict[str, ProfileAttribute] = {}

        for pref_type, patterns in _PREFERENCE_PATTERNS.items():
            scores: Counter[str] = Counter()
            evidence: list[str] = []

            for mem in memories:
                content = mem.content.lower()
                for value, keywords in patterns.items():
                    score = sum(1 for kw in keywords if kw in content)
                    if score > 0:
                        scores[value] += score
                        evidence.append(mem.id)

            if scores:
                top_value, top_count = scores.most_common(1)[0]
                total = sum(scores.values())
                strength = min(0.95, top_count / max(total * 0.6, 1))
                confidence = min(0.9, 0.4 + strength * 0.5)

                pref = Preference(
                    preference_type=pref_type,
                    value=top_value,
                    strength=round(strength, 3),
                    context="general",
                    confidence=round(confidence, 3),
                )
                attr = pref.to_attribute()
                # Override evidence with actual memory IDs
                attr_with_evidence = ProfileAttribute(
                    name=attr.name,
                    value=attr.value,
                    confidence=attr.confidence,
                    source=attr.source,
                    timestamp=attr.timestamp,
                    evidence=tuple(sorted(set(evidence)))[:5],
                )
                results[attr.name] = attr_with_evidence

        return results

    def infer_knowledge(self, memories: Sequence[Memory]) -> dict[str, ProfileAttribute]:
        """Infer user knowledge domains and levels from memories.

        Args:
            memories: Sequence of memories to analyze.

        Returns:
            Dict mapping knowledge attribute names to ProfileAttribute instances.

        Raises:
            ValueError: If memories is empty.
        """
        if not memories:
            raise ValueError("memories must not be empty for knowledge inference")

        domain_scores: Counter[str] = Counter()
        evidence_map: dict[str, list[str]] = {}

        for mem in memories:
            content = mem.content.lower()
            for domain, keywords in _KNOWLEDGE_PATTERNS.items():
                score = sum(1 for kw in keywords if kw in content)
                if score > 0:
                    domain_scores[domain] += score
                    evidence_map.setdefault(domain, []).append(mem.id)

        results: dict[str, ProfileAttribute] = {}

        for domain, count in domain_scores.most_common(5):
            # Infer level based on content complexity indicators
            level = self._estimate_knowledge_level(memories, domain)
            confidence = min(0.85, 0.4 + count * 0.1)

            knowledge = Knowledge(
                domain=domain,
                level=level,
                topics=tuple(),
                gaps=tuple(),
                confidence=round(confidence, 3),
            )
            attr = knowledge.to_attribute()
            evidence = tuple(sorted(set(evidence_map.get(domain, []))))[:5]
            results[attr.name] = ProfileAttribute(
                name=attr.name,
                value=attr.value,
                confidence=attr.confidence,
                source=attr.source,
                timestamp=attr.timestamp,
                evidence=evidence,
            )

        return results

    def infer_interaction_style(self, memories: Sequence[Memory]) -> dict[str, ProfileAttribute]:
        """Infer user interaction style from memory patterns.

        Args:
            memories: Sequence of memories to analyze.

        Returns:
            Dict with interaction style attribute.

        Raises:
            ValueError: If memories is empty.
        """
        if not memories:
            raise ValueError("memories must not be empty for style inference")

        style_scores: Counter[CommunicationStyle] = Counter()
        total_length = 0
        question_count = 0
        follow_up_indicators = 0

        for mem in memories:
            content = mem.content
            total_length += len(content)

            # Count questions
            question_count += content.count("?") + content.count("？")

            # Check for follow-up indicators
            follow_words = ["also", "another", "additionally", "还有", "另外", "再"]
            follow_up_indicators += sum(1 for w in follow_words if w in content.lower())

            for style, indicators in _STYLE_INDICATORS.items():
                score = sum(1 for ind in indicators if ind in content.lower())
                if score > 0:
                    style_scores[style] += score

        # Determine primary style
        if style_scores:
            primary_style, style_count = style_scores.most_common(1)[0]
            total_style = sum(style_scores.values())
            style_confidence = min(0.9, style_count / max(total_style * 0.5, 1))
        else:
            primary_style = CommunicationStyle.CONVERSATIONAL
            style_confidence = 0.3

        # Estimate detail level from average message length
        avg_length = total_length / len(memories) if memories else 0
        if avg_length > 200:
            detail_level = 0.8
        elif avg_length > 100:
            detail_level = 0.6
        else:
            detail_level = 0.4

        # Estimate initiative from question ratio
        question_ratio = question_count / len(memories) if memories else 0
        initiative = min(0.9, 0.3 + question_ratio * 0.3)

        # Follow-up preference
        follow_up = follow_up_indicators >= len(memories) * 0.1

        style = InteractionStyle(
            communication=primary_style,
            response_speed="thoughtful",
            initiative=round(initiative, 3),
            detail_level=round(detail_level, 3),
            follow_up=follow_up,
            confidence=round(style_confidence, 3),
        )

        return {"interaction_style": style.to_attribute()}

    def build_profile(
        self,
        user_id: str,
        memories: Sequence[Memory],
    ) -> dict[str, ProfileAttribute]:
        """Build a complete user profile from memories.

        Args:
            user_id: The user identifier.
            memories: All available memories for the user.

        Returns:
            Combined dict of all inferred profile attributes.

        Raises:
            ValueError: If memories is empty.
        """
        if not memories:
            raise ValueError("memories must not be empty to build profile")

        _logger.info("Building profile for user %s from %d memories", user_id, len(memories))

        profile: dict[str, ProfileAttribute] = {}

        try:
            intent_attrs = self.infer_intent(memories)
            profile.update(intent_attrs)
        except Exception as exc:
            _logger.warning("Intent inference failed: %s", exc)

        try:
            pref_attrs = self.infer_preferences(memories)
            profile.update(pref_attrs)
        except Exception as exc:
            _logger.warning("Preference inference failed: %s", exc)

        try:
            knowledge_attrs = self.infer_knowledge(memories)
            profile.update(knowledge_attrs)
        except Exception as exc:
            _logger.warning("Knowledge inference failed: %s", exc)

        try:
            style_attrs = self.infer_interaction_style(memories)
            profile.update(style_attrs)
        except Exception as exc:
            _logger.warning("Style inference failed: %s", exc)

        _logger.info("Profile built with %d attributes for user %s", len(profile), user_id)
        return profile

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _estimate_knowledge_level(
        self, memories: Sequence[Memory], domain: str
    ) -> KnowledgeLevel:
        """Estimate knowledge level for a domain from memory content.

        Args:
            memories: Sequence of memories.
            domain: The knowledge domain.

        Returns:
            Inferred KnowledgeLevel.
        """
        beginner_indicators = ["beginner", "new to", "start", "入门", "新手", "基础"]
        advanced_indicators = [
            "advanced", "optimize", "architecture", "deep dive",
            "高级", "优化", "架构", "深入",
        ]
        expert_indicators = [
            "expert", "contribute", "design pattern", "internals",
            "专家", "源码", "原理", "内核",
        ]

        beginner_score = 0
        advanced_score = 0
        expert_score = 0

        for mem in memories:
            content = mem.content.lower()
            beginner_score += sum(1 for ind in beginner_indicators if ind in content)
            advanced_score += sum(1 for ind in advanced_indicators if ind in content)
            expert_score += sum(1 for ind in expert_indicators if ind in content)

        if expert_score > 0 and expert_score >= advanced_score:
            return KnowledgeLevel.EXPERT
        if advanced_score > 0 and advanced_score >= beginner_score:
            return KnowledgeLevel.ADVANCED
        if beginner_score > 0:
            return KnowledgeLevel.BEGINNER
        return KnowledgeLevel.INTERMEDIATE

    def _fetch_user_memories(self, user_id: str, limit: int = 1000) -> list[Memory]:
        """Fetch memories for a user from storage.

        Args:
            user_id: The user identifier.
            limit: Maximum number of memories to fetch.

        Returns:
            List of Memory objects.

        Raises:
            RuntimeError: If storage backend is not configured.
        """
        if self.storage is None:
            raise RuntimeError("Storage backend not configured")
        try:
            return self.storage.list_memories(user_id=user_id, limit=limit)
        except Exception as exc:
            _logger.error("Failed to fetch memories for user %s: %s", user_id, exc)
            raise RuntimeError(f"Failed to fetch memories: {exc}") from exc
