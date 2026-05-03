"""对话记忆提取器 — 从多轮对话中提取结构化记忆"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from neuralmem.core.types import MemoryType

_logger = logging.getLogger(__name__)

# Language detection patterns
_CJK_RANGES = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff'
    r'\u3000-\u303f\uff00-\uffef]'
)
_CYRILLIC_RANGES = re.compile(r'[\u0400-\u04ff]')
_ARABIC_RANGES = re.compile(r'[\u0600-\u06ff]')

# Preference signal keywords per language
_PREFERENCE_SIGNALS = {
    'zh': [
        '喜欢', '偏好', '爱', '更喜欢', '首选', '习惯',
        '倾向于', '觉得好', '感觉', '希望', '想要',
    ],
    'en': [
        'prefer', 'like', 'love', 'favorite', 'want',
        'enjoy', 'rather', 'better', 'wish', 'usually',
    ],
}

# Fact signal keywords per language
_FACT_SIGNALS = {
    'zh': [
        '是', '在', '有', '叫做', '属于', '位于',
        '成立于', '大约', '总共', '包含', '使用',
    ],
    'en': [
        'is', 'are', 'was', 'were', 'has', 'have',
        'works', 'lives', 'located', 'founded', 'uses',
        'contains', 'includes', 'consists', 'born',
    ],
}

# Procedural signal keywords
_PROCEDURAL_SIGNALS = {
    'zh': ['步骤', '首先', '然后', '最后', '方法', '流程', '先', '再', '接着'],
    'en': [
        'step', 'first', 'then', 'finally', 'how to',
        'process', 'method', 'procedure', 'next', 'after',
    ],
}

# Episodic signal keywords
_EPISODIC_SIGNALS = {
    'zh': [
        '昨天', '今天', '上周', '去年', '刚才',
        '之前', '发生', '记得', '那时候', '当时',
    ],
    'en': [
        'yesterday', 'today', 'last week', 'last year',
        'ago', 'happened', 'remember', 'that time',
        'earlier', 'before',
    ],
}


@dataclass
class ExtractedMemory:
    """A single memory extracted from conversation."""
    content: str
    memory_type: MemoryType
    source_role: str = "user"
    confidence: float = 0.8
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class ConversationExtractor:
    """Extract structured memories from multi-turn dialogue.

    Takes a list of message dicts with 'role' and 'content' keys,
    and produces ExtractedMemory items with type classification.

    Supports language detection and extracts facts in the same
    language as the source conversation.
    """

    def __init__(self) -> None:
        self._lang_detectors = {
            'zh': _CJK_RANGES,
            'ru': _CYRILLIC_RANGES,
            'ar': _ARABIC_RANGES,
        }

    def extract(
        self,
        messages: list[dict[str, str]],
        memory_type: MemoryType | None = None,
    ) -> list[ExtractedMemory]:
        """Extract memories from a conversation (list of messages).

        Args:
            messages: List of dicts with 'role' and 'content' keys.
            memory_type: Override memory type for all extracted items.
                         If None, type is inferred per statement.

        Returns:
            List of ExtractedMemory items.
        """
        if not messages:
            return []

        # Detect primary language from the conversation
        all_text = " ".join(
            m.get("content", "") for m in messages if m.get("content")
        )
        lang = self._detect_language(all_text)

        memories: list[ExtractedMemory] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if not content or len(content) < 3:
                continue

            statements = self._split_statements(content, lang)
            for stmt in statements:
                stmt = stmt.strip()
                if not stmt or len(stmt) < 3:
                    continue
                mtype = memory_type or self._infer_type(stmt, lang)
                conf = self._score_confidence(stmt, role, mtype)
                tags = self._extract_tags(stmt, lang)
                memories.append(ExtractedMemory(
                    content=stmt,
                    memory_type=mtype,
                    source_role=role,
                    confidence=conf,
                    tags=tags,
                    metadata={"detected_language": lang},
                ))

        # Merge adjacent short statements from same role
        memories = self._merge_short_adjacent(memories)
        return memories

    def _detect_language(self, text: str) -> str:
        """Detect the dominant language of a text block."""
        if not text:
            return 'en'

        total_chars = max(len(text), 1)
        for lang, pattern in self._lang_detectors.items():
            cjk_count = len(pattern.findall(text))
            if cjk_count / total_chars > 0.1:
                return lang
        return 'en'

    def _split_statements(self, text: str, lang: str) -> list[str]:
        """Split text into individual statements."""
        if lang == 'zh':
            # Split on Chinese sentence terminators
            parts = re.split(r'[。！？；\n]+', text)
        else:
            # Split on common sentence terminators
            parts = re.split(r'[.!?\n]+', text)
        return [p.strip() for p in parts if p.strip()]

    def _infer_type(self, text: str, lang: str) -> MemoryType:
        """Infer memory type from text content and language."""
        text_lower = text.lower()

        # Check preference signals
        for kw in _PREFERENCE_SIGNALS.get(lang, _PREFERENCE_SIGNALS['en']):
            if kw in text_lower:
                return MemoryType.PREFERENCE

        # Check procedural signals
        for kw in _PROCEDURAL_SIGNALS.get(lang, _PROCEDURAL_SIGNALS['en']):
            if kw in text_lower:
                return MemoryType.PROCEDURAL

        # Check episodic signals
        for kw in _EPISODIC_SIGNALS.get(lang, _EPISODIC_SIGNALS['en']):
            if kw in text_lower:
                return MemoryType.EPISODIC

        # Check fact signals (less specific, lower priority)
        for kw in _FACT_SIGNALS.get(lang, _FACT_SIGNALS['en']):
            if kw in text_lower:
                return MemoryType.FACT

        # Default to SEMANTIC for general knowledge
        return MemoryType.SEMANTIC

    def _score_confidence(
        self, text: str, role: str, mtype: MemoryType
    ) -> float:
        """Score confidence of the extracted memory."""
        base = 0.7

        # User statements are generally higher confidence for
        # preferences and facts about themselves
        if role == "user":
            if mtype is MemoryType.PREFERENCE:
                base += 0.1
            elif mtype is MemoryType.FACT:
                base += 0.05

        # Assistant statements are informational
        if role == "assistant":
            base += 0.05

        # Longer statements tend to be more informative
        if len(text) > 50:
            base += 0.05
        if len(text) > 100:
            base += 0.05

        return min(base, 1.0)

    def _extract_tags(self, text: str, lang: str) -> list[str]:
        """Extract relevant tags from the statement."""
        tags: list[str] = []
        text_lower = text.lower()

        # Common topic tags
        tech_kw = [
            'python', 'javascript', 'typescript', 'react', 'vue',
            'docker', 'kubernetes', 'api', 'sql', 'git',
        ]
        if any(kw in text_lower for kw in tech_kw):
            tags.append('technology')

        if lang == 'zh':
            if any(w in text_lower for w in ['工作', '项目', '任务']):
                tags.append('work')
            if any(w in text_lower for w in ['问题', '错误', 'bug']):
                tags.append('issue')
        else:
            if any(w in text_lower for w in ['work', 'project', 'task']):
                tags.append('work')
            if any(w in text_lower for w in ['error', 'bug', 'issue']):
                tags.append('issue')

        return tags

    def _merge_short_adjacent(
        self, memories: list[ExtractedMemory]
    ) -> list[ExtractedMemory]:
        """Merge adjacent very short statements from same role."""
        if len(memories) <= 1:
            return memories

        merged: list[ExtractedMemory] = []
        i = 0
        while i < len(memories):
            curr = memories[i]
            # Try to merge with next if both are short
            if (
                i + 1 < len(memories)
                and len(curr.content) < 30
                and len(memories[i + 1].content) < 30
                and curr.source_role == memories[i + 1].source_role
                and curr.memory_type == memories[i + 1].memory_type
            ):
                next_mem = memories[i + 1]
                combined = ExtractedMemory(
                    content=f"{curr.content}，{next_mem.content}",
                    memory_type=curr.memory_type,
                    source_role=curr.source_role,
                    confidence=max(curr.confidence, next_mem.confidence),
                    tags=list(set(curr.tags + next_mem.tags)),
                    metadata=curr.metadata,
                )
                merged.append(combined)
                i += 2
            else:
                merged.append(curr)
                i += 1
        return merged
