"""用户画像基类 — UserProfile ABC 与 ProfileAttribute dataclass.

参考 multimodal/base.py 的 ABC 模式，定义画像系统的抽象基类和属性模型。
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuralmem.core.types import Memory

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProfileAttribute:
    """画像维度的单个属性值。

    Attributes:
        name: 属性名称（如 "programming_language"）。
        value: 属性值（如 "Python"）。
        confidence: 置信度，0.0 ~ 1.0。
        source: 属性来源描述（如 "memory_analysis", "explicit_input"）。
        timestamp: 属性首次推断时间。
        evidence: 支撑此属性的证据记忆 ID 列表。
    """

    name: str
    value: Any
    confidence: float = field(default=0.5)
    source: str = field(default="inferred")
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate confidence range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def with_confidence(self, confidence: float) -> "ProfileAttribute":
        """Return a new attribute with updated confidence."""
        return ProfileAttribute(
            name=self.name,
            value=self.value,
            confidence=confidence,
            source=self.source,
            timestamp=self.timestamp,
            evidence=self.evidence,
        )


class UserProfile(abc.ABC):
    """用户画像抽象基类。

    所有具体画像实现必须继承此类。画像从 memory 对象中提取行为特征，
    通过 ``analyze`` 方法分析记忆集合，返回推断出的画像维度。

    Subclasses must implement:
        - ``analyze(self, memories: Sequence[Memory]) -> dict[str, ProfileAttribute]``
        - ``dimension(self) -> str`` — 返回画像维度名称。
    """

    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id
        self._attributes: dict[str, ProfileAttribute] = {}
        self._last_updated: datetime | None = None

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def analyze(self, memories: Sequence[Memory]) -> dict[str, ProfileAttribute]:
        """Analyze a sequence of memories and extract profile attributes.

        Args:
            memories: A sequence of Memory objects to analyze.

        Returns:
            A dict mapping attribute names to ProfileAttribute instances.

        Raises:
            ValueError: If memories is empty or contains invalid types.
        """
        ...

    @property
    @abc.abstractmethod
    def dimension(self) -> str:
        """Return the profile dimension name (e.g. 'intent', 'preference')."""
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def get_attribute(self, name: str) -> ProfileAttribute | None:
        """Retrieve an attribute by name.

        Args:
            name: The attribute name.

        Returns:
            The ProfileAttribute if found, else None.
        """
        return self._attributes.get(name)

    def set_attribute(self, attr: ProfileAttribute) -> None:
        """Set or update an attribute.

        Args:
            attr: The ProfileAttribute to store.
        """
        self._attributes[attr.name] = attr
        self._last_updated = datetime.now(timezone.utc)

    def list_attributes(self) -> list[ProfileAttribute]:
        """Return all stored attributes as a list."""
        return list(self._attributes.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a dictionary."""
        return {
            "user_id": self.user_id,
            "dimension": self.dimension,
            "last_updated": self._last_updated.isoformat() if self._last_updated else None,
            "attributes": {
                name: {
                    "name": attr.name,
                    "value": attr.value,
                    "confidence": attr.confidence,
                    "source": attr.source,
                    "timestamp": attr.timestamp.isoformat(),
                    "evidence": list(attr.evidence),
                }
                for name, attr in self._attributes.items()
            },
        }

    @classmethod
    def _extract_keywords(cls, text: str, min_length: int = 3) -> list[str]:
        """Extract candidate keywords from text.

        Args:
            text: Input text.
            min_length: Minimum keyword length.

        Returns:
            List of lowercase keywords.
        """
        if not text:
            return []
        words = text.lower().split()
        return [w.strip(".,!?;:\"'()[]") for w in words if len(w) >= min_length]

    @classmethod
    def _frequency_count(cls, items: Sequence[str]) -> dict[str, int]:
        """Count frequency of items.

        Args:
            items: Sequence of strings.

        Returns:
            Dict mapping item to count.
        """
        counts: dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return counts

    @staticmethod
    def _check_dependency(name: str) -> bool:
        """Return True if *name* can be imported.

        Logs a warning when a dependency is missing.
        """
        try:
            __import__(name)
            return True
        except ImportError:
            _logger.warning("Optional dependency %r is not installed.", name)
            return False
