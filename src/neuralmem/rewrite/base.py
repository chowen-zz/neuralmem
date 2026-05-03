"""ContextRewriter 抽象基类与共享数据模型。"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Memory, MemoryType
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)


class LLMCaller(Protocol):
    """协议：任何能接收 prompt 返回字符串的 LLM 调用器。"""

    def __call__(self, prompt: str) -> str: ...


@dataclass
class RewriteResult:
    """一次重写/扫描的结果。"""

    new_summaries: list[Memory] = field(default_factory=list)
    updated_summaries: list[Memory] = field(default_factory=list)
    connections_found: list[dict[str, Any]] = field(default_factory=list)
    memories_archived: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def merge(self, other: RewriteResult) -> RewriteResult:
        return RewriteResult(
            new_summaries=self.new_summaries + other.new_summaries,
            updated_summaries=self.updated_summaries + other.updated_summaries,
            connections_found=self.connections_found + other.connections_found,
            memories_archived=self.memories_archived + other.memories_archived,
            errors=self.errors + other.errors,
        )


class ContextRewriter(ABC):
    """上下文重写器抽象基类。

    子类实现 ``rewrite()`` 以执行具体的重写逻辑：
    - 扫描记忆库中相关记忆
    - 合并为更高层次摘要
    - 发现隐性连接
    """

    def __init__(
        self,
        config: NeuralMemConfig,
        storage: StorageBackend,
        llm_caller: LLMCaller | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._llm = llm_caller

    # ------------------------------------------------------------------
    # 共享工具
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        if self._llm is None:
            raise RuntimeError("No LLM caller available")
        raw = self._llm(prompt)
        return (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

    def _parse_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            _logger.warning("LLM returned invalid JSON: %s", exc)
            return {}

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _make_summary_memory(
        self,
        content: str,
        source_memory_ids: list[str],
        importance: float = 0.7,
        tags: tuple[str, ...] = (),
        user_id: str | None = None,
    ) -> Memory:
        return Memory(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            tags=tags + ("summary", "auto-generated"),
            source="context_rewriter",
            importance=importance,
            user_id=user_id,
            entity_ids=tuple(source_memory_ids),
        )

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def rewrite(
        self,
        user_id: str | None = None,
        memory_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> RewriteResult:
        """执行重写/扫描并返回结果。"""
        ...
