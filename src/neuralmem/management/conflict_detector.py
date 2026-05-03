"""Conflict detection and resolution between memory sets."""
from __future__ import annotations

import json
import logging
from typing import Any

from neuralmem.management.llm_manager import (
    LLMMemoryManager,
    MemoryOperation,
    OperationType,
)
from neuralmem.management.prompts import CONFLICT_PROMPT

_logger = logging.getLogger(__name__)


class ConflictDetector:
    """Detects and resolves conflicts between new and existing memories.

    Uses LLM to identify contradictions, supersessions, and negations,
    then generates UPDATE/DELETE operations to resolve them.
    """

    def __init__(
        self,
        llm_backend: str = "ollama",
        model: str | None = None,
        **kwargs: object,
    ) -> None:
        self._manager = LLMMemoryManager(
            llm_backend=llm_backend,
            model=model,
            **kwargs,
        )

    async def detect_conflicts(
        self,
        new_memories: list[str],
        existing_memories: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Detect conflicts between new and existing memories.

        Args:
            new_memories: List of new memory content strings.
            existing_memories: List of dicts with 'id' and 'content' keys.

        Returns:
            List of conflict dicts with keys:
                new_memory, existing_memory_id, conflict_type, resolution
        """
        if not new_memories or not existing_memories:
            return []

        new_str = json.dumps(new_memories, indent=2)
        existing_str = json.dumps(existing_memories, indent=2)

        prompt = CONFLICT_PROMPT.format(
            new_memories=new_str,
            existing_memories=existing_str,
        )

        raw = await self._manager._call_llm(prompt)
        result = await self._manager._parse_json_response(raw)

        if not isinstance(result, list):
            return []

        conflicts: list[dict[str, Any]] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            conflict_type = item.get("conflict_type", "contradiction")
            # Validate conflict_type
            valid_types = {"contradiction", "supersession", "negation"}
            if conflict_type not in valid_types:
                conflict_type = "contradiction"
            conflicts.append({
                "new_memory": item.get("new_memory", ""),
                "existing_memory_id": item.get("existing_memory_id", ""),
                "conflict_type": conflict_type,
                "resolution": item.get("resolution", "update"),
            })

        return conflicts

    async def auto_resolve(
        self,
        conflicts: list[dict[str, Any]],
    ) -> list[MemoryOperation]:
        """Generate operations to resolve detected conflicts.

        For contradictions and supersessions: creates UPDATE operations.
        For negations: creates DELETE operations.
        """
        operations: list[MemoryOperation] = []
        for conflict in conflicts:
            conflict_type = conflict.get("conflict_type", "contradiction")
            resolution = conflict.get("resolution", "update")
            existing_id = conflict.get("existing_memory_id")
            new_memory = conflict.get("new_memory", "")

            if conflict_type == "negation" or resolution == "delete":
                operations.append(
                    MemoryOperation(
                        op_type=OperationType.DELETE,
                        content=new_memory,
                        old_memory_id=existing_id,
                        confidence=0.9,
                    )
                )
            else:
                # contradiction or supersession → UPDATE
                operations.append(
                    MemoryOperation(
                        op_type=OperationType.UPDATE,
                        content=new_memory,
                        old_memory_id=existing_id,
                        confidence=0.85,
                    )
                )

        return operations
