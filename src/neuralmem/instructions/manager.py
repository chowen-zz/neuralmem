"""InstructionManager — CRUD for memory extraction instructions.

Instructions are injected into LLM prompts to guide how memories
are extracted and stored (e.g. language, focus areas, privacy).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone

from neuralmem.core.types import _generate_ulid


class InstructionScope(enum.Enum):
    """Scope determines which context an instruction applies to."""

    GLOBAL = "global"
    USER = "user"
    AGENT = "agent"
    SESSION = "session"


@dataclass
class Instruction:
    """A single extraction instruction."""

    id: str
    name: str
    content: str
    scope: InstructionScope = InstructionScope.GLOBAL
    scope_id: str | None = None
    priority: int = 0
    enabled: bool = True
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class InstructionManager:
    """Manage extraction instructions with CRUD operations."""

    def __init__(self) -> None:
        self._instructions: dict[str, Instruction] = {}

    def add(
        self,
        name: str,
        content: str,
        scope: InstructionScope = InstructionScope.GLOBAL,
        scope_id: str | None = None,
        priority: int = 0,
    ) -> Instruction:
        """Create and store a new instruction. Returns the Instruction."""
        inst = Instruction(
            id=_generate_ulid(),
            name=name,
            content=content,
            scope=scope,
            scope_id=scope_id,
            priority=priority,
        )
        self._instructions[inst.id] = inst
        return inst

    def remove(self, instruction_id: str) -> bool:
        """Remove an instruction by id. Returns True if found."""
        return self._instructions.pop(instruction_id, None) is not None

    def get(self, instruction_id: str) -> Instruction | None:
        """Get an instruction by id."""
        return self._instructions.get(instruction_id)

    def list(
        self,
        scope: InstructionScope | None = None,
        scope_id: str | None = None,
    ) -> list[Instruction]:
        """List instructions, optionally filtered by scope and scope_id."""
        result = list(self._instructions.values())
        if scope is not None:
            result = [i for i in result if i.scope is scope]
        if scope_id is not None:
            result = [i for i in result if i.scope_id == scope_id]
        return result

    def get_prompt_injection(
        self,
        scope: InstructionScope = InstructionScope.GLOBAL,
        scope_id: str | None = None,
    ) -> str:
        """Return a formatted instruction string to inject into LLM prompts.

        Only enabled instructions matching the given scope/scope_id are
        included, sorted by priority (highest first).
        """
        items = [
            i
            for i in self.list(scope=scope, scope_id=scope_id)
            if i.enabled
        ]
        items.sort(key=lambda i: i.priority, reverse=True)
        if not items:
            return ""
        lines = [f"- {i.content}" for i in items]
        return "Instructions:\n" + "\n".join(lines)

    def clear(self, scope: InstructionScope | None = None) -> int:
        """Remove instructions, optionally filtered by scope.

        Returns count of removed instructions.
        """
        if scope is None:
            count = len(self._instructions)
            self._instructions.clear()
            return count
        to_remove = [
            k
            for k, v in self._instructions.items()
            if v.scope is scope
        ]
        for k in to_remove:
            del self._instructions[k]
        return len(to_remove)

    def to_dict(self) -> dict:
        """Serialize all instructions to a dict."""
        return {
            "instructions": [
                {
                    "id": i.id,
                    "name": i.name,
                    "content": i.content,
                    "scope": i.scope.value,
                    "scope_id": i.scope_id,
                    "priority": i.priority,
                    "enabled": i.enabled,
                    "created_at": i.created_at.isoformat(),
                }
                for i in self._instructions.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> InstructionManager:
        """Deserialize an InstructionManager from a dict."""
        mgr = cls()
        for item in data.get("instructions", []):
            inst = Instruction(
                id=item["id"],
                name=item["name"],
                content=item["content"],
                scope=InstructionScope(item["scope"]),
                scope_id=item.get("scope_id"),
                priority=item.get("priority", 0),
                enabled=item.get("enabled", True),
                created_at=datetime.fromisoformat(item["created_at"]),
            )
            mgr._instructions[inst.id] = inst
        return mgr
