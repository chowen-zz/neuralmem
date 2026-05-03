"""Predefined built-in instructions for common extraction scenarios."""
from __future__ import annotations

from dataclasses import dataclass

from neuralmem.instructions.manager import Instruction, InstructionManager, InstructionScope


@dataclass
class BuiltinInstruction:
    """A predefined instruction template."""

    name: str
    description: str
    content: str
    scope: InstructionScope


BUILTIN_INSTRUCTIONS: list[BuiltinInstruction] = [
    BuiltinInstruction(
        name="language_en",
        description="Extract and store memories in English",
        content="Always extract and store memories in English",
        scope=InstructionScope.GLOBAL,
    ),
    BuiltinInstruction(
        name="language_zh",
        description="Extract and store memories in Chinese",
        content="\u59cb\u7ec8\u4f7f\u7528\u4e2d\u6587\u63d0\u53d6\u548c\u5b58\u50a8\u8bb0\u5fc6",
        scope=InstructionScope.GLOBAL,
    ),
    BuiltinInstruction(
        name="preference_focus",
        description="Focus on extracting user preferences and opinions",
        content="Focus on extracting user preferences and opinions",
        scope=InstructionScope.USER,
    ),
    BuiltinInstruction(
        name="fact_focus",
        description="Focus on extracting factual information",
        content="Focus on extracting factual information",
        scope=InstructionScope.GLOBAL,
    ),
    BuiltinInstruction(
        name="privacy_mode",
        description="Never store personal identifying information",
        content="Never store personal identifying information",
        scope=InstructionScope.GLOBAL,
    ),
    BuiltinInstruction(
        name="detailed_extraction",
        description="Extract detailed memories with context and reasoning",
        content=(
            "Extract detailed memories with context and reasoning"
        ),
        scope=InstructionScope.GLOBAL,
    ),
    BuiltinInstruction(
        name="concise_extraction",
        description="Extract only the core fact, keep memories concise",
        content="Extract only the core fact, keep memories concise",
        scope=InstructionScope.GLOBAL,
    ),
    BuiltinInstruction(
        name="no_duplicates",
        description="Skip memories that are semantically similar to existing ones",
        content=(
            "Skip memories that are semantically similar to existing ones"
        ),
        scope=InstructionScope.GLOBAL,
    ),
]

_BUILTIN_MAP = {b.name: b for b in BUILTIN_INSTRUCTIONS}


def get_builtin(name: str) -> BuiltinInstruction | None:
    """Look up a built-in instruction by name."""
    return _BUILTIN_MAP.get(name)


def list_builtins() -> list[BuiltinInstruction]:
    """Return all available built-in instructions."""
    return list(BUILTIN_INSTRUCTIONS)


def apply_builtin(
    manager: InstructionManager,
    name: str,
    **kwargs: object,
) -> Instruction | None:
    """Register a built-in instruction into a manager.

    kwargs are forwarded to manager.add() (e.g. priority, scope_id).
    Returns the created Instruction, or None if the builtin doesn't exist.
    """
    builtin = get_builtin(name)
    if builtin is None:
        return None
    return manager.add(
        name=builtin.name,
        content=builtin.content,
        scope=builtin.scope,
        **kwargs,  # type: ignore[arg-type]
    )
