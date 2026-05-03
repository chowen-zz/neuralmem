"""Memory instructions — manage per-user/agent/global extraction instructions."""
from neuralmem.instructions.builtins import (
    BUILTIN_INSTRUCTIONS,
    BuiltinInstruction,
    apply_builtin,
    get_builtin,
    list_builtins,
)
from neuralmem.instructions.manager import Instruction, InstructionManager, InstructionScope

__all__ = [
    "InstructionManager",
    "Instruction",
    "InstructionScope",
    "BuiltinInstruction",
    "BUILTIN_INSTRUCTIONS",
    "get_builtin",
    "list_builtins",
    "apply_builtin",
]
