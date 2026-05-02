"""NeuralMem — Memory as Infrastructure. Local-first, MCP-native agent memory."""
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.memory import NeuralMem
from neuralmem.core.types import (
    Entity,
    Memory,
    MemoryScope,
    MemoryType,
    Relation,
    SearchResult,
    SessionLayer,
)

__version__ = "0.2.0"

__all__ = [
    "NeuralMem",
    "Memory",
    "MemoryType",
    "MemoryScope",
    "SessionLayer",
    "Entity",
    "Relation",
    "SearchResult",
    "NeuralMemConfig",
    "__version__",
]
