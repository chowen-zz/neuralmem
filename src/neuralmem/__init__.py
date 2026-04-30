"""NeuralMem — Memory as Infrastructure. Local-first, MCP-native agent memory."""
from neuralmem.core.memory import NeuralMem
from neuralmem.core.types import Memory, MemoryType, MemoryScope, Entity, Relation, SearchResult
from neuralmem.core.config import NeuralMemConfig

__version__ = "0.2.0"

__all__ = [
    "NeuralMem",
    "Memory",
    "MemoryType",
    "MemoryScope",
    "Entity",
    "Relation",
    "SearchResult",
    "NeuralMemConfig",
    "__version__",
]
