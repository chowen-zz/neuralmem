"""NeuralMem storage backends."""
from neuralmem.storage.sqlite import SQLiteStorage

try:
    from neuralmem.storage.pgvector import PgVectorStorage
except ImportError:
    PgVectorStorage = None  # type: ignore[assignment,misc]

try:
    from neuralmem.storage.factory import VectorStoreFactory
except ImportError:
    VectorStoreFactory = None  # type: ignore[assignment,misc]

__all__ = [
    "SQLiteStorage",
    "PgVectorStorage",
    "VectorStoreFactory",
]
