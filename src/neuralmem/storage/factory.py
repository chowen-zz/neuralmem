"""Vector store backend factory."""
from __future__ import annotations

import importlib
import logging
from typing import Any

from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

# Maps backend name -> (module_path, class_name, pip_extra)
_BACKEND_REGISTRY: dict[str, tuple[str, str, str]] = {}


class VectorStoreFactory:
    """Registry and factory for vector store backends.

    Usage::

        VectorStoreFactory.register(
            'chroma',
            'neuralmem.storage.chroma',
            'ChromaVectorStore',
            'chroma',
        )
        backend = VectorStoreFactory.create('chroma', config={...})
    """

    _registry: dict[str, tuple[str, str, str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        module_path: str,
        class_name: str,
        pip_extra: str,
    ) -> None:
        """Register a backend under *name*.

        Parameters
        ----------
        name:
            Short identifier used in ``create()``.
        module_path:
            Fully-qualified module, e.g. ``'neuralmem.storage.chroma'``.
        class_name:
            Class inside that module.
        pip_extra:
            Extra to recommend in error messages, e.g. ``'chroma'``.
        """
        cls._registry[name] = (module_path, class_name, pip_extra)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> StorageBackend:
        """Instantiate and return a backend by *name*.

        Raises ``ValueError`` if the backend is unknown or its dependency
        is not installed.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise ValueError(
                f"Unknown vector store backend '{name}'. "
                f"Available: {available}"
            )

        module_path, class_name, pip_extra = cls._registry[name]

        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Backend '{name}' requires extra dependencies. "
                f"Install with: pip install neuralmem[{pip_extra}]"
            ) from exc

        cls_: type[StorageBackend] = getattr(mod, class_name)
        return cls_(**kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def list_backends(cls) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(cls._registry)

    @classmethod
    def available_backends(cls) -> list[str]:
        """Return backends whose dependencies are currently importable."""
        result: list[str] = []
        for name, (module_path, _cls, _extra) in cls._registry.items():
            try:
                importlib.import_module(module_path)
                result.append(name)
            except ImportError:
                pass
        return sorted(result)

    @classmethod
    def check_dependency(cls, name: str) -> bool:
        """Return *True* if the backend's dependency is installed."""
        if name not in cls._registry:
            return False
        module_path, _cls, _extra = cls._registry[name]
        try:
            importlib.import_module(module_path)
            return True
        except ImportError:
            return False

    @classmethod
    def registry_info(cls) -> dict[str, dict[str, str]]:
        """Return a dict of ``{name: {module, class, extra, available}}``."""
        info: dict[str, dict[str, str]] = {}
        for name, (mod, cls_name, extra) in cls._registry.items():
            info[name] = {
                "module": mod,
                "class": cls_name,
                "extra": extra,
                "available": str(cls.check_dependency(name)),
            }
        return info


# ------------------------------------------------------------------
# Auto-register all built-in vector store backends
# ------------------------------------------------------------------

_BUILTIN_BACKENDS: list[tuple[str, str, str, str]] = [
    ("chroma", "neuralmem.storage.chroma", "ChromaVectorStore", "chroma"),
    ("qdrant", "neuralmem.storage.qdrant_store", "QdrantVectorStore", "qdrant"),
    ("faiss", "neuralmem.storage.faiss_store", "FAISSVectorStore", "faiss"),
    ("redis", "neuralmem.storage.redis_store", "RedisVectorStore", "redis"),
]

for _name, _mod, _cls, _extra in _BUILTIN_BACKENDS:
    VectorStoreFactory.register(_name, _mod, _cls, _extra)
