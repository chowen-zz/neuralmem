"""NeuralMem Spaces — project-level memory containers with role-based access."""
from __future__ import annotations

from neuralmem.spaces.space import Space, SpaceVisibility
from neuralmem.spaces.membership import SpaceMembership, SpaceRole
from neuralmem.spaces.manager import SpaceManager

__all__ = [
    "Space",
    "SpaceVisibility",
    "SpaceMembership",
    "SpaceRole",
    "SpaceManager",
]
