"""NeuralMem community package exports (V1.3)."""
from __future__ import annotations

from neuralmem.community.sharing import MemorySharing, ShareRecord
from neuralmem.community.collaboration import (
    CollaborationSpace,
    MemberRole,
    SpaceManager,
    SpaceMember,
)
from neuralmem.community.feedback import FeedbackEntry, FeedbackLoop

__all__ = [
    "MemorySharing",
    "ShareRecord",
    "CollaborationSpace",
    "MemberRole",
    "SpaceManager",
    "SpaceMember",
    "FeedbackEntry",
    "FeedbackLoop",
]
