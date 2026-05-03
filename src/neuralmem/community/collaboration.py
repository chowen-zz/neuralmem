"""NeuralMem community features — shared workspace for team memories.

V1.3: CollaborationSpace
  • Create named spaces for teams/projects
  • Add/remove members with roles (owner, editor, viewer)
  • Associate memories to a space
  • In-memory implementation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from neuralmem.core.types import Memory


class MemberRole(str, Enum):
    """Roles within a collaboration space."""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"


@dataclass
class SpaceMember:
    """A member of a collaboration space."""
    principal_id: str
    principal_type: str  # "user" | "agent"
    role: MemberRole
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationSpace:
    """A shared workspace."""
    space_id: str
    name: str
    description: str = ""
    owner_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    members: dict[str, SpaceMember] = field(default_factory=dict)
    memory_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


class SpaceManager:
    """In-memory manager for collaboration spaces.

    Usage:
        mgr = SpaceManager()
        space = mgr.create_space("proj-x", owner_id="u1")
        mgr.add_member(space.space_id, "u2", MemberRole.EDITOR)
        mgr.add_memory(space.space_id, memory.id)
    """

    def __init__(self) -> None:
        self._spaces: dict[str, CollaborationSpace] = {}
        # external memory store reference (populated via register_memory)
        self._memories: dict[str, Memory] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _next_id(self) -> str:
        self._counter += 1
        return f"spc-{self._counter:06d}"

    def _get_space(self, space_id: str) -> CollaborationSpace:
        if space_id not in self._spaces:
            raise KeyError(f"Space {space_id} not found")
        return self._spaces[space_id]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def register_memory(self, memory: Memory) -> None:
        """Index a memory so it can be attached to spaces."""
        self._memories[memory.id] = memory

    def create_space(
        self,
        name: str,
        *,
        owner_id: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CollaborationSpace:
        """Create a new collaboration space."""
        space = CollaborationSpace(
            space_id=self._next_id(),
            name=name,
            description=description,
            owner_id=owner_id,
            metadata=metadata or {},
        )
        # owner is automatically a member
        space.members[owner_id] = SpaceMember(
            principal_id=owner_id,
            principal_type="user",
            role=MemberRole.OWNER,
        )
        self._spaces[space.space_id] = space
        return space

    def delete_space(self, space_id: str) -> bool:
        """Delete a space. Returns True if it existed."""
        if space_id in self._spaces:
            del self._spaces[space_id]
            return True
        return False

    def get_space(self, space_id: str) -> CollaborationSpace | None:
        """Retrieve a space by ID."""
        return self._spaces.get(space_id)

    def list_spaces(self, principal_id: str | None = None) -> list[CollaborationSpace]:
        """List all spaces, optionally filtered by membership."""
        spaces = list(self._spaces.values())
        if principal_id is not None:
            spaces = [s for s in spaces if principal_id in s.members]
        return spaces

    # ------------------------------------------------------------------ #
    # Membership
    # ------------------------------------------------------------------ #
    def add_member(
        self,
        space_id: str,
        principal_id: str,
        role: MemberRole,
        *,
        principal_type: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> SpaceMember:
        """Add a member to a space."""
        space = self._get_space(space_id)
        member = SpaceMember(
            principal_id=principal_id,
            principal_type=principal_type,
            role=role,
            metadata=metadata or {},
        )
        space.members[principal_id] = member
        return member

    def remove_member(self, space_id: str, principal_id: str) -> bool:
        """Remove a member. Cannot remove the owner."""
        space = self._get_space(space_id)
        if principal_id == space.owner_id:
            return False
        if principal_id in space.members:
            del space.members[principal_id]
            return True
        return False

    def get_member_role(self, space_id: str, principal_id: str) -> MemberRole | None:
        """Return the role of a principal in a space."""
        space = self._get_space(space_id)
        member = space.members.get(principal_id)
        return member.role if member else None

    def update_member_role(
        self, space_id: str, principal_id: str, new_role: MemberRole
    ) -> bool:
        """Change a member's role."""
        space = self._get_space(space_id)
        member = space.members.get(principal_id)
        if member is None:
            return False
        member.role = new_role
        return True

    def list_members(self, space_id: str) -> list[SpaceMember]:
        """Return all members of a space."""
        space = self._get_space(space_id)
        return list(space.members.values())

    # ------------------------------------------------------------------ #
    # Memory association
    # ------------------------------------------------------------------ #
    def add_memory(self, space_id: str, memory_id: str) -> bool:
        """Associate a memory with a space."""
        space = self._get_space(space_id)
        space.memory_ids.add(memory_id)
        return True

    def remove_memory(self, space_id: str, memory_id: str) -> bool:
        """Disassociate a memory from a space."""
        space = self._get_space(space_id)
        if memory_id in space.memory_ids:
            space.memory_ids.discard(memory_id)
            return True
        return False

    def get_memories(self, space_id: str) -> list[Memory]:
        """Return all Memory objects attached to a space."""
        space = self._get_space(space_id)
        return [self._memories[mid] for mid in space.memory_ids if mid in self._memories]

    def can_edit(self, space_id: str, principal_id: str) -> bool:
        """True if principal has editor or owner role."""
        role = self.get_member_role(space_id, principal_id)
        return role in (MemberRole.OWNER, MemberRole.EDITOR)

    def can_view(self, space_id: str, principal_id: str) -> bool:
        """True if principal is any member."""
        role = self.get_member_role(space_id, principal_id)
        return role is not None

    def get_space_stats(self, space_id: str) -> dict[str, Any]:
        """Return statistics for a space."""
        space = self._get_space(space_id)
        return {
            "space_id": space.space_id,
            "name": space.name,
            "member_count": len(space.members),
            "memory_count": len(space.memory_ids),
            "owner_id": space.owner_id,
        }
