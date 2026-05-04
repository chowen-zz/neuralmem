"""OrgManager — CRUD, member management, and settings for organizations.

V2.0: OrgManager
  • Create, read, update, delete organizations
  • Add, remove, update members with role-based access
  • Organization settings management
  • In-memory + optional SQLite persistence
  • Similar to Supermemory's org support with Better Auth patterns
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Any

from neuralmem.organization.org import Organization, OrgSettings, OrgStatus
from neuralmem.organization.member import (
    MemberRole,
    MembershipStatus,
    OrganizationMember,
)


class OrgNotFoundError(KeyError):
    """Raised when an organization is not found."""
    pass


class MemberNotFoundError(KeyError):
    """Raised when a member is not found in an organization."""
    pass


class DuplicateMemberError(ValueError):
    """Raised when adding a member that already exists."""
    pass


class RoleAssignmentError(PermissionError):
    """Raised when a role assignment violates organization rules."""
    pass


class OrgManager:
    """Manager for organizations and their members.

    Supports in-memory storage or optional SQLite persistence.

    Usage:
        mgr = OrgManager()
        org = mgr.create_org("Acme Corp", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR, invited_by="u1")
        mgr.update_org_settings(org.org_id, max_members=50)
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._lock = threading.RLock()
        self._orgs: dict[str, Organization] = {}
        self._members: dict[str, dict[str, OrganizationMember]] = {}
        self._use_db = db_path is not None
        if self._use_db:
            assert db_path is not None
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._init_db()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        """Initialize SQLite tables."""
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS organizations (
                org_id TEXT PRIMARY KEY,
                name TEXT,
                slug TEXT,
                description TEXT,
                owner_id TEXT,
                status TEXT,
                settings TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS org_members (
                user_id TEXT,
                org_id TEXT,
                role TEXT,
                status TEXT,
                invited_by TEXT,
                joined_at TEXT,
                updated_at TEXT,
                metadata TEXT,
                PRIMARY KEY (user_id, org_id)
            )"""
        )
        self._conn.commit()

    def _get_org(self, org_id: str) -> Organization:
        """Retrieve an organization or raise OrgNotFoundError."""
        org = self._orgs.get(org_id)
        if org is None:
            raise OrgNotFoundError(f"Organization '{org_id}' not found")
        return org

    def _get_members(self, org_id: str) -> dict[str, OrganizationMember]:
        """Retrieve member dict for an org, initializing if needed."""
        if org_id not in self._members:
            self._members[org_id] = {}
        return self._members[org_id]

    def _persist_org(self, org: Organization) -> None:
        """Persist an organization to SQLite."""
        if not self._use_db:
            return
        import json
        self._conn.execute(
            """INSERT OR REPLACE INTO organizations
                (org_id, name, slug, description, owner_id, status,
                 settings, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                org.org_id,
                org.name,
                org.slug,
                org.description,
                org.owner_id,
                org.status.value,
                json.dumps(org.settings.to_dict()),
                org.created_at.isoformat(),
                org.updated_at.isoformat(),
                json.dumps(org.metadata),
            ),
        )
        self._conn.commit()

    def _persist_member(self, member: OrganizationMember) -> None:
        """Persist a member to SQLite."""
        if not self._use_db:
            return
        import json
        self._conn.execute(
            """INSERT OR REPLACE INTO org_members
                (user_id, org_id, role, status, invited_by,
                 joined_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                member.user_id,
                member.org_id,
                member.role.value,
                member.status.value,
                member.invited_by,
                member.joined_at.isoformat(),
                member.updated_at.isoformat(),
                json.dumps(member.metadata),
            ),
        )
        self._conn.commit()

    def _delete_member_db(self, org_id: str, user_id: str) -> None:
        """Delete a member from SQLite."""
        if not self._use_db:
            return
        self._conn.execute(
            "DELETE FROM org_members WHERE org_id = ? AND user_id = ?",
            (org_id, user_id),
        )
        self._conn.commit()

    def _load_from_db(self) -> None:
        """Load all organizations and members from SQLite."""
        if not self._use_db:
            return
        import json
        from datetime import datetime

        # Load organizations
        rows = self._conn.execute(
            "SELECT org_id, name, slug, description, owner_id, status, "
            "settings, created_at, updated_at, metadata FROM organizations"
        ).fetchall()
        for row in rows:
            settings = OrgSettings.from_dict(json.loads(row[6]))
            org = Organization(
                org_id=row[0],
                name=row[1],
                slug=row[2],
                description=row[3],
                owner_id=row[4],
                status=OrgStatus(row[5]),
                settings=settings,
                created_at=datetime.fromisoformat(row[7]),
                updated_at=datetime.fromisoformat(row[8]),
                metadata=json.loads(row[9]),
            )
            self._orgs[org.org_id] = org

        # Load members
        rows = self._conn.execute(
            "SELECT user_id, org_id, role, status, invited_by, "
            "joined_at, updated_at, metadata FROM org_members"
        ).fetchall()
        for row in rows:
            member = OrganizationMember(
                user_id=row[0],
                org_id=row[1],
                role=MemberRole(row[2]),
                status=MembershipStatus(row[3]),
                invited_by=row[4],
                joined_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                metadata=json.loads(row[7]),
            )
            self._members.setdefault(member.org_id, {})[member.user_id] = member

    # ------------------------------------------------------------------ #
    # Organization CRUD
    # ------------------------------------------------------------------ #
    def create_org(
        self,
        name: str,
        *,
        owner_id: str,
        description: str = "",
        settings: OrgSettings | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Organization:
        """Create a new organization.

        Args:
            name: Organization name.
            owner_id: User ID of the owner.
            description: Optional description.
            settings: Optional custom settings.
            metadata: Optional extra metadata.

        Returns:
            The created Organization.
        """
        with self._lock:
            org = Organization(
                name=name,
                description=description,
                owner_id=owner_id,
                settings=settings or OrgSettings(),
                metadata=metadata or {},
            )
            self._orgs[org.org_id] = org
            # Owner is automatically a member
            owner_member = OrganizationMember(
                user_id=owner_id,
                org_id=org.org_id,
                role=MemberRole.OWNER,
                status=MembershipStatus.ACTIVE,
            )
            self._get_members(org.org_id)[owner_id] = owner_member
            self._persist_org(org)
            self._persist_member(owner_member)
            return org

    def get_org(self, org_id: str) -> Organization | None:
        """Retrieve an organization by ID."""
        with self._lock:
            return self._orgs.get(org_id)

    def list_orgs(
        self,
        *,
        user_id: str | None = None,
        status: OrgStatus | None = None,
    ) -> list[Organization]:
        """List organizations, optionally filtered by member or status.

        Args:
            user_id: If provided, only list orgs this user is a member of.
            status: If provided, filter by organization status.

        Returns:
            List of matching organizations.
        """
        with self._lock:
            results = list(self._orgs.values())
            if status is not None:
                results = [o for o in results if o.status == status]
            if user_id is not None:
                user_orgs = []
                for org in results:
                    members = self._members.get(org.org_id, {})
                    if user_id in members:
                        user_orgs.append(org)
                results = user_orgs
            return results

    def update_org(
        self,
        org_id: str,
        **kwargs: Any,
    ) -> Organization:
        """Update organization fields.

        Args:
            org_id: Organization ID.
            **kwargs: Fields to update (name, description, status, metadata, owner_id).

        Returns:
            The updated Organization.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
        """
        with self._lock:
            org = self._get_org(org_id)
            org.update(**kwargs)
            self._persist_org(org)
            return org

    def update_org_settings(
        self,
        org_id: str,
        **kwargs: Any,
    ) -> Organization:
        """Update organization settings.

        Args:
            org_id: Organization ID.
            **kwargs: Settings fields to update.

        Returns:
            The updated Organization.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
        """
        with self._lock:
            org = self._get_org(org_id)
            org.update_settings(**kwargs)
            self._persist_org(org)
            return org

    def delete_org(self, org_id: str) -> bool:
        """Delete an organization and all its memberships.

        Returns:
            True if the organization existed and was deleted.
        """
        with self._lock:
            if org_id not in self._orgs:
                return False
            del self._orgs[org_id]
            self._members.pop(org_id, None)
            if self._use_db:
                self._conn.execute(
                    "DELETE FROM organizations WHERE org_id = ?", (org_id,)
                )
                self._conn.execute(
                    "DELETE FROM org_members WHERE org_id = ?", (org_id,)
                )
                self._conn.commit()
            return True

    # ------------------------------------------------------------------ #
    # Member management
    # ------------------------------------------------------------------ #
    def add_member(
        self,
        org_id: str,
        user_id: str,
        role: MemberRole = MemberRole.VIEWER,
        *,
        invited_by: str = "",
        status: MembershipStatus = MembershipStatus.ACTIVE,
        metadata: dict[str, Any] | None = None,
    ) -> OrganizationMember:
        """Add a member to an organization.

        Args:
            org_id: Organization ID.
            user_id: User ID to add.
            role: Role to assign.
            invited_by: User ID of the inviter.
            status: Initial membership status.
            metadata: Optional extra metadata.

        Returns:
            The created OrganizationMember.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
            DuplicateMemberError: If the user is already a member.
            RoleAssignmentError: If assigning owner role improperly.
        """
        with self._lock:
            org = self._get_org(org_id)
            members = self._get_members(org_id)
            if user_id in members:
                raise DuplicateMemberError(
                    f"User '{user_id}' is already a member of org '{org_id}'"
                )
            if role == MemberRole.OWNER and org.owner_id != user_id:
                # Only existing owner can assign owner, and only one owner allowed
                existing_owner = [m for m in members.values() if m.role == MemberRole.OWNER]
                if existing_owner:
                    raise RoleAssignmentError(
                        "Organization already has an owner. Transfer ownership first."
                    )
            member = OrganizationMember(
                user_id=user_id,
                org_id=org_id,
                role=role,
                status=status,
                invited_by=invited_by,
                metadata=metadata or {},
            )
            members[user_id] = member
            self._persist_member(member)
            return member

    def remove_member(self, org_id: str, user_id: str) -> bool:
        """Remove a member from an organization.

        Cannot remove the owner. Transfer ownership first.

        Returns:
            True if the member was removed.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
        """
        with self._lock:
            org = self._get_org(org_id)
            members = self._get_members(org_id)
            if user_id == org.owner_id:
                return False
            if user_id not in members:
                return False
            del members[user_id]
            self._delete_member_db(org_id, user_id)
            return True

    def get_member(self, org_id: str, user_id: str) -> OrganizationMember | None:
        """Retrieve a member by org and user ID."""
        with self._lock:
            members = self._get_members(org_id)
            return members.get(user_id)

    def update_member_role(
        self,
        org_id: str,
        user_id: str,
        new_role: MemberRole,
        *,
        changed_by: str = "",
    ) -> OrganizationMember:
        """Update a member's role.

        Args:
            org_id: Organization ID.
            user_id: User ID.
            new_role: New role to assign.
            changed_by: User ID making the change (for audit).

        Returns:
            The updated OrganizationMember.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
            MemberNotFoundError: If the user is not a member.
            RoleAssignmentError: If changing owner role improperly.
        """
        with self._lock:
            org = self._get_org(org_id)
            members = self._get_members(org_id)
            member = members.get(user_id)
            if member is None:
                raise MemberNotFoundError(
                    f"User '{user_id}' is not a member of org '{org_id}'"
                )
            # Cannot demote owner without transferring ownership
            if member.role == MemberRole.OWNER and new_role != MemberRole.OWNER:
                if user_id == org.owner_id:
                    raise RoleAssignmentError(
                        "Cannot demote the current owner. Transfer ownership first."
                    )
            # Cannot promote to owner if org already has a different owner
            if new_role == MemberRole.OWNER:
                if org.owner_id and org.owner_id != user_id:
                    raise RoleAssignmentError(
                        f"Organization already owned by '{org.owner_id}'. "
                        "Transfer ownership first."
                    )
                org.owner_id = user_id
                self._persist_org(org)
            member.update_role(new_role)
            member.metadata["role_changed_by"] = changed_by
            self._persist_member(member)
            return member

    def update_member_status(
        self,
        org_id: str,
        user_id: str,
        new_status: MembershipStatus,
    ) -> OrganizationMember:
        """Update a member's status.

        Args:
            org_id: Organization ID.
            user_id: User ID.
            new_status: New status to set.

        Returns:
            The updated OrganizationMember.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
            MemberNotFoundError: If the user is not a member.
        """
        with self._lock:
            self._get_org(org_id)
            members = self._get_members(org_id)
            member = members.get(user_id)
            if member is None:
                raise MemberNotFoundError(
                    f"User '{user_id}' is not a member of org '{org_id}'"
                )
            member.update_status(new_status)
            self._persist_member(member)
            return member

    def list_members(
        self,
        org_id: str,
        *,
        status: MembershipStatus | None = None,
        role: MemberRole | None = None,
    ) -> list[OrganizationMember]:
        """List members of an organization.

        Args:
            org_id: Organization ID.
            status: Filter by membership status.
            role: Filter by role.

        Returns:
            List of matching members.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
        """
        with self._lock:
            self._get_org(org_id)
            members = self._get_members(org_id).values()
            if status is not None:
                members = [m for m in members if m.status == status]
            if role is not None:
                members = [m for m in members if m.role == role]
            return list(members)

    def list_user_orgs(self, user_id: str) -> list[Organization]:
        """List all organizations a user is a member of."""
        with self._lock:
            return self.list_orgs(user_id=user_id)

    # ------------------------------------------------------------------ #
    # Permission checking
    # ------------------------------------------------------------------ #
    def check_permission(
        self,
        org_id: str,
        user_id: str,
        required_role: MemberRole,
    ) -> bool:
        """Check if a user has at least the required role in an org.

        Args:
            org_id: Organization ID.
            user_id: User ID.
            required_role: Minimum role required.

        Returns:
            True if the user has sufficient privileges.
        """
        with self._lock:
            member = self.get_member(org_id, user_id)
            if member is None:
                return False
            return member.can(required_role)

    def is_owner(self, org_id: str, user_id: str) -> bool:
        """Check if a user is the owner of an organization."""
        return self.check_permission(org_id, user_id, MemberRole.OWNER)

    def is_admin(self, org_id: str, user_id: str) -> bool:
        """Check if a user is an admin (or owner) of an organization."""
        return self.check_permission(org_id, user_id, MemberRole.ADMIN)

    def can_edit(self, org_id: str, user_id: str) -> bool:
        """Check if a user can edit (editor, admin, or owner)."""
        return self.check_permission(org_id, user_id, MemberRole.EDITOR)

    def can_view(self, org_id: str, user_id: str) -> bool:
        """Check if a user can view (any active member)."""
        return self.check_permission(org_id, user_id, MemberRole.VIEWER)

    # ------------------------------------------------------------------ #
    # Ownership transfer
    # ------------------------------------------------------------------ #
    def transfer_ownership(
        self,
        org_id: str,
        new_owner_id: str,
        *,
        current_owner_id: str = "",
    ) -> Organization:
        """Transfer ownership of an organization to another member.

        Args:
            org_id: Organization ID.
            new_owner_id: User ID of the new owner (must be a member).
            current_owner_id: Optional current owner for validation.

        Returns:
            The updated Organization.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
            MemberNotFoundError: If the new owner is not a member.
            RoleAssignmentError: If validation fails.
        """
        with self._lock:
            org = self._get_org(org_id)
            members = self._get_members(org_id)
            # Validate current owner if provided
            if current_owner_id and org.owner_id != current_owner_id:
                raise RoleAssignmentError(
                    f"Current owner mismatch: expected '{current_owner_id}', "
                    f"got '{org.owner_id}'"
                )
            # Validate new owner is a member
            new_owner = members.get(new_owner_id)
            if new_owner is None:
                raise MemberNotFoundError(
                    f"User '{new_owner_id}' is not a member of org '{org_id}'"
                )
            # Demote current owner to admin
            current_owner = members.get(org.owner_id)
            if current_owner is not None:
                current_owner.update_role(MemberRole.ADMIN)
                self._persist_member(current_owner)
            # Promote new owner
            new_owner.update_role(MemberRole.OWNER)
            self._persist_member(new_owner)
            # Update org owner
            org.owner_id = new_owner_id
            self._persist_org(org)
            return org

    # ------------------------------------------------------------------ #
    # Statistics & queries
    # ------------------------------------------------------------------ #
    def get_org_stats(self, org_id: str) -> dict[str, Any]:
        """Return statistics for an organization.

        Raises:
            OrgNotFoundError: If the organization doesn't exist.
        """
        with self._lock:
            org = self._get_org(org_id)
            members = self._get_members(org_id)
            active_members = [m for m in members.values() if m.is_active()]
            return {
                "org_id": org.org_id,
                "name": org.name,
                "slug": org.slug,
                "status": org.status.value,
                "owner_id": org.owner_id,
                "total_members": len(members),
                "active_members": len(active_members),
                "pending_members": len(
                    [m for m in members.values() if m.status == MembershipStatus.PENDING]
                ),
                "suspended_members": len(
                    [m for m in members.values() if m.status == MembershipStatus.SUSPENDED]
                ),
                "settings": org.settings.to_dict(),
                "created_at": org.created_at.isoformat(),
                "updated_at": org.updated_at.isoformat(),
            }

    def get_member_count(self, org_id: str) -> int:
        """Return the total number of members in an organization."""
        with self._lock:
            self._get_org(org_id)
            return len(self._get_members(org_id))

    def get_active_member_count(self, org_id: str) -> int:
        """Return the number of active members."""
        with self._lock:
            self._get_org(org_id)
            members = self._get_members(org_id)
            return len([m for m in members.values() if m.is_active()])
