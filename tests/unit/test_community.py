"""Unit tests for NeuralMem V1.3 community features — all mock-based.

Covers:
  • MemorySharing (share, revoke, list, access control)
  • SpaceManager / CollaborationSpace (CRUD, membership, memory association)
  • FeedbackLoop (submit, aggregate, flag, summary)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.community import (
    MemorySharing,
    ShareRecord,
    SpaceManager,
    CollaborationSpace,
    MemberRole,
    SpaceMember,
    FeedbackLoop,
    FeedbackEntry,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_memory(memory_id: str = "mem-001", content: str = "test content") -> MagicMock:
    """Return a mock Memory object with the given id."""
    mem = MagicMock()
    mem.id = memory_id
    mem.content = content
    return mem


# =============================================================================
# MemorySharing
# =============================================================================

class TestMemorySharingShare:
    def test_share_memory_returns_record(self):
        sharing = MemorySharing()
        mem = make_memory("m1")
        rec = sharing.share_memory(mem, owner_id="u1", recipient_id="u2")
        assert isinstance(rec, ShareRecord)
        assert rec.memory_id == "m1"
        assert rec.owner_id == "u1"
        assert rec.recipient_id == "u2"
        assert rec.permission == "read"

    def test_share_memory_auto_registers(self):
        sharing = MemorySharing()
        mem = make_memory("m2")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2")
        assert "m2" in sharing._memories

    def test_share_memory_with_options(self):
        sharing = MemorySharing()
        mem = make_memory("m3")
        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        rec = sharing.share_memory(
            mem,
            owner_id="u1",
            recipient_id="a1",
            recipient_type="agent",
            permission="write",
            expires_at=expires,
            metadata={"note": "urgent"},
        )
        assert rec.recipient_type == "agent"
        assert rec.permission == "write"
        assert rec.expires_at == expires
        assert rec.metadata["note"] == "urgent"


class TestMemorySharingRevoke:
    def test_revoke_share_removes_matching(self):
        sharing = MemorySharing()
        mem = make_memory("m4")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2")
        assert sharing.revoke_share("m4", "u2") is True
        assert sharing.list_shared_to("u2") == []

    def test_revoke_share_no_match(self):
        sharing = MemorySharing()
        assert sharing.revoke_share("m4", "u2") is False

    def test_revoke_all_for_memory(self):
        sharing = MemorySharing()
        mem = make_memory("m5")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u3")
        assert sharing.revoke_all_for_memory("m5") == 2
        assert sharing.list_shared_to("u2") == []
        assert sharing.list_shared_to("u3") == []


class TestMemorySharingList:
    def test_list_shared_to(self):
        sharing = MemorySharing()
        mem = make_memory("m6")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2")
        results = sharing.list_shared_to("u2")
        assert len(results) == 1
        assert results[0][0].id == "m6"
        assert results[0][1].recipient_id == "u2"

    def test_list_shared_from(self):
        sharing = MemorySharing()
        mem = make_memory("m7")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2")
        results = sharing.list_shared_from("u1")
        assert len(results) == 1
        assert results[0][0].id == "m7"

    def test_list_shared_to_filters_expired(self):
        sharing = MemorySharing()
        mem = make_memory("m8")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2", expires_at=past)
        assert sharing.list_shared_to("u2") == []
        assert len(sharing.list_shared_to("u2", include_expired=True)) == 1

    def test_list_shared_to_filters_type(self):
        sharing = MemorySharing()
        mem = make_memory("m9")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2", recipient_type="user")
        sharing.share_memory(mem, owner_id="u1", recipient_id="a1", recipient_type="agent")
        user_results = sharing.list_shared_to("u2", recipient_type="user")
        agent_results = sharing.list_shared_to("a1", recipient_type="agent")
        assert len(user_results) == 1
        assert len(agent_results) == 1


class TestMemorySharingAccess:
    def test_can_access_read(self):
        sharing = MemorySharing()
        mem = make_memory("m10")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2", permission="read")
        assert sharing.can_access("m10", "u2", required_permission="read") is True
        assert sharing.can_access("m10", "u2", required_permission="write") is False

    def test_can_access_write_implies_read(self):
        sharing = MemorySharing()
        mem = make_memory("m11")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2", permission="write")
        assert sharing.can_access("m11", "u2", required_permission="read") is True
        assert sharing.can_access("m11", "u2", required_permission="write") is True

    def test_can_access_expired_denied(self):
        sharing = MemorySharing()
        mem = make_memory("m12")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2", expires_at=past)
        assert sharing.can_access("m12", "u2") is False


class TestMemorySharingStats:
    def test_get_share_stats(self):
        sharing = MemorySharing()
        mem = make_memory("m13")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2", permission="read")
        sharing.share_memory(mem, owner_id="u1", recipient_id="u3", permission="write")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        sharing.share_memory(mem, owner_id="u1", recipient_id="u4", expires_at=past)
        stats = sharing.get_share_stats("m13")
        assert stats["total_shares"] == 3
        assert stats["active_shares"] == 2
        assert stats["expired_shares"] == 1
        assert stats["read_grants"] == 1
        assert stats["write_grants"] == 1


# =============================================================================
# SpaceManager / CollaborationSpace
# =============================================================================

class TestSpaceManagerCRUD:
    def test_create_space(self):
        mgr = SpaceManager()
        space = mgr.create_space("team-alpha", owner_id="u1")
        assert isinstance(space, CollaborationSpace)
        assert space.name == "team-alpha"
        assert space.owner_id == "u1"
        assert "u1" in space.members

    def test_get_space(self):
        mgr = SpaceManager()
        space = mgr.create_space("proj-x", owner_id="u1")
        fetched = mgr.get_space(space.space_id)
        assert fetched is not None
        assert fetched.space_id == space.space_id

    def test_delete_space(self):
        mgr = SpaceManager()
        space = mgr.create_space("tmp", owner_id="u1")
        assert mgr.delete_space(space.space_id) is True
        assert mgr.get_space(space.space_id) is None
        assert mgr.delete_space("bogus") is False

    def test_list_spaces(self):
        mgr = SpaceManager()
        s1 = mgr.create_space("s1", owner_id="u1")
        mgr.create_space("s2", owner_id="u2")
        assert len(mgr.list_spaces()) == 2
        assert len(mgr.list_spaces(principal_id="u1")) == 1
        assert mgr.list_spaces(principal_id="u1")[0].space_id == s1.space_id


class TestSpaceManagerMembership:
    def test_add_member(self):
        mgr = SpaceManager()
        space = mgr.create_space("s3", owner_id="u1")
        member = mgr.add_member(space.space_id, "u2", MemberRole.EDITOR)
        assert isinstance(member, SpaceMember)
        assert member.role == MemberRole.EDITOR
        assert mgr.get_member_role(space.space_id, "u2") == MemberRole.EDITOR

    def test_remove_member(self):
        mgr = SpaceManager()
        space = mgr.create_space("s4", owner_id="u1")
        mgr.add_member(space.space_id, "u2", MemberRole.VIEWER)
        assert mgr.remove_member(space.space_id, "u2") is True
        assert mgr.get_member_role(space.space_id, "u2") is None

    def test_cannot_remove_owner(self):
        mgr = SpaceManager()
        space = mgr.create_space("s5", owner_id="u1")
        assert mgr.remove_member(space.space_id, "u1") is False
        assert mgr.get_member_role(space.space_id, "u1") == MemberRole.OWNER

    def test_update_member_role(self):
        mgr = SpaceManager()
        space = mgr.create_space("s6", owner_id="u1")
        mgr.add_member(space.space_id, "u2", MemberRole.VIEWER)
        assert mgr.update_member_role(space.space_id, "u2", MemberRole.EDITOR) is True
        assert mgr.get_member_role(space.space_id, "u2") == MemberRole.EDITOR

    def test_list_members(self):
        mgr = SpaceManager()
        space = mgr.create_space("s7", owner_id="u1")
        mgr.add_member(space.space_id, "u2", MemberRole.EDITOR)
        members = mgr.list_members(space.space_id)
        assert len(members) == 2
        ids = {m.principal_id for m in members}
        assert ids == {"u1", "u2"}


class TestSpaceManagerPermissions:
    def test_can_edit(self):
        mgr = SpaceManager()
        space = mgr.create_space("s8", owner_id="u1")
        mgr.add_member(space.space_id, "u2", MemberRole.EDITOR)
        mgr.add_member(space.space_id, "u3", MemberRole.VIEWER)
        assert mgr.can_edit(space.space_id, "u1") is True
        assert mgr.can_edit(space.space_id, "u2") is True
        assert mgr.can_edit(space.space_id, "u3") is False
        assert mgr.can_edit(space.space_id, "u4") is False

    def test_can_view(self):
        mgr = SpaceManager()
        space = mgr.create_space("s9", owner_id="u1")
        mgr.add_member(space.space_id, "u2", MemberRole.VIEWER)
        assert mgr.can_view(space.space_id, "u1") is True
        assert mgr.can_view(space.space_id, "u2") is True
        assert mgr.can_view(space.space_id, "u3") is False


class TestSpaceManagerMemories:
    def test_add_memory(self):
        mgr = SpaceManager()
        space = mgr.create_space("s10", owner_id="u1")
        assert mgr.add_memory(space.space_id, "mem-001") is True
        assert "mem-001" in space.memory_ids

    def test_remove_memory(self):
        mgr = SpaceManager()
        space = mgr.create_space("s11", owner_id="u1")
        mgr.add_memory(space.space_id, "mem-002")
        assert mgr.remove_memory(space.space_id, "mem-002") is True
        assert "mem-002" not in space.memory_ids
        assert mgr.remove_memory(space.space_id, "mem-002") is False

    def test_get_memories(self):
        mgr = SpaceManager()
        space = mgr.create_space("s12", owner_id="u1")
        mem = make_memory("mem-003", "content-a")
        mgr.register_memory(mem)
        mgr.add_memory(space.space_id, "mem-003")
        memories = mgr.get_memories(space.space_id)
        assert len(memories) == 1
        assert memories[0].id == "mem-003"

    def test_get_memories_missing_ignored(self):
        mgr = SpaceManager()
        space = mgr.create_space("s13", owner_id="u1")
        mgr.add_memory(space.space_id, "mem-missing")
        assert mgr.get_memories(space.space_id) == []


class TestSpaceManagerStats:
    def test_get_space_stats(self):
        mgr = SpaceManager()
        space = mgr.create_space("s14", owner_id="u1")
        mgr.add_member(space.space_id, "u2", MemberRole.EDITOR)
        mgr.add_memory(space.space_id, "mem-004")
        stats = mgr.get_space_stats(space.space_id)
        assert stats["name"] == "s14"
        assert stats["member_count"] == 2
        assert stats["memory_count"] == 1
        assert stats["owner_id"] == "u1"


class TestSpaceManagerErrors:
    def test_get_space_missing(self):
        mgr = SpaceManager()
        assert mgr.get_space("nonexistent") is None

    def test_operations_on_missing_space(self):
        mgr = SpaceManager()
        with pytest.raises(KeyError):
            mgr.add_member("bad-id", "u2", MemberRole.VIEWER)
        with pytest.raises(KeyError):
            mgr.get_memories("bad-id")


# =============================================================================
# FeedbackLoop
# =============================================================================

class TestFeedbackSubmit:
    def test_submit_feedback(self):
        fb = FeedbackLoop()
        entry = fb.submit_feedback("m1", "u1", helpful=True, comment="good")
        assert isinstance(entry, FeedbackEntry)
        assert entry.memory_id == "m1"
        assert entry.user_id == "u1"
        assert entry.helpful is True
        assert entry.comment == "good"

    def test_submit_with_rating(self):
        fb = FeedbackLoop()
        entry = fb.submit_feedback("m2", "u1", rating=5, helpful=True)
        assert entry.rating == 5

    def test_submit_with_metadata(self):
        fb = FeedbackLoop()
        entry = fb.submit_feedback("m3", "u1", metadata={"source": "web"})
        assert entry.metadata["source"] == "web"


class TestFeedbackAggregate:
    def test_aggregate_no_feedback(self):
        fb = FeedbackLoop()
        assert fb.aggregate_score("m1") == 0.5

    def test_aggregate_all_positive(self):
        fb = FeedbackLoop()
        fb.submit_feedback("m1", "u1", helpful=True)
        fb.submit_feedback("m1", "u2", helpful=True)
        assert fb.aggregate_score("m1") == 1.0

    def test_aggregate_all_negative(self):
        fb = FeedbackLoop()
        fb.submit_feedback("m1", "u1", helpful=False)
        fb.submit_feedback("m1", "u2", helpful=False)
        assert fb.aggregate_score("m1") == 0.0

    def test_aggregate_mixed(self):
        fb = FeedbackLoop()
        fb.submit_feedback("m1", "u1", helpful=True)
        fb.submit_feedback("m1", "u2", helpful=False)
        # helpful_norm = (0/2 + 1)/2 = 0.5, no ratings
        assert fb.aggregate_score("m1") == 0.5

    def test_aggregate_with_ratings(self):
        fb = FeedbackLoop()
        fb.submit_feedback("m1", "u1", helpful=True, rating=5)
        fb.submit_feedback("m1", "u2", helpful=True, rating=3)
        # helpful_norm = 1.0, rating_norm = (1.0 + 0.5)/2 = 0.75
        # blend = 0.6*1.0 + 0.4*0.75 = 0.9
        assert fb.aggregate_score("m1") == 0.9


class TestFeedbackFlagging:
    def test_is_flagged(self):
        fb = FeedbackLoop(flag_threshold=0.3)
        fb.submit_feedback("m1", "u1", helpful=False)
        fb.submit_feedback("m1", "u2", helpful=False)
        assert fb.is_flagged("m1") is True

    def test_not_flagged(self):
        fb = FeedbackLoop(flag_threshold=0.3)
        fb.submit_feedback("m1", "u1", helpful=True)
        assert fb.is_flagged("m1") is False

    def test_list_flagged(self):
        fb = FeedbackLoop(flag_threshold=0.3)
        fb.submit_feedback("m1", "u1", helpful=False)
        fb.submit_feedback("m1", "u2", helpful=False)
        fb.submit_feedback("m2", "u1", helpful=True)
        flagged = fb.list_flagged()
        assert "m1" in flagged
        assert "m2" not in flagged

    def test_list_flagged_custom_threshold(self):
        fb = FeedbackLoop(flag_threshold=0.3)
        fb.submit_feedback("m1", "u1", helpful=True, rating=3)
        fb.submit_feedback("m1", "u2", helpful=False, rating=1)
        # score should be around 0.55, not flagged at 0.3 but flagged at 0.6
        assert "m1" not in fb.list_flagged()
        assert "m1" in fb.list_flagged(threshold=0.6)


class TestFeedbackSummary:
    def test_get_summary(self):
        fb = FeedbackLoop()
        fb.submit_feedback("m1", "u1", helpful=True, rating=5, comment="nice")
        fb.submit_feedback("m1", "u2", helpful=False, rating=1)
        summary = fb.get_summary("m1")
        assert summary["memory_id"] == "m1"
        assert summary["total_feedback"] == 2
        assert summary["thumbs_up"] == 1
        assert summary["thumbs_down"] == 1
        assert summary["average_rating"] == 3.0
        assert summary["flagged"] is False  # score ~0.55

    def test_get_summary_no_ratings(self):
        fb = FeedbackLoop()
        fb.submit_feedback("m1", "u1", helpful=True)
        summary = fb.get_summary("m1")
        assert summary["average_rating"] is None


class TestFeedbackQueries:
    def test_get_feedback(self):
        fb = FeedbackLoop()
        fb.submit_feedback("m1", "u1", helpful=True)
        fb.submit_feedback("m1", "u2", helpful=False)
        entries = fb.get_feedback("m1")
        assert len(entries) == 2

    def test_get_user_feedback(self):
        fb = FeedbackLoop()
        fb.submit_feedback("m1", "u1", helpful=True)
        fb.submit_feedback("m2", "u1", helpful=False)
        fb.submit_feedback("m3", "u2", helpful=True)
        entries = fb.get_user_feedback("u1")
        assert len(entries) == 2
        assert all(e.user_id == "u1" for e in entries)


class TestFeedbackGlobalStats:
    def test_get_global_stats(self):
        fb = FeedbackLoop(flag_threshold=0.3)
        fb.submit_feedback("m1", "u1", helpful=False)
        fb.submit_feedback("m1", "u2", helpful=False)
        fb.submit_feedback("m2", "u1", helpful=True)
        stats = fb.get_global_stats()
        assert stats["total_feedback_entries"] == 3
        assert stats["memories_with_feedback"] == 2
        assert stats["flagged_memory_count"] == 1
        assert stats["flagged_memory_ids"] == ["m1"]
