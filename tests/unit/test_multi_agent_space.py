"""Tests for NeuralMem V1.8 multi-agent memory space."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.multi_agent.space import AgentMemorySpace, AgentMemoryPool, Permission


class TestAgentRegistration:
    def test_register_agent(self):
        space = AgentMemorySpace()
        agent = space.register_agent("a1", "Agent1")
        assert agent.agent_id == "a1"
        assert agent.name == "Agent1"

    def test_create_pool(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        pool = space.create_pool("a1", "pool1")
        assert pool.pool_id == "pool1"
        assert "a1" in pool.agents

    def test_create_pool_unregistered_agent(self):
        space = AgentMemorySpace()
        with pytest.raises(ValueError):
            space.create_pool("a1", "pool1")


class TestPoolMembership:
    def test_add_to_pool(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        space.register_agent("a2", "Agent2")
        pool = space.create_pool("a1")
        space.add_to_pool(pool.pool_id, "a2", Permission.READ)
        assert "a2" in pool.agents

    def test_get_agent_pools(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        pool = space.create_pool("a1")
        pools = space.get_agent_pools("a1")
        assert pool.pool_id in pools


class TestMemoryStorage:
    def test_store_private(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        space.store_private("a1", {"content": "secret"})
        memories = space.query_private("a1")
        assert len(memories) == 1
        assert memories[0]["content"] == "secret"
        assert memories[0]["_private"] is True

    def test_store_shared(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        pool = space.create_pool("a1")
        space.store_shared(pool.pool_id, {"content": "shared"}, "a1")
        memories = space.query_shared(pool.pool_id, "a1")
        assert len(memories) == 1
        assert memories[0]["content"] == "shared"

    def test_store_shared_no_permission(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        space.register_agent("a2", "Agent2")
        pool = space.create_pool("a1")
        space.add_to_pool(pool.pool_id, "a2", Permission.READ)
        with pytest.raises(PermissionError):
            space.store_shared(pool.pool_id, {"content": "x"}, "a2")

    def test_query_shared_no_read_permission(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        space.register_agent("a2", "Agent2")
        pool = space.create_pool("a1")
        with pytest.raises(PermissionError):
            space.query_shared(pool.pool_id, "a2")


class TestAgentMemoryPool:
    def test_pool_wrapper(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        pool = space.create_pool("a1")
        wrapper = AgentMemoryPool(space, pool.pool_id)
        wrapper.share({"content": "test"}, "a1")
        results = wrapper.query("a1")
        assert len(results) == 1

    def test_reset(self):
        space = AgentMemorySpace()
        space.register_agent("a1", "Agent1")
        space.create_pool("a1")
        space.reset()
        assert len(space.get_agent_pools("a1")) == 0
