"""Agent memory spaces with isolation and sharing."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import time
import uuid


class Permission(Enum):
    READ = auto()
    WRITE = auto()
    ADMIN = auto()


@dataclass
class Agent:
    agent_id: str
    name: str
    permissions: dict[str, list[Permission]] = field(default_factory=dict)
    private_memory: list[dict] = field(default_factory=list)


@dataclass
class SharedPool:
    pool_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    memories: list[dict] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class AgentMemorySpace:
    """Manages private and shared memory for multiple agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._pools: dict[str, SharedPool] = {}

    def register_agent(self, agent_id: str, name: str) -> Agent:
        agent = Agent(agent_id=agent_id, name=name)
        self._agents[agent_id] = agent
        return agent

    def create_pool(self, agent_id: str, pool_name: str | None = None) -> SharedPool:
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not registered")
        pool = SharedPool(pool_id=pool_name or str(uuid.uuid4())[:8])
        pool.agents.append(agent_id)
        self._pools[pool.pool_id] = pool
        # Grant admin permission
        self._agents[agent_id].permissions[pool.pool_id] = [Permission.ADMIN]
        return pool

    def add_to_pool(self, pool_id: str, agent_id: str, permission: Permission = Permission.READ) -> None:
        if pool_id not in self._pools:
            raise ValueError(f"Pool {pool_id} not found")
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not registered")
        pool = self._pools[pool_id]
        if agent_id not in pool.agents:
            pool.agents.append(agent_id)
        perms = self._agents[agent_id].permissions.setdefault(pool_id, [])
        if permission not in perms:
            perms.append(permission)

    def store_private(self, agent_id: str, memory: dict) -> None:
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not registered")
        self._agents[agent_id].private_memory.append({
            **memory,
            "_stored_at": time.time(),
            "_private": True,
        })

    def store_shared(self, pool_id: str, memory: dict, agent_id: str) -> None:
        if pool_id not in self._pools:
            raise ValueError(f"Pool {pool_id} not found")
        perms = self._agents.get(agent_id, Agent(agent_id=agent_id, name="")).permissions.get(pool_id, [])
        if Permission.WRITE not in perms and Permission.ADMIN not in perms:
            raise PermissionError(f"Agent {agent_id} cannot write to pool {pool_id}")
        self._pools[pool_id].memories.append({
            **memory,
            "_stored_at": time.time(),
            "_agent_id": agent_id,
        })

    def query_private(self, agent_id: str, limit: int = 10) -> list[dict]:
        if agent_id not in self._agents:
            return []
        return self._agents[agent_id].private_memory[-limit:]

    def query_shared(self, pool_id: str, agent_id: str, limit: int = 10) -> list[dict]:
        if pool_id not in self._pools:
            return []
        perms = self._agents.get(agent_id, Agent(agent_id=agent_id, name="")).permissions.get(pool_id, [])
        if Permission.READ not in perms and Permission.WRITE not in perms and Permission.ADMIN not in perms:
            raise PermissionError(f"Agent {agent_id} cannot read pool {pool_id}")
        return self._pools[pool_id].memories[-limit:]

    def get_agent_pools(self, agent_id: str) -> list[str]:
        if agent_id not in self._agents:
            return []
        return list(self._agents[agent_id].permissions.keys())

    def get_pool_agents(self, pool_id: str) -> list[str]:
        if pool_id not in self._pools:
            return []
        return list(self._pools[pool_id].agents)

    def reset(self) -> None:
        self._agents.clear()
        self._pools.clear()


class AgentMemoryPool:
    """Convenience wrapper for a specific pool."""

    def __init__(self, space: AgentMemorySpace, pool_id: str) -> None:
        self._space = space
        self._pool_id = pool_id

    def share(self, memory: dict, agent_id: str) -> None:
        self._space.store_shared(self._pool_id, memory, agent_id)

    def query(self, agent_id: str, limit: int = 10) -> list[dict]:
        return self._space.query_shared(self._pool_id, agent_id, limit)
