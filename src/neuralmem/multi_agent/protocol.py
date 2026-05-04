"""Inter-agent memory communication protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
import time
import uuid


class MessageType(Enum):
    QUERY = auto()
    UPDATE = auto()
    BROADCAST = auto()
    ARBITRATE = auto()


@dataclass
class AgentMessage:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    recipient_id: str | None = None  # None = broadcast
    message_type: MessageType = MessageType.QUERY
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueryResult:
    message_id: str = ""
    agent_id: str = ""
    results: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


class AgentMemoryProtocol:
    """Protocol for agent-to-agent memory communication."""

    def __init__(self, message_handler: Callable[[AgentMessage], Any] | None = None) -> None:
        self._handler = message_handler
        self._messages: list[AgentMessage] = []
        self._callbacks: dict[str, Callable] = {}

    def send_query(self, sender_id: str, recipient_id: str, query: dict, callback: Callable | None = None) -> AgentMessage:
        msg = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.QUERY,
            payload={"query": query},
        )
        self._messages.append(msg)
        if callback:
            self._callbacks[msg.message_id] = callback
        if self._handler:
            self._handler(msg)
        return msg

    def send_update(self, sender_id: str, recipient_id: str | None, update: dict) -> AgentMessage:
        msg = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.UPDATE,
            payload={"update": update},
        )
        self._messages.append(msg)
        if self._handler:
            self._handler(msg)
        return msg

    def broadcast(self, sender_id: str, payload: dict) -> AgentMessage:
        msg = AgentMessage(
            sender_id=sender_id,
            recipient_id=None,
            message_type=MessageType.BROADCAST,
            payload=payload,
        )
        self._messages.append(msg)
        if self._handler:
            self._handler(msg)
        return msg

    def arbitrate_conflict(self, sender_id: str, conflict: dict, agents: list[str]) -> AgentMessage:
        msg = AgentMessage(
            sender_id=sender_id,
            recipient_id=None,
            message_type=MessageType.ARBITRATE,
            payload={"conflict": conflict, "agents": agents},
        )
        self._messages.append(msg)
        if self._handler:
            self._handler(msg)
        return msg

    def get_messages(self, agent_id: str | None = None, msg_type: MessageType | None = None) -> list[AgentMessage]:
        out = self._messages
        if agent_id:
            out = [m for m in out if m.sender_id == agent_id or m.recipient_id == agent_id]
        if msg_type:
            out = [m for m in out if m.message_type == msg_type]
        return out

    def reset(self) -> None:
        self._messages.clear()
        self._callbacks.clear()
