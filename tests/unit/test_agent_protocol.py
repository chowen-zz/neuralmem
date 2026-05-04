"""Tests for NeuralMem V1.8 agent memory protocol."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.multi_agent.protocol import AgentMemoryProtocol, AgentMessage, MessageType


class TestSendQuery:
    def test_send_query(self):
        handler = MagicMock()
        proto = AgentMemoryProtocol(message_handler=handler)
        msg = proto.send_query("a1", "a2", {"q": "test"})
        assert msg.message_type == MessageType.QUERY
        assert msg.sender_id == "a1"
        assert msg.recipient_id == "a2"
        handler.assert_called_once()

    def test_callback_registered(self):
        cb = MagicMock()
        proto = AgentMemoryProtocol()
        msg = proto.send_query("a1", "a2", {}, callback=cb)
        assert msg.message_id in proto._callbacks


class TestBroadcast:
    def test_broadcast(self):
        handler = MagicMock()
        proto = AgentMemoryProtocol(message_handler=handler)
        msg = proto.broadcast("a1", {"update": "x"})
        assert msg.message_type == MessageType.BROADCAST
        assert msg.recipient_id is None
        handler.assert_called_once()


class TestArbitrate:
    def test_arbitrate_conflict(self):
        handler = MagicMock()
        proto = AgentMemoryProtocol(message_handler=handler)
        msg = proto.arbitrate_conflict("a1", {"key": "x"}, ["a2", "a3"])
        assert msg.message_type == MessageType.ARBITRATE
        handler.assert_called_once()


class TestGetMessages:
    def test_filter_by_agent(self):
        proto = AgentMemoryProtocol()
        proto.send_query("a1", "a2", {})
        proto.send_query("a3", "a1", {})
        msgs = proto.get_messages(agent_id="a1")
        assert len(msgs) == 2

    def test_filter_by_type(self):
        proto = AgentMemoryProtocol()
        proto.send_query("a1", "a2", {})
        proto.broadcast("a1", {})
        msgs = proto.get_messages(msg_type=MessageType.BROADCAST)
        assert len(msgs) == 1

    def test_reset(self):
        proto = AgentMemoryProtocol()
        proto.send_query("a1", "a2", {})
        proto.reset()
        assert len(proto.get_messages()) == 0
