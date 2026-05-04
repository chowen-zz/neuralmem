"""Tests for NeuralMem V1.8 streaming pipeline."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.streaming.pipeline import StreamingMemoryPipeline, StreamEvent, BackpressureStrategy


class TestStreamEvent:
    def test_create_event(self):
        e = StreamEvent(event_type="test", payload={"k": "v"})
        assert e.event_type == "test"
        assert e.event_id is not None


class TestPipelineSubmit:
    def test_submit_when_running(self):
        pipe = StreamingMemoryPipeline()
        pipe.start()
        e = pipe.submit("memory_update", {"content": "hello"})
        assert e is not None
        assert e.event_type == "memory_update"

    def test_submit_when_stopped(self):
        pipe = StreamingMemoryPipeline()
        e = pipe.submit("test", {})
        assert e is None

    def test_buffer_limit_drop_oldest(self):
        pipe = StreamingMemoryPipeline(max_buffer_size=2, backpressure=BackpressureStrategy.DROP_OLDEST)
        pipe.start()
        pipe.submit("a", {})
        pipe.submit("b", {})
        pipe.submit("c", {})
        assert pipe.get_buffer_size() == 2

    def test_buffer_limit_drop_latest(self):
        pipe = StreamingMemoryPipeline(max_buffer_size=2, backpressure=BackpressureStrategy.DROP_LATEST)
        pipe.start()
        pipe.submit("a", {})
        pipe.submit("b", {})
        e = pipe.submit("c", {})
        assert e is None
        assert pipe.get_buffer_size() == 2


class TestPipelineProcess:
    def test_process_next(self):
        handler = MagicMock(return_value="processed")
        pipe = StreamingMemoryPipeline(event_handler=handler)
        pipe.start()
        pipe.submit("test", {"k": "v"})
        result = pipe.process_next()
        assert result == "processed"
        handler.assert_called_once()

    def test_process_batch(self):
        pipe = StreamingMemoryPipeline()
        pipe.start()
        for i in range(5):
            pipe.submit("test", {"i": i})
        results = pipe.process_batch(batch_size=3)
        assert len(results) <= 3

    def test_process_empty(self):
        pipe = StreamingMemoryPipeline()
        pipe.start()
        assert pipe.process_next() is None


class TestStats:
    def test_get_stats(self):
        pipe = StreamingMemoryPipeline()
        pipe.start()
        pipe.submit("test", {})
        pipe.process_next()
        stats = pipe.get_stats()
        assert stats.events_processed == 1

    def test_reset(self):
        pipe = StreamingMemoryPipeline()
        pipe.start()
        pipe.submit("test", {})
        pipe.reset()
        assert pipe.get_buffer_size() == 0
