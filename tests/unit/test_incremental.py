"""Tests for NeuralMem V1.8 incremental memory updater."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.streaming.incremental import IncrementalMemoryUpdater, MicroBatch


class TestIngestAndBatch:
    def test_ingest(self):
        updater = IncrementalMemoryUpdater()
        updater.ingest({"content": "hello"})
        assert updater.get_pending_count() == 1

    def test_create_batch(self):
        updater = IncrementalMemoryUpdater(batch_size=2)
        updater.ingest({"a": 1})
        updater.ingest({"b": 2})
        batch = updater.create_batch()
        assert batch is not None
        assert len(batch.events) == 2

    def test_create_batch_empty(self):
        updater = IncrementalMemoryUpdater()
        assert updater.create_batch() is None


class TestProcessBatch:
    def test_process_with_default_processor(self):
        updater = IncrementalMemoryUpdater()
        updater.ingest({"a": 1})
        batch = updater.create_batch()
        result = updater.process_batch(batch)
        assert result["processed"] == 1

    def test_process_with_custom_processor(self):
        processor = MagicMock(return_value={"custom": True})
        updater = IncrementalMemoryUpdater(processor=processor)
        updater.ingest({"a": 1})
        updater.process_batch()
        processor.assert_called_once()

    def test_process_none_batch(self):
        updater = IncrementalMemoryUpdater()
        result = updater.process_batch()
        assert result == {}


class TestCheckpoint:
    def test_checkpoint(self):
        updater = IncrementalMemoryUpdater()
        updater.ingest({"a": 1})
        updater.process_batch()
        cp = updater.checkpoint()
        assert cp.sequence_num == 1
        assert any(k.startswith("last_batch") for k in cp.state)

    def test_restore(self):
        updater = IncrementalMemoryUpdater()
        updater.ingest({"a": 1})
        updater.process_batch()
        cp = updater.checkpoint()
        updater.reset()
        updater.restore(cp)
        assert updater.get_state() == cp.state


class TestReset:
    def test_reset(self):
        updater = IncrementalMemoryUpdater()
        updater.ingest({"a": 1})
        updater.reset()
        assert updater.get_pending_count() == 0
        assert len(updater._checkpoints) == 0
