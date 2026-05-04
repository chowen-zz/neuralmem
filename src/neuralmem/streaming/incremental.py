"""Incremental memory updater with checkpointing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time


@dataclass
class Checkpoint:
    checkpoint_id: str
    sequence_num: int
    state: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class MicroBatch:
    batch_id: str
    events: list[dict]
    start_seq: int
    end_seq: int
    timestamp: float = field(default_factory=time.time)


class IncrementalMemoryUpdater:
    """Micro-batch incremental processing with state store."""

    def __init__(
        self,
        batch_size: int = 100,
        state_store: dict | None = None,
        processor: Callable[[list[dict]], dict] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self._state = state_store or {}
        self._processor = processor
        self._checkpoints: list[Checkpoint] = []
        self._pending: list[dict] = []
        self._sequence = 0
        self._batch_count = 0

    def ingest(self, event: dict) -> None:
        event["_seq"] = self._sequence
        self._sequence += 1
        self._pending.append(event)

    def create_batch(self) -> MicroBatch | None:
        if not self._pending:
            return None
        batch_events = self._pending[: self.batch_size]
        self._pending = self._pending[self.batch_size :]
        batch = MicroBatch(
            batch_id=f"batch_{self._batch_count}",
            events=batch_events,
            start_seq=batch_events[0]["_seq"],
            end_seq=batch_events[-1]["_seq"],
        )
        self._batch_count += 1
        return batch

    def process_batch(self, batch: MicroBatch | None = None) -> dict:
        if batch is None:
            batch = self.create_batch()
        if batch is None:
            return {}
        if self._processor:
            result = self._processor(batch.events)
        else:
            result = {"processed": len(batch.events), "batch_id": batch.batch_id}
        # Update state
        self._state[f"last_batch_{batch.batch_id}"] = result
        return result

    def checkpoint(self) -> Checkpoint:
        cp = Checkpoint(
            checkpoint_id=f"cp_{len(self._checkpoints)}",
            sequence_num=self._sequence,
            state=dict(self._state),
        )
        self._checkpoints.append(cp)
        return cp

    def restore(self, checkpoint: Checkpoint) -> None:
        self._state = dict(checkpoint.state)
        self._sequence = checkpoint.sequence_num

    def get_state(self) -> dict:
        return dict(self._state)

    def get_pending_count(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()
        self._checkpoints.clear()
        self._state.clear()
        self._sequence = 0
        self._batch_count = 0
