"""Event-driven streaming memory pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
import time
import uuid


class BackpressureStrategy(Enum):
    DROP_OLDEST = auto()
    DROP_LATEST = auto()
    BLOCK = auto()
    BUFFER = auto()


@dataclass
class StreamEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    event_type: str = "memory_update"
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence_num: int = 0


@dataclass
class PipelineStats:
    events_processed: int = 0
    events_dropped: int = 0
    avg_latency_ms: float = 0.0
    backpressure_triggered: int = 0


class StreamingMemoryPipeline:
    """Event-driven memory pipeline with backpressure handling."""

    def __init__(
        self,
        max_buffer_size: int = 1000,
        backpressure: BackpressureStrategy = BackpressureStrategy.BUFFER,
        event_handler: Callable[[StreamEvent], Any] | None = None,
    ) -> None:
        self.max_buffer_size = max_buffer_size
        self.backpressure = backpressure
        self._event_handler = event_handler
        self._buffer: list[StreamEvent] = []
        self._stats = PipelineStats()
        self._sequence = 0
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def submit(self, event_type: str, payload: dict) -> StreamEvent | None:
        if not self._running:
            return None
        event = StreamEvent(
            event_type=event_type,
            payload=payload,
            sequence_num=self._sequence,
        )
        self._sequence += 1
        if len(self._buffer) >= self.max_buffer_size:
            self._stats.backpressure_triggered += 1
            if self.backpressure == BackpressureStrategy.DROP_OLDEST:
                self._buffer.pop(0)
                self._stats.events_dropped += 1
            elif self.backpressure == BackpressureStrategy.DROP_LATEST:
                self._stats.events_dropped += 1
                return None
            elif self.backpressure == BackpressureStrategy.BLOCK:
                # In real implementation would block; mock just drops
                self._stats.events_dropped += 1
                return None
        self._buffer.append(event)
        return event

    def process_next(self) -> Any | None:
        if not self._buffer:
            return None
        event = self._buffer.pop(0)
        start = time.time()
        result = None
        if self._event_handler:
            result = self._event_handler(event)
        latency = (time.time() - start) * 1000
        self._stats.events_processed += 1
        # Update avg latency
        total = self._stats.events_processed
        self._stats.avg_latency_ms = (
            (self._stats.avg_latency_ms * (total - 1) + latency) / total
        )
        return result

    def process_batch(self, batch_size: int = 10) -> list[Any]:
        results = []
        for _ in range(min(batch_size, len(self._buffer))):
            result = self.process_next()
            if result is not None:
                results.append(result)
            elif self._event_handler:
                # event_handler returned None but still processed
                results.append(None)
        # Filter out None values if handler didn't return anything meaningful
        return [r for r in results if r is not None]

    def get_stats(self) -> PipelineStats:
        return PipelineStats(
            events_processed=self._stats.events_processed,
            events_dropped=self._stats.events_dropped,
            avg_latency_ms=round(self._stats.avg_latency_ms, 2),
            backpressure_triggered=self._stats.backpressure_triggered,
        )

    def get_buffer_size(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._stats = PipelineStats()
        self._sequence = 0
