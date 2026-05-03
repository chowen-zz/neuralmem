"""6-state governance state machine for NeuralMem memories.

States: DRAFT -> ACTIVE -> REVIEW / STALE / SUPERSEDED -> ARCHIVED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from neuralmem.core.exceptions import NeuralMemError
from neuralmem.core.types import Memory


class MemoryState(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    REVIEW = "review"
    STALE = "stale"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class StateTransitionError(NeuralMemError):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: MemoryState, to_state: MemoryState, reason: str = ""):
        self.from_state = from_state
        self.to_state = to_state
        msg = f"Invalid transition: {from_state.value} -> {to_state.value}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


# Valid transitions: from_state -> set of allowed to_states
_VALID_TRANSITIONS: dict[MemoryState, set[MemoryState]] = {
    MemoryState.DRAFT: {MemoryState.ACTIVE},
    MemoryState.ACTIVE: {MemoryState.REVIEW, MemoryState.STALE, MemoryState.SUPERSEDED},
    MemoryState.REVIEW: {MemoryState.ACTIVE, MemoryState.ARCHIVED},
    MemoryState.STALE: {MemoryState.ACTIVE, MemoryState.ARCHIVED},
    MemoryState.SUPERSEDED: set(),  # terminal
    MemoryState.ARCHIVED: set(),    # terminal
}


@dataclass
class StateHistoryEntry:
    memory_id: str
    from_state: MemoryState
    to_state: MemoryState
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GovernanceState:
    """Manages governance state transitions and history for memories."""

    def __init__(self) -> None:
        self._current_states: dict[str, MemoryState] = {}
        self._history: list[StateHistoryEntry] = []

    @staticmethod
    def infer_state(memory: Memory) -> MemoryState:
        """Infer the governance state of a memory from its fields."""
        if memory.superseded_by is not None or not memory.is_active:
            return MemoryState.SUPERSEDED
        if memory.expires_at is not None:
            now = datetime.now(timezone.utc)
            if memory.expires_at < now:
                return MemoryState.STALE
        return MemoryState.ACTIVE

    def get_state(self, memory_id: str) -> MemoryState | None:
        """Get the current tracked state of a memory, or None if untracked."""
        return self._current_states.get(memory_id)

    def track(self, memory_id: str, state: MemoryState) -> None:
        """Register a memory with a known state (without recording a transition)."""
        self._current_states[memory_id] = state

    def transition(
        self,
        memory_id: str,
        from_state: MemoryState,
        to_state: MemoryState,
        reason: str,
    ) -> StateHistoryEntry:
        """Validate and record a state transition.

        Raises StateTransitionError if the transition is invalid.
        """
        allowed = _VALID_TRANSITIONS.get(from_state, set())
        if to_state not in allowed:
            raise StateTransitionError(from_state, to_state, reason)

        entry = StateHistoryEntry(
            memory_id=memory_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
        )
        self._history.append(entry)
        self._current_states[memory_id] = to_state
        return entry

    def get_history(self, memory_id: str | None = None) -> list[StateHistoryEntry]:
        """Return state transition history, optionally filtered by memory_id."""
        if memory_id is None:
            return list(self._history)
        return [e for e in self._history if e.memory_id == memory_id]
