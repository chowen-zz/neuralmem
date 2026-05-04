"""Edge memory node with on-device training, periodic sync, and bandwidth adaptation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from neuralmem.federated.aggregator import ModelUpdate
from neuralmem.federated.privacy import PrivacyBudgetManager


class NodeState(Enum):
    """Lifecycle states of an edge memory node."""

    IDLE = "idle"
    TRAINING = "training"
    SYNCING = "syncing"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class SyncMetrics:
    """Metrics from a synchronization event."""

    bytes_sent: int = 0
    bytes_received: int = 0
    duration_sec: float = 0.0
    bandwidth_mbps: float = 0.0
    dropped_updates: int = 0


@dataclass
class TrainingResult:
    """Result of a local training round."""

    loss: float
    sample_count: int
    epochs: int
    weights_delta: dict[str, np.ndarray]


class EdgeMemoryNode:
    """Edge node for federated memory learning.

    Features:
    - On-device memory training with local weight updates
    - Periodic sync strategy with configurable intervals
    - Bandwidth-adaptive transmission (skip sync if bandwidth is low)
    - Privacy budget integration
    - Lifecycle management (train/sync/idle/offline/error)
    """

    def __init__(
        self,
        node_id: str,
        local_weights: dict[str, np.ndarray] | None = None,
        sync_interval_sec: float = 60.0,
        min_bandwidth_mbps: float = 0.5,
        privacy_manager: PrivacyBudgetManager | None = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.node_id = node_id
        self.local_weights = local_weights or {}
        self.sync_interval_sec = sync_interval_sec
        self.min_bandwidth_mbps = min_bandwidth_mbps
        self.privacy = privacy_manager or PrivacyBudgetManager()
        self.max_grad_norm = max_grad_norm

        self.state = NodeState.IDLE
        self.last_sync_time: float = 0.0
        self.sync_count = 0
        self.training_count = 0
        self.metrics_history: list[SyncMetrics] = []
        self._error_message: str = ""

    def _estimate_bandwidth(self) -> float:
        """Estimate current bandwidth in Mbps (mock-friendly, overrideable)."""
        # Default mock: assume adequate bandwidth
        return 10.0

    def train(
        self,
        data_samples: int,
        epochs: int = 1,
        learning_rate: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> TrainingResult:
        """Perform on-device training and produce weight deltas.

        Args:
            data_samples: Number of local samples trained on.
            epochs: Training epochs.
            learning_rate: Learning rate for SGD-style update.
            rng: Optional RNG for reproducible tests.

        Returns:
            TrainingResult with loss, sample count, and weight deltas.
        """
        self.state = NodeState.TRAINING
        self.training_count += 1
        rng = rng or np.random.default_rng()

        if not self.local_weights:
            # Initialize dummy weights for testing
            self.local_weights = {"mem_layer_0": np.zeros(64)}

        deltas: dict[str, np.ndarray] = {}
        total_loss = 0.0
        for _ in range(epochs):
            for key, w in self.local_weights.items():
                # Simulate gradient as small random perturbation
                grad = rng.standard_normal(size=w.shape) * learning_rate
                # Clip gradient
                norm = float(np.sqrt(np.sum(grad * grad)))
                if norm > self.max_grad_norm:
                    grad = grad * (self.max_grad_norm / norm)
                deltas[key] = grad
                self.local_weights[key] = w - grad
                total_loss += float(np.sum(grad * grad))

        self.state = NodeState.IDLE
        return TrainingResult(
            loss=total_loss / max(epochs, 1),
            sample_count=data_samples,
            epochs=epochs,
            weights_delta=deltas,
        )

    def should_sync(self, current_time: float | None = None) -> bool:
        """Determine if the node should sync based on interval and bandwidth.

        Args:
            current_time: Optional timestamp; defaults to time.time().

        Returns:
            True if sync should proceed.
        """
        now = current_time if current_time is not None else time.time()
        if now - self.last_sync_time < self.sync_interval_sec:
            return False
        bandwidth = self._estimate_bandwidth()
        return bandwidth >= self.min_bandwidth_mbps

    def prepare_update(self, round_number: int = 0) -> ModelUpdate | None:
        """Prepare a ModelUpdate from current local weights for aggregation.

        Args:
            round_number: Current federated round number.

        Returns:
            ModelUpdate or None if privacy budget is exhausted.
        """
        if self.privacy.budget_exhausted:
            return None

        # Auto-spend privacy budget for this update
        auto_result = self.privacy.auto_spend_and_noise(
            sensitivity=1.0,
            mechanism="gaussian",
            description=f"node {self.node_id} update round {round_number}",
        )
        if not auto_result["success"]:
            return None

        return ModelUpdate(
            node_id=self.node_id,
            weights={k: v.copy() for k, v in self.local_weights.items()},
            sample_count=100,  # default mock sample count
            round_number=round_number,
        )

    def sync(
        self,
        aggregator: Any,
        current_time: float | None = None,
        bandwidth_mbps: float | None = None,
    ) -> SyncMetrics:
        """Synchronize local weights with the federated aggregator.

        Args:
            aggregator: FederatedAggregator instance (duck-typed for mocks).
            current_time: Optional timestamp.
            bandwidth_mbps: Optional override bandwidth estimate.

        Returns:
            SyncMetrics from the sync operation.
        """
        self.state = NodeState.SYNCING
        now = current_time if current_time is not None else time.time()
        bw = bandwidth_mbps if bandwidth_mbps is not None else self._estimate_bandwidth()

        metrics = SyncMetrics(
            bytes_sent=0,
            bytes_received=0,
            duration_sec=0.0,
            bandwidth_mbps=bw,
            dropped_updates=0,
        )

        if bw < self.min_bandwidth_mbps:
            metrics.dropped_updates = 1
            self.state = NodeState.IDLE
            return metrics

        update = self.prepare_update(round_number=getattr(aggregator, "round_number", 0))
        if update is None:
            metrics.dropped_updates = 1
            self.state = NodeState.IDLE
            return metrics

        # Simulate transmission size
        for arr in update.weights.values():
            metrics.bytes_sent += arr.nbytes

        start = time.time()
        try:
            # Duck-type call: aggregator.aggregate([update], ...)
            result = aggregator.aggregate([update])
            # Apply aggregated weights back
            if hasattr(result, "aggregated_weights"):
                self.local_weights = {
                    k: v.copy() for k, v in result.aggregated_weights.items()
                }
            metrics.bytes_received = metrics.bytes_sent  # symmetric mock
        except Exception as exc:
            self.state = NodeState.ERROR
            self._error_message = str(exc)
            metrics.dropped_updates = 1
            return metrics
        finally:
            metrics.duration_sec = time.time() - start

        self.last_sync_time = now
        self.sync_count += 1
        self.metrics_history.append(metrics)
        self.state = NodeState.IDLE
        return metrics

    def get_state(self) -> NodeState:
        """Return current lifecycle state."""
        return self.state

    def get_metrics(self) -> list[SyncMetrics]:
        """Return all recorded sync metrics."""
        return list(self.metrics_history)

    def get_privacy_report(self) -> Any:
        """Return the privacy manager's audit report."""
        return self.privacy.generate_report()

    def set_bandwidth_estimator(self, estimator: Callable[[], float]) -> None:
        """Override the bandwidth estimation function (useful for mocks/tests)."""
        self._estimate_bandwidth = estimator  # type: ignore[method-assign]

    def reset(self) -> None:
        """Reset node state and history."""
        self.state = NodeState.IDLE
        self.last_sync_time = 0.0
        self.sync_count = 0
        self.training_count = 0
        self.metrics_history.clear()
        self._error_message = ""
        self.privacy.reset()
