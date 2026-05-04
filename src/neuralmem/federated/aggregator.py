"""Federated aggregator with differential privacy and secure multi-party computation."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ModelUpdate:
    """A single model update from an edge node."""

    node_id: str
    weights: dict[str, np.ndarray]
    sample_count: int = 0
    round_number: int = 0
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute a simple checksum of the weights for integrity verification."""
        h = hashlib.sha256()
        for key in sorted(self.weights):
            h.update(key.encode())
            h.update(self.weights[key].tobytes())
        return h.hexdigest()[:16]


@dataclass
class AggregationResult:
    """Result of a federated aggregation round."""

    aggregated_weights: dict[str, np.ndarray]
    participating_nodes: list[str]
    dropped_nodes: list[str]
    noise_scale: float
    round_number: int
    privacy_spent: float


@dataclass
class ConflictResolution:
    """Record of a conflict resolution decision."""

    node_id: str
    field: str
    strategy: str
    accepted_value: Any
    rejected_value: Any
    reason: str


@dataclass
class AggregationAudit:
    """Audit log entry for an aggregation round."""

    round_number: int
    timestamp: float
    node_count: int
    noise_scale: float
    privacy_spent: float
    conflicts: list[ConflictResolution] = field(default_factory=list)


class FederatedAggregator:
    """Federated learning aggregator with differential privacy and SMPC basics.

    Features:
    - Differential privacy via Gaussian noise injection
    - Secure aggregation basics (weight clipping, checksum validation)
    - Model update conflict resolution (last-write-wins, weighted-merge)
    - Audit trail for privacy budget consumption
    """

    def __init__(
        self,
        global_weights: dict[str, np.ndarray] | None = None,
        noise_multiplier: float = 1.0,
        max_gradient_norm: float = 1.0,
        conflict_strategy: str = "weighted_merge",
    ) -> None:
        self.global_weights = global_weights or {}
        self.noise_multiplier = noise_multiplier
        self.max_gradient_norm = max_gradient_norm
        self.conflict_strategy = conflict_strategy
        self.round_number = 0
        self.audit_log: list[AggregationAudit] = []
        self._conflict_history: list[ConflictResolution] = []

    def _clip_weights(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Clip weight updates to bound their L2 norm (DP requirement)."""
        total_norm = 0.0
        for w in weights.values():
            total_norm += float(np.sum(w * w))
        total_norm = float(np.sqrt(total_norm))
        if total_norm == 0.0:
            return weights
        clip_factor = min(1.0, self.max_gradient_norm / total_norm)
        return {k: w * clip_factor for k, w in weights.items()}

    def _inject_noise(
        self,
        aggregated: dict[str, np.ndarray],
        noise_scale: float,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Inject Gaussian noise for differential privacy."""
        rng = rng or np.random.default_rng()
        noised: dict[str, np.ndarray] = {}
        for key, arr in aggregated.items():
            noise = rng.normal(loc=0.0, scale=noise_scale, size=arr.shape)
            noised[key] = arr + noise.astype(arr.dtype)
        return noised

    def _validate_update(self, update: ModelUpdate) -> bool:
        """Validate a model update checksum and basic integrity."""
        if not update.weights:
            return False
        expected = update._compute_checksum()
        return update.checksum == expected

    def _resolve_conflict(
        self,
        node_id: str,
        field: str,
        current: np.ndarray,
        incoming: np.ndarray,
        node_weight: float,
    ) -> tuple[np.ndarray, ConflictResolution | None]:
        """Resolve a weight conflict between current and incoming values."""
        if self.conflict_strategy == "last_write_wins":
            resolution = ConflictResolution(
                node_id=node_id,
                field=field,
                strategy="last_write_wins",
                accepted_value=incoming.copy(),
                rejected_value=current.copy(),
                reason="last write wins policy",
            )
            return incoming, resolution

        # Default: weighted_merge
        merged = current * (1 - node_weight) + incoming * node_weight
        resolution = ConflictResolution(
            node_id=node_id,
            field=field,
            strategy="weighted_merge",
            accepted_value=merged.copy(),
            rejected_value=current.copy(),
            reason=f"weighted merge with node_weight={node_weight:.4f}",
        )
        return merged, resolution

    def aggregate(
        self,
        updates: list[ModelUpdate],
        privacy_budget: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> AggregationResult:
        """Aggregate model updates with DP noise and conflict resolution.

        Args:
            updates: List of ModelUpdate from edge nodes.
            privacy_budget: Optional epsilon cap; if exceeded, aggregation is skipped.
            rng: Optional numpy RNG for reproducible tests.

        Returns:
            AggregationResult with aggregated weights and metadata.
        """
        self.round_number += 1
        valid_updates: list[ModelUpdate] = []
        dropped_nodes: list[str] = []

        for up in updates:
            if self._validate_update(up):
                valid_updates.append(up)
            else:
                dropped_nodes.append(up.node_id)

        if not valid_updates:
            return AggregationResult(
                aggregated_weights=copy.deepcopy(self.global_weights),
                participating_nodes=[],
                dropped_nodes=dropped_nodes,
                noise_scale=0.0,
                round_number=self.round_number,
                privacy_spent=0.0,
            )

        total_samples = sum(u.sample_count for u in valid_updates)
        aggregated: dict[str, np.ndarray] = {}
        conflicts: list[ConflictResolution] = []

        # Compute weighted average of clipped updates
        for up in valid_updates:
            node_weight = up.sample_count / max(total_samples, 1)
            clipped = self._clip_weights(up.weights)
            for key, arr in clipped.items():
                if key not in aggregated:
                    aggregated[key] = np.zeros_like(arr)
                aggregated[key] += arr * node_weight

        # Resolve conflicts between new aggregated weights and existing global weights
        for key in aggregated:
            if key in self.global_weights and not np.array_equal(
                aggregated[key], self.global_weights[key]
            ):
                resolved, conflict = self._resolve_conflict(
                    node_id="aggregator",
                    field=key,
                    current=self.global_weights[key],
                    incoming=aggregated[key],
                    node_weight=1.0,
                )
                aggregated[key] = resolved
                if conflict:
                    conflicts.append(conflict)

        # Noise scale based on number of participants and noise multiplier
        noise_scale = self.noise_multiplier / max(len(valid_updates), 1)
        privacy_spent = noise_scale  # Simplified privacy accounting

        if privacy_budget is not None and privacy_spent > privacy_budget:
            # Budget exceeded: return current global weights without update
            return AggregationResult(
                aggregated_weights=copy.deepcopy(self.global_weights),
                participating_nodes=[u.node_id for u in valid_updates],
                dropped_nodes=dropped_nodes,
                noise_scale=noise_scale,
                round_number=self.round_number,
                privacy_spent=privacy_spent,
            )

        aggregated = self._inject_noise(aggregated, noise_scale, rng)
        self.global_weights = aggregated

        audit = AggregationAudit(
            round_number=self.round_number,
            timestamp=0.0,  # caller can override
            node_count=len(valid_updates),
            noise_scale=noise_scale,
            privacy_spent=privacy_spent,
            conflicts=conflicts,
        )
        self.audit_log.append(audit)
        self._conflict_history.extend(conflicts)

        return AggregationResult(
            aggregated_weights=copy.deepcopy(aggregated),
            participating_nodes=[u.node_id for u in valid_updates],
            dropped_nodes=dropped_nodes,
            noise_scale=noise_scale,
            round_number=self.round_number,
            privacy_spent=privacy_spent,
        )

    def secure_aggregate_mask(
        self,
        updates: list[ModelUpdate],
        mask_seed: int | None = None,
    ) -> AggregationResult:
        """SMPC-style aggregation using random masks that cancel out.

        Each update is masked with a random vector; masks are designed to
        sum to zero across participants so the true aggregate is recovered.
        This is a basic demonstration (full SMPC requires pairwise keys).
        """
        if not updates:
            return AggregationResult(
                aggregated_weights=copy.deepcopy(self.global_weights),
                participating_nodes=[],
                dropped_nodes=[],
                noise_scale=0.0,
                round_number=self.round_number,
                privacy_spent=0.0,
            )

        self.round_number += 1
        rng = np.random.default_rng(mask_seed)
        total_samples = sum(u.sample_count for u in updates)

        # Generate zero-sum masks
        masks: dict[str, np.ndarray] = {}
        first_weights = updates[0].weights
        for key, arr in first_weights.items():
            mask_vals = rng.standard_normal(size=(len(updates), *arr.shape))
            mask_vals -= mask_vals.mean(axis=0)
            masks[key] = mask_vals

        aggregated: dict[str, np.ndarray] = {}
        for idx, up in enumerate(updates):
            node_weight = up.sample_count / max(total_samples, 1)
            for key, arr in up.weights.items():
                masked = arr + masks[key][idx]
                if key not in aggregated:
                    aggregated[key] = np.zeros_like(arr)
                aggregated[key] += masked * node_weight

        # Masks cancel out, leaving true weighted average
        privacy_spent = 0.0  # No additional DP noise in pure SMPC path
        noise_scale = 0.0
        self.global_weights = aggregated

        audit = AggregationAudit(
            round_number=self.round_number,
            timestamp=0.0,
            node_count=len(updates),
            noise_scale=noise_scale,
            privacy_spent=privacy_spent,
            conflicts=[],
        )
        self.audit_log.append(audit)

        return AggregationResult(
            aggregated_weights=copy.deepcopy(aggregated),
            participating_nodes=[u.node_id for u in updates],
            dropped_nodes=[],
            noise_scale=noise_scale,
            round_number=self.round_number,
            privacy_spent=privacy_spent,
        )

    def get_audit_log(self) -> list[AggregationAudit]:
        """Return the full aggregation audit log."""
        return list(self.audit_log)

    def get_conflict_history(self) -> list[ConflictResolution]:
        """Return all conflict resolutions performed."""
        return list(self._conflict_history)

    def reset(self) -> None:
        """Reset aggregator state (weights, round counter, audit log)."""
        self.global_weights = {}
        self.round_number = 0
        self.audit_log.clear()
        self._conflict_history.clear()
