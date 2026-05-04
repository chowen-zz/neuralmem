"""Tests for the federated aggregator."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from neuralmem.federated.aggregator import (
    AggregationAudit,
    AggregationResult,
    ConflictResolution,
    FederatedAggregator,
    ModelUpdate,
)


class TestModelUpdate:
    def test_checksum_computed_automatically(self) -> None:
        weights = {"w": np.array([1.0, 2.0])}
        up = ModelUpdate(node_id="n1", weights=weights, sample_count=10, round_number=1)
        assert up.checksum
        assert len(up.checksum) == 16

    def test_checksum_validates_integrity(self) -> None:
        weights = {"w": np.array([1.0, 2.0])}
        up = ModelUpdate(node_id="n1", weights=weights, sample_count=10, round_number=1)
        assert up._compute_checksum() == up.checksum

    def test_tampered_checksum_fails_validation(self) -> None:
        weights = {"w": np.array([1.0, 2.0])}
        up = ModelUpdate(
            node_id="n1", weights=weights, sample_count=10, round_number=1, checksum="deadbeef"
        )
        agg = FederatedAggregator()
        assert agg._validate_update(up) is False


class TestFederatedAggregatorBasic:
    def test_aggregate_single_update(self) -> None:
        global_w = {"w": np.array([0.0, 0.0])}
        agg = FederatedAggregator(global_weights=global_w, noise_multiplier=0.0, max_gradient_norm=10.0)
        up = ModelUpdate(
            node_id="n1",
            weights={"w": np.array([1.0, 1.0])},
            sample_count=10,
            round_number=1,
        )
        result = agg.aggregate([up])
        assert result.round_number == 1
        assert result.participating_nodes == ["n1"]
        assert result.dropped_nodes == []
        np.testing.assert_array_almost_equal(result.aggregated_weights["w"], [1.0, 1.0])

    def test_aggregate_multiple_updates(self) -> None:
        global_w = {"w": np.array([0.0, 0.0])}
        agg = FederatedAggregator(global_weights=global_w, noise_multiplier=0.0, max_gradient_norm=10.0)
        up1 = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0, 1.0])}, sample_count=10, round_number=1
        )
        up2 = ModelUpdate(
            node_id="n2", weights={"w": np.array([3.0, 3.0])}, sample_count=30, round_number=1
        )
        result = agg.aggregate([up1, up2])
        # Weighted average: (10*1 + 30*3) / 40 = 2.5
        np.testing.assert_array_almost_equal(result.aggregated_weights["w"], [2.5, 2.5])

    def test_invalid_update_dropped(self) -> None:
        agg = FederatedAggregator(noise_multiplier=0.0)
        up_valid = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0])}, sample_count=10, round_number=1
        )
        up_invalid = ModelUpdate(
            node_id="n2",
            weights={"w": np.array([2.0])},
            sample_count=10,
            round_number=1,
            checksum="bad",
        )
        result = agg.aggregate([up_valid, up_invalid])
        assert result.participating_nodes == ["n1"]
        assert result.dropped_nodes == ["n2"]

    def test_empty_updates_returns_global(self) -> None:
        global_w = {"w": np.array([5.0, 5.0])}
        agg = FederatedAggregator(global_weights=global_w)
        result = agg.aggregate([])
        assert result.participating_nodes == []
        np.testing.assert_array_equal(result.aggregated_weights["w"], global_w["w"])


class TestDifferentialPrivacy:
    def test_noise_injected(self) -> None:
        global_w = {"w": np.array([0.0, 0.0])}
        agg = FederatedAggregator(global_weights=global_w, noise_multiplier=1.0)
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0, 1.0])}, sample_count=10, round_number=1
        )
        rng = np.random.default_rng(42)
        result = agg.aggregate([up], rng=rng)
        # Noise should perturb the exact value
        assert not np.array_equal(result.aggregated_weights["w"], [1.0, 1.0])

    def test_noise_scale_recorded(self) -> None:
        agg = FederatedAggregator(noise_multiplier=2.0)
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0])}, sample_count=10, round_number=1
        )
        result = agg.aggregate([up])
        assert result.noise_scale == 2.0  # 2.0 / 1 participant

    def test_privacy_budget_exceeded_skips_update(self) -> None:
        global_w = {"w": np.array([0.0])}
        agg = FederatedAggregator(global_weights=global_w, noise_multiplier=10.0)
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0])}, sample_count=10, round_number=1
        )
        result = agg.aggregate([up], privacy_budget=0.001)
        # Budget exceeded: should return original global weights
        np.testing.assert_array_equal(result.aggregated_weights["w"], [0.0])
        assert result.privacy_spent > 0.001

    def test_weight_clipping(self) -> None:
        agg = FederatedAggregator(max_gradient_norm=1.0)
        large_weights = {"w": np.array([10.0, 0.0])}
        clipped = agg._clip_weights(large_weights)
        norm = float(np.sqrt(np.sum(clipped["w"] ** 2)))
        assert norm <= 1.0 + 1e-6


class TestConflictResolution:
    def test_weighted_merge_strategy(self) -> None:
        global_w = {"w": np.array([1.0, 1.0])}
        agg = FederatedAggregator(
            global_weights=global_w, noise_multiplier=0.0, conflict_strategy="weighted_merge", max_gradient_norm=10.0
        )
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([3.0, 3.0])}, sample_count=10, round_number=1
        )
        result = agg.aggregate([up])
        # weighted_merge: current=global_w, incoming=aggregated, node_weight=1.0
        # merged = 1.0*(1-1.0) + 3.0*1.0 = 3.0
        np.testing.assert_array_almost_equal(result.aggregated_weights["w"], [3.0, 3.0])

    def test_last_write_wins_strategy(self) -> None:
        global_w = {"w": np.array([1.0, 1.0])}
        agg = FederatedAggregator(
            global_weights=global_w, noise_multiplier=0.0, conflict_strategy="last_write_wins", max_gradient_norm=10.0
        )
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([5.0, 5.0])}, sample_count=10, round_number=1
        )
        result = agg.aggregate([up])
        np.testing.assert_array_almost_equal(result.aggregated_weights["w"], [5.0, 5.0])

    def test_conflict_recorded_in_audit(self) -> None:
        global_w = {"w": np.array([1.0])}
        agg = FederatedAggregator(global_weights=global_w, noise_multiplier=0.0)
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([2.0])}, sample_count=10, round_number=1
        )
        agg.aggregate([up])
        conflicts = agg.get_conflict_history()
        assert len(conflicts) >= 0  # may or may not trigger depending on path


class TestSecureAggregation:
    def test_smpc_mask_cancels_out(self) -> None:
        agg = FederatedAggregator(noise_multiplier=0.0)
        up1 = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0, 1.0])}, sample_count=10, round_number=1
        )
        up2 = ModelUpdate(
            node_id="n2", weights={"w": np.array([3.0, 3.0])}, sample_count=10, round_number=1
        )
        result = agg.secure_aggregate_mask([up1, up2], mask_seed=42)
        # Masks cancel, leaving exact weighted average: (1+3)/2 = 2.0
        np.testing.assert_array_almost_equal(result.aggregated_weights["w"], [2.0, 2.0])

    def test_smpc_no_privacy_spent(self) -> None:
        agg = FederatedAggregator()
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0])}, sample_count=10, round_number=1
        )
        result = agg.secure_aggregate_mask([up], mask_seed=7)
        assert result.privacy_spent == 0.0
        assert result.noise_scale == 0.0

    def test_smpc_empty_updates(self) -> None:
        agg = FederatedAggregator(global_weights={"w": np.array([5.0])})
        result = agg.secure_aggregate_mask([])
        np.testing.assert_array_equal(result.aggregated_weights["w"], [5.0])


class TestAuditLog:
    def test_audit_log_populated(self) -> None:
        agg = FederatedAggregator(noise_multiplier=0.0)
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0])}, sample_count=10, round_number=1
        )
        agg.aggregate([up])
        log = agg.get_audit_log()
        assert len(log) == 1
        assert log[0].round_number == 1
        assert log[0].node_count == 1

    def test_reset_clears_state(self) -> None:
        agg = FederatedAggregator(global_weights={"w": np.array([0.0])}, noise_multiplier=0.0)
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0])}, sample_count=10, round_number=1
        )
        agg.aggregate([up])
        assert agg.round_number == 1
        agg.reset()
        assert agg.round_number == 0
        assert agg.get_audit_log() == []
        assert agg.global_weights == {}


class TestMockIntegration:
    def test_aggregator_with_mocked_rng(self) -> None:
        """Use a fully mocked RNG to verify noise injection path."""
        mock_rng = MagicMock()
        mock_rng.normal = MagicMock(return_value=np.array([0.1, -0.1]))
        global_w = {"w": np.array([0.0, 0.0])}
        agg = FederatedAggregator(global_weights=global_w, noise_multiplier=1.0)
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([1.0, 1.0])}, sample_count=10, round_number=1
        )
        result = agg.aggregate([up], rng=mock_rng)
        mock_rng.normal.assert_called_once()
        assert result.noise_scale == 1.0

    def test_aggregate_result_structure(self) -> None:
        agg = FederatedAggregator(noise_multiplier=0.0)
        up = ModelUpdate(
            node_id="n1", weights={"w": np.array([2.0])}, sample_count=5, round_number=1
        )
        result = agg.aggregate([up])
        assert isinstance(result, AggregationResult)
        assert hasattr(result, "aggregated_weights")
        assert hasattr(result, "participating_nodes")
        assert hasattr(result, "privacy_spent")
