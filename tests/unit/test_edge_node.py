"""Tests for the edge memory node."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from neuralmem.federated.edge_node import (
    EdgeMemoryNode,
    NodeState,
    SyncMetrics,
    TrainingResult,
)
from neuralmem.federated.privacy import PrivacyBudgetManager


class TestEdgeNodeLifecycle:
    def test_initial_state_is_idle(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1")
        assert node.get_state() == NodeState.IDLE

    def test_training_changes_state(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1")
        result = node.train(data_samples=10, epochs=1)
        assert isinstance(result, TrainingResult)
        assert node.get_state() == NodeState.IDLE
        assert node.training_count == 1

    def test_training_produces_deltas(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1", local_weights={"w": np.array([0.0, 0.0])})
        result = node.train(data_samples=10, epochs=2, learning_rate=0.1)
        assert result.sample_count == 10
        assert result.epochs == 2
        assert "w" in result.weights_delta

    def test_training_with_rng_reproducible(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1", local_weights={"w": np.array([0.0])})
        rng = np.random.default_rng(123)
        result1 = node.train(data_samples=5, epochs=1, rng=rng)
        node2 = EdgeMemoryNode(node_id="edge-2", local_weights={"w": np.array([0.0])})
        rng2 = np.random.default_rng(123)
        result2 = node2.train(data_samples=5, epochs=1, rng=rng2)
        np.testing.assert_array_almost_equal(
            result1.weights_delta["w"], result2.weights_delta["w"]
        )


class TestSyncScheduling:
    def test_should_sync_after_interval(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1", sync_interval_sec=10.0)
        assert node.should_sync(current_time=15.0) is True

    def test_should_not_sync_before_interval(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1", sync_interval_sec=10.0)
        node.last_sync_time = 10.0
        assert node.should_sync(current_time=15.0) is False

    def test_should_not_sync_with_low_bandwidth(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1", sync_interval_sec=1.0, min_bandwidth_mbps=5.0)
        node.set_bandwidth_estimator(lambda: 1.0)
        assert node.should_sync(current_time=10.0) is False

    def test_should_sync_with_adequate_bandwidth(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1", sync_interval_sec=1.0, min_bandwidth_mbps=5.0)
        node.set_bandwidth_estimator(lambda: 10.0)
        assert node.should_sync(current_time=10.0) is True


class TestPrepareUpdate:
    def test_prepare_update_returns_model_update(self) -> None:
        node = EdgeMemoryNode(
            node_id="edge-1",
            local_weights={"w": np.array([1.0, 2.0])},
            privacy_manager=PrivacyBudgetManager(epsilon_budget=10.0),
        )
        up = node.prepare_update(round_number=3)
        assert up is not None
        assert up.node_id == "edge-1"
        assert up.round_number == 3
        np.testing.assert_array_equal(up.weights["w"], [1.0, 2.0])

    def test_prepare_update_none_when_budget_exhausted(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=0.01)
        pm.spend(epsilon=0.01)
        node = EdgeMemoryNode(
            node_id="edge-1",
            local_weights={"w": np.array([1.0])},
            privacy_manager=pm,
        )
        assert node.prepare_update() is None


class TestSyncWithAggregator:
    def test_sync_success(self) -> None:
        mock_agg = MagicMock()
        mock_agg.round_number = 5
        mock_result = MagicMock()
        mock_result.aggregated_weights = {"w": np.array([2.0, 2.0])}
        mock_agg.aggregate.return_value = mock_result

        node = EdgeMemoryNode(
            node_id="edge-1",
            local_weights={"w": np.array([1.0, 1.0])},
            sync_interval_sec=1.0,
            privacy_manager=PrivacyBudgetManager(epsilon_budget=10.0),
        )
        metrics = node.sync(mock_agg, current_time=10.0, bandwidth_mbps=10.0)
        assert isinstance(metrics, SyncMetrics)
        assert metrics.dropped_updates == 0
        assert metrics.bandwidth_mbps == 10.0
        assert node.sync_count == 1
        mock_agg.aggregate.assert_called_once()

    def test_sync_skipped_due_to_low_bandwidth(self) -> None:
        mock_agg = MagicMock()
        node = EdgeMemoryNode(
            node_id="edge-1",
            local_weights={"w": np.array([1.0])},
            sync_interval_sec=1.0,
            min_bandwidth_mbps=5.0,
        )
        metrics = node.sync(mock_agg, current_time=10.0, bandwidth_mbps=1.0)
        assert metrics.dropped_updates == 1
        assert node.sync_count == 0
        mock_agg.aggregate.assert_not_called()

    def test_sync_error_state(self) -> None:
        mock_agg = MagicMock()
        mock_agg.aggregate.side_effect = RuntimeError("aggregator failure")

        node = EdgeMemoryNode(
            node_id="edge-1",
            local_weights={"w": np.array([1.0])},
            sync_interval_sec=1.0,
            privacy_manager=PrivacyBudgetManager(epsilon_budget=10.0),
        )
        metrics = node.sync(mock_agg, current_time=10.0, bandwidth_mbps=10.0)
        assert node.get_state() == NodeState.ERROR
        assert metrics.dropped_updates == 1

    def test_sync_applies_aggregated_weights(self) -> None:
        mock_agg = MagicMock()
        mock_agg.round_number = 2
        mock_result = MagicMock()
        mock_result.aggregated_weights = {"w": np.array([9.0])}
        mock_agg.aggregate.return_value = mock_result

        node = EdgeMemoryNode(
            node_id="edge-1",
            local_weights={"w": np.array([1.0])},
            sync_interval_sec=1.0,
            privacy_manager=PrivacyBudgetManager(epsilon_budget=10.0),
        )
        node.sync(mock_agg, current_time=10.0, bandwidth_mbps=10.0)
        np.testing.assert_array_equal(node.local_weights["w"], [9.0])


class TestMetricsAndHistory:
    def test_metrics_recorded_after_sync(self) -> None:
        mock_agg = MagicMock()
        mock_agg.round_number = 1
        mock_result = MagicMock()
        mock_result.aggregated_weights = {"w": np.array([0.0])}
        mock_agg.aggregate.return_value = mock_result

        node = EdgeMemoryNode(
            node_id="edge-1",
            local_weights={"w": np.array([1.0])},
            sync_interval_sec=1.0,
            privacy_manager=PrivacyBudgetManager(epsilon_budget=10.0),
        )
        node.sync(mock_agg, current_time=10.0, bandwidth_mbps=10.0)
        history = node.get_metrics()
        assert len(history) == 1
        assert history[0].bandwidth_mbps == 10.0

    def test_get_privacy_report(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=5.0)
        node = EdgeMemoryNode(node_id="edge-1", privacy_manager=pm)
        node.train(data_samples=5)
        report = node.get_privacy_report()
        assert report.total_epsilon_spent >= 0.0
        assert report.risk_level in ("low", "medium", "high", "exhausted")


class TestReset:
    def test_reset_clears_state(self) -> None:
        node = EdgeMemoryNode(
            node_id="edge-1",
            local_weights={"w": np.array([1.0])},
            privacy_manager=PrivacyBudgetManager(epsilon_budget=1.0),
        )
        node.train(data_samples=5)
        node.reset()
        assert node.get_state() == NodeState.IDLE
        assert node.training_count == 0
        assert node.sync_count == 0
        assert node.get_metrics() == []
        assert node.privacy.remaining_epsilon == 1.0


class TestMockBandwidthEstimator:
    def test_custom_bandwidth_estimator(self) -> None:
        node = EdgeMemoryNode(node_id="edge-1", sync_interval_sec=1.0, min_bandwidth_mbps=2.0)
        node.set_bandwidth_estimator(lambda: 1.0)
        assert node.should_sync(current_time=10.0) is False
        node.set_bandwidth_estimator(lambda: 100.0)
        assert node.should_sync(current_time=10.0) is True
