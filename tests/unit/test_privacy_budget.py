"""Tests for the privacy budget manager."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neuralmem.federated.privacy import (
    PrivacyAuditReport,
    PrivacyBudgetManager,
    PrivacyEvent,
)


class TestBudgetSpending:
    def test_spend_reduces_remaining(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        result = pm.spend(epsilon=1.0)
        assert result["success"] is True
        assert result["spent"] == 1.0
        assert result["remaining"] == 3.0
        assert pm.remaining_epsilon == 3.0

    def test_spend_exceeding_budget_fails(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=0.5)
        result = pm.spend(epsilon=1.0)
        assert result["success"] is False
        assert result["spent"] == 0.0
        assert result["reason"] == "budget_exceeded"

    def test_spend_negative_epsilon_raises(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        with pytest.raises(ValueError, match="non-negative"):
            pm.spend(epsilon=-0.1)

    def test_budget_exhausted_property(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=1.0)
        assert pm.budget_exhausted is False
        pm.spend(epsilon=1.0)
        assert pm.budget_exhausted is True


class TestNoiseCalibration:
    def test_calibrate_noise_gaussian(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0, delta=1e-5)
        sigma = pm.calibrate_noise(sensitivity=1.0, target_epsilon=1.0, mechanism="gaussian")
        assert sigma > 0.0
        assert sigma <= pm.max_noise_scale

    def test_calibrate_noise_laplace(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        sigma = pm.calibrate_noise(sensitivity=1.0, target_epsilon=1.0, mechanism="laplace")
        assert sigma > 0.0
        assert sigma == pytest.approx(1.0, rel=1e-6)

    def test_calibrate_noise_clipped_to_max(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0, max_noise_scale=0.5)
        sigma = pm.calibrate_noise(sensitivity=10.0, target_epsilon=0.01, mechanism="gaussian")
        assert sigma == 0.5

    def test_calibrate_noise_clipped_to_min(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0, min_noise_scale=5.0)
        sigma = pm.calibrate_noise(sensitivity=0.001, target_epsilon=10.0, mechanism="gaussian")
        assert sigma == 5.0

    def test_calibrate_noise_zero_epsilon_returns_max(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=0.0)
        sigma = pm.calibrate_noise(sensitivity=1.0, mechanism="gaussian")
        assert sigma == pm.max_noise_scale


class TestAutoSpendAndNoise:
    def test_auto_spend_success(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=10.0)
        result = pm.auto_spend_and_noise(sensitivity=1.0, mechanism="gaussian")
        assert result["success"] is True
        assert result["noise_scale"] > 0.0
        assert result["epsilon_spent"] > 0.0
        assert result["event_id"] is not None

    def test_auto_spend_exhausted_budget(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=0.0)
        result = pm.auto_spend_and_noise(sensitivity=1.0)
        assert result["success"] is False
        assert result["reason"] == "budget_exhausted"
        assert result["noise_scale"] == pm.max_noise_scale

    def test_auto_spend_records_event(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=5.0)
        pm.auto_spend_and_noise(description="test query")
        trail = pm.get_audit_trail()
        assert len(trail) == 1
        assert trail[0].description == "test query"
        assert trail[0].mechanism == "gaussian"


class TestAuditTrail:
    def test_get_audit_trail_empty(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        assert pm.get_audit_trail() == []

    def test_get_audit_trail_populated(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        pm.spend(epsilon=0.5, mechanism="gaussian", description="q1")
        pm.spend(epsilon=1.0, mechanism="laplace", description="q2")
        trail = pm.get_audit_trail()
        assert len(trail) == 2
        assert trail[0].mechanism == "gaussian"
        assert trail[1].mechanism == "laplace"
        assert all(isinstance(e, PrivacyEvent) for e in trail)

    def test_event_ids_are_unique(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=10.0)
        for i in range(5):
            pm.spend(epsilon=0.1)
        ids = [e.event_id for e in pm.get_audit_trail()]
        assert len(set(ids)) == len(ids)


class TestReportGeneration:
    def test_report_structure(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        report = pm.generate_report()
        assert isinstance(report, PrivacyAuditReport)
        assert report.total_epsilon_spent == 0.0
        assert report.remaining_epsilon == 4.0
        assert report.total_events == 0
        assert report.risk_level == "low"

    def test_report_risk_levels(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        # 20% -> low
        pm.spend(epsilon=0.8)
        assert pm.generate_report().risk_level == "low"
        # 45% -> medium
        pm.spend(epsilon=1.0)
        assert pm.generate_report().risk_level == "medium"
        # 85% -> high
        pm.spend(epsilon=1.6)
        assert pm.generate_report().risk_level == "high"
        # Exhausted (spend remaining exactly to hit >= 0.9)
        pm.spend(epsilon=pm.remaining_epsilon)
        assert pm.generate_report().risk_level == "exhausted"

    def test_report_recommendations(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        # Low usage -> healthy
        pm.spend(epsilon=0.5)
        recs = pm.generate_report().recommendations
        assert any("healthy" in r for r in recs)

        # High usage -> warning
        pm2 = PrivacyBudgetManager(epsilon_budget=4.0)
        pm2.spend(epsilon=3.5)
        recs2 = pm2.generate_report().recommendations
        assert any("increasing" in r for r in recs2)

    def test_report_event_breakdown(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=10.0)
        pm.spend(epsilon=0.5, mechanism="gaussian")
        pm.spend(epsilon=0.5, mechanism="gaussian")
        pm.spend(epsilon=0.5, mechanism="laplace")
        report = pm.generate_report()
        assert report.event_breakdown["gaussian"] == 2
        assert report.event_breakdown["laplace"] == 1


class TestResetAndClone:
    def test_reset_clears_all(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        pm.spend(epsilon=1.0)
        pm.reset()
        assert pm.remaining_epsilon == 4.0
        assert pm.get_audit_trail() == []
        assert pm.budget_exhausted is False

    def test_clone_has_same_config_fresh_state(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0, delta=1e-6, min_noise_scale=0.05)
        pm.spend(epsilon=1.0)
        clone = pm.clone()
        assert clone.epsilon_budget == 4.0
        assert clone.delta == 1e-6
        assert clone.min_noise_scale == 0.05
        assert clone.remaining_epsilon == 4.0
        assert clone.get_audit_trail() == []


class TestMockIntegration:
    def test_mocked_numpy_clip(self) -> None:
        """Verify noise calibration with mocked numpy clip."""
        pm = PrivacyBudgetManager(epsilon_budget=4.0, min_noise_scale=2.0, max_noise_scale=2.0)
        with patch.object(np, "clip", wraps=np.clip) as mock_clip:
            sigma = pm.calibrate_noise(sensitivity=1.0, target_epsilon=1.0)
            mock_clip.assert_called_once()
            assert sigma == 2.0

    def test_mocked_spend_with_custom_timestamp(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=4.0)
        mock_time = 1234567890.0
        result = pm.spend(epsilon=0.5, timestamp=mock_time, description="mocked event")
        assert result["success"] is True
        trail = pm.get_audit_trail()
        assert trail[0].timestamp == mock_time
        assert trail[0].description == "mocked event"

    def test_mocked_auto_spend_with_magic_mock(self) -> None:
        pm = PrivacyBudgetManager(epsilon_budget=10.0)
        mock_calibrate = MagicMock(return_value=0.42)
        pm.calibrate_noise = mock_calibrate  # type: ignore[method-assign]
        result = pm.auto_spend_and_noise(sensitivity=2.0, mechanism="laplace")
        mock_calibrate.assert_called_once_with(2.0, pytest.approx(1.0), "laplace")
        assert result["noise_scale"] == 0.42
