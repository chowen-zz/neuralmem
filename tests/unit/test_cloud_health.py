"""Unit tests for neuralmem.cloud.health — DistributedHealthChecker."""

from unittest.mock import MagicMock

import pytest

from neuralmem.cloud.health import (
    DeploymentBackend,
    DistributedHealthChecker,
    FailoverAction,
    FailoverBackend,
    HealthCheckResult,
    HealthProbeBackend,
    HealthStatus,
    InstanceInfo,
)


@pytest.fixture
def checker():
    return DistributedHealthChecker()


@pytest.fixture
def checker_with_mocks():
    probe = MagicMock(spec=HealthProbeBackend)
    failover = MagicMock(spec=FailoverBackend)
    deploy = MagicMock(spec=DeploymentBackend)
    return DistributedHealthChecker(
        probe_backend=probe,
        failover_backend=failover,
        deployment_backend=deploy,
    ), probe, failover, deploy


class TestRegisterInstance:
    def test_register_and_list(self, checker):
        inst = InstanceInfo("i-1", "host-1", 8080)
        checker.register_instance(inst)
        assert len(checker.list_instances()) == 1
        assert checker.get_instance("i-1").host == "host-1"

    def test_unregister(self, checker):
        checker.register_instance(InstanceInfo("i-1"))
        checker.unregister_instance("i-1")
        assert checker.list_instances() == []
        assert checker.get_instance("i-1") is None

    def test_update_existing(self, checker):
        checker.register_instance(InstanceInfo("i-1", "host-a", 8080))
        checker.register_instance(InstanceInfo("i-1", "host-b", 9090))
        assert checker.get_instance("i-1").port == 9090


class TestHealthChecks:
    def test_check_healthy(self, checker_with_mocks):
        chk, probe, _, _ = checker_with_mocks
        chk.register_instance(InstanceInfo("i-1"))
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.HEALTHY, latency_ms=12.0
        )
        result = chk.check_instance("i-1")
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 12.0
        probe.probe.assert_called_once()

    def test_check_unregistered(self, checker):
        result = checker.check_instance("missing")
        assert result.status == HealthStatus.UNKNOWN

    def test_check_all(self, checker_with_mocks):
        chk, probe, _, _ = checker_with_mocks
        chk.register_instance(InstanceInfo("i-1"))
        chk.register_instance(InstanceInfo("i-2"))
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.HEALTHY
        )
        results = chk.check_all()
        assert len(results) == 2
        assert results["i-1"].status == HealthStatus.HEALTHY

    def test_get_health(self, checker_with_mocks):
        chk, probe, _, _ = checker_with_mocks
        chk.register_instance(InstanceInfo("i-1"))
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.DEGRADED
        )
        chk.check_instance("i-1")
        assert chk.get_health("i-1").status == HealthStatus.DEGRADED

    def test_history_trimming(self, checker_with_mocks):
        chk, probe, _, _ = checker_with_mocks
        chk.register_instance(InstanceInfo("i-1"))
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.HEALTHY
        )
        for _ in range(25):
            chk.check_instance("i-1")
        assert len(chk.get_health_history("i-1")) == 20


class TestFailover:
    def test_no_action_when_healthy(self, checker_with_mocks):
        chk, probe, failover, _ = checker_with_mocks
        chk.register_instance(InstanceInfo("i-1"))
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.HEALTHY
        )
        chk.check_instance("i-1")
        failover.drain_instance.assert_not_called()

    def test_drain_when_unhealthy_with_peer(self, checker_with_mocks):
        chk, probe, failover, _ = checker_with_mocks
        chk.register_instance(InstanceInfo("i-1", weight=100, is_active=True))
        chk.register_instance(InstanceInfo("i-2", weight=100, is_active=True))
        # i-1 healthy first so it is considered a peer
        probe.probe.side_effect = lambda inst: HealthCheckResult(
            instance_id=inst.instance_id,
            status=HealthStatus.HEALTHY if inst.instance_id == "i-2" else HealthStatus.UNHEALTHY,
        )
        # Seed healthy history for i-2
        chk.check_instance("i-2")
        # Fail i-1 repeatedly
        for _ in range(3):
            chk.check_instance("i-1")
        failover.drain_instance.assert_called_with("i-1")
        failover.redirect_traffic.assert_called()
        assert chk.get_instance("i-1").is_active is False
        assert chk.get_instance("i-1").is_draining is True

    def test_scale_up_when_no_healthy_peer(self, checker_with_mocks):
        chk, probe, failover, _ = checker_with_mocks
        chk.register_instance(InstanceInfo("i-1"))
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.UNHEALTHY
        )
        for _ in range(3):
            chk.check_instance("i-1")
        failover.scale_up.assert_called_once()

    def test_recommended_action(self, checker_with_mocks):
        chk, probe, _, _ = checker_with_mocks
        chk.register_instance(InstanceInfo("i-1"))
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.DEGRADED
        )
        chk.check_instance("i-1")
        assert chk.get_recommended_action("i-1") == FailoverAction.RESTART

    def test_recommended_action_unhealthy_no_peer(self, checker):
        checker.register_instance(InstanceInfo("i-1"))
        # No health data yet
        assert checker.get_recommended_action("i-1") == FailoverAction.NONE


class TestZeroDowntimeDeployment:
    def test_start_deployment(self, checker_with_mocks):
        chk, _, _, deploy = checker_with_mocks
        old = InstanceInfo("old-1", is_active=True)
        new = InstanceInfo("new-1")
        chk.register_instance(old)
        chk.start_zero_downtime_deployment("dep-1", [new])
        deploy.start_deployment.assert_called_once()
        assert chk.get_instance("new-1") is not None
        assert chk.get_instance("old-1").is_draining is True

    def test_validate_deployment_all_healthy(self, checker_with_mocks):
        chk, probe, _, _ = checker_with_mocks
        new = InstanceInfo("new-1")
        chk.register_instance(new)
        probe.probe.return_value = HealthCheckResult(
            instance_id="new-1", status=HealthStatus.HEALTHY
        )
        chk.start_zero_downtime_deployment("dep-1", [new])
        chk.check_instance("new-1")
        result = chk.validate_deployment("dep-1")
        assert result["valid"] is True
        assert "HEALTHY" in result["details"][0]

    def test_validate_deployment_unhealthy(self, checker_with_mocks):
        chk, probe, _, _ = checker_with_mocks
        new = InstanceInfo("new-1")
        chk.register_instance(new)
        probe.probe.return_value = HealthCheckResult(
            instance_id="new-1", status=HealthStatus.UNHEALTHY
        )
        chk.start_zero_downtime_deployment("dep-1", [new])
        chk.check_instance("new-1")
        result = chk.validate_deployment("dep-1")
        assert result["valid"] is False

    def test_commit_deployment(self, checker_with_mocks):
        chk, _, _, deploy = checker_with_mocks
        old = InstanceInfo("old-1")
        new = InstanceInfo("new-1")
        chk.register_instance(old)
        chk.start_zero_downtime_deployment("dep-1", [new])
        result = chk.commit_deployment("dep-1")
        deploy.finish_deployment.assert_called_once_with("dep-1")
        assert chk.get_instance("old-1") is None
        assert chk.get_instance("new-1") is not None

    def test_rollback_deployment(self, checker_with_mocks):
        chk, _, _, deploy = checker_with_mocks
        old = InstanceInfo("old-1")
        new = InstanceInfo("new-1")
        chk.register_instance(old)
        chk.start_zero_downtime_deployment("dep-1", [new])
        result = chk.rollback_deployment("dep-1")
        deploy.rollback_deployment.assert_called_once_with("dep-1")
        assert chk.get_instance("new-1") is None
        assert chk.get_instance("old-1").is_active is True
        assert chk.get_instance("old-1").is_draining is False

    def test_get_deployment_status(self, checker_with_mocks):
        chk, _, _, _ = checker_with_mocks
        new = InstanceInfo("new-1")
        chk.register_instance(new)
        chk.start_zero_downtime_deployment("dep-1", [new])
        status = chk.get_deployment_status("dep-1")
        assert status["active"] is True
        assert "new_instances" in status

    def test_get_deployment_status_not_found(self, checker_with_mocks):
        chk, _, _, deploy = checker_with_mocks
        deploy.get_deployment_status.return_value = {"status": "unknown"}
        status = chk.get_deployment_status("dep-missing")
        assert status["status"] == "unknown"


class TestBackgroundLoop:
    def test_start_stop(self, checker):
        checker.start()
        assert checker._running is True
        checker.stop()
        assert checker._running is False

    def test_stop_idempotent(self, checker):
        checker.stop()
        checker.stop()
        assert checker._running is False


class TestHooks:
    def test_status_change_hook(self, checker_with_mocks):
        chk, probe, _, _ = checker_with_mocks
        hook = MagicMock()
        chk.set_status_change_hook(hook)
        chk.register_instance(InstanceInfo("i-1"))
        # First check establishes baseline (no prior status)
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.HEALTHY
        )
        chk.check_instance("i-1")
        hook.assert_not_called()  # first check has no prior status
        # Second check changes from HEALTHY -> DEGRADED
        probe.probe.return_value = HealthCheckResult(
            instance_id="i-1", status=HealthStatus.DEGRADED
        )
        chk.check_instance("i-1")
        hook.assert_called_once_with("i-1", HealthStatus.HEALTHY, HealthStatus.DEGRADED)


class TestReset:
    def test_reset_clears_all(self, checker):
        checker.register_instance(InstanceInfo("i-1"))
        checker.start()
        checker.reset()
        assert checker.list_instances() == []
        assert checker._running is False
        assert checker.get_health("i-1") is None
