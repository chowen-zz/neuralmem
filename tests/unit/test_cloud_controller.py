"""Unit tests for neuralmem.cloud.controller — HostingController."""

from unittest.mock import MagicMock

import pytest

from neuralmem.cloud.controller import (
    DeploymentStrategy,
    HostingController,
    K8sBackend,
    ResourceQuota,
    RolloutStatus,
    TenantCRD,
    TenantPhase,
)
from neuralmem.tenancy.models import TenantConfig


@pytest.fixture
def controller():
    return HostingController()


@pytest.fixture
def controller_with_mock_backend():
    backend = MagicMock(spec=K8sBackend)
    backend.create_deployment.return_value = {"ready": True}
    backend.update_deployment.return_value = {"ready": True}
    backend.delete_deployment.return_value = {"deleted": True}
    backend.scale_deployment.return_value = {"ready": True}
    backend.create_service.return_value = {"cluster_ip": "10.0.0.1"}
    backend.delete_service.return_value = {"deleted": True}
    return HostingController(backend=backend), backend


class TestCreateTenant:
    def test_create_tenant_basic(self, controller):
        crd = controller.create_tenant("acme")
        assert crd.tenant_id == "acme"
        assert crd.phase == TenantPhase.RUNNING
        assert crd.generation == 1
        assert crd.observed_generation == 1

    def test_create_tenant_with_config(self, controller):
        cfg = TenantConfig(tenant_id="acme", max_memories=500)
        crd = controller.create_tenant("acme", config=cfg)
        assert crd.config.max_memories == 500

    def test_create_tenant_with_quota(self, controller):
        quota = ResourceQuota(max_replicas=5)
        crd = controller.create_tenant("acme", quota=quota)
        assert crd.quota.max_replicas == 5

    def test_create_duplicate_raises(self, controller):
        controller.create_tenant("acme")
        with pytest.raises(ValueError, match="already exists"):
            controller.create_tenant("acme")

    def test_create_triggers_reconcile(self, controller_with_mock_backend):
        ctrl, backend = controller_with_mock_backend
        ctrl.create_tenant("acme")
        backend.create_deployment.assert_called_once()
        backend.create_service.assert_called_once()


class TestDeleteTenant:
    def test_delete_existing(self, controller):
        controller.create_tenant("acme")
        controller.delete_tenant("acme")
        with pytest.raises(KeyError):
            controller.get_tenant("acme")

    def test_delete_missing_raises(self, controller):
        with pytest.raises(KeyError, match="not found"):
            controller.delete_tenant("missing")

    def test_delete_calls_backend(self, controller_with_mock_backend):
        ctrl, backend = controller_with_mock_backend
        ctrl.create_tenant("acme")
        ctrl.delete_tenant("acme")
        backend.delete_deployment.assert_called_once_with("acme")
        backend.delete_service.assert_called_once_with("acme")


class TestGetAndList:
    def test_get_tenant(self, controller):
        controller.create_tenant("acme")
        crd = controller.get_tenant("acme")
        assert isinstance(crd, TenantCRD)

    def test_get_missing_raises(self, controller):
        with pytest.raises(KeyError, match="not found"):
            controller.get_tenant("missing")

    def test_list_tenants(self, controller):
        controller.create_tenant("a")
        controller.create_tenant("b")
        assert len(controller.list_tenants()) == 2


class TestUpdateSpec:
    def test_update_replicas(self, controller):
        crd = controller.create_tenant("acme", replicas=1)
        updated = controller.update_tenant_spec("acme", replicas=3)
        assert updated.replicas == 3
        assert updated.generation == 2
        assert updated.phase == TenantPhase.RUNNING

    def test_update_image(self, controller):
        crd = controller.create_tenant("acme", image="v1")
        updated = controller.update_tenant_spec("acme", image="v2")
        assert updated.image == "v2"

    def test_update_labels(self, controller):
        controller.create_tenant("acme")
        updated = controller.update_tenant_spec("acme", labels={"env": "prod"})
        assert updated.labels["env"] == "prod"

    def test_update_respects_max_replicas(self, controller):
        controller.create_tenant("acme", quota=ResourceQuota(max_replicas=3), replicas=1)
        updated = controller.update_tenant_spec("acme", replicas=10)
        assert updated.replicas == 3

    def test_update_missing_raises(self, controller):
        with pytest.raises(KeyError, match="not found"):
            controller.update_tenant_spec("missing", replicas=2)


class TestReconcile:
    def test_reconcile_pending_tenant(self, controller_with_mock_backend):
        ctrl, backend = controller_with_mock_backend
        crd = ctrl.create_tenant("acme")
        # Force back to pending to test reconcile
        crd.phase = TenantPhase.PENDING
        crd.observed_generation = 0
        ctrl.reconcile()
        assert crd.phase == TenantPhase.RUNNING
        assert crd.observed_generation == crd.generation

    def test_reconcile_failed_to_running(self, controller_with_mock_backend):
        ctrl, backend = controller_with_mock_backend
        crd = ctrl.create_tenant("acme")
        crd.phase = TenantPhase.FAILED
        crd.observed_generation = 0
        ctrl.reconcile()
        assert crd.phase == TenantPhase.RUNNING

    def test_reconcile_noop_when_current(self, controller):
        controller.create_tenant("acme")
        # Should not raise and phase stays RUNNING
        controller.reconcile()
        assert controller.get_tenant("acme").phase == TenantPhase.RUNNING

    def test_pre_post_hooks(self, controller_with_mock_backend):
        ctrl, _ = controller_with_mock_backend
        pre = MagicMock()
        post = MagicMock()
        ctrl.set_pre_reconcile_hook(pre)
        ctrl.set_post_reconcile_hook(post)
        ctrl.create_tenant("acme")
        pre.assert_called_once()
        post.assert_called_once()
        assert post.call_args[0][1] is True  # success=True


class TestRollout:
    def test_rollout_created_on_update(self, controller_with_mock_backend):
        ctrl, backend = controller_with_mock_backend
        ctrl.create_tenant("acme", replicas=1)
        ctrl.update_tenant_spec("acme", image="v2")
        status = ctrl.get_rollout_status("acme")
        assert isinstance(status, RolloutStatus)
        assert status.target_generation == 2

    def test_rollout_progress(self, controller_with_mock_backend):
        ctrl, _ = controller_with_mock_backend
        ctrl.create_tenant("acme", replicas=2)
        ctrl.update_tenant_spec("acme", image="v2")
        progress = ctrl.rollout_progress("acme")
        assert progress["tenant_id"] == "acme"
        assert progress["target_generation"] == 2
        assert "percent_ready" in progress
        assert "duration_sec" in progress

    def test_rollout_status_none(self, controller):
        assert controller.get_rollout_status("none") is None

    def test_blue_green_strategy(self, controller_with_mock_backend):
        ctrl, _ = controller_with_mock_backend
        crd = ctrl.create_tenant("acme", strategy=DeploymentStrategy.BLUE_GREEN)
        assert crd.strategy == DeploymentStrategy.BLUE_GREEN


class TestQuotas:
    def test_set_quota(self, controller):
        controller.create_tenant("acme")
        new_quota = ResourceQuota(max_cpu_cores=4.0, max_memory_mb=8192)
        crd = controller.set_quota("acme", new_quota)
        assert crd.quota.max_cpu_cores == 4.0
        assert crd.quota.max_memory_mb == 8192

    def test_check_quota_allowed(self, controller):
        controller.create_tenant("acme", quota=ResourceQuota(max_replicas=5))
        allowed, reason = controller.check_quota("acme", desired_replicas=3)
        assert allowed is True
        assert reason == ""

    def test_check_quota_denied(self, controller):
        controller.create_tenant("acme", quota=ResourceQuota(max_replicas=2))
        allowed, reason = controller.check_quota("acme", desired_replicas=5)
        assert allowed is False
        assert "exceeds" in reason

    def test_check_quota_missing(self, controller):
        allowed, reason = controller.check_quota("missing", desired_replicas=1)
        assert allowed is False
        assert "not found" in reason


class TestBackgroundLoop:
    def test_start_stop(self, controller):
        controller.start()
        assert controller._running is True
        controller.stop()
        assert controller._running is False

    def test_stop_idempotent(self, controller):
        controller.stop()
        controller.stop()
        assert controller._running is False

    def test_reconcile_in_loop(self, controller_with_mock_backend):
        ctrl, backend = controller_with_mock_backend
        ctrl.create_tenant("acme")
        backend.create_deployment.reset_mock()
        # Manually tick reconcile
        ctrl.reconcile()
        # Already reconciled — should not call create again
        backend.create_deployment.assert_not_called()


class TestReset:
    def test_reset_clears_state(self, controller):
        controller.create_tenant("a")
        controller.create_tenant("b")
        controller.reset()
        assert controller.list_tenants() == []
        assert controller._running is False
