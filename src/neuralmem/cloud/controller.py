"""NeuralMem Cloud — Hosting Controller (K8s operator pattern).

Manages tenant lifecycle, resource quotas, and deployment rollouts using a
CRD-style abstraction.  All Kubernetes interactions are optional and can be
mocked for unit testing.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from neuralmem.tenancy.models import TenantConfig

_logger = logging.getLogger(__name__)


class TenantPhase(Enum):
    """Lifecycle phases for a tenant deployment."""

    PENDING = auto()
    CREATING = auto()
    RUNNING = auto()
    UPDATING = auto()
    SCALING = auto()
    DELETING = auto()
    DELETED = auto()
    FAILED = auto()


class DeploymentStrategy(Enum):
    """Rollout strategy for tenant deployments."""

    ROLLING_UPDATE = "rolling_update"
    RECREATE = "recreate"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"


@dataclass
class ResourceQuota:
    """Resource limits for a tenant workload.

    Attributes
    ----------
    max_cpu_cores : float
    max_memory_mb : int
    max_replicas : int
    max_storage_gb : int
    """

    max_cpu_cores: float = 2.0
    max_memory_mb: int = 4096
    max_replicas: int = 3
    max_storage_gb: int = 10


@dataclass
class TenantCRD:
    """Custom Resource Definition representation for a hosted tenant.

    Attributes
    ----------
    tenant_id : str
    config : TenantConfig
    quota : ResourceQuota
    replicas : int
    image : str
        Container image to run.
    strategy : DeploymentStrategy
    labels : dict[str, str]
    annotations : dict[str, str]
    phase : TenantPhase
    generation : int
        Monotonically incremented on every spec change.
    observed_generation : int
        Generation last reconciled by the controller.
    """

    tenant_id: str
    config: TenantConfig = field(
        default_factory=lambda: TenantConfig(tenant_id="")
    )
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    replicas: int = 1
    image: str = "neuralmem:latest"
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    phase: TenantPhase = TenantPhase.PENDING
    generation: int = 1
    observed_generation: int = 0


@dataclass
class RolloutStatus:
    """Status of an in-flight rollout.

    Attributes
    ----------
    tenant_id : str
    target_generation : int
    updated_replicas : int
    ready_replicas : int
    unavailable_replicas : int
    start_time : float
    completion_time : float | None
    strategy : DeploymentStrategy
    """

    tenant_id: str
    target_generation: int
    updated_replicas: int = 0
    ready_replicas: int = 0
    unavailable_replicas: int = 0
    start_time: float = field(default_factory=time.time)
    completion_time: float | None = None
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE


# ------------------------------------------------------------------
# Pluggable K8s backend interface (mock-friendly)
# ------------------------------------------------------------------

class K8sBackend:
    """Abstract Kubernetes backend.

    Subclass or monkey-patch for tests.  The default implementation is a no-op
    in-memory simulation.
    """

    def create_deployment(
        self, crd: TenantCRD
    ) -> dict[str, Any]:
        """Create a deployment for *crd* and return its status."""
        return {"name": crd.tenant_id, "replicas": crd.replicas, "ready": True}

    def update_deployment(
        self, crd: TenantCRD
    ) -> dict[str, Any]:
        """Update an existing deployment to match *crd*."""
        return {"name": crd.tenant_id, "replicas": crd.replicas, "ready": True}

    def delete_deployment(self, tenant_id: str) -> dict[str, Any]:
        """Delete the deployment for *tenant_id*."""
        return {"name": tenant_id, "deleted": True}

    def scale_deployment(
        self, tenant_id: str, replicas: int
    ) -> dict[str, Any]:
        """Scale the deployment for *tenant_id* to *replicas*."""
        return {"name": tenant_id, "replicas": replicas, "ready": True}

    def get_deployment_status(
        self, tenant_id: str
    ) -> dict[str, Any]:
        """Return current deployment status."""
        return {"name": tenant_id, "replicas": 1, "ready": True}

    def list_pods(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return pods belonging to *tenant_id*."""
        return []

    def create_service(self, tenant_id: str, port: int) -> dict[str, Any]:
        """Create a K8s Service for *tenant_id*."""
        return {"name": tenant_id, "port": port, "cluster_ip": "10.0.0.1"}

    def delete_service(self, tenant_id: str) -> dict[str, Any]:
        """Delete the Service for *tenant_id*."""
        return {"name": tenant_id, "deleted": True}


class HostingController:
    """K8s-style operator for NeuralMem tenant hosting.

    Manages the full tenant lifecycle (CRD → reconcile → deployment) and
    supports zero-downtime rolling updates via pluggable strategies.

    Parameters
    ----------
    backend : K8sBackend | None
        Kubernetes backend to use.  Defaults to an in-memory no-op.
    reconcile_interval : float
        Seconds between automatic reconcile loops (if running).
    """

    def __init__(
        self,
        backend: K8sBackend | None = None,
        reconcile_interval: float = 30.0,
    ) -> None:
        self._backend = backend or K8sBackend()
        self._reconcile_interval = reconcile_interval
        self._crds: dict[str, TenantCRD] = {}
        self._rollouts: dict[str, RolloutStatus] = {}
        self._lock = threading.Lock()
        self._running = False
        self._reconcile_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Hooks for testing / observability
        self._pre_reconcile_hook: Callable[[TenantCRD], None] | None = None
        self._post_reconcile_hook: Callable[[TenantCRD, bool], None] | None = None

    # ------------------------------------------------------------------
    # CRD lifecycle
    # ------------------------------------------------------------------

    def create_tenant(
        self,
        tenant_id: str,
        config: TenantConfig | None = None,
        quota: ResourceQuota | None = None,
        replicas: int = 1,
        image: str = "neuralmem:latest",
        strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> TenantCRD:
        """Create a new tenant CRD and trigger initial reconciliation.

        Raises ``ValueError`` if the tenant already exists.
        """
        with self._lock:
            if tenant_id in self._crds:
                raise ValueError(f"Tenant '{tenant_id}' already exists")
            crd = TenantCRD(
                tenant_id=tenant_id,
                config=config or TenantConfig(tenant_id=tenant_id),
                quota=quota or ResourceQuota(),
                replicas=replicas,
                image=image,
                strategy=strategy,
                labels=labels or {},
                annotations=annotations or {},
                phase=TenantPhase.PENDING,
                generation=1,
                observed_generation=0,
            )
            self._crds[tenant_id] = crd
            _logger.info("Created tenant CRD: %s", tenant_id)

        self._reconcile_single(crd)
        return crd

    def delete_tenant(self, tenant_id: str) -> None:
        """Delete a tenant and its underlying resources.

        Raises ``KeyError`` if the tenant does not exist.
        """
        with self._lock:
            if tenant_id not in self._crds:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            crd = self._crds[tenant_id]
            crd.phase = TenantPhase.DELETING

        # Tear down K8s resources outside the lock
        self._backend.delete_service(tenant_id)
        self._backend.delete_deployment(tenant_id)

        with self._lock:
            crd.phase = TenantPhase.DELETED
            self._rollouts.pop(tenant_id, None)
            del self._crds[tenant_id]
            _logger.info("Deleted tenant: %s", tenant_id)

    def get_tenant(self, tenant_id: str) -> TenantCRD:
        """Return the CRD for *tenant_id*.

        Raises ``KeyError`` if the tenant does not exist.
        """
        with self._lock:
            if tenant_id not in self._crds:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            return self._crds[tenant_id]

    def list_tenants(self) -> list[TenantCRD]:
        """Return all active tenant CRDs."""
        with self._lock:
            return list(self._crds.values())

    def update_tenant_spec(
        self,
        tenant_id: str,
        replicas: int | None = None,
        image: str | None = None,
        quota: ResourceQuota | None = None,
        labels: dict[str, str] | None = None,
    ) -> TenantCRD:
        """Mutate a tenant's spec and bump its generation.

        Reconciliation is triggered automatically.
        """
        with self._lock:
            if tenant_id not in self._crds:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            crd = self._crds[tenant_id]
            if replicas is not None:
                crd.replicas = max(1, min(replicas, crd.quota.max_replicas))
            if image is not None:
                crd.image = image
            if quota is not None:
                crd.quota = quota
            if labels is not None:
                crd.labels.update(labels)
            crd.generation += 1
            crd.phase = TenantPhase.UPDATING
            _logger.info("Updated tenant spec: %s (gen=%d)", tenant_id, crd.generation)

        self._reconcile_single(crd)
        return crd

    # ------------------------------------------------------------------
    # Reconciliation (operator pattern)
    # ------------------------------------------------------------------

    def reconcile(self) -> None:
        """Reconcile all tenants whose ``observed_generation < generation``."""
        with self._lock:
            pending = [
                crd for crd in self._crds.values()
                if crd.observed_generation < crd.generation
                or crd.phase in (TenantPhase.PENDING, TenantPhase.FAILED)
            ]
        for crd in pending:
            self._reconcile_single(crd)

    def _reconcile_single(self, crd: TenantCRD) -> bool:
        """Reconcile a single tenant CRD.

        Returns ``True`` if reconciliation succeeded.
        """
        if self._pre_reconcile_hook:
            self._pre_reconcile_hook(crd)

        try:
            if crd.phase == TenantPhase.PENDING:
                # Initial creation
                self._backend.create_service(crd.tenant_id, port=8080)
                self._backend.create_deployment(crd)
                crd.phase = TenantPhase.RUNNING

            elif crd.phase == TenantPhase.UPDATING:
                # Rolling update
                self._start_rollout(crd)
                status = self._backend.update_deployment(crd)
                if status.get("ready"):
                    self._finish_rollout(crd)
                    crd.phase = TenantPhase.RUNNING
                else:
                    crd.phase = TenantPhase.FAILED

            elif crd.phase == TenantPhase.SCALING:
                self._backend.scale_deployment(crd.tenant_id, crd.replicas)
                crd.phase = TenantPhase.RUNNING

            elif crd.phase == TenantPhase.FAILED:
                # Retry once
                self._backend.update_deployment(crd)
                crd.phase = TenantPhase.RUNNING

            crd.observed_generation = crd.generation
            success = True
        except Exception as exc:
            _logger.error("Reconcile failed for %s: %s", crd.tenant_id, exc)
            crd.phase = TenantPhase.FAILED
            success = False

        if self._post_reconcile_hook:
            self._post_reconcile_hook(crd, success)
        return success

    # ------------------------------------------------------------------
    # Rollout / zero-downtime updates
    # ------------------------------------------------------------------

    def _start_rollout(self, crd: TenantCRD) -> RolloutStatus:
        """Initialise a :class:`RolloutStatus` for *crd*."""
        status = RolloutStatus(
            tenant_id=crd.tenant_id,
            target_generation=crd.generation,
            strategy=crd.strategy,
        )
        with self._lock:
            self._rollouts[crd.tenant_id] = status
        _logger.info("Rollout started for %s (gen=%d)", crd.tenant_id, crd.generation)
        return status

    def _finish_rollout(self, crd: TenantCRD) -> None:
        """Mark the current rollout for *crd* as complete."""
        with self._lock:
            status = self._rollouts.get(crd.tenant_id)
            if status is not None:
                status.completion_time = time.time()
                status.ready_replicas = crd.replicas
                status.updated_replicas = crd.replicas
                status.unavailable_replicas = 0
        _logger.info("Rollout finished for %s", crd.tenant_id)

    def get_rollout_status(self, tenant_id: str) -> RolloutStatus | None:
        """Return the current rollout status for *tenant_id*."""
        with self._lock:
            return self._rollouts.get(tenant_id)

    def rollout_progress(self, tenant_id: str) -> dict[str, Any]:
        """Return human-readable rollout progress for *tenant_id*."""
        status = self.get_rollout_status(tenant_id)
        if status is None:
            return {"status": "none"}
        total = status.updated_replicas + status.unavailable_replicas
        pct = (
            (status.ready_replicas / total * 100)
            if total > 0 else 0
        )
        return {
            "tenant_id": status.tenant_id,
            "target_generation": status.target_generation,
            "updated_replicas": status.updated_replicas,
            "ready_replicas": status.ready_replicas,
            "unavailable_replicas": status.unavailable_replicas,
            "percent_ready": round(pct, 1),
            "duration_sec": (
                (status.completion_time or time.time()) - status.start_time
            ),
            "strategy": status.strategy.value,
        }

    # ------------------------------------------------------------------
    # Resource quotas
    # ------------------------------------------------------------------

    def set_quota(self, tenant_id: str, quota: ResourceQuota) -> TenantCRD:
        """Apply a new resource quota to *tenant_id*.

        Reconciles automatically.
        """
        return self.update_tenant_spec(tenant_id, quota=quota)

    def check_quota(self, tenant_id: str, desired_replicas: int) -> tuple[bool, str]:
        """Check whether *desired_replicas* is within the tenant's quota.

        Returns ``(allowed, reason)``.
        """
        with self._lock:
            if tenant_id not in self._crds:
                return False, f"Tenant '{tenant_id}' not found"
            crd = self._crds[tenant_id]
        if desired_replicas > crd.quota.max_replicas:
            return (
                False,
                f"Desired replicas {desired_replicas} exceeds max {crd.quota.max_replicas}",
            )
        return True, ""

    # ------------------------------------------------------------------
    # Background reconcile loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background reconcile loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._reconcile_thread = threading.Thread(
            target=self._reconcile_loop, daemon=True
        )
        self._reconcile_thread.start()
        _logger.info("HostingController started")

    def stop(self) -> None:
        """Stop the background reconcile loop."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._reconcile_thread is not None:
            self._reconcile_thread.join(timeout=5.0)
        _logger.info("HostingController stopped")

    def _reconcile_loop(self) -> None:
        while not self._stop_event.is_set():
            self.reconcile()
            self._stop_event.wait(timeout=self._reconcile_interval)

    # ------------------------------------------------------------------
    # Observability hooks
    # ------------------------------------------------------------------

    def set_pre_reconcile_hook(
        self, hook: Callable[[TenantCRD], None] | None
    ) -> None:
        """Set a hook called before each single-tenant reconcile."""
        self._pre_reconcile_hook = hook

    def set_post_reconcile_hook(
        self, hook: Callable[[TenantCRD, bool], None] | None
    ) -> None:
        """Set a hook called after each single-tenant reconcile."""
        self._post_reconcile_hook = hook

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state and stop the background loop."""
        self.stop()
        with self._lock:
            self._crds.clear()
            self._rollouts.clear()
        _logger.info("HostingController state reset")
