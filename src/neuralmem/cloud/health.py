"""NeuralMem Cloud — Distributed Health Checker.

Performs health checks, failover, and zero-downtime deployment coordination.
All infrastructure interactions are abstracted behind pluggable backends so
unit tests can run with pure mocks.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

_logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Discrete health states for a service instance."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


class FailoverAction(Enum):
    """Actions the health checker can recommend."""

    NONE = auto()
    RESTART = auto()
    REDIRECT = auto()
    SCALE_UP = auto()
    DRAIN = auto()


@dataclass
class HealthCheckResult:
    """Result of a single health probe.

    Attributes
    ----------
    instance_id : str
    status : HealthStatus
    latency_ms : float
    timestamp : float
    message : str
    metadata : dict
    """

    instance_id: str
    status: HealthStatus
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class InstanceInfo:
    """Metadata for a managed service instance.

    Attributes
    ----------
    instance_id : str
    host : str
    port : int
    weight : int
        Load-balancer weight (0 = drained).
    is_active : bool
        Whether the instance is receiving traffic.
    is_draining : bool
        Whether the instance is in the process of being removed.
    """

    instance_id: str
    host: str = "localhost"
    port: int = 8080
    weight: int = 100
    is_active: bool = True
    is_draining: bool = False


# ------------------------------------------------------------------
# Pluggable backends
# ------------------------------------------------------------------

class HealthProbeBackend:
    """Abstract health probe backend.

    Override ``probe`` in tests to return synthetic results.
    """

    def probe(self, instance: InstanceInfo) -> HealthCheckResult:
        """Probe *instance* and return a :class:`HealthCheckResult`."""
        start = time.time()
        # Default no-op: assume healthy
        latency = (time.time() - start) * 1000
        return HealthCheckResult(
            instance_id=instance.instance_id,
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="default probe ok",
        )


class FailoverBackend:
    """Abstract failover backend.

    Override methods in tests to simulate failover actions.
    """

    def restart_instance(self, instance_id: str) -> dict[str, Any]:
        _logger.info("FailoverBackend: restarting %s", instance_id)
        return {"instance_id": instance_id, "action": "restart", "ok": True}

    def redirect_traffic(
        self, from_instance: str, to_instance: str
    ) -> dict[str, Any]:
        _logger.info(
            "FailoverBackend: redirect %s -> %s", from_instance, to_instance
        )
        return {
            "from": from_instance,
            "to": to_instance,
            "action": "redirect",
            "ok": True,
        }

    def drain_instance(self, instance_id: str) -> dict[str, Any]:
        _logger.info("FailoverBackend: draining %s", instance_id)
        return {"instance_id": instance_id, "action": "drain", "ok": True}

    def scale_up(self, count: int = 1) -> dict[str, Any]:
        _logger.info("FailoverBackend: scale up %d", count)
        return {"count": count, "action": "scale_up", "ok": True}


class DeploymentBackend:
    """Abstract deployment backend for zero-downtime coordination.

    Override methods in tests to simulate deployment state machines.
    """

    def start_deployment(
        self, deployment_id: str, instances: list[InstanceInfo]
    ) -> dict[str, Any]:
        _logger.info("DeploymentBackend: start %s", deployment_id)
        return {"deployment_id": deployment_id, "status": "started"}

    def update_instance(
        self, deployment_id: str, instance: InstanceInfo
    ) -> dict[str, Any]:
        return {
            "deployment_id": deployment_id,
            "instance_id": instance.instance_id,
            "status": "updated",
        }

    def finish_deployment(self, deployment_id: str) -> dict[str, Any]:
        _logger.info("DeploymentBackend: finish %s", deployment_id)
        return {"deployment_id": deployment_id, "status": "finished"}

    def rollback_deployment(self, deployment_id: str) -> dict[str, Any]:
        _logger.info("DeploymentBackend: rollback %s", deployment_id)
        return {"deployment_id": deployment_id, "status": "rolled_back"}

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        return {"deployment_id": deployment_id, "status": "unknown"}


# ------------------------------------------------------------------
# Distributed Health Checker
# ------------------------------------------------------------------

class DistributedHealthChecker:
    """Distributed health checker with failover and zero-downtime deployment.

    Parameters
    ----------
    probe_backend : HealthProbeBackend | None
    failover_backend : FailoverBackend | None
    deployment_backend : DeploymentBackend | None
    check_interval : float
        Seconds between automatic health check sweeps.
    failure_threshold : int
        Consecutive failures before an instance is marked UNHEALTHY.
    latency_threshold_ms : float
        Latency above which an instance is considered DEGRADED.
    """

    def __init__(
        self,
        probe_backend: HealthProbeBackend | None = None,
        failover_backend: FailoverBackend | None = None,
        deployment_backend: DeploymentBackend | None = None,
        check_interval: float = 5.0,
        failure_threshold: int = 3,
        latency_threshold_ms: float = 500.0,
    ) -> None:
        self._probe = probe_backend or HealthProbeBackend()
        self._failover = failover_backend or FailoverBackend()
        self._deploy = deployment_backend or DeploymentBackend()
        self._check_interval = check_interval
        self._failure_threshold = failure_threshold
        self._latency_threshold_ms = latency_threshold_ms
        self._instances: dict[str, InstanceInfo] = {}
        self._health_history: dict[str, list[HealthCheckResult]] = {}
        self._current_health: dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()
        self._running = False
        self._check_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Deployment tracking
        self._active_deployments: dict[str, dict[str, Any]] = {}
        # Hooks
        self._on_status_change: Callable[[str, HealthStatus, HealthStatus], None] | None = None

    # ------------------------------------------------------------------
    # Instance registry
    # ------------------------------------------------------------------

    def register_instance(self, info: InstanceInfo) -> None:
        """Register (or update) an instance to be monitored."""
        with self._lock:
            self._instances[info.instance_id] = info
            if info.instance_id not in self._health_history:
                self._health_history[info.instance_id] = []
        _logger.info("Registered instance %s", info.instance_id)

    def unregister_instance(self, instance_id: str) -> None:
        """Stop monitoring *instance_id* and discard its history."""
        with self._lock:
            self._instances.pop(instance_id, None)
            self._health_history.pop(instance_id, None)
            self._current_health.pop(instance_id, None)
        _logger.info("Unregistered instance %s", instance_id)

    def list_instances(self) -> list[InstanceInfo]:
        """Return all registered instances."""
        with self._lock:
            return list(self._instances.values())

    def get_instance(self, instance_id: str) -> InstanceInfo | None:
        """Return the :class:`InstanceInfo` for *instance_id*."""
        with self._lock:
            return self._instances.get(instance_id)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def check_instance(self, instance_id: str) -> HealthCheckResult:
        """Run a single health probe against *instance_id*.

        Stores the result in history and triggers failover logic if needed.
        """
        with self._lock:
            instance = self._instances.get(instance_id)
        if instance is None:
            return HealthCheckResult(
                instance_id=instance_id,
                status=HealthStatus.UNKNOWN,
                message="Instance not registered",
            )

        result = self._probe.probe(instance)
        # Detect status change before overwriting current health
        with self._lock:
            old = self._current_health.get(instance_id)
            if old is not None and old.status != result.status:
                if self._on_status_change is not None:
                    self._on_status_change(instance_id, old.status, result.status)
            self._current_health[instance_id] = result
            history = self._health_history.setdefault(instance_id, [])
            history.append(result)
            # Keep last 20 results
            if len(history) > 20:
                history.pop(0)

        self._evaluate_failover(instance_id, result)
        return result

    def check_all(self) -> dict[str, HealthCheckResult]:
        """Run health probes for every registered instance."""
        with self._lock:
            ids = list(self._instances.keys())
        return {iid: self.check_instance(iid) for iid in ids}

    def get_health(self, instance_id: str) -> HealthCheckResult | None:
        """Return the most recent health result for *instance_id*."""
        with self._lock:
            return self._current_health.get(instance_id)

    def get_health_history(
        self, instance_id: str
    ) -> list[HealthCheckResult]:
        """Return the stored health history for *instance_id*."""
        with self._lock:
            return list(self._health_history.get(instance_id, []))

    # ------------------------------------------------------------------
    # Failover logic
    # ------------------------------------------------------------------

    def _evaluate_failover(
        self, instance_id: str, result: HealthCheckResult
    ) -> FailoverAction:
        """Decide whether to trigger failover for *instance_id*.

        Returns the action taken.
        """
        # Detect status changes and fire hook
        old_status = None
        with self._lock:
            old = self._current_health.get(instance_id)
        if old is not None:
            old_status = old.status
        if old_status is not None and old_status != result.status:
            if self._on_status_change is not None:
                self._on_status_change(instance_id, old_status, result.status)

        if result.status == HealthStatus.HEALTHY:
            return FailoverAction.NONE

        # Count consecutive failures
        with self._lock:
            history = self._health_history.get(instance_id, [])
        consecutive = 0
        for h in reversed(history):
            if h.status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN):
                consecutive += 1
            else:
                break

        if consecutive >= self._failure_threshold:
            # Trigger drain + redirect if possible
            healthy_peers = [
                i for i in self.list_instances()
                if i.instance_id != instance_id and self._is_healthy(i.instance_id)
            ]
            if healthy_peers:
                target = healthy_peers[0]
                self._failover.drain_instance(instance_id)
                self._failover.redirect_traffic(instance_id, target.instance_id)
                with self._lock:
                    inst = self._instances.get(instance_id)
                    if inst is not None:
                        inst.is_active = False
                        inst.is_draining = True
                return FailoverAction.DRAIN
            else:
                self._failover.scale_up(count=1)
                return FailoverAction.SCALE_UP

        if result.status == HealthStatus.DEGRADED:
            return FailoverAction.RESTART

        return FailoverAction.NONE

    def _is_healthy(self, instance_id: str) -> bool:
        """Return ``True`` if *instance_id* is currently HEALTHY."""
        h = self.get_health(instance_id)
        if h is None:
            return False
        return h.status == HealthStatus.HEALTHY

    def get_recommended_action(self, instance_id: str) -> FailoverAction:
        """Return the recommended failover action for *instance_id*.

        Does **not** execute the action.
        """
        h = self.get_health(instance_id)
        if h is None:
            return FailoverAction.NONE
        if h.status == HealthStatus.HEALTHY:
            return FailoverAction.NONE
        if h.status == HealthStatus.DEGRADED:
            return FailoverAction.RESTART
        # UNHEALTHY or UNKNOWN
        healthy_peers = [
            i for i in self.list_instances()
            if i.instance_id != instance_id and self._is_healthy(i.instance_id)
        ]
        if healthy_peers:
            return FailoverAction.REDIRECT
        return FailoverAction.SCALE_UP

    # ------------------------------------------------------------------
    # Zero-downtime deployment
    # ------------------------------------------------------------------

    def start_zero_downtime_deployment(
        self,
        deployment_id: str,
        new_instances: list[InstanceInfo],
    ) -> dict[str, Any]:
        """Begin a rolling replacement of instances.

        1. Register new instances.
        2. Mark old non-draining instances as draining.
        3. Start deployment tracking.
        """
        with self._lock:
            old_instances = [
                i for i in self._instances.values() if not i.is_draining
            ]

        # Register new instances first
        for inst in new_instances:
            self.register_instance(inst)

        # Drain old instances
        for inst in old_instances:
            with self._lock:
                inst.is_draining = True
                inst.weight = 0
            self._failover.drain_instance(inst.instance_id)

        status = self._deploy.start_deployment(deployment_id, new_instances)
        with self._lock:
            self._active_deployments[deployment_id] = {
                "status": status,
                "new_instances": [i.instance_id for i in new_instances],
                "old_instances": [i.instance_id for i in old_instances],
                "start_time": time.time(),
            }
        return status

    def validate_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Validate that all new instances in *deployment_id* are healthy.

        Returns a dict with ``valid`` (bool) and ``details`` (list).
        """
        with self._lock:
            deploy = self._active_deployments.get(deployment_id)
        if deploy is None:
            return {"valid": False, "details": ["Deployment not found"]}

        details: list[str] = []
        all_healthy = True
        for iid in deploy["new_instances"]:
            h = self.get_health(iid)
            if h is None:
                all_healthy = False
                details.append(f"{iid}: no health data")
            elif h.status != HealthStatus.HEALTHY:
                all_healthy = False
                details.append(f"{iid}: {h.status.name}")
            else:
                details.append(f"{iid}: HEALTHY")

        return {"valid": all_healthy, "details": details}

    def commit_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Commit *deployment_id*: remove old instances, finish deployment."""
        with self._lock:
            deploy = self._active_deployments.get(deployment_id)
        if deploy is None:
            return {"error": "Deployment not found"}

        # Unregister old instances
        for iid in deploy["old_instances"]:
            self.unregister_instance(iid)

        result = self._deploy.finish_deployment(deployment_id)
        with self._lock:
            self._active_deployments.pop(deployment_id, None)
        return result

    def rollback_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Rollback *deployment_id*: remove new instances, restore old ones."""
        with self._lock:
            deploy = self._active_deployments.get(deployment_id)
        if deploy is None:
            return {"error": "Deployment not found"}

        # Unregister new instances
        for iid in deploy["new_instances"]:
            self.unregister_instance(iid)

        # Restore old instances
        for iid in deploy["old_instances"]:
            with self._lock:
                inst = self._instances.get(iid)
                if inst is not None:
                    inst.is_draining = False
                    inst.weight = 100
                    inst.is_active = True

        result = self._deploy.rollback_deployment(deployment_id)
        with self._lock:
            self._active_deployments.pop(deployment_id, None)
        return result

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Return current status of *deployment_id*."""
        with self._lock:
            deploy = self._active_deployments.get(deployment_id)
        if deploy is None:
            return self._deploy.get_deployment_status(deployment_id)
        return {
            "deployment_id": deployment_id,
            "active": True,
            "new_instances": deploy["new_instances"],
            "old_instances": deploy["old_instances"],
            "duration_sec": time.time() - deploy["start_time"],
        }

    # ------------------------------------------------------------------
    # Background health check loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background health check loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._check_thread = threading.Thread(
            target=self._check_loop, daemon=True
        )
        self._check_thread.start()
        _logger.info("DistributedHealthChecker started")

    def stop(self) -> None:
        """Stop the background health check loop."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._check_thread is not None:
            self._check_thread.join(timeout=5.0)
        _logger.info("DistributedHealthChecker stopped")

    def _check_loop(self) -> None:
        while not self._stop_event.is_set():
            self.check_all()
            self._stop_event.wait(timeout=self._check_interval)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def set_status_change_hook(
        self,
        hook: Callable[[str, HealthStatus, HealthStatus], None] | None,
    ) -> None:
        """Set a hook called when an instance's health status changes.

        Signature: ``hook(instance_id, old_status, new_status)``.
        """
        self._on_status_change = hook

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state and stop the background loop."""
        self.stop()
        with self._lock:
            self._instances.clear()
            self._health_history.clear()
            self._current_health.clear()
            self._active_deployments.clear()
        _logger.info("DistributedHealthChecker state reset")
