"""NeuralMem Cloud — Serverless API Gateway.

Multi-tenant routing, rate limiting, billing metering, and auto-scaling signals.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from neuralmem.auth.ratelimit import RateLimitConfig, RateLimiter
from neuralmem.tenancy.models import TenantConfig

_logger = logging.getLogger(__name__)


class ScalingSignal(Enum):
    """Auto-scaling recommendation signals."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"
    EMERGENCY = "emergency"


@dataclass
class RouteRule:
    """A routing rule for tenant traffic.

    Attributes
    ----------
    tenant_id : str
        Target tenant identifier.
    path_prefix : str
        URL path prefix that maps to this tenant (e.g. ``/v1/acme``).
    backend_host : str
        Backend service host (can be a service name in K8s).
    backend_port : int
        Backend service port.
    priority : int
        Rule priority (higher wins on conflict).
    """

    tenant_id: str
    path_prefix: str
    backend_host: str = "localhost"
    backend_port: int = 8000
    priority: int = 0


@dataclass
class BillingRecord:
    """Metered billing record for a single request.

    Attributes
    ----------
    tenant_id : str
    timestamp : float
    request_count : int
    token_count : int
    compute_ms : int
    endpoint : str
    """

    tenant_id: str
    timestamp: float
    request_count: int = 1
    token_count: int = 0
    compute_ms: int = 0
    endpoint: str = ""


@dataclass
class ScalingMetrics:
    """Snapshot of metrics used for auto-scaling decisions.

    Attributes
    ----------
    qps : float
        Queries per second over the last window.
    cpu_percent : float
        Normalised CPU utilisation (0-100).
    memory_percent : float
        Normalised memory utilisation (0-100).
    latency_p99_ms : float
        99th percentile latency in milliseconds.
    active_connections : int
    """

    qps: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    latency_p99_ms: float = 0.0
    active_connections: int = 0


@dataclass
class AutoScaleThresholds:
    """Thresholds that trigger scaling signals.

    All values are inclusive — exceeding a threshold fires the associated signal.
    """

    cpu_scale_up: float = 75.0
    cpu_scale_down: float = 25.0
    memory_scale_up: float = 80.0
    memory_scale_down: float = 30.0
    qps_scale_up: float = 1000.0
    qps_scale_down: float = 100.0
    latency_p99_scale_up_ms: float = 500.0


@dataclass
class GatewayResponse:
    """Response envelope from the gateway.

    Attributes
    ----------
    status_code : int
    headers : dict
    body : Any
    billing_record : BillingRecord | None
    scaling_signal : ScalingSignal | None
    """

    status_code: int = 200
    headers: dict = field(default_factory=dict)
    body: Any = None
    billing_record: BillingRecord | None = None
    scaling_signal: ScalingSignal | None = None


class ServerlessAPIGateway:
    """Multi-tenant serverless API gateway.

    Handles:
    * Tenant routing via prefix rules
    * Per-tenant token-bucket rate limiting
    * Billing metering (request count, token usage)
    * Auto-scaling signal generation from synthetic metrics

    Parameters
    ----------
    default_rate_limit : RateLimitConfig
        Fallback rate-limit configuration for tenants that do not supply one.
    scale_thresholds : AutoScaleThresholds
        Thresholds used to compute :class:`ScalingSignal` recommendations.
    """

    def __init__(
        self,
        default_rate_limit: RateLimitConfig | None = None,
        scale_thresholds: AutoScaleThresholds | None = None,
    ) -> None:
        self._default_rate_limit = default_rate_limit or RateLimitConfig(
            requests_per_minute=60, burst_size=10
        )
        self._scale_thresholds = scale_thresholds or AutoScaleThresholds()
        self._routes: list[RouteRule] = []
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._billing: dict[str, list[BillingRecord]] = defaultdict(list)
        self._metrics_history: list[ScalingMetrics] = []
        self._lock = threading.Lock()
        self._billing_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        # Synthetic metric generators (injected for testing)
        self._metric_provider: Callable[[], ScalingMetrics] | None = None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def add_route(self, rule: RouteRule) -> None:
        """Register a routing rule.

        Rules are sorted by *priority* descending so higher-priority rules
        match first.
        """
        with self._lock:
            self._routes.append(rule)
            self._routes.sort(key=lambda r: r.priority, reverse=True)
            _logger.info("Added route for tenant %s → %s", rule.tenant_id, rule.path_prefix)

    def remove_route(self, tenant_id: str) -> None:
        """Remove all routing rules for *tenant_id*."""
        with self._lock:
            before = len(self._routes)
            self._routes = [r for r in self._routes if r.tenant_id != tenant_id]
            removed = before - len(self._routes)
            _logger.info("Removed %d route(s) for tenant %s", removed, tenant_id)

    def resolve_tenant(self, path: str) -> RouteRule | None:
        """Return the matching :class:`RouteRule` for *path*.

        The first rule whose ``path_prefix`` is a prefix of *path* wins.
        """
        with self._lock:
            for rule in self._routes:
                if path.startswith(rule.path_prefix):
                    return rule
        return None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def register_tenant_rate_limit(
        self, tenant_id: str, config: RateLimitConfig | None = None
    ) -> None:
        """Attach a rate limiter to *tenant_id*."""
        cfg = config or self._default_rate_limit
        with self._lock:
            self._rate_limiters[tenant_id] = RateLimiter(cfg)

    def check_rate_limit(self, tenant_id: str) -> tuple[bool, dict]:
        """Check whether a request for *tenant_id* is allowed.

        Returns ``(allowed, info)``.
        """
        with self._lock:
            limiter = self._rate_limiters.get(tenant_id)
        if limiter is None:
            # Auto-register with default on first encounter
            self.register_tenant_rate_limit(tenant_id)
            limiter = self._rate_limiters[tenant_id]
        return limiter.consume(tenant_id)

    # ------------------------------------------------------------------
    # Billing / metering
    # ------------------------------------------------------------------

    def record_billing(
        self,
        tenant_id: str,
        token_count: int = 0,
        compute_ms: int = 0,
        endpoint: str = "",
    ) -> BillingRecord:
        """Record a billing event for *tenant_id*.

        Returns the created :class:`BillingRecord`.
        """
        record = BillingRecord(
            tenant_id=tenant_id,
            timestamp=time.time(),
            request_count=1,
            token_count=token_count,
            compute_ms=compute_ms,
            endpoint=endpoint,
        )
        with self._billing_lock:
            self._billing[tenant_id].append(record)
        return record

    def get_billing_summary(
        self, tenant_id: str, since: float | None = None
    ) -> dict:
        """Return aggregated billing metrics for *tenant_id*.

        Keys: ``total_requests``, ``total_tokens``, ``total_compute_ms``,
        ``endpoints`` (counter dict).
        """
        cutoff = since or 0.0
        with self._billing_lock:
            records = [r for r in self._billing[tenant_id] if r.timestamp >= cutoff]
        if not records:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_compute_ms": 0,
                "endpoints": {},
            }
        endpoints: dict[str, int] = defaultdict(int)
        total_tokens = 0
        total_compute = 0
        for r in records:
            total_tokens += r.token_count
            total_compute += r.compute_ms
            endpoints[r.endpoint] += r.request_count
        return {
            "total_requests": len(records),
            "total_tokens": total_tokens,
            "total_compute_ms": total_compute,
            "endpoints": dict(endpoints),
        }

    def flush_billing(self, tenant_id: str) -> list[BillingRecord]:
        """Remove and return all billing records for *tenant_id*."""
        with self._billing_lock:
            records = self._billing[tenant_id][:]
            self._billing[tenant_id].clear()
        return records

    # ------------------------------------------------------------------
    # Auto-scaling signals
    # ------------------------------------------------------------------

    def set_metric_provider(
        self, provider: Callable[[], ScalingMetrics] | None
    ) -> None:
        """Inject a callable that returns current :class:`ScalingMetrics`.

        Used in tests to avoid relying on real infrastructure metrics.
        """
        self._metric_provider = provider

    def collect_metrics(self) -> ScalingMetrics:
        """Return current scaling metrics.

        If a metric provider is injected, its result is returned verbatim.
        Otherwise returns a zero-valued snapshot.
        """
        if self._metric_provider is not None:
            return self._metric_provider()
        return ScalingMetrics()

    def evaluate_scaling(self, metrics: ScalingMetrics | None = None) -> ScalingSignal:
        """Evaluate *metrics* against thresholds and return a signal.

        Priority order:
        1. ``EMERGENCY`` — CPU or memory > 95 %
        2. ``SCALE_UP`` — any threshold exceeded
        3. ``SCALE_DOWN`` — all metrics below down thresholds
        4. ``HOLD`` — otherwise
        """
        m = metrics or self.collect_metrics()
        th = self._scale_thresholds

        # Emergency
        if m.cpu_percent >= 95.0 or m.memory_percent >= 95.0:
            return ScalingSignal.EMERGENCY

        # Scale up
        if (
            m.cpu_percent >= th.cpu_scale_up
            or m.memory_percent >= th.memory_scale_up
            or m.qps >= th.qps_scale_up
            or m.latency_p99_ms >= th.latency_p99_scale_up_ms
        ):
            return ScalingSignal.SCALE_UP

        # Scale down
        if (
            m.cpu_percent <= th.cpu_scale_down
            and m.memory_percent <= th.memory_scale_down
            and m.qps <= th.qps_scale_down
        ):
            return ScalingSignal.SCALE_DOWN

        return ScalingSignal.HOLD

    def record_metrics(self, metrics: ScalingMetrics) -> None:
        """Store a metrics snapshot in the rolling history window."""
        with self._metrics_lock:
            self._metrics_history.append(metrics)
            # Keep last 60 snapshots (e.g. 1 minute at 1/s)
            if len(self._metrics_history) > 60:
                self._metrics_history.pop(0)

    def get_metrics_history(self) -> list[ScalingMetrics]:
        """Return the stored metrics history."""
        with self._metrics_lock:
            return list(self._metrics_history)

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def process_request(
        self,
        path: str,
        tenant_id: str | None = None,
        token_count: int = 0,
        compute_ms: int = 0,
        endpoint: str = "",
    ) -> GatewayResponse:
        """End-to-end gateway processing for a single request.

        1. Resolve tenant from *path* (or use supplied *tenant_id*).
        2. Check rate limit.
        3. Record billing.
        4. Evaluate scaling signal.
        5. Return a :class:`GatewayResponse`.
        """
        # 1. Resolve
        if tenant_id is None:
            rule = self.resolve_tenant(path)
            if rule is None:
                return GatewayResponse(
                    status_code=404,
                    body={"error": "No route matches the request path"},
                )
            tenant_id = rule.tenant_id

        # 2. Rate limit
        allowed, info = self.check_rate_limit(tenant_id)
        if not allowed:
            return GatewayResponse(
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(info.get("limit", "")),
                    "X-RateLimit-Remaining": str(info.get("remaining", "")),
                    "Retry-After": str(int(info.get("retry_after", 1))),
                },
                body={"error": "Rate limit exceeded", "info": info},
            )

        # 3. Billing
        billing = self.record_billing(tenant_id, token_count, compute_ms, endpoint)

        # 4. Scaling
        metrics = self.collect_metrics()
        self.record_metrics(metrics)
        signal = self.evaluate_scaling(metrics)

        return GatewayResponse(
            status_code=200,
            headers={
                "X-RateLimit-Limit": str(info.get("limit", "")),
                "X-RateLimit-Remaining": str(info.get("remaining", "")),
                "X-Scaling-Signal": signal.value,
            },
            body={"tenant_id": tenant_id, "routed": True},
            billing_record=billing,
            scaling_signal=signal,
        )

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def list_routes(self) -> list[RouteRule]:
        """Return a snapshot of all registered routes."""
        with self._lock:
            return list(self._routes)

    def list_tenants(self) -> list[str]:
        """Return all tenant IDs with registered rate limiters."""
        with self._lock:
            return list(self._rate_limiters.keys())

    def reset(self) -> None:
        """Clear all routes, rate limiters, billing, and metrics."""
        with self._lock:
            self._routes.clear()
            self._rate_limiters.clear()
        with self._billing_lock:
            self._billing.clear()
        with self._metrics_lock:
            self._metrics_history.clear()
        _logger.info("Gateway state reset")
