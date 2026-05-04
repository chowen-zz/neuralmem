"""Unit tests for neuralmem.cloud.gateway — ServerlessAPIGateway."""

from unittest.mock import MagicMock

import pytest

from neuralmem.cloud.gateway import (
    AutoScaleThresholds,
    BillingRecord,
    GatewayResponse,
    RouteRule,
    ScalingMetrics,
    ScalingSignal,
    ServerlessAPIGateway,
)
from neuralmem.auth.ratelimit import RateLimitConfig


@pytest.fixture
def gateway():
    return ServerlessAPIGateway()


@pytest.fixture
def gateway_with_routes(gateway):
    gateway.add_route(RouteRule("tenant-a", "/v1/a", "svc-a", 8080, priority=10))
    gateway.add_route(RouteRule("tenant-b", "/v1/b", "svc-b", 8080, priority=5))
    return gateway


class TestRouting:
    def test_add_route(self, gateway):
        gateway.add_route(RouteRule("t1", "/api/v1"))
        routes = gateway.list_routes()
        assert len(routes) == 1
        assert routes[0].tenant_id == "t1"

    def test_add_route_priority_sort(self, gateway):
        gateway.add_route(RouteRule("low", "/api", priority=1))
        gateway.add_route(RouteRule("high", "/api/v2", priority=10))
        assert gateway.list_routes()[0].tenant_id == "high"

    def test_resolve_tenant_exact_prefix(self, gateway_with_routes):
        rule = gateway_with_routes.resolve_tenant("/v1/a/query")
        assert rule is not None
        assert rule.tenant_id == "tenant-a"

    def test_resolve_tenant_no_match(self, gateway_with_routes):
        assert gateway_with_routes.resolve_tenant("/v1/c") is None

    def test_remove_route(self, gateway_with_routes):
        gateway_with_routes.remove_route("tenant-a")
        assert gateway_with_routes.resolve_tenant("/v1/a/query") is None

    def test_list_routes_isolation(self, gateway):
        gateway.add_route(RouteRule("t1", "/x"))
        snapshot = gateway.list_routes()
        snapshot.pop()
        assert len(gateway.list_routes()) == 1


class TestRateLimiting:
    def test_auto_register_default_limiter(self, gateway):
        allowed, info = gateway.check_rate_limit("new-tenant")
        assert allowed is True
        assert info["limit"] == 10  # default burst

    def test_rate_limit_exceeded(self, gateway):
        gateway.register_tenant_rate_limit(
            "tight", RateLimitConfig(requests_per_minute=1, burst_size=1)
        )
        gateway.check_rate_limit("tight")  # consume the only token
        allowed, _ = gateway.check_rate_limit("tight")
        assert allowed is False

    def test_custom_rate_limit(self, gateway):
        gateway.register_tenant_rate_limit(
            "loose", RateLimitConfig(requests_per_minute=1000, burst_size=100)
        )
        allowed, info = gateway.check_rate_limit("loose")
        assert allowed is True
        assert info["limit"] == 100


class TestBilling:
    def test_record_billing(self, gateway):
        record = gateway.record_billing("t1", token_count=42, compute_ms=120, endpoint="/chat")
        assert isinstance(record, BillingRecord)
        assert record.tenant_id == "t1"
        assert record.token_count == 42
        assert record.compute_ms == 120
        assert record.endpoint == "/chat"

    def test_get_billing_summary(self, gateway):
        gateway.record_billing("t1", token_count=10, endpoint="/a")
        gateway.record_billing("t1", token_count=20, endpoint="/b")
        summary = gateway.get_billing_summary("t1")
        assert summary["total_requests"] == 2
        assert summary["total_tokens"] == 30
        assert summary["endpoints"]["/a"] == 1
        assert summary["endpoints"]["/b"] == 1

    def test_get_billing_summary_since(self, gateway):
        import time
        now = time.time()
        gateway.record_billing("t1", token_count=5)
        summary = gateway.get_billing_summary("t1", since=now + 3600)
        assert summary["total_requests"] == 0

    def test_flush_billing(self, gateway):
        gateway.record_billing("t1")
        flushed = gateway.flush_billing("t1")
        assert len(flushed) == 1
        assert gateway.get_billing_summary("t1")["total_requests"] == 0


class TestScalingSignals:
    def test_evaluate_scale_up_cpu(self, gateway):
        metrics = ScalingMetrics(cpu_percent=80.0)
        assert gateway.evaluate_scaling(metrics) == ScalingSignal.SCALE_UP

    def test_evaluate_scale_up_memory(self, gateway):
        metrics = ScalingMetrics(memory_percent=85.0)
        assert gateway.evaluate_scaling(metrics) == ScalingSignal.SCALE_UP

    def test_evaluate_scale_up_qps(self, gateway):
        metrics = ScalingMetrics(qps=1200.0)
        assert gateway.evaluate_scaling(metrics) == ScalingSignal.SCALE_UP

    def test_evaluate_scale_up_latency(self, gateway):
        metrics = ScalingMetrics(latency_p99_ms=600.0)
        assert gateway.evaluate_scaling(metrics) == ScalingSignal.SCALE_UP

    def test_evaluate_emergency(self, gateway):
        metrics = ScalingMetrics(cpu_percent=96.0)
        assert gateway.evaluate_scaling(metrics) == ScalingSignal.EMERGENCY

    def test_evaluate_scale_down(self, gateway):
        metrics = ScalingMetrics(cpu_percent=20.0, memory_percent=25.0, qps=50.0)
        assert gateway.evaluate_scaling(metrics) == ScalingSignal.SCALE_DOWN

    def test_evaluate_hold(self, gateway):
        metrics = ScalingMetrics(cpu_percent=50.0, memory_percent=50.0, qps=500.0)
        assert gateway.evaluate_scaling(metrics) == ScalingSignal.HOLD

    def test_collect_metrics_with_provider(self, gateway):
        mock_metrics = ScalingMetrics(qps=999.0)
        gateway.set_metric_provider(lambda: mock_metrics)
        assert gateway.collect_metrics() == mock_metrics

    def test_record_and_get_history(self, gateway):
        m = ScalingMetrics(qps=10.0)
        gateway.record_metrics(m)
        assert gateway.get_metrics_history() == [m]


class TestProcessRequest:
    def test_successful_request(self, gateway_with_routes):
        resp = gateway_with_routes.process_request("/v1/a/search", token_count=5, compute_ms=10, endpoint="/search")
        assert resp.status_code == 200
        assert resp.billing_record is not None
        assert resp.billing_record.token_count == 5
        assert resp.scaling_signal is not None

    def test_404_no_route(self, gateway):
        resp = gateway.process_request("/unknown")
        assert resp.status_code == 404

    def test_429_rate_limited(self, gateway):
        gateway.register_tenant_rate_limit(
            "blocked", RateLimitConfig(requests_per_minute=1, burst_size=1)
        )
        gateway.process_request("/x", tenant_id="blocked")
        resp = gateway.process_request("/x", tenant_id="blocked")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_explicit_tenant_id(self, gateway):
        gateway.register_tenant_rate_limit("explicit")
        resp = gateway.process_request("/any", tenant_id="explicit")
        assert resp.status_code == 200
        assert resp.body["tenant_id"] == "explicit"

    def test_response_headers(self, gateway_with_routes):
        resp = gateway_with_routes.process_request("/v1/a")
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-Scaling-Signal" in resp.headers


class TestAdmin:
    def test_list_tenants(self, gateway):
        gateway.register_tenant_rate_limit("t1")
        gateway.register_tenant_rate_limit("t2")
        assert set(gateway.list_tenants()) == {"t1", "t2"}

    def test_reset(self, gateway_with_routes):
        gateway_with_routes.record_billing("tenant-a")
        gateway_with_routes.record_metrics(ScalingMetrics())
        gateway_with_routes.reset()
        assert gateway_with_routes.list_routes() == []
        assert gateway_with_routes.list_tenants() == []
        assert gateway_with_routes.get_billing_summary("tenant-a")["total_requests"] == 0
        assert gateway_with_routes.get_metrics_history() == []
