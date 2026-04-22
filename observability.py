"""
Observability — Prometheus metrics + health probes + OpenTelemetry hook
(Production-Readiness T4.5).

Three concerns:

1. **Prometheus exposition** — a module-level registry + canonical metrics
   for HTTP traffic, optimiser solve time, and ML prediction latency.
   ``/metrics`` returns the Prometheus text format.  Metrics library is
   optional: if ``prometheus_client`` isn't installed, the helpers
   degrade to no-ops rather than breaking the import.

2. **Health probes** — ``/health/live`` (is-the-process-up) and
   ``/health/ready`` (are-core-subsystems-ready).  Kept lightweight so
   load balancers can hit them 10× per second without load.

3. **OpenTelemetry hook** — env-gated (``OTEL_ENABLED=true``), optional;
   when active, every request gets a span.  Downstream callers can
   ``observe_optimizer_solve('cpsat')`` or
   ``observe_ml_prediction('noshow_model')`` around their hot path to
   emit a latency histogram + (optional) OTel span at once.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Feature gates
# --------------------------------------------------------------------------- #


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def metrics_enabled() -> bool:
    return _env_bool("METRICS_ENABLED", True)


def otel_enabled() -> bool:
    return _env_bool("OTEL_ENABLED", False)


# --------------------------------------------------------------------------- #
# Prometheus metrics (lazy import)
# --------------------------------------------------------------------------- #


_prom_available = False
try:  # pragma: no cover — exercised in tests with pytest.importorskip
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _prom_available = True
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    REGISTRY = None

    def generate_latest(*_a, **_kw):                 # pragma: no cover
        return b""


# HTTP metrics — labels narrow enough that cardinality stays bounded
# (endpoint is the Flask rule template, never the full URL with IDs).
if _prom_available:
    HTTP_REQUESTS = Counter(
        "sact_http_requests_total",
        "Total HTTP requests handled, by endpoint + method + status",
        ("method", "endpoint", "status"),
    )
    HTTP_LATENCY = Histogram(
        "sact_http_request_duration_seconds",
        "Wall-clock duration of an HTTP request, by endpoint + method",
        ("method", "endpoint"),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    OPTIMIZER_LATENCY = Histogram(
        "sact_optimizer_solve_seconds",
        "Optimisation solver wall time, by solver name",
        ("solver",),
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    ML_LATENCY = Histogram(
        "sact_ml_prediction_seconds",
        "ML prediction wall time, by model name",
        ("model",),
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    )
    APP_READY = Gauge(
        "sact_app_ready",
        "1 if the application has finished startup, 0 otherwise",
    )
else:                                                # pragma: no cover
    HTTP_REQUESTS = HTTP_LATENCY = OPTIMIZER_LATENCY = ML_LATENCY = APP_READY = None


# --------------------------------------------------------------------------- #
# OpenTelemetry (optional)
# --------------------------------------------------------------------------- #


_otel_tracer = None


def _try_init_otel() -> Optional[Any]:               # pragma: no cover — network setup
    """Return an OpenTelemetry Tracer if ``OTEL_ENABLED=true`` + SDK is present."""
    global _otel_tracer
    if _otel_tracer is not None:
        return _otel_tracer
    if not otel_enabled():
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        logger.warning("otel: OpenTelemetry SDK not installed")
        return None
    service_name = os.environ.get("OTEL_SERVICE_NAME", "sact-scheduler")
    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    # OTLP endpoint is the normal export path; fall back to no-op exporter
    # when the dependency is absent so OTEL=true without an endpoint is safe.
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else None
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception:
        pass
    trace.set_tracer_provider(provider)
    _otel_tracer = trace.get_tracer(service_name)
    return _otel_tracer


# --------------------------------------------------------------------------- #
# Context managers for hot-path instrumentation
# --------------------------------------------------------------------------- #


@contextmanager
def observe_optimizer_solve(solver: str) -> Iterator[None]:
    """Record optimiser wall time + optionally emit an OTel span."""
    start = time.perf_counter()
    tracer = _try_init_otel()
    span_cm = tracer.start_as_current_span(
        f"optimizer.{solver}",
        attributes={"sact.solver": solver},
    ) if tracer is not None else None                 # pragma: no cover
    if span_cm is not None:                           # pragma: no cover
        span_cm.__enter__()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if OPTIMIZER_LATENCY is not None:
            OPTIMIZER_LATENCY.labels(solver=solver).observe(elapsed)
        if span_cm is not None:                       # pragma: no cover
            span_cm.__exit__(None, None, None)


@contextmanager
def observe_ml_prediction(model: str) -> Iterator[None]:
    """Record ML prediction wall time + optionally emit an OTel span."""
    start = time.perf_counter()
    tracer = _try_init_otel()
    span_cm = tracer.start_as_current_span(
        f"ml.{model}",
        attributes={"sact.model": model},
    ) if tracer is not None else None                 # pragma: no cover
    if span_cm is not None:                           # pragma: no cover
        span_cm.__enter__()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if ML_LATENCY is not None:
            ML_LATENCY.labels(model=model).observe(elapsed)
        if span_cm is not None:                       # pragma: no cover
            span_cm.__exit__(None, None, None)


# --------------------------------------------------------------------------- #
# Flask wiring
# --------------------------------------------------------------------------- #


_readiness_checks: Dict[str, Callable[[], bool]] = {}


def register_readiness_check(name: str, check_fn: Callable[[], bool]) -> None:
    """Register a subsystem check consumed by ``/health/ready``.

    ``check_fn`` must return ``True`` iff the subsystem is usable.  It
    MUST be cheap (no network calls); probes fire many times per minute
    and a slow check pages the on-call for a non-problem.
    """
    _readiness_checks[name] = check_fn


def unregister_readiness_check(name: str) -> None:
    _readiness_checks.pop(name, None)


def readiness_snapshot() -> Dict[str, Any]:
    """Aggregate all readiness checks; each runs in a try/except."""
    results: Dict[str, Any] = {}
    all_ok = True
    for name, fn in _readiness_checks.items():
        try:
            ok = bool(fn())
            results[name] = {"ok": ok}
        except Exception as exc:
            ok = False
            results[name] = {"ok": False, "error": str(exc)}
        if not ok:
            all_ok = False
    return {"ready": all_ok, "checks": results}


def attach_observability(app) -> None:
    """Register ``/metrics``, ``/health/live``, ``/health/ready`` + request hooks."""
    from flask import jsonify, request, Response

    # Time each request — endpoint label is the Flask rule template, not the URL.
    @app.before_request
    def _obs_start():                                 # pragma: no cover — trivial
        request._obs_start = time.perf_counter()

    @app.after_request
    def _obs_record(resp):                            # pragma: no cover — trivial
        if HTTP_REQUESTS is None:
            return resp
        try:
            endpoint = request.url_rule.rule if request.url_rule else "unknown"
            HTTP_REQUESTS.labels(
                method=request.method,
                endpoint=endpoint,
                status=str(resp.status_code),
            ).inc()
            start = getattr(request, "_obs_start", None)
            if start is not None and HTTP_LATENCY is not None:
                HTTP_LATENCY.labels(
                    method=request.method, endpoint=endpoint
                ).observe(time.perf_counter() - start)
        except Exception:
            pass
        return resp

    @app.route("/metrics")
    def _metrics():
        if not metrics_enabled() or not _prom_available:
            return Response("", content_type="text/plain", status=503)
        return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)

    @app.route("/health/live")
    def _health_live():
        return jsonify({"status": "alive"}), 200

    @app.route("/health/ready")
    def _health_ready():
        snap = readiness_snapshot()
        status = 200 if snap["ready"] else 503
        return jsonify(snap), status


def mark_app_ready() -> None:
    """Signal the app has finished its startup routine (drives ``sact_app_ready``)."""
    if APP_READY is not None:
        APP_READY.set(1)


def mark_app_not_ready() -> None:
    if APP_READY is not None:
        APP_READY.set(0)
