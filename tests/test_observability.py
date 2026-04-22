"""
Tests for observability.py — Prometheus metrics + health probes + OTel gate
(Production-Readiness T4.5).

Structure:
- TestEnvGates         — metrics_enabled / otel_enabled env parsing
- TestReadinessChecks  — register/unregister, snapshot aggregates, failures
- TestFlaskEndpoints   — /health/live, /health/ready (ok + degraded), /metrics
- TestInstrumentation  — observe_optimizer_solve / observe_ml_prediction
                         record to Prometheus histograms
- TestAppReadyGauge    — mark_app_ready / mark_app_not_ready flip the gauge
- TestGrafanaDashboard — the committed JSON parses and references our metrics
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from flask import Flask

import observability as obs


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clear_readiness_checks():
    # Reset the module-level registry per test so tests don't contaminate each other.
    obs._readiness_checks.clear()
    yield
    obs._readiness_checks.clear()


def _metrics_text(app):
    """Scrape /metrics via the test client and decode."""
    resp = app.test_client().get("/metrics")
    return resp.status_code, resp.get_data(as_text=True)


def _build_app():
    app = Flask(__name__)
    app.secret_key = "test"
    obs.attach_observability(app)
    return app


# --------------------------------------------------------------------------- #
# Env gates
# --------------------------------------------------------------------------- #


class TestEnvGates:
    def test_metrics_default_on(self, monkeypatch):
        monkeypatch.delenv("METRICS_ENABLED", raising=False)
        assert obs.metrics_enabled() is True

    def test_metrics_toggle_off(self, monkeypatch):
        monkeypatch.setenv("METRICS_ENABLED", "false")
        assert obs.metrics_enabled() is False

    def test_otel_default_off(self, monkeypatch):
        monkeypatch.delenv("OTEL_ENABLED", raising=False)
        assert obs.otel_enabled() is False

    def test_otel_toggle_on(self, monkeypatch):
        monkeypatch.setenv("OTEL_ENABLED", "true")
        assert obs.otel_enabled() is True


# --------------------------------------------------------------------------- #
# Readiness checks
# --------------------------------------------------------------------------- #


class TestReadinessChecks:
    def test_empty_registry_is_ready(self):
        snap = obs.readiness_snapshot()
        assert snap["ready"] is True
        assert snap["checks"] == {}

    def test_all_ok(self):
        obs.register_readiness_check("a", lambda: True)
        obs.register_readiness_check("b", lambda: True)
        snap = obs.readiness_snapshot()
        assert snap["ready"] is True
        assert snap["checks"]["a"] == {"ok": True}
        assert snap["checks"]["b"] == {"ok": True}

    def test_one_failure_flips_ready(self):
        obs.register_readiness_check("a", lambda: True)
        obs.register_readiness_check("b", lambda: False)
        snap = obs.readiness_snapshot()
        assert snap["ready"] is False
        assert snap["checks"]["b"] == {"ok": False}

    def test_exception_treated_as_failure(self):
        def boom():
            raise RuntimeError("cache unreachable")
        obs.register_readiness_check("cache", boom)
        snap = obs.readiness_snapshot()
        assert snap["ready"] is False
        assert snap["checks"]["cache"]["ok"] is False
        assert "cache unreachable" in snap["checks"]["cache"]["error"]

    def test_unregister(self):
        obs.register_readiness_check("a", lambda: True)
        obs.unregister_readiness_check("a")
        obs.unregister_readiness_check("noop")  # idempotent
        snap = obs.readiness_snapshot()
        assert "a" not in snap["checks"]


# --------------------------------------------------------------------------- #
# Flask endpoints
# --------------------------------------------------------------------------- #


class TestFlaskEndpoints:
    def test_health_live_always_200(self):
        app = _build_app()
        resp = app.test_client().get("/health/live")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "alive"}

    def test_health_ready_empty_is_ready(self):
        app = _build_app()
        resp = app.test_client().get("/health/ready")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ready"] is True

    def test_health_ready_degraded_returns_503(self):
        app = _build_app()
        obs.register_readiness_check("db", lambda: False)
        resp = app.test_client().get("/health/ready")
        assert resp.status_code == 503
        body = resp.get_json()
        assert body["ready"] is False
        assert body["checks"]["db"] == {"ok": False}

    def test_metrics_200_when_enabled(self, monkeypatch):
        pytest.importorskip("prometheus_client")
        monkeypatch.setenv("METRICS_ENABLED", "true")
        app = _build_app()
        # Hit a route first so we have at least one labeled metric.
        app.test_client().get("/health/live")
        status, text = _metrics_text(app)
        assert status == 200
        # Canonical Prometheus HELP/TYPE lines should be present.
        assert "sact_http_requests_total" in text

    def test_metrics_503_when_disabled(self, monkeypatch):
        monkeypatch.setenv("METRICS_ENABLED", "false")
        app = _build_app()
        status, _ = _metrics_text(app)
        assert status == 503


# --------------------------------------------------------------------------- #
# Hot-path instrumentation
# --------------------------------------------------------------------------- #


class TestInstrumentation:
    def test_observe_optimizer_records_sample(self):
        pytest.importorskip("prometheus_client")
        before = _sum_samples("sact_optimizer_solve_seconds_count", {"solver": "cpsat"})
        with obs.observe_optimizer_solve("cpsat"):
            time.sleep(0.001)
        after = _sum_samples("sact_optimizer_solve_seconds_count", {"solver": "cpsat"})
        assert after == pytest.approx(before + 1)

    def test_observe_ml_records_sample(self):
        pytest.importorskip("prometheus_client")
        before = _sum_samples("sact_ml_prediction_seconds_count", {"model": "noshow"})
        with obs.observe_ml_prediction("noshow"):
            time.sleep(0.001)
        after = _sum_samples("sact_ml_prediction_seconds_count", {"model": "noshow"})
        assert after == pytest.approx(before + 1)

    def test_observe_handles_exception_inside_block(self):
        pytest.importorskip("prometheus_client")
        before = _sum_samples("sact_optimizer_solve_seconds_count", {"solver": "boom"})
        with pytest.raises(ValueError):
            with obs.observe_optimizer_solve("boom"):
                raise ValueError("kaboom")
        after = _sum_samples("sact_optimizer_solve_seconds_count", {"solver": "boom"})
        # Exception path still records — crucial for debugging slow-failing solves.
        assert after == pytest.approx(before + 1)


# --------------------------------------------------------------------------- #
# App-ready gauge
# --------------------------------------------------------------------------- #


class TestAppReadyGauge:
    def test_gauge_can_be_set_and_cleared(self):
        pytest.importorskip("prometheus_client")
        obs.mark_app_ready()
        assert _gauge_value("sact_app_ready") == 1
        obs.mark_app_not_ready()
        assert _gauge_value("sact_app_ready") == 0
        # Reset state for downstream tests that assume "ready".
        obs.mark_app_ready()


# --------------------------------------------------------------------------- #
# Grafana dashboard sanity
# --------------------------------------------------------------------------- #


class TestGrafanaDashboard:
    def test_parses_and_references_our_metrics(self):
        path = Path(__file__).parent.parent / "docs" / "observability" / "grafana_dashboard.json"
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["title"].startswith("SACT")
        flat = json.dumps(payload)
        for metric in (
            "sact_http_requests_total",
            "sact_http_request_duration_seconds",
            "sact_optimizer_solve_seconds",
            "sact_ml_prediction_seconds",
            "sact_app_ready",
        ):
            assert metric in flat, f"dashboard missing {metric}"


# --------------------------------------------------------------------------- #
# Low-level helpers for reading the default Prometheus registry
# --------------------------------------------------------------------------- #


def _sum_samples(metric_name: str, label_filter: dict) -> float:
    """Sum the samples of a Prometheus metric matching every label in the filter."""
    from prometheus_client import REGISTRY
    total = 0.0
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name != metric_name:
                continue
            if all(sample.labels.get(k) == v for k, v in label_filter.items()):
                total += sample.value
    return total


def _gauge_value(metric_name: str) -> float:
    from prometheus_client import REGISTRY
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == metric_name:
                return sample.value
    return float("nan")
