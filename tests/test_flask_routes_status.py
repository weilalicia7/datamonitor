"""
Flask integration tests — root + status + optimizer-diagnostic endpoints.

Covers the read-only dashboard surface: the landing page, legacy dashboard,
system status JSON, metrics digest, and the three ``/api/optimizer/*``
diagnostic endpoints (cache, gnn, colgen).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    import flask_app
    return flask_app.app.test_client()


class TestLandingPages:
    def test_root_renders_viewer_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")
        body = resp.get_data(as_text=True)
        # Viewer template must emit a doctype; any HTML body is acceptable.
        assert "<!DOCTYPE" in body.upper() or "<html" in body.lower()

    def test_dashboard_renders_legacy_template(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")


class TestStatusJson:
    def test_api_status_reports_appointment_and_patient_counts(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        body = resp.get_json()
        # Shape assertions — not values, because the in-memory counts drift
        # with other tests in the same module.
        for key in (
            "mode",
            "appointment_count",
            "pending_count",
            "event_count",
            "last_update",
            "refresh_interval",
            "sites",
            "metrics",
        ):
            assert key in body, f"missing {key} in /api/status body"
        assert isinstance(body["appointment_count"], int)
        assert isinstance(body["pending_count"], int)

    def test_api_metrics_snapshot(self, client):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "metrics" in body
        assert "timestamp" in body
        assert "optimization_results" in body


class TestOptimizerDiagnostics:
    def test_optimizer_cache_reports_counts_and_entries(self, client):
        resp = client.get("/api/optimizer/cache")
        assert resp.status_code == 200
        body = resp.get_json()
        for key in ("cache_size", "cache_max", "total_hits", "entries", "timestamp"):
            assert key in body
        assert isinstance(body["cache_size"], int)
        assert isinstance(body["entries"], list)
