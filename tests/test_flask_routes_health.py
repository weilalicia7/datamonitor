"""
Flask integration tests — health, metrics, auth endpoints.

These routes are the public surface of the observability / auth layer.  They
are exercised BEFORE any other test suite so failures here flag import-time
issues (import flask_app) rather than business-logic regressions.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    """Flask test client over the already-initialised flask_app singleton."""
    import flask_app
    return flask_app.app.test_client()


class TestHealthProbes:
    def test_health_live_always_200(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body == {"status": "alive"}

    def test_health_ready_reports_subsystems(self, client):
        resp = client.get("/health/ready")
        # In dev both pandas + ml_available are True → 200; otherwise 503.
        assert resp.status_code in (200, 503)
        body = resp.get_json()
        assert "ready" in body and "checks" in body
        # The two baked-in readiness checks must be present.
        assert "pandas_available" in body["checks"]
        assert "ml_available" in body["checks"]


class TestMetricsExposition:
    def test_metrics_returns_prometheus_text(self, client):
        resp = client.get("/metrics")
        # If prometheus_client absent the module returns 503 — we accept that.
        if resp.status_code == 503:
            pytest.skip("prometheus_client not installed")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/plain")
        text = resp.get_data(as_text=True)
        # At least one HELP line indicates live exposition.
        assert "# HELP" in text


class TestAuthRoutesAreOpen:
    def test_auth_whoami_unauth_when_disabled(self, client):
        resp = client.get("/auth/whoami")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["authenticated"] is False
        assert body["username"] is None
        # The default dev deployment has AUTH_ENABLED=false.
        assert body["auth_enabled"] in (True, False)

    def test_auth_login_noop_when_auth_disabled(self, client):
        resp = client.post(
            "/auth/login",
            json={"username": "x", "password": "y"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        # The 'note' field only appears when AUTH_ENABLED=false (the default
        # dev condition).  When auth is enabled the route enforces real creds.
        if not body.get("note"):
            # Auth is actually enabled — the login path returns a role.
            assert "role" in body
        else:
            assert "no-op" in body["note"].lower()

    def test_auth_logout_succeeds(self, client):
        resp = client.post("/auth/logout")
        assert resp.status_code == 200
        assert resp.get_json() == {"success": True}
