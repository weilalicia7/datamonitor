"""
Flask integration tests — mutating optimization endpoints.

``/api/optimize`` + ``/api/optimize/run`` + ``/api/optimize/pareto`` and the
weight-tuning endpoint comprise the write side of the scheduling core.  The
underlying CP-SAT solve is stubbed out in the test ``conftest.py`` so these
tests finish in <1s each; we verify that the routes return success, carry
the expected response shape, and surface the ML-integration metadata.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    import flask_app
    return flask_app.app.test_client()


class TestOptimizePost:
    def test_api_optimize_runs_full_pipeline(self, client):
        resp = client.post("/api/optimize", json={})
        assert resp.status_code == 200
        body = resp.get_json()
        # Two paths: patients exist → full result dict; none → success + msg.
        assert body.get("success") is True or "scheduled" in body
        if "scheduled" in body:
            # Full-pipeline response must include ML-integration metadata.
            assert "ml_integration" in body
            integ = body["ml_integration"]
            assert "data_channel" in integ
            assert "noshow_predictions_applied" in integ


class TestOptimizeRun:
    def test_api_optimize_run_returns_success_envelope(self, client):
        resp = client.post("/api/optimize/run")
        assert resp.status_code == 200
        body = resp.get_json()
        for key in ("success", "message", "scheduled", "appointments_count"):
            assert key in body


class TestOptimizeWeights:
    def test_get_weights_returns_description(self, client):
        resp = client.get("/api/optimize/weights")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "description" in body
        assert "weights" in body or "default_weights" in body
        descr = body["description"]
        for key in ("priority", "utilization", "noshow_risk", "travel"):
            assert key in descr

    def test_post_weights_accepts_normalised_vector(self, client):
        payload = {
            "priority": 0.2,
            "utilization": 0.2,
            "noshow_risk": 0.2,
            "waiting_time": 0.1,
            "robustness": 0.2,
            "travel": 0.1,
        }
        resp = client.post("/api/optimize/weights", json=payload)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        assert "weights" in body
        returned = body["weights"]
        # The six-objective vector must be preserved (even if normalised).
        for k in payload.keys():
            assert k in returned
