"""
Flask integration tests — digital-twin policy evaluation endpoints.

The twin simulates the in-memory schedule forward under a policy spec and
returns guardrail / KPI metrics.  We exercise the three user-facing routes
(``evaluate``, ``compare``, ``evaluations``) with a small horizon so each
call finishes in <2s, and separately verify the ``horizon_days`` cap
enforcement: 999999 is silently clamped to the ENDPOINT_BOUNDS value (365),
NOT a 400.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    import flask_app
    return flask_app.app.test_client()


class TestTwinEvaluate:
    def test_evaluate_minimal_horizon_returns_metrics(self, client):
        resp = client.post(
            "/api/twin/evaluate",
            json={"horizon_days": 2, "step_hours": 24, "rng_seed": 42},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        ev = body["evaluation"]
        # Canonical KPI keys must be present on every evaluation.
        for key in (
            "horizon_days",
            "accept_rate",
            "mean_utilization_pct",
            "noshow_rate_realised",
            "num_steps",
            "policy",
        ):
            assert key in ev
        assert ev["horizon_days"] == 2

    def test_horizon_cap_is_silently_clamped(self, client):
        """ENDPOINT_BOUNDS caps horizon_days at 365; oversized → 200."""
        resp = client.post(
            "/api/twin/evaluate",
            json={"horizon_days": 999_999, "step_hours": 24, "rng_seed": 42},
        )
        # clamp_int clamps DOWN to the cap rather than raising — 200 expected.
        assert resp.status_code == 200
        body = resp.get_json()
        ev = body["evaluation"]
        # Cap is 365 per validators.ENDPOINT_BOUNDS["horizon_days"].
        assert ev["horizon_days"] == 365


class TestTwinCompare:
    def test_compare_two_policies_returns_comparison(self, client):
        payload = {
            "policies": [
                {"name": "baseline", "double_book_threshold": 0.3},
                {"name": "aggressive", "double_book_threshold": 0.05},
            ],
            "horizon_days": 2,
            "step_hours": 24,
            "rng_seed": 42,
        }
        resp = client.post("/api/twin/compare", json=payload)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        comp = body["comparison"]
        assert "policies" in comp
        assert len(comp["policies"]) == 2
        assert "best_policy" in comp

    def test_compare_rejects_empty_policies(self, client):
        resp = client.post(
            "/api/twin/compare",
            json={"policies": [], "horizon_days": 1, "rng_seed": 42},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body.get("success") is False
        assert "no policies" in body.get("error", "").lower()


class TestTwinEvaluations:
    def test_evaluations_history_returns_list(self, client):
        # Seed at least one evaluation first so the list is guaranteed non-empty.
        client.post(
            "/api/twin/evaluate",
            json={"horizon_days": 1, "step_hours": 24, "rng_seed": 7},
        )
        resp = client.get("/api/twin/evaluations")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        assert "evaluations" in body
        evs = body["evaluations"]
        assert isinstance(evs, list)
        # Every historical record carries a timestamp + policy summary.
        if evs:
            sample = evs[0]
            assert "evaluated_ts" in sample or "file" in sample
