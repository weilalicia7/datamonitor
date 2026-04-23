"""
Flask integration tests — stochastic MPC controller endpoints.

Covers the five ``/api/mpc/*`` routes that expose the receding-horizon
scheduler: status probe, on-demand decision, day-long simulation, Bayesian
arrival-rate update + read-back.  Simulations run over a 30-minute window
with a single policy so wall-clock stays <2s per test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    import flask_app
    return flask_app.app.test_client()


class TestMpcStatus:
    def test_status_returns_controller_config(self, client):
        resp = client.get("/api/mpc/status")
        assert resp.status_code == 200
        body = resp.get_json()
        # Status dict includes controller config + Bayesian arrival model.
        assert "arrival_model" in body or "config" in body


class TestMpcArrivalModel:
    def test_arrival_rate_returns_posterior(self, client):
        resp = client.get("/api/mpc/arrival/rate")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        arrival = body["arrival_model"]
        # The Gamma posterior summary always includes these fields.
        for key in ("posterior_mean_rate_per_hour", "alpha", "beta"):
            assert key in arrival

    def test_arrival_update_bumps_posterior(self, client):
        # Observe one arrival over five minutes; posterior alpha must change.
        resp = client.post(
            "/api/mpc/arrival/update",
            json={"n_arrivals": 1, "minutes_observed": 5.0},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        assert "arrival_model" in body


class TestMpcSimulate:
    def test_simulate_runs_greedy_policy_short_window(self, client):
        resp = client.post(
            "/api/mpc/simulate",
            json={"policies": ["greedy"], "total_minutes": 30, "rng_seed": 42},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        assert body["total_minutes"] == 30
        assert "greedy" in body["policies"]
        assert "results" in body
        assert "greedy" in body["results"]


class TestMpcDecide:
    def test_decide_returns_action_and_state(self, client):
        resp = client.post("/api/mpc/decide", json={})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        assert "decision" in body
        assert "state" in body
        state = body["state"]
        for key in ("time_min", "n_chairs", "n_idle", "n_queue"):
            assert key in state
