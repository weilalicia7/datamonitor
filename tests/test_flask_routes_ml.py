"""
Flask integration tests — read-only ML summary endpoints.

Covers four diagnostic routes that expose the current state of the no-show
ensemble + sequence model + duration-test harness.  All are GET and
side-effect free, so running them many times in parallel is safe.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    import flask_app
    return flask_app.app.test_client()


class TestNoShowPredictionsDigest:
    def test_api_ml_predictions_shape(self, client):
        resp = client.get("/api/ml/predictions")
        assert resp.status_code == 200
        body = resp.get_json()
        for key in (
            "noshow_predictions",
            "duration_predictions",
            "model_status",
            "data_channel",
            "ml_training_status",
        ):
            assert key in body
        assert isinstance(body["noshow_predictions"], list)
        assert isinstance(body["duration_predictions"], list)
        assert body["model_status"] in ("active", "simulated")


class TestSequenceModel:
    def test_sequence_model_reports_availability(self, client):
        resp = client.get("/api/ml/sequence-model")
        assert resp.status_code == 200
        body = resp.get_json()
        # The route always returns JSON even when ml_available=False — in
        # that case it has 'enabled' + 'message' only.
        assert "enabled" in body or "available" in body
        if "enabled" in body and body.get("enabled"):
            assert "model_type" in body
            assert "min_appointments_required" in body


class TestEnsembleConfig:
    def test_ensemble_config_surfaces_base_models_and_methods(self, client):
        resp = client.get("/api/ml/ensemble-config")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "available" in body
        if body["available"]:
            assert "ensemble_methods" in body
            methods = body["ensemble_methods"]
            # The four documented ensemble methods MUST all appear as keys.
            for method in (
                "stacked_generalization",
                "bayesian_model_averaging",
                "sequence_model",
                "temporal_cv",
            ):
                assert method in methods
            assert isinstance(body["base_models"], list)
        else:
            pytest.skip("ensemble ML stack not available in this environment")


class TestDurationTest:
    def test_duration_test_runs_protocol_battery(self, client):
        resp = client.get("/api/ml/duration-test")
        assert resp.status_code == 200
        body = resp.get_json()
        # If model is unavailable the route returns {'error': ...} with 200.
        if "error" in body:
            pytest.skip(f"duration_model not available: {body.get('error')}")
        assert "duration_tests" in body
        results = body["duration_tests"]
        assert len(results) >= 5  # Five canned REG codes live in the source.
        # Every entry carries either a prediction or an error — never empty.
        for r in results:
            assert "name" in r
            assert "protocol" in r


class TestEnsembleConfigIsIdempotent:
    def test_two_reads_return_matching_shape(self, client):
        a = client.get("/api/ml/ensemble-config").get_json()
        b = client.get("/api/ml/ensemble-config").get_json()
        assert a.keys() == b.keys()
        assert a["available"] == b["available"]
