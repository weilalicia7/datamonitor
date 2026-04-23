"""
Flask integration tests — validator guardrails (T4.2).

Three concerns, five tests:

* **DML whitelist rejection.**  ``validators.DML_TREATMENT_ALLOWED`` pins
  the set of column names the DML endpoints accept.  An attempt to smuggle
  SQL fragments in via ``treatment`` must return 400 with ``field='treatment'``.

* **Input-cap silent clamp.**  ``clamp_int`` clamps DOWN to the endpoint
  cap rather than rejecting — so ``horizon_days=999999`` on
  ``/api/twin/evaluate`` returns 200 (not 400).

* **MAX_CONTENT_LENGTH.**  A 17 MB body on a JSON-reading endpoint is
  rejected by Werkzeug with 413 before the view function runs.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    import flask_app
    return flask_app.app.test_client()


class TestDmlWhitelistRejection:
    def test_dml_estimate_rejects_sql_fragment(self, client):
        """Attempted SQLi payload on treatment name → 400 + field='treatment'."""
        resp = client.post(
            "/api/ml/dml/estimate",
            json={"treatment": "DROP TABLE users", "outcome": "no_show"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body.get("success") is False
        assert body.get("field") == "treatment"
        assert "treatment" in body.get("error", "").lower()

    def test_dml_estimate_rejects_unknown_outcome(self, client):
        """Outcome column not in DML_OUTCOME_ALLOWED → 400 + field='outcome'."""
        resp = client.post(
            "/api/ml/dml/estimate",
            json={"treatment": "SMS_Reminder", "outcome": "'; DROP"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body.get("success") is False
        assert body.get("field") == "outcome"


class TestInputCapClamping:
    def test_horizon_days_oversized_is_clamped_not_rejected(self, client):
        """horizon_days=999999 → clamped to 365, NOT a 400 error."""
        resp = client.post(
            "/api/twin/evaluate",
            json={"horizon_days": 999_999, "step_hours": 24, "rng_seed": 42},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("success") is True
        assert body["evaluation"]["horizon_days"] == 365


class TestMaxContentLength:
    def test_oversized_json_body_returns_413(self, client):
        """A 17 MB body on a route that reads request.json triggers Werkzeug 413."""
        # Build a JSON blob just above the 16 MB MAX_CONTENT_LENGTH ceiling.
        padding = "x" * (17 * 1024 * 1024)
        body_bytes = (
            b'{"treatment":"SMS_Reminder","outcome":"no_show","pad":"'
            + padding.encode("ascii")
            + b'"}'
        )
        resp = client.post(
            "/api/ml/dml/estimate",
            data=body_bytes,
            content_type="application/json",
        )
        assert resp.status_code == 413


class TestNegativeIntegerRejection:
    def test_negative_n_folds_rejected_with_400(self, client):
        """clamp_int raises when coerced value < min_value → 400."""
        resp = client.post(
            "/api/ml/dml/estimate",
            json={
                "treatment": "SMS_Reminder",
                "outcome": "no_show",
                "n_folds": -1,
            },
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body.get("success") is False
        assert body.get("field") == "n_folds"
        assert "below minimum" in body.get("error", "").lower()
