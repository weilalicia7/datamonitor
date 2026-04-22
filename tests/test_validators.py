"""
Tests for validators.py — input caps, string whitelists, rate-limiter factory
(Production-Readiness T4.2).

Structure:
- TestClampInt        — integer coercion + clamping + error cases
- TestClampFloat      — same for floats
- TestEndpointBounds  — per-key caps + env-var overrides
- TestWhitelist       — single-value + many-value string whitelists
- TestErrorResponse   — translation to Flask 400 JSON
- TestRateLimiter     — env gating of Flask-Limiter integration
- TestFlaskIntegration — MAX_CONTENT_LENGTH enforcement via the test client
"""

from __future__ import annotations

import os

import pytest
from flask import Flask, jsonify, request

from validators import (
    DEFAULT_MAX_CONTENT_LENGTH,
    DML_COVARIATES_ALLOWED,
    DML_OUTCOME_ALLOWED,
    DML_TREATMENT_ALLOWED,
    ENDPOINT_BOUNDS,
    ValidationError,
    clamp_float,
    clamp_int,
    get_cap,
    init_rate_limiter,
    rate_limit_enabled,
    validate_whitelist,
    validate_whitelist_many,
    validation_error_response,
)


# --------------------------------------------------------------------------- #
# clamp_int
# --------------------------------------------------------------------------- #


class TestClampInt:
    def test_passthrough_inside_bounds(self):
        assert clamp_int(30, field="horizon_days", default=7) == 30

    def test_clamps_above_cap_silently(self):
        # horizon_days cap = 365 by default
        assert clamp_int(1_000_000, field="horizon_days", default=7) == 365

    def test_uses_default_when_none(self):
        assert clamp_int(None, field="horizon_days", default=7) == 7

    def test_explicit_max_overrides_get_cap(self):
        assert clamp_int(50, field="horizon_days", default=7, max_value=10) == 10

    def test_accepts_string_integers(self):
        assert clamp_int("42", field="horizon_days", default=7) == 42

    def test_raises_on_non_numeric(self):
        with pytest.raises(ValidationError) as exc:
            clamp_int("not-a-number", field="horizon_days", default=7)
        assert exc.value.field == "horizon_days"

    def test_raises_on_below_minimum(self):
        with pytest.raises(ValidationError) as exc:
            clamp_int(-5, field="horizon_days", default=7, min_value=0)
        assert "below minimum" in str(exc.value)
        assert exc.value.field == "horizon_days"

    def test_unknown_field_uses_fallback_cap(self):
        # Fields not in ENDPOINT_BOUNDS fall through to 1_000_000 default.
        assert clamp_int(999_999, field="unseen_param", default=1) == 999_999
        assert clamp_int(5_000_000, field="unseen_param", default=1) == 1_000_000

    def test_rejects_floats_silently_truncated(self):
        # int(3.7) → 3; we accept this because json numerics can arrive as float
        assert clamp_int(3.7, field="horizon_days", default=7) == 3


# --------------------------------------------------------------------------- #
# clamp_float
# --------------------------------------------------------------------------- #


class TestClampFloat:
    def test_passthrough_inside_bounds(self):
        assert clamp_float(0.5, field="epsilon", default=0.1) == pytest.approx(0.5)

    def test_clamps_above_cap(self):
        # epsilon cap = 1.0
        assert clamp_float(999.0, field="epsilon", default=0.1) == pytest.approx(1.0)

    def test_uses_default_when_none(self):
        assert clamp_float(None, field="epsilon", default=0.1) == pytest.approx(0.1)

    def test_accepts_string_floats(self):
        assert clamp_float("0.25", field="epsilon", default=0.1) == pytest.approx(0.25)

    def test_raises_on_non_numeric(self):
        with pytest.raises(ValidationError):
            clamp_float("abc", field="epsilon", default=0.1)

    def test_raises_on_below_minimum(self):
        with pytest.raises(ValidationError):
            clamp_float(-0.1, field="epsilon", default=0.1, min_value=0.0)


# --------------------------------------------------------------------------- #
# ENDPOINT_BOUNDS + env override
# --------------------------------------------------------------------------- #


class TestEndpointBounds:
    def test_defaults_reasonable(self):
        # Sanity: caps are finite + policy-sensible.
        assert ENDPOINT_BOUNDS["horizon_days"] == 365
        assert ENDPOINT_BOUNDS["total_minutes"] == 1_440
        assert ENDPOINT_BOUNDS["n_scenarios"] <= 1_000
        assert ENDPOINT_BOUNDS["limit"] == 1_000

    def test_get_cap_returns_default(self):
        assert get_cap("horizon_days") == 365

    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("VALIDATOR_CAP_HORIZON_DAYS", "14")
        assert get_cap("horizon_days") == 14

    def test_env_override_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("VALIDATOR_CAP_HORIZON_DAYS", "not-a-number")
        assert get_cap("horizon_days") == 365

    def test_unknown_field_returns_million(self):
        assert get_cap("never_defined_xyz") == 1_000_000


# --------------------------------------------------------------------------- #
# Whitelists
# --------------------------------------------------------------------------- #


class TestWhitelist:
    def test_accepts_whitelisted_value(self):
        assert validate_whitelist(
            "Reminder_Sent",
            field="treatment",
            allowed=DML_TREATMENT_ALLOWED,
        ) == "Reminder_Sent"

    def test_rejects_non_whitelisted_value(self):
        with pytest.raises(ValidationError) as exc:
            validate_whitelist(
                "DROP TABLE users",
                field="treatment",
                allowed=DML_TREATMENT_ALLOWED,
            )
        assert exc.value.field == "treatment"

    def test_rejects_empty_by_default(self):
        with pytest.raises(ValidationError):
            validate_whitelist("", field="treatment", allowed={"A", "B"})

    def test_rejects_none_by_default(self):
        with pytest.raises(ValidationError):
            validate_whitelist(None, field="treatment", allowed={"A", "B"})

    def test_outcome_whitelist_accepts(self):
        assert validate_whitelist(
            "no_show", field="outcome", allowed=DML_OUTCOME_ALLOWED
        ) == "no_show"

    def test_many_accepts_list(self):
        out = validate_whitelist_many(
            ["age", "Priority"],
            field="covariates",
            allowed=DML_COVARIATES_ALLOWED,
        )
        assert out == ["age", "Priority"]

    def test_many_rejects_mixed_good_and_bad(self):
        with pytest.raises(ValidationError):
            validate_whitelist_many(
                ["age", "evil_column"],
                field="covariates",
                allowed=DML_COVARIATES_ALLOWED,
            )

    def test_many_rejects_non_list(self):
        with pytest.raises(ValidationError):
            validate_whitelist_many(
                "age",  # scalar, not a list
                field="covariates",
                allowed=DML_COVARIATES_ALLOWED,
            )

    def test_many_accepts_none_as_empty(self):
        assert validate_whitelist_many(
            None, field="covariates", allowed=DML_COVARIATES_ALLOWED,
        ) == []


# --------------------------------------------------------------------------- #
# validation_error_response
# --------------------------------------------------------------------------- #


class TestErrorResponse:
    def test_shape(self):
        app = Flask(__name__)
        with app.test_request_context():
            resp, status = validation_error_response(
                ValidationError("bad param", field="horizon_days")
            )
            assert status == 400
            payload = resp.get_json()
            assert payload["success"] is False
            assert payload["field"] == "horizon_days"
            assert "bad param" in payload["error"]

    def test_no_field(self):
        app = Flask(__name__)
        with app.test_request_context():
            resp, status = validation_error_response(ValidationError("oops"))
            assert status == 400
            payload = resp.get_json()
            assert payload["field"] is None


# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #


class TestRateLimiter:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
        assert rate_limit_enabled() is False

    def test_enabled_various_truthy(self, monkeypatch):
        for val in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("RATE_LIMIT_ENABLED", val)
            assert rate_limit_enabled() is True, f"value={val!r}"

    def test_disabled_on_falsy(self, monkeypatch):
        for val in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("RATE_LIMIT_ENABLED", val)
            assert rate_limit_enabled() is False, f"value={val!r}"

    def test_init_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
        app = Flask(__name__)
        limiter, enabled = init_rate_limiter(app)
        assert limiter is None
        assert enabled is False

    def test_init_when_enabled(self, monkeypatch):
        pytest.importorskip("flask_limiter")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        app = Flask(__name__)
        limiter, enabled = init_rate_limiter(app)
        assert enabled is True
        assert limiter is not None


# --------------------------------------------------------------------------- #
# MAX_CONTENT_LENGTH enforcement
# --------------------------------------------------------------------------- #


class TestMaxContentLength:
    def test_default_is_16mb(self):
        assert DEFAULT_MAX_CONTENT_LENGTH == 16 * 1024 * 1024

    def test_flask_rejects_oversized_body(self):
        app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 1024  # 1 KB for the test
        app.config["TESTING"] = True

        @app.route("/echo", methods=["POST"])
        def echo():
            return jsonify({"len": len(request.get_data())})

        client = app.test_client()
        # Exceed the cap: Flask returns 413 Request Entity Too Large.
        resp = client.post("/echo", data=b"X" * 2048,
                           content_type="application/octet-stream")
        assert resp.status_code == 413

    def test_flask_accepts_undersized_body(self):
        app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_CONTENT_LENGTH
        app.config["TESTING"] = True

        @app.route("/echo", methods=["POST"])
        def echo():
            return jsonify({"len": len(request.get_data())})

        resp = app.test_client().post(
            "/echo", data=b"hello", content_type="application/octet-stream"
        )
        assert resp.status_code == 200
        assert resp.get_json()["len"] == 5


# --------------------------------------------------------------------------- #
# End-to-end: decorator registers errorhandler → 400 JSON
# --------------------------------------------------------------------------- #


class TestErrorHandlerIntegration:
    def test_validation_error_becomes_400(self):
        app = Flask(__name__)

        @app.errorhandler(ValidationError)
        def _handle(exc):
            return validation_error_response(exc)

        @app.route("/go", methods=["POST"])
        def go():
            data = request.json or {}
            clamp_int(data.get("horizon"), field="horizon_days",
                      default=7, min_value=0)
            return jsonify({"ok": True})

        client = app.test_client()

        # Negative value triggers ValidationError → 400.
        resp = client.post("/go", json={"horizon": -1})
        assert resp.status_code == 400
        payload = resp.get_json()
        assert payload["success"] is False
        assert payload["field"] == "horizon_days"

        # Above-cap value is silently clamped, not an error.
        resp = client.post("/go", json={"horizon": 999_999})
        assert resp.status_code == 200

        # Sensible value passes through.
        resp = client.post("/go", json={"horizon": 14})
        assert resp.status_code == 200
