"""
Tests for session_config.py — session cookie hardening + CSRF protection
(Production-Readiness T4.3).

Structure:
- TestCookieConfig       — apply_session_cookie_config + env overrides
- TestSameSiteNoneFixup  — SameSite=None must force Secure=True
- TestCSRFInit           — env-gating of CSRFProtect instantiation
- TestExemptJSONRoutes   — post-registration exemption of /api/*, /auth/*
- TestIntegration        — end-to-end: browser POST needs CSRF token,
                           API POST doesn't
"""

from __future__ import annotations

import pytest
from flask import Flask, jsonify, request

from session_config import (
    DEFAULT_EXEMPT_PREFIXES,
    DEFAULT_HTTPONLY,
    DEFAULT_SAMESITE,
    DEFAULT_SECURE,
    DEFAULT_SESSION_LIFETIME_SECONDS,
    apply_session_cookie_config,
    csrf_enabled,
    exempt_json_api_routes,
    init_csrf,
)


# --------------------------------------------------------------------------- #
# apply_session_cookie_config
# --------------------------------------------------------------------------- #


class TestCookieConfig:
    def test_defaults(self, monkeypatch):
        for v in (
            "SESSION_COOKIE_HTTPONLY", "SESSION_COOKIE_SAMESITE",
            "SESSION_COOKIE_SECURE", "SESSION_LIFETIME_SECONDS",
        ):
            monkeypatch.delenv(v, raising=False)
        app = Flask(__name__)
        out = apply_session_cookie_config(app)
        assert out["SESSION_COOKIE_HTTPONLY"] == DEFAULT_HTTPONLY
        assert out["SESSION_COOKIE_SAMESITE"] == DEFAULT_SAMESITE
        assert out["SESSION_COOKIE_SECURE"] == DEFAULT_SECURE
        assert out["PERMANENT_SESSION_LIFETIME"] == DEFAULT_SESSION_LIFETIME_SECONDS
        assert app.config["SESSION_COOKIE_HTTPONLY"] is True
        assert app.config["SESSION_COOKIE_SAMESITE"] == "Strict"

    def test_env_flip_secure(self, monkeypatch):
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")
        app = Flask(__name__)
        out = apply_session_cookie_config(app)
        assert out["SESSION_COOKIE_SECURE"] is True
        assert app.config["SESSION_COOKIE_SECURE"] is True

    def test_env_override_lifetime(self, monkeypatch):
        monkeypatch.setenv("SESSION_LIFETIME_SECONDS", "600")
        app = Flask(__name__)
        out = apply_session_cookie_config(app)
        assert out["PERMANENT_SESSION_LIFETIME"] == 600

    def test_invalid_lifetime_falls_back(self, monkeypatch):
        monkeypatch.setenv("SESSION_LIFETIME_SECONDS", "not-an-int")
        app = Flask(__name__)
        out = apply_session_cookie_config(app)
        assert out["PERMANENT_SESSION_LIFETIME"] == DEFAULT_SESSION_LIFETIME_SECONDS

    def test_samesite_choice_validated(self, monkeypatch):
        # Unknown SameSite value falls back to default.
        monkeypatch.setenv("SESSION_COOKIE_SAMESITE", "NotAReal")
        app = Flask(__name__)
        out = apply_session_cookie_config(app)
        assert out["SESSION_COOKIE_SAMESITE"] == DEFAULT_SAMESITE

    def test_samesite_lax_accepted(self, monkeypatch):
        monkeypatch.setenv("SESSION_COOKIE_SAMESITE", "Lax")
        app = Flask(__name__)
        out = apply_session_cookie_config(app)
        assert out["SESSION_COOKIE_SAMESITE"] == "Lax"


class TestSameSiteNoneFixup:
    def test_none_without_secure_forces_secure_on(self, monkeypatch):
        monkeypatch.setenv("SESSION_COOKIE_SAMESITE", "None")
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "false")
        app = Flask(__name__)
        out = apply_session_cookie_config(app)
        # Chromium rejects SameSite=None without Secure — we force Secure on.
        assert out["SESSION_COOKIE_SAMESITE"] == "None"
        assert out["SESSION_COOKIE_SECURE"] is True

    def test_none_with_secure_honored(self, monkeypatch):
        monkeypatch.setenv("SESSION_COOKIE_SAMESITE", "None")
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")
        app = Flask(__name__)
        out = apply_session_cookie_config(app)
        assert out["SESSION_COOKIE_SAMESITE"] == "None"
        assert out["SESSION_COOKIE_SECURE"] is True


# --------------------------------------------------------------------------- #
# init_csrf
# --------------------------------------------------------------------------- #


class TestCSRFInit:
    def test_csrf_enabled_default(self, monkeypatch):
        monkeypatch.delenv("CSRF_ENABLED", raising=False)
        assert csrf_enabled() is True

    def test_csrf_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("CSRF_ENABLED", "false")
        assert csrf_enabled() is False

    def test_init_csrf_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("CSRF_ENABLED", "false")
        app = Flask(__name__)
        app.secret_key = "test-key"
        assert init_csrf(app) is None

    def test_init_csrf_returns_protect_when_enabled(self, monkeypatch):
        pytest.importorskip("flask_wtf")
        monkeypatch.setenv("CSRF_ENABLED", "true")
        app = Flask(__name__)
        app.secret_key = "test-key"
        csrf = init_csrf(app)
        assert csrf is not None
        # The instance behaves like Flask-WTF's CSRFProtect.
        assert hasattr(csrf, "exempt")


# --------------------------------------------------------------------------- #
# exempt_json_api_routes
# --------------------------------------------------------------------------- #


class TestExemptJSONRoutes:
    def _build_app(self):
        app = Flask(__name__)
        app.secret_key = "test-key"

        @app.route("/api/foo", methods=["POST"])
        def api_foo():
            return jsonify({"ok": True})

        @app.route("/auth/login", methods=["POST"])
        def auth_login():
            return jsonify({"ok": True})

        @app.route("/health/live")
        def health():
            return jsonify({"ok": True})

        @app.route("/dashboard")
        def dashboard():
            return "<html></html>"

        @app.route("/report", methods=["POST"])
        def report():
            return "<html></html>"

        return app

    def test_exempts_expected_prefixes(self, monkeypatch):
        pytest.importorskip("flask_wtf")
        monkeypatch.setenv("CSRF_ENABLED", "true")
        app = self._build_app()
        csrf = init_csrf(app)
        exempted = exempt_json_api_routes(app, csrf)
        assert "/api/foo" in exempted
        assert "/auth/login" in exempted
        assert "/health/live" in exempted
        # Browser routes are NOT exempted.
        assert "/dashboard" not in exempted
        assert "/report" not in exempted

    def test_none_csrf_is_noop(self):
        app = self._build_app()
        assert exempt_json_api_routes(app, None) == []

    def test_custom_prefixes(self, monkeypatch):
        pytest.importorskip("flask_wtf")
        monkeypatch.setenv("CSRF_ENABLED", "true")
        app = self._build_app()
        csrf = init_csrf(app)
        exempted = exempt_json_api_routes(app, csrf, prefixes=("/api/",))
        assert "/api/foo" in exempted
        assert "/auth/login" not in exempted
        assert "/health/live" not in exempted

    def test_default_prefixes_cover_standard_surface(self):
        assert set(DEFAULT_EXEMPT_PREFIXES) >= {"/api/", "/auth/"}


# --------------------------------------------------------------------------- #
# End-to-end: form POST blocked without token, API POST allowed
# --------------------------------------------------------------------------- #


class TestIntegration:
    def _build_app(self, monkeypatch):
        pytest.importorskip("flask_wtf")
        monkeypatch.setenv("CSRF_ENABLED", "true")
        app = Flask(__name__)
        app.secret_key = "test-key"
        app.config["WTF_CSRF_ENABLED"] = True
        apply_session_cookie_config(app)

        @app.route("/api/echo", methods=["POST"])
        def api_echo():
            return jsonify({"ok": True, "body": request.get_json(silent=True) or {}})

        @app.route("/form/submit", methods=["POST"])
        def form_submit():
            return jsonify({"ok": True})

        csrf = init_csrf(app)
        exempt_json_api_routes(app, csrf)
        return app

    def test_api_post_allowed_without_csrf_token(self, monkeypatch):
        app = self._build_app(monkeypatch)
        client = app.test_client()
        resp = client.post("/api/echo", json={"x": 1})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True, "body": {"x": 1}}

    def test_form_post_blocked_without_csrf_token(self, monkeypatch):
        app = self._build_app(monkeypatch)
        client = app.test_client()
        resp = client.post("/form/submit", data={"hello": "world"})
        # CSRFProtect returns 400 by default on missing/invalid token.
        assert resp.status_code == 400

    def test_session_cookie_attributes_set_on_response(self, monkeypatch):
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "false")
        monkeypatch.setenv("SESSION_COOKIE_SAMESITE", "Strict")
        app = self._build_app(monkeypatch)

        @app.route("/set-session")
        def set_session():
            from flask import session
            session["x"] = 1
            return jsonify({"ok": True})

        resp = app.test_client().get("/set-session")
        # Flask sets a Set-Cookie header; extract the raw header value.
        cookies = [h for h in resp.headers.getlist("Set-Cookie") if "session=" in h]
        assert cookies, "expected a session cookie to be set"
        raw = cookies[0]
        assert "HttpOnly" in raw
        assert "SameSite=Strict" in raw
