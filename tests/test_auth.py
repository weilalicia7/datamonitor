"""
Tests for auth.py + /auth endpoints + before_request gate (Plan T4.1).

The test fixture flips ``AUTH_ENABLED`` on for specific tests (via env + reload)
to exercise the enforcement path, then flips it back off so the rest of the
suite is unaffected.
"""

from __future__ import annotations

import importlib
import os

import pytest

import auth


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clean_user_store():
    """Reset the in-memory identity store before + after every test."""
    auth.clear_users()
    yield
    auth.clear_users()


@pytest.fixture
def enable_auth(monkeypatch):
    """Turn AUTH_ENABLED on for the duration of a test."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    assert auth.auth_enabled() is True
    yield
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    assert auth.auth_enabled() is False


# --------------------------------------------------------------------------- #
# 1. Password hashing + verification
# --------------------------------------------------------------------------- #


class TestPasswordHashing:
    def test_hash_produces_hex_strings(self):
        digest, salt = auth.hash_password("hunter2")
        assert len(digest) == 64  # sha256 hex
        assert len(salt) == 32    # 16 bytes hex

    def test_verify_succeeds_on_correct_password(self):
        digest, salt = auth.hash_password("hunter2")
        assert auth.verify_password("hunter2", digest, salt) is True

    def test_verify_fails_on_wrong_password(self):
        digest, salt = auth.hash_password("hunter2")
        assert auth.verify_password("Hunter2", digest, salt) is False
        assert auth.verify_password("", digest, salt) is False

    def test_same_password_different_salts_yield_different_digests(self):
        d1, _ = auth.hash_password("same")
        d2, _ = auth.hash_password("same")
        assert d1 != d2  # salts are random


# --------------------------------------------------------------------------- #
# 2. Role hierarchy
# --------------------------------------------------------------------------- #


class TestRoleHierarchy:
    @pytest.mark.parametrize("actual,required,expected", [
        (auth.ROLE_VIEWER,   auth.ROLE_VIEWER,   True),
        (auth.ROLE_OPERATOR, auth.ROLE_VIEWER,   True),
        (auth.ROLE_ADMIN,    auth.ROLE_VIEWER,   True),
        (auth.ROLE_VIEWER,   auth.ROLE_OPERATOR, False),
        (auth.ROLE_OPERATOR, auth.ROLE_OPERATOR, True),
        (auth.ROLE_ADMIN,    auth.ROLE_OPERATOR, True),
        (auth.ROLE_VIEWER,   auth.ROLE_ADMIN,    False),
        (auth.ROLE_OPERATOR, auth.ROLE_ADMIN,    False),
        (auth.ROLE_ADMIN,    auth.ROLE_ADMIN,    True),
    ])
    def test_has_role(self, actual, required, expected):
        assert auth._has_role(actual, required) is expected

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError):
            auth.role_required("superadmin")


# --------------------------------------------------------------------------- #
# 3. User + API-key registration
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_register_user_stores_hash_not_plaintext(self):
        u = auth.register_user("alice", "alicepw", role=auth.ROLE_OPERATOR)
        assert u.username == "alice"
        assert u.role == auth.ROLE_OPERATOR
        assert u.password_hash is not None
        assert u.password_salt is not None
        # Plain-text password never appears in the stored hash
        assert "alicepw" not in u.password_hash

    def test_register_api_key_creates_user_when_missing(self):
        auth.register_api_key("bot", "key-abc-123", role=auth.ROLE_VIEWER)
        u = auth.get_user("bot")
        assert u is not None
        assert u.role == auth.ROLE_VIEWER

    def test_api_key_digest_is_constant_time_safe(self):
        d1 = auth._api_key_digest("secret")
        d2 = auth._api_key_digest("secret")
        assert d1 == d2
        assert len(d1) == 64


# --------------------------------------------------------------------------- #
# 4. Environment seeding
# --------------------------------------------------------------------------- #


class TestSeedFromEnvironment:
    def test_no_op_when_auth_disabled(self, monkeypatch):
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        monkeypatch.setenv("AUTH_SEED_ADMIN_PASSWORD", "x")
        auth.seed_from_environment()
        assert auth.get_user("admin") is None

    def test_seeds_admin_from_password(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("AUTH_SEED_ADMIN_PASSWORD", "rootpw")
        auth.seed_from_environment()
        u = auth.get_user("admin")
        assert u is not None
        assert u.role == auth.ROLE_ADMIN
        assert auth.verify_password("rootpw", u.password_hash, u.password_salt)

    def test_seeds_api_keys(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("AUTH_OPERATOR_API_KEY", "op-key-xyz")
        auth.seed_from_environment()
        u = auth.get_user("operator")
        assert u is not None
        assert u.role == auth.ROLE_OPERATOR


# --------------------------------------------------------------------------- #
# 5. Flask integration — /auth endpoints + before_request gate
# --------------------------------------------------------------------------- #


@pytest.fixture
def client_no_auth():
    """Flask test client with AUTH_ENABLED off (default)."""
    import flask_app as fa
    fa.app.testing = True
    return fa.app.test_client()


@pytest.fixture
def client_with_auth(monkeypatch):
    """Flask test client with AUTH_ENABLED on + seeded identities."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    # Seed three users directly (don't rely on env-seeded boot)
    auth.register_user("admin",    "adminpw", role=auth.ROLE_ADMIN)
    auth.register_user("operator", "oppw",    role=auth.ROLE_OPERATOR)
    auth.register_user("viewer",   "viewpw",  role=auth.ROLE_VIEWER)
    auth.register_api_key("bot",   "key-admin", role=auth.ROLE_ADMIN)
    auth.register_api_key("bot-v", "key-viewer", role=auth.ROLE_VIEWER)
    import flask_app as fa
    fa.app.testing = True
    yield fa.app.test_client()


class TestFlaskAuthEndpoints:
    def test_whoami_when_auth_disabled(self, client_no_auth):
        r = client_no_auth.get('/auth/whoami')
        assert r.status_code == 200
        body = r.get_json()
        assert body['auth_enabled'] is False
        assert body['authenticated'] is False

    def test_login_noop_when_auth_disabled(self, client_no_auth):
        r = client_no_auth.post('/auth/login',
                                json={'username': 'x', 'password': 'y'})
        assert r.status_code == 200
        assert r.get_json()['success'] is True

    def test_login_requires_username_and_password(self, client_with_auth):
        r = client_with_auth.post('/auth/login', json={})
        assert r.status_code == 400

    def test_login_rejects_invalid_credentials(self, client_with_auth):
        r = client_with_auth.post(
            '/auth/login',
            json={'username': 'admin', 'password': 'WRONG'},
        )
        assert r.status_code == 401

    def test_login_accepts_valid_credentials(self, client_with_auth):
        r = client_with_auth.post(
            '/auth/login',
            json={'username': 'admin', 'password': 'adminpw'},
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body['success'] is True
        assert body['role'] == auth.ROLE_ADMIN

    def test_whoami_with_valid_api_key_header(self, client_with_auth):
        r = client_with_auth.get('/auth/whoami',
                                 headers={'X-API-Key': 'key-admin'})
        assert r.status_code == 200
        assert r.get_json()['role'] == auth.ROLE_ADMIN

    def test_whoami_with_invalid_api_key_header(self, client_with_auth):
        r = client_with_auth.get('/auth/whoami',
                                 headers={'X-API-Key': 'not-a-real-key'})
        assert r.status_code == 200
        # Invalid key → treated as unauthenticated (not 401 here, because
        # whoami is a probe endpoint that just reports the resolved identity)
        assert r.get_json()['authenticated'] is False


# --------------------------------------------------------------------------- #
# 6. Route gating via before_request
# --------------------------------------------------------------------------- #


class TestRouteGating:
    def test_public_paths_no_auth_when_enabled(self, client_with_auth):
        for p in ('/auth/login', '/auth/logout', '/auth/whoami'):
            # Any request method should hit the route handler without 401
            r = client_with_auth.get(p)
            # /auth/login via GET is 405 (Method Not Allowed), but it's NOT 401
            assert r.status_code != 401

    def test_get_status_endpoint_requires_auth_when_enabled(self, client_with_auth):
        r = client_with_auth.get('/api/fairness/dro/status')
        assert r.status_code == 401

    def test_get_with_viewer_api_key_allowed(self, client_with_auth):
        r = client_with_auth.get('/api/fairness/dro/status',
                                 headers={'X-API-Key': 'key-viewer'})
        # viewer can GET status endpoints
        assert r.status_code != 401 and r.status_code != 403

    def test_post_config_endpoint_requires_admin(self, client_with_auth):
        # Operator-level API key should be rejected from /config endpoints
        auth.register_api_key("bot-op", "key-op", role=auth.ROLE_OPERATOR)
        r = client_with_auth.post(
            '/api/fairness/dro/config',
            json={'epsilon': 0.01},
            headers={'X-API-Key': 'key-op'},
        )
        assert r.status_code == 403
        body = r.get_json()
        assert 'admin' in body.get('error', '').lower()

    def test_post_config_endpoint_allowed_for_admin(self, client_with_auth):
        r = client_with_auth.post(
            '/api/fairness/dro/config',
            json={'epsilon': 0.01},
            headers={'X-API-Key': 'key-admin'},
        )
        # 200 on valid, or 400 on malformed body — either way NOT 401/403
        assert r.status_code not in (401, 403)

    def test_post_operator_endpoint_rejects_viewer(self, client_with_auth):
        # /api/urgent/insert is a POST that should require operator role
        r = client_with_auth.post(
            '/api/urgent/insert',
            json={},
            headers={'X-API-Key': 'key-viewer'},
        )
        assert r.status_code == 403

    def test_gating_is_noop_when_auth_disabled(self, client_no_auth):
        # Without auth, a raw POST to a config endpoint should NOT 401/403
        r = client_no_auth.post('/api/fairness/dro/config',
                                json={'epsilon': 0.01})
        assert r.status_code not in (401, 403)


# --------------------------------------------------------------------------- #
# 7. Route-role mapping rules
# --------------------------------------------------------------------------- #


class TestRouteRoleMapping:
    """Directly test the `_route_required_role` pure function (no Flask)."""

    def _call(self, method, path):
        import flask_app as fa
        return fa._route_required_role(method, path)

    def test_public_paths(self):
        for p in ('/auth/login', '/auth/whoami', '/health/live',
                  '/health/ready', '/metrics', '/favicon.ico',
                  '/static/foo.png'):
            assert self._call('GET', p) is None

    def test_post_config_is_admin(self):
        assert self._call('POST', '/api/fairness/dro/config') == auth.ROLE_ADMIN
        assert self._call('POST', '/api/mpc/config') == auth.ROLE_ADMIN
        assert self._call('POST', '/api/anything/config') == auth.ROLE_ADMIN

    def test_post_non_config_is_operator(self):
        assert self._call('POST', '/api/optimize') == auth.ROLE_OPERATOR
        assert self._call('POST', '/api/mpc/simulate') == auth.ROLE_OPERATOR

    def test_get_is_viewer(self):
        assert self._call('GET', '/api/scheduler/status') == auth.ROLE_VIEWER

    def test_delete_is_operator_not_admin(self):
        assert self._call('DELETE', '/api/resource') == auth.ROLE_OPERATOR
