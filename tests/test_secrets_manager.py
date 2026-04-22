"""
Tests for secrets_manager.py — unified secret loading + rotation-safe API
(Production-Readiness T4.7).

Structure:
- TestDotenv                — .env autoload; missing file returns False
- TestGetSecretFromEnv      — env-var hit is authoritative
- TestGetSecretMissing      — default, required=False, required=True paths
- TestBackendSelector       — secrets_backend() parses env correctly
- TestAssertRequired        — raises on missing, passes on present
- TestIsProductionLike      — env-hint detection
- TestBackendLookupSkipped  — backend=env short-circuits remote calls
"""

from __future__ import annotations

import os

import pytest

import secrets_manager as sm


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    # Drop module-level caches so tests don't observe each other's writes.
    sm._aws_cache.clear()
    sm._vault_cache.clear()
    # Clear any prior test-injected env var that could survive through fixtures.
    for k in list(os.environ):
        if k.startswith("TEST_SECRET_"):
            monkeypatch.delenv(k, raising=False)
    yield
    sm._aws_cache.clear()
    sm._vault_cache.clear()


# --------------------------------------------------------------------------- #
# .env autoload
# --------------------------------------------------------------------------- #


class TestDotenv:
    def test_missing_returns_false(self, tmp_path):
        non_existent = str(tmp_path / "never_exists.env")
        assert sm.load_dotenv_if_present(non_existent) is False

    def test_loads_values_into_env(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_SECRET_FROM_FILE=value-from-dotenv\n")
        monkeypatch.delenv("TEST_SECRET_FROM_FILE", raising=False)
        assert sm.load_dotenv_if_present(str(env_file)) is True
        assert os.environ["TEST_SECRET_FROM_FILE"] == "value-from-dotenv"

    def test_respects_override_false(self, tmp_path, monkeypatch):
        # When override=False (default), existing env wins over .env.
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_SECRET_COLLIDE=from-file\n")
        monkeypatch.setenv("TEST_SECRET_COLLIDE", "from-env")
        sm.load_dotenv_if_present(str(env_file), override=False)
        assert os.environ["TEST_SECRET_COLLIDE"] == "from-env"

    def test_override_true_replaces_env(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_SECRET_OVERRIDE=from-file\n")
        monkeypatch.setenv("TEST_SECRET_OVERRIDE", "from-env")
        sm.load_dotenv_if_present(str(env_file), override=True)
        assert os.environ["TEST_SECRET_OVERRIDE"] == "from-file"


# --------------------------------------------------------------------------- #
# get_secret
# --------------------------------------------------------------------------- #


class TestGetSecretFromEnv:
    def test_env_wins(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_X", "from-env")
        assert sm.get_secret("TEST_SECRET_X") == "from-env"

    def test_stripped_empty_is_missing(self, monkeypatch):
        # Empty env value should NOT satisfy a required lookup.
        monkeypatch.setenv("TEST_SECRET_EMPTY", "")
        with pytest.raises(sm.MissingSecretError):
            sm.get_secret("TEST_SECRET_EMPTY", required=True)


class TestGetSecretMissing:
    def test_returns_default_when_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_SECRET_MISSING", raising=False)
        assert sm.get_secret("TEST_SECRET_MISSING", default="fallback") == "fallback"

    def test_returns_none_by_default(self, monkeypatch):
        monkeypatch.delenv("TEST_SECRET_MISSING", raising=False)
        assert sm.get_secret("TEST_SECRET_MISSING") is None

    def test_required_raises(self, monkeypatch):
        monkeypatch.delenv("TEST_SECRET_MUST_EXIST", raising=False)
        monkeypatch.setenv("SECRETS_BACKEND", "env")
        with pytest.raises(sm.MissingSecretError) as exc:
            sm.get_secret("TEST_SECRET_MUST_EXIST", required=True)
        assert "TEST_SECRET_MUST_EXIST" in str(exc.value)
        assert "SECRETS_ROTATION" in str(exc.value)


class TestBackendSelector:
    def test_default_is_env(self, monkeypatch):
        monkeypatch.delenv("SECRETS_BACKEND", raising=False)
        assert sm.secrets_backend() == "env"

    def test_aws_parsed(self, monkeypatch):
        monkeypatch.setenv("SECRETS_BACKEND", "AWS")
        assert sm.secrets_backend() == "aws"

    def test_vault_parsed(self, monkeypatch):
        monkeypatch.setenv("SECRETS_BACKEND", "vault")
        assert sm.secrets_backend() == "vault"


class TestAssertRequired:
    def test_passes_when_all_present(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_A", "a")
        monkeypatch.setenv("TEST_SECRET_B", "b")
        sm.assert_required_secrets_set(["TEST_SECRET_A", "TEST_SECRET_B"])

    def test_raises_when_any_missing(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_A", "a")
        monkeypatch.delenv("TEST_SECRET_B", raising=False)
        with pytest.raises(sm.MissingSecretError) as exc:
            sm.assert_required_secrets_set(["TEST_SECRET_A", "TEST_SECRET_B"])
        assert "TEST_SECRET_B" in str(exc.value)
        assert "TEST_SECRET_A" not in str(exc.value)


class TestIsProductionLike:
    def test_default_not_prod(self, monkeypatch):
        for v in (
            "GUNICORN_CMD_ARGS", "KUBERNETES_SERVICE_HOST",
            "SACT_PROD_MODE", "FLASK_ENV",
        ):
            monkeypatch.delenv(v, raising=False)
        assert sm.is_production_like() is False

    def test_gunicorn_hint(self, monkeypatch):
        for v in (
            "KUBERNETES_SERVICE_HOST", "SACT_PROD_MODE", "FLASK_ENV",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("GUNICORN_CMD_ARGS", "--workers 4")
        assert sm.is_production_like() is True

    def test_k8s_hint(self, monkeypatch):
        for v in ("GUNICORN_CMD_ARGS", "SACT_PROD_MODE", "FLASK_ENV"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        assert sm.is_production_like() is True

    def test_flask_env_production(self, monkeypatch):
        for v in (
            "GUNICORN_CMD_ARGS", "KUBERNETES_SERVICE_HOST", "SACT_PROD_MODE",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("FLASK_ENV", "production")
        assert sm.is_production_like() is True

    def test_flask_env_dev_is_not(self, monkeypatch):
        for v in (
            "GUNICORN_CMD_ARGS", "KUBERNETES_SERVICE_HOST", "SACT_PROD_MODE",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("FLASK_ENV", "development")
        assert sm.is_production_like() is False


class TestBackendLookupSkipped:
    """When SECRETS_BACKEND=env, the AWS/Vault helpers must NOT be called.

    Guards against a misconfiguration where a non-prod container accidentally
    hits AWS Secrets Manager with no IAM perms.
    """

    def test_env_backend_does_not_invoke_aws(self, monkeypatch):
        calls = {"aws": 0, "vault": 0}

        def _fail_aws(name):
            calls["aws"] += 1
            return None

        def _fail_vault(name):
            calls["vault"] += 1
            return None

        monkeypatch.setattr(sm, "_load_from_aws", _fail_aws)
        monkeypatch.setattr(sm, "_load_from_vault", _fail_vault)
        monkeypatch.setenv("SECRETS_BACKEND", "env")
        monkeypatch.delenv("TEST_SECRET_SKIP", raising=False)
        assert sm.get_secret("TEST_SECRET_SKIP") is None
        assert calls == {"aws": 0, "vault": 0}

    def test_aws_backend_invoked_when_env_missing(self, monkeypatch):
        def _stub_aws(name):
            return f"aws-value-for-{name}"

        monkeypatch.setattr(sm, "_load_from_aws", _stub_aws)
        monkeypatch.setenv("SECRETS_BACKEND", "aws")
        monkeypatch.delenv("TEST_SECRET_AWS_LOAD", raising=False)
        value = sm.get_secret("TEST_SECRET_AWS_LOAD")
        assert value == "aws-value-for-TEST_SECRET_AWS_LOAD"
        # The resolved value is cached back into os.environ.
        assert os.environ["TEST_SECRET_AWS_LOAD"] == value

    def test_vault_backend_invoked_when_selected(self, monkeypatch):
        def _stub_vault(name):
            return f"vault-value-for-{name}"

        monkeypatch.setattr(sm, "_load_from_vault", _stub_vault)
        monkeypatch.setenv("SECRETS_BACKEND", "vault")
        monkeypatch.delenv("TEST_SECRET_VAULT_LOAD", raising=False)
        value = sm.get_secret("TEST_SECRET_VAULT_LOAD")
        assert value == "vault-value-for-TEST_SECRET_VAULT_LOAD"
