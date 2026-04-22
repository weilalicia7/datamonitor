"""
Unified secrets loading (Production-Readiness T4.7).

SACT Scheduler reads secrets (Flask session key, API keys, seed passwords,
external-service tokens) from a **single entrypoint** so we can swap the
backend without touching every ``os.environ.get`` site:

    value = get_secret("FLASK_SECRET_KEY", required=True)

Resolution order (first non-empty wins):

1. **Process env** (``os.environ``) — the highest priority.  Operators
   who set a secret at the shell / systemd / k8s manifest level win.
2. **``.env`` file** loaded via python-dotenv (dev convenience only).
   The file is never committed (``.gitignore`` blocks it; pre-commit
   hook adds a second layer of defence).
3. **Configured backend** — when ``SECRETS_BACKEND=aws`` /
   ``SECRETS_BACKEND=vault``, we look up the secret in AWS Secrets
   Manager or HashiCorp Vault.  Library import is lazy; the app still
   boots without boto3 / hvac installed.
4. **Default value** — only for non-required secrets.

Fail-fast: :func:`assert_required_secrets_set` raises at startup if any
secret marked ``required=True`` cannot be resolved.  Catches "deployed
without FLASK_SECRET_KEY" misconfigurations before the first request.

Rotation semantics + operator runbook live in
``docs/SECRETS_ROTATION.md`` (linked from SECURITY.md).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# .env autoload (dev convenience)
# --------------------------------------------------------------------------- #


def load_dotenv_if_present(path: str = ".env", override: bool = False) -> bool:
    """Read ``path`` into ``os.environ`` if the file exists.

    Returns True if a file was actually loaded.  In production this is a
    no-op because ``.env`` isn't shipped.  The function never raises —
    a corrupt ``.env`` produces a warning log and is ignored so the app
    keeps booting from real environment variables.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        from dotenv import load_dotenv as _dotenv_load
    except Exception:                                # pragma: no cover
        logger.warning("secrets: .env present but python-dotenv not installed")
        return False
    try:
        _dotenv_load(dotenv_path=str(p), override=override)
        return True
    except Exception as exc:                         # pragma: no cover
        logger.warning("secrets: failed to parse .env (%s); ignoring", exc)
        return False


# --------------------------------------------------------------------------- #
# Backend selector
# --------------------------------------------------------------------------- #


def secrets_backend() -> str:
    """Which remote backend to consult (``env`` disables remote lookup)."""
    return (os.environ.get("SECRETS_BACKEND") or "env").strip().lower()


# --------------------------------------------------------------------------- #
# AWS Secrets Manager (optional, lazy)
# --------------------------------------------------------------------------- #

_aws_cache: Dict[str, str] = {}


def _load_from_aws(name: str) -> Optional[str]:     # pragma: no cover — requires boto3 + AWS creds
    if name in _aws_cache:
        return _aws_cache[name]
    try:
        import boto3
    except Exception:
        logger.warning("secrets: boto3 not installed; AWS backend unavailable")
        return None
    try:
        region = os.environ.get("AWS_REGION", "eu-west-2")
        prefix = os.environ.get("AWS_SECRETS_PREFIX", "sact/")
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=f"{prefix}{name}")
        value = resp.get("SecretString")
        if value is not None:
            _aws_cache[name] = value
        return value
    except Exception as exc:
        logger.warning("secrets: AWS lookup failed for %s: %s", name, exc)
        return None


# --------------------------------------------------------------------------- #
# HashiCorp Vault (optional, lazy)
# --------------------------------------------------------------------------- #

_vault_cache: Dict[str, str] = {}


def _load_from_vault(name: str) -> Optional[str]:   # pragma: no cover — requires hvac + VAULT_*
    if name in _vault_cache:
        return _vault_cache[name]
    try:
        import hvac
    except Exception:
        logger.warning("secrets: hvac not installed; Vault backend unavailable")
        return None
    addr = os.environ.get("VAULT_ADDR")
    token = os.environ.get("VAULT_TOKEN")
    if not addr or not token:
        logger.warning("secrets: VAULT_ADDR / VAULT_TOKEN not set")
        return None
    try:
        client = hvac.Client(url=addr, token=token)
        prefix = os.environ.get("VAULT_SECRETS_PATH", "secret/data/sact")
        read = client.read(f"{prefix}/{name}")
        if read is None:
            return None
        value = read.get("data", {}).get("data", {}).get("value")
        if value is not None:
            _vault_cache[name] = value
        return value
    except Exception as exc:
        logger.warning("secrets: Vault lookup failed for %s: %s", name, exc)
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class MissingSecretError(RuntimeError):
    """Raised at startup when a required secret cannot be resolved."""


def get_secret(
    name: str,
    default: Optional[str] = None,
    *,
    required: bool = False,
) -> Optional[str]:
    """Fetch a secret by name.

    Resolution: env > .env (if loaded) > remote backend > default.
    If ``required=True`` and no source provides a value, raises
    :class:`MissingSecretError`.  Non-required secrets return ``default``.
    """
    # 1. Process env (always authoritative).
    value = os.environ.get(name)
    if value:
        return value

    # 2. Remote backend (AWS/Vault).  .env has already been merged into
    # os.environ by ``load_dotenv_if_present``, so it is covered by step 1.
    backend = secrets_backend()
    if backend == "aws":
        value = _load_from_aws(name)
    elif backend == "vault":
        value = _load_from_vault(name)

    if value:
        # Cache the resolved value in the process env so subsequent lookups
        # short-circuit.  This is safe because workers don't fork after
        # secrets are resolved.
        os.environ[name] = value
        return value

    if required:
        raise MissingSecretError(
            f"Required secret '{name}' not set in env, .env, or backend "
            f"{backend!r}.  See docs/SECRETS_ROTATION.md."
        )
    return default


def assert_required_secrets_set(names: Iterable[str]) -> None:
    """Fail-fast wrapper: raise if any secret in ``names`` is missing."""
    missing: List[str] = []
    for n in names:
        try:
            v = get_secret(n, required=True)
            if not v:
                missing.append(n)
        except MissingSecretError:
            missing.append(n)
    if missing:
        raise MissingSecretError(
            f"Missing required secrets: {missing}.  See docs/SECRETS_ROTATION.md."
        )


# --------------------------------------------------------------------------- #
# Dev / prod detection helper
# --------------------------------------------------------------------------- #


def is_production_like() -> bool:
    """Crude heuristic: treat as 'prod-ish' when running under gunicorn or k8s.

    Used by the Flask bootstrap to decide whether missing secrets should
    fatal-out or just warn.
    """
    hints = (
        os.environ.get("GUNICORN_CMD_ARGS"),
        os.environ.get("KUBERNETES_SERVICE_HOST"),
        os.environ.get("SACT_PROD_MODE"),
    )
    if any(hints):
        return True
    return (os.environ.get("FLASK_ENV") or "").lower() == "production"
