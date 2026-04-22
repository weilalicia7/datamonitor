"""
Authentication + authorisation layer (Production-Readiness Plan T4.1).

Opt-in via ``AUTH_ENABLED`` env var — when unset/false, all decorators are
no-ops so the 441 existing tests keep working.  When ``AUTH_ENABLED=true``
the decorators enforce three roles:

* ``viewer``   — read-only (GET on most routes)
* ``operator`` — may mutate the schedule / run ML / commit config
* ``admin``    — everything the operator can do, plus policy-level config
                 (fairness / DRO / pentest / data-protection endpoints)

Two authentication paths are supported:

1. **Session auth** (browser) via Flask-Login.  Users log in at
   ``POST /auth/login`` with ``{"username":..., "password":...}``;
   session cookie is issued.
2. **API key auth** (machine-to-machine) via ``X-API-Key`` header.  Each
   key is bound to a user + role in the :data:`API_KEYS` store.

Both paths converge on :class:`User`, which exposes ``.role`` and the
standard Flask-Login interface (``is_authenticated``, ``get_id()``, etc.).

The user store is in-memory and seeded from environment variables:

* ``AUTH_SEED_ADMIN_PASSWORD``    (required when AUTH_ENABLED=true)
* ``AUTH_SEED_OPERATOR_PASSWORD`` (optional; creates 'operator' user)
* ``AUTH_SEED_VIEWER_PASSWORD``   (optional; creates 'viewer' user)
* ``AUTH_ADMIN_API_KEY`` / ``AUTH_OPERATOR_API_KEY`` / ``AUTH_VIEWER_API_KEY``

All passwords are stored as PBKDF2-HMAC-SHA256 hashes; API keys are stored
verbatim (rotate often).  See ``docs/PRODUCTION_READINESS_PLAN.md`` T4.7 for
how this will be superseded by a real secrets backend.
"""

from __future__ import annotations

import hmac
import hashlib
import os
import secrets
import threading
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set

from flask import current_app, g, jsonify, request, session

try:
    from flask_login import LoginManager, UserMixin, current_user, login_user, logout_user
    FLASK_LOGIN_AVAILABLE = True
except Exception:                              # pragma: no cover - optional dep
    FLASK_LOGIN_AVAILABLE = False

    class UserMixin:                           # type: ignore[no-redef]
        pass

    current_user = None                        # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

ROLE_VIEWER = "viewer"
ROLE_OPERATOR = "operator"
ROLE_ADMIN = "admin"

#: Role hierarchy — admin ⊇ operator ⊇ viewer.  `_has_role(u, "operator")`
#: is True iff the user's role is 'operator' or 'admin'.
_ROLE_RANK: Dict[str, int] = {
    ROLE_VIEWER:   1,
    ROLE_OPERATOR: 2,
    ROLE_ADMIN:    3,
}

ALL_ROLES: Set[str] = set(_ROLE_RANK.keys())


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------


@dataclass
class User(UserMixin):
    username: str
    role: str
    password_hash: Optional[str] = None      # PBKDF2 hash or None for API-key-only users
    password_salt: Optional[str] = None

    # Flask-Login interface -------------------------------------------------

    def get_id(self) -> str:                    # pragma: no cover
        return self.username

    @property
    def is_authenticated(self) -> bool:         # pragma: no cover
        return True

    @property
    def is_active(self) -> bool:                # pragma: no cover
        return True

    @property
    def is_anonymous(self) -> bool:             # pragma: no cover
        return False

    # Role helpers ----------------------------------------------------------

    def has_role(self, required: str) -> bool:
        return _has_role(self.role, required)


def _has_role(actual: str, required: str) -> bool:
    return _ROLE_RANK.get(actual, 0) >= _ROLE_RANK.get(required, 999)


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, cost = 200_000 iterations)
# ---------------------------------------------------------------------------

_HASH_ITERATIONS = 200_000
_HASH_ALGO = "sha256"


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Return ``(hex_hash, hex_salt)``.  ``salt`` is generated when None."""
    salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        _HASH_ALGO,
        password.encode("utf-8"),
        bytes.fromhex(salt),
        _HASH_ITERATIONS,
    )
    return derived.hex(), salt


def verify_password(password: str, password_hash: str, password_salt: str) -> bool:
    candidate, _ = hash_password(password, salt=password_salt)
    return hmac.compare_digest(candidate, password_hash)


# ---------------------------------------------------------------------------
# In-memory user store + API-key store
# ---------------------------------------------------------------------------

_USERS: Dict[str, User] = {}
_API_KEYS: Dict[str, str] = {}                # hashed_key -> username
_STORE_LOCK = threading.Lock()


def _api_key_digest(key: str) -> str:
    """HMAC-SHA256 hash of an API key for constant-time comparison."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def register_user(username: str, password: str, role: str = ROLE_VIEWER) -> User:
    if role not in ALL_ROLES:
        raise ValueError(f"unknown role: {role!r}")
    pw_hash, salt = hash_password(password)
    user = User(username=username, role=role,
                password_hash=pw_hash, password_salt=salt)
    with _STORE_LOCK:
        _USERS[username] = user
    return user


def register_api_key(username: str, api_key: str, role: str = ROLE_VIEWER) -> User:
    if role not in ALL_ROLES:
        raise ValueError(f"unknown role: {role!r}")
    digest = _api_key_digest(api_key)
    with _STORE_LOCK:
        if username not in _USERS:
            _USERS[username] = User(username=username, role=role)
        elif _USERS[username].role != role:
            # Upgrade / downgrade ok as long as explicit
            _USERS[username].role = role
        _API_KEYS[digest] = username
    return _USERS[username]


def get_user(username: str) -> Optional[User]:
    with _STORE_LOCK:
        return _USERS.get(username)


def clear_users() -> None:
    """Test helper — wipe the user + API-key stores."""
    with _STORE_LOCK:
        _USERS.clear()
        _API_KEYS.clear()


def _user_from_api_key(api_key: str) -> Optional[User]:
    digest = _api_key_digest(api_key)
    with _STORE_LOCK:
        username = _API_KEYS.get(digest)
        if username is None:
            return None
        return _USERS.get(username)


# ---------------------------------------------------------------------------
# AUTH_ENABLED flag
# ---------------------------------------------------------------------------


def auth_enabled() -> bool:
    """Single source of truth for whether to enforce auth.

    Default **False** so existing tests / dev workflows keep working.  Flip
    via env var ``AUTH_ENABLED=true`` (or any truthy variant) when running
    in production.
    """
    val = os.environ.get("AUTH_ENABLED", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def _current_identity() -> Optional[User]:
    """Resolve the current request's identity, trying API key first.

    Returns None if no credential was supplied.  Callers enforce 401/403.
    """
    # 1. API key via header  X-API-Key: <value>
    api_key = request.headers.get("X-API-Key") if request else None
    if api_key:
        user = _user_from_api_key(api_key)
        if user is not None:
            g.current_user = user
            return user
    # 2. Fall back to Flask-Login session
    if FLASK_LOGIN_AVAILABLE and current_user is not None and \
            current_user.is_authenticated:
        g.current_user = current_user
        return current_user          # type: ignore[return-value]
    return None


def login_required(view: Callable[..., Any]) -> Callable[..., Any]:
    """Require an authenticated user; 401 otherwise.

    When ``AUTH_ENABLED`` is off, acts as a no-op.
    """
    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any):
        if not auth_enabled():
            return view(*args, **kwargs)
        user = _current_identity()
        if user is None:
            return jsonify({
                "success": False,
                "error": "authentication required",
            }), 401
        return view(*args, **kwargs)
    return wrapper


def role_required(required: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Require the caller's role to be >= ``required`` (admin ⊇ operator ⊇ viewer)."""
    if required not in ALL_ROLES:
        raise ValueError(f"unknown role: {required!r}")

    def decorator(view: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(view)
        def wrapper(*args: Any, **kwargs: Any):
            if not auth_enabled():
                return view(*args, **kwargs)
            user = _current_identity()
            if user is None:
                return jsonify({
                    "success": False,
                    "error": "authentication required",
                }), 401
            if not user.has_role(required):
                return jsonify({
                    "success": False,
                    "error": f"role '{required}' required (current: '{user.role}')",
                }), 403
            return view(*args, **kwargs)
        return wrapper
    return decorator


def api_key_required(view: Callable[..., Any]) -> Callable[..., Any]:
    """Strict variant: require a valid ``X-API-Key`` header specifically.

    Useful for M2M endpoints that should not accept a session cookie.
    """
    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any):
        if not auth_enabled():
            return view(*args, **kwargs)
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({
                "success": False,
                "error": "X-API-Key header required",
            }), 401
        user = _user_from_api_key(api_key)
        if user is None:
            return jsonify({
                "success": False,
                "error": "invalid API key",
            }), 401
        g.current_user = user
        return view(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Seed helper for production boot
# ---------------------------------------------------------------------------


def seed_from_environment(logger=None) -> None:
    """Populate the user store from env vars if configured.

    Looks for:
        AUTH_SEED_ADMIN_PASSWORD       → 'admin'    (admin role)
        AUTH_SEED_OPERATOR_PASSWORD    → 'operator' (operator role)
        AUTH_SEED_VIEWER_PASSWORD      → 'viewer'   (viewer role)
        AUTH_ADMIN_API_KEY             → admin M2M key
        AUTH_OPERATOR_API_KEY          → operator M2M key
        AUTH_VIEWER_API_KEY            → viewer M2M key

    No-op when AUTH_ENABLED is false.  Warnings are logged for missing
    admin credentials in auth-enabled mode.
    """
    if not auth_enabled():
        return

    seeded = []

    # Password seeds
    for username, role in (
        ("admin",    ROLE_ADMIN),
        ("operator", ROLE_OPERATOR),
        ("viewer",   ROLE_VIEWER),
    ):
        env_var = f"AUTH_SEED_{username.upper()}_PASSWORD"
        pw = os.environ.get(env_var)
        if pw:
            register_user(username, pw, role=role)
            seeded.append(f"{username}@password")

    # API-key seeds
    for env_key, role in (
        ("AUTH_ADMIN_API_KEY",    ROLE_ADMIN),
        ("AUTH_OPERATOR_API_KEY", ROLE_OPERATOR),
        ("AUTH_VIEWER_API_KEY",   ROLE_VIEWER),
    ):
        key = os.environ.get(env_key)
        if key:
            username = env_key.split("_")[1].lower()
            register_api_key(username, key, role=role)
            seeded.append(f"{username}@apikey")

    if logger and seeded:
        logger.info("auth: seeded %d identities (%s)",
                    len(seeded), ", ".join(seeded))
    elif logger and not seeded:
        logger.warning(
            "AUTH_ENABLED=true but no identities seeded; every request "
            "will 401. Set AUTH_SEED_ADMIN_PASSWORD or AUTH_ADMIN_API_KEY."
        )


# ---------------------------------------------------------------------------
# LoginManager factory
# ---------------------------------------------------------------------------


def init_login_manager(app) -> Optional[Any]:
    """Wire Flask-Login into ``app`` if the dep is present.  Idempotent."""
    if not FLASK_LOGIN_AVAILABLE:
        return None
    lm = LoginManager()
    lm.login_view = "auth_login"
    lm.init_app(app)

    @lm.user_loader
    def _load(user_id: str) -> Optional[User]:
        return get_user(user_id)

    return lm
