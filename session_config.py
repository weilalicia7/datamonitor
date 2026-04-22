"""
Session cookie hardening + CSRF protection (Production-Readiness T4.3).

Two concerns:

1. **Session cookie attributes** — set sensible defaults that browsers enforce:

   - ``HTTPONLY=True``  → JavaScript can't read ``document.cookie``, so stolen
     XSS reads can't lift the session token.
   - ``SAMESITE=Strict`` → browsers never send the cookie on cross-site
     navigation, killing classical CSRF via top-level navigation.
   - ``SECURE=True``   → cookies only travel over HTTPS.  Off by default for
     local development; flip on via ``SESSION_COOKIE_SECURE=true`` in prod.
   - ``PERMANENT_SESSION_LIFETIME=1800`` (30 min idle) → bounded exposure
     window for a compromised session.

2. **CSRF** — ``Flask-WTF`` ``CSRFProtect`` is enabled by default so any
   future browser-form endpoint is protected out of the box.  JSON-first
   endpoints under ``/api/*`` and ``/auth/*`` are exempted post-hoc via
   :func:`exempt_json_api_routes`; they authenticate via API key or session
   and, because browsers forbid cross-origin ``application/json`` submits
   without CORS preflight (which we don't enable), they aren't reachable
   by a classical CSRF form-submit attack.

Every setting is env-var configurable so operators can tighten or relax
the defaults without a code change.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Env-var helpers
# --------------------------------------------------------------------------- #


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_choice(name: str, default: str, allowed: Iterable[str]) -> str:
    raw = os.environ.get(name, default).strip()
    if raw not in set(allowed):
        return default
    return raw


# --------------------------------------------------------------------------- #
# Defaults (single source of truth)
# --------------------------------------------------------------------------- #

#: SameSite policy.  ``Strict`` is the safest — never sends the cookie on
#: cross-site requests.  ``Lax`` permits top-level GET navigation (common
#: pattern for OAuth redirects).  ``None`` requires ``SECURE=True`` and
#: allows cross-site use (iframes, SaaS dashboards); too loose for default.
DEFAULT_SAMESITE: str = "Strict"

#: 30-minute idle session TTL.  Tightenable via env var for clinical /
#: financial settings; relaxable for staff portals where 8-hour shifts
#: would re-auth repeatedly.
DEFAULT_SESSION_LIFETIME_SECONDS: int = 30 * 60

#: ``HTTPONLY`` should always be True — there is no sane reason to expose
#: the session cookie to JavaScript.  Env-gated only for test use cases
#: that mock the cookie jar.
DEFAULT_HTTPONLY: bool = True

#: ``SECURE`` off by default because local dev runs on plain HTTP.  All
#: production deployments MUST set ``SESSION_COOKIE_SECURE=true`` (or
#: equivalent via the infra secrets layer).
DEFAULT_SECURE: bool = False

#: CSRF protection on by default — fail-safe.
DEFAULT_CSRF_ENABLED: bool = True

#: Prefixes we exempt from CSRF after all routes are registered.
#: - ``/api/*`` — JSON endpoints; API-key auth; cross-origin JSON submit is
#:   blocked by browsers without CORS preflight.
#: - ``/auth/*`` — login/logout/whoami; JSON only.
#: - ``/health`` / ``/metrics`` — unauth probes; no state change.
DEFAULT_EXEMPT_PREFIXES: Tuple[str, ...] = ("/api/", "/auth/", "/health", "/metrics")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def apply_session_cookie_config(app) -> dict:
    """Write session-cookie hardening config into ``app.config``.

    Returns the dict of values actually written (useful for logging).
    """
    written = {
        "SESSION_COOKIE_HTTPONLY": _env_bool(
            "SESSION_COOKIE_HTTPONLY", DEFAULT_HTTPONLY
        ),
        "SESSION_COOKIE_SAMESITE": _env_choice(
            "SESSION_COOKIE_SAMESITE", DEFAULT_SAMESITE, {"Strict", "Lax", "None"}
        ),
        "SESSION_COOKIE_SECURE": _env_bool(
            "SESSION_COOKIE_SECURE", DEFAULT_SECURE
        ),
        "PERMANENT_SESSION_LIFETIME": _env_int(
            "SESSION_LIFETIME_SECONDS", DEFAULT_SESSION_LIFETIME_SECONDS
        ),
    }
    # ``SameSite=None`` without ``Secure=True`` is rejected by Chromium —
    # correct the combination here rather than surface a confusing browser
    # error at first login.
    if (
        written["SESSION_COOKIE_SAMESITE"] == "None"
        and not written["SESSION_COOKIE_SECURE"]
    ):
        logger.warning(
            "session: SameSite=None requires Secure=True; forcing Secure on."
        )
        written["SESSION_COOKIE_SECURE"] = True

    for key, value in written.items():
        app.config[key] = value
    return written


def csrf_enabled() -> bool:
    return _env_bool("CSRF_ENABLED", DEFAULT_CSRF_ENABLED)


def init_csrf(app):
    """Attach Flask-WTF's ``CSRFProtect`` iff the library is available.

    Returns the ``CSRFProtect`` instance (or ``None`` if disabled / missing).
    The caller uses the instance later — e.g. to call
    :func:`exempt_json_api_routes` or to opt-in-protect a browser form.
    """
    if not csrf_enabled():
        return None
    try:
        from flask_wtf.csrf import CSRFProtect
    except Exception:                                  # pragma: no cover
        logger.warning("csrf: flask_wtf not installed — CSRF protection OFF")
        return None

    csrf = CSRFProtect(app)
    return csrf


def exempt_json_api_routes(
    app,
    csrf,
    prefixes: Optional[Iterable[str]] = None,
) -> List[str]:
    """Exempt every view whose URL matches an API prefix from CSRF checks.

    Call this AFTER all routes are registered.  Returns the list of matched
    rule strings (for logging / test assertions).
    """
    if csrf is None:
        return []
    pfx = tuple(prefixes) if prefixes is not None else DEFAULT_EXEMPT_PREFIXES

    matched: List[str] = []
    seen_endpoints = set()
    for rule in app.url_map.iter_rules():
        if any(rule.rule.startswith(p) for p in pfx):
            if rule.endpoint in seen_endpoints:
                continue
            seen_endpoints.add(rule.endpoint)
            view = app.view_functions.get(rule.endpoint)
            if view is not None:
                csrf.exempt(view)
                matched.append(rule.rule)
    return matched
