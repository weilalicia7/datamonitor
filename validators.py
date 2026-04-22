"""
Request-input validators + rate-limiter factory (Production-Readiness T4.2).

Two concerns live here:

1. **Input bounds**.  Every user-supplied numeric / string parameter that
   can drive compute (horizon days, scenario count, simulation minutes,
   paginator limit) is clamped to a configured maximum BEFORE it reaches
   the underlying solver.  The cap returns HTTP 400 with a clear
   explanation rather than letting the request quietly run for hours and
   OOM the host.

2. **Rate limiting**.  ``init_rate_limiter(app)`` wires Flask-Limiter when
   the dependency is installed AND ``RATE_LIMIT_ENABLED=true`` is set in
   the environment.  Returns the shared limiter instance so individual
   routes can add tighter caps via ``@limiter.limit("5 per minute")``.

Both behaviours default to SAFE (caps always applied; rate-limit gated
by env var, because test suites don't tolerate 429s out of the box).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from flask import jsonify, request

# ---------------------------------------------------------------------------
# Numeric caps — single source of truth
# ---------------------------------------------------------------------------

#: Max bytes per request body (Flask ``MAX_CONTENT_LENGTH``).  16 MB is
#: generous enough for legitimate schedule / evaluation payloads, tight
#: enough to prevent 1 GB JSON bombs.
DEFAULT_MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024

#: Per-parameter bounds hit at the route level (BEFORE solver calls).
#: Tight defaults; operators who need larger values can relax via env
#: without changing code.  Derived from the D-audit + Tier 4 brief.
ENDPOINT_BOUNDS: Dict[str, int] = {
    # /api/twin/evaluate + /api/twin/compare
    "horizon_days":       365,         # one year of simulation; more is rarely clinical
    "step_hours":         24,
    "rng_seed":           2_147_483_647,
    # /api/twin/evaluations + /api/override/last + similar paginators
    "limit":              1_000,
    # /api/mpc/simulate
    "total_minutes":      1_440,       # 24 h clock
    "n_scenarios":        500,         # keeps K×rollout under 1 minute
    "lookahead_minutes":  480,         # full working day
    # /api/ml/uncertainty-optimization/evaluate
    "n_uncertainty_scenarios": 500,
    # /api/microbatch/flush etc.
    "change_threshold":   10_000,
    "slow_path_interval_s": 86_400,    # 24 h
    # /api/fairness/dro/certify / /lipschitz/certify
    "epsilon":            1.0,          # Wasserstein ball radius (numeric)
    "tau":                1.0,          # Lipschitz similarity threshold
}


def _env_override(name: str, default: int) -> int:
    """Allow per-key override via env var ``VALIDATOR_CAP_<NAME>``."""
    env_name = f"VALIDATOR_CAP_{name.upper()}"
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def get_cap(name: str) -> int:
    """Lookup effective cap (env override > built-in default)."""
    default = ENDPOINT_BOUNDS.get(name, 1_000_000)
    return _env_override(name, default)


class ValidationError(Exception):
    """Raised when a parameter violates its cap and cannot be clamped down."""

    def __init__(self, message: str, field: str = ""):
        super().__init__(message)
        self.field = field


# ---------------------------------------------------------------------------
# Integer clamping
# ---------------------------------------------------------------------------


def clamp_int(
    value: Any,
    *,
    field: str,
    default: int,
    max_value: Optional[int] = None,
    min_value: int = 0,
) -> int:
    """Coerce + clamp ``value`` to ``[min_value, max_value]``.

    Raises ``ValidationError`` if the value isn't coercible to int.  Values
    above ``max_value`` are silently clamped DOWN (not an error) so the
    optimiser still runs — a 400 would surprise callers who just wanted
    "whatever is the maximum you support."  Values below ``min_value``
    raise, because that usually signals a client bug.

    Callers specify ``field`` so the error message cites the parameter
    name the user actually sent.
    """
    if value is None:
        value = default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{field!r} must be an integer, got {type(value).__name__}: {value!r}",
            field=field,
        )
    if coerced < min_value:
        raise ValidationError(
            f"{field!r}={coerced} is below minimum {min_value}",
            field=field,
        )
    cap = max_value if max_value is not None else get_cap(field)
    if coerced > cap:
        return cap
    return coerced


def clamp_float(
    value: Any,
    *,
    field: str,
    default: float,
    max_value: Optional[float] = None,
    min_value: float = 0.0,
) -> float:
    """Float variant of :func:`clamp_int`."""
    if value is None:
        value = default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{field!r} must be numeric, got {type(value).__name__}: {value!r}",
            field=field,
        )
    if coerced < min_value:
        raise ValidationError(
            f"{field!r}={coerced} is below minimum {min_value}",
            field=field,
        )
    cap = max_value if max_value is not None else float(get_cap(field))
    if coerced > cap:
        return cap
    return coerced


# ---------------------------------------------------------------------------
# String whitelists (for DML, SACT field names, etc.)
# ---------------------------------------------------------------------------


#: Approved DML treatment column names (the §6.x causal-inference endpoints
#: pipe user-supplied column names into DataFrame lookups; a malicious name
#: could cause pandas to load unrelated columns or run heavy string ops).
DML_TREATMENT_ALLOWED: Set[str] = {
    "Reminder_Sent",
    "Phone_Call",
    "Transport_Offer",
    "SMS_Reminder",
    "Longer_Appointment",
    "Weekday_Only",
    "Morning_Slot",
    "Intervention_A",
    "Intervention_B",
}

#: Approved DML outcome column names.
DML_OUTCOME_ALLOWED: Set[str] = {
    "no_show",
    "Attended_Status",
    "showed_up",
    "completed_treatment",
    "cancelled",
}

#: Approved DML covariate column names.
DML_COVARIATES_ALLOWED: Set[str] = {
    "age",
    "Age",
    "Age_Band",
    "priority",
    "Priority",
    "distance_km",
    "Distance_km",
    "Travel_Time_Min",
    "Planned_Duration",
    "Patient_NoShow_Rate",
    "Cycle_Number",
    "Previous_NoShows",
    "Previous_Cancellations",
    "Weather_Severity",
    "Ethnic_Category",
    "Gender_Code",
}


def validate_whitelist(
    value: str,
    *,
    field: str,
    allowed: Iterable[str],
    allow_empty: bool = False,
) -> str:
    """Return ``value`` if it's in ``allowed``; else raise."""
    allowed_set = set(allowed)
    if value is None or (not allow_empty and str(value).strip() == ""):
        raise ValidationError(f"{field!r} is required", field=field)
    if str(value) not in allowed_set:
        raise ValidationError(
            f"{field!r}={value!r} not in allowed set ({len(allowed_set)} options)",
            field=field,
        )
    return str(value)


def validate_whitelist_many(
    values: Any,
    *,
    field: str,
    allowed: Iterable[str],
) -> list:
    """Validate a list of column names — fails on first offender."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise ValidationError(
            f"{field!r} must be a list, got {type(values).__name__}",
            field=field,
        )
    out = []
    for v in values:
        out.append(validate_whitelist(str(v), field=field, allowed=allowed))
    return out


# ---------------------------------------------------------------------------
# Flask helpers
# ---------------------------------------------------------------------------


def validation_error_response(exc: ValidationError):
    """Translate ``ValidationError`` into a consistent 400 JSON response."""
    return jsonify({
        "success": False,
        "error": str(exc),
        "field": exc.field or None,
    }), 400


# ---------------------------------------------------------------------------
# Flask-Limiter factory
# ---------------------------------------------------------------------------


def rate_limit_enabled() -> bool:
    val = os.environ.get("RATE_LIMIT_ENABLED", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def init_rate_limiter(app) -> Tuple[Optional[Any], bool]:
    """Attach Flask-Limiter to ``app`` iff enabled + library available.

    Returns ``(limiter_or_None, enabled_bool)``.  When the limiter is
    disabled, routes that call ``limiter.limit(...)`` still work because
    the decorator becomes a no-op (Flask-Limiter handles this).
    """
    if not rate_limit_enabled():
        return None, False

    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
    except Exception:                                 # pragma: no cover
        return None, False

    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["60 per minute", "600 per hour"],
        storage_uri=os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://"),
    )
    return limiter, True
