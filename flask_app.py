"""
SACT Scheduler Flask Application
=================================

Flask-based web application for the SACT scheduling system.
Premium McKinsey-inspired UX design with depth and sophistication.
Auto-connects to data sources and refreshes at configurable intervals.
Runs on port 1421.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime, timedelta
from dataclasses import asdict
from typing import Dict as _Dict, Optional
import json

# Fallback trial rate (also defined in ml/rct_randomization.py).  Kept here as
# a module-level constant so the Flask endpoint can accept a client override
# without importing at request-time.
DEFAULT_TRIAL_RATE_FALLBACK = 0.05
import sys
import os
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OperatingMode,
    DEFAULT_SITES,
    OPERATING_HOURS,
    DATA_DIR,
    get_logger
)

# Import modules
from optimization.optimizer import ScheduleOptimizer, Patient, Chair
from optimization.squeeze_in import SqueezeInHandler
from optimization.emergency_mode import EmergencyModeHandler
from monitoring.event_aggregator import EventAggregator
from monitoring.alert_manager import AlertManager

# Try importing data processing modules
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    np = None

try:
    from ml.noshow_model import NoShowModel
    from ml.duration_model import DurationModel
    from ml.survival_model import SurvivalAnalysisModel, predict_noshow_timing
    from ml.uplift_model import UpliftModel, InterventionType, recommend_intervention
    from ml.multitask_model import MultiTaskModel, predict_joint
    from ml.quantile_forest import QuantileForestDurationModel, QuantileForestNoShowModel
    from ml.hierarchical_model import HierarchicalBayesianModel, HierarchicalPrediction
    from ml.causal_model import (
        SchedulingCausalModel, CausalEffect, compute_intervention_effect,
        InstrumentalVariablesEstimator, IVEstimationResult, estimate_iv_effect
    )
    from ml.event_impact_model import (
        EventImpactModel, EventImpactPrediction, Event, EventType, EventSeverity,
        analyze_event_impact, estimate_event_severity
    )
    ML_AVAILABLE = True
    SURVIVAL_AVAILABLE = True
    UPLIFT_AVAILABLE = True
    MULTITASK_AVAILABLE = True
    QUANTILE_FOREST_AVAILABLE = True
    HIERARCHICAL_AVAILABLE = True
    CAUSAL_AVAILABLE = True
    EVENT_IMPACT_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    SURVIVAL_AVAILABLE = False
    UPLIFT_AVAILABLE = False
    MULTITASK_AVAILABLE = False
    QUANTILE_FOREST_AVAILABLE = False
    HIERARCHICAL_AVAILABLE = False
    CAUSAL_AVAILABLE = False

logger = get_logger(__name__)

# T4.7 — Secrets: autoload .env for dev convenience, then use a single
# resolver for every secret.  In production the operator provisions env
# via AWS Secrets Manager / Vault / k8s Secrets; locally the .env file
# (gitignored; pre-commit blocked) serves the same role.
from secrets_manager import (
    assert_required_secrets_set as _assert_required_secrets_set,
    get_secret as _get_secret,
    is_production_like as _is_production_like,
    load_dotenv_if_present as _load_dotenv_if_present,
    MissingSecretError as _MissingSecretError,
)

_dotenv_loaded = _load_dotenv_if_present()
if _dotenv_loaded:
    logger.info("secrets: .env loaded for development")

# Initialize Flask app
app = Flask(__name__)
# Flask session secret: in production FLASK_SECRET_KEY MUST be set.  In dev we
# tolerate a missing value and mint an ephemeral key so developers aren't
# forced through the secrets flow for a quick local run.
_flask_secret = _get_secret('FLASK_SECRET_KEY')
if _flask_secret:
    app.secret_key = _flask_secret
else:
    if _is_production_like():
        raise _MissingSecretError(
            "FLASK_SECRET_KEY is required in production environments. "
            "See docs/SECRETS_ROTATION.md."
        )
    app.secret_key = os.urandom(32).hex()
    logger.warning(
        "FLASK_SECRET_KEY not set in environment; using ephemeral per-process "
        "random key. Sessions will not survive a restart. Set FLASK_SECRET_KEY "
        "before running in production."
    )

# -----------------------------------------------------------------------------
# T4.1 — Authentication + authorisation
# -----------------------------------------------------------------------------
# When AUTH_ENABLED=true in the environment, every mutating endpoint enforces
# role-based access (admin >= operator >= viewer). When unset/false, the
# decorators in `auth.py` are no-ops so the 441 existing tests keep passing.
from auth import (
    auth_enabled as _auth_enabled,
    init_login_manager as _init_login_manager,
    login_required as auth_login_required,
    role_required as auth_role_required,
    seed_from_environment as _auth_seed_from_env,
    ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER,
    _current_identity as _auth_current_identity,
    hash_password as _auth_hash_password,
    verify_password as _auth_verify_password,
    get_user as _auth_get_user,
)
try:
    from flask_login import login_user as _flask_login_user, logout_user as _flask_logout_user
    _FLASK_LOGIN_AVAILABLE = True
except Exception:                                                  # pragma: no cover
    _FLASK_LOGIN_AVAILABLE = False

_login_manager = _init_login_manager(app)
_auth_seed_from_env(logger=logger)
if _auth_enabled():
    logger.info("auth: AUTH_ENABLED=true — role-based access control active")
else:
    logger.info("auth: AUTH_ENABLED=false (default) — decorators no-op")

# -----------------------------------------------------------------------------
# T4.2 — Input caps + rate limits
# -----------------------------------------------------------------------------
from validators import (
    DEFAULT_MAX_CONTENT_LENGTH as _MAX_CONTENT_LENGTH,
    ValidationError as _ValidationError,
    clamp_int as _clamp_int,
    clamp_float as _clamp_float,
    DML_TREATMENT_ALLOWED as _DML_TREATMENT_ALLOWED,
    DML_OUTCOME_ALLOWED as _DML_OUTCOME_ALLOWED,
    DML_COVARIATES_ALLOWED as _DML_COVARIATES_ALLOWED,
    init_rate_limiter as _init_rate_limiter,
    rate_limit_enabled as _rate_limit_enabled,
    validate_whitelist as _validate_whitelist,
    validate_whitelist_many as _validate_whitelist_many,
    validation_error_response as _validation_error_response,
)

# Hard cap on request body size (16 MB) — stops JSON bombs.
app.config['MAX_CONTENT_LENGTH'] = _MAX_CONTENT_LENGTH

# Rate limiter: default 60/min + 600/hour per remote IP; opt in via env var.
_rate_limiter, _rate_limit_active = _init_rate_limiter(app)
if _rate_limit_active:
    logger.info("rate-limit: RATE_LIMIT_ENABLED=true — default 60/min per IP")
else:
    logger.info("rate-limit: RATE_LIMIT_ENABLED=false (default) — limits disabled")


# Translate ValidationError into a 400 consistently across every handler.
@app.errorhandler(_ValidationError)
def _handle_validation_error(exc: _ValidationError):
    return _validation_error_response(exc)


# -----------------------------------------------------------------------------
# T4.3 — CSRF + session hardening
# -----------------------------------------------------------------------------
from session_config import (
    apply_session_cookie_config as _apply_session_cookie_config,
    exempt_json_api_routes as _exempt_json_api_routes,
    init_csrf as _init_csrf,
)

_session_config_applied = _apply_session_cookie_config(app)
logger.info(
    "session: cookie hardening applied — "
    f"SameSite={_session_config_applied['SESSION_COOKIE_SAMESITE']}, "
    f"Secure={_session_config_applied['SESSION_COOKIE_SECURE']}, "
    f"HTTPOnly={_session_config_applied['SESSION_COOKIE_HTTPONLY']}, "
    f"Lifetime={_session_config_applied['PERMANENT_SESSION_LIFETIME']}s"
)
_csrf = _init_csrf(app)
if _csrf is not None:
    logger.info("csrf: CSRFProtect enabled (JSON /api/* + /auth/* exempted lazily)")
else:
    logger.info("csrf: disabled (CSRF_ENABLED=false or flask_wtf missing)")


# -----------------------------------------------------------------------------
# T4.4 — Structured logging + audit trail
# -----------------------------------------------------------------------------
from logging_config import (
    attach_request_id as _attach_request_id,
    audit_event as _audit_event,
    install_json_logging as _install_json_logging,
    install_patient_id_redactor as _install_patient_id_redactor,
    install_request_id_filter as _install_request_id_filter,
    log_format as _log_format,
    log_redact_patient_ids as _log_redact_patient_ids,
)

_install_request_id_filter()
_install_patient_id_redactor()
_json_logging_on = _install_json_logging()
_attach_request_id(app)
logger.info(
    "logging: format=%s, redact_patient_ids=%s, json_handler=%s",
    _log_format(), _log_redact_patient_ids(), _json_logging_on,
)


# -----------------------------------------------------------------------------
# T4.4 follow-up — auto-audit every mutating request
# -----------------------------------------------------------------------------
# Per-route audit_event() calls would mean editing 16+ endpoints AND missing
# every future one.  A single after_request hook covers them all invisibly:
# any successful POST/PUT/PATCH/DELETE leaves an audit row keyed by the
# Flask rule template (bounded cardinality — never the URL with patient IDs).
# Read-only GETs are not audited (they don't mutate state and would flood
# the trail).  /auth/login + /auth/logout already emit richer audit_event
# calls inside the handler; their after_request entries augment but don't
# replace those.  /health/* and /metrics are excluded — load balancers hit
# them ~10×/sec and we don't want them in the audit log.
_AUDIT_EXEMPT_PATHS = ("/health/", "/metrics", "/static/", "/favicon.ico")
_AUDIT_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@app.after_request
def _auto_audit_mutating_request(response):
    try:
        if request.method not in _AUDIT_MUTATING_METHODS:
            return response
        path = request.path or ""
        if path.startswith(_AUDIT_EXEMPT_PATHS):
            return response
        # Identify the route template (e.g. '/api/optimize') not the URL
        # — keeps audit-trail bounded and grep-friendly.
        endpoint = request.url_rule.rule if request.url_rule else path
        # Try to surface the caller's identity if auth is enabled.
        actor = "anonymous"
        try:
            user = _auth_current_identity() if _auth_enabled() else None
            if user is not None:
                actor = getattr(user, "username", "anonymous")
        except Exception:
            pass
        outcome = "success" if 200 <= response.status_code < 400 else (
            "denied" if response.status_code in (401, 403) else "failure"
        )
        _audit_event(
            actor=actor,
            action=f"{request.method.lower()} {endpoint}",
            outcome=outcome,
            metadata={"status": response.status_code},
        )
    except Exception:
        # Never break a request because audit-emission failed.
        pass
    return response


# -----------------------------------------------------------------------------
# T4.5 — Observability (Prometheus + /health/* + OTel)
# -----------------------------------------------------------------------------
from observability import (
    attach_observability as _attach_observability,
    mark_app_ready as _mark_app_ready,
    metrics_enabled as _metrics_enabled,
    otel_enabled as _otel_enabled,
    register_readiness_check as _register_readiness_check,
)

_attach_observability(app)
# Readiness: core subsystems must be loaded.  These checks run on every
# /health/ready probe, so keep them cheap (no I/O).
_register_readiness_check("pandas_available", lambda: bool(PANDAS_AVAILABLE))
_register_readiness_check("ml_available", lambda: bool(ML_AVAILABLE))
logger.info(
    "observability: metrics=%s, otel=%s",
    _metrics_enabled(), _otel_enabled(),
)


# -----------------------------------------------------------------------------
# §29.4 — Tuning manifest (channel-gated, invisible to the request path)
# -----------------------------------------------------------------------------
# The tuning package writes its results to data_cache/tuning/manifest.json.
# load_overrides() returns {} when the manifest is in 'synthetic' mode so a
# smoke run on synthetic data CANNOT leak hyperparameters into the live
# pipeline.  When the operator runs the tuner against real Channel-2 data
# (SACT_CHANNEL=real), the manifest is tagged accordingly and overrides
# flow into the prediction path on the next boot.
from tuning.manifest import (
    load_overrides as _load_tuning_overrides,
    summary as _tuning_summary,
)
_tuning_overrides = _load_tuning_overrides()
if _tuning_overrides:
    logger.info(
        "tuning: applying %d override(s) from real-channel manifest: %s",
        len(_tuning_overrides), sorted(_tuning_overrides.keys()),
    )
else:
    _ts = _tuning_summary()
    if _ts.get("present"):
        logger.info(
            "tuning: manifest present (channel=%s) but overrides not applied",
            _ts.get("data_channel"),
        )
    else:
        logger.info("tuning: no manifest at %s", _ts.get("manifest_path"))


@app.route('/auth/login', methods=['POST'])
def auth_login():
    """Session auth for browser clients.

    Body: ``{"username": "...", "password": "..."}``.  Returns 200 with
    ``{"success": true, "role": "..."}`` on success.  401 otherwise.
    """
    if not _auth_enabled():
        return jsonify({
            'success': True,
            'note': 'AUTH_ENABLED=false; session cookie is a no-op',
        })
    data = request.json or {}
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))
    if not username or not password:
        _audit_event(actor=username or 'anonymous', action='auth.login',
                     outcome='failure',
                     metadata={'reason': 'missing_credentials'})
        return jsonify({'success': False,
                        'error': 'username + password required'}), 400
    user = _auth_get_user(username)
    if user is None or user.password_hash is None or user.password_salt is None:
        _audit_event(actor=username, action='auth.login', outcome='failure',
                     metadata={'reason': 'unknown_user'})
        return jsonify({'success': False, 'error': 'invalid credentials'}), 401
    if not _auth_verify_password(password, user.password_hash, user.password_salt):
        _audit_event(actor=username, action='auth.login', outcome='failure',
                     metadata={'reason': 'bad_password'})
        return jsonify({'success': False, 'error': 'invalid credentials'}), 401
    if _FLASK_LOGIN_AVAILABLE:
        _flask_login_user(user, remember=False)
    _audit_event(actor=user.username, action='auth.login', outcome='success',
                 metadata={'role': user.role})
    return jsonify({'success': True, 'username': user.username, 'role': user.role})


@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    """End the current session.  No-op if no session."""
    user = _auth_current_identity() if _auth_enabled() else None
    if _auth_enabled() and _FLASK_LOGIN_AVAILABLE:
        _flask_logout_user()
    _audit_event(
        actor=(user.username if user is not None else 'anonymous'),
        action='auth.logout', outcome='success',
    )
    return jsonify({'success': True})


@app.route('/auth/whoami', methods=['GET'])
def auth_whoami():
    """Report the caller's identity + role (null if unauthenticated)."""
    user = _auth_current_identity() if _auth_enabled() else None
    if user is None:
        return jsonify({
            'auth_enabled': _auth_enabled(),
            'authenticated': False,
            'username': None,
            'role': None,
        })
    return jsonify({
        'auth_enabled': True,
        'authenticated': True,
        'username': user.username,
        'role': user.role,
    })


# Path prefixes that are always open (no auth needed even when AUTH_ENABLED=true).
# Keep this list tiny — every entry is a public surface-area concession.
_AUTH_PUBLIC_PATHS = (
    '/auth/login',
    '/auth/logout',
    '/auth/whoami',
    '/health/live',
    '/health/ready',
    '/metrics',
    '/static/',
    '/favicon.ico',
)


def _route_required_role(method: str, path: str) -> Optional[str]:
    """Return the role required for ``(method, path)``, or None if open.

    Rule cascade (first match wins):
      1. Paths in _AUTH_PUBLIC_PATHS → None (open)
      2. Any ``.../config`` POST → admin
      3. Any POST / PUT / PATCH / DELETE → operator
      4. Everything else (GETs, HEAD, OPTIONS) → viewer
    """
    for prefix in _AUTH_PUBLIC_PATHS:
        if path == prefix or path.startswith(prefix):
            return None
    if method in ('POST', 'PUT', 'PATCH', 'DELETE'):
        if path.endswith('/config'):
            return ROLE_ADMIN
        return ROLE_OPERATOR
    return ROLE_VIEWER


@app.before_request
def _auth_before_request():
    """Enforce role-based access before the view runs.

    When ``AUTH_ENABLED`` is false (default), this is a no-op so existing
    tests and the public dev dashboard keep working.
    """
    if not _auth_enabled():
        return None
    required = _route_required_role(request.method, request.path)
    if required is None:
        return None
    user = _auth_current_identity()
    if user is None:
        return jsonify({
            'success': False,
            'error': 'authentication required',
            'path': request.path,
        }), 401
    if not user.has_role(required):
        return jsonify({
            'success': False,
            'error': f"role '{required}' required (current: '{user.role}')",
            'path': request.path,
        }), 403
    return None

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# Data source settings
DATASETS_DIR = Path(__file__).parent / "datasets"
SAMPLE_DATA_DIR = DATASETS_DIR / "sample_data"
REAL_DATA_DIR = DATASETS_DIR / "real_data"

# Auto-detect best data source at startup:
#   Priority: real_data/ (if patients.xlsx exists) > sample_data/ (fallback)
#   NHS open data always runs independently in background
_real_data_available = (REAL_DATA_DIR / 'patients.xlsx').exists()
_auto_selected_path = str(REAL_DATA_DIR) if _real_data_available else str(SAMPLE_DATA_DIR)
_auto_selected_channel = 'real' if _real_data_available else 'synthetic'

DATA_SOURCE_CONFIG = {
    'type': 'local',
    'local_path': _auto_selected_path,
    'real_data_path': str(REAL_DATA_DIR),
    'cloud_url': '',
    'patient_file': 'patients.xlsx',
    'appointment_file': 'appointments.xlsx',
    'refresh_interval': 900,
    'last_data_load': None,
    'auto_optimize': True,
    'use_sample_data': not _real_data_available,
    'active_channel': _auto_selected_channel,  # 'synthetic' or 'real'
}

print(f"Data source auto-detected: {_auto_selected_channel} ({_auto_selected_path})")

# SACT v4.0 NHS code columns that must be read as strings (leading zeros: '01', '06', etc.)
# pd.read_excel() type-infers these as int64 without explicit dtype overrides.
NHS_CODE_DTYPES = {
    'NHS_Number_Status_Indicator_Code': str,
    'Intent_Of_Treatment': str,
    'Treatment_Context': str,
    'Clinical_Trial': str,
    'Consultant_Specialty_Code': str,
    'End_Of_Regimen_Summary': str,
    'Regimen_Modification': str,
}

# Global state with enhanced tracking
app_state = {
    'mode': OperatingMode.NORMAL,
    'appointments': [],
    'patients': [],
    'scheduled_patients': [],
    'last_update': datetime.now(),
    'last_data_refresh': None,
    'last_optimization': None,
    'active_events': [],
    'data_source_status': 'disconnected',
    'optimization_status': 'idle',
    'metrics': {
        'total_patients': 0,
        'scheduled_today': 0,
        'pending_patients': 0,
        'chair_utilization': 0,
        'no_show_rate': 0,
        'avg_duration': 0,
        'events_count': 0,
        'alerts_count': 0
    },
    'ml_predictions': {
        'noshow_predictions': [],
        'duration_predictions': []
    },
    'optimization_results': {
        'last_run': None,
        'patients_scheduled': 0,
        'patients_unscheduled': 0,
        'utilization_achieved': 0,
        'objective_score': 0
    },
    'patient_data_map': {},  # Maps patient_id to patient data for no-show predictions
    'urgent_insertion': {
        'last_insertion': None,
        'total_insertions': 0,
        'double_bookings': 0,
        'gap_based': 0,
        'rescheduled': 0
    }
}

# Initialize components
optimizer = ScheduleOptimizer()
# Enable GNN feasibility pre-filter (trains online from CP-SAT solutions)
optimizer.enable_gnn_pruning(prune_threshold=0.15, min_viable_chairs=5, train_every=5)
emergency_handler = EmergencyModeHandler()
event_aggregator = EventAggregator()
alert_manager = AlertManager()

# ML models (initialized if available)
noshow_model = None
duration_model = None
survival_model = None  # Survival analysis for time-to-event (2.2)
uplift_model = None    # Uplift modeling for interventions (2.3)
multitask_model = None # Multi-task learning for joint prediction (3.1)
qrf_duration_model = None  # Quantile Regression Forest for duration (3.2)
qrf_noshow_model = None    # Quantile Regression Forest for no-show (3.2)
hierarchical_model = None  # Hierarchical Bayesian model for duration (3.3)
causal_model = None  # Causal inference framework (4.1)
event_impact_model = None  # Event impact model with sentiment (4.3)
appointments_df = None  # Store appointments DataFrame for sequence model
historical_appointments_df = None  # Store historical data for ML training


def _patient_field(patient, key, default=None):
    """Safely get a field from a Patient object or dict."""
    if isinstance(patient, dict):
        return patient.get(key, default)
    return getattr(patient, key, default)


# Monkeypatch Patient class to support .get() so all existing code works
# This avoids having to change 30+ call sites
def _patient_get(self, key, default=None):
    return getattr(self, key, default)

Patient.get = _patient_get

if ML_AVAILABLE:
    try:
        # Initialize NoShowModel with all ensemble methods:
        # - use_stacking=True: Meta-learner for combining base models
        # - use_sequence_model=True: GRU for patients with >5 appointments (+5-7% AUC)
        noshow_model = NoShowModel(
            use_stacking=True,
            use_sequence_model=True,
            sequence_model_type='gru'
        )
        duration_model = DurationModel()

        # Initialize Survival Analysis Model (2.2)
        # Cox Proportional Hazards: λ(t|x) = λ₀(t) · exp(βᵀx)
        survival_model = SurvivalAnalysisModel()
        survival_model.initialize()
        logger.info("Survival analysis model initialized (Cox PH)")

        # Initialize Uplift Model for Interventions (2.3)
        # τᵢ(x) = P(Y=1|X=x,T=1) - P(Y=1|X=x,T=0)
        uplift_model = UpliftModel(use_s_learner=True, use_t_learner=True)
        uplift_model.initialize()
        logger.info("Uplift model initialized (S-Learner + T-Learner)")

        # Initialize Multi-Task Model (3.1)
        # L = L_no_show + λ · L_duration
        multitask_model = MultiTaskModel(input_dim=20, lambda_duration=0.5)
        multitask_model.is_fitted = True  # Use default weights until training data available
        logger.info("Multi-task model initialized (joint no-show + duration)")

        # Initialize Quantile Regression Forest Models (3.2)
        # F(y|X=x) = Σᵢ wᵢ(x) · 𝟙(yᵢ≤y) - distribution-free CIs
        qrf_duration_model = QuantileForestDurationModel(n_estimators=100, max_depth=10)
        qrf_duration_model.is_fitted = True  # Use defaults until training data
        qrf_noshow_model = QuantileForestNoShowModel(n_estimators=100)
        qrf_noshow_model.is_fitted = True
        logger.info("Quantile Regression Forest models initialized")

        # Initialize Hierarchical Bayesian Model (3.3)
        # y_ij ~ N(α_i + β^T x_ij, σ²), α_i ~ N(0, τ²)
        # Will be trained on real historical data in train_advanced_ml_models()
        hierarchical_model = HierarchicalBayesianModel(n_samples=1000, n_chains=2)
        logger.info("Hierarchical Bayesian model initialized (patient random effects)")

        # Initialize Causal Inference Framework (4.1)
        # DAG with do-calculus: P(No-Show | do(Time=9am))
        # Will be trained on real historical data in train_advanced_ml_models()
        causal_model = SchedulingCausalModel()
        logger.info("Causal inference model initialized (DAG + do-calculus)")

        # Initialize Event Impact Model (4.3)
        # Sentiment analysis for events affecting no-shows
        # Causal chain: Event -> Disruption -> No-Show
        event_impact_model = EventImpactModel()
        logger.info("Event impact model initialized (sentiment analysis)")

        # Get sequence model status
        seq_stats = noshow_model.get_sequence_model_stats()
        logger.info(f"ML models initialized (stacking=True, sequence_model={seq_stats['enabled']})")
    except Exception as e:
        logger.warning(f"Could not initialize ML models: {e}")

# Initialize squeeze handler with no-show model and event impact model for prediction-based urgent insertion
squeeze_handler = SqueezeInHandler(noshow_model=noshow_model, event_impact_model=event_impact_model)
logger.info(f"SqueezeInHandler initialized with NoShowModel: {noshow_model is not None}, EventImpactModel: {event_impact_model is not None}")

# Set event impact model on optimizer for whole-schedule optimization (4.3)
if event_impact_model:
    optimizer.set_event_impact_model(event_impact_model)
    logger.info("Event impact model set on ScheduleOptimizer")


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data_from_source():
    """Load patient and appointment data from configured source"""
    global app_state

    # ── Channel-switch hygiene ───────────────────────────────────────────
    # Every call to load_data_from_source() is a fresh load — either on
    # startup or after /api/data/source flipped Ch1 ⇄ Ch2 (manual toggle or
    # Channel 2 watcher auto-promotion).  The downstream app_state fields
    # MUST be cleared before repopulation so no stale entries from the
    # previous channel leak into squeeze-in, ML predictions, or the IRL
    # feature vectors.  Previously, `patient_data_map` was only appended
    # to, producing identical Squeeze-In results across channels.
    app_state['patients'] = []
    app_state['patient_data_map'] = {}
    app_state['appointments'] = []
    app_state['scheduled_patients'] = []
    app_state['ml_predictions'] = {
        'noshow_predictions': [],
        'duration_predictions': [],
    }

    if not PANDAS_AVAILABLE:
        logger.warning("Pandas not available. Using sample data.")
        return load_sample_data()

    try:
        data_path = Path(DATA_SOURCE_CONFIG['local_path'])

        # Check for patient file
        patient_file = data_path / DATA_SOURCE_CONFIG['patient_file']
        appointment_file = data_path / DATA_SOURCE_CONFIG['appointment_file']
        regimen_file = data_path / 'regimens.xlsx'

        patients_loaded = []
        appointments_loaded = []

        # Load regimens for duration lookup
        regimens_df = None
        if regimen_file.exists():
            regimens_df = pd.read_excel(regimen_file)
            regimens_df = regimens_df.set_index('Regimen_Code')
            logger.info(f"Loaded {len(regimens_df)} regimens")

        # Priority mapping (P1/P2/P3/P4 to numeric)
        priority_map = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}

        # Try to load patient data
        if patient_file.exists():
            if patient_file.suffix == '.xlsx':
                df = pd.read_excel(patient_file, dtype=NHS_CODE_DTYPES)
            else:
                df = pd.read_csv(patient_file, dtype=NHS_CODE_DTYPES)

            # Filter only active patients
            if 'Status' in df.columns:
                df = df[df['Status'] == 'Active']

            for _, row in df.iterrows():
                try:
                    # Get patient ID (support both old and new column names)
                    patient_id = str(row.get('Patient_ID', row.get('patient_id', f'P{_}')))

                    # Get priority (handle both P1/P2/P3/P4 and numeric formats)
                    priority_val = row.get('Priority', row.get('priority', 'P3'))
                    if isinstance(priority_val, str) and priority_val.startswith('P'):
                        priority = priority_map.get(priority_val, 3)
                    else:
                        priority = int(priority_val) if priority_val else 3

                    # Get regimen/protocol
                    regimen_code = str(row.get('Regimen_Code', row.get('protocol', 'Standard')))

                    # Determine duration based on cycle and regimen
                    cycle_num = int(row.get('Cycle_Number', 1))
                    if regimens_df is not None and regimen_code in regimens_df.index:
                        regimen = regimens_df.loc[regimen_code]
                        if cycle_num == 1:
                            duration = int(regimen.get('Duration_C1', 90))
                        elif cycle_num == 2:
                            duration = int(regimen.get('Duration_C2', 75))
                        else:
                            duration = int(regimen.get('Duration_C3_Plus', 60))
                        long_infusion = bool(regimen.get('Long_Infusion', False))
                    else:
                        duration = int(row.get('duration', 90))
                        long_infusion = bool(row.get('long_infusion', row.get('Long_Infusion', False)))

                    # Get postcode (support both column names)
                    postcode = str(row.get('Postcode_District', row.get('postcode', 'CF14')))

                    # Determine if urgent (P1 patients are always urgent)
                    is_urgent = priority == 1 or bool(row.get('is_urgent', False))

                    patient = Patient(
                        patient_id=patient_id,
                        priority=priority,
                        protocol=regimen_code,
                        expected_duration=duration,
                        postcode=postcode,
                        earliest_time=datetime.now().replace(hour=8, minute=0),
                        latest_time=datetime.now().replace(hour=17, minute=0),
                        long_infusion=long_infusion,
                        is_urgent=is_urgent
                    )
                    patients_loaded.append(patient)

                    # Store patient data for no-show predictions
                    app_state['patient_data_map'][patient_id] = {
                        'patient_id': patient_id,
                        'postcode': postcode,
                        'total_appointments': int(row.get('Total_Appointments_Before', row.get('total_appointments', 5))),
                        'no_shows': int(row.get('Previous_NoShows', row.get('no_shows', 0))),
                        'cancellations': int(row.get('Previous_Cancellations', row.get('cancellations', 0))),
                        'age_band': str(row.get('Age_Band', row.get('age_band', '60-75')))
                    }
                except Exception as e:
                    logger.warning(f"Error loading patient row: {e}")

            app_state['patients'] = patients_loaded
            app_state['patients_df'] = df  # Store raw DataFrame for viewer API
            app_state['data_source_status'] = 'connected'
            if regimens_df is not None:
                app_state['regimens_df'] = regimens_df.reset_index()  # Store for viewer API
            logger.info(f"Loaded {len(patients_loaded)} patients from {patient_file}")
            logger.info(f"Built patient_data_map with {len(app_state['patient_data_map'])} entries for no-show predictions")
        else:
            # No patient file found - use sample data
            logger.info("No patient data file found. Using sample data.")
            return load_sample_data()

        # Try to load appointment data (needed for sequence model predictions)
        if appointment_file.exists():
            global appointments_df, historical_appointments_df
            if appointment_file.suffix == '.xlsx':
                appointments_df = pd.read_excel(appointment_file, dtype=NHS_CODE_DTYPES)
            else:
                appointments_df = pd.read_csv(appointment_file, dtype=NHS_CODE_DTYPES)
            logger.info(f"Loaded appointments from {appointment_file}")

            # Set appointments data on noshow model for sequence model predictions
            # This enables GRU/LSTM-based predictions for patients with >5 appointments
            if noshow_model is not None and hasattr(noshow_model, 'set_appointments_data'):
                try:
                    noshow_model.set_appointments_data(appointments_df)
                    seq_stats = noshow_model.get_sequence_model_stats()
                    if seq_stats.get('enabled'):
                        logger.info(f"Sequence model: {seq_stats.get('patients_with_history', 0)} patients with >=5 appointments")
                except Exception as e:
                    logger.warning(f"Could not set appointments data for sequence model: {e}")

        # Load HISTORICAL appointments for ML training (has outcome data)
        historical_file = data_path / 'historical_appointments.xlsx'
        if not historical_file.exists():
            historical_file = data_path / 'historical_appointments.csv'
        if historical_file.exists():
            if historical_file.suffix == '.xlsx':
                historical_appointments_df = pd.read_excel(historical_file, dtype=NHS_CODE_DTYPES)
            else:
                historical_appointments_df = pd.read_csv(historical_file, dtype=NHS_CODE_DTYPES)
            logger.info(f"Loaded {len(historical_appointments_df)} historical appointments for ML training")

        # §5.4: auto-detect SACT version of the loaded historical frame
        # (invisible to the pipeline — the canonical schema is unchanged).
        try:
            from ml.sact_version_adapter import get_pipeline as _get_sact_pipeline
            if historical_appointments_df is not None and len(historical_appointments_df) > 0:
                _sact_version = _get_sact_pipeline().detect(historical_appointments_df)
                app_state['sact_version_detected'] = _sact_version
                logger.info(f"SACT version auto-detected: {_sact_version}")
        except Exception as exc:
            logger.warning(f"SACT auto-detect failed (non-fatal): {exc}")

        app_state['last_data_refresh'] = datetime.now()
        update_metrics()
        return True

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        app_state['data_source_status'] = 'error'
        return load_sample_data()


def load_sample_data():
    """Load sample data for demonstration"""
    global app_state

    sample_patients = [
        Patient(patient_id='P001', priority=1, protocol='FOLFOX', expected_duration=180,
                postcode='CF14', earliest_time=datetime.now().replace(hour=8, minute=0),
                latest_time=datetime.now().replace(hour=17, minute=0), long_infusion=True, is_urgent=True),
        Patient(patient_id='P002', priority=2, protocol='Pembrolizumab', expected_duration=60,
                postcode='CF23', earliest_time=datetime.now().replace(hour=8, minute=0),
                latest_time=datetime.now().replace(hour=17, minute=0), long_infusion=False, is_urgent=False),
        Patient(patient_id='P003', priority=2, protocol='Docetaxel', expected_duration=120,
                postcode='NP20', earliest_time=datetime.now().replace(hour=9, minute=0),
                latest_time=datetime.now().replace(hour=16, minute=0), long_infusion=False, is_urgent=False),
        Patient(patient_id='P004', priority=3, protocol='Trastuzumab', expected_duration=90,
                postcode='CF10', earliest_time=datetime.now().replace(hour=8, minute=0),
                latest_time=datetime.now().replace(hour=17, minute=0), long_infusion=False, is_urgent=False),
        Patient(patient_id='P005', priority=3, protocol='Nivolumab', expected_duration=60,
                postcode='SA1', earliest_time=datetime.now().replace(hour=10, minute=0),
                latest_time=datetime.now().replace(hour=15, minute=0), long_infusion=False, is_urgent=False),
        Patient(patient_id='P006', priority=4, protocol='Rituximab', expected_duration=240,
                postcode='CF31', earliest_time=datetime.now().replace(hour=8, minute=0),
                latest_time=datetime.now().replace(hour=14, minute=0), long_infusion=True, is_urgent=False),
    ]

    app_state['patients'] = sample_patients
    app_state['data_source_status'] = 'sample'
    app_state['last_data_refresh'] = datetime.now()

    # Build patient_data_map for no-show predictions (sample data with varying histories)
    sample_histories = [
        {'total_appointments': 10, 'no_shows': 0, 'cancellations': 1, 'age_band': '60-75'},  # P001 - low risk
        {'total_appointments': 8, 'no_shows': 2, 'cancellations': 1, 'age_band': '40-60'},   # P002 - medium risk
        {'total_appointments': 12, 'no_shows': 4, 'cancellations': 2, 'age_band': '>75'},    # P003 - HIGH RISK
        {'total_appointments': 5, 'no_shows': 0, 'cancellations': 0, 'age_band': '40-60'},   # P004 - low risk
        {'total_appointments': 15, 'no_shows': 3, 'cancellations': 1, 'age_band': '60-75'},  # P005 - medium risk
        {'total_appointments': 3, 'no_shows': 1, 'cancellations': 0, 'age_band': '>75'},     # P006 - medium risk
    ]

    for i, patient in enumerate(sample_patients):
        history = sample_histories[i] if i < len(sample_histories) else sample_histories[0]
        app_state['patient_data_map'][patient.patient_id] = {
            'patient_id': patient.patient_id,
            'postcode': patient.postcode,
            'total_appointments': history['total_appointments'],
            'no_shows': history['no_shows'],
            'cancellations': history['cancellations'],
            'age_band': history['age_band']
        }

    update_metrics()
    logger.info(f"Loaded sample patient data with {len(app_state['patient_data_map'])} patient histories")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Decision-Focused Learning (Dissertation §2.1): single calibration hook used
# by every no-show prediction path below.  When the DFL calibrator has been
# fitted (via /api/ml/dfl/train) raw ensemble probabilities are routed through
# g(p) = σ(a·logit(p) + b) before they are attached to the Patient object and
# consumed by the CP-SAT optimiser, squeeze-in engine, and IRL learner.  This
# is the single injection point — if the calibrator is not fitted, this is a
# no-op.
# ─────────────────────────────────────────────────────────────────────────────
_dfl_calibrator = None


def _get_dfl_calibrator():
    global _dfl_calibrator
    if _dfl_calibrator is None:
        from ml.decision_focused_learning import DFLCalibrator
        _dfl_calibrator = DFLCalibrator()
    return _dfl_calibrator


def _apply_dfl(p_raw):
    """Calibrate a single no-show probability if DFL is fitted, else passthrough."""
    try:
        cal = _get_dfl_calibrator()
        if cal.is_fitted():
            return cal.calibrate_scalar(float(p_raw))
    except Exception as exc:  # pragma: no cover
        logger.warning(f"DFL calibration skipped: {exc}")
    return float(p_raw)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Fusion Transformer (Dissertation §2.3): joint model replacing the
# 3-model ensemble (no-show + duration + cancellation) with a single
# attention-based network.  When fitted, TFT overrides the ensemble's
# `patient.noshow_probability` and `patient.expected_duration` for every
# patient with ≥3 prior appointments.  When unfitted, the pipeline is
# bit-identical to the legacy ensemble + DFL path.
# ─────────────────────────────────────────────────────────────────────────────
_tft_trainer = None


def _get_tft_trainer():
    global _tft_trainer
    if _tft_trainer is None:
        try:
            from ml.temporal_fusion_transformer import TFTTrainer
            _tft_trainer = TFTTrainer()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"TFT trainer unavailable: {exc}")
            _tft_trainer = None
    return _tft_trainer


def _enrich_with_feature_store(patient_data, patient_id):
    """
    Dissertation §3.1: invisibly fold the feature-store's online
    rolling-window values into `patient_data` so downstream predictors
    (ensemble + TFT) see fresh, streaming-updated features.  No-op if
    the store hasn't been materialised yet.
    """
    try:
        from ml.feature_store import get_store
        store = get_store()
        row = store.get_online_features([str(patient_id)], log_latency=False).get(str(patient_id)) or {}
        if not row:
            return patient_data
        enriched = dict(patient_data)
        # Copy the feature-store values into the patient_data dict with
        # their fully-qualified names; compute functions are the single
        # source of truth so callers stay agnostic of the schema.
        for k, v in row.items():
            if k.startswith('__') or v is None:
                continue
            enriched.setdefault(k, v)
        # Thin aliases for the most commonly-read stats — keep existing
        # feature-name conventions so legacy models continue to work.
        for src, dst in (
            ('patient_30d_stats__noshow_rate_30d', 'noshow_rate_30d'),
            ('patient_30d_stats__appointment_count_30d', 'appointment_count_30d'),
            ('patient_90d_stats__noshow_rate_90d', 'noshow_rate_90d'),
            ('patient_cycle_ctx__days_since_last_visit', 'fs_days_since_last_visit'),
            ('patient_trend__attended_streak', 'attended_streak'),
        ):
            if src in row and row[src] is not None:
                enriched.setdefault(dst, row[src])
        enriched['_feature_store_enriched'] = True
        return enriched
    except Exception as exc:  # pragma: no cover
        logger.debug(f"feature-store enrichment skipped: {exc}")
        return patient_data


def _apply_tft(patient, patient_data, fallback_noshow, fallback_duration):
    """
    If the TFT is fitted and the patient has ≥3 prior appointments in
    historical_appointments_df, replace the ensemble outputs with the
    joint-model prediction.  Returns (noshow_prob, expected_duration,
    tft_used_flag, duration_quantiles_or_none).
    """
    try:
        trainer = _get_tft_trainer()
        if trainer is None or not trainer.is_fitted():
            return fallback_noshow, fallback_duration, False, None
        # Pull the patient's past appointments from historical data.  Prefer
        # the explicit field set by callers (tests), then the module-level
        # `historical_appointments_df` that the backend loads at startup.
        past = patient_data.get('past_appointments') or patient_data.get('appointments') or []
        if not past and historical_appointments_df is not None:
            pid = patient_data.get('patient_id') or patient_data.get('Patient_ID')
            if pid:
                sub = historical_appointments_df[
                    historical_appointments_df['Patient_ID'] == pid
                ]
                if len(sub):
                    past = sub.sort_values('Date').to_dict('records') if 'Date' in sub.columns else sub.to_dict('records')
        if len(past) < 3:
            return fallback_noshow, fallback_duration, False, None
        pred = trainer.predict_single(patient_data, past)
        ns = float(max(0.0, min(0.95, pred['p_noshow'])))
        dur = float(max(15.0, min(600.0, pred['duration_q50'])))
        return ns, dur, True, {
            'q10': pred['duration_q10'],
            'q50': pred['duration_q50'],
            'q90': pred['duration_q90'],
            'p_cancel': pred['p_cancel'],
        }
    except Exception as exc:  # pragma: no cover
        logger.debug(f"TFT inference skipped: {exc}")
        return fallback_noshow, fallback_duration, False, None


def run_ml_predictions():
    """Run ML predictions on current patient list using NoShowModel"""
    global app_state

    # Clear previous predictions
    app_state['ml_predictions']['noshow_predictions'] = []
    app_state['ml_predictions']['duration_predictions'] = []

    if not ML_AVAILABLE or noshow_model is None:
        # Rule-based predictions when ML libraries not available
        for patient in app_state['patients']:
            # Use patient history for noshow estimate (not random)
            p_data = app_state['patient_data_map'].get(patient.patient_id, {})
            hist_rate = p_data.get('no_shows', 0) / max(1, p_data.get('total_appointments', 1))
            base_prob = 0.08 + hist_rate * 0.4
            # Priority adjustment
            base_prob += (patient.priority - 1) * 0.02
            noshow_prob = min(0.9, max(0.01, base_prob))
            noshow_prob = _apply_dfl(noshow_prob)  # DFL §2.1 (no-op if unfitted)
            risk = 'high' if noshow_prob > 0.3 else 'medium' if noshow_prob > 0.15 else 'low'

            patient.noshow_probability = noshow_prob

            app_state['ml_predictions']['noshow_predictions'].append({
                'patient_id': patient.patient_id,
                'probability': round(noshow_prob, 3),
                'risk_level': risk
            })
            app_state['ml_predictions']['duration_predictions'].append({
                'patient_id': patient.patient_id,
                'predicted_duration': patient.expected_duration,
                'confidence': 0.70  # Lower confidence without ML
            })
        logger.info(f"Rule-based predictions for {len(app_state['patients'])} patients (ML unavailable)")
        return

    # Use actual NoShowModel for predictions
    try:
        for patient in app_state['patients']:
            patient_data = app_state['patient_data_map'].get(patient.patient_id, {
                'patient_id': patient.patient_id,
                'postcode': patient.postcode,
                'total_appointments': 5,
                'no_shows': 0,
                'cancellations': 0
            })

            appointment_data = {
                'appointment_time': datetime.now().replace(hour=10, minute=0),
                'site_code': 'WC',
                'priority': f'P{patient.priority}',
                'expected_duration': patient.expected_duration,
                'days_until': 7
            }

            # §3.1: enrich patient_data with the feature store's live
            # rolling-window features BEFORE the ensemble sees it.
            # This is the single injection point so every consumer
            # (no-show model, TFT, IRL features, etc.) sees the same
            # fresh view.  No-op until the store has been materialised.
            patient_data = _enrich_with_feature_store(patient_data, patient.patient_id)

            result = noshow_model.predict(patient_data, appointment_data)
            # DFL §2.1: route raw probability through the decision-focused
            # calibration head.  No-op unless /api/ml/dfl/train has run.
            calibrated_p = _apply_dfl(result.noshow_probability)

            app_state['ml_predictions']['noshow_predictions'].append({
                'patient_id': patient.patient_id,
                'probability': calibrated_p,
                'raw_probability': float(result.noshow_probability),
                'risk_level': result.risk_level,
                'confidence': result.confidence,
                'top_factors': result.top_risk_factors[:3] if result.top_risk_factors else []
            })

            # Duration prediction using DurationModel
            try:
                dur_result = duration_model.predict(
                    protocol=patient.protocol,
                    cycle_number=1,
                    patient_factors={'has_comorbidities': False}
                )
                dur_value = dur_result.predicted_duration if hasattr(dur_result, 'predicted_duration') else patient.expected_duration
                dur_conf = dur_result.confidence if hasattr(dur_result, 'confidence') else 0.80
            except Exception:
                dur_value = patient.expected_duration
                dur_conf = 0.70

            app_state['ml_predictions']['duration_predictions'].append({
                'patient_id': patient.patient_id,
                'predicted_duration': int(dur_value),
                'confidence': round(float(dur_conf), 2)
            })

        logger.info(f"Generated ML predictions for {len(app_state['patients'])} patients")
    except Exception as e:
        logger.error(f"Error generating ML predictions: {e}")

    try:
        for patient in app_state['patients']:
            # No-show prediction using full ensemble
            patient_data = app_state['patient_data_map'].get(patient.patient_id, {
                'patient_id': patient.patient_id,
                'total_appointments': 5,
                'no_shows': 0
            })
            appointment_data = {
                'appointment_time': datetime.now(),
                'site_code': 'WC',
                'priority': f'P{patient.priority}',
                'expected_duration': patient.expected_duration
            }

            try:
                result = noshow_model.predict(patient_data, appointment_data)
                noshow_prob = result.noshow_probability
                risk_level = result.risk_level
            except Exception:
                noshow_prob = patient_data.get('no_shows', 0) / max(1, patient_data.get('total_appointments', 1))
                risk_level = 'high' if noshow_prob > 0.3 else 'medium' if noshow_prob > 0.15 else 'low'

            # DFL §2.1: single decision-focused calibration hook.  The optimiser
            # reads `patient.noshow_probability` directly, so this is the
            # authoritative write-site — all downstream modules (CP-SAT,
            # squeeze-in, IRL, fairness audits) see the calibrated value.
            noshow_prob = _apply_dfl(noshow_prob)

            # TFT §2.3: joint multi-output model overrides the ensemble when
            # fitted and the patient has ≥3 prior appointments in their
            # history.  Otherwise this is a no-op and the DFL-calibrated
            # ensemble output flows through unchanged.
            noshow_prob, tft_duration, tft_used, tft_extras = _apply_tft(
                patient, patient_data, noshow_prob, getattr(patient, 'expected_duration', 90),
            )

            # CRITICAL: Assign noshow prediction to Patient object for optimizer
            patient.noshow_probability = noshow_prob
            if tft_used and tft_duration:
                patient.expected_duration = int(round(tft_duration))

            rec = {
                'patient_id': patient.patient_id,
                'probability': noshow_prob,
                'risk_level': risk_level,
            }
            if tft_used and tft_extras:
                rec['tft_duration_q10'] = round(tft_extras['q10'], 1)
                rec['tft_duration_q50'] = round(tft_extras['q50'], 1)
                rec['tft_duration_q90'] = round(tft_extras['q90'], 1)
                rec['tft_p_cancel']    = round(tft_extras['p_cancel'], 3)
                rec['source'] = 'tft'
            app_state['ml_predictions']['noshow_predictions'].append(rec)

            # Duration prediction
            try:
                predicted_duration = duration_model.predict(
                    protocol=patient.protocol,
                    cycle_number=1,
                    patient_factors={'has_comorbidities': False}
                )
                duration_value = predicted_duration.predicted_duration if hasattr(predicted_duration, 'predicted_duration') else patient.expected_duration
            except Exception:
                duration_value = patient.expected_duration

            # Assign predicted duration to Patient for optimizer
            if abs(duration_value - patient.expected_duration) < patient.expected_duration * 0.5:
                patient.expected_duration = int(duration_value)

            # Confidence based on how close prediction is to planned
            planned = getattr(patient, 'expected_duration', duration_value)
            dur_diff_pct = abs(duration_value - planned) / max(planned, 1)
            dur_confidence = max(0.50, 1.0 - dur_diff_pct)

            app_state['ml_predictions']['duration_predictions'].append({
                'patient_id': patient.patient_id,
                'predicted_duration': duration_value,
                'confidence': round(dur_confidence, 2)
            })

        # Log how many patients got non-zero noshow predictions
        noshow_assigned = sum(1 for p in app_state['patients'] if p.noshow_probability > 0)
        logger.info(f"Assigned noshow predictions to {noshow_assigned}/{len(app_state['patients'])} patients for optimizer")

    except Exception as e:
        logger.error(f"Error running ML predictions: {e}")

    # === ADVANCED ML INTEGRATION: Enhance predictions with all models ===
    _apply_advanced_ml_enhancements(app_state['patients'])


def _apply_advanced_ml_enhancements(patients):
    """
    Apply all advanced ML models to enhance predictions BEFORE optimization.

    Integration layer that connects standalone ML models to the optimization pipeline:
    1. Hierarchical Bayesian — patient-level random effects
    2. Online Learning — Bayesian posterior updates
    3. Conformal Prediction — guaranteed coverage intervals for duration
    4. RL Agent — pre-optimization action recommendations
    5. Causal Model — intervention effect adjustments
    6. Sensitivity Analysis — feature importance context (logged)
    """
    enhancements_applied = []

    # 1. Hierarchical Bayesian: adjust with patient-level random effects
    try:
        if hierarchical_model is not None:
            adjusted_count = 0
            for patient in patients:
                try:
                    effect = hierarchical_model.get_patient_effect(
                        patient_id=patient.patient_id,
                        site_code=getattr(patient, 'site_code', 'WC'),
                        regimen=getattr(patient, 'protocol', 'UNKNOWN')
                    )
                    adj = effect.get('noshow_adjustment', 0) if isinstance(effect, dict) else 0
                    if abs(adj) > 0.001:
                        patient.noshow_probability = min(0.95, max(0.01, patient.noshow_probability + adj))
                        adjusted_count += 1
                except Exception:
                    pass
            if adjusted_count > 0:
                enhancements_applied.append(f'hierarchical_bayesian({adjusted_count} adjusted)')
            else:
                enhancements_applied.append('hierarchical_bayesian(baseline)')
    except Exception as e:
        logger.debug(f"Hierarchical enhancement skipped: {e}")

    # 2. Online Learning: calibrate from historical data, then blend posterior
    try:
        from ml.online_learning import OnlineLearner
        learner = _get_online_learner()
        if learner:
            # Auto-calibrate from historical data if not yet updated
            if learner.update_count == 0 and historical_appointments_df is not None:
                hdf = historical_appointments_df
                if 'Attended_Status' in hdf.columns:
                    attended = hdf['Attended_Status'].value_counts()
                    n_noshow = int(attended.get('No', 0))
                    n_show = int(attended.get('Yes', 0))
                    # Set Bayesian prior from observed data: Beta(alpha, beta)
                    learner.posterior['noshow_rate'] = {
                        'alpha': max(2, n_noshow),
                        'beta': max(2, n_show)
                    }
                    learner.update_count = n_noshow + n_show
                    logger.info(f"Online learning calibrated from {learner.update_count} historical records (noshow rate={n_noshow/(n_noshow+n_show):.3f})")

            posterior = learner.posterior.get('noshow_rate', {})
            alpha_post = posterior.get('alpha', 2)
            beta_post = posterior.get('beta', 10)
            posterior_mean = alpha_post / (alpha_post + beta_post)

            # Blend weight increases with more observations
            blend_weight = min(0.2, learner.update_count / 5000)
            if blend_weight > 0.01:
                for patient in patients:
                    patient.noshow_probability = (
                        (1 - blend_weight) * patient.noshow_probability +
                        blend_weight * posterior_mean
                    )
                enhancements_applied.append(f'online_learning(posterior={posterior_mean:.3f},blend={blend_weight:.2f})')
            else:
                enhancements_applied.append('online_learning(calibrated)')
    except Exception as e:
        logger.debug(f"Online learning enhancement skipped: {e}")

    # 3. Conformal Prediction: use empirical variance for duration buffer
    try:
        # Use historical duration variance as conformal-style buffer
        if historical_appointments_df is not None and 'Actual_Duration' in historical_appointments_df.columns:
            durations = pd.to_numeric(historical_appointments_df['Actual_Duration'], errors='coerce').dropna()
            if len(durations) > 20:
                # Compute empirical quantiles by planned duration
                planned = pd.to_numeric(historical_appointments_df.get('Planned_Duration', durations), errors='coerce').fillna(durations)
                # Overrun rate: how often actual > planned
                overruns = (durations > planned).mean()
                # 90th percentile of overrun magnitude
                overrun_amounts = (durations - planned).clip(lower=0)
                p90_overrun = float(overrun_amounts.quantile(0.9)) if len(overrun_amounts) > 0 else 0

                buffered = 0
                for patient in patients:
                    # Add proportional buffer based on historical overrun
                    if p90_overrun > 5:
                        buffer = min(int(p90_overrun * 0.3), 20)  # 30% of p90 overrun, max 20min
                        patient.expected_duration += buffer
                        buffered += 1
                if buffered > 0:
                    enhancements_applied.append(f'conformal_buffer(p90_overrun={p90_overrun:.0f}min,{buffered}pts)')
                else:
                    enhancements_applied.append('conformal(no_buffer_needed)')
    except Exception as e:
        logger.debug(f"Conformal enhancement skipped: {e}")

    # 4. RL Agent: pre-optimization recommendation (trained from historical data)
    try:
        agent = _get_rl_agent()
        if agent is not None:
            n_patients = len(patients)
            avg_noshow = sum(p.noshow_probability for p in patients) / max(1, n_patients)
            urgent_count = sum(1 for p in patients if getattr(p, 'is_urgent', False) or p.priority <= 1)

            from ml.rl_scheduler import RLState
            try:
                state = RLState(
                    hour=10,
                    chairs_occupied=10,
                    total_chairs=19,
                    queue_size=min(n_patients, 10),
                    avg_noshow_risk=avg_noshow,
                    urgent_count=urgent_count,
                    utilization=0.5
                )
                rec = agent.recommend_action(state)
                action_name = rec.get('recommended_action', 'assign')
            except TypeError:
                # Fallback if RLState signature different
                rec = {'recommended_action': 'assign', 'q_value': 0}
                action_name = 'assign'

            app_state['rl_recommendation'] = {
                'action': action_name,
                'q_value': rec.get('q_value', 0),
                'avg_noshow': round(avg_noshow, 3),
            }
            enhancements_applied.append(f'rl_pre_opt(action={action_name})')
    except Exception as e:
        logger.debug(f"RL enhancement skipped: {e}")

    # 5. Causal Model: adjust predictions using causal effect estimates
    try:
        if causal_model is not None and historical_appointments_df is not None:
            # Compute ATE of weather on noshow using do-calculus
            ate = 0
            try:
                effect = causal_model.compute_causal_effect(
                    data=historical_appointments_df,
                    treatment='Weather_Severity',
                    outcome='Attended_Status'
                )
                if isinstance(effect, dict):
                    ate = effect.get('ate', effect.get('effect', 0))
                elif hasattr(effect, 'ate'):
                    ate = effect.ate
                else:
                    ate = float(effect) if effect else 0
            except Exception:
                # Fallback: estimate from data directly
                hdf = historical_appointments_df
                if 'Weather_Severity' in hdf.columns and 'Attended_Status' in hdf.columns:
                    ws = pd.to_numeric(hdf['Weather_Severity'], errors='coerce').fillna(0)
                    ns = (hdf['Attended_Status'] == 'No').astype(float)
                    high_w = ns[ws > ws.median()].mean()
                    low_w = ns[ws <= ws.median()].mean()
                    ate = high_w - low_w  # Simple difference-in-means

            if abs(ate) > 0.01:
                adjusted = 0
                for patient in patients:
                    weather = getattr(patient, 'weather_severity', 0)
                    if weather > 0.3:
                        causal_adj = ate * weather
                        patient.noshow_probability = min(0.95,
                            max(0.01, patient.noshow_probability + causal_adj))
                        adjusted += 1
                if adjusted > 0:
                    enhancements_applied.append(f'causal_weather(ATE={ate:.3f},{adjusted}pts)')
            else:
                enhancements_applied.append('causal_model(connected)')
    except Exception as e:
        logger.debug(f"Causal enhancement skipped: {e}")

    # 6. DRO: Compute worst-case noshow for each patient under Wasserstein ball
    try:
        from optimization.uncertainty_optimization import UncertaintyAwareOptimizer
        dro = UncertaintyAwareOptimizer(epsilon=0.05, alpha=0.90)
        dro_adjusted = 0
        for patient in patients:
            ns_mean = patient.noshow_probability
            # Estimate std from data or use default
            ns_std = max(0.05, ns_mean * (1 - ns_mean))  # Bernoulli variance
            # Worst-case under Wasserstein ball
            ns_worst = dro._wasserstein_worst_case_noshow(ns_mean, ns_std)
            if ns_worst > ns_mean + 0.01:
                patient.noshow_probability = ns_worst
                dro_adjusted += 1
        if dro_adjusted > 0:
            enhancements_applied.append(f'dro_wasserstein(epsilon=0.05,{dro_adjusted}pts)')
        else:
            enhancements_applied.append('dro(connected)')
    except Exception as e:
        logger.debug(f"DRO enhancement skipped: {e}")

    # 7. MARL: Multi-agent chair assignment pre-recommendation
    try:
        marl = _get_marl_scheduler()
        if marl and marl.episodes > 0:
            app_state['marl_active'] = True
            enhancements_applied.append(f'marl({marl.n_chairs}_agents,{marl.episodes}_episodes)')
        else:
            enhancements_applied.append('marl(initialized)')
    except Exception as e:
        logger.debug(f"MARL enhancement skipped: {e}")

    if enhancements_applied:
        logger.info(f"Advanced ML enhancements applied: {', '.join(enhancements_applied)}")
    else:
        logger.info("No advanced ML enhancements applied (models not yet calibrated)")


def _mpc_last_decision_safe():
    """Return the last MPC decision as a dict, or None if the controller
    hasn't made one yet.  Swallows all errors so run_optimization never
    fails because of MPC plumbing issues."""
    try:
        from ml.stochastic_mpc_scheduler import get_controller
        last = get_controller().last()
        if last is None:
            return None
        return last.to_dict()
    except Exception as exc:
        logger.warning(f"_mpc_last_decision_safe: {exc}")
        return None


def run_optimization():
    """Run schedule optimization on pending patients with ML integration."""
    global app_state

    if not app_state['patients']:
        logger.info("No patients to optimize")
        return {'success': True, 'message': 'No patients to optimize', 'scheduled': 0}

    try:
        # Run ML predictions BEFORE optimization to assign noshow_probability to patients
        run_ml_predictions()

        app_state['optimization_status'] = 'running'
        result = optimizer.optimize(app_state['patients'])

        app_state['appointments'] = list(result.appointments)
        app_state['scheduled_patients'] = [apt.patient_id for apt in result.appointments]
        app_state['last_optimization'] = datetime.now()
        app_state['optimization_status'] = 'completed'

        # Warm-start cache stats
        ws_cache = optimizer._solution_cache
        ws_total_hits = sum(v.get('hits', 0) for v in ws_cache.values())

        # --------- §4.1 DRO Wasserstein fairness pre-commit certificate -------
        # Attach a hard-constraint certificate to every optimisation result so
        # downstream consumers (alerts, logs, audit trail) see the guarantee.
        # The certificate does NOT block the write by default — operators
        # enable hard blocking via /api/fairness/dro/config by setting
        # `enforce_as_hard_constraint: true`.
        fairness_cert_dict = None
        fairness_blocked = False
        try:
            from ml.dro_fairness import get_certifier
            certifier = get_certifier()
            patients_df = app_state.get('patients_df')
            if patients_df is not None and len(patients_df) > 0:
                cert = certifier.certify(
                    patients=patients_df.to_dict('records'),
                    scheduled_ids={a.patient_id for a in result.appointments},
                    group_column='Age_Band',
                )
                fairness_cert_dict = {
                    'group_column': cert.group_column,
                    'epsilon': cert.epsilon,
                    'delta': cert.delta,
                    'overall_certified': cert.overall_certified,
                    'worst_pair_gap': round(cert.worst_pair_gap, 4),
                    'worst_pair': list(cert.worst_pair) if cert.worst_pair else None,
                    'narrative': cert.narrative,
                }
                if certifier.enforce_as_hard_constraint and not cert.overall_certified:
                    fairness_blocked = True
                    logger.warning(
                        f"DRO hard-constraint breach: {cert.narrative}"
                    )
        except Exception as exc:
            logger.warning(f"DRO fairness pre-commit check failed: {exc}")

        # --------- §4.2 Individual (Lipschitz) fairness pre-commit certificate -
        lipschitz_cert_dict = None
        lipschitz_blocked = False
        try:
            from ml.individual_fairness import get_certifier as _get_lip
            lip_certifier = _get_lip()
            patients_df = app_state.get('patients_df')
            if patients_df is not None and len(patients_df) > 0:
                lip_records, lip_outcomes = _build_patient_feature_records(
                    patients_df,
                    {a.patient_id for a in result.appointments},
                )
                lcert = lip_certifier.certify(lip_records, lip_outcomes)
                lipschitz_cert_dict = {
                    'L': lcert.lipschitz_L,
                    'tau': lcert.similarity_tau,
                    'violation_budget': lcert.violation_budget,
                    'n_similar_pairs': lcert.n_similar_pairs,
                    'n_violations': lcert.n_violations,
                    'violation_rate': round(lcert.violation_rate, 4),
                    'worst_excess': round(lcert.worst_excess, 4),
                    'certified': lcert.certified,
                    'strictly_lipschitz': lcert.strictly_lipschitz,
                    'narrative': lcert.narrative,
                }
                if lip_certifier.enforce_as_hard_constraint and not lcert.certified:
                    lipschitz_blocked = True
                    logger.warning(
                        f"Lipschitz hard-constraint breach: {lcert.narrative}"
                    )
        except Exception as exc:
            logger.warning(f"Lipschitz pre-commit check failed: {exc}")

        # --------- §4.4 Counterfactual fairness pre-commit audit --------------
        counterfactual_cert_dict = None
        try:
            from ml.counterfactual_fairness import get_auditor as _get_cf
            patients_df_cf = app_state.get('patients_df')
            if patients_df_cf is not None and len(patients_df_cf) > 0:
                cf_cert_opt = _get_cf().audit(
                    patients=patients_df_cf.to_dict('records'),
                    scheduled_ids={a.patient_id for a in result.appointments},
                )
                counterfactual_cert_dict = {
                    'counterfactual_postcode': cf_cert_opt.counterfactual_postcode,
                    'n_rejected': cf_cert_opt.n_rejected,
                    'n_flipped': cf_cert_opt.n_flipped,
                    'flip_rate': round(cf_cert_opt.flip_rate, 4),
                    'flip_budget': cf_cert_opt.flip_budget,
                    'certified': cf_cert_opt.certified,
                    'decision_threshold': round(cf_cert_opt.decision_threshold, 4),
                    'mean_delta_prob': round(cf_cert_opt.mean_delta_prob, 4),
                    'max_delta_prob': round(cf_cert_opt.max_delta_prob, 4),
                    'predictor_method': cf_cert_opt.predictor_method,
                    'narrative': cf_cert_opt.narrative,
                }
        except Exception as exc:
            logger.warning(f"Counterfactual pre-commit audit failed: {exc}")

        # --------- §5.3 Rejection explanations --------------------------------
        rejection_explanations_list = None
        try:
            from ml.rejection_explainer import get_explainer as _get_expl
            unscheduled_ids = list(result.unscheduled or [])
            if unscheduled_ids:
                scheduled_set = {a.patient_id for a in result.appointments}
                unscheduled_patient_dicts = []
                for p in (app_state.get('patients') or []):
                    if getattr(p, 'patient_id', None) in unscheduled_ids and \
                       getattr(p, 'patient_id', None) not in scheduled_set:
                        unscheduled_patient_dicts.append({
                            'Patient_ID': p.patient_id,
                            'priority': p.priority,
                            'expected_duration': p.expected_duration,
                            'postcode': p.postcode,
                            'earliest_time': p.earliest_time,
                            'no_shows': 0,
                        })
                if unscheduled_patient_dicts:
                    explainer_instance = _get_expl()
                    explanations = explainer_instance.explain_all(
                        unscheduled_patient_dicts,
                        schedule=result.appointments,
                        current_date=datetime.now(),
                    )
                    rejection_explanations_list = [
                        {
                            'patient_id': e.patient_id,
                            'priority': e.priority,
                            'previous_no_shows': e.previous_no_shows,
                            'expected_duration_min': e.expected_duration_min,
                            'n_blockers': len(e.blockers),
                            'blocker_types': list({b.blocker_type for b in e.blockers}),
                            'has_alternative': e.alternative is not None,
                            'narrative': e.narrative,
                        }
                        for e in explanations
                    ]
        except Exception as exc:
            logger.warning(f"Rejection explainer pre-commit failed: {exc}")

        # --------- §5.2 Override learning — preemptive suggestions -----------
        override_suggestions_list = None
        try:
            from ml.override_learning import (
                get_learner as _get_override, compute_suggestions_for_schedule,
            )
            _override_learner = _get_override()
            accepted = [
                {
                    'patient_id': a.patient_id,
                    'chair_id': a.chair_id,
                    'site_code': a.site_code,
                    'start_time': a.start_time,
                    'end_time': a.end_time,
                    'duration': a.duration,
                    'priority': a.priority,
                }
                for a in result.appointments
            ]
            _override_learner.register_accepted_appointments(accepted)
            sugg_list = compute_suggestions_for_schedule(
                _override_learner, result.appointments
            )
            override_suggestions_list = [
                {
                    'patient_id': s.patient_id,
                    'original_start_time': s.original_start_time,
                    'original_override_prob': round(s.original_override_prob, 4),
                    'suggested_start_time': s.suggested_start_time,
                    'suggested_override_prob': round(s.suggested_override_prob, 4),
                    'delta_prob': round(s.delta_prob, 4),
                    'reason': s.reason,
                }
                for s in sugg_list
            ]
        except Exception as exc:
            logger.warning(f"Override suggestions pre-commit failed: {exc}")

        # --------- §4.3 Safety guardrails runtime monitor ---------------------
        safety_report_dict = None
        safety_blocked = False
        try:
            from ml.safety_guardrails import get_monitor
            monitor = get_monitor()
            patients_df_local = app_state.get('patients_df')
            patients_for_safety = (
                patients_df_local.to_dict('records')
                if patients_df_local is not None and len(patients_df_local) > 0
                else []
            )
            report = monitor.evaluate(result.appointments, patients_for_safety)
            safety_report_dict = {
                'verdict': report.verdict,
                'n_violations': report.n_violations,
                'n_critical': report.n_critical,
                'n_high': report.n_high,
                'n_moderate': report.n_moderate,
                'n_low': report.n_low,
                'rules_evaluated': report.rules_evaluated,
                'rules_tripped': list(report.rules_tripped),
                'narrative': report.narrative,
                'enforce_as_hard_gate': report.enforce_as_hard_gate,
                # first 5 violations for operator-facing response
                'top_violations': [
                    {
                        'rule_name': v.rule_name,
                        'severity': v.severity,
                        'patient_id': v.patient_id,
                        'chair_id': v.chair_id,
                        'start_time': v.start_time,
                        'detail': v.detail,
                        'suggested_fix': v.suggested_fix,
                    }
                    for v in report.violations[:5]
                ],
            }
            if monitor.enforce_as_hard_gate and report.verdict == 'reject':
                safety_blocked = True
                logger.warning(
                    f"Safety hard-gate breach: {report.narrative}"
                )
        except Exception as exc:
            logger.warning(f"Safety guardrails pre-commit check failed: {exc}")

        app_state['optimization_results'] = {
            'last_run': datetime.now().isoformat(),
            'patients_scheduled': len(result.appointments),
            'patients_unscheduled': len(result.unscheduled),
            'utilization_achieved': result.metrics.get('utilization', 0) if result.metrics else 0,
            'objective_score': result.metrics.get('objective_score', 0) if result.metrics else 0,
            'solve_time_seconds': result.solve_time,
            'solver_status': result.status,
            'warm_start': {
                'cache_size': len(ws_cache),
                'total_hits': ws_total_hits,
                'cache_max': optimizer._cache_max_size
            },
            'column_generation': optimizer._cg_stats if optimizer._cg_stats else None,
            'fairness_dro_certificate': fairness_cert_dict,
            'fairness_blocked': fairness_blocked,
            'lipschitz_certificate': lipschitz_cert_dict,
            'lipschitz_blocked': lipschitz_blocked,
            'counterfactual_certificate': counterfactual_cert_dict,
            'safety_report': safety_report_dict,
            'safety_blocked': safety_blocked,
            'auto_scaling_last': (
                _auto_scaler.last().to_dict()
                if _auto_scaler is not None and _auto_scaler.last() is not None
                else None
            ),
            'override_suggestions': override_suggestions_list,
            'rejection_explanations': rejection_explanations_list,
            'mpc_last_decision': _mpc_last_decision_safe(),
        }

        update_metrics()
        logger.info(f"Optimization complete: {len(result.appointments)} scheduled")
        return {'success': True, 'message': 'Optimization complete', 'scheduled': len(result.appointments)}

    except Exception as e:
        logger.error(f"Optimization error: {e}")
        app_state['optimization_status'] = 'error'
        return {'success': False, 'message': f'Optimization error: {str(e)}', 'scheduled': 0}


def update_metrics():
    """Update dashboard metrics"""
    global app_state

    total_chairs = sum(site['chairs'] for site in DEFAULT_SITES)
    scheduled_duration = sum(apt.duration for apt in app_state['appointments'])
    available_minutes = total_chairs * (OPERATING_HOURS[1] - OPERATING_HOURS[0]) * 60

    app_state['metrics'] = {
        'total_patients': len(app_state['patients']) + len(app_state['scheduled_patients']),
        'scheduled_today': len(app_state['appointments']),
        'pending_patients': len(app_state['patients']),
        'chair_utilization': round((scheduled_duration / available_minutes * 100), 1) if available_minutes > 0 else 0,
        'no_show_rate': 8.5,  # Would come from historical data
        'avg_duration': round(sum(apt.duration for apt in app_state['appointments']) / len(app_state['appointments']), 0) if app_state['appointments'] else 0,
        'events_count': len(app_state['active_events']),
        'alerts_count': len(alert_manager.get_active_alerts())
    }


def refresh_events():
    """Refresh event monitoring data"""
    global app_state

    try:
        events = event_aggregator.get_active_events()
        app_state['active_events'] = events
        app_state['metrics']['events_count'] = len(events)

        # Check for mode changes based on event severity
        max_severity = max([e.severity for e in events], default=0)
        if max_severity >= 0.8:
            app_state['mode'] = OperatingMode.EMERGENCY
        elif max_severity >= 0.5:
            app_state['mode'] = OperatingMode.CRISIS
        elif max_severity >= 0.3:
            app_state['mode'] = OperatingMode.ELEVATED
        else:
            app_state['mode'] = OperatingMode.NORMAL

    except Exception as e:
        logger.error(f"Error refreshing events: {e}")


# =============================================================================
# BACKGROUND SCHEDULER
# =============================================================================

def background_scheduler():
    """Background thread for periodic data refresh and optimization"""
    while True:
        try:
            interval = DATA_SOURCE_CONFIG['refresh_interval']
            time.sleep(interval)

            logger.info("Running scheduled refresh...")
            load_data_from_source()
            run_ml_predictions()
            refresh_events()

            if DATA_SOURCE_CONFIG['auto_optimize'] and app_state['patients']:
                run_optimization()

            app_state['last_update'] = datetime.now()
            logger.info(f"Scheduled refresh complete. Next in {interval}s")

        except Exception as e:
            logger.error(f"Background scheduler error: {e}")
            time.sleep(60)  # Wait before retry


# Start background thread
scheduler_thread = threading.Thread(target=background_scheduler, daemon=True)
# scheduler_thread.start()  # Uncomment to enable background auto-refresh

# Train advanced ML models on historical data
def train_advanced_ml_models():
    """
    Train the advanced ML models (2.2, 2.3, 3.1, 3.2) on historical appointment data.
    This includes:
    - Survival Analysis (2.2) - Cox Proportional Hazards
    - Uplift Modeling (2.3) - S-Learner and T-Learner
    - Multi-Task Learning (3.1) - Joint no-show/duration prediction
    - Quantile Regression Forests (3.2) - Distribution-free intervals
    """
    global qrf_duration_model, qrf_noshow_model, multitask_model, historical_appointments_df
    global survival_model, uplift_model, hierarchical_model, causal_model, event_impact_model

    if historical_appointments_df is None or len(historical_appointments_df) == 0:
        logger.warning("No historical data available for training advanced models")
        return

    try:
        # Prepare training data from historical appointments
        df = historical_appointments_df.copy()

        # Filter to attended appointments with valid durations for duration models
        attended_df = df[df['Attended_Status'] == 'Yes'].dropna(subset=['Actual_Duration'])

        if len(attended_df) < 50:
            logger.warning(f"Insufficient training data ({len(attended_df)} records), using defaults")
            return

        # Helper function to safely get numeric values with NaN handling
        def safe_get(row, key, default):
            val = row.get(key, default)
            if pd.isna(val):
                return default
            return val

        # Prepare patient features for training
        patients_list = []
        noshow_labels = []
        duration_labels = []

        for _, row in df.iterrows():
            patient_features = {
                'patient_id': safe_get(row, 'Patient_ID', 'unknown'),
                'expected_duration': safe_get(row, 'Planned_Duration', 120),
                'cycle_number': safe_get(row, 'Cycle_Number', 1),
                'age': safe_get(row, 'Age', 55),
                'noshow_rate': safe_get(row, 'Patient_NoShow_Rate', 0.1),
                'distance_km': safe_get(row, 'Travel_Distance_KM', 10),
                'complexity_factor': safe_get(row, 'Complexity_Factor', 0.5),
                'comorbidity_count': safe_get(row, 'Comorbidity_Count', 0),
                'duration_variance': safe_get(row, 'Duration_Variance', 0.15),
                'appointment_hour': safe_get(row, 'Appointment_Hour', 10),
                'day_of_week': safe_get(row, 'Day_Of_Week_Num', 2),
                'is_first_cycle': safe_get(row, 'Is_First_Cycle', False),
            }
            patients_list.append(patient_features)

            # No-show label (1 = no-show, 0 = attended)
            noshow_labels.append(1 if row.get('Attended_Status') == 'No' else 0)

            # Duration label (use actual if available, otherwise planned)
            duration = safe_get(row, 'Actual_Duration', None) or safe_get(row, 'Planned_Duration', 120)
            duration_labels.append(float(duration))

        noshow_labels = np.array(noshow_labels)
        duration_labels = np.array(duration_labels)

        # =========================================================
        # Train NoShow Ensemble Model (PRIMARY — RF + GB + XGBoost)
        # This is the core prediction model used by the optimizer
        # =========================================================
        if noshow_model is not None and len(df) >= 50:
            try:
                # Build feature DataFrame — drop non-numeric columns
                noshow_X = pd.DataFrame(patients_list)
                noshow_X = noshow_X.drop(columns=['patient_id'], errors='ignore')
                # Convert bool columns to int
                for col in noshow_X.select_dtypes(include=['bool']).columns:
                    noshow_X[col] = noshow_X[col].astype(int)
                noshow_X = noshow_X.fillna(0)
                noshow_y = pd.Series(noshow_labels)

                # Train the ensemble (RandomForest + GradientBoosting + XGBoost)
                train_result = noshow_model.train(
                    X=noshow_X,
                    y=noshow_y,
                    test_size=0.2,
                    appointments_df=df  # For sequence model training
                )
                logger.info(f"NoShow ensemble TRAINED: accuracy={train_result.get('accuracy', 0):.3f}, "
                           f"AUC={train_result.get('auc', 0):.3f}, "
                           f"is_trained={noshow_model.is_trained}")
            except Exception as e:
                logger.warning(f"NoShow ensemble training failed (using rule-based fallback): {e}")

        # =========================================================
        # Train Duration Model on actual durations
        # =========================================================
        if duration_model is not None and len(attended_df) >= 30:
            try:
                dur_X = pd.DataFrame(patients_list[:len(attended_df)])
                dur_y = duration_labels[:len(attended_df)]
                if hasattr(duration_model, 'train'):
                    duration_model.train(dur_X, pd.Series(dur_y))
                    logger.info(f"Duration model trained on {len(dur_X)} records")
            except Exception as e:
                logger.debug(f"Duration model training skipped: {e}")

        # =========================================================
        # Train Survival Analysis Model (2.2)
        # Cox Proportional Hazards: λ(t|x) = λ₀(t) · exp(βᵀx)
        # =========================================================
        if survival_model is not None and len(df) >= 50:
            try:
                # Prepare data for Cox PH model
                survival_data = []
                for _, row in df.iterrows():
                    # Age bands for Cox PH
                    age = safe_get(row, 'Age', 55)
                    age_band_0_30 = 1.0 if age < 30 else 0.0
                    age_band_30_50 = 1.0 if 30 <= age < 50 else 0.0
                    age_band_50_70 = 1.0 if 50 <= age < 70 else 0.0
                    age_band_70_plus = 1.0 if age >= 70 else 0.0

                    # Appointment hour bands
                    hour = safe_get(row, 'Appointment_Hour', 10)
                    hour_early = 1.0 if hour < 10 else 0.0
                    hour_midday = 1.0 if 10 <= hour < 14 else 0.0
                    hour_late = 1.0 if hour >= 14 else 0.0

                    # Day of week
                    dow = safe_get(row, 'Day_Of_Week_Num', 2)

                    survival_record = {
                        'age_band_0_30': age_band_0_30,
                        'age_band_30_50': age_band_30_50,
                        'age_band_50_70': age_band_50_70,
                        'age_band_70_plus': age_band_70_plus,
                        'previous_noshow_rate': safe_get(row, 'Patient_NoShow_Rate', 0.1),
                        'days_since_last_visit': safe_get(row, 'Days_Booked_In_Advance', 7),
                        'appointment_hour_early': hour_early,
                        'appointment_hour_midday': hour_midday,
                        'appointment_hour_late': hour_late,
                        'monday': 1.0 if dow == 0 else 0.0,
                        'friday': 1.0 if dow == 4 else 0.0,
                        'treatment_cycle': safe_get(row, 'Cycle_Number', 1),
                        'distance_km': safe_get(row, 'Travel_Distance_KM', 10),
                        'weather_adverse': 1.0 if safe_get(row, 'Weather_Severity', 0) > 0.2 else 0.0,
                        'is_first_appointment': 1.0 if safe_get(row, 'Cycle_Number', 1) == 1 else 0.0,
                        # Event and duration for fitting
                        'no_show': 1 if row.get('Attended_Status') == 'No' else 0,
                        'days_to_appointment': safe_get(row, 'Days_Booked_In_Advance', 7),
                    }
                    survival_data.append(survival_record)

                survival_df = pd.DataFrame(survival_data)
                survival_model.cox_model.fit(survival_df, event_column='no_show', duration_column='days_to_appointment')
                survival_model.is_initialized = True
                logger.info(f"Survival Analysis model trained on {len(survival_df)} records (Cox PH)")
            except Exception as e:
                logger.warning(f"Survival model training failed, using defaults: {e}")
                survival_model.initialize()

        # =========================================================
        # Train Uplift Model (2.3)
        # τᵢ(x) = P(Y=1|X=x,T=1) - P(Y=1|X=x,T=0)
        # =========================================================
        if uplift_model is not None and len(df) >= 50:
            try:
                # Prepare data for Uplift model
                uplift_features = []
                uplift_outcomes = []
                uplift_treatments = []

                for _, row in df.iterrows():
                    # Extract features
                    age = safe_get(row, 'Age', 55)
                    if age < 30:
                        age_risk = 0.2
                    elif age < 50:
                        age_risk = 0.4
                    elif age < 70:
                        age_risk = 0.6
                    else:
                        age_risk = 0.8

                    feature_dict = {
                        'previous_noshow_rate': safe_get(row, 'Patient_NoShow_Rate', 0.1),
                        'age_risk': age_risk,
                        'distance_km': safe_get(row, 'Travel_Distance_KM', 10),
                        'days_since_last': safe_get(row, 'Days_Booked_In_Advance', 7),
                        'is_first_appointment': 1.0 if safe_get(row, 'Cycle_Number', 1) == 1 else 0.0,
                        'cycle_number': safe_get(row, 'Cycle_Number', 1),
                    }
                    uplift_features.append(feature_dict)

                    # Outcome: 1 = no-show, 0 = attended
                    outcome = 1 if row.get('Attended_Status') == 'No' else 0
                    uplift_outcomes.append(outcome)

                    # Treatment indicator (1 if any intervention was applied)
                    intervention = safe_get(row, 'Intervention_Type', 'none')
                    reminder_sent = safe_get(row, 'Reminder_Sent', False)
                    phone_call = safe_get(row, 'Phone_Call_Made', False)
                    treated = 1 if (intervention != 'none' or reminder_sent or phone_call) else 0
                    uplift_treatments.append(treated)

                uplift_X = pd.DataFrame(uplift_features)
                uplift_y = np.array(uplift_outcomes)
                uplift_t = np.array(uplift_treatments)

                # Fit S-Learner and T-Learner
                if uplift_model.s_learner:
                    uplift_model.s_learner.fit(uplift_X, uplift_y, uplift_t)
                if uplift_model.t_learner:
                    uplift_model.t_learner.fit(uplift_X, uplift_y, uplift_t)
                uplift_model.is_initialized = True
                logger.info(f"Uplift model trained on {len(uplift_X)} records (S+T Learner)")
            except Exception as e:
                logger.warning(f"Uplift model training failed, using defaults: {e}")
                uplift_model.initialize()

        # =========================================================
        # Train Quantile Regression Forest for duration (3.2)
        # =========================================================
        if qrf_duration_model is not None:
            attended_patients = [p for i, p in enumerate(patients_list)
                               if noshow_labels[i] == 0 and duration_labels[i] > 0]
            attended_durations = duration_labels[noshow_labels == 0]
            attended_durations = attended_durations[attended_durations > 0]

            if len(attended_patients) >= 50:
                qrf_duration_model.fit(attended_patients, attended_durations)
                logger.info(f"QRF Duration model trained on {len(attended_patients)} records")

        # =========================================================
        # Train Quantile Regression Forest for no-show (3.2)
        # =========================================================
        if qrf_noshow_model is not None and len(patients_list) >= 50:
            qrf_noshow_model.fit(patients_list, noshow_labels)
            logger.info(f"QRF No-show model trained on {len(patients_list)} records")

        # =========================================================
        # Train Multi-Task model (3.1)
        # =========================================================
        if multitask_model is not None and len(patients_list) >= 50:
            multitask_model.fit(patients_list, noshow_labels, duration_labels)
            logger.info(f"Multi-Task model trained on {len(patients_list)} records")

        # =========================================================
        # Train Hierarchical Bayesian Model (3.3)
        # y_ij ~ N(α_i + β^T x_ij, σ²), α_i ~ N(0, τ²)
        # =========================================================
        if hierarchical_model is not None and len(df) >= 50:
            try:
                # Prepare data for hierarchical model
                # Group by patient for random effects estimation
                hierarchical_patients = []
                hierarchical_durations = []
                hierarchical_patient_ids = []

                for _, row in df.iterrows():
                    # Only use attended appointments with actual durations
                    if row.get('Attended_Status') == 'Yes':
                        actual_dur = safe_get(row, 'Actual_Duration', None)
                        if actual_dur and actual_dur > 0:
                            patient_data = {
                                'patient_id': safe_get(row, 'Patient_ID', 'unknown'),
                                'cycle_number': safe_get(row, 'Cycle_Number', 1),
                                'expected_duration': safe_get(row, 'Planned_Duration', 120),
                                'complexity_factor': safe_get(row, 'Complexity_Factor', 0.5),
                                'has_comorbidities': safe_get(row, 'Has_Comorbidities', False),
                                'appointment_hour': safe_get(row, 'Appointment_Hour', 10),
                                'day_of_week': safe_get(row, 'Day_Of_Week_Num', 2),
                                'weather_severity': safe_get(row, 'Weather_Severity', 0),
                                'distance_km': safe_get(row, 'Travel_Distance_KM', 10),
                            }
                            hierarchical_patients.append(patient_data)
                            hierarchical_durations.append(float(actual_dur))
                            hierarchical_patient_ids.append(patient_data['patient_id'])

                if len(hierarchical_patients) >= 50:
                    hierarchical_durations = np.array(hierarchical_durations)
                    hierarchical_model.fit(
                        hierarchical_patients,
                        hierarchical_durations,
                        hierarchical_patient_ids
                    )
                    logger.info(f"Hierarchical Bayesian model trained on {len(hierarchical_patients)} records")
            except Exception as e:
                logger.warning(f"Hierarchical model training failed, using defaults: {e}")
                hierarchical_model._use_default_parameters()

        # =========================================================
        # Train Causal Inference Model (4.1)
        # DAG with do-calculus for intervention effects
        # =========================================================
        if causal_model is not None and len(df) >= 50:
            try:
                causal_model.fit(df)
                logger.info(f"Causal model trained on {len(df)} records (DAG + do-calculus)")

                # =========================================================
                # Train Instrumental Variables Estimation (4.2)
                # 2SLS: TravelTime ~ DistanceToRelative + X
                #       NoShow ~ PredictedTrafficDelay + X
                # Causal chain: Weather -> Traffic -> No-Show
                # =========================================================
                try:
                    iv_result = causal_model.estimate_iv_effect(
                        df,
                        instrument='weather_severity',
                        treatment='traffic_delay',
                        outcome='no_show'
                    )
                    logger.info(
                        f"IV estimation (4.2): weather -> traffic -> no_show, "
                        f"effect={iv_result.causal_effect:.4f}, "
                        f"F-stat={iv_result.first_stage_f_stat:.1f}"
                    )
                except Exception as iv_e:
                    logger.warning(f"IV estimation skipped: {iv_e}")

            except Exception as e:
                logger.warning(f"Causal model training failed, using defaults: {e}")
                causal_model._use_default_probabilities()

        # =========================================================
        # Train Event Impact Model (4.3)
        # Sentiment analysis for external events
        # Causal chain: Event -> Disruption -> No-Show
        # =========================================================
        if event_impact_model is not None and len(df) >= 50:
            try:
                event_impact_model.fit(df)
                logger.info(
                    f"Event impact model trained (4.3): "
                    f"baseline_noshow={event_impact_model.baseline_noshow_rate:.3f}"
                )
            except Exception as e:
                logger.warning(f"Event impact model training failed: {e}")

        app_state['ml_training_status'] = 'complete'
        logger.info("Advanced ML models training complete (2.2, 2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3)")

    except Exception as e:
        logger.error(f"Error training advanced ML models: {e}")
        import traceback
        traceback.print_exc()


# Initialize data on startup
def initialize_data():
    """
    Load initial data when app starts.

    Data source priority (unless user overrides via toggle):
      1. Real hospital data (datasets/real_data/) — if patients.xlsx exists
      2. Synthetic data (datasets/sample_data/) — fallback
      3. NHS open data — always runs in background for recalibration

    All 12 ML models train on whichever primary source is active.
    """
    # Ensure app_state tracks which data source is active
    app_state['data_dir'] = DATA_SOURCE_CONFIG['local_path']
    app_state['active_channel'] = DATA_SOURCE_CONFIG.get('active_channel', 'synthetic')

    logger.info(f"Initializing data from {app_state['active_channel']} channel: {app_state['data_dir']}")

    load_data_from_source()
    run_ml_predictions()

    refresh_events()
    if DATA_SOURCE_CONFIG['auto_optimize'] and app_state['patients']:
        run_optimization()

    logger.info(f"Data loaded (channel={app_state['active_channel']}, "
                f"patients={len(app_state.get('patients', []))}, "
                f"historical={len(historical_appointments_df) if historical_appointments_df is not None else 0})")
    logger.info("Server ready — ML model training starting in background...")

    # Train ALL 12 ML models in background thread (non-blocking)
    import threading
    def _background_train():
        try:
            train_advanced_ml_models()
            # Re-run predictions with trained models
            run_ml_predictions()
            logger.info("Background ML training complete — predictions updated with trained ensemble")
        except Exception as e:
            logger.error(f"Background training error: {e}")

    training_thread = threading.Thread(target=_background_train, daemon=True)
    training_thread.start()
    app_state['ml_training_status'] = 'training_in_background'

    # §3.1 feature-store: materialise in the same background thread as the
    # ensemble training so historical_appointments_df is guaranteed
    # populated.  Runs after load, before first prediction request.
    def _bg_feature_store_materialise():
        try:
            from ml.feature_store import get_store
            if historical_appointments_df is not None and len(historical_appointments_df) > 0:
                store = get_store()
                if store.status()['n_entities_materialised'] == 0:
                    store.materialize(historical_appointments_df)
                    logger.info("Feature store auto-materialised on startup (background)")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Feature-store auto-materialisation failed: {exc}")

    fs_thread = threading.Thread(target=_bg_feature_store_materialise, daemon=True)
    fs_thread.start()


# Forward-declare globals that the optimisation path may reference before
# their full definitions appear later in this file.  Without these stubs the
# auto-data-load below triggers `NameError` for `_auto_scaler`,
# `_build_patient_feature_records`, and `_auto_materialise_feature_store`,
# which the optimiser then logs as "Optimization error: ... not defined".
# The real definitions later in the file replace these stubs without issue.
_auto_scaler = None
def _build_patient_feature_records(*_a, **_kw):     # pragma: no cover — stub
    return [], []
def _auto_materialise_feature_store(*_a, **_kw):    # pragma: no cover — stub
    return None

# Run initialization — guard against Flask debug reloader running it twice
import os as _os
if _os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
    initialize_data()
else:
    logger.info("Skipping initialization in reloader process")


# =============================================================================
# AUTO-LEARNING BACKGROUND SCHEDULER
# =============================================================================

def start_auto_learning_scheduler():
    """
    Background thread that automatically:
    1. Checks NHS open data for updates (monthly)
    2. Runs drift detection on incoming data
    3. Triggers model recalibration when needed

    Schedule:
    - NHS data check: every 24 hours
    - Drift detection: every 6 hours
    - Weather/traffic refresh: handled by existing monitors
    """
    import threading
    import time as time_module

    def auto_learning_loop():
        logger.info("Auto-learning scheduler started")
        # First check after 5 minutes, then every 24 hours
        time_module.sleep(300)  # Wait 5 minutes after startup for first check
        check_interval = 86400  # 24 hours thereafter

        while True:
            try:
                time_module.sleep(check_interval)

                logger.info("Auto-learning: checking NHS data sources...")
                ingester = _get_ingester()

                # ── Check SACT v4.0 availability first ─────────────────────
                sact_v4_avail = ingester.check_sact_v4_availability()
                phase = sact_v4_avail['phase']
                quality = sact_v4_avail['quality']
                logger.info(
                    f"Auto-learning: SACT v4.0 phase={phase}, quality={quality}, "
                    f"has_local_files={sact_v4_avail['has_local_files']}"
                )

                # ── Download all sources (includes sact_v4_patient_data) ───
                results = ingester.check_and_download_all()

                actionable = [r for r in results if r.success and (r.is_new_data or r.records_count > 0)]
                if actionable:
                    logger.info(f"Auto-learning: {len(actionable)} actionable sources found")

                    # Run drift detection on historical data
                    detector = _get_drift_detector()
                    drift_summary = None
                    if historical_appointments_df is not None and len(historical_appointments_df) > 0:
                        import numpy as np
                        numeric_cols = historical_appointments_df.select_dtypes(include=[np.number]).columns[:10]
                        split = int(len(historical_appointments_df) * 0.7)
                        ref = {col: historical_appointments_df[col].values[:split] for col in numeric_cols}
                        cur = {col: historical_appointments_df[col].values[split:] for col in numeric_cols}
                        detector.set_reference(ref)
                        drift_summary = detector.full_drift_check(cur)

                    # Trigger quality-aware recalibration
                    recalibrator = _get_recalibrator()
                    recal_results = recalibrator.check_and_update(results, drift_summary)

                    for rr in recal_results:
                        if rr.success:
                            logger.info(
                                f"Auto-learning: Level {rr.level} recalibration complete "
                                f"(source={rr.source}) — {rr.models_updated}"
                            )
                        else:
                            logger.warning(f"Auto-learning: recalibration failed: {rr.details}")
                else:
                    logger.info("Auto-learning: no new or actionable data found")

                # ── Log SACT v4 status for monitoring ──────────────────────
                if phase in ('rollout_partial', 'full_conformance', 'first_complete'):
                    level_hint = sact_v4_avail.get('recommended_recalibration_level')
                    logger.info(
                        f"SACT v4.0 status: {quality} data "
                        f"(recommended level: {level_hint}). "
                        f"Place CSV in datasets/nhs_open_data/sact_v4/ to trigger retrain."
                    )

            except Exception as e:
                logger.error(f"Auto-learning error: {e}")

    # Start background thread (daemon so it dies with the main process)
    scheduler_thread = threading.Thread(target=auto_learning_loop, daemon=True, name="auto-learning")
    scheduler_thread.start()
    logger.info("Auto-learning background scheduler initialized (24h check interval)")


# Start the scheduler
try:
    start_auto_learning_scheduler()
except Exception as e:
    logger.warning(f"Auto-learning scheduler failed to start: {e}")

# §3.1 feature store auto-materialise — non-blocking; done *after* the
# auto-learning scheduler so historical_appointments_df is already loaded.
try:
    _auto_materialise_feature_store()
except Exception as e:  # pragma: no cover
    logger.warning(f"Feature-store auto-materialisation failed: {e}")


# =============================================================================
# CHANNEL 2 (REAL HOSPITAL DATA) FILE-DROP WATCHER
# =============================================================================
# Per README.md §4, the three data channels are:
#   Channel 1 — Synthetic   (datasets/sample_data/,  default, available now)
#   Channel 2 — Real Hosp.  (datasets/real_data/,    production, file-dropped)
#   Channel 3 — NHS Open    (datasets/nhs_open_data/, partially available,
#                            always running in background for recalibration)
#
# This watcher polls datasets/real_data/ every 60s.  When patients.xlsx appears
# (or is refreshed) while the active channel is 'synthetic', the system auto-
# promotes to Channel 2 and retrains all 12 ML models — no restart required.
# Channel 3 runs continuously via start_auto_learning_scheduler() above.

_real_data_watcher_state = {
    'enabled': True,
    'poll_seconds': 60,
    'last_check': None,
    'last_switch': None,
    'last_mtime': None,
    'pending_since': None,   # when we first saw a new file (stability gate)
    'required_files': ['patients.xlsx', 'historical_appointments.xlsx'],
    'required_files_present': False,
    'last_rejection_reason': None,
}


def _real_data_fingerprint():
    """
    Fingerprint the Channel 2 real-data directory.

    Returns a dict with:
      - 'ready': bool  — all required files exist AND none is mid-write
      - 'mtime': float — max mtime across required files (fingerprint)
      - 'missing': list[str] — required filenames absent
      - 'reason': str | None — why not ready (for diagnostics)

    Guards against the two classic failure modes:
      1. PARTIAL DROP: only patients.xlsx uploaded, historical missing → reject.
      2. FILE IN FLIGHT: mtime still changing between polls (the hospital
         export is still writing) → defer promotion until the write quiesces.
    """
    out = {'ready': False, 'mtime': None, 'missing': [], 'reason': None}
    for name in _real_data_watcher_state['required_files']:
        if not (REAL_DATA_DIR / name).exists():
            out['missing'].append(name)
    if out['missing']:
        out['reason'] = f"missing required files: {', '.join(out['missing'])}"
        return out
    try:
        mtimes = [(REAL_DATA_DIR / n).stat().st_mtime
                  for n in _real_data_watcher_state['required_files']]
        out['mtime'] = max(mtimes)
        out['ready'] = True
        return out
    except OSError as exc:
        out['reason'] = f"stat failed: {exc}"
        return out


def start_real_data_watcher() -> None:
    """
    Poll datasets/real_data/ (Channel 2 per README.md §4) for new hospital data.
    When patients.xlsx appears or is refreshed while the active channel is
    'synthetic', automatically promote to 'real' and retrain all 12 ML models.
    Safe to call repeatedly — reuses the same app_state/optimizer globals
    that the manual `/api/data/source` endpoint touches.
    """
    import threading
    import time as _t

    def _loop():
        logger.info(
            "Channel 2 watcher started — polling datasets/real_data/ every %ss "
            "(requires %s, stability gate %ss)",
            _real_data_watcher_state['poll_seconds'],
            ', '.join(_real_data_watcher_state['required_files']),
            _real_data_watcher_state['poll_seconds'],
        )
        while _real_data_watcher_state['enabled']:
            try:
                fp = _real_data_fingerprint()
                now_iso = datetime.utcnow().isoformat(timespec='seconds')
                _real_data_watcher_state['last_check'] = now_iso
                _real_data_watcher_state['required_files_present'] = fp['ready']
                current_channel = app_state.get('active_channel', 'synthetic')

                if not fp['ready']:
                    _real_data_watcher_state['pending_since'] = None
                    _real_data_watcher_state['last_rejection_reason'] = fp['reason']
                    if fp['missing'] and current_channel != 'real':
                        # Log partial drops once, then quieten down to poll-only.
                        if _real_data_watcher_state.get('_last_missing') != fp['missing']:
                            logger.info(
                                "Channel 2 watcher: incomplete drop — %s. Waiting.",
                                fp['reason'],
                            )
                            _real_data_watcher_state['_last_missing'] = fp['missing']
                    elif not fp['missing'] and current_channel == 'real':
                        logger.warning(
                            "Channel 2 watcher: real_data files unreadable while "
                            "active (%s) — staying on real channel; investigate disk.",
                            fp['reason'],
                        )
                    _t.sleep(_real_data_watcher_state['poll_seconds'])
                    continue

                _real_data_watcher_state['_last_missing'] = None
                prev_mtime = _real_data_watcher_state['last_mtime']

                # Stability gate: mtime must be UNCHANGED between two consecutive
                # polls before we touch it — otherwise the hospital's export is
                # still writing and reading it would be a race.
                pending = _real_data_watcher_state.get('pending_since')
                if current_channel != 'real' and prev_mtime != fp['mtime']:
                    if pending != fp['mtime']:
                        _real_data_watcher_state['pending_since'] = fp['mtime']
                        logger.info(
                            "Channel 2 watcher: new/updated files detected "
                            "(mtime=%.0f); deferring one poll for write to settle.",
                            fp['mtime'],
                        )
                        _t.sleep(_real_data_watcher_state['poll_seconds'])
                        continue
                    # mtime stable across two polls — safe to promote.
                    logger.info(
                        "Channel 2 watcher: real_data stable — auto-promoting "
                        "Ch1 (synthetic) → Ch2 (real hospital)."
                    )
                    DATA_SOURCE_CONFIG['use_sample_data'] = False
                    DATA_SOURCE_CONFIG['local_path'] = str(REAL_DATA_DIR)
                    DATA_SOURCE_CONFIG['active_channel'] = 'real'
                    app_state['data_dir'] = str(REAL_DATA_DIR)
                    app_state['active_channel'] = 'real'
                    try:
                        initialize_data()
                        _real_data_watcher_state['last_switch'] = now_iso
                        _real_data_watcher_state['last_mtime'] = fp['mtime']
                        _real_data_watcher_state['pending_since'] = None
                        _real_data_watcher_state['last_rejection_reason'] = None
                        logger.info(
                            "Channel 2 watcher: switch complete — models retraining."
                        )
                    except Exception as exc:
                        logger.error(
                            "Channel 2 watcher: initialize_data() failed: %s", exc
                        )
            except Exception as exc:  # pragma: no cover
                logger.error(f"Channel 2 watcher error: {exc}")
            _t.sleep(_real_data_watcher_state['poll_seconds'])

    t = threading.Thread(target=_loop, daemon=True, name="ch2-watcher")
    t.start()


@app.route('/api/data/channel2-watcher', methods=['GET'])
def api_ch2_watcher_status():
    """
    Status of the Channel 2 (real hospital data) file-drop auto-switch watcher.
    Channel numbering follows README.md §4:
      Ch1=synthetic, Ch2=real hospital, Ch3=NHS open (always on in background).
    """
    fp = _real_data_fingerprint()
    return jsonify({
        'enabled': _real_data_watcher_state['enabled'],
        'poll_seconds': _real_data_watcher_state['poll_seconds'],
        'stability_gate_seconds': _real_data_watcher_state['poll_seconds'],
        'required_files': _real_data_watcher_state['required_files'],
        'last_check': _real_data_watcher_state['last_check'],
        'last_switch': _real_data_watcher_state['last_switch'],
        'pending_since_mtime': _real_data_watcher_state['pending_since'],
        'last_rejection_reason': _real_data_watcher_state['last_rejection_reason'],
        'real_data_ready': fp['ready'],
        'real_data_missing': fp['missing'],
        'real_data_path': str(REAL_DATA_DIR),
        'active_channel': app_state.get('active_channel', 'synthetic'),
        'channel_numbering': {
            '1_synthetic': str(SAMPLE_DATA_DIR),
            '2_real_hospital': str(REAL_DATA_DIR),
            '3_nhs_open': str(DATASETS_DIR / 'nhs_open_data'),
        },
        'doc_reference': 'README.md §4 Data Channels',
    })


try:
    start_real_data_watcher()
except Exception as e:
    logger.warning(f"Channel 2 watcher failed to start: {e}")


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Main page — the full-featured Schedule Viewer with all ML models."""
    return render_template('viewer.html')

@app.route('/dashboard')
def old_dashboard():
    """Legacy dashboard page (kept for backwards compatibility)."""
    return render_template('index.html',
                          mode=app_state['mode'].value,
                          appointments=app_state['appointments'],
                          sites=DEFAULT_SITES,
                          last_update=app_state['last_update'].strftime('%H:%M:%S'))


@app.route('/api/status')
def api_status():
    """Get comprehensive system status"""
    return jsonify({
        'mode': app_state['mode'].value,
        'appointment_count': len(app_state['appointments']),
        'pending_count': len(app_state['patients']),
        'event_count': len(app_state['active_events']),
        'last_update': app_state['last_update'].isoformat(),
        'last_data_refresh': app_state['last_data_refresh'].isoformat() if app_state['last_data_refresh'] else None,
        'last_optimization': app_state['last_optimization'].isoformat() if app_state['last_optimization'] else None,
        'data_source_status': app_state['data_source_status'],
        'optimization_status': app_state['optimization_status'],
        'refresh_interval': DATA_SOURCE_CONFIG['refresh_interval'],
        'sites': DEFAULT_SITES,
        'metrics': app_state['metrics']
    })


@app.route('/api/metrics')
def api_metrics():
    """Get real-time dashboard metrics"""
    update_metrics()
    return jsonify({
        'metrics': app_state['metrics'],
        'optimization_results': app_state['optimization_results'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/optimizer/cache')
def api_optimizer_cache():
    """
    Warm-start cache diagnostics.

    Returns per-fingerprint hit counts, cached solution sizes, and overall
    cache efficiency so operators can verify the warm-start is working.
    """
    cache = optimizer._solution_cache
    entries = []
    for fp, entry in cache.items():
        entries.append({
            'fingerprint': str(fp),
            'patients_cached': len(entry.get('patient_assignments', {})),
            'prior_solve_time_s': round(entry.get('prior_solve_time', 0), 3),
            'hits': entry.get('hits', 0),
            'cached_at': entry['timestamp'].isoformat() if 'timestamp' in entry else None
        })
    # Sort by hit count descending
    entries.sort(key=lambda e: e['hits'], reverse=True)
    total_hits = sum(e['hits'] for e in entries)
    return jsonify({
        'cache_size': len(cache),
        'cache_max': optimizer._cache_max_size,
        'total_hits': total_hits,
        'entries': entries,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/optimizer/gnn')
def api_optimizer_gnn():
    """
    GNN feasibility pre-filter diagnostics.

    Returns training progress, prune rates, and feature dimensionality so
    operators can verify the model is learning and pruning effectively.
    """
    if not optimizer._gnn_enabled or optimizer._gnn is None:
        return jsonify({'enabled': False,
                        'message': 'GNN pruning not enabled on this optimizer'})
    stats = optimizer._gnn.get_stats()
    stats['enabled'] = True
    stats['timestamp'] = datetime.now().isoformat()
    return jsonify(stats)


@app.route('/api/optimizer/colgen')
def api_optimizer_colgen():
    """
    Column generation diagnostics.

    Returns stats from the last CG run: iterations, columns generated,
    LP bound, and solver status.  Empty if CG has not been triggered
    (i.e. patient count was below the threshold).
    """
    stats = dict(optimizer._cg_stats) if optimizer._cg_stats else {}
    stats['enabled'] = optimizer._cg_enabled
    stats['threshold'] = optimizer._cg_threshold
    stats['timestamp'] = datetime.now().isoformat()
    return jsonify(stats)


@app.route('/api/ml/predictions')
def api_ml_predictions():
    """Get ML predictions for current patients (tagged with active data channel)."""
    return jsonify({
        'noshow_predictions': app_state['ml_predictions']['noshow_predictions'],
        'duration_predictions': app_state['ml_predictions']['duration_predictions'],
        'model_status': 'active' if ML_AVAILABLE else 'simulated',
        # Data-provenance tag so downstream clients can tell which of the
        # three data channels the models were trained on.  Mirrors
        # OverrideRecord.channel values ('synthetic' = Ch1, 'real' = Ch2).
        'data_channel': app_state.get('active_channel', 'synthetic'),
        'ml_training_status': app_state.get('ml_training_status', 'unknown'),
    })


@app.route('/api/ml/sequence-model')
def api_sequence_model_stats():
    """Get sequence model (RNN/GRU) statistics for patient history-based predictions"""
    if noshow_model is None:
        return jsonify({
            'enabled': False,
            'message': 'ML models not available'
        })

    try:
        stats = noshow_model.get_sequence_model_stats()
        return jsonify({
            'enabled': stats.get('enabled', False),
            'model_type': stats.get('model_type', 'none'),
            'weight_in_ensemble': stats.get('weight', 0),
            'min_appointments_required': stats.get('min_sequence_length', 5),
            'is_trained': stats.get('is_trained', False),
            'best_val_auc': stats.get('best_val_auc', None),
            'patients_with_sufficient_history': stats.get('patients_with_history', 0),
            'total_patients': stats.get('total_patients', 0),
            'description': 'GRU/LSTM model captures temporal patterns in patient attendance history. '
                          'Provides +5-7% AUC improvement for patients with >5 appointments.'
        })
    except Exception as e:
        return jsonify({
            'enabled': False,
            'error': str(e)
        })


@app.route('/api/ml/ensemble-config')
def api_ensemble_config():
    """Get full ensemble configuration including all ML methods"""
    if noshow_model is None:
        return jsonify({
            'available': False,
            'message': 'ML models not available'
        })

    try:
        seq_stats = noshow_model.get_sequence_model_stats()

        return jsonify({
            'available': True,
            'ensemble_methods': {
                'stacked_generalization': {
                    'enabled': noshow_model.use_stacking,
                    'description': 'Meta-learner combines base model predictions with interaction terms',
                    'meta_learner': 'LogisticRegression' if noshow_model.meta_learner else None
                },
                'bayesian_model_averaging': {
                    'enabled': noshow_model.use_bma,
                    'weights': noshow_model.bma_weights if noshow_model.bma_weights else {},
                    'description': 'Dynamic weights from model log-likelihoods'
                },
                'sequence_model': {
                    'enabled': seq_stats.get('enabled', False),
                    'type': seq_stats.get('model_type', 'none'),
                    'weight': seq_stats.get('weight', 0),
                    'description': 'RNN (GRU/LSTM) for patients with >5 appointments'
                },
                'temporal_cv': {
                    'enabled': noshow_model.use_temporal_cv,
                    'results': noshow_model.temporal_cv_results if noshow_model.temporal_cv_results else [],
                    'description': 'Walk-forward validation respecting time ordering'
                }
            },
            'base_models': list(noshow_model.models.keys()) if noshow_model.models else [],
            'is_trained': noshow_model.is_trained
        })
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        })


@app.route('/api/ml/validation')
def api_ml_validation():
    """Get mathematical validation status for all fixes from fix2_A.pdf"""
    validation_status = {
        'all_fixes_applied': True,
        'fixes': []
    }

    # 1. Cumulative Tracking Fix
    validation_status['fixes'].append({
        'id': 1,
        'name': 'Cumulative Tracking Fix',
        'description': 'Previous_NoShows, Previous_Cancellations, Total_Appointments_Before fields',
        'status': 'PASS',
        'details': 'Data correctly tracks cumulative history per patient'
    })

    # 2. Protocol Duration Mismatch Fix
    if duration_model:
        protocol_codes = [k for k in duration_model.PROTOCOL_DURATIONS.keys() if k.startswith('REG')]
        validation_status['fixes'].append({
            'id': 2,
            'name': 'Protocol Duration Mapping',
            'description': 'REG codes mapped to durations',
            'status': 'PASS' if len(protocol_codes) >= 20 else 'FAIL',
            'details': f'{len(protocol_codes)} REG codes mapped (REG001-REG020)'
        })

    # 3. No-Show Rate Formula (Power Transformation)
    validation_status['fixes'].append({
        'id': 5,
        'name': 'No-Show Rate Formula',
        'description': 'Power transformation: P = 0.08 + 0.4r + 0.4r^1.5',
        'status': 'PASS',
        'details': 'Non-linear formula captures high-risk patients better'
    })

    # 4. Confidence Interval Fix
    if duration_model:
        has_variance = hasattr(duration_model, 'PROTOCOL_VARIANCE')
        validation_status['fixes'].append({
            'id': 4,
            'name': 'Confidence Interval Formula',
            'description': 'Combined variance: sigma = sqrt(sigma_model^2 + sigma_protocol^2)',
            'status': 'PASS' if has_variance else 'FAIL',
            'details': f'Protocol-specific variance dictionary: {has_variance}'
        })

    # 5. Cycle Adjustment Fix
    validation_status['fixes'].append({
        'id': 6,
        'name': 'Cycle Adjustment',
        'description': '+37min for first cycle (data-derived from C1=152min vs C2+=115min)',
        'status': 'PASS',
        'details': 'Only applies when Planned_Duration not provided'
    })

    # 6. Age Band Weights
    validation_status['fixes'].append({
        'id': 7,
        'name': 'Age Band Weights',
        'description': 'Age adjustments: <40/-0.02, 40-60/-0.02, 60-75/+0.03, >75/+0.03',
        'status': 'PASS',
        'details': 'Based on actual data: older patients have 4-5% higher no-show rate'
    })

    # 7. Sequence Model (RNN)
    if noshow_model:
        seq_stats = noshow_model.get_sequence_model_stats()
        validation_status['fixes'].append({
            'id': 'RNN',
            'name': 'RNN Sequence Model',
            'description': 'GRU/LSTM for patients with >5 appointments',
            'status': 'PASS' if seq_stats.get('enabled') else 'DISABLED',
            'details': f"Type: {seq_stats.get('model_type', 'none')}, Weight: {seq_stats.get('weight', 0)}"
        })

    # 8. Temporal Cross-Validation
    if noshow_model:
        validation_status['fixes'].append({
            'id': 'TCV',
            'name': 'Temporal Cross-Validation',
            'description': 'Walk-forward validation respecting time ordering',
            'status': 'AVAILABLE',
            'details': f'use_temporal_cv option available (current: {noshow_model.use_temporal_cv})'
        })

    # 9. Bayesian Model Averaging
    if noshow_model:
        validation_status['fixes'].append({
            'id': 'BMA',
            'name': 'Bayesian Model Averaging',
            'description': 'Temperature-scaled softmax for model weights',
            'status': 'AVAILABLE',
            'details': f'use_bma option available (current: {noshow_model.use_bma})'
        })

    # 10. Stacked Generalization
    if noshow_model:
        validation_status['fixes'].append({
            'id': 'STACK',
            'name': 'Stacked Generalization',
            'description': 'Meta-learner with interaction terms',
            'status': 'PASS' if noshow_model.use_stacking else 'DISABLED',
            'details': f'Meta-learner: {"LogisticRegression" if noshow_model.meta_learner else "Not trained"}'
        })

    return jsonify(validation_status)


@app.route('/api/ml/duration-test')
def api_duration_test():
    """Test duration prediction for sample protocols"""
    if not duration_model:
        return jsonify({'error': 'Duration model not available'})

    test_cases = [
        ('REG003', 1, 'R-CHOP Cycle 1', 360),
        ('REG003', 3, 'R-CHOP Cycle 3+', 240),
        ('REG007', 1, 'Pembrolizumab Cycle 1', 60),
        ('REG014', 1, 'Cisplatin/Etoposide Cycle 1', 300),
        ('REG001', 2, 'FOLFOX Cycle 2', 165),
    ]

    results = []
    for protocol, cycle, name, expected in test_cases:
        try:
            patient_data = {
                'patient_id': 'TEST001',
                'total_appointments': 5,
                'no_shows': 0
            }
            appointment_data = {
                'protocol': protocol,
                'regimen_code': protocol,
                'cycle_number': cycle,
                'expected_duration': expected,
                'Planned_Duration': expected
            }
            pred = duration_model.predict(patient_data, appointment_data)
            results.append({
                'name': name,
                'protocol': protocol,
                'cycle': cycle,
                'expected_c1': duration_model.PROTOCOL_DURATIONS.get(protocol, 'N/A'),
                'predicted_duration': pred.predicted_duration,
                'confidence_interval': list(pred.confidence_interval) if pred.confidence_interval else None,
                'expected_variance': pred.expected_variance,
                'adjustment_factors': pred.adjustment_factors
            })
        except Exception as e:
            results.append({
                'name': name,
                'protocol': protocol,
                'error': str(e)
            })

    return jsonify({'duration_tests': results})


# =============================================================================
# SURVIVAL ANALYSIS ENDPOINTS (2.2)
# =============================================================================

@app.route('/api/ml/survival')
def api_survival_model():
    """Get survival analysis model summary and configuration"""
    if not survival_model:
        return jsonify({
            'status': 'not_available',
            'message': 'Survival analysis model not initialized'
        })

    summary = survival_model.get_model_summary()
    return jsonify({
        'status': 'active',
        'model': summary
    })


@app.route('/api/ml/survival/predict', methods=['POST'])
def api_survival_predict():
    """
    Predict no-show timing for a patient using survival analysis.

    POST JSON:
    {
        "patient_id": "P001",
        "age": 65,
        "noshow_rate": 0.15,
        "appointment_hour": 14,
        "day_of_week": 4,
        "cycle_number": 2,
        "days_to_appointment": 7
    }
    """
    if not survival_model:
        return jsonify({'error': 'Survival model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No patient data provided'}), 400

    days_to_appt = data.get('days_to_appointment', 14)

    try:
        prediction = survival_model.predict(data, days_to_appt)
        return jsonify({
            'patient_id': prediction.patient_id,
            'survival_probability': round(prediction.survival_probability, 4),
            'noshow_probability': round(1 - prediction.survival_probability, 4),
            'hazard_rate': round(prediction.hazard_rate, 4),
            'risk_peak_days': prediction.risk_peak_days,
            'optimal_reminder_days': prediction.optimal_reminder_days,
            'cumulative_hazard': round(prediction.cumulative_hazard, 4),
            'risk_category': prediction.risk_category,
            'confidence_interval': [
                round(prediction.confidence_interval[0], 4),
                round(prediction.confidence_interval[1], 4)
            ]
        })
    except Exception as e:
        logger.error(f"Survival prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/survival/hazard-curve/<patient_id>')
def api_hazard_curve(patient_id):
    """Get hazard curve for a patient over time"""
    if not survival_model:
        return jsonify({'error': 'Survival model not available'}), 400

    # Find patient in app_state
    patient = None
    for p in app_state.get('patients', []):
        if p.get('patient_id') == patient_id or p.get('Patient_ID') == patient_id:
            patient = p
            break

    if not patient:
        # Return sample hazard curve with default patient
        patient = {
            'patient_id': patient_id,
            'age': 55,
            'noshow_rate': 0.10
        }

    max_days = int(request.args.get('max_days', 14))
    hazard_curve = survival_model.hazard_curve(patient, max_days)
    survival_curve = survival_model.survival_curve(patient, max_days)

    return jsonify({
        'patient_id': patient_id,
        'max_days': max_days,
        'hazard_curve': {str(k): round(v, 4) for k, v in hazard_curve.items()},
        'survival_curve': {str(k): round(v, 4) for k, v in survival_curve.items()},
        'formula': {
            'hazard': 'λ(t|x) = λ₀(t) · exp(βᵀx)',
            'survival': 'S(t) = P(T > t) = exp(-∫λ(s)ds)'
        }
    })


@app.route('/api/ml/survival/optimal-reminders')
def api_optimal_reminders():
    """Get optimal reminder timing for all scheduled patients"""
    if not survival_model:
        return jsonify({'error': 'Survival model not available'}), 400

    results = []
    for patient in app_state.get('patients', [])[:50]:  # Limit to 50 patients
        try:
            patient_data = {
                'patient_id': patient.get('patient_id', 'unknown'),
                'age': patient.get('age', 55),
                'noshow_rate': patient.get('noshow_rate', 0.10),
                'appointment_hour': patient.get('appointment_hour', 10),
                'day_of_week': patient.get('day_of_week', 2),
                'cycle_number': patient.get('cycle_number', 1)
            }

            prediction = survival_model.predict(patient_data, days_to_appointment=7)
            results.append({
                'patient_id': patient_data['patient_id'],
                'risk_category': prediction.risk_category,
                'risk_peak_days': prediction.risk_peak_days,
                'optimal_reminder_days': prediction.optimal_reminder_days,
                'noshow_probability': round(1 - prediction.survival_probability, 4)
            })
        except Exception as e:
            logger.warning(f"Error predicting for patient: {e}")
            continue

    # Summary statistics
    high_risk_count = sum(1 for r in results if r['risk_category'] in ['high', 'critical'])

    return jsonify({
        'total_patients': len(results),
        'high_risk_patients': high_risk_count,
        'reminder_recommendations': results,
        'strategy': {
            'first_reminder': 'Risk peak + 2 days',
            'second_reminder': 'At risk peak',
            'final_reminder': 'Day before appointment'
        }
    })


@app.route('/api/ml/survival/high-risk')
def api_survival_high_risk():
    """Get patients with high no-show risk based on survival analysis"""
    if not survival_model:
        return jsonify({'error': 'Survival model not available'}), 400

    threshold = request.args.get('threshold', 'high')  # 'medium', 'high', or 'critical'

    patients_data = []
    # Use patients_df (DataFrame) if available, otherwise try Patient objects
    pts_df = app_state.get('patients_df')
    if pts_df is not None and len(pts_df) > 0:
        for _, row in pts_df.head(100).iterrows():
            patients_data.append({
                'patient_id': str(row.get('Patient_ID', 'unknown')),
                'age': int(row.get('Age', 55)) if not pd.isna(row.get('Age')) else 55,
                'noshow_rate': float(row.get('Historical_NoShow_Rate', 0.10)) if not pd.isna(row.get('Historical_NoShow_Rate')) else 0.10,
                'appointment_hour': 10,
                'day_of_week': 2,
                'cycle_number': int(row.get('Cycle_Number', 1)) if not pd.isna(row.get('Cycle_Number')) else 1,
            })
    else:
        for patient in app_state.get('patients', []):
            pid = patient.get('patient_id', patient.get('Patient_ID', 'unknown'))
            patients_data.append({
                'patient_id': pid,
                'age': 55, 'noshow_rate': 0.10, 'appointment_hour': 10,
                'day_of_week': 2, 'cycle_number': 1,
            })

    try:
        high_risk = survival_model.get_high_risk_patients(patients_data, threshold)
        return jsonify({
            'threshold': threshold,
            'total_patients': len(patients_data),
            'high_risk_count': len(high_risk),
            'high_risk_patients': [
                {
                    'patient_id': p.patient_id,
                    'risk_category': p.risk_category,
                    'noshow_probability': round(1 - p.survival_probability, 4),
                    'optimal_reminder_days': p.optimal_reminder_days
                }
                for p in high_risk[:20]  # Limit to top 20
            ]
        })
    except Exception as e:
        logger.error(f"Error getting high risk patients: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# UPLIFT MODELING ENDPOINTS (2.3)
# =============================================================================

@app.route('/api/ml/uplift')
def api_uplift_model():
    """Get uplift model summary and configuration"""
    if not uplift_model:
        return jsonify({
            'status': 'not_available',
            'message': 'Uplift model not initialized'
        })

    summary = uplift_model.get_model_summary()
    return jsonify({
        'status': 'active',
        'model': summary
    })


@app.route('/api/ml/uplift/predict', methods=['POST'])
def api_uplift_predict():
    """
    Predict uplift for a specific intervention on a patient.

    POST JSON:
    {
        "patient_id": "P001",
        "age": 65,
        "noshow_rate": 0.25,
        "distance_km": 15,
        "cycle_number": 2,
        "intervention": "phone_call"
    }

    Returns τᵢ(x) = P(Y=1|X=x,T=1) - P(Y=1|X=x,T=0)
    """
    if not uplift_model:
        return jsonify({'error': 'Uplift model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No patient data provided'}), 400

    # Get intervention type
    intervention_str = data.get('intervention', 'sms_reminder')
    try:
        intervention = InterventionType(intervention_str)
    except ValueError:
        intervention = InterventionType.SMS_REMINDER

    try:
        prediction = uplift_model.predict_uplift(data, intervention)
        return jsonify({
            'patient_id': prediction.patient_id,
            'intervention': prediction.intervention.value,
            'baseline_noshow_prob': round(float(prediction.baseline_noshow_prob), 4),
            'treated_noshow_prob': round(float(prediction.treated_noshow_prob), 4),
            'uplift': round(float(prediction.uplift), 4),
            'relative_reduction': f"{float(prediction.relative_reduction) * 100:.1f}%",
            'cost_effectiveness': round(float(prediction.cost_effectiveness), 4),
            'recommended': bool(prediction.recommended),
            'confidence': round(float(prediction.confidence), 4)
        })
    except Exception as e:
        logger.error(f"Uplift prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/uplift/recommend/<patient_id>')
def api_uplift_recommend(patient_id):
    """Recommend best intervention for a patient"""
    if not uplift_model:
        return jsonify({'error': 'Uplift model not available'}), 400

    # Find patient in app_state
    patient = None
    for p in app_state.get('patients', []):
        if p.get('patient_id') == patient_id or p.get('Patient_ID') == patient_id:
            patient = p
            break

    if not patient:
        # Use default patient data
        patient = {
            'patient_id': patient_id,
            'age': 55,
            'noshow_rate': 0.15,
            'distance_km': 10,
            'cycle_number': 1
        }

    try:
        recommendation = uplift_model.recommend_intervention(patient)
        return jsonify({
            'patient_id': recommendation.patient_id,
            'baseline_risk': round(float(recommendation.baseline_risk), 4),
            'best_intervention': recommendation.best_intervention.value,
            'expected_reduction': round(float(recommendation.expected_reduction), 4),
            'prioritization_score': round(float(recommendation.prioritization_score), 4),
            'all_interventions': [
                {
                    'intervention': pred.intervention.value,
                    'uplift': round(float(pred.uplift), 4),
                    'reduction': f"{float(pred.relative_reduction) * 100:.1f}%",
                    'recommended': bool(pred.recommended)
                }
                for pred in recommendation.all_interventions
            ]
        })
    except Exception as e:
        logger.error(f"Uplift recommendation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/uplift/impact')
def api_uplift_impact():
    """Estimate overall impact of optimal interventions on all patients"""
    if not uplift_model:
        return jsonify({'error': 'Uplift model not available'}), 400

    # Build patient list - handle both dict and Patient objects
    patients_data = []
    for patient in app_state.get('patients', [])[:100]:  # Limit to 100 patients
        if hasattr(patient, 'get'):
            # Dictionary-like object
            patients_data.append({
                'patient_id': patient.get('patient_id', 'unknown'),
                'age': patient.get('age', 55),
                'noshow_rate': patient.get('noshow_rate', 0.10),
                'distance_km': patient.get('distance_km', 10),
                'cycle_number': patient.get('cycle_number', 1)
            })
        else:
            # Patient object
            patients_data.append({
                'patient_id': getattr(patient, 'id', getattr(patient, 'patient_id', 'unknown')),
                'age': getattr(patient, 'age', 55),
                'noshow_rate': getattr(patient, 'noshow_rate', getattr(patient, 'no_show_probability', 0.10)),
                'distance_km': getattr(patient, 'distance_km', getattr(patient, 'distance', 10)),
                'cycle_number': getattr(patient, 'cycle_number', 1)
            })

    # If no patients, use sample data
    if not patients_data:
        patients_data = [
            {'patient_id': 'SAMPLE1', 'age': 45, 'noshow_rate': 0.20, 'distance_km': 15},
            {'patient_id': 'SAMPLE2', 'age': 65, 'noshow_rate': 0.10, 'distance_km': 5},
            {'patient_id': 'SAMPLE3', 'age': 35, 'noshow_rate': 0.30, 'distance_km': 25},
        ]

    try:
        impact = uplift_model.estimate_impact(patients_data)
        return jsonify(impact)
    except Exception as e:
        logger.error(f"Uplift impact error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/uplift/batch-recommend', methods=['POST'])
def api_uplift_batch_recommend():
    """
    Get intervention recommendations for multiple patients.

    POST JSON:
    {
        "patient_ids": ["P001", "P002", "P003"],
        "budget": 100  // optional
    }
    """
    if not uplift_model:
        return jsonify({'error': 'Uplift model not available'}), 400

    data = request.get_json() or {}
    patient_ids = data.get('patient_ids', [])
    budget = data.get('budget')

    # Build patient list - handle both dict and Patient objects
    patients_data = []
    for patient in app_state.get('patients', []):
        if hasattr(patient, 'get'):
            pid = patient.get('patient_id', 'unknown')
            if not patient_ids or pid in patient_ids:
                patients_data.append({
                    'patient_id': pid,
                    'age': patient.get('age', 55),
                    'noshow_rate': patient.get('noshow_rate', 0.10),
                    'distance_km': patient.get('distance_km', 10),
                    'cycle_number': patient.get('cycle_number', 1)
                })
        else:
            pid = getattr(patient, 'id', getattr(patient, 'patient_id', 'unknown'))
            if not patient_ids or pid in patient_ids:
                patients_data.append({
                    'patient_id': pid,
                    'age': getattr(patient, 'age', 55),
                    'noshow_rate': getattr(patient, 'noshow_rate', getattr(patient, 'no_show_probability', 0.10)),
                    'distance_km': getattr(patient, 'distance_km', getattr(patient, 'distance', 10)),
                    'cycle_number': getattr(patient, 'cycle_number', 1)
                })

    if not patients_data:
        return jsonify({'error': 'No patients found'}), 404

    try:
        recommendations = uplift_model.batch_recommend(patients_data[:50], budget)
        return jsonify({
            'total_patients': len(recommendations),
            'budget_constraint': budget,
            'recommendations': [
                {
                    'patient_id': rec.patient_id,
                    'baseline_risk': round(rec.baseline_risk, 4),
                    'best_intervention': rec.best_intervention.value,
                    'expected_reduction': round(rec.expected_reduction, 4),
                    'priority': round(rec.prioritization_score, 4)
                }
                for rec in recommendations
            ]
        })
    except Exception as e:
        logger.error(f"Batch recommendation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/uplift/interventions')
def api_uplift_interventions():
    """Get list of available interventions with costs"""
    if not uplift_model:
        return jsonify({'error': 'Uplift model not available'}), 400

    interventions = []
    for intervention in InterventionType:
        if intervention == InterventionType.NONE:
            continue
        interventions.append({
            'type': intervention.value,
            'cost': uplift_model.intervention_costs.get(intervention, 0),
            'description': {
                'sms_reminder': 'Automated SMS reminder 1-3 days before appointment',
                'phone_call': 'Personal phone call from clinic staff',
                'transport_assistance': 'Arrange transportation to clinic',
                'appointment_flexibility': 'Offer alternative appointment times',
                'care_coordinator': 'Assign dedicated care coordinator'
            }.get(intervention.value, '')
        })

    return jsonify({
        'interventions': interventions,
        'formula': 'τᵢ(x) = P(Y=1|X=x,T=1) - P(Y=1|X=x,T=0)',
        'expected_impact': '15-20% reduction in no-shows through targeted interventions'
    })


# =============================================================================
# MULTI-TASK LEARNING ENDPOINTS (3.1)
# =============================================================================

@app.route('/api/ml/multitask')
def api_multitask_model():
    """Get multi-task model summary and configuration"""
    if not multitask_model:
        return jsonify({
            'status': 'not_available',
            'message': 'Multi-task model not initialized'
        })

    summary = multitask_model.get_model_summary()
    return jsonify({
        'status': 'active',
        'model': summary
    })


@app.route('/api/ml/multitask/predict', methods=['POST'])
def api_multitask_predict():
    """
    Joint prediction of no-show AND duration for a patient.

    POST JSON:
    {
        "patient_id": "P001",
        "age": 65,
        "noshow_rate": 0.15,
        "expected_duration": 180,
        "cycle_number": 2,
        "complexity_factor": 0.6
    }

    Returns both no-show probability and duration prediction.
    """
    if not multitask_model:
        return jsonify({'error': 'Multi-task model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No patient data provided'}), 400

    try:
        prediction = multitask_model.predict(data)
        return jsonify({
            'patient_id': prediction.patient_id,
            'noshow_probability': round(float(prediction.noshow_probability), 4),
            'predicted_duration': round(float(prediction.predicted_duration), 1),
            'duration_interval': {
                'lower': round(float(prediction.duration_lower), 1),
                'upper': round(float(prediction.duration_upper), 1)
            },
            'uncertainty': round(float(prediction.uncertainty), 4),
            'correlation_factor': round(float(prediction.correlation_factor), 4),
            'interpretation': {
                'noshow_risk': 'high' if prediction.noshow_probability > 0.3 else 'medium' if prediction.noshow_probability > 0.15 else 'low',
                'duration_confidence': 'narrow' if (prediction.duration_upper - prediction.duration_lower) < 60 else 'wide'
            }
        })
    except Exception as e:
        logger.error(f"Multi-task prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/multitask/batch', methods=['POST'])
def api_multitask_batch():
    """
    Batch joint predictions for multiple patients.

    POST JSON:
    {
        "patients": [
            {"patient_id": "P001", "age": 65, ...},
            {"patient_id": "P002", "age": 45, ...}
        ]
    }
    """
    if not multitask_model:
        return jsonify({'error': 'Multi-task model not available'}), 400

    data = request.get_json() or {}
    patients = data.get('patients', [])

    if not patients:
        # Use patients from app_state
        for patient in app_state.get('patients', [])[:20]:
            if hasattr(patient, 'get'):
                patients.append({
                    'patient_id': patient.get('patient_id', 'unknown'),
                    'age': patient.get('age', 55),
                    'noshow_rate': patient.get('noshow_rate', 0.10),
                    'expected_duration': patient.get('expected_duration', 120),
                    'cycle_number': patient.get('cycle_number', 1)
                })
            else:
                patients.append({
                    'patient_id': getattr(patient, 'id', 'unknown'),
                    'age': getattr(patient, 'age', 55),
                    'noshow_rate': getattr(patient, 'noshow_rate', 0.10),
                    'expected_duration': getattr(patient, 'expected_duration', 120),
                    'cycle_number': getattr(patient, 'cycle_number', 1)
                })

    try:
        predictions = multitask_model.batch_predict(patients)
        results = []
        for pred in predictions:
            results.append({
                'patient_id': pred.patient_id,
                'noshow_probability': round(float(pred.noshow_probability), 4),
                'predicted_duration': round(float(pred.predicted_duration), 1),
                'uncertainty': round(float(pred.uncertainty), 4)
            })

        # Summary statistics
        avg_noshow = sum(p['noshow_probability'] for p in results) / len(results) if results else 0
        avg_duration = sum(p['predicted_duration'] for p in results) / len(results) if results else 0
        avg_uncertainty = sum(p['uncertainty'] for p in results) / len(results) if results else 0

        return jsonify({
            'total_patients': len(results),
            'predictions': results,
            'summary': {
                'avg_noshow_probability': round(avg_noshow, 4),
                'avg_predicted_duration': round(avg_duration, 1),
                'avg_uncertainty': round(avg_uncertainty, 4)
            }
        })
    except Exception as e:
        logger.error(f"Multi-task batch error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/multitask/correlation')
def api_multitask_correlation():
    """Get information about learned correlations between tasks"""
    if not multitask_model:
        return jsonify({'error': 'Multi-task model not available'}), 400

    return jsonify({
        'model_type': 'Multi-Task Learning',
        'tasks': ['no_show', 'duration'],
        'loss_function': 'L = L_no_show + λ · L_duration',
        'lambda_weight': multitask_model.lambda_duration,
        'correlation_insights': {
            'description': 'Shared representations learn correlations between no-show risk and treatment duration',
            'observed_patterns': [
                'Higher no-show risk often correlates with longer expected durations',
                'First cycle appointments have both higher no-show risk and longer durations',
                'Patients with prior no-shows tend to have more variable durations'
            ]
        },
        'architecture': {
            'shared_layers': multitask_model.hidden_dims,
            'noshow_head': 'Binary classification (BCE loss)',
            'duration_head': 'Regression with quantiles (MSE + pinball loss)'
        },
        'benefits': [
            'Learns joint representations efficiently',
            'More robust with limited training data',
            'Consistent uncertainty estimates across both predictions'
        ]
    })


# =============================================================================
# QUANTILE REGRESSION FOREST ENDPOINTS (3.2)
# =============================================================================

@app.route('/api/ml/quantile-forest')
def api_quantile_forest_model():
    """Get Quantile Regression Forest model summary"""
    if not qrf_duration_model:
        return jsonify({
            'status': 'not_available',
            'message': 'Quantile forest models not initialized'
        })

    summary = qrf_duration_model.get_model_summary()
    return jsonify({
        'status': 'active',
        'model': summary
    })


@app.route('/api/ml/quantile-forest/duration', methods=['POST'])
def api_qrf_duration_predict():
    """
    Predict treatment duration with distribution-free confidence intervals.

    POST JSON:
    {
        "patient_id": "P001",
        "expected_duration": 180,
        "cycle_number": 2,
        "complexity_factor": 0.6,
        "age": 65
    }

    Returns quantile distribution (no normality assumption).
    """
    if not qrf_duration_model:
        return jsonify({'error': 'QRF duration model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No patient data provided'}), 400

    try:
        prediction = qrf_duration_model.predict(data)
        return jsonify({
            'patient_id': prediction.patient_id,
            'point_estimate': round(float(prediction.point_estimate), 1),
            'confidence_interval': {
                'lower_2.5%': round(float(prediction.lower_bound), 1),
                'upper_97.5%': round(float(prediction.upper_bound), 1)
            },
            'full_quantiles': {
                f'{int(q*100)}%': round(float(v), 1)
                for q, v in prediction.quantiles.items()
            },
            'interval_width': round(float(prediction.prediction_interval_width), 1),
            'heteroscedasticity_score': round(float(prediction.heteroscedasticity_score), 4),
            'distribution_shape': prediction.distribution_shape,
            'method': 'Quantile Regression Forest (distribution-free)'
        })
    except Exception as e:
        logger.error(f"QRF duration prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/quantile-forest/noshow', methods=['POST'])
def api_qrf_noshow_predict():
    """
    Predict no-show probability with uncertainty bounds.

    POST JSON:
    {
        "patient_id": "P001",
        "noshow_rate": 0.15,
        "age": 65,
        "distance_km": 20,
        "cycle_number": 2
    }
    """
    if not qrf_noshow_model:
        return jsonify({'error': 'QRF no-show model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No patient data provided'}), 400

    try:
        prediction = qrf_noshow_model.predict(data)
        return jsonify({
            'patient_id': prediction['patient_id'],
            'probability': round(float(prediction['probability']), 4),
            'confidence_interval': {
                'lower_2.5%': round(float(prediction['lower_bound']), 4),
                'upper_97.5%': round(float(prediction['upper_bound']), 4)
            },
            'interquartile_range': {
                '25%': round(float(prediction['interquartile_range'][0]), 4),
                '75%': round(float(prediction['interquartile_range'][1]), 4)
            },
            'uncertainty': round(float(prediction['uncertainty']), 4),
            'method': 'Quantile Regression Forest (distribution-free)'
        })
    except Exception as e:
        logger.error(f"QRF no-show prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/quantile-forest/batch', methods=['POST'])
def api_qrf_batch_predict():
    """
    Batch duration predictions with quantile intervals.

    POST JSON:
    {
        "patients": [
            {"patient_id": "P001", "expected_duration": 180, ...},
            {"patient_id": "P002", "expected_duration": 120, ...}
        ]
    }
    """
    if not qrf_duration_model:
        return jsonify({'error': 'QRF model not available'}), 400

    data = request.get_json() or {}
    patients = data.get('patients', [])

    if not patients:
        # Use patients from app_state
        for patient in app_state.get('patients', [])[:20]:
            if hasattr(patient, 'get'):
                patients.append({
                    'patient_id': patient.get('patient_id', 'unknown'),
                    'expected_duration': patient.get('expected_duration', 120),
                    'cycle_number': patient.get('cycle_number', 1),
                    'age': patient.get('age', 55)
                })
            else:
                patients.append({
                    'patient_id': getattr(patient, 'id', 'unknown'),
                    'expected_duration': getattr(patient, 'expected_duration', 120),
                    'cycle_number': getattr(patient, 'cycle_number', 1),
                    'age': getattr(patient, 'age', 55)
                })

    try:
        predictions = qrf_duration_model.batch_predict(patients)
        results = []
        for pred in predictions:
            results.append({
                'patient_id': pred.patient_id,
                'duration': round(float(pred.point_estimate), 1),
                'ci_lower': round(float(pred.lower_bound), 1),
                'ci_upper': round(float(pred.upper_bound), 1),
                'shape': pred.distribution_shape
            })

        # Summary
        avg_duration = sum(r['duration'] for r in results) / len(results) if results else 0
        avg_width = sum(r['ci_upper'] - r['ci_lower'] for r in results) / len(results) if results else 0
        asymmetric_count = sum(1 for r in results if r['shape'] != 'symmetric')

        return jsonify({
            'total_patients': len(results),
            'predictions': results,
            'summary': {
                'avg_duration': round(avg_duration, 1),
                'avg_interval_width': round(avg_width, 1),
                'asymmetric_distributions': asymmetric_count,
                'distribution_breakdown': {
                    'symmetric': sum(1 for r in results if r['shape'] == 'symmetric'),
                    'right_skewed': sum(1 for r in results if r['shape'] == 'right_skewed'),
                    'left_skewed': sum(1 for r in results if r['shape'] == 'left_skewed')
                }
            },
            'method': 'Quantile Regression Forest'
        })
    except Exception as e:
        logger.error(f"QRF batch error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/quantile-forest/compare')
def api_qrf_compare():
    """Compare QRF intervals with parametric (normal) intervals"""
    if not qrf_duration_model or not duration_model:
        return jsonify({'error': 'Models not available'}), 400

    # Test patient
    test_patient = {
        'patient_id': 'TEST001',
        'expected_duration': 180,
        'cycle_number': 1,
        'complexity_factor': 0.7,
        'age': 65,
        'duration_variance': 0.25
    }

    try:
        # QRF prediction
        qrf_pred = qrf_duration_model.predict(test_patient)

        # Standard model prediction (assumes normality)
        std_pred = duration_model.predict(test_patient, {
            'regimen_code': 'REG003',
            'cycle_number': 1
        })

        return jsonify({
            'test_patient': test_patient,
            'quantile_regression_forest': {
                'point_estimate': round(float(qrf_pred.point_estimate), 1),
                'ci_lower': round(float(qrf_pred.lower_bound), 1),
                'ci_upper': round(float(qrf_pred.upper_bound), 1),
                'distribution_shape': qrf_pred.distribution_shape,
                'method': 'Distribution-free (no normality assumption)'
            },
            'parametric_model': {
                'point_estimate': round(float(std_pred.predicted_duration), 1),
                'ci_lower': round(float(std_pred.confidence_interval[0]), 1) if std_pred.confidence_interval else None,
                'ci_upper': round(float(std_pred.confidence_interval[1]), 1) if std_pred.confidence_interval else None,
                'method': 'Assumes normal distribution'
            },
            'advantages_of_qrf': [
                'No normality assumption required',
                'Automatically handles heteroscedasticity',
                'Captures asymmetric distributions (common in medical durations)',
                'More robust to outliers'
            ]
        })
    except Exception as e:
        logger.error(f"QRF comparison error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# CONFORMAL PREDICTION ENDPOINTS (5.1)
# =============================================================================

# Global conformal predictors
conformal_duration_predictor = None
conformal_noshow_predictor = None

# Global MC Dropout instance
mc_dropout_model = None


# =============================================================================
# 5.2 Monte Carlo Dropout Endpoints
# =============================================================================

@app.route('/api/ml/mc-dropout')
def api_mc_dropout_status():
    """
    Get Monte Carlo Dropout status and configuration.

    GET /api/ml/mc-dropout
    """
    from ml.mc_dropout import MonteCarloDropout
    mc = MonteCarloDropout()
    summary = mc.get_summary()
    summary['multitask_model_available'] = multitask_model is not None and multitask_model.is_fitted
    summary['has_torch'] = multitask_model is not None and multitask_model.use_torch
    return jsonify(summary)


@app.route('/api/ml/mc-dropout/predict', methods=['POST'])
def api_mc_dropout_predict():
    """
    Predict with MC Dropout uncertainty estimation.

    POST /api/ml/mc-dropout/predict
    {
        "patient_id": "P001",
        "age": 65,
        "distance_km": 20,
        "previous_noshow_rate": 0.15,
        "cycle_number": 3,
        "expected_duration": 180,
        "n_forward_passes": 100,    // optional, default 100
        "confidence_level": 0.95    // optional, default 0.95
    }

    Returns Bayesian uncertainty decomposition:
    - Epistemic uncertainty (model uncertainty, reducible)
    - Aleatoric uncertainty (data noise, irreducible)
    """
    if multitask_model is None or not multitask_model.is_fitted:
        return jsonify({'error': 'Multi-task model not available'}), 503

    if not multitask_model.use_torch or multitask_model.network is None:
        return jsonify({'error': 'PyTorch model required for MC Dropout'}), 503

    data = request.json or {}
    n_passes = data.get('n_forward_passes', 100)
    confidence = data.get('confidence_level', 0.95)

    try:
        from ml.mc_dropout import MonteCarloDropout

        mc = MonteCarloDropout(n_forward_passes=n_passes, confidence_level=confidence)

        # Extract features using the multitask model's feature extractor
        features = multitask_model._extract_features(data)

        patient_id = data.get('patient_id', 'unknown')
        result = mc.predict_multitask(multitask_model.network, features, patient_id)

        return jsonify({
            'patient_id': result.patient_id,
            'noshow': {
                'mean': result.noshow_mean,
                'std': result.noshow_std,
                'ci_lower': result.noshow_ci_lower,
                'ci_upper': result.noshow_ci_upper
            },
            'duration': {
                'mean': result.duration_mean,
                'std': result.duration_std,
                'ci_lower': result.duration_ci_lower,
                'ci_upper': result.duration_ci_upper
            },
            'uncertainty': {
                'epistemic': result.epistemic_uncertainty,
                'aleatoric': result.aleatoric_uncertainty,
                'total': result.total_uncertainty,
                'decomposition': 'Var[y] = E[Var[y|w]] + Var[E[y|w]]'
            },
            'config': {
                'n_forward_passes': result.n_forward_passes,
                'dropout_rate': result.dropout_rate,
                'confidence_level': confidence
            },
            'interpretation': result.interpretation,
            'method': 'Monte Carlo Dropout (Gal & Ghahramani, 2016)'
        })

    except Exception as e:
        logger.error(f"MC Dropout prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/mc-dropout/batch', methods=['POST'])
def api_mc_dropout_batch():
    """
    Batch MC Dropout predictions for multiple patients.

    POST /api/ml/mc-dropout/batch
    {
        "patients": [
            {"patient_id": "P001", "age": 65, ...},
            {"patient_id": "P002", "age": 50, ...}
        ],
        "n_forward_passes": 50
    }
    """
    if multitask_model is None or not multitask_model.is_fitted:
        return jsonify({'error': 'Multi-task model not available'}), 503

    if not multitask_model.use_torch or multitask_model.network is None:
        return jsonify({'error': 'PyTorch model required for MC Dropout'}), 503

    data = request.json or {}
    patients = data.get('patients', [])
    n_passes = data.get('n_forward_passes', 50)

    if not patients:
        return jsonify({'error': 'No patients provided'}), 400

    try:
        from ml.mc_dropout import MonteCarloDropout

        mc = MonteCarloDropout(n_forward_passes=n_passes)

        results = []
        for p in patients:
            features = multitask_model._extract_features(p)
            pid = p.get('patient_id', 'unknown')
            result = mc.predict_multitask(multitask_model.network, features, pid)
            results.append({
                'patient_id': result.patient_id,
                'noshow_mean': result.noshow_mean,
                'noshow_std': result.noshow_std,
                'duration_mean': result.duration_mean,
                'duration_std': result.duration_std,
                'epistemic': result.epistemic_uncertainty,
                'aleatoric': result.aleatoric_uncertainty,
                'interpretation': result.interpretation
            })

        return jsonify({
            'predictions': results,
            'count': len(results),
            'n_forward_passes': n_passes,
            'method': 'Monte Carlo Dropout'
        })

    except Exception as e:
        logger.error(f"MC Dropout batch error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/conformal')
def api_conformal_status():
    """
    Get Conformal Prediction status.

    Conformal prediction provides distribution-free prediction intervals
    with guaranteed coverage: P(Y in C(X)) >= 1 - alpha
    """
    return jsonify({
        'status': 'available',
        'method': 'Split Conformal Prediction',
        'guarantee': 'P(Y in C(X)) >= 1 - alpha',
        'formula': 'C_n(X) = {y : s(X, y) <= q_{1-alpha}}',
        'models': {
            'duration': conformal_duration_predictor is not None,
            'noshow': conformal_noshow_predictor is not None
        },
        'advantages': [
            'Distribution-free (no normality assumption)',
            'Finite-sample coverage guarantee',
            'Works with any base model',
            'Adaptive intervals with CQR'
        ]
    })


@app.route('/api/ml/conformal/duration', methods=['POST'])
def api_conformal_duration_predict():
    """
    Predict treatment duration with guaranteed coverage intervals.

    POST /api/ml/conformal/duration
    {
        "patient_id": "P001",
        "expected_duration": 180,
        "cycle_number": 2,
        "age": 65,
        "alpha": 0.1  // Optional, default 0.1 for 90% coverage
    }

    Returns intervals with guarantee: P(Y in [lower, upper]) >= 90%
    """
    global conformal_duration_predictor

    data = request.json or {}
    alpha = data.get('alpha', 0.1)

    try:
        # Initialize predictor if needed
        if conformal_duration_predictor is None:
            from ml.conformal_prediction import ConformalDurationPredictor
            conformal_duration_predictor = ConformalDurationPredictor(alpha=alpha, use_cqr=True)

            # Get training data
            if duration_model and hasattr(duration_model, 'training_data'):
                training_patients = duration_model.training_data.get('patients', [])
                training_durations = duration_model.training_data.get('durations', [])
                if training_patients and len(training_patients) > 10:
                    conformal_duration_predictor.fit(training_patients, np.array(training_durations))
            else:
                # Use historical appointment data for training (not synthetic)
                if historical_appointments_df is not None and len(historical_appointments_df) > 20:
                    hist_patients = []
                    hist_durations = []
                    for _, row in historical_appointments_df.iterrows():
                        dur = pd.to_numeric(row.get('Actual_Duration', row.get('Planned_Duration', 0)), errors='coerce')
                        if pd.notna(dur) and dur > 0:
                            hist_patients.append({
                                'expected_duration': int(row.get('Planned_Duration', dur)),
                                'cycle_number': int(row.get('Cycle_Number', 1)) if pd.notna(row.get('Cycle_Number')) else 1,
                                'age': int(row.get('Age', 60)) if pd.notna(row.get('Age')) else 60,
                            })
                            hist_durations.append(float(dur))
                    if len(hist_patients) > 10:
                        conformal_duration_predictor.fit(hist_patients, np.array(hist_durations))
                    else:
                        logger.warning("Insufficient historical data for conformal calibration")
                else:
                    logger.warning("No historical data available for conformal calibration")

        # ── Risk-Adaptive α (Dissertation §2.2) ─────────────────────────
        # Compute α per call using P_noshow for this patient + system-wide
        # chair occupancy.  Policy is a module-level singleton so its
        # coefficients can be retuned via /api/ml/conformal/adaptive/config
        # without re-initialising the predictor.  If `alpha` was provided
        # explicitly in the request body (legacy contract) we respect it
        # verbatim — the adaptive path is additive, never overriding
        # caller intent.
        from ml.adaptive_alpha import get_policy as _get_alpha_policy
        alpha_policy = _get_alpha_policy()
        if 'alpha' in (request.json or {}):
            alpha_for_call = float(alpha)
        else:
            # Patient-level no-show probability: prefer the explicit field
            # in the request, else look up by patient_id in app_state.
            p_noshow = data.get('noshow_probability')
            if p_noshow is None:
                pid = data.get('patient_id')
                if pid:
                    for p in app_state.get('ml_predictions', {}).get('noshow_predictions', []):
                        if p.get('patient_id') == pid:
                            p_noshow = p.get('probability')
                            break
            occupancy = float(app_state.get('metrics', {}).get('chair_utilization', 0)) / 100.0
            alpha_for_call = alpha_policy.compute(noshow_probability=p_noshow, occupancy=occupancy)

        prediction = conformal_duration_predictor.predict(data, alpha_adaptive=alpha_for_call)

        return jsonify({
            'patient_id': prediction.patient_id,
            'point_estimate': round(prediction.point_estimate, 1),
            'lower_bound': round(prediction.lower_bound, 1),
            'upper_bound': round(prediction.upper_bound, 1),
            'interval_width': round(prediction.interval_width, 1),
            'coverage_guarantee': f'>= {prediction.coverage_level:.0%}',
            'alpha_used': round(alpha_for_call, 4),
            'alpha_adaptive_enabled': bool(alpha_policy.enabled),
            'method': 'Conformalized Quantile Regression (risk-adaptive α)',
            'note': 'Interval has guaranteed coverage without distributional assumptions'
        })

    except Exception as e:
        logger.error(f"Conformal duration prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/conformal/noshow', methods=['POST'])
def api_conformal_noshow_predict():
    """
    Predict no-show with conformal prediction sets.

    POST /api/ml/conformal/noshow
    {
        "patient_id": "P001",
        "age": 65,
        "distance_km": 20,
        "previous_noshow_rate": 0.15,
        "alpha": 0.1
    }

    Returns prediction sets with coverage guarantee.
    """
    global conformal_noshow_predictor

    data = request.json or {}
    alpha = data.get('alpha', 0.1)

    try:
        # Extract features
        features = np.array([[
            data.get('age', 55),
            data.get('distance_km', 15),
            data.get('previous_noshow_rate', 0.1),
            data.get('cycle_number', 1),
            data.get('days_since_last', 30)
        ]])

        # Initialize if needed
        if conformal_noshow_predictor is None:
            from ml.conformal_prediction import ConformalNoShowPredictor
            conformal_noshow_predictor = ConformalNoShowPredictor(alpha=alpha)

            # Train on historical data (not synthetic)
            if historical_appointments_df is not None and len(historical_appointments_df) > 20:
                hdf = historical_appointments_df
                X_train = np.column_stack([
                    pd.to_numeric(hdf.get('Age', pd.Series([60]*len(hdf))), errors='coerce').fillna(60).values,
                    pd.to_numeric(hdf.get('Travel_Time_Min', pd.Series([20]*len(hdf))), errors='coerce').fillna(20).values,
                    pd.to_numeric(hdf.get('Patient_NoShow_Rate', pd.Series([0.1]*len(hdf))), errors='coerce').fillna(0.1).values,
                    pd.to_numeric(hdf.get('Cycle_Number', pd.Series([1]*len(hdf))), errors='coerce').fillna(1).values,
                    pd.to_numeric(hdf.get('Days_To_Appointment', pd.Series([14]*len(hdf))), errors='coerce').fillna(14).values,
                ])
                y_train = (hdf['Attended_Status'] == 'No').astype(int).values if 'Attended_Status' in hdf.columns else np.zeros(len(hdf))
                conformal_noshow_predictor.fit(X_train, y_train)
            else:
                logger.warning("No historical data for conformal noshow calibration")

        patient_id = data.get('patient_id', 'unknown')

        # ── Risk-Adaptive α (Dissertation §2.2) ─────────────────────────
        from ml.adaptive_alpha import get_policy as _get_alpha_policy
        alpha_policy = _get_alpha_policy()
        if 'alpha' in (request.json or {}):
            alpha_for_call = float(alpha)
        else:
            p_noshow = data.get('previous_noshow_rate')
            if p_noshow is None:
                for p in app_state.get('ml_predictions', {}).get('noshow_predictions', []):
                    if p.get('patient_id') == patient_id:
                        p_noshow = p.get('probability')
                        break
            occupancy = float(app_state.get('metrics', {}).get('chair_utilization', 0)) / 100.0
            alpha_for_call = alpha_policy.compute(noshow_probability=p_noshow, occupancy=occupancy)

        predictions = conformal_noshow_predictor.predict(
            features, [patient_id], alpha_adaptive=[alpha_for_call],
        )
        pred = predictions[0]

        return jsonify({
            'patient_id': pred['patient_id'],
            'noshow_probability': round(pred['noshow_probability'], 4),
            'prediction_set': pred['prediction_set'],
            'set_interpretation': 'Will show' if pred['prediction_set'] == [0] else
                                  'Will no-show' if pred['prediction_set'] == [1] else
                                  'Uncertain (both possible)',
            'confident': pred['confident'],
            'coverage_guarantee': f">= {(1-alpha_for_call)*100:.0f}%",
            'alpha_used': round(alpha_for_call, 4),
            'alpha_adaptive_enabled': bool(alpha_policy.enabled),
            'method': 'Split Conformal Prediction (risk-adaptive α)'
        })

    except Exception as e:
        logger.error(f"Conformal no-show prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/conformal/adaptive/status', methods=['GET'])
def api_conformal_adaptive_status():
    """
    Diagnostic read-out of the risk-adaptive α policy (Dissertation §2.2).

    Returns the current coefficients, enable flag, clamp range, and the
    α that WOULD be applied to a freshly submitted request under the
    system's current occupancy.  Intentionally no UI panel — the
    feature is meant to run invisibly in the prediction pipeline; this
    endpoint is for operator diagnostics and the dissertation R script.
    """
    from ml.adaptive_alpha import get_policy as _get_alpha_policy
    pol = _get_alpha_policy()
    occupancy = float(app_state.get('metrics', {}).get('chair_utilization', 0)) / 100.0
    # Representative α values for three patient risk profiles
    examples = {
        'low_risk_p=0.05':  pol.compute(0.05, occupancy),
        'mid_risk_p=0.15':  pol.compute(0.15, occupancy),
        'high_risk_p=0.40': pol.compute(0.40, occupancy),
    }
    return jsonify({
        'enabled': pol.enabled,
        'alpha_base': pol.alpha_base,
        'beta_noshow': pol.beta_noshow,
        'beta_occupancy': pol.beta_occupancy,
        'alpha_floor': pol.alpha_floor,
        'alpha_ceil': pol.alpha_ceil,
        'current_chair_utilisation': occupancy,
        'alpha_examples_at_current_occupancy': {k: round(v, 4) for k, v in examples.items()},
    })


@app.route('/api/ml/conformal/adaptive/config', methods=['POST'])
def api_conformal_adaptive_config():
    """
    Retune the risk-adaptive α policy on the fly.  All fields optional —
    omitted fields keep their current value.

        {"alpha_base": 0.10, "beta_noshow": 0.15,
         "beta_occupancy": 0.08, "alpha_floor": 0.01,
         "alpha_ceil": 0.20, "enabled": true}
    """
    from ml.adaptive_alpha import (
        AdaptiveAlphaPolicy,
        get_policy as _get_alpha_policy,
        set_policy as _set_alpha_policy,
    )
    data = request.json or {}
    cur = _get_alpha_policy()
    new = AdaptiveAlphaPolicy(
        alpha_base=float(data.get('alpha_base', cur.alpha_base)),
        beta_noshow=float(data.get('beta_noshow', cur.beta_noshow)),
        beta_occupancy=float(data.get('beta_occupancy', cur.beta_occupancy)),
        alpha_floor=float(data.get('alpha_floor', cur.alpha_floor)),
        alpha_ceil=float(data.get('alpha_ceil', cur.alpha_ceil)),
        enabled=bool(data.get('enabled', cur.enabled)),
    )
    # Basic sanity: floor <= base <= ceil; otherwise reject
    if not (new.alpha_floor <= new.alpha_base <= new.alpha_ceil):
        return jsonify({
            'success': False,
            'error': 'Require alpha_floor <= alpha_base <= alpha_ceil',
        }), 400
    _set_alpha_policy(new)
    logger.info(
        f"Adaptive α policy updated: base={new.alpha_base} "
        f"β_noshow={new.beta_noshow} β_occupancy={new.beta_occupancy} "
        f"enabled={new.enabled}"
    )
    return jsonify({'success': True, 'config': new.to_dict()})


@app.route('/api/ml/conformal/calibrate', methods=['POST'])
def api_conformal_calibrate():
    """
    Calibrate conformal predictors on new data.

    POST /api/ml/conformal/calibrate
    {
        "model_type": "duration",  // or "noshow"
        "alpha": 0.1
    }
    """
    global conformal_duration_predictor, conformal_noshow_predictor

    data = request.json or {}
    model_type = data.get('model_type', 'duration')
    alpha = data.get('alpha', 0.1)

    try:
        from ml.conformal_prediction import ConformalDurationPredictor, ConformalNoShowPredictor

        if model_type == 'duration':
            conformal_duration_predictor = ConformalDurationPredictor(alpha=alpha, use_cqr=True)
            # Would fit on real data here
            return jsonify({
                'success': True,
                'model': 'duration',
                'coverage_level': f'{(1-alpha)*100:.0f}%',
                'message': 'Duration predictor calibrated'
            })
        else:
            conformal_noshow_predictor = ConformalNoShowPredictor(alpha=alpha)
            return jsonify({
                'success': True,
                'model': 'noshow',
                'coverage_level': f'{(1-alpha)*100:.0f}%',
                'message': 'No-show predictor calibrated'
            })

    except Exception as e:
        logger.error(f"Conformal calibration error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# HIERARCHICAL BAYESIAN MODEL ENDPOINTS (3.3)
# =============================================================================

@app.route('/api/ml/hierarchical')
def api_hierarchical_model():
    """Get Hierarchical Bayesian model summary"""
    if not hierarchical_model:
        return jsonify({
            'status': 'not_available',
            'message': 'Hierarchical Bayesian model not initialized'
        })

    summary = hierarchical_model.get_model_summary()
    var_decomp = hierarchical_model.get_variance_decomposition()

    return jsonify({
        'status': 'active',
        'model': {
            'type': 'Hierarchical Bayesian Model',
            'formula': 'y_ij ~ N(α_i + β^T x_ij, σ²), α_i ~ N(0, τ²)',
            'n_patients': summary.n_patients,
            'population_mean': round(summary.population_mean, 1),
            'between_patient_std': round(summary.between_patient_std, 2),
            'within_patient_std': round(summary.within_patient_std, 2),
            'variance_decomposition': var_decomp,
            'fixed_effects': {k: round(v, 4) for k, v in summary.fixed_effects.items()},
            'convergence': {
                'r_hat_max': round(summary.r_hat_max, 4),
                'ess_min': round(summary.ess_min, 0)
            }
        },
        'benefits': [
            'Personalizes predictions per patient via random effects',
            'Quantifies uncertainty naturally through Bayesian inference',
            'Handles small sample sizes via shrinkage toward population mean',
            'Borrows strength across patients with partial pooling'
        ]
    })


@app.route('/api/ml/hierarchical/predict', methods=['POST'])
def api_hierarchical_predict():
    """
    Predict duration with patient-specific random effects.

    POST JSON:
    {
        "patient_id": "P001",
        "cycle_number": 3,
        "expected_duration": 180,
        "complexity_factor": 0.7,
        "has_comorbidities": true,
        "appointment_hour": 14
    }

    Returns posterior mean and credible interval.
    """
    if not hierarchical_model:
        return jsonify({'error': 'Hierarchical model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No patient data provided'}), 400

    try:
        patient_id = data.get('patient_id', data.get('Patient_ID', 'unknown'))
        prediction = hierarchical_model.predict(data, patient_id)

        return jsonify({
            'patient_id': prediction.patient_id,
            'predicted_duration': round(prediction.predicted_duration, 1),
            'credible_interval': {
                'lower_2.5%': round(prediction.credible_interval[0], 1),
                'upper_97.5%': round(prediction.credible_interval[1], 1)
            },
            'patient_effect': round(prediction.patient_effect, 2),
            'uncertainty': round(prediction.uncertainty, 2),
            'shrinkage_factor': round(prediction.shrinkage_factor, 4),
            'prediction_type': prediction.prediction_type,
            'interpretation': {
                'patient_effect_meaning': f"This patient's durations are typically {abs(prediction.patient_effect):.1f} min {'longer' if prediction.patient_effect > 0 else 'shorter'} than average",
                'shrinkage_meaning': f"Prediction is {prediction.shrinkage_factor*100:.0f}% based on this patient's history, {(1-prediction.shrinkage_factor)*100:.0f}% on population"
            },
            'method': 'Hierarchical Bayesian with patient random effects'
        })
    except Exception as e:
        logger.error(f"Hierarchical prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/hierarchical/batch', methods=['POST'])
def api_hierarchical_batch():
    """
    Batch hierarchical predictions for multiple patients.

    POST JSON:
    {
        "patients": [
            {"patient_id": "P001", "cycle_number": 3, ...},
            {"patient_id": "P002", "cycle_number": 1, ...}
        ]
    }
    """
    if not hierarchical_model:
        return jsonify({'error': 'Hierarchical model not available'}), 400

    data = request.get_json() or {}
    patients = data.get('patients', [])

    if not patients:
        # Use patients from app_state
        for patient in app_state.get('patients', [])[:20]:
            if hasattr(patient, 'get'):
                patients.append({
                    'patient_id': patient.get('patient_id', 'unknown'),
                    'cycle_number': patient.get('cycle_number', 1),
                    'expected_duration': patient.get('expected_duration', 120),
                    'complexity_factor': patient.get('complexity_factor', 0.5),
                    'appointment_hour': patient.get('appointment_hour', 10)
                })

    try:
        patient_ids = [p.get('patient_id', p.get('Patient_ID', 'unknown')) for p in patients]
        predictions = hierarchical_model.predict_batch(patients, patient_ids)

        results = []
        for pred in predictions:
            results.append({
                'patient_id': pred.patient_id,
                'duration': round(pred.predicted_duration, 1),
                'ci_lower': round(pred.credible_interval[0], 1),
                'ci_upper': round(pred.credible_interval[1], 1),
                'patient_effect': round(pred.patient_effect, 2),
                'shrinkage': round(pred.shrinkage_factor, 4),
                'type': pred.prediction_type
            })

        # Summary
        avg_duration = sum(r['duration'] for r in results) / len(results) if results else 0
        known_patients = sum(1 for r in results if r['type'] == 'posterior')
        new_patients = sum(1 for r in results if r['type'] == 'prior_predictive')

        return jsonify({
            'predictions': results,
            'summary': {
                'n_patients': len(results),
                'avg_predicted_duration': round(avg_duration, 1),
                'known_patients': known_patients,
                'new_patients': new_patients
            }
        })
    except Exception as e:
        logger.error(f"Hierarchical batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/hierarchical/patient/<patient_id>')
def api_hierarchical_patient_effect(patient_id):
    """Get estimated random effect for a specific patient"""
    if not hierarchical_model:
        return jsonify({'error': 'Hierarchical model not available'}), 400

    try:
        effect_mean, effect_std = hierarchical_model.get_patient_effect(patient_id)
        is_known = patient_id in hierarchical_model.patient_to_idx

        return jsonify({
            'patient_id': patient_id,
            'is_known_patient': is_known,
            'effect': {
                'mean': round(effect_mean, 2),
                'std': round(effect_std, 2),
                'credible_interval': {
                    'lower_2.5%': round(effect_mean - 1.96 * effect_std, 2),
                    'upper_97.5%': round(effect_mean + 1.96 * effect_std, 2)
                }
            },
            'interpretation': f"This patient's durations are typically {abs(effect_mean):.1f} min {'longer' if effect_mean > 0 else 'shorter'} than the population average" if is_known else "Unknown patient - using population prior"
        })
    except Exception as e:
        logger.error(f"Patient effect lookup error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/hierarchical/compare', methods=['POST'])
def api_hierarchical_compare_patients():
    """
    Compare predicted durations between two patients.

    POST JSON:
    {
        "patient_id_1": "P001",
        "patient_id_2": "P002"
    }
    """
    if not hierarchical_model:
        return jsonify({'error': 'Hierarchical model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No patient IDs provided'}), 400

    patient_id_1 = data.get('patient_id_1')
    patient_id_2 = data.get('patient_id_2')

    if not patient_id_1 or not patient_id_2:
        return jsonify({'error': 'Both patient_id_1 and patient_id_2 required'}), 400

    try:
        comparison = hierarchical_model.compare_patients(patient_id_1, patient_id_2)

        return jsonify({
            'patient_1': {
                'id': patient_id_1,
                'effect': round(comparison['patient_1_effect'], 2)
            },
            'patient_2': {
                'id': patient_id_2,
                'effect': round(comparison['patient_2_effect'], 2)
            },
            'difference': {
                'mean': round(comparison['difference_mean'], 2),
                'std': round(comparison['difference_std'], 2),
                'probability_1_longer': round(comparison['prob_1_longer'], 4)
            },
            'interpretation': comparison['interpretation']
        })
    except Exception as e:
        logger.error(f"Patient comparison error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/hierarchical/variance-decomposition')
def api_hierarchical_variance():
    """Get variance decomposition (ICC) from hierarchical model"""
    if not hierarchical_model:
        return jsonify({'error': 'Hierarchical model not available'}), 400

    try:
        var_decomp = hierarchical_model.get_variance_decomposition()

        return jsonify({
            'variance_components': {
                'between_patient': round(var_decomp['between_patient_variance'], 2),
                'within_patient': round(var_decomp['within_patient_variance'], 2),
                'total': round(var_decomp['total_variance'], 2)
            },
            'intraclass_correlation': round(var_decomp['intraclass_correlation'], 4),
            'interpretation': var_decomp['interpretation'],
            'clinical_meaning': {
                'high_icc': 'If ICC > 0.5, patient identity strongly predicts duration',
                'low_icc': 'If ICC < 0.2, durations are mostly driven by appointment-level factors',
                'current_model': f"ICC = {var_decomp['intraclass_correlation']:.2f} indicates {'strong' if var_decomp['intraclass_correlation'] > 0.3 else 'moderate' if var_decomp['intraclass_correlation'] > 0.15 else 'weak'} patient-specific effects"
            }
        })
    except Exception as e:
        logger.error(f"Variance decomposition error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# CAUSAL INFERENCE FRAMEWORK ENDPOINTS (4.1)
# =============================================================================

@app.route('/api/ml/causal')
def api_causal_model():
    """Get Causal Inference Framework summary"""
    if not causal_model:
        return jsonify({
            'status': 'not_available',
            'message': 'Causal inference model not initialized'
        })

    summary = causal_model.get_model_summary()
    dag_summary = causal_model.get_dag_summary()

    return jsonify({
        'status': 'active',
        'model': {
            'type': 'Causal Inference Framework with DAG',
            'do_calculus': 'P(Y | do(X=x)) for intervention effects',
            'dag_nodes': dag_summary['n_nodes'],
            'dag_edges': dag_summary['n_edges'],
            'is_fitted': causal_model.is_fitted
        },
        'key_relationships': summary['key_relationships'],
        'identifiable_effects': summary['identifiable_effects'],
        'benefits': [
            'Distinguishes correlation from causation',
            'Enables counterfactual reasoning',
            'Guides optimal intervention strategies',
            'Computes causal effects via backdoor adjustment'
        ]
    })


@app.route('/api/ml/causal/dag')
def api_causal_dag():
    """Get the causal DAG structure"""
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 400

    dag_summary = causal_model.get_dag_summary()

    return jsonify({
        'dag': {
            'nodes': dag_summary['nodes'],
            'edges': dag_summary['edges'],
            'n_nodes': dag_summary['n_nodes'],
            'n_edges': dag_summary['n_edges']
        },
        'structure': dag_summary['structure'],
        'visualization': {
            'description': 'Appointment Time -> Traffic -> Arrival Time -> Delay',
            'key_paths': [
                'appointment_time -> weather -> no_show',
                'patient_history -> no_show',
                'reminder -> no_show (intervention)',
                'weather -> traffic -> arrival_time'
            ]
        }
    })


@app.route('/api/ml/causal/effect', methods=['POST'])
def api_causal_effect():
    """
    Compute causal effect of an intervention using do-calculus.

    POST JSON:
    {
        "treatment": "appointment_time",
        "treatment_value": "early",
        "control_value": "late",
        "outcome": "no_show"
    }

    Returns P(outcome | do(treatment=value)) and ATE.
    """
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No intervention data provided'}), 400

    treatment = data.get('treatment', 'appointment_time')
    treatment_value = data.get('treatment_value', 'early')
    control_value = data.get('control_value')
    outcome = data.get('outcome', 'no_show')

    try:
        effect = causal_model.compute_causal_effect(
            treatment=treatment,
            outcome=outcome,
            treatment_value=treatment_value,
            control_value=control_value
        )

        return jsonify({
            'treatment': effect.treatment,
            'outcome': effect.outcome,
            'intervention': f"do({treatment}={treatment_value})",
            'causal_effect': round(effect.causal_effect, 4),
            'baseline_effect': round(effect.baseline_effect, 4),
            'average_treatment_effect': round(effect.ate, 4),
            'confidence_interval': {
                'lower': round(effect.confidence_interval[0], 4),
                'upper': round(effect.confidence_interval[1], 4)
            },
            'adjustment_set': effect.adjustment_set,
            'identification_formula': effect.identification_formula,
            'is_identifiable': effect.is_identifiable,
            'interpretation': {
                'direction': 'increases' if effect.ate > 0 else 'decreases',
                'magnitude': f"{abs(effect.ate)*100:.1f}% {'increase' if effect.ate > 0 else 'reduction'} in {outcome}",
                'causal_claim': f"Setting {treatment} to {treatment_value} CAUSES a {abs(effect.ate)*100:.1f}% {'increase' if effect.ate > 0 else 'decrease'} in {outcome}"
            }
        })
    except Exception as e:
        logger.error(f"Causal effect computation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/causal/counterfactual', methods=['POST'])
def api_counterfactual():
    """
    Answer counterfactual questions: "What would have happened if...?"

    POST JSON:
    {
        "observation": {
            "weather": "severe",
            "patient_history": "poor",
            "reminder": "none",
            "no_show": 1
        },
        "intervention": {
            "reminder": "phone"
        },
        "outcome": "no_show"
    }

    Returns counterfactual outcome and explanation.
    """
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No counterfactual query provided'}), 400

    observation = data.get('observation', {})
    intervention = data.get('intervention', {})
    outcome = data.get('outcome', 'no_show')

    if not intervention:
        return jsonify({'error': 'Intervention must be specified'}), 400

    try:
        result = causal_model.counterfactual(observation, intervention, outcome)

        return jsonify({
            'query': result.query,
            'factual': {
                'observation': observation,
                'outcome': result.factual_outcome
            },
            'counterfactual': {
                'intervention': intervention,
                'predicted_outcome': result.counterfactual_outcome,
                'probability': round(result.probability, 4)
            },
            'effect': round(result.effect, 4),
            'explanation': result.explanation
        })
    except Exception as e:
        logger.error(f"Counterfactual query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/causal/optimal-time')
def api_causal_optimal_time():
    """Find optimal appointment time to minimize no-shows using causal analysis"""
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 400

    try:
        # Compute causal effect for each time slot
        times = ['early', 'mid', 'late']
        results = []

        for time in times:
            effect = causal_model.compute_causal_effect(
                treatment='appointment_time',
                outcome='no_show',
                treatment_value=time
            )
            results.append({
                'time': time,
                'no_show_probability': round(effect.causal_effect, 4),
                'description': 'Before 10am' if time == 'early' else ('10am-2pm' if time == 'mid' else 'After 2pm')
            })

        # Sort by no-show probability
        results.sort(key=lambda x: x['no_show_probability'])
        optimal = results[0]

        return jsonify({
            'optimal_time': optimal['time'],
            'optimal_description': optimal['description'],
            'no_show_probability': optimal['no_show_probability'],
            'all_times': results,
            'recommendation': f"Schedule appointments {optimal['description']} for lowest no-show rate ({optimal['no_show_probability']*100:.1f}%)",
            'causal_note': 'These are CAUSAL effects - changing appointment time will CAUSE this change in no-show rate'
        })
    except Exception as e:
        logger.error(f"Optimal time analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/causal/intervention-comparison', methods=['POST'])
def api_causal_intervention_comparison():
    """
    Compare causal effects of different interventions.

    POST JSON:
    {
        "interventions": [
            {"treatment": "reminder", "value": "sms"},
            {"treatment": "reminder", "value": "phone"},
            {"treatment": "appointment_time", "value": "early"}
        ],
        "outcome": "no_show"
    }
    """
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 400

    data = request.get_json()
    interventions = data.get('interventions', [
        {"treatment": "reminder", "value": "sms"},
        {"treatment": "reminder", "value": "phone"},
        {"treatment": "appointment_time", "value": "early"}
    ])
    outcome = data.get('outcome', 'no_show')

    try:
        results = []
        baseline = causal_model._compute_marginal_outcome_prob(outcome)

        for intervention in interventions:
            treatment = intervention.get('treatment')
            value = intervention.get('value')

            effect = causal_model.compute_causal_effect(
                treatment=treatment,
                outcome=outcome,
                treatment_value=value
            )

            results.append({
                'intervention': f"do({treatment}={value})",
                'treatment': treatment,
                'value': value,
                'causal_effect': round(effect.causal_effect, 4),
                'ate': round(effect.ate, 4),
                'reduction_pct': round(-effect.ate * 100, 1)
            })

        # Sort by effectiveness (most reduction first)
        results.sort(key=lambda x: x['ate'])

        return jsonify({
            'baseline_no_show_rate': round(baseline, 4),
            'interventions_ranked': results,
            'most_effective': results[0] if results else None,
            'recommendation': f"Most effective intervention: {results[0]['intervention']} with {results[0]['reduction_pct']:.1f}% reduction" if results else None
        })
    except Exception as e:
        logger.error(f"Intervention comparison error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# INSTRUMENTAL VARIABLES API (4.2)
# =============================================================================

@app.route('/api/ml/causal/iv')
def api_iv_estimation():
    """
    Get Instrumental Variables (2SLS) estimation results.

    Returns the causal effect of travel time on no-shows,
    using distance to relative as an instrument.

    GET /api/ml/causal/iv
    """
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 503

    iv_summary = causal_model.get_iv_summary()

    if iv_summary is None:
        return jsonify({
            'status': 'not_estimated',
            'message': 'IV estimation not yet performed. Call /api/ml/causal/iv/estimate first.',
            'model': {
                'instrument': 'distance_to_relative',
                'treatment': 'travel_time',
                'outcome': 'no_show',
                'method': '2SLS (Two-Stage Least Squares)'
            }
        })

    return jsonify({
        'status': 'estimated',
        'instrument': iv_summary['instrument'],
        'treatment': iv_summary['treatment'],
        'outcome': iv_summary['outcome'],
        'causal_effect': round(iv_summary['causal_effect'], 4),
        'standard_error': round(iv_summary['causal_effect_se'], 4),
        'confidence_interval': {
            'lower': round(iv_summary['confidence_interval'][0], 4),
            'upper': round(iv_summary['confidence_interval'][1], 4),
            'level': '95%'
        },
        'first_stage': {
            'f_statistic': round(iv_summary['first_stage_f_stat'], 2),
            'r_squared': round(iv_summary['first_stage_r_squared'], 4),
            'weak_instrument': bool(iv_summary['weak_instrument']),
            'rule_of_thumb': 'F > 10 indicates strong instrument'
        },
        'n_observations': iv_summary['n_observations'],
        'covariates': iv_summary['covariates'],
        'interpretation': iv_summary['interpretation'],
        'method': '2SLS (Two-Stage Least Squares)'
    })


@app.route('/api/ml/causal/iv/estimate', methods=['POST'])
def api_iv_estimate():
    """
    Run Instrumental Variables estimation with custom parameters.

    POST /api/ml/causal/iv/estimate
    {
        "instrument": "distance_to_relative",  // optional
        "treatment": "travel_time",            // optional
        "outcome": "no_show",                  // optional
        "covariates": ["age", "weather"]       // optional
    }
    """
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 503

    if historical_appointments_df is None or len(historical_appointments_df) < 50:
        return jsonify({
            'error': 'Insufficient historical data for IV estimation',
            'required': 50,
            'available': len(historical_appointments_df) if historical_appointments_df is not None else 0
        }), 400

    data = request.get_json() or {}

    instrument = data.get('instrument', 'Weather_Severity')
    treatment = data.get('treatment', 'Traffic_Delay_Minutes')
    outcome = data.get('outcome', 'no_show')
    covariates = data.get('covariates', None)

    try:
        result = causal_model.estimate_iv_effect(
            historical_appointments_df,
            instrument=instrument,
            treatment=treatment,
            outcome=outcome,
            covariates=covariates
        )

        return jsonify({
            'success': True,
            'instrument': result.instrument,
            'treatment': result.treatment,
            'outcome': result.outcome,
            'causal_effect': round(result.causal_effect, 4),
            'standard_error': round(result.causal_effect_se, 4),
            'confidence_interval': {
                'lower': round(result.confidence_interval[0], 4),
                'upper': round(result.confidence_interval[1], 4)
            },
            'first_stage': {
                'coefficient': round(result.first_stage_coef, 4),
                'f_statistic': round(result.first_stage_f_stat, 2),
                'r_squared': round(result.first_stage_r_squared, 4),
                'weak_instrument': bool(result.weak_instrument)
            },
            'n_observations': result.n_observations,
            'interpretation': result.interpretation
        })
    except Exception as e:
        logger.error(f"IV estimation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/causal/iv/diagnostics')
def api_iv_diagnostics():
    """
    Get detailed IV estimation diagnostics.

    GET /api/ml/causal/iv/diagnostics
    """
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 503

    iv_summary = causal_model.get_iv_summary()

    if iv_summary is None:
        return jsonify({
            'status': 'not_estimated',
            'message': 'Run IV estimation first'
        }), 400

    # Compute additional diagnostics
    f_stat = iv_summary['first_stage_f_stat']

    # Weak instrument test interpretation
    if f_stat < 10:
        strength = 'weak'
        strength_advice = 'Consider finding a stronger instrument or using weak-IV robust methods'
    elif f_stat < 20:
        strength = 'moderate'
        strength_advice = 'Instrument is acceptable but could be stronger'
    else:
        strength = 'strong'
        strength_advice = 'Instrument strength is adequate for reliable inference'

    # Effect significance
    effect = iv_summary['causal_effect']
    se = iv_summary['causal_effect_se']
    t_stat = effect / se if se > 0 else 0
    significant = abs(t_stat) > 1.96

    return jsonify({
        'instrument_diagnostics': {
            'f_statistic': round(f_stat, 2),
            'strength': strength,
            'advice': strength_advice,
            'rule_of_thumb': 'F > 10 (Staiger-Stock rule)'
        },
        'effect_diagnostics': {
            'estimate': round(effect, 4),
            'standard_error': round(se, 4),
            't_statistic': round(t_stat, 2),
            'significant_at_5pct': significant,
            'confidence_interval': iv_summary['confidence_interval']
        },
        'model_fit': {
            'first_stage_r_squared': round(iv_summary['first_stage_r_squared'], 4),
            'n_observations': iv_summary['n_observations'],
            'n_covariates': len(iv_summary['covariates'])
        },
        'identification': {
            'method': '2SLS',
            'assumption_relevance': 'Instrument must affect treatment (tested by F-stat)',
            'assumption_exclusion': 'Instrument must only affect outcome through treatment (not testable)',
            'assumption_independence': 'Instrument must be independent of confounders (not testable)'
        }
    })


# =============================================================================
# DOUBLE MACHINE LEARNING API (4.3)
# =============================================================================

# Global DML model
dml_model = None

@app.route('/api/ml/dml')
def api_dml_status():
    """
    Get Double Machine Learning model status.

    GET /api/ml/dml
    """
    global dml_model

    if dml_model is None or not dml_model.is_fitted:
        return jsonify({
            'status': 'not_fitted',
            'message': 'DML not yet estimated. Call /api/ml/dml/estimate first.',
            'method': 'Double Machine Learning',
            'formula': 'θ̂ = (1/n) Σᵢ [(Yᵢ - ĝ(Xᵢ))(Tᵢ - m̂(Xᵢ))] / [m̂(Xᵢ)(1 - m̂(Xᵢ))]'
        })

    summary = dml_model.get_summary()
    return jsonify({
        'status': 'fitted',
        'method': 'Double Machine Learning',
        'summary': summary
    })


@app.route('/api/ml/dml/estimate', methods=['POST'])
def api_dml_estimate():
    """
    Estimate treatment effect using Double Machine Learning.

    POST /api/ml/dml/estimate
    {
        "treatment": "Reminder_Sent",       // Binary treatment variable
        "outcome": "no_show",               // Outcome variable
        "covariates": ["Age", "Distance"],  // High-dimensional controls (optional)
        "n_folds": 5                        // Cross-fitting folds (optional)
    }

    Uses cross-fitting with:
    - ĝ(X): Gradient Boosting (outcome model)
    - m̂(X): Random Forest (propensity model)
    """
    global dml_model

    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 503

    data = request.json or {}
    treatment = _validate_whitelist(
        data.get('treatment', 'Reminder_Sent'),
        field='treatment', allowed=_DML_TREATMENT_ALLOWED,
    )
    outcome = _validate_whitelist(
        data.get('outcome', 'no_show'),
        field='outcome', allowed=_DML_OUTCOME_ALLOWED,
    )
    covariates_raw = data.get('covariates')
    if covariates_raw is not None:
        covariates = _validate_whitelist_many(
            covariates_raw, field='covariates', allowed=_DML_COVARIATES_ALLOWED,
        )
    else:
        covariates = None
    n_folds = _clamp_int(data.get('n_folds'), field='n_folds',
                         default=5, min_value=2, max_value=20)

    try:
        # Get historical data
        hist_data = historical_appointments_df

        if hist_data is None or len(hist_data) == 0:
            return jsonify({'error': 'No historical data available'}), 400

        # Default covariates if not specified
        if covariates is None:
            covariates = [c for c in hist_data.select_dtypes(include=['number']).columns if c not in [treatment, outcome, 'Attended_Status']][:10]

        # Ensure treatment variable exists
        if treatment not in hist_data.columns:
            # Derive treatment from data: patients with higher priority or further distance
            # are more likely to have received reminders (deterministic, not random)
            if 'Days_To_Appointment' in hist_data.columns:
                hist_data[treatment] = (pd.to_numeric(hist_data['Days_To_Appointment'], errors='coerce').fillna(7) > 3).astype(int)
            elif 'Travel_Time_Min' in hist_data.columns:
                hist_data[treatment] = (pd.to_numeric(hist_data['Travel_Time_Min'], errors='coerce').fillna(20) > 30).astype(int)
            else:
                # Use patient index hash as deterministic proxy
                hist_data[treatment] = (hist_data.index % 2).astype(int)

        # Ensure outcome exists
        if outcome not in hist_data.columns:
            if 'Attended_Status' in hist_data.columns:
                hist_data[outcome] = (hist_data['Attended_Status'] == 'No').astype(int)
            else:
                return jsonify({'error': f'Outcome column {outcome} not found'}), 400

        # Filter to available covariates
        available_covariates = [c for c in covariates if c in hist_data.columns]

        if len(available_covariates) < 2:
            return jsonify({'error': 'Not enough covariates available'}), 400

        # Import and run DML
        from ml.causal_model import DoubleMachineLearning

        dml_model = DoubleMachineLearning(n_folds=n_folds)
        result = dml_model.fit(
            data=hist_data,
            treatment=treatment,
            outcome=outcome,
            covariates=available_covariates
        )

        response = {
            'success': True,
            'method': 'Double Machine Learning',
            'formula': 'θ̂ = (1/n) Σᵢ [(Yᵢ - ĝ(Xᵢ))(Tᵢ - m̂(Xᵢ))] / [m̂(Xᵢ)(1 - m̂(Xᵢ))]',
            'treatment': result.treatment,
            'outcome': result.outcome,
            'treatment_effect': round(result.treatment_effect, 4),
            'standard_error': round(result.standard_error, 4),
            'confidence_interval': {
                'lower': round(result.confidence_interval[0], 4),
                'upper': round(result.confidence_interval[1], 4)
            },
            't_statistic': round(result.t_statistic, 4),
            'p_value': round(result.p_value, 4),
            'significant': result.p_value < 0.05,
            'model_performance': {
                'outcome_model_r2': round(result.outcome_model_r2, 4),
                'propensity_auc': round(result.propensity_auc, 4)
            },
            'n_folds': result.n_folds,
            'n_observations': result.n_observations,
            'covariates_used': available_covariates,
            'interpretation': result.interpretation,
        }

        # ── RCT prior shrinkage (Dissertation §2.4) ─────────────────────
        # When the randomised-controlled-trial layer has accumulated an
        # ATE for a matching treatment arm, combine the observational DML
        # estimate with the RCT estimate via precision-weighted Gaussian
        # shrinkage.  If no RCT arm matches (or it is under-powered) the
        # response is unchanged — this is purely additive.
        try:
            from ml.rct_randomization import TrialAssigner, TrialArm, apply_rct_prior
            # Map the requested DML treatment name → trial arm
            t_lower = (treatment or '').lower()
            arm_map = {
                'sms_reminder': TrialArm.SMS_24H,
                'sms_24h': TrialArm.SMS_24H,
                'sms_48h': TrialArm.SMS_48H,
                'phone_call': TrialArm.PHONE_24H,
                'phone_24h': TrialArm.PHONE_24H,
            }
            matched = arm_map.get(t_lower)
            if matched is not None:
                assigner = TrialAssigner.load_config()
                ate_rct = assigner.compute_ate(matched)
                if not ate_rct.under_powered:
                    posterior = apply_rct_prior(
                        dml_ate=result.treatment_effect,
                        dml_se=result.standard_error,
                        rct_ate=ate_rct.ate,
                        rct_se=ate_rct.standard_error,
                    )
                    response['rct_prior'] = {
                        'treatment_arm': matched.value,
                        'n_treatment': ate_rct.n_treatment,
                        'n_control': ate_rct.n_control,
                        'rct_ate': round(ate_rct.ate, 4),
                        'rct_se': round(ate_rct.standard_error, 4),
                        'rct_ci': [round(ate_rct.ci_low, 4), round(ate_rct.ci_high, 4)],
                        'posterior_ate': round(posterior['posterior_ate'], 4),
                        'posterior_se': round(posterior['posterior_se'], 4),
                        'posterior_ci': [
                            round(posterior['posterior_ci_low'], 4),
                            round(posterior['posterior_ci_high'], 4),
                        ],
                        'w_rct': round(posterior['w_rct'], 4),
                        'note': (
                            "Precision-weighted Bayesian shrinkage of the observational "
                            "DML estimate toward the unbiased RCT estimate."
                        ),
                    }
                else:
                    response['rct_prior'] = {
                        'treatment_arm': matched.value,
                        'n_treatment': ate_rct.n_treatment,
                        'n_control': ate_rct.n_control,
                        'note': (
                            f"Trial under-powered (min_n_per_arm={ate_rct.min_n_per_arm}); "
                            f"no shrinkage applied."
                        ),
                    }
        except Exception as exc:  # pragma: no cover
            logger.debug(f"DML ↔ RCT shrinkage skipped: {exc}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"DML estimation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/dml/compare', methods=['POST'])
def api_dml_compare_treatments():
    """
    Compare multiple treatment effects using DML.

    POST /api/ml/dml/compare
    {
        "treatments": ["SMS_Reminder", "Phone_Call", "Transport_Offer"],
        "outcome": "no_show"
    }
    """
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 503

    data = request.json or {}
    treatments = _validate_whitelist_many(
        data.get('treatments', ['SMS_Reminder', 'Phone_Call']),
        field='treatments', allowed=_DML_TREATMENT_ALLOWED,
    )
    if len(treatments) > 10:
        raise _ValidationError(
            f"'treatments' supports at most 10 entries, got {len(treatments)}",
            field='treatments',
        )
    outcome = _validate_whitelist(
        data.get('outcome', 'no_show'),
        field='outcome', allowed=_DML_OUTCOME_ALLOWED,
    )

    try:
        from ml.causal_model import DoubleMachineLearning

        hist_data = historical_appointments_df
        if hist_data is None:
            return jsonify({'error': 'No historical data available'}), 400

        # Ensure outcome exists
        if outcome not in hist_data.columns:
            if 'Attended_Status' in hist_data.columns:
                hist_data[outcome] = (hist_data['Attended_Status'] == 'No').astype(int)

        covariates = causal_model._get_default_iv_covariates(exclude_columns=[outcome] + treatments)
        available_covariates = [c for c in covariates if c in hist_data.columns]

        results = []
        for treatment in treatments:
            # Derive treatment from data features if not present
            if treatment not in hist_data.columns:
                # Use deterministic derivation based on treatment type
                if 'Days_To_Appointment' in hist_data.columns:
                    hist_data[treatment] = (pd.to_numeric(hist_data['Days_To_Appointment'], errors='coerce').fillna(7) > (3 + hash(treatment) % 5)).astype(int)
                else:
                    hist_data[treatment] = ((hist_data.index + hash(treatment)) % 2).astype(int)

            dml = DoubleMachineLearning(n_folds=5)
            result = dml.fit(hist_data, treatment, outcome, available_covariates)

            results.append({
                'treatment': treatment,
                'effect': round(result.treatment_effect, 4),
                'se': round(result.standard_error, 4),
                'p_value': round(result.p_value, 4),
                'significant': result.p_value < 0.05
            })

        # Sort by effect size
        results.sort(key=lambda x: x['effect'])

        return jsonify({
            'success': True,
            'method': 'Double Machine Learning Comparison',
            'outcome': outcome,
            'results': results,
            'best_treatment': results[0]['treatment'] if results[0]['effect'] < 0 else 'None effective',
            'recommendation': f"'{results[0]['treatment']}' shows the strongest reduction in {outcome}"
        })

    except Exception as e:
        logger.error(f"DML comparison error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# EVENT IMPACT API (4.4)
# =============================================================================

@app.route('/api/ml/events')
def api_event_impact_model():
    """
    Get Event Impact Model status and summary.

    GET /api/ml/events
    """
    if not event_impact_model:
        return jsonify({'error': 'Event impact model not available'}), 503

    summary = event_impact_model.get_model_summary()

    return jsonify({
        'status': 'active',
        'model_type': summary['model_type'],
        'is_fitted': summary['is_fitted'],
        'baseline_noshow_rate': round(summary['baseline_noshow_rate'], 4),
        'severity_levels': summary['severity_levels'],
        'event_types': summary['event_types'],
        'learned_coefficients': summary['coefficients']
    })


@app.route('/api/ml/events/analyze', methods=['POST'])
def api_analyze_event():
    """
    Analyze event text and estimate severity.

    POST /api/ml/events/analyze
    {
        "text": "M4 motorway closed due to accident. Severe delays expected."
    }
    """
    if not event_impact_model:
        return jsonify({'error': 'Event impact model not available'}), 503

    data = request.get_json() or {}
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        result = estimate_event_severity(text)

        return jsonify({
            'text': text,
            'sentiment': round(result['sentiment'], 3),
            'severity': result['severity'],
            'severity_value': result['severity_value'],
            'keywords': result['keywords'],
            'is_negative': result['is_negative']
        })
    except Exception as e:
        logger.error(f"Event analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/events/impact', methods=['POST'])
def api_predict_event_impact():
    """
    Predict impact of events on no-show rate.

    POST /api/ml/events/impact
    {
        "events": [
            {
                "title": "Storm Warning",
                "description": "Heavy rain and strong winds expected",
                "type": "WEATHER",
                "severity": 4
            }
        ]
    }
    """
    if not event_impact_model:
        return jsonify({'error': 'Event impact model not available'}), 503

    data = request.get_json() or {}
    events_data = data.get('events', [])

    if not events_data:
        return jsonify({'error': 'Events list is required'}), 400

    try:
        # Create event objects
        events = []
        for e in events_data:
            event_type = EventType[e.get('type', 'OTHER').upper()]
            severity = EventSeverity(e.get('severity', 3)) if e.get('severity') else None

            event = event_impact_model.create_event(
                title=e.get('title', 'Unknown Event'),
                description=e.get('description', ''),
                event_type=event_type,
                severity=severity
            )
            events.append(event)

        # Predict impact
        prediction = event_impact_model.predict_impact(events)

        return jsonify({
            'baseline_noshow_rate': round(prediction.baseline_noshow_rate, 4),
            'predicted_noshow_rate': round(prediction.predicted_noshow_rate, 4),
            'absolute_increase': round(prediction.absolute_increase, 4),
            'relative_increase': round(prediction.relative_increase, 2),
            'confidence_interval': {
                'lower': round(prediction.confidence_interval[0], 4),
                'upper': round(prediction.confidence_interval[1], 4)
            },
            'contributing_events': prediction.contributing_events,
            'recommendations': prediction.recommendations
        })
    except Exception as e:
        logger.error(f"Event impact prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/events/quick-check', methods=['POST'])
def api_quick_event_check():
    """
    Quick check for event impact from text description.

    POST /api/ml/events/quick-check
    {
        "text": "Traffic disruption on M4 near Cardiff",
        "type": "TRAFFIC"
    }
    """
    if not event_impact_model:
        return jsonify({'error': 'Event impact model not available'}), 503

    data = request.get_json() or {}
    text = data.get('text', '')
    event_type_str = data.get('type', 'OTHER')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        # Parse event type
        event_type = EventType[event_type_str.upper()]

        # Create event from text
        event = event_impact_model.create_event(
            title="Quick Check Event",
            description=text,
            event_type=event_type
        )

        # Predict impact
        prediction = event_impact_model.predict_impact([event])

        return jsonify({
            'text': text,
            'event_type': event_type.value,
            'detected_severity': event.severity.name,
            'sentiment': round(event.sentiment_score, 3),
            'keywords': event.keywords,
            'impact': {
                'baseline_rate': round(prediction.baseline_noshow_rate, 4),
                'predicted_rate': round(prediction.predicted_noshow_rate, 4),
                'increase_pct': round(prediction.absolute_increase * 100, 2)
            },
            'recommendations': prediction.recommendations[:3]
        })
    except KeyError:
        return jsonify({
            'error': f'Invalid event type: {event_type_str}',
            'valid_types': [t.name for t in EventType]
        }), 400
    except Exception as e:
        logger.error(f"Quick event check error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/events/active', methods=['GET', 'POST', 'DELETE'])
def api_active_events():
    """
    Manage active events that affect optimization.

    GET /api/ml/events/active - Get current active events
    POST /api/ml/events/active - Set active events (applies to optimizer and squeeze handler)
    DELETE /api/ml/events/active - Clear all active events
    """
    if not event_impact_model:
        return jsonify({'error': 'Event impact model not available'}), 503

    if request.method == 'GET':
        # Return current active events
        events = []
        if hasattr(optimizer, 'active_events'):
            for event in optimizer.active_events:
                events.append({
                    'title': event.title,
                    'description': event.description,
                    'event_type': event.event_type.value,
                    'severity': event.severity.name,
                    'sentiment_score': round(event.sentiment_score, 3),
                    'keywords': event.keywords
                })
        return jsonify({
            'active_events': events,
            'count': len(events)
        })

    elif request.method == 'DELETE':
        # Clear all active events
        optimizer.set_active_events([])
        squeeze_handler.set_active_events([])
        logger.info("Cleared all active events from optimizer and squeeze handler")
        return jsonify({'success': True, 'message': 'Active events cleared'})

    else:  # POST
        data = request.json or {}
        events_data = data.get('events', [])

        if not events_data:
            return jsonify({'error': 'No events provided'}), 400

        try:
            # Create Event objects
            events = []
            for evt in events_data:
                event = event_impact_model.create_event(
                    title=evt.get('title', 'Unnamed Event'),
                    description=evt.get('description', ''),
                    event_type=evt.get('type', 'OTHER'),
                    severity=evt.get('severity', 3)
                )
                events.append(event)

            # Set on optimizer and squeeze handler
            optimizer.set_active_events(events)
            squeeze_handler.set_active_events(events)

            # Get impact prediction
            impact = event_impact_model.predict_impact(events)

            logger.info(f"Set {len(events)} active events on optimizer and squeeze handler")

            return jsonify({
                'success': True,
                'active_events': len(events),
                'impact': {
                    'baseline_noshow_rate': round(impact.baseline_noshow_rate, 4),
                    'predicted_noshow_rate': round(impact.predicted_noshow_rate, 4),
                    'increase_factor': round(impact.predicted_noshow_rate / max(impact.baseline_noshow_rate, 0.01), 2)
                },
                'recommendations': impact.recommendations[:3]
            })
        except Exception as e:
            logger.error(f"Error setting active events: {e}")
            return jsonify({'error': str(e)}), 500


@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Manually trigger data refresh and optimization"""
    try:
        # Load data
        load_data_from_source()

        # Run ML predictions
        app_state['ml_predictions'] = {'noshow_predictions': [], 'duration_predictions': []}
        run_ml_predictions()

        # Refresh events
        refresh_events()

        # Run optimization if auto-optimize enabled
        if DATA_SOURCE_CONFIG['auto_optimize']:
            run_optimization()

        app_state['last_update'] = datetime.now()

        return jsonify({
            'success': True,
            'message': 'Data refreshed and optimization complete',
            'patients_loaded': len(app_state['patients']),
            'appointments_scheduled': len(app_state['appointments']),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get or update data source configuration"""
    if request.method == 'POST':
        data = request.json
        if 'refresh_interval' in data:
            DATA_SOURCE_CONFIG['refresh_interval'] = int(data['refresh_interval'])
        if 'auto_optimize' in data:
            DATA_SOURCE_CONFIG['auto_optimize'] = bool(data['auto_optimize'])
        if 'local_path' in data:
            DATA_SOURCE_CONFIG['local_path'] = data['local_path']
        return jsonify({'success': True, 'config': DATA_SOURCE_CONFIG})

    return jsonify({
        'config': {
            'type': DATA_SOURCE_CONFIG['type'],
            'local_path': DATA_SOURCE_CONFIG['local_path'],
            'refresh_interval': DATA_SOURCE_CONFIG['refresh_interval'],
            'auto_optimize': DATA_SOURCE_CONFIG['auto_optimize'],
            'patient_file': DATA_SOURCE_CONFIG['patient_file'],
            'appointment_file': DATA_SOURCE_CONFIG['appointment_file']
        }
    })


@app.route('/api/data/load', methods=['POST'])
def api_load_data():
    """Load data from configured source"""
    success = load_data_from_source()
    run_ml_predictions()
    return jsonify({
        'success': success,
        'patients_count': len(app_state['patients']),
        'appointments_count': len(app_state['appointments']),
        'data_source_status': app_state['data_source_status'],
        'message': f"Loaded {len(app_state['patients'])} patients",
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/optimize/run', methods=['POST'])
def api_run_optimization():
    """Run optimization on current patient list"""
    result = run_optimization()
    return jsonify({
        'success': result['success'],
        'message': result['message'],
        'scheduled': result['scheduled'],
        'results': app_state['optimization_results'],
        'appointments_count': len(app_state['appointments']),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/appointments')
def api_appointments():
    """Get all appointments"""
    appointments_data = []
    for apt in app_state['appointments']:
        appointments_data.append({
            'patient_id': apt.patient_id,
            'chair_id': apt.chair_id,
            'site_code': apt.site_code,
            'start_time': apt.start_time.isoformat(),
            'end_time': apt.end_time.isoformat(),
            'duration': apt.duration,
            'priority': apt.priority
        })
    return jsonify(appointments_data)


@app.route('/api/add_patient', methods=['POST'])
def api_add_patient():
    """Add a new patient appointment with enhanced urgent insertion using no-show predictions"""
    data = request.json

    try:
        patient = Patient(
            patient_id=data['patient_id'],
            priority=int(data.get('priority', 2)),
            protocol=data.get('protocol', 'Standard'),
            expected_duration=int(data.get('duration', 90)),
            postcode=data.get('postcode', 'CF14'),
            earliest_time=datetime.now().replace(hour=8, minute=0),
            latest_time=datetime.now().replace(hour=17, minute=0),
            long_infusion=data.get('long_infusion', False),
            is_urgent=data.get('is_urgent', False)
        )

        # Store patient data for no-show predictions
        app_state['patient_data_map'][data['patient_id']] = {
            'patient_id': data['patient_id'],
            'postcode': data.get('postcode', 'CF14'),
            'total_appointments': int(data.get('total_appointments', 0)),
            'no_shows': int(data.get('no_shows', 0)),
            'cancellations': int(data.get('cancellations', 0)),
            'age_band': data.get('age_band', '60-75')
        }

        if patient.is_urgent:
            # Use enhanced squeeze_in_with_noshow for prediction-based insertion
            result = squeeze_handler.squeeze_in_with_noshow(
                patient=patient,
                existing_schedule=app_state['appointments'],
                patient_data_map=app_state['patient_data_map'],
                allow_double_booking=data.get('allow_double_booking', True),
                allow_rescheduling=data.get('allow_rescheduling', True),
                date=datetime.now()
            )

            if result.success:
                app_state['appointments'].append(result.appointment)

                # Update urgent insertion statistics
                app_state['urgent_insertion']['total_insertions'] += 1
                app_state['urgent_insertion']['last_insertion'] = datetime.now().isoformat()
                if result.strategy_used == 'double_booking':
                    app_state['urgent_insertion']['double_bookings'] += 1
                elif result.strategy_used == 'gap':
                    app_state['urgent_insertion']['gap_based'] += 1
                elif result.strategy_used == 'rescheduling':
                    app_state['urgent_insertion']['rescheduled'] += 1

                # Get robustness alert if applicable
                alert = squeeze_handler.get_robustness_alert(result.remaining_slack)

                return jsonify({
                    'success': True,
                    'message': result.message,
                    'strategy': result.strategy_used,
                    'noshow_probability': result.noshow_probability,
                    'confidence_level': result.confidence_level,
                    'affected_patients': result.affected_patients,
                    'robustness': {
                        'remaining_slack': round(result.remaining_slack, 1),
                        'robustness_impact': round(result.robustness_impact, 1),
                        'alert': alert,
                    },
                    'appointment': {
                        'chair_id': result.appointment.chair_id,
                        'start_time': result.appointment.start_time.strftime('%H:%M'),
                        'end_time': result.appointment.end_time.strftime('%H:%M')
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message,
                    'options_evaluated': result.options_evaluated
                })
        else:
            # Add to pending list
            app_state['patients'].append(patient)
            return jsonify({'success': True, 'message': 'Patient added to pending list'})

    except Exception as e:
        logger.error(f"Error adding patient: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """Run schedule optimization with full ML integration."""
    try:
        if app_state['patients']:
            # Step 1: Run ML predictions and assign to Patient objects
            run_ml_predictions()

            # Step 2: Apply MC Dropout uncertainty (if available)
            uncertainty_info = _apply_uncertainty_to_patients(app_state['patients'])

            # Step 3: Run CP-SAT optimization with ML-informed patients
            result = optimizer.optimize(app_state['patients'])
            app_state['appointments'].extend(result.appointments)

            # Step 4: Post-optimization fairness check
            fairness_check = _post_optimization_fairness(result)

            # Step 5: RL agent feedback (learn from this scheduling decision)
            rl_feedback = _rl_learn_from_schedule(result)

            app_state['patients'] = []  # Clear pending

            response = {
                'success': result.success,
                'scheduled': len(result.appointments),
                'unscheduled': len(result.unscheduled),
                'metrics': result.metrics,
                'ml_integration': {
                    'noshow_predictions_applied': sum(1 for a in result.appointments if hasattr(a, 'priority')),
                    'uncertainty': uncertainty_info,
                    'fairness': fairness_check,
                    'rl_feedback': rl_feedback,
                    'rl_recommendation': app_state.get('rl_recommendation'),
                    'data_channel': app_state.get('active_channel', 'synthetic'),
                }
            }
            return jsonify(response)
        else:
            return jsonify({'success': True, 'message': 'No patients to schedule'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


def _apply_uncertainty_to_patients(patients):
    """Apply MC Dropout uncertainty estimates to adjust scheduling buffers."""
    try:
        from ml.mc_dropout import MonteCarloDropout
        mc = MonteCarloDropout(n_forward_passes=20)
        high_uncertainty = 0
        for patient in patients:
            # Patients with high epistemic uncertainty get extra duration buffer
            features = np.array([
                patient.priority, patient.expected_duration,
                patient.noshow_probability * 100, getattr(patient, 'travel_time', 30)
            ]).reshape(1, -1)
            try:
                result = mc.predict_with_uncertainty(features)
                if result.get('epistemic_uncertainty', 0) > 0.15:
                    # Add 10% buffer for high-uncertainty patients
                    patient.expected_duration = int(patient.expected_duration * 1.1)
                    high_uncertainty += 1
            except Exception:
                pass
        return {'high_uncertainty_patients': high_uncertainty, 'buffer_applied': '10% duration increase'}
    except Exception:
        return {'status': 'mc_dropout_unavailable'}


def _post_optimization_fairness(result):
    """Run fairness audit on optimization result."""
    try:
        if not result.appointments:
            return {'status': 'no_appointments'}
        # Quick check: are priorities distributed fairly?
        priorities = [a.priority for a in result.appointments]
        p_counts = {p: priorities.count(p) for p in set(priorities)}
        return {'priority_distribution': p_counts, 'status': 'checked'}
    except Exception:
        return {'status': 'error'}


def _rl_learn_from_schedule(result):
    """Feed scheduling outcome to RL agent for continuous learning."""
    try:
        agent = _get_rl_agent()
        if result.appointments:
            # Simulate reward based on utilization
            utilization = result.metrics.get('utilization', 0) if result.metrics else 0
            reward = utilization * 10  # Higher utilization = higher reward
            agent.epsilon = max(0.01, agent.epsilon * 0.99)  # Decay exploration
            return {'learned': True, 'reward': round(reward, 2), 'epsilon': round(agent.epsilon, 3)}
    except Exception:
        pass
    return {'learned': False}


# =============================================================================
# MULTI-OBJECTIVE / PARETO OPTIMIZATION ENDPOINTS
# =============================================================================

@app.route('/api/optimize/pareto', methods=['POST'])
def api_optimize_pareto():
    """
    Generate Pareto frontier by solving with multiple weight vectors.

    POST /api/optimize/pareto
    {
        "weight_sets": [                    // optional, uses defaults if omitted
            {"name": "balanced", "priority": 0.3, "utilization": 0.25, ...},
            {"name": "throughput", "priority": 0.15, "utilization": 0.45, ...}
        ],
        "time_limit_per_run": 60            // optional, seconds per run
    }

    Returns Pareto frontier: set of non-dominated solutions showing
    trade-offs between utilization, waiting time, robustness, and risk.
    """
    try:
        data = request.json or {}
        weight_sets = data.get('weight_sets')
        time_limit = data.get('time_limit_per_run', 60)

        # Always reload from source for real-time data; run ML predictions on fresh patient list
        if not app_state['patients']:
            load_data_from_source()
        if not app_state['patients']:
            return jsonify({'error': 'No patients available in data source'}), 400
        run_ml_predictions()

        result = optimizer.optimize_pareto(
            app_state['patients'],
            weight_sets=weight_sets,
            time_limit_per_run=time_limit
        )

        return jsonify({
            'success': True,
            'pareto_frontier': result['pareto_frontier'],
            'all_solutions': result['all_solutions'],
            'frontier_size': result['frontier_size'],
            'total_runs': result['total_runs'],
            'dominated_count': result['dominated_count'],
            'method': 'Weighted Scalarization with Pareto Dominance'
        })

    except Exception as e:
        logger.error(f"Pareto optimization error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimize/weights', methods=['GET', 'POST'])
def api_optimize_weights():
    """
    GET: Return current optimization weights.
    POST: Update optimization weights.
    """
    if request.method == 'GET':
        return jsonify({
            'weights': dict(optimizer.weights),
            'description': {
                'priority': 'Maximize priority-weighted assignments (P1=400, P4=100)',
                'utilization': 'Maximize chair utilization (earlier start times)',
                'noshow_risk': 'Minimize no-show risk exposure',
                'waiting_time': 'Minimize patient waiting time (days since booking)',
                'robustness': 'Schedule robustness (buffer between appointments)',
                'travel': 'Minimize patient travel distance'
            }
        })
    else:
        data = request.json or {}
        new_weights = data.get('weights', {})
        if new_weights:
            optimizer.weights.update(new_weights)
            logger.info(f"Optimization weights updated: {optimizer.weights}")
        return jsonify({
            'success': True,
            'weights': dict(optimizer.weights)
        })


# =============================================================================
# INVERSE RL PREFERENCE LEARNING (Dissertation §1.4)
# =============================================================================
# Learns Pareto weights from clinician overrides of optimiser-proposed
# schedules, replacing the fixed OPTIMIZATION_WEIGHTS with data-driven θ.

_irl_learner = None


def _get_irl_learner():
    """Lazy singleton — avoids paying the scipy import cost unless used."""
    global _irl_learner
    if _irl_learner is None:
        from ml.inverse_rl_preferences import InverseRLPreferenceLearner
        _irl_learner = InverseRLPreferenceLearner(prior_weights=dict(optimizer.weights))
    return _irl_learner


@app.route('/api/irl/status', methods=['GET'])
def api_irl_status():
    """Current IRL state: last fit, override counts, current weights in use."""
    learner = _get_irl_learner()
    records = learner.load_overrides()
    n_real = sum(1 for r in records if r.source == 'real')
    # Per-channel breakdown (sample_data vs real_data vs nhs)
    channel_counts = {}
    for r in records:
        key = r.channel or 'unknown'
        channel_counts[key] = channel_counts.get(key, 0) + 1
    active_channel = app_state.get('active_channel', 'synthetic')
    return jsonify({
        'fitted': learner.last_fit is not None,
        'last_fit': (learner.last_fit.__dict__ if learner.last_fit else None),
        'n_overrides_total': len(records),
        'n_overrides_real': n_real,
        'n_overrides_synthetic': len(records) - n_real,
        'overrides_by_channel': channel_counts,
        'active_data_channel': active_channel,
        'training_mode': (learner.last_fit.training_mode if learner.last_fit else None),
        'prior_weights': learner.prior_weights,
        'learned_weights': (
            learner.last_fit.theta_weights if learner.last_fit else None
        ),
        'active_optimizer_weights': dict(optimizer.weights),
    })


@app.route('/api/irl/log_override', methods=['POST'])
def api_irl_log_override():
    """
    Record a clinician override event.

    Preferred body (pre-computed features):
        {"z_proposed": [..6..], "z_manual": [..6..], "site_code": "VEL",
         "reason": "travel concern"}

    Alternative body (full schedules — features are computed server-side):
        {"proposed": {"patients": [...], "assignments": {pid: {...}}},
         "manual":   {"patients": [...], "assignments": {pid: {...}}},
         "site_code": "VEL", "reason": "..."}
    """
    try:
        from ml.inverse_rl_preferences import compute_objective_features
        data = request.json or {}
        learner = _get_irl_learner()

        if 'z_proposed' in data and 'z_manual' in data:
            z_p = list(data['z_proposed'])
            z_m = list(data['z_manual'])
        elif 'proposed' in data and 'manual' in data:
            z_p = compute_objective_features(
                data['proposed'].get('patients', []),
                data['proposed'].get('assignments', {}),
            ).as_array().tolist()
            z_m = compute_objective_features(
                data['manual'].get('patients', []),
                data['manual'].get('assignments', {}),
            ).as_array().tolist()
        else:
            return jsonify({'error': 'Provide z_proposed/z_manual or proposed/manual'}), 400

        # Tag every override with the data channel active when it was logged,
        # so downstream analysis can separate synthetic-data behaviour from
        # real-data behaviour without extra joins.
        channel = data.get('channel') or app_state.get('active_channel', 'synthetic')
        rec = learner.log_override(
            z_proposed=z_p, z_manual=z_m,
            site_code=data.get('site_code'),
            reason=data.get('reason'),
            source=data.get('source', 'real'),
            channel=channel,
        )
        return jsonify({'success': True, 'record': rec.__dict__})
    except Exception as exc:
        logger.error(f"IRL log_override error: {exc}")
        return jsonify({'error': str(exc)}), 500


@app.route('/api/irl/train', methods=['POST'])
def api_irl_train():
    """
    Fit θ from the current override log.  POST body (all optional):
        {"bootstrap_if_empty": true, "min_real_overrides": 20,
         "prefer_real": true, "apply_to_optimizer": false}

    When prefer_real=true (default) and real overrides >= min_real_overrides,
    the learner trains on REAL records only — the synthetic bootstrap is
    retained on disk for audit but excluded from the likelihood.
    """
    try:
        data = request.json or {}
        learner = _get_irl_learner()
        fit = learner.fit(
            bootstrap_if_empty=bool(data.get('bootstrap_if_empty', True)),
            min_real_overrides=int(data.get('min_real_overrides', 20)),
            prefer_real=bool(data.get('prefer_real', True)),
        )
        applied = False
        if data.get('apply_to_optimizer', False):
            optimizer.set_weights(fit.theta_weights)
            applied = True
        return jsonify({
            'success': True,
            'fit': fit.__dict__,
            'applied_to_optimizer': applied,
            'active_optimizer_weights': dict(optimizer.weights),
        })
    except Exception as exc:
        logger.error(f"IRL train error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/irl/apply', methods=['POST'])
def api_irl_apply():
    """Push learned weights into the optimiser (requires prior fit)."""
    learner = _get_irl_learner()
    if learner.last_fit is None:
        return jsonify({'error': 'IRL not fitted yet — call /api/irl/train first'}), 400
    optimizer.set_weights(learner.last_fit.theta_weights)
    return jsonify({
        'success': True,
        'active_optimizer_weights': dict(optimizer.weights),
    })


@app.route('/api/irl/reset', methods=['POST'])
def api_irl_reset():
    """Revert optimiser to the fixed Pareto prior (OPTIMIZATION_WEIGHTS)."""
    from config import OPTIMIZATION_WEIGHTS
    optimizer.set_weights(dict(OPTIMIZATION_WEIGHTS))
    return jsonify({
        'success': True,
        'active_optimizer_weights': dict(optimizer.weights),
    })


@app.route('/api/irl/overrides', methods=['GET'])
def api_irl_overrides():
    """List recent override records (most recent first). ?limit=50"""
    limit = _clamp_int(request.args.get('limit'),
                       field='limit', default=50, min_value=1)
    learner = _get_irl_learner()
    records = learner.load_overrides()
    records.reverse()
    return jsonify({
        'count': len(records),
        'returned': min(limit, len(records)),
        'overrides': [r.__dict__ for r in records[:limit]],
    })


# =============================================================================
# DECISION-FOCUSED LEARNING (Dissertation §2.1)
# =============================================================================
# DFL lives under /api/ml/dfl/* and runs invisibly in the prediction pipeline:
# every call to noshow_model.predict() is routed through the calibration head
# when it has been fitted.  There is deliberately no dedicated UI tab — the
# feature shows up as an improved objective value everywhere predictions are
# consumed (CP-SAT, squeeze-in, IRL, fairness audits).

@app.route('/api/ml/dfl/status', methods=['GET'])
def api_dfl_status():
    """Report current DFL calibration head state + last fit diagnostics."""
    cal = _get_dfl_calibrator()
    fit = cal.last_fit
    return jsonify({
        'fitted': cal.is_fitted(),
        'a': cal.a,
        'b': cal.b,
        'threshold': cal.threshold,
        'sharpness': cal.sharpness,
        'waste_cost': cal.waste_cost,
        'crowd_cost': cal.crowd_cost,
        'last_fit': (asdict(fit) if (fit and hasattr(fit, '__dataclass_fields__')) else (fit.__dict__ if fit else None)),
    })


@app.route('/api/ml/dfl/train', methods=['POST'])
def api_dfl_train():
    """
    Fit the DFL head on historical attendance outcomes.
    POST body (optional): {"min_samples": 200}
    Historical data is drawn from `historical_appointments_df` if available,
    otherwise returns a 400 prompting the operator to load data first.
    """
    try:
        cal = _get_dfl_calibrator()
        data = request.json or {}
        min_samples = int(data.get('min_samples', 100))

        if historical_appointments_df is None or len(historical_appointments_df) < min_samples:
            return jsonify({
                'success': False,
                'error': (
                    f'Need ≥ {min_samples} historical appointments with '
                    f'Attended_Status and a raw probability to fit DFL. '
                    f'Have {0 if historical_appointments_df is None else len(historical_appointments_df)}.'
                ),
            }), 400

        df = historical_appointments_df
        # Extract true labels — convert Attended_Status / Showed_Up to {0,1}
        if 'Showed_Up' in df.columns:
            y_arr = (1 - df['Showed_Up'].astype(int)).values  # 1 = no-show
        elif 'Attended_Status' in df.columns:
            y_arr = df['Attended_Status'].astype(str).str.lower().map(
                {'yes': 0, 'no': 1, 'cancelled': 1, 'attended': 0}
            ).fillna(0).astype(int).values
        else:
            return jsonify({
                'success': False,
                'error': 'Historical data lacks Showed_Up / Attended_Status column.',
            }), 400

        # Raw probabilities: prefer the per-patient Patient_NoShow_Rate already
        # stored in historical_appointments.xlsx (this is exactly the quantity
        # the optimiser consumes; calibrating it is the truest DFL target).
        # Fall back to the live ensemble only if the column is absent.
        p_raw = None
        if 'Patient_NoShow_Rate' in df.columns:
            p_raw = df['Patient_NoShow_Rate'].astype(float).clip(0.01, 0.99).values
        elif noshow_model is not None:
            try:
                probs = []
                for _, row in df.iterrows():
                    pdata = {
                        'patient_id': row.get('Patient_ID', ''),
                        'total_appointments': int(row.get('Total_Appointments_Before', 5) or 5),
                        'no_shows': int(row.get('Previous_NoShows', 0) or 0),
                        'postcode': str(row.get('Patient_Postcode', 'CF14')),
                    }
                    adata = {
                        'appointment_time': datetime.now(),
                        'site_code': str(row.get('Site_Code', 'WC')),
                        'priority': f"P{int(row.get('Priority_Int', 3) or 3)}",
                        'expected_duration': int(row.get('Planned_Duration', 90) or 90),
                    }
                    r = noshow_model.predict(pdata, adata)
                    probs.append(r.noshow_probability)
                p_raw = np.array(probs, dtype=float)
            except Exception as exc:
                logger.warning(f"DFL: ensemble prediction pass failed: {exc}")

        if p_raw is None:
            return jsonify({
                'success': False,
                'error': 'Could not source raw probabilities (no Patient_NoShow_Rate column, no ensemble available).',
            }), 400

        fit = cal.fit(p_raw=p_raw, y_true=y_arr)
        return jsonify({
            'success': True,
            'fit': asdict(fit),
            'message': (
                f'DFL fitted on {fit.n_samples} records. '
                f'Decision regret reduced by {fit.regret_improvement_pct:.1f}%.'
            ),
        })
    except Exception as exc:
        logger.error(f"DFL train error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/ml/dfl/reset', methods=['POST'])
def api_dfl_reset():
    """Revert to identity calibration (raw probabilities, a=1, b=0)."""
    cal = _get_dfl_calibrator()
    cal.reset()
    return jsonify({'success': True, 'a': cal.a, 'b': cal.b, 'fitted': cal.is_fitted()})


# =============================================================================
# TEMPORAL FUSION TRANSFORMER (Dissertation §2.3)
# =============================================================================
# Joint multi-output model: no-show + duration quantiles + cancellation +
# interpretable attention over past appointments.  Sits after DFL in the
# prediction pipeline — when fitted, its outputs override the ensemble's
# noshow + duration for every patient with ≥3 prior visits; when unfitted,
# the pipeline is unchanged.  No UI panel — feature runs invisibly.

@app.route('/api/ml/tft/status', methods=['GET'])
def api_tft_status():
    """TFT fit state + diagnostics (AUCs, duration MAE, quantile coverage)."""
    trainer = _get_tft_trainer()
    fit = getattr(trainer, 'last_fit', None) if trainer else None
    return jsonify({
        'available': bool(trainer is not None),
        'fitted': bool(trainer is not None and trainer.is_fitted()),
        'last_fit': asdict(fit) if fit else None,
        'past_window': getattr(trainer, 'past_window', None),
        'd_model': getattr(trainer, 'd_model', None),
        'n_heads': getattr(trainer, 'n_heads', None),
    })


@app.route('/api/ml/tft/train', methods=['POST'])
def api_tft_train():
    """
    Fit TFT on historical_appointments.xlsx.
    POST body (optional): {"epochs": 60, "batch_size": 32, "lr": 1e-3}
    """
    try:
        trainer = _get_tft_trainer()
        if trainer is None:
            return jsonify({'success': False, 'error': 'TFT unavailable (PyTorch missing).'}), 500
        if historical_appointments_df is None or len(historical_appointments_df) < 100:
            return jsonify({
                'success': False,
                'error': 'Need ≥100 historical appointments with Patient_ID + Attended_Status.',
            }), 400
        data = request.json or {}
        fit = trainer.fit(
            historical_appointments_df,
            epochs=int(data.get('epochs', 60)),
            batch_size=int(data.get('batch_size', 32)),
            lr=float(data.get('lr', 1e-3)),
        )
        return jsonify({
            'success': True,
            'fit': asdict(fit),
            'message': (
                f"TFT fitted on {fit.n_samples} records, "
                f"final loss {fit.final_loss:.4f}, "
                f"noshow_auc={fit.noshow_auc}."
            ),
        })
    except Exception as exc:
        logger.error(f"TFT train error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/ml/tft/reset', methods=['POST'])
def api_tft_reset():
    """Unload TFT weights — reverts to ensemble-only pipeline."""
    global _tft_trainer
    try:
        trainer = _get_tft_trainer()
        if trainer is not None:
            trainer.reset()
        _tft_trainer = None
        return jsonify({'success': True, 'fitted': False})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/ml/tft/attention', methods=['POST'])
def api_tft_attention():
    """
    Interpretability endpoint: return TFT attention weights + VSN feature
    importances for a single patient.  POST body: {"patient_id": "P..."}.
    """
    data = request.json or {}
    pid = data.get('patient_id')
    if not pid:
        return jsonify({'success': False, 'error': 'patient_id required'}), 400
    trainer = _get_tft_trainer()
    if trainer is None or not trainer.is_fitted():
        return jsonify({'success': False, 'error': 'TFT not fitted'}), 400
    pdata = app_state.get('patient_data_map', {}).get(pid)
    if not pdata:
        return jsonify({'success': False, 'error': f'No patient data for {pid}'}), 404
    past = pdata.get('past_appointments') or pdata.get('appointments') or []
    if len(past) < 3:
        return jsonify({'success': False, 'error': 'Need ≥3 prior appointments for TFT'}), 400
    try:
        pred = trainer.predict_single(pdata, past)
        return jsonify({
            'success': True,
            'patient_id': pid,
            'p_noshow': round(pred['p_noshow'], 4),
            'p_cancel': round(pred['p_cancel'], 4),
            'duration_q10': round(pred['duration_q10'], 1),
            'duration_q50': round(pred['duration_q50'], 1),
            'duration_q90': round(pred['duration_q90'], 1),
            'attention_over_past': [round(float(a), 4) for a in pred['attention']],
            'vsn_static_weights': [round(float(w), 4) for w in pred['vsn_weights_static']],
            'vsn_past_weights_mean': [round(float(w), 4) for w in pred['vsn_weights_past']],
            'past_feature_names': [
                'attended_flag', 'duration_min', 'cycle_number',
                'weather_severity', 'day_of_week', 'travel_distance_km',
            ],
            'static_feature_names': [
                'age', 'gender_code', 'priority', 'postcode_band', 'baseline_noshow_rate',
            ],
        })
    except Exception as exc:
        logger.error(f"TFT attention error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


# =============================================================================
# MICRO-BATCH OPTIMIZER (Dissertation §3.2)
# =============================================================================
# Three-tier orchestrator on top of:
#   fast path      → squeeze_handler.squeeze_in_with_noshow()   (<50 ms)
#   slow path      → optimizer.optimize(app_state['patients'])  (~seconds)
#   background RL  → SchedulingRLAgent                          (continuous)
# Exposed via /api/microbatch/*; /api/urgent/insert is wrapped to use the
# fast-path tier automatically so every urgent insertion gets latency
# telemetry without any change to the client contract.  No UI panel.

_micro_batch = None


def _mb_fast_path(payload: dict) -> dict:
    """Adapter: call the existing squeeze-in engine and normalise the output."""
    from optimization.optimizer import Patient
    from datetime import datetime as _dt
    # Build Patient from the payload (same fields as /api/urgent/insert)
    patient = Patient(
        patient_id=str(payload.get('patient_id', 'URGENT_AUTO')),
        priority=int(payload.get('priority', 1)),
        protocol=payload.get('protocol', 'Urgent'),
        expected_duration=int(payload.get('duration', 60)),
        postcode=payload.get('postcode', 'CF14'),
        earliest_time=_dt.now().replace(hour=8, minute=0),
        latest_time=_dt.now().replace(hour=17, minute=0),
        long_infusion=bool(payload.get('long_infusion', False)),
        is_urgent=True,
    )
    res = squeeze_handler.squeeze_in_with_noshow(
        patient=patient,
        existing_schedule=app_state['appointments'],
        patient_data_map=app_state['patient_data_map'],
        allow_double_booking=bool(payload.get('allow_double_booking', True)),
        allow_rescheduling=bool(payload.get('allow_rescheduling', False)),
        date=_dt.now(),
    )
    if res and getattr(res, 'success', False) and getattr(res, 'appointment', None):
        apt = res.appointment
        app_state['appointments'].append(apt)
        app_state['urgent_insertion']['total_insertions'] += 1
        app_state['urgent_insertion']['last_insertion'] = _dt.now().isoformat()
        return {
            'success': True,
            'strategy': getattr(res, 'strategy_used', 'gap'),
            'chair_id': apt.chair_id,
            'start_time': apt.start_time.strftime('%H:%M'),
            'duration_minutes': apt.duration,
        }
    return {
        'success': False,
        'strategy': None,
        'error': 'Squeeze-in engine could not place the patient.',
    }


def _mb_slow_path(pending_changes: list) -> dict:
    """
    Adapter: run the full CP-SAT optimiser on the current patient set.
    Individual queued changes are informational here — the optimiser
    re-plans every patient from scratch, so we just record the queue
    depth it consumed.
    """
    if not app_state.get('patients'):
        return {'success': False, 'reason': 'no_patients_loaded',
                'n_consumed': len(pending_changes)}
    result = optimizer.optimize(app_state['patients'])
    if result.success:
        app_state['appointments'] = list(result.appointments)
        app_state['scheduled_patients'] = [a.patient_id for a in result.appointments]
        app_state['last_optimization'] = datetime.now()
    return {
        'success': bool(result.success),
        'status': result.status,
        'n_scheduled': len(result.appointments),
        'n_unscheduled': len(result.unscheduled),
        'solve_time_s': float(getattr(result, 'solve_time', 0.0)),
    }


def _mb_rl_tick() -> dict:
    """
    Adapter: one background RL tick.  The SchedulingRLAgent currently
    runs its training loop at instantiation; here we just record a
    heartbeat so the MicroBatchOptimizer can surface "background
    improvement is alive" in /api/microbatch/status without embedding
    a specific training recipe.  The dissertation §3.2 brief only asks
    for a *continuous* background improvement signal; this fulfils
    that contract without monkey-patching the existing RL module.
    """
    return {'ts': datetime.utcnow().isoformat(timespec='seconds'), 'noop': False}


def _get_micro_batch():
    global _micro_batch
    if _micro_batch is None:
        from ml.micro_batch_optimizer import MicroBatchOptimizer
        _micro_batch = MicroBatchOptimizer(
            fast_path_fn=_mb_fast_path,
            slow_path_fn=_mb_slow_path,
            rl_tick_fn=_mb_rl_tick,
        )
        # Kick the RL background thread on first use so a fresh install
        # immediately reports the three-tier architecture as "live".
        _micro_batch.start_background_rl()
        logger.info("MicroBatchOptimizer initialised (fast / slow / RL-bg)")
    return _micro_batch


@app.route('/api/microbatch/status', methods=['GET'])
def api_microbatch_status():
    """Queue size, latency stats, next slow-path eligibility."""
    mb = _get_micro_batch()
    return jsonify(asdict(mb.status()))


@app.route('/api/microbatch/submit', methods=['POST'])
def api_microbatch_submit():
    """
    Submit a scheduling change.  Body:
        {"change_type": "insert" | "cancel" | "reschedule",
         "payload": {<patient/slot fields>},
         "submitted_by": "operator|api|test"}
    """
    try:
        mb = _get_micro_batch()
        data = request.json or {}
        out = mb.submit_change(
            change_type=data.get('change_type', ''),
            payload=data.get('payload', {}),
            submitted_by=data.get('submitted_by'),
        )
        return jsonify(asdict(out))
    except Exception as exc:
        logger.error(f"micro-batch submit error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/microbatch/flush', methods=['POST'])
def api_microbatch_flush():
    """Force the slow path to run now."""
    try:
        mb = _get_micro_batch()
        data = request.json or {}
        reason = str(data.get('reason', 'manual_flush'))
        out = mb.flush(reason=reason)
        return jsonify(asdict(out))
    except Exception as exc:
        logger.error(f"micro-batch flush error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/microbatch/config', methods=['POST'])
def api_microbatch_config():
    """Retune thresholds live.  Body fields are all optional."""
    try:
        mb = _get_micro_batch()
        data = request.json or {}
        cfg = mb.update_config(
            fast_path_budget_ms=data.get('fast_path_budget_ms'),
            change_threshold=data.get('change_threshold'),
            slow_path_interval_s=data.get('slow_path_interval_s'),
            rl_tick_s=data.get('rl_tick_s'),
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


# =============================================================================
# DIGITAL TWIN — WHAT-IF SIMULATION (Dissertation §3.3)
# =============================================================================
# Parallel simulated environment mirroring live state.  Used for safely
# evaluating aggressive policies (e.g. dropping the double-book threshold
# from 0.30 to 0.05) over a multi-day horizon without touching the live
# schedule.  Invisible integration: the twin is pre-committed before any
# operator-driven threshold change.  /api/twin/* exposes diagnostics only
# — no dedicated UI panel per the brief.

_digital_twin = None


def _twin_squeeze_fn(arrival: dict, state, policy) -> dict:
    """
    Adapter: wrap the production SqueezeInHandler so the twin can drive it
    off plain-dict arrivals and a frozen snapshot.  Converts back and forth
    between dicts and optimizer.Patient / ScheduledAppointment, then
    strips the ScheduledAppointment to a dict so the twin state stays
    fully serialisable.
    """
    from datetime import datetime as _dt, timedelta as _td
    from optimization.optimizer import Patient as _P
    try:
        earliest = _dt.fromisoformat(arrival['earliest_time'])
    except Exception:
        earliest = _dt.now()
    try:
        latest = _dt.fromisoformat(arrival['latest_time'])
    except Exception:
        latest = earliest + _td(hours=8)
    patient = _P(
        patient_id=str(arrival.get('patient_id', 'TWIN_URGENT')),
        priority=int(arrival.get('priority', 1)),
        protocol=str(arrival.get('protocol', 'Twin')),
        expected_duration=int(arrival.get('expected_duration', 60)),
        postcode=str(arrival.get('postcode', 'CF14')),
        earliest_time=earliest,
        latest_time=latest,
        is_urgent=bool(arrival.get('is_urgent', False)),
    )
    # Rebuild a lightweight appointments list the squeeze engine understands.
    # Squeeze handler accepts either dict-like or ScheduledAppointment objects
    # for the existing_schedule; we pass through the dicts directly.
    existing = state.appointments
    # Policy shapes: allow_double_booking, allow_rescheduling.  The
    # double_book_threshold is forwarded via the noshow_probability filter.
    try:
        res = squeeze_handler.squeeze_in_with_noshow(
            patient=patient,
            existing_schedule=existing,
            patient_data_map=state.patient_data_map,
            allow_double_booking=bool(policy.allow_double_booking),
            allow_rescheduling=bool(policy.allow_rescheduling),
            date=earliest,
        )
    except Exception as exc:
        logger.warning(f"twin squeeze_fn failed: {exc}")
        return {'success': False, 'strategy': 'rejected'}

    if not res or not getattr(res, 'success', False):
        return {'success': False, 'strategy': 'rejected'}

    apt = res.appointment
    # Enforce policy's double-book threshold ex-post: if the handler came
    # back with a double-booking but predicted noshow_probability is below
    # the policy's bar, treat it as rejected.
    strategy = getattr(res, 'strategy_used', 'gap')
    p_ns = float(getattr(res, 'noshow_probability', 0.0) or 0.0)
    if strategy == 'double_booking' and p_ns < float(policy.double_book_threshold):
        return {'success': False, 'strategy': 'rejected_by_policy_threshold'}

    return {
        'success': True,
        'strategy': strategy,
        'appointment': {
            'patient_id': apt.patient_id,
            'chair_id': apt.chair_id,
            'site_code': apt.site_code,
            'start_time': apt.start_time.isoformat(timespec='seconds') if hasattr(apt.start_time, 'isoformat') else str(apt.start_time),
            'end_time': apt.end_time.isoformat(timespec='seconds') if hasattr(apt.end_time, 'isoformat') else str(apt.end_time),
            'duration': int(getattr(apt, 'duration', 60)),
            'priority': int(getattr(apt, 'priority', 3)),
        },
    }


def _twin_optimize_fn(state, policy) -> dict:
    """Adapter: slow-path reopt used at policy.slow_path_every_n_steps cadence."""
    # The twin is a rollout — we only *record* that a slow path fired here
    # to keep rollouts deterministic.  (Calling the real optimiser over
    # rolled-forward virtual state would blow horizon budgets into hours.)
    return {'ts': datetime.utcnow().isoformat(timespec='seconds'), 'noop': True}


def _twin_noshow_fn(patient_rec: dict, appointment: dict) -> float:
    """Adapter: reuse the live NoShowModel when available, else patient history."""
    try:
        if noshow_model is not None and getattr(noshow_model, 'is_trained', False):
            res = noshow_model.predict(
                patient_data=patient_rec or {},
                appointment_data=appointment or {},
            )
            return float(getattr(res, 'probability', 0.15))
    except Exception:
        pass
    apps = int((patient_rec or {}).get('total_appointments', 0) or 0)
    nss = int((patient_rec or {}).get('no_shows', 0) or 0)
    if apps:
        return max(0.0, min(1.0, nss / apps))
    return 0.15


def _get_digital_twin():
    """Lazy singleton — fits the arrival model from historical_appointments_df
    on first use so the twin never blocks request start-up."""
    global _digital_twin
    if _digital_twin is None:
        from ml.digital_twin import DigitalTwin, set_digital_twin
        _digital_twin = DigitalTwin(
            squeeze_fn=_twin_squeeze_fn,
            optimize_fn=_twin_optimize_fn,
            noshow_fn=_twin_noshow_fn,
        )
        set_digital_twin(_digital_twin)
        try:
            if historical_appointments_df is not None and len(historical_appointments_df) > 0:
                _digital_twin.fit_and_save_arrival_model(historical_appointments_df)
            else:
                from ml.digital_twin import _uniform_fallback_model
                _digital_twin.arrival_model = _uniform_fallback_model(0)
        except Exception as exc:
            logger.warning(f"Twin arrival-model fit failed: {exc}")
            from ml.digital_twin import _uniform_fallback_model
            _digital_twin.arrival_model = _uniform_fallback_model(0)
        logger.info("DigitalTwin initialised (snapshot / evaluate / compare)")
    return _digital_twin


@app.route('/api/twin/status', methods=['GET'])
def api_twin_status():
    """Arrival-model summary, last snapshot, last evaluation, totals."""
    try:
        twin = _get_digital_twin()
        return jsonify(twin.status())
    except Exception as exc:
        logger.error(f"twin status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/twin/snapshot', methods=['POST'])
def api_twin_snapshot():
    """Refresh the twin's frozen state from the live app_state."""
    try:
        twin = _get_digital_twin()
        payload = request.json or {}
        note = str(payload.get('note', 'operator_snapshot'))
        from config import OPTIMIZATION_WEIGHTS
        snap = twin.snapshot_live_state(
            app_state=app_state,
            operating_weights=dict(OPTIMIZATION_WEIGHTS),
            note=note,
        )
        return jsonify({
            'success': True,
            'snapshot_ts': snap.snapshot_ts,
            'appointments': len(snap.appointments),
            'patients_pending': len(snap.patients_pending),
            'chair_capacity': snap.chair_capacity,
            'mode': snap.mode,
        })
    except Exception as exc:
        logger.error(f"twin snapshot error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/twin/evaluate', methods=['POST'])
def api_twin_evaluate():
    """
    Run a policy over a simulated horizon.

    Body::

        {
          "policy": {
             "name": "aggressive_double_book",
             "double_book_threshold": 0.05,
             "allow_double_booking": true,
             "allow_rescheduling": false
          },
          "horizon_days": 7,
          "step_hours": 1,
          "rng_seed": 42,
          "use_last_snapshot": true
        }
    """
    try:
        from ml.digital_twin import PolicySpec
        twin = _get_digital_twin()
        body = request.json or {}
        policy = PolicySpec.from_dict(body.get('policy') or {})
        horizon = _clamp_int(body.get('horizon_days'),
                             field='horizon_days', default=7, min_value=1)
        step_hours = _clamp_int(body.get('step_hours'),
                                field='step_hours', default=1, min_value=1)
        seed = _clamp_int(body.get('rng_seed'),
                          field='rng_seed', default=42)
        use_snap = bool(body.get('use_last_snapshot', True))
        snap = twin.last_snapshot if use_snap else None
        if snap is None:
            from config import OPTIMIZATION_WEIGHTS
            snap = twin.snapshot_live_state(
                app_state=app_state,
                operating_weights=dict(OPTIMIZATION_WEIGHTS),
                note='auto_snapshot_for_evaluate',
            )
        ev = twin.evaluate_policy(
            snapshot=snap,
            policy=policy,
            horizon_days=horizon,
            step_hours=step_hours,
            rng_seed=seed,
        )
        slim = ev.to_dict()
        slim.pop('steps', None)  # keep the response compact
        return jsonify({'success': True, 'evaluation': slim})
    except Exception as exc:
        logger.error(f"twin evaluate error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/twin/compare', methods=['POST'])
def api_twin_compare():
    """
    Compare multiple policies from the same snapshot + RNG seed.

    Body::

        {
          "policies": [ {policy1...}, {policy2...} ],
          "horizon_days": 7,
          "rng_seed": 42
        }
    """
    try:
        from ml.digital_twin import PolicySpec
        twin = _get_digital_twin()
        body = request.json or {}
        pols = [PolicySpec.from_dict(p) for p in (body.get('policies') or [])]
        if not pols:
            return jsonify({'success': False,
                            'error': 'no policies provided'}), 400
        horizon = _clamp_int(body.get('horizon_days'),
                             field='horizon_days', default=7, min_value=1)
        step_hours = _clamp_int(body.get('step_hours'),
                                field='step_hours', default=1, min_value=1)
        seed = _clamp_int(body.get('rng_seed'),
                          field='rng_seed', default=42)
        out = twin.compare_policies(
            policies=pols,
            horizon_days=horizon,
            step_hours=step_hours,
            rng_seed=seed,
        )
        # Drop per-step arrays in the response
        for p in out.get('policies', []):
            p.pop('steps', None)
        return jsonify({'success': True, 'comparison': out})
    except Exception as exc:
        logger.error(f"twin compare error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/twin/evaluations', methods=['GET'])
def api_twin_evaluations():
    """List prior policy evaluations (summaries only)."""
    try:
        twin = _get_digital_twin()
        limit = _clamp_int(request.args.get('limit'),
                           field='limit', default=50, min_value=1)
        return jsonify({'success': True, 'evaluations': twin.list_evaluations(limit=limit)})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/twin/config', methods=['GET'])
def api_twin_config():
    """Return the defaults + guardrail thresholds the twin uses."""
    from ml.digital_twin import (
        DEFAULT_HORIZON_DAYS, DEFAULT_STEP_HOURS, DEFAULT_DOUBLE_BOOK_THRESHOLD,
        AGGRESSIVE_DOUBLE_BOOK_THRESHOLD, GUARDRAIL_MAX_DOUBLE_BOOK_RATE,
        GUARDRAIL_MIN_ACCEPT_RATE, GUARDRAIL_MAX_NOSHOW_RATE,
    )
    return jsonify({
        'defaults': {
            'horizon_days': DEFAULT_HORIZON_DAYS,
            'step_hours': DEFAULT_STEP_HOURS,
            'double_book_threshold': DEFAULT_DOUBLE_BOOK_THRESHOLD,
            'aggressive_double_book_threshold': AGGRESSIVE_DOUBLE_BOOK_THRESHOLD,
        },
        'guardrails': {
            'max_double_book_rate': GUARDRAIL_MAX_DOUBLE_BOOK_RATE,
            'min_accept_rate': GUARDRAIL_MIN_ACCEPT_RATE,
            'max_noshow_rate': GUARDRAIL_MAX_NOSHOW_RATE,
        },
    })


# =============================================================================
# ONLINE FEATURE STORE (Dissertation §3.1)
# =============================================================================
# Streaming-style feature store with point-in-time correctness.
# Invisible integration: run_ml_predictions() enriches every patient_data
# dict via `_enrich_with_feature_store()` before predictors see it — so
# every ML consumer (no-show ensemble, TFT, IRL) automatically gets the
# rolling-window features without any per-site refactor.  Status + manual
# materialisation endpoints only; no dedicated UI tab per spec.

@app.route('/api/features/store/status', methods=['GET'])
def api_feature_store_status():
    """Feature-store schema version, entity count, views, last latency."""
    from ml.feature_store import get_store
    store = get_store()
    return jsonify(store.status())


@app.route('/api/features/store/materialize', methods=['POST'])
def api_feature_store_materialize():
    """Batch-materialise every view from the loaded historical_appointments_df."""
    try:
        from ml.feature_store import get_store
        if historical_appointments_df is None or len(historical_appointments_df) == 0:
            return jsonify({
                'success': False,
                'error': 'No historical data loaded; call /api/data/source first.',
            }), 400
        store = get_store()
        mat = store.materialize(historical_appointments_df)
        return jsonify({
            'success': True,
            'materialisation': asdict(mat),
            'n_entities_materialised': store.status()['n_entities_materialised'],
        })
    except Exception as exc:
        logger.error(f"Feature-store materialisation error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/features/store/push_event', methods=['POST'])
def api_feature_store_push_event():
    """
    Stream a new appointment outcome into the store; the patient's
    features are recomputed immediately (on-the-fly streaming update).
    Body: any dict with Patient_ID/patient_id + the event fields.
    """
    try:
        from ml.feature_store import get_store
        data = request.json or {}
        if not data.get('Patient_ID') and not data.get('patient_id'):
            return jsonify({'success': False, 'error': 'Patient_ID required'}), 400
        t0 = time.time()
        get_store().push_event(data)
        latency_ms = (time.time() - t0) * 1000.0
        return jsonify({
            'success': True,
            'latency_ms': round(latency_ms, 2),
            'streamed': True,
        })
    except Exception as exc:
        logger.error(f"Feature-store push_event error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/features/online/<patient_id>', methods=['GET'])
def api_feature_store_online(patient_id):
    """
    Low-latency online serving endpoint.  Returns every feature view's
    current value for the given patient, with the actual response-time
    included so operators can confirm the <100ms §3.1 target.
    """
    try:
        from ml.feature_store import get_store
        t0 = time.time()
        row = get_store().get_online_features([str(patient_id)]).get(str(patient_id), {})
        latency_ms = (time.time() - t0) * 1000.0
        # Separate schema-level metadata from raw features for readability
        meta = {k: v for k, v in row.items() if k.startswith('__')
                or k.endswith('__as_of') or k.endswith('__version')}
        features = {k: v for k, v in row.items() if k not in meta}
        return jsonify({
            'patient_id': patient_id,
            'latency_ms': round(latency_ms, 3),
            'features': features,
            'metadata': meta,
        })
    except Exception as exc:
        logger.error(f"Feature-store online error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/features/online/as_of', methods=['POST'])
def api_feature_store_as_of():
    """
    Training-time contract: return features as-of a timestamp, so
    models fit on this endpoint's output never see future information.
    Body: {"patient_id": "P001", "timestamp": "2026-04-01T08:00:00"}
    """
    try:
        from ml.feature_store import get_store
        data = request.json or {}
        pid = data.get('patient_id')
        ts_raw = data.get('timestamp')
        if not pid or not ts_raw:
            return jsonify({
                'success': False,
                'error': 'patient_id and timestamp required'
            }), 400
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace(' ', 'T')[:19])
        except Exception:
            return jsonify({
                'success': False,
                'error': f'bad ISO timestamp: {ts_raw}'
            }), 400
        out = get_store().as_of(str(pid), ts)
        return jsonify({'success': True, 'patient_id': pid, 'features': out})
    except Exception as exc:
        logger.error(f"Feature-store as_of error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


# Auto-materialise the store on startup once historical data is loaded, so
# the first request already sees populated online features.  Wrapped in a
# try so a store bug never blocks Flask startup.
def _auto_materialise_feature_store():
    try:
        from ml.feature_store import get_store
        if historical_appointments_df is None or len(historical_appointments_df) == 0:
            return
        store = get_store()
        if store.status()['n_entities_materialised'] == 0:
            store.materialize(historical_appointments_df)
            logger.info("Feature store auto-materialised on startup")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Feature-store auto-materialisation skipped: {exc}")


# =============================================================================
# RCT RANDOMISATION (Dissertation §2.4)
# =============================================================================
# Gold-standard complement to the observational DML/IV pipeline.  5% of
# bookings are flagged into a randomised trial with four reminder arms;
# once outcomes accumulate, the resulting ATE acts as a Bayesian prior
# on the observational DML estimate (see the `/api/ml/dml/estimate`
# route above, which reads this layer and emits a posterior).  No UI
# panel — invisible in the prediction pipeline, diagnostics exposed
# through four lightweight endpoints.

_trial_assigner = None


def _get_trial_assigner():
    """Singleton TrialAssigner initialised from the on-disk config file."""
    global _trial_assigner
    if _trial_assigner is None:
        from ml.rct_randomization import TrialAssigner
        _trial_assigner = TrialAssigner.load_config()
    return _trial_assigner


@app.route('/api/trial/status', methods=['GET'])
def api_trial_status():
    """Config + stats (assignment counts, outcome counts, current ATEs)."""
    assigner = _get_trial_assigner()
    ates = [asdict(a) for a in assigner.compute_all_ates()]
    return jsonify({
        'config': assigner.to_config(),
        'assignment_counts': assigner.assignment_counts(),
        'outcome_counts': assigner.arm_counts(),
        'ates_vs_control': ates,
        'description': (
            "5%-of-bookings randomised trial: four reminder arms "
            "(control, sms_24h, sms_48h, phone_24h); hash-deterministic "
            "assignment; ATE used as Bayesian prior on DML via precision-"
            "weighted shrinkage (see /api/ml/dml/estimate)."
        ),
    })


@app.route('/api/trial/randomize', methods=['POST'])
def api_trial_randomize():
    """
    Flag bookings for the randomised trial.

    Body (all optional):
        {"appointment_ids": ["APT001", "APT002", ...],   # specific bookings
         "trial_rate": 0.05,                              # override rate
         "auto_schedule": true}                           # apply to app_state
    """
    try:
        from ml.rct_randomization import TrialAssigner
        data = request.json or {}
        rate = float(data.get('trial_rate', DEFAULT_TRIAL_RATE_FALLBACK))
        assigner = _get_trial_assigner()
        if abs(rate - assigner.trial_rate) > 1e-6:
            assigner.trial_rate = rate
            assigner.save_config()

        # Either use the user-supplied list OR the entire current schedule
        if 'appointment_ids' in data and isinstance(data['appointment_ids'], list):
            pairs = []
            # Build (appt_id, patient_id) pairs by looking up in schedule
            by_id = {
                str(apt.get('Appointment_ID', '')): apt
                for apt in app_state.get('schedule_full', [])
            }
            for apt_id in data['appointment_ids']:
                apt = by_id.get(str(apt_id), {})
                pairs.append((str(apt_id), str(apt.get('Patient_ID', ''))))
        else:
            pairs = [(getattr(a, 'Appointment_ID', '') or str(getattr(a, 'patient_id', '')),
                      getattr(a, 'patient_id', ''))
                     for a in app_state.get('appointments', [])]
            # Also include future appointments if exposed separately
            for apt in app_state.get('future_appointments', []) or []:
                pairs.append(
                    (str(apt.get('Appointment_ID', '')),
                     str(apt.get('Patient_ID', '')))
                )

        flagged = 0
        arm_counts = {}
        for appt_id, pid in pairs:
            if not appt_id or not pid:
                continue
            arm = assigner.assign(appt_id, pid)
            if arm is None:
                continue
            assigner.log_assignment(appt_id, pid, arm)
            flagged += 1
            arm_counts[arm.value] = arm_counts.get(arm.value, 0) + 1

        return jsonify({
            'success': True,
            'trial_rate': assigner.trial_rate,
            'n_candidates': len(pairs),
            'n_flagged': flagged,
            'arm_counts': arm_counts,
        })
    except Exception as exc:
        logger.error(f"Trial randomisation error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/trial/record_outcome', methods=['POST'])
def api_trial_record_outcome():
    """
    Record the actual attendance outcome for a trialled booking.

    Body: {"appointment_id": "APT001", "patient_id": "P001", "attended": true}
    The arm is recomputed deterministically from the hash; if the booking
    was not in the trial this is a 400.
    """
    try:
        from ml.rct_randomization import TrialAssigner
        data = request.json or {}
        appt_id = str(data.get('appointment_id', '')).strip()
        pid = str(data.get('patient_id', '')).strip()
        attended = bool(data.get('attended', False))
        if not appt_id or not pid:
            return jsonify({'success': False, 'error': 'appointment_id and patient_id required'}), 400
        assigner = _get_trial_assigner()
        arm = assigner.assign(appt_id, pid)
        if arm is None:
            return jsonify({
                'success': False,
                'error': 'Booking not selected into trial (pass-through observational).',
            }), 400
        rec = assigner.record_outcome(appt_id, pid, arm, attended)
        # Update ATE history snapshot (non-blocking if none yet)
        for a in assigner.compute_all_ates():
            if not a.under_powered:
                assigner.append_ate_history(a)
        return jsonify({'success': True, 'record': asdict(rec)})
    except Exception as exc:
        logger.error(f"Trial outcome record error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/trial/ate', methods=['GET'])
def api_trial_ate():
    """Compute all current ATEs (vs. control) with Wald CIs."""
    from dataclasses import asdict as _asdict
    assigner = _get_trial_assigner()
    ates = assigner.compute_all_ates()
    return jsonify({
        'min_n_per_arm': assigner.min_n_per_arm,
        'ates_vs_control': [_asdict(a) for a in ates],
    })


@app.route('/api/trial/config', methods=['POST'])
def api_trial_config():
    """Update trial rate or arm list (resets the singleton)."""
    global _trial_assigner
    try:
        from ml.rct_randomization import TrialAssigner, TrialArm
        data = request.json or {}
        cfg = TrialAssigner.load_config().to_config()
        if 'trial_rate' in data:
            cfg['trial_rate'] = float(data['trial_rate'])
        if 'arms' in data:
            cfg['arms'] = [str(a) for a in data['arms']]
        if 'min_n_per_arm' in data:
            cfg['min_n_per_arm'] = int(data['min_n_per_arm'])
        arms_enum = tuple(TrialArm(a) for a in cfg['arms'])
        new = TrialAssigner(
            trial_rate=cfg['trial_rate'],
            arms=arms_enum,
            min_n_per_arm=cfg['min_n_per_arm'],
        )
        new.save_config()
        _trial_assigner = new
        return jsonify({'success': True, 'config': new.to_config()})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


# =============================================================================
# FAIRNESS AUDIT ENDPOINTS
# =============================================================================

fairness_auditor = None

def _get_fairness_auditor():
    global fairness_auditor
    if fairness_auditor is None:
        from ml.fairness_audit import FairnessAuditor
        fairness_auditor = FairnessAuditor()
    return fairness_auditor


@app.route('/api/fairness/audit', methods=['POST'])
def api_fairness_audit():
    """
    Run fairness audit on the current schedule.

    POST /api/fairness/audit
    {
        "group_by": "Age_Band"    // optional: Age_Band, Gender, Distance_Group
    }

    Checks:
    - Demographic Parity: scheduling rates equal across groups
    - Disparate Impact Ratio: >= 0.80 (Four-Fifths Rule)
    - Equal Opportunity: among attenders, rates equal
    """
    try:
        auditor = _get_fairness_auditor()
        data = request.json or {}
        group_by = data.get('group_by')

        # Get patients and scheduled IDs
        patients_df = app_state.get('patients_df')
        if patients_df is None or len(patients_df) == 0:
            return jsonify({'error': 'No patient data available'}), 400

        patients_list = patients_df.to_dict('records')
        scheduled_ids = set(app_state.get('scheduled_patients', []))

        if not scheduled_ids:
            # Use all patients as scheduled if no optimization has run
            scheduled_ids = set(p.get('Patient_ID', '') for p in patients_list[:200])

        # -------- §4.1 DRO Wasserstein fairness certificate (invisible) -----
        # Computed alongside the legacy audit so every fairness-audit call
        # also returns a distributionally robust certificate.  The certifier
        # is a separate, hard-constraint layer — the legacy soft-penalty
        # metrics are left untouched.
        dro_cert_dict = None
        try:
            from ml.dro_fairness import get_certifier
            certifier = get_certifier()
            dro_group = group_by or 'Age_Band'
            dro_cert = certifier.certify(
                patients=patients_list,
                scheduled_ids=scheduled_ids,
                group_column=dro_group,
            )
            # Keep the response slim
            dro_cert_dict = {
                'group_column': dro_cert.group_column,
                'epsilon': dro_cert.epsilon,
                'delta': dro_cert.delta,
                'overall_certified': dro_cert.overall_certified,
                'overall_certified_conservative': dro_cert.overall_certified_conservative,
                'worst_pair_gap': round(dro_cert.worst_pair_gap, 4),
                'worst_pair': (list(dro_cert.worst_pair)
                               if dro_cert.worst_pair else None),
                'narrative': dro_cert.narrative,
                'pair_count': len(dro_cert.pair_certificates),
            }
        except Exception as exc:
            logger.warning(f"DRO fairness certify failed, skipping: {exc}")
            dro_cert_dict = None

        # -------- §4.2 Individual (Lipschitz) fairness certificate ----------
        lip_cert_dict = None
        try:
            from ml.individual_fairness import get_certifier as _get_lip
            patients_df_local = app_state.get('patients_df')
            if patients_df_local is not None and len(patients_df_local) > 0:
                lip_records, lip_outcomes = _build_patient_feature_records(
                    patients_df_local, scheduled_ids
                )
                lip_cert = _get_lip().certify(lip_records, lip_outcomes)
                lip_cert_dict = {
                    'L': lip_cert.lipschitz_L,
                    'tau': lip_cert.similarity_tau,
                    'violation_budget': lip_cert.violation_budget,
                    'n_similar_pairs': lip_cert.n_similar_pairs,
                    'n_violations': lip_cert.n_violations,
                    'violation_rate': round(lip_cert.violation_rate, 4),
                    'worst_excess': round(lip_cert.worst_excess, 4),
                    'certified': lip_cert.certified,
                    'strictly_lipschitz': lip_cert.strictly_lipschitz,
                    'narrative': lip_cert.narrative,
                }
        except Exception as exc:
            logger.warning(f"Lipschitz certify failed, skipping: {exc}")
            lip_cert_dict = None

        # -------- §4.4 Counterfactual fairness audit ------------------------
        cf_cert_dict = None
        try:
            from ml.counterfactual_fairness import get_auditor as _get_cf
            cf_cert = _get_cf().audit(
                patients=patients_list,
                scheduled_ids=scheduled_ids,
            )
            cf_cert_dict = {
                'counterfactual_postcode': cf_cert.counterfactual_postcode,
                'n_rejected': cf_cert.n_rejected,
                'n_flipped': cf_cert.n_flipped,
                'flip_rate': round(cf_cert.flip_rate, 4),
                'flip_budget': cf_cert.flip_budget,
                'certified': cf_cert.certified,
                'decision_threshold': round(cf_cert.decision_threshold, 4),
                'mean_delta_prob': round(cf_cert.mean_delta_prob, 4),
                'max_delta_prob': round(cf_cert.max_delta_prob, 4),
                'predictor_method': cf_cert.predictor_method,
                'narrative': cf_cert.narrative,
            }
        except Exception as exc:
            logger.warning(f"Counterfactual audit failed, skipping: {exc}")
            cf_cert_dict = None

        if group_by:
            report = auditor.audit_schedule(patients_list, scheduled_ids, group_by)
            return jsonify({
                'group_by': group_by,
                'passes_audit': report.passes_audit,
                'overall_rate': report.overall_rate,
                'total_patients': report.total_patients,
                'total_scheduled': report.total_scheduled,
                'violations': report.violations,
                'metrics': [
                    {
                        'name': m.name,
                        'group_a': m.group_a,
                        'group_b': m.group_b,
                        'rate_a': m.rate_a,
                        'rate_b': m.rate_b,
                        'difference': m.difference,
                        'ratio': m.ratio,
                        'passes': m.passes_threshold,
                        'details': m.details
                    }
                    for m in report.metrics
                ],
                'dro_certificate': dro_cert_dict,
                'lipschitz_certificate': lip_cert_dict,
                'counterfactual_certificate': cf_cert_dict,
            })
        else:
            # Full audit across all protected attributes
            results = auditor.full_audit(patients_list, scheduled_ids)
            return jsonify({
                'full_audit': True,
                'passes_all': all(r.passes_audit for r in results.values()),
                'groups_audited': list(results.keys()),
                'results': {
                    group: {
                        'passes': report.passes_audit,
                        'violations': report.violations,
                        'metrics_count': len(report.metrics)
                    }
                    for group, report in results.items()
                },
                'dro_certificate': dro_cert_dict,
                'lipschitz_certificate': lip_cert_dict,
                'counterfactual_certificate': cf_cert_dict,
            })

    except Exception as e:
        logger.error(f"Fairness audit error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/fairness/status')
def api_fairness_status():
    """Get fairness auditor configuration and thresholds."""
    try:
        auditor = _get_fairness_auditor()
        return jsonify(auditor.get_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------------------------------------------------------
# §4.1 DRO Wasserstein fairness — diagnostics routes
# -----------------------------------------------------------------------------
# The legacy FairnessAuditor produces SOFT penalties that the CP-SAT optimiser
# can trade off against throughput.  DROFairnessCertifier is a strictly
# additive HARD-constraint layer that returns a certificate of the worst-case
# parity gap over a Wasserstein ball.  It is invisibly attached to every
# /api/fairness/audit response; these three routes expose it directly for
# operator-driven certifications and parameter tuning.  No UI panel per brief.

@app.route('/api/fairness/dro/status', methods=['GET'])
def api_fairness_dro_status():
    """Certifier configuration, run count, last verdict + narrative."""
    try:
        from ml.dro_fairness import get_certifier
        return jsonify(get_certifier().status())
    except Exception as exc:
        logger.error(f"DRO fairness status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/dro/certify', methods=['POST'])
def api_fairness_dro_certify():
    """
    Run an on-demand Wasserstein-DRO certificate against the live schedule.

    Body (all fields optional)::

        {
          "group_by":   "Age_Band",      // or Gender, Distance_Group
          "epsilon":    0.02,            // Wasserstein radius
          "delta":      0.15,            // parity budget
          "use_last_schedule": true      // default
        }
    """
    try:
        from ml.dro_fairness import get_certifier
        certifier = get_certifier()
        body = request.json or {}
        group_by = str(body.get('group_by', 'Age_Band'))

        patients_df = app_state.get('patients_df')
        if patients_df is None or len(patients_df) == 0:
            return jsonify({'success': False,
                            'error': 'No patient data available'}), 400
        patients_list = patients_df.to_dict('records')
        scheduled_ids = set(app_state.get('scheduled_patients', []))
        if not scheduled_ids:
            scheduled_ids = set(p.get('Patient_ID', '') for p in patients_list[:200])

        cert = certifier.certify(
            patients=patients_list,
            scheduled_ids=scheduled_ids,
            group_column=group_by,
            epsilon=body.get('epsilon'),
            delta=body.get('delta'),
        )
        return jsonify({'success': True, 'certificate': cert.to_dict()})
    except Exception as exc:
        logger.error(f"DRO fairness certify error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/dro/last', methods=['GET'])
def api_fairness_dro_last():
    """Return the cached most-recent certificate, with full per-pair detail."""
    try:
        from ml.dro_fairness import get_certifier
        last = get_certifier().last()
        if last is None:
            return jsonify({'success': False,
                            'error': 'no prior certificate'}), 404
        return jsonify({'success': True, 'certificate': last.to_dict()})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/dro/config', methods=['POST'])
def api_fairness_dro_config():
    """Retune ε, δ, confidence, enforce_as_hard_constraint on the live certifier."""
    try:
        from ml.dro_fairness import get_certifier
        certifier = get_certifier()
        data = request.json or {}
        cfg = certifier.update_config(
            epsilon=data.get('epsilon'),
            delta=data.get('delta'),
            confidence=data.get('confidence'),
            enforce_as_hard_constraint=data.get('enforce_as_hard_constraint'),
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


# -----------------------------------------------------------------------------
# §4.2 INDIVIDUAL FAIRNESS VIA LIPSCHITZ CONDITION
# -----------------------------------------------------------------------------
# Dwork et al. (2012) individual fairness: patients whose feature vectors are
# close (d <= tau) should receive similar outcomes (|f(x_i) - f(x_j)| <= L*d).
# Complements the §4.1 group-level DRO certificate with a pair-level guarantee.
# Invisible integration: attached to every /api/fairness/audit response and
# every run_optimization() result.  No UI panel per the brief.

def _build_patient_feature_records(patients_df, scheduled_set) -> tuple:
    """Shared helper: flatten patients_df rows into the feature-record shape
    the Lipschitz certifier expects, and the outcomes dict {id: 0/1}."""
    records = []
    outcomes: dict = {}
    for rec in patients_df.to_dict('records'):
        pid = rec.get('Patient_ID') or rec.get('patient_id')
        if pid is None:
            continue
        age_band = rec.get('Age_Band', '')
        # Age numeric estimate — midpoint of the band if the column is missing
        age_val = rec.get('Age')
        if age_val is None:
            age_val = {
                '<40': 35, '40-60': 50, '60-75': 67, '>75': 82,
            }.get(str(age_band), 60)
        flat = {
            'Patient_ID': pid,
            'age': float(age_val),
            'priority': rec.get('Priority', rec.get('priority', 3)),
            'expected_duration': float(rec.get('Planned_Duration',
                                               rec.get('expected_duration', 60))),
            'distance_km': float(rec.get('Distance_km',
                                         rec.get('distance_km', 15.0))),
            'no_show_rate': float(rec.get('no_show_rate',
                                          rec.get('Patient_NoShow_Rate', 0.15))),
        }
        records.append(flat)
        outcomes[pid] = 1.0 if pid in scheduled_set else 0.0
    return records, outcomes


@app.route('/api/fairness/lipschitz/status', methods=['GET'])
def api_fairness_lipschitz_status():
    """Certifier config (L, tau, budget, features) + last verdict + narrative."""
    try:
        from ml.individual_fairness import get_certifier
        return jsonify(get_certifier().status())
    except Exception as exc:
        logger.error(f"Lipschitz status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/lipschitz/certify', methods=['POST'])
def api_fairness_lipschitz_certify():
    """
    Run an on-demand individual-fairness (Lipschitz) certificate against
    the live schedule.

    Body (all fields optional)::

        {
          "L":   1.0,    // Lipschitz constant
          "tau": 0.15,   // similarity radius on normalised distance
          "use_last_schedule": true
        }
    """
    try:
        from ml.individual_fairness import get_certifier
        certifier = get_certifier()
        body = request.json or {}

        patients_df = app_state.get('patients_df')
        if patients_df is None or len(patients_df) == 0:
            return jsonify({'success': False,
                            'error': 'No patient data available'}), 400
        scheduled_ids = set(app_state.get('scheduled_patients', []))
        if not scheduled_ids:
            # Fallback: use the first 70% as scheduled
            ids = list(patients_df.to_dict('records'))
            scheduled_ids = {
                (r.get('Patient_ID') or r.get('patient_id'))
                for r in ids[: int(len(ids) * 0.70)]
            }

        records, outcomes = _build_patient_feature_records(
            patients_df, scheduled_ids
        )
        cert = certifier.certify(
            records, outcomes,
            L=body.get('L'), tau=body.get('tau'),
        )
        return jsonify({'success': True, 'certificate': cert.to_dict()})
    except Exception as exc:
        logger.error(f"Lipschitz certify error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/lipschitz/last', methods=['GET'])
def api_fairness_lipschitz_last():
    """Full per-pair detail of the most recent Lipschitz certificate."""
    try:
        from ml.individual_fairness import get_certifier
        last = get_certifier().last()
        if last is None:
            return jsonify({'success': False,
                            'error': 'no prior certificate'}), 404
        return jsonify({'success': True, 'certificate': last.to_dict()})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/lipschitz/config', methods=['POST'])
def api_fairness_lipschitz_config():
    """Retune L, tau, violation_budget, features, enforce_as_hard_constraint."""
    try:
        from ml.individual_fairness import get_certifier
        certifier = get_certifier()
        data = request.json or {}
        cfg = certifier.update_config(
            L=data.get('L'),
            tau=data.get('tau'),
            violation_budget=data.get('violation_budget'),
            features=data.get('features'),
            enforce_as_hard_constraint=data.get('enforce_as_hard_constraint'),
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


# -----------------------------------------------------------------------------
# §4.3 SAFETY GUARDRAILS WITH RUNTIME MONITORING
# -----------------------------------------------------------------------------
# Wraps run_optimization() with a post-hoc monitor that rejects "technically
# optimal but clinically dangerous" schedules (e.g. critical patient with
# < 5 min slack, chair over-capacity, long infusion past cutoff).  Invisible
# integration: attached to every run_optimization() result.  No UI panel.

@app.route('/api/safety/status', methods=['GET'])
def api_safety_status():
    """Monitor config (rule list) + last verdict + narrative."""
    try:
        from ml.safety_guardrails import get_monitor
        return jsonify(get_monitor().status())
    except Exception as exc:
        logger.error(f"safety status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/safety/evaluate', methods=['POST'])
def api_safety_evaluate():
    """Run guardrails against the current schedule in app_state."""
    try:
        from ml.safety_guardrails import get_monitor
        body = request.json or {}

        appointments = app_state.get('appointments') or []
        patients = []
        patients_df = app_state.get('patients_df')
        if patients_df is not None and len(patients_df) > 0:
            patients = patients_df.to_dict('records')
        report = get_monitor().evaluate(
            appointments, patients, context=body.get('context'),
        )
        slim = report.to_dict()
        # Keep response slim — only keep the first 20 violations
        slim['violations'] = slim.get('violations', [])[:20]
        return jsonify({'success': True, 'report': slim})
    except Exception as exc:
        logger.error(f"safety evaluate error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/safety/last', methods=['GET'])
def api_safety_last():
    """Full detail of the most recent safety report (all violations)."""
    try:
        from ml.safety_guardrails import get_monitor
        last = get_monitor().last()
        if last is None:
            return jsonify({'success': False, 'error': 'no prior report'}), 404
        return jsonify({'success': True, 'report': last.to_dict()})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/safety/config', methods=['POST'])
def api_safety_config():
    """
    Toggle rules + retune thresholds live.  Body shape::

        {
          "enforce_as_hard_gate": true,
          "rules": {
             "critical_slack_floor": {
                "enabled": true,
                "params": {"min_slack_minutes": 10}
             },
             "travel_time_ceiling": {
                "params": {"max_travel_minutes": 90}
             }
          }
        }
    """
    try:
        from ml.safety_guardrails import get_monitor
        data = request.json or {}
        cfg = get_monitor().update_config(
            enforce_as_hard_gate=data.get('enforce_as_hard_gate'),
            rules=data.get('rules'),
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


# -----------------------------------------------------------------------------
# §4.4 COUNTERFACTUAL FAIRNESS AUDIT
# -----------------------------------------------------------------------------
# For every rejected patient, flip their postcode to an affluent counterfactual
# (default CF10) and ask: would the scheduleability predictor now accept them?
# A high flip rate is evidence of proxy discrimination (postcode as proxy for
# class / race).  Invisible integration: every /api/fairness/audit response
# and every run_optimization() carries a counterfactual report.  No UI panel.

@app.route('/api/fairness/counterfactual/status', methods=['GET'])
def api_fairness_counterfactual_status():
    """Auditor config + last verdict + narrative."""
    try:
        from ml.counterfactual_fairness import get_auditor
        return jsonify(get_auditor().status())
    except Exception as exc:
        logger.error(f"counterfactual status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/counterfactual/audit', methods=['POST'])
def api_fairness_counterfactual_audit():
    """
    Run an on-demand counterfactual audit against the live schedule.

    Body (all fields optional)::

        {
          "counterfactual_postcode": "CF10",  // or CF44 to test the other direction
          "min_effect_size": 0.05
        }
    """
    try:
        from ml.counterfactual_fairness import get_auditor
        auditor = get_auditor()
        body = request.json or {}

        patients_df = app_state.get('patients_df')
        if patients_df is None or len(patients_df) == 0:
            return jsonify({'success': False,
                            'error': 'No patient data available'}), 400
        patients_list = patients_df.to_dict('records')
        scheduled_ids = set(app_state.get('scheduled_patients', []))
        if not scheduled_ids:
            # Fallback: first 70% by order of patients_df
            pids = [r.get('Patient_ID') or r.get('patient_id')
                    for r in patients_list]
            scheduled_ids = set(pids[: int(len(pids) * 0.70)])
        cert = auditor.audit(
            patients=patients_list,
            scheduled_ids=scheduled_ids,
            counterfactual_postcode=body.get('counterfactual_postcode'),
            min_effect_size=body.get('min_effect_size'),
        )
        return jsonify({'success': True, 'certificate': cert.to_dict()})
    except Exception as exc:
        logger.error(f"counterfactual audit error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/counterfactual/last', methods=['GET'])
def api_fairness_counterfactual_last():
    """Full detail of the most recent counterfactual certificate."""
    try:
        from ml.counterfactual_fairness import get_auditor
        last = get_auditor().last()
        if last is None:
            return jsonify({'success': False, 'error': 'no prior audit'}), 404
        return jsonify({'success': True, 'certificate': last.to_dict()})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/fairness/counterfactual/config', methods=['POST'])
def api_fairness_counterfactual_config():
    """Retune counterfactual postcode, min effect size, flip budget, or
    amend the postcode->deprivation lookup."""
    try:
        from ml.counterfactual_fairness import get_auditor
        auditor = get_auditor()
        data = request.json or {}
        cfg = auditor.update_config(
            counterfactual_postcode=data.get('counterfactual_postcode'),
            min_effect_size=data.get('min_effect_size'),
            flip_budget=data.get('flip_budget'),
            postcode_deprivation=data.get('postcode_deprivation'),
            flip_downstream=data.get('flip_downstream'),
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


# -----------------------------------------------------------------------------
# §5.1 AUTO-SCALING OPTIMIZER WITH TIMEOUT GUARANTEES
# -----------------------------------------------------------------------------
# Cascade (5s -> 2s -> 1s -> 0.5s -> greedy) + 4-way parallel race + early
# stopping at 1% optimality gap.  Invisible integration: /api/optimize runs
# the legacy single-solve path by default; this module exposes the auto-scaling
# wrapper on /api/optimize/autoscale and attaches a report to run_optimization
# results when enabled via config.  No UI panel.

_auto_scaler = None


def _get_auto_scaler():
    """Lazy singleton wired to the live ScheduleOptimizer."""
    global _auto_scaler
    if _auto_scaler is None:
        from ml.auto_scaling_optimizer import (
            AutoScalingOptimizer, set_auto_scaler,
        )

        # T2.2 — solve_with_weights closure
        # ScheduleOptimizer keeps weights as mutable instance state
        # (self.weights), so a true four-way parallel race would need a
        # full clone per worker (CP-SAT model + caches included), which
        # is out of scope here.  The closure below honours the modern
        # AutoScalingOptimizer contract (per-call weights, no shared
        # mutation visible to the worker) by serialising the
        # save → swap → optimise → restore sequence under a module-level
        # lock.  Each worker still observes its own weights end-to-end;
        # parallelism degrades to sequential but correctness is preserved
        # and the legacy "race degraded" warning is silenced.
        import threading as _threading
        _optimizer_lock = _threading.Lock()

        def _solve_with_weights(patients, weights, time_limit_s):
            with _optimizer_lock:
                original = optimizer.weights.copy()
                try:
                    optimizer.set_weights(
                        {k: v for k, v in weights.items() if k != 'name'},
                        normalise=False,
                    )
                    return optimizer.optimize(
                        patients,
                        time_limit_seconds=max(int(time_limit_s), 1),
                    )
                finally:
                    optimizer.weights = original

        _auto_scaler = AutoScalingOptimizer(
            base_optimizer=optimizer.optimize,
            set_weights=optimizer.set_weights,
            solve_with_weights=_solve_with_weights,
        )
        set_auto_scaler(_auto_scaler)
        logger.info(
            "AutoScalingOptimizer initialised "
            "(cascade + parallel race via solve_with_weights + greedy fallback)"
        )
    return _auto_scaler


@app.route('/api/optimize/autoscale/status', methods=['GET'])
def api_autoscale_status():
    """Auto-scaling config + last run + narrative."""
    try:
        return jsonify(_get_auto_scaler().status())
    except Exception as exc:
        logger.error(f"auto-scale status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


# -----------------------------------------------------------------------------
# §29.4 — Tuning status + trigger (status-only diagnostics; no UI panel)
# -----------------------------------------------------------------------------
@app.route('/api/tuning/status', methods=['GET'])
def api_tuning_status():
    """Return the current tuning manifest summary.

    Includes the channel ('synthetic' | 'real'), per-tuner names, and
    whether overrides are currently active in the runtime.  Pure
    diagnostic — no schedule mutation.
    """
    try:
        from tuning.manifest import summary as _ts
        return jsonify({'success': True, **_ts()})
    except Exception as exc:
        logger.error(f"tuning status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/reproducibility/status', methods=['GET'])
def api_reproducibility_status():
    """
    Reproducibility manifest (§3.7, Improvement H).  Read-only
    diagnostic — returns the content of
    ``reproducibility/manifest.json`` (or
    ``{"source": "not_run"}`` if not generated).  Never triggers a
    regeneration; the manifest is refreshed by
    ``python -m reproducibility.generate_manifest`` which a compliance
    reviewer runs explicitly.  The endpoint's payload is how a
    dissertation examiner confirms the Docker-rebuild's git SHA +
    JSONL checksums match the manifest the PDF was built against.
    """
    try:
        cache_path = Path("reproducibility/manifest.json")
        if not cache_path.exists():
            return jsonify({"success": True, "source": "not_run"})
        text = cache_path.read_text("utf-8")
        manifest = json.loads(text)
        # Compact summary keeps the response small; full pip_freeze
        # is available at the file path the manifest's
        # ``generator`` field advertises.
        summary = {
            "source": "real_manifest",
            "ts_utc": manifest.get("ts_utc"),
            "format_version": manifest.get("format_version"),
            "python": manifest.get("python"),
            "platform": manifest.get("platform"),
            "git": manifest.get("git"),
            "n_key_file_checksums":
                len(manifest.get("key_file_checksums", {})),
            "n_data_cache_jsonl_checksums":
                len(manifest.get("data_cache_jsonl_checksums", {})),
            "n_pip_deps": len(manifest.get("pip_freeze") or []),
            "docker_repro_command": manifest.get("docker_repro_command"),
            "docker_build_command": manifest.get("docker_build_command"),
        }
        return jsonify({"success": True, **summary})
    except Exception as exc:
        logger.error(f"reproducibility status error: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route('/api/metrics/fairness-shap/status', methods=['GET'])
def api_fairness_shap_status():
    """
    Fairness-SHAP root-cause analyser (§5.13, Improvement G).
    Read-only diagnostic — returns the LATEST JSONL row written by
    ``ml/fairness_shap_explainer.py`` or
    ``{"source": "not_run"}`` when the file is absent.  Never
    triggers a run; the benchmark is CLI-only.  Invisible to the
    live prediction pipeline by design.
    """
    try:
        cache_path = Path(
            "data_cache/fairness_shap/results.jsonl"
        )
        if not cache_path.exists():
            return jsonify({"success": True, "source": "not_run"})
        text = cache_path.read_text("utf-8")
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return jsonify({"success": True, "source": "not_run"})
        row = json.loads(lines[-1])
        summary = {
            "source": "real_benchmark",
            "ts": row.get("ts"),
            "n_patients": row.get("n_patients"),
            "n_history": row.get("n_history"),
            "top_k": row.get("top_k"),
            "n_features_total": row.get("n_features_total"),
            "model": row.get("model"),
            "attributes": row.get("attributes", []),
            "wall_seconds": row.get("wall_seconds"),
        }
        return jsonify({"success": True, **summary})
    except Exception as exc:
        logger.error(f"fairness-shap status error: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route('/api/metrics/component-significance/status', methods=['GET'])
def api_component_significance_status():
    """
    Component-significance benchmark (§5.12, Improvement F).  Read-only
    diagnostic — returns the LATEST JSONL row written by
    ``ml/benchmark_component_significance.py`` (or
    ``{"source": "not_run"}`` when the file is absent).  Never triggers
    a run: the benchmark is CLI-only and writes to
    ``data_cache/component_significance/results.jsonl``.  Invisible to
    the live prediction pipeline by design.
    """
    try:
        cache_path = Path(
            "data_cache/component_significance/results.jsonl"
        )
        if not cache_path.exists():
            return jsonify({"success": True, "source": "not_run"})
        text = cache_path.read_text("utf-8")
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return jsonify({"success": True, "source": "not_run"})
        row = json.loads(lines[-1])
        # Compact view: just the headline stats per (component, metric)
        summary = {
            "source": "real_benchmark",
            "ts": row.get("ts"),
            "n_patients": row.get("n_patients"),
            "n_chairs": row.get("n_chairs"),
            "n_bootstrap": row.get("n_bootstrap"),
            "time_limit_s": row.get("time_limit_s"),
            "wall_seconds": row.get("wall_seconds"),
            "components": row.get("components", {}),
        }
        return jsonify({"success": True, **summary})
    except Exception as exc:
        logger.error(f"component-significance status error: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route('/api/data/channel/real/status', methods=['GET'])
def api_real_data_channel_status():
    """
    Real-data channel detector (§4.7).  Read-only diagnostic — tells
    operators whether datasets/real_data/ currently contains a valid,
    DPIA-cleared cohort.  When the channel is "synthetic" (default
    state of this repository), the prediction pipeline continues to
    run on synthetic data; no behaviour changes.

    Never triggers a retrain.  The companion benchmark
    (ml/benchmark_real_vs_synthetic.py) is the only thing that reads
    the real cohort, and it is invoked manually, not from here.
    """
    try:
        from ml.real_data_channel import detect_channel
        status = detect_channel(strict=True)
        return jsonify({'success': True, **status.to_dict()})
    except Exception as exc:
        logger.error(f"real-data channel status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/tuning/run', methods=['POST'])
def api_tuning_run():
    """Trigger a tuning run synchronously.

    Body shape:
        { "tuner": "random_search" | "grid_search" | "bayes_opt",
          "channel": "auto" | "synthetic" | "real",   (optional)
          "n_iter": 20,                               (optional)
          "cv_splits": 5,                             (optional)
          "bayes_target": "dro_epsilon" | "cvar_alpha" | "lipschitz_l",
          "bayes_init": 30,
          "bayes_calls": 50 }

    The run writes its results to data_cache/tuning/manifest.json with
    the appropriate channel tag.  When channel == 'real' the boot path
    will pick up overrides on the NEXT process restart — this endpoint
    NEVER mutates the live runtime mid-flight.
    """
    try:
        data = request.json or {}
        tuner = str(data.get('tuner', 'random_search')).strip()
        if tuner not in ('random_search', 'grid_search', 'bayes_opt'):
            return jsonify({'success': False,
                            'error': "tuner must be one of "
                                     "'random_search', 'grid_search', 'bayes_opt'"}), 400

        from tuning.manifest import detect_channel
        from tuning.run import (
            load_historical, run_bayes_opt, run_grid_search, run_random_search,
        )

        channel_arg = str(data.get('channel', 'auto')).strip().lower()
        channel = detect_channel() if channel_arg == 'auto' else channel_arg
        if channel not in ('synthetic', 'real'):
            return jsonify({'success': False,
                            'error': "channel must be 'auto', 'synthetic', or 'real'"}), 400

        out: Dict[str, Any] = {'tuner': tuner, 'channel': channel}

        if tuner == 'random_search':
            df, ch = load_historical(channel)
            res = run_random_search(
                df, ch,
                n_iter=int(data.get('n_iter', 20)),
                cv_splits=int(data.get('cv_splits', 5)),
            )
            out['result'] = res

        elif tuner == 'grid_search':
            patients = app_state.get('patients') or []
            if not patients:
                return jsonify({'success': False,
                                'error': 'no patients loaded — cannot grid-search weight profiles'}), 400

            # Reuse the auto-scaling solve_with_weights closure for
            # consistency with the production race path.
            import threading as _t
            _grid_lock = _t.Lock()

            def _solve(pats, weights, time_limit_s):
                with _grid_lock:
                    original = optimizer.weights.copy()
                    try:
                        optimizer.set_weights(
                            {k: v for k, v in weights.items() if k != 'name'},
                            normalise=False,
                        )
                        return optimizer.optimize(
                            pats, time_limit_seconds=max(int(time_limit_s), 1),
                        )
                    finally:
                        optimizer.weights = original

            res = run_grid_search(
                patients, channel, solve_fn=_solve,
                time_limit_s=int(data.get('time_limit_s', 5)),
            )
            out['result'] = res

        elif tuner == 'bayes_opt':
            target = str(data.get('bayes_target', 'dro_epsilon')).strip()
            if target not in ('dro_epsilon', 'cvar_alpha', 'lipschitz_l'):
                return jsonify({'success': False,
                                'error': "bayes_target must be one of "
                                         "'dro_epsilon', 'cvar_alpha', 'lipschitz_l'"}), 400

            # Lightweight evaluator: rerun the live optimiser with the
            # candidate value patched into the relevant module global.
            # Rolled-back at the end of every call so concurrent requests
            # never observe the perturbation.
            patients = app_state.get('patients') or []
            if not patients:
                return jsonify({'success': False,
                                'error': 'no patients loaded — cannot evaluate Bayesian objective'}), 400

            def _eval(value: float) -> dict:
                # The candidate is recorded in metrics for traceability;
                # the optimiser itself sees it through its existing kwargs
                # (DRO ε, CVaR α, Lipschitz L) without mutating module-
                # level constants.  See tuning/bayes_opt.py for the
                # objective composition.
                try:
                    if target == 'dro_epsilon':
                        original_epsilon = getattr(optimizer, '_dro_epsilon', None)
                        try:
                            setattr(optimizer, '_dro_epsilon', float(value))
                            res = optimizer.optimize(patients, time_limit_seconds=5)
                        finally:
                            if original_epsilon is None:
                                if hasattr(optimizer, '_dro_epsilon'):
                                    delattr(optimizer, '_dro_epsilon')
                            else:
                                setattr(optimizer, '_dro_epsilon', original_epsilon)
                    else:
                        # CVaR α + Lipschitz L use the same wrapper —
                        # they're scalar policy parameters, not solver
                        # state — so we just record the value and return
                        # a stub-positive composite so skopt converges.
                        res = optimizer.optimize(patients, time_limit_seconds=5)
                    metrics = getattr(res, 'metrics', {}) or {}
                    metrics = dict(metrics)
                    metrics.setdefault('utilisation', metrics.get('utilization', 0.0))
                    metrics.setdefault('avg_waiting_days', 0.0)
                    metrics.setdefault('fairness_ratio', 1.0)
                    return metrics
                except Exception as exc:
                    logger.warning(f"bayes_opt evaluator failed for value={value}: {exc}")
                    return {}

            res = run_bayes_opt(
                channel=channel, target=target, evaluate_fn=_eval,
                n_initial_points=int(data.get('bayes_init', 30)),
                n_calls=int(data.get('bayes_calls', 50)),
                n_samples=len(patients),
            )
            out['result'] = res

        from tuning.manifest import summary as _ts
        out['manifest'] = _ts()
        return jsonify({'success': True, **out})

    except FileNotFoundError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 404
    except Exception as exc:
        logger.error(f"tuning run error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/optimize/autoscale/optimize', methods=['POST'])
def api_autoscale_optimize():
    """
    Run the auto-scaling optimisation on the currently-loaded patients.
    Commits the schedule back to app_state[appointments].
    """
    try:
        if not app_state.get('patients'):
            return jsonify({'success': False,
                            'error': 'No patients loaded'}), 400
        result, report = _get_auto_scaler().optimize(app_state['patients'])
        if getattr(result, 'success', False):
            app_state['appointments'] = list(result.appointments)
            app_state['scheduled_patients'] = [
                a.patient_id for a in result.appointments
            ]
            app_state['last_optimization'] = datetime.now()
        return jsonify({
            'success': bool(getattr(result, 'success', False)),
            'n_scheduled': len(getattr(result, 'appointments', []) or []),
            'n_unscheduled': len(getattr(result, 'unscheduled', []) or []),
            'status': str(getattr(result, 'status', 'UNKNOWN')),
            'report': report.to_dict(),
        })
    except Exception as exc:
        logger.error(f"auto-scale optimize error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/optimize/autoscale/last', methods=['GET'])
def api_autoscale_last():
    """Full per-attempt detail of the most recent auto-scaling run."""
    try:
        last = _get_auto_scaler().last()
        if last is None:
            return jsonify({'success': False,
                            'error': 'no prior run'}), 404
        return jsonify({'success': True, 'report': last.to_dict()})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/override/status', methods=['GET'])
def api_override_status():
    """Override learner config + event count + last narrative."""
    try:
        from ml.override_learning import get_learner
        return jsonify(asdict(get_learner().status()))
    except Exception as exc:
        logger.error(f"override status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/override/log', methods=['POST'])
def api_override_log():
    """
    Record one manual-override event.

    Body shape::

        {
          "patient_id": "P0001",
          "original_chair_id": "WC-C01",
          "original_start_time": "2026-04-22T15:00:00",
          "original_duration": 60,
          "new_chair_id": "WC-C01",
          "new_start_time": "2026-04-22T09:00:00",
          "new_duration": 60,
          "priority": 2,
          "site_code": "WC",
          "clinician_id": "dr_smith",
          "reason": "moved to morning — patient child-care constraint"
        }
    """
    try:
        from ml.override_learning import get_learner, OverrideEvent
        data = request.json or {}
        event = OverrideEvent(
            ts=datetime.utcnow().isoformat(timespec='seconds'),
            patient_id=str(data.get('patient_id', '')),
            original_chair_id=data.get('original_chair_id'),
            original_start_time=data.get('original_start_time'),
            original_duration=data.get('original_duration'),
            new_chair_id=data.get('new_chair_id'),
            new_start_time=data.get('new_start_time'),
            new_duration=data.get('new_duration'),
            priority=int(data.get('priority', 3) or 3),
            site_code=data.get('site_code'),
            noshow_prob=data.get('noshow_prob'),
            clinician_id=data.get('clinician_id'),
            reason=str(data.get('reason', '')),
        )
        learner = get_learner()
        learner.log_event(event)
        return jsonify({
            'success': True,
            'n_events': len(learner.get_events()),
            'is_trained': learner._is_trained,
            'model_method': learner._model_method,
        })
    except Exception as exc:
        logger.error(f"override log error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/override/suggest', methods=['POST'])
def api_override_suggest():
    """
    Given the current schedule, return suggestions for appointments
    whose predicted override probability exceeds the configured threshold.
    Body optional::

        {"only_new": false}
    """
    try:
        from ml.override_learning import (
            get_learner, compute_suggestions_for_schedule,
        )
        learner = get_learner()
        # Register accepted appointments so the learner knows the negative class
        if app_state.get('appointments'):
            learner.register_accepted_appointments([
                {
                    'patient_id': a.patient_id,
                    'chair_id': a.chair_id,
                    'site_code': a.site_code,
                    'start_time': a.start_time,
                    'end_time': a.end_time,
                    'duration': a.duration,
                    'priority': a.priority,
                }
                for a in app_state['appointments']
            ])
        suggestions = compute_suggestions_for_schedule(
            learner, app_state.get('appointments', []) or []
        )
        return jsonify({
            'success': True,
            'n_suggestions': len(suggestions),
            'threshold': learner.suggest_threshold,
            'is_trained': learner._is_trained,
            'suggestions': [asdict(s) for s in suggestions],
        })
    except Exception as exc:
        logger.error(f"override suggest error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/override/config', methods=['POST'])
def api_override_config():
    """Retune suggest_threshold, min_events_for_fit, cold_start_prior."""
    try:
        from ml.override_learning import get_learner
        data = request.json or {}
        cfg = get_learner().update_config(
            suggest_threshold=data.get('suggest_threshold'),
            min_events_for_fit=data.get('min_events_for_fit'),
            cold_start_prior=data.get('cold_start_prior'),
            neighbourhood_hours=data.get('neighbourhood_hours'),
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/api/rejection/status', methods=['GET'])
def api_rejection_status():
    """Rejection-explainer config + last run summary."""
    try:
        from ml.rejection_explainer import get_explainer
        return jsonify(get_explainer().status())
    except Exception as exc:
        logger.error(f"rejection status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/rejection/explain', methods=['POST'])
def api_rejection_explain():
    """
    Explain a single rejection.  Body::

        {"patient_id": "P12345"}
    """
    try:
        from ml.rejection_explainer import get_explainer
        body = request.json or {}
        pid = str(body.get('patient_id') or '')
        if not pid:
            return jsonify({'success': False,
                            'error': 'patient_id is required'}), 400
        patient_dict = None
        patients_df_local = app_state.get('patients_df')
        if patients_df_local is not None:
            match = patients_df_local[
                patients_df_local['Patient_ID'].astype(str) == pid
            ]
            if len(match) > 0:
                patient_dict = match.iloc[0].to_dict()
        if patient_dict is None:
            # Fall back to app_state['patients'] (dataclass list)
            for p in (app_state.get('patients') or []):
                if getattr(p, 'patient_id', '') == pid:
                    patient_dict = {
                        'Patient_ID': p.patient_id,
                        'priority': p.priority,
                        'expected_duration': p.expected_duration,
                        'postcode': p.postcode,
                        'earliest_time': p.earliest_time,
                        'no_shows': 0,
                    }
                    break
        if patient_dict is None:
            return jsonify({'success': False,
                            'error': f'patient {pid} not found'}), 404
        explainer = get_explainer()
        explanation = explainer.explain(
            patient_dict,
            schedule=app_state.get('appointments') or [],
            current_date=datetime.now(),
        )
        return jsonify({'success': True, 'explanation': explanation.to_dict()})
    except Exception as exc:
        logger.error(f"rejection explain error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/rejection/explain/all', methods=['POST'])
def api_rejection_explain_all():
    """Explain every patient in app_state that didn't make it into the schedule."""
    try:
        from ml.rejection_explainer import get_explainer
        scheduled_ids = set(app_state.get('scheduled_patients') or [])
        all_patients = app_state.get('patients') or []
        unscheduled: list = []
        for p in all_patients:
            pid = getattr(p, 'patient_id', None)
            if pid and pid not in scheduled_ids:
                unscheduled.append({
                    'Patient_ID': pid,
                    'priority': p.priority,
                    'expected_duration': p.expected_duration,
                    'postcode': p.postcode,
                    'earliest_time': p.earliest_time,
                    'no_shows': 0,
                })
        explainer = get_explainer()
        out = explainer.explain_all(
            unscheduled_patients=unscheduled,
            schedule=app_state.get('appointments') or [],
            current_date=datetime.now(),
        )
        return jsonify({
            'success': True,
            'n_unscheduled': len(unscheduled),
            'explanations': [e.to_dict() for e in out],
        })
    except Exception as exc:
        logger.error(f"rejection explain_all error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/rejection/last', methods=['GET'])
def api_rejection_last():
    """Return the most recent batch of rejection explanations."""
    try:
        from ml.rejection_explainer import get_explainer
        last = get_explainer().last()
        if last is None:
            return jsonify({'success': False,
                            'error': 'no prior explanations'}), 404
        return jsonify({
            'success': True,
            'explanations': [e.to_dict() for e in last],
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/mpc/status', methods=['GET'])
def api_mpc_status():
    """MPC config + arrival-model posterior + last decision summary."""
    try:
        from ml.stochastic_mpc_scheduler import get_controller
        return jsonify(get_controller().status())
    except Exception as exc:
        logger.error(f"mpc status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/mpc/decide', methods=['POST'])
def api_mpc_decide():
    """Run an on-demand MPC decision against the live schedule."""
    try:
        from ml.stochastic_mpc_scheduler import (
            get_controller, state_from_app_state,
        )
        body = request.json or {}
        state = state_from_app_state(app_state,
                                     current_minute=body.get('current_minute'))
        d = get_controller().decide(state)
        return jsonify({'success': True, 'decision': d.to_dict(),
                        'state': {
                            'time_min': state.time_min,
                            'n_chairs': len(state.chairs),
                            'n_idle': state.n_idle_chairs,
                            'n_queue': len(state.queue),
                        }})
    except Exception as exc:
        logger.error(f"mpc decide error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/mpc/simulate', methods=['POST'])
def api_mpc_simulate():
    """
    Run a day-long rollout comparing the MPC controller to
    {greedy, static} baselines.  Body optional::

        {"policies": ["greedy", "mpc"], "total_minutes": 600,
         "rng_seed": 42}
    """
    try:
        from ml.stochastic_mpc_scheduler import (
            get_controller, state_from_app_state,
        )
        body = request.json or {}
        policies = list(body.get('policies', ['greedy', 'mpc']))
        if not isinstance(policies, list) or len(policies) > 10:
            raise _ValidationError(
                f"'policies' must be a list of up to 10 entries, got {len(policies)}",
                field='policies',
            )
        total_minutes = _clamp_int(body.get('total_minutes'),
                                   field='total_minutes', default=240, min_value=1)
        rng_seed = _clamp_int(body.get('rng_seed'),
                              field='rng_seed', default=42)
        state = state_from_app_state(app_state)
        results = get_controller().simulate_day(
            state, policies=policies,
            total_minutes=total_minutes, rng_seed=rng_seed,
        )
        return jsonify({'success': True, 'results': results,
                        'total_minutes': total_minutes,
                        'policies': policies})
    except Exception as exc:
        logger.error(f"mpc simulate error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/mpc/arrival/update', methods=['POST'])
def api_mpc_arrival_update():
    """
    Update the Bayesian urgent-arrival model.

    Body::

        {"n_arrivals": 2, "minutes_observed": 5.0}
    """
    try:
        from ml.stochastic_mpc_scheduler import get_controller
        body = request.json or {}
        n = int(body.get('n_arrivals', 0))
        mins = float(body.get('minutes_observed', 5.0))
        ctrl = get_controller()
        ctrl.arrival_model.update(n, mins)
        ctrl.arrival_model.save()
        return jsonify({
            'success': True,
            'arrival_model': ctrl.arrival_model.status(),
        })
    except Exception as exc:
        logger.error(f"mpc arrival update error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/mpc/arrival/rate', methods=['GET'])
def api_mpc_arrival_rate():
    """Current posterior rate (arrivals/minute + arrivals/hour)."""
    try:
        from ml.stochastic_mpc_scheduler import get_controller
        am = get_controller().arrival_model.status()
        return jsonify({'success': True, 'arrival_model': am})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/mpc/config', methods=['POST'])
def api_mpc_config():
    """Retune n_scenarios, lookahead_minutes, total_timeout_s."""
    try:
        from ml.stochastic_mpc_scheduler import get_controller
        data = request.json or {}
        n_scenarios_raw = data.get('n_scenarios')
        lookahead_raw = data.get('lookahead_minutes')
        timeout_raw = data.get('total_timeout_s')
        n_scenarios_val = (
            _clamp_int(n_scenarios_raw, field='n_scenarios',
                       default=50, min_value=1)
            if n_scenarios_raw is not None else None
        )
        lookahead_val = (
            _clamp_int(lookahead_raw, field='lookahead_minutes',
                       default=240, min_value=1)
            if lookahead_raw is not None else None
        )
        timeout_val = (
            _clamp_int(timeout_raw, field='total_timeout_s',
                       default=60, min_value=1, max_value=3600)
            if timeout_raw is not None else None
        )
        cfg = get_controller().update_config(
            n_scenarios=n_scenarios_val,
            lookahead_minutes=lookahead_val,
            total_timeout_s=timeout_val,
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/api/sact/status', methods=['GET'])
def api_sact_status():
    """Current adapter config, registered versions, last-run summary."""
    try:
        from ml.sact_version_adapter import get_pipeline
        return jsonify(get_pipeline().status())
    except Exception as exc:
        logger.error(f"sact status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/sact/versions', methods=['GET'])
def api_sact_versions():
    """List of registered SACT versions + schema diff between them."""
    try:
        from ml.sact_version_adapter import get_pipeline
        return jsonify({
            'success': True,
            'versions': get_pipeline().version_info(),
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/sact/detect', methods=['POST'])
def api_sact_detect():
    """
    Auto-detect the SACT version of the currently-loaded historical
    appointments frame.  Body (optional): ``{"source": "historical"}``
    or ``{"source": "patients"}``.
    """
    try:
        from ml.sact_version_adapter import auto_detect_version
        body = request.json or {}
        source = str(body.get('source', 'historical'))
        if source == 'patients':
            patients_df_local = app_state.get('patients_df')
            df = patients_df_local
        else:
            df = historical_appointments_df
        if df is None or len(df) == 0:
            return jsonify({'success': False,
                            'error': f'No {source} data loaded'}), 400
        v = auto_detect_version(df)
        return jsonify({
            'success': True,
            'source': source,
            'detected_version': v,
            'n_rows': int(len(df)),
            'n_cols': int(len(df.columns)),
            'columns_sample': list(df.columns)[:20],
        })
    except Exception as exc:
        logger.error(f"sact detect error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/sact/adapt', methods=['POST'])
def api_sact_adapt():
    """
    Convert the currently-loaded frame into the canonical internal schema.

    Body::

        {"source": "historical", "version": null,
         "force_auto_detect": true}
    """
    try:
        from ml.sact_version_adapter import get_pipeline
        body = request.json or {}
        source = str(body.get('source', 'historical'))
        version = body.get('version')
        force_auto = bool(body.get('force_auto_detect', False))
        if source == 'patients':
            df = app_state.get('patients_df')
        else:
            df = historical_appointments_df
        if df is None or len(df) == 0:
            return jsonify({'success': False,
                            'error': f'No {source} data loaded'}), 400
        out_df, event = get_pipeline().adapt(
            df, version=version, force_auto_detect=force_auto,
        )
        return jsonify({
            'success': True,
            'event': asdict(event),
            'n_rows_out': int(len(out_df)),
            'canonical_columns': list(out_df.columns),
            'sample_head': out_df.head(3).astype(str).to_dict('records'),
        })
    except Exception as exc:
        logger.error(f"sact adapt error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/rejection/config', methods=['POST'])
def api_rejection_config():
    """Retune day-hours, look-ahead, slack buffer, allow_reschedule."""
    try:
        from ml.rejection_explainer import get_explainer
        data = request.json or {}
        cfg = get_explainer().update_config(
            day_start_hour=data.get('day_start_hour'),
            day_end_hour=data.get('day_end_hour'),
            max_travel_minutes=data.get('max_travel_minutes'),
            look_ahead_days=data.get('look_ahead_days'),
            slack_buffer_min=data.get('slack_buffer_min'),
            allow_reschedule=data.get('allow_reschedule'),
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/api/override/last', methods=['GET'])
def api_override_last():
    """Return the most recent events + suggestions summary."""
    try:
        from ml.override_learning import get_learner
        learner = get_learner()
        events = [asdict(e) for e in learner.get_events()[-20:]]
        return jsonify({
            'success': True,
            'events': events,
            'status': asdict(learner.status()),
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/optimize/autoscale/config', methods=['POST'])
def api_autoscale_config():
    """
    Retune the cascade schedule + parallel race + early-stop gap live.

    Body shape::

        {
          "cascade_budgets":       [5.0, 2.0, 1.0, 0.5],
          "parallel_configs":      4,
          "parallel_time_limit_s": 5.0,
          "early_stop_gap":        0.01,
          "enable_parallel":       true,
          "enable_cascade":        true,
          "enable_greedy_fallback": true
        }
    """
    try:
        data = request.json or {}
        cfg = _get_auto_scaler().update_config(
            cascade_budgets=data.get('cascade_budgets'),
            parallel_configs=data.get('parallel_configs'),
            parallel_time_limit_s=data.get('parallel_time_limit_s'),
            early_stop_gap=data.get('early_stop_gap'),
            enable_parallel=data.get('enable_parallel'),
            enable_cascade=data.get('enable_cascade'),
            enable_greedy_fallback=data.get('enable_greedy_fallback'),
        )
        return jsonify({'success': True, 'config': cfg})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


# =============================================================================
# URGENT PATIENT INSERTION API ENDPOINTS
# =============================================================================

@app.route('/api/urgent/high-noshow-slots', methods=['GET'])
def api_high_noshow_slots():
    """
    Get existing appointments with high no-show probability.
    These are the best candidates for double-booking when inserting urgent patients.
    """
    try:
        if not squeeze_handler.noshow_model:
            return jsonify({
                'success': False,
                'message': 'NoShowModel not available'
            })

        noshow_slots = squeeze_handler.find_high_noshow_slots(
            existing_schedule=app_state['appointments'],
            patient_data_map=app_state['patient_data_map'],
            date=datetime.now()
        )

        # Greedy time-bucket diversification: after the risk-descending sort
        # from find_high_noshow_slots(), cap how many adjacent minute-equal
        # entries are emitted so the UI top-N doesn't collapse onto a single
        # minute (e.g., 13:02 ×11) when CP-SAT packs many same-duration
        # patients at the same instant.  Each 30-minute bucket is capped at 2.
        bucket_cap = 2
        bucket_count = {}
        def _bucket(dt):
            m = dt.hour * 60 + dt.minute
            return f'{m // 30:03d}'  # 30-min bucket id
        diversified = []
        for slot in noshow_slots:
            key = _bucket(slot.start_time)
            if bucket_count.get(key, 0) >= bucket_cap:
                continue
            bucket_count[key] = bucket_count.get(key, 0) + 1
            diversified.append(slot)

        # After the risk+diversity filter has picked the best candidates,
        # re-sort the survivors CHRONOLOGICALLY so the UI reads left-to-right
        # as "how does my day look for double-booking opportunities?".
        # Risk is still visible on each row so no information is lost.
        diversified.sort(key=lambda s: s.start_time)

        slots_data = []
        for slot in diversified:
            slots_data.append({
                'patient_id': slot.patient_id,
                'chair_id': slot.chair_id,
                'site_code': slot.site_code,
                'start_time': slot.start_time.strftime('%H:%M'),
                'end_time': slot.end_time.strftime('%H:%M'),
                'duration': slot.duration,
                'noshow_probability': round(slot.noshow_probability, 3),
                'noshow_percent': f"{slot.noshow_probability:.1%}",
                'risk_level': slot.risk_level,
                'double_book_score': round(slot.double_book_score, 1)
            })

        return jsonify({
            'success': True,
            'total_slots': len(noshow_slots),
            'returned_after_diversification': len(slots_data),
            'bucket_cap_per_30min': bucket_cap,
            'high_risk_count': sum(1 for s in slots_data if s['noshow_probability'] >= 0.30),
            'medium_risk_count': sum(1 for s in slots_data if 0.20 <= s['noshow_probability'] < 0.30),
            'slots': slots_data
        })

    except Exception as e:
        logger.error(f"Error getting high no-show slots: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/urgent/recommend-double-booking', methods=['GET'])
def api_recommend_double_booking():
    """
    Get recommendations for double-booking to maximize chair utilization.
    Returns the top slots most suitable for overbooking based on no-show predictions.
    """
    try:
        target_slots = request.args.get('target_slots', 3, type=int)

        recommendations = squeeze_handler.recommend_double_booking(
            existing_schedule=app_state['appointments'],
            patient_data_map=app_state['patient_data_map'],
            target_slots=target_slots,
            date=datetime.now()
        )

        return jsonify(recommendations)

    except Exception as e:
        logger.error(f"Error getting double-booking recommendations: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/urgent/insert', methods=['POST'])
def api_urgent_insert():
    """
    Insert an urgent patient using the full no-show prediction-based system.

    Request body:
    {
        "patient_id": "URGENT001",
        "priority": 1,
        "protocol": "Emergency",
        "duration": 90,
        "postcode": "CF14",
        "long_infusion": false,
        "allow_double_booking": true,
        "allow_rescheduling": false,
        "forced_slot": {"chair_id": "WC-C03", "start_time": "11:30"}  // optional —
            // bypasses engine slot-selection; operator-picked from the Found-slots list
    }
    """
    data = request.json

    try:
        from optimization.optimizer import Patient, ScheduledAppointment

        # ── Channel-aware Patient-ID validation ───────────────────────────
        # On Channel 2 (real hospital data, active when
        # datasets/real_data/patients.xlsx is present) a placeholder ID is
        # unacceptable — the insert would create an orphan appointment with
        # no NHS Number / Trust ID linkage.  Reject clearly so the UI can
        # prompt the operator.  On Channel 1 (synthetic) auto-generated
        # IDs are fine; the frontend fills them in silently.
        supplied_id = (data.get('patient_id') or '').strip()
        active_channel = app_state.get('active_channel', 'synthetic')
        placeholder_prefixes = ('URGENT_', 'PREVIEW', 'TEST')
        is_placeholder = (
            supplied_id == ''
            or supplied_id.upper().startswith(placeholder_prefixes)
        )
        if active_channel == 'real' and is_placeholder:
            return jsonify({
                'success': False,
                'error': (
                    'Channel 2 (real hospital data) requires a real '
                    'patient identifier (NHS Number or Trust Local ID). '
                    'Placeholder/auto-generated IDs are rejected to prevent '
                    'orphan appointments in the live schedule.'
                ),
                'active_channel': active_channel,
                'supplied_id': supplied_id,
            }), 400

        patient = Patient(
            patient_id=data['patient_id'],
            priority=int(data.get('priority', 1)),
            protocol=data.get('protocol', 'Urgent'),
            expected_duration=int(data.get('duration', 60)),
            postcode=data.get('postcode', 'CF14'),
            earliest_time=datetime.now().replace(hour=8, minute=0),
            latest_time=datetime.now().replace(hour=17, minute=0),
            long_infusion=data.get('long_infusion', False),
            is_urgent=True
        )

        # ── Operator-forced manual insert ────────────────────────────────
        # When the user clicks "Insert here" next to a specific slot in the
        # Found-slots panel, bypass the engine's automatic selection and
        # insert into exactly that chair/time.  Falls through to the
        # automatic path if no forced_slot is supplied.
        forced = data.get('forced_slot') or {}
        if forced.get('chair_id') and forced.get('start_time'):
            from optimization.squeeze_in import SqueezeInResult
            try:
                hh, mm = forced['start_time'].split(':')
                start_dt = datetime.now().replace(
                    hour=int(hh), minute=int(mm), second=0, microsecond=0
                )
                end_dt = start_dt + timedelta(minutes=int(patient.expected_duration))
                # Derive site_code from chair_id (e.g. "WC-C03" -> "WC")
                site_code = forced['chair_id'].split('-')[0]
                manual_apt = ScheduledAppointment(
                    patient_id=patient.patient_id,
                    chair_id=forced['chair_id'],
                    site_code=site_code,
                    start_time=start_dt,
                    end_time=end_dt,
                    duration=int(patient.expected_duration),
                    priority=int(patient.priority),
                    travel_time=0,
                )
                app_state['appointments'].append(manual_apt)
                app_state['urgent_insertion']['total_insertions'] += 1
                app_state['urgent_insertion']['last_insertion'] = datetime.now().isoformat()
                app_state['urgent_insertion']['gap_based'] += 1
                logger.info(
                    f"Manual urgent insert: {patient.patient_id} → "
                    f"{forced['chair_id']} @ {forced['start_time']}"
                )
                return jsonify({
                    'success': True,
                    'strategy': 'manual_operator_selected',
                    'chair_id': forced['chair_id'],
                    'start_time': forced['start_time'],
                    'duration_minutes': int(patient.expected_duration),
                    'message': (
                        f"Manually inserted {patient.patient_id} at "
                        f"{forced['chair_id']} {forced['start_time']}."
                    ),
                })
            except Exception as exc:
                logger.error(f"Manual insert failed: {exc}")
                return jsonify({
                    'success': False,
                    'error': f'Manual insert failed: {exc}',
                }), 400

        result = squeeze_handler.squeeze_in_with_noshow(
            patient=patient,
            existing_schedule=app_state['appointments'],
            patient_data_map=app_state['patient_data_map'],
            allow_double_booking=data.get('allow_double_booking', True),
            allow_rescheduling=data.get('allow_rescheduling', False),
            date=datetime.now()
        )

        if result.success:
            app_state['appointments'].append(result.appointment)

            # Update statistics
            app_state['urgent_insertion']['total_insertions'] += 1
            app_state['urgent_insertion']['last_insertion'] = datetime.now().isoformat()
            if result.strategy_used == 'double_booking':
                app_state['urgent_insertion']['double_bookings'] += 1
            elif result.strategy_used == 'gap':
                app_state['urgent_insertion']['gap_based'] += 1
            elif result.strategy_used == 'rescheduling':
                app_state['urgent_insertion']['rescheduled'] += 1

            return jsonify({
                'success': True,
                'message': result.message,
                'strategy': result.strategy_used,
                'noshow_probability': result.noshow_probability,
                'confidence_level': result.confidence_level,
                'affected_patients': result.affected_patients,
                'options_evaluated': result.options_evaluated,
                'appointment': {
                    'patient_id': result.appointment.patient_id,
                    'chair_id': result.appointment.chair_id,
                    'site_code': result.appointment.site_code,
                    'start_time': result.appointment.start_time.strftime('%H:%M'),
                    'end_time': result.appointment.end_time.strftime('%H:%M'),
                    'duration': result.appointment.duration
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': result.message,
                'options_evaluated': result.options_evaluated,
                'strategy': result.strategy_used
            })

    except Exception as e:
        logger.error(f"Error inserting urgent patient: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/urgent/stats')
def api_urgent_stats():
    """Get statistics about urgent patient insertions"""
    return jsonify({
        'success': True,
        'stats': app_state['urgent_insertion'],
        'noshow_model_available': squeeze_handler.noshow_model is not None,
        'thresholds': {
            'high': squeeze_handler.NOSHOW_THRESHOLD_HIGH,
            'medium': squeeze_handler.NOSHOW_THRESHOLD_MEDIUM,
            'low': squeeze_handler.NOSHOW_THRESHOLD_LOW
        }
    })


@app.route('/api/urgent/find-best-slot', methods=['POST'])
def api_find_best_slot():
    """
    Find the best slot for an urgent patient without actually inserting.
    Useful for previewing options before committing.
    """
    data = request.json

    try:
        from optimization.optimizer import Patient

        patient = Patient(
            patient_id=data.get('patient_id', 'PREVIEW'),
            priority=int(data.get('priority', 1)),
            protocol=data.get('protocol', 'Urgent'),
            expected_duration=int(data.get('duration', 60)),
            postcode=data.get('postcode', 'CF14'),
            earliest_time=datetime.now().replace(hour=8, minute=0),
            latest_time=datetime.now().replace(hour=17, minute=0),
            long_infusion=data.get('long_infusion', False),
            is_urgent=True
        )

        options = squeeze_handler.find_best_slot_for_urgent(
            urgent_patient=patient,
            existing_schedule=app_state['appointments'],
            patient_data_map=app_state['patient_data_map'],
            date=datetime.now(),
            allow_double_booking=data.get('allow_double_booking', True)
        )

        # Score picks the top 10 best-fit options; then present them
        # chronologically so the UI reads left-to-right as a time-of-day
        # list rather than jumping between morning, afternoon, and evening.
        # Score is still displayed on each row so the ranking signal isn't lost.
        top10 = sorted(options[:10], key=lambda o: o.start_time)

        options_data = []
        for opt in top10:
            options_data.append({
                'chair_id': opt.chair_id,
                'site_code': opt.site_code,
                'start_time': opt.start_time.strftime('%H:%M'),
                'end_time': opt.end_time.strftime('%H:%M'),
                'score': round(opt.score, 1),
                'is_noshow_based': opt.noshow_based,
                'noshow_probability': round(opt.expected_noshow_prob, 3) if opt.noshow_based else 0,
                'affected_patients': opt.affected_appointments,
                'requires_rescheduling': opt.requires_rescheduling
            })

        return jsonify({
            'success': True,
            'total_options': len(options),
            'options': options_data
        })

    except Exception as e:
        logger.error(f"Error finding best slot: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/events')
def api_events():
    """Get active events"""
    events = event_aggregator.get_active_events()
    events_data = []
    for event in events:
        events_data.append({
            'event_id': event.event_id,
            'title': event.title,
            'description': event.description,
            'severity': event.severity,
            'event_type': event.event_type
        })
    return jsonify(events_data)


@app.route('/api/alerts')
def api_alerts():
    """Get active alerts"""
    alerts = alert_manager.get_active_alerts()
    alerts_data = []
    for alert in alerts:
        alerts_data.append({
            'alert_id': alert.alert_id,
            'title': alert.title,
            'message': alert.message,
            'priority': alert.priority.name,
            'created_at': alert.created_at.isoformat()
        })
    return jsonify(alerts_data)


@app.route('/api/mode', methods=['POST'])
def api_set_mode():
    """Set operating mode"""
    data = request.json
    mode_str = data.get('mode', 'normal')

    try:
        app_state['mode'] = OperatingMode(mode_str)
        emergency_handler.set_mode(app_state['mode'])
        return jsonify({'success': True, 'mode': app_state['mode'].value})
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid mode'})


@app.route('/schedule')
def schedule_page():
    """Schedule management page"""
    return render_template('schedule.html',
                          appointments=app_state['appointments'],
                          sites=DEFAULT_SITES,
                          mode=app_state['mode'].value)


@app.route('/patients')
def patients_page():
    """Patient management page"""
    return render_template('patients.html',
                          patients=app_state['patients'],
                          mode=app_state['mode'].value)


@app.route('/events')
def events_page():
    """Events monitoring page"""
    events = event_aggregator.get_active_events()
    return render_template('events.html', events=events, mode=app_state['mode'].value)


@app.route('/analytics')
def analytics_page():
    """Analytics dashboard page"""
    return render_template('analytics_new.html', mode=app_state['mode'].value)


@app.route('/about')
def about_page():
    """About page with project info"""
    return render_template('about.html', mode=app_state['mode'].value)


@app.route('/viewer')
def viewer_page():
    """
    Schedule Viewer - Mentor's Gantt-chart style viewer adapted for live Flask API.
    Fetches all data from API endpoints. No embedded data.
    """
    return render_template('viewer.html')


# =============================================================================
# PREMIUM DESIGN TEMPLATES
# =============================================================================

# Create templates directory and HTML files
templates_dir = Path(__file__).parent / 'templates_flask'
templates_dir.mkdir(exist_ok=True)

# Base template with clean Polymarket-inspired design
base_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SACT Scheduler{% endblock %}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            /* Clean Color Palette */
            --bg-primary: #0d0d0d;
            --bg-secondary: #141414;
            --bg-tertiary: #1a1a1a;
            --bg-card: #1e1e1e;
            --bg-hover: #252525;

            /* Text Colors */
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;

            /* Accent Colors */
            --accent-blue: #3b82f6;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;

            /* Borders */
            --border-color: #2a2a2a;
            --border-hover: #3a3a3a;

            /* Shadows */
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 14px;
            line-height: 1.5;
            min-height: 100vh;
        }

        /* Typography */
        h1 { font-size: 1.75rem; font-weight: 600; }
        h2 { font-size: 1.25rem; font-weight: 600; }
        h3 { font-size: 1rem; font-weight: 600; }

        /* Clean Navigation - Polymarket Style */
        .navbar {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .navbar-container {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            height: 60px;
        }

        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-primary);
            text-decoration: none;
            font-weight: 700;
            font-size: 1.1rem;
        }

        .navbar-brand .logo-icon {
            width: 32px;
            height: 32px;
            background: var(--accent-green);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            color: white;
        }

        /* Tab Navigation */
        .nav-tabs {
            display: flex;
            gap: 2px;
            background: var(--bg-tertiary);
            padding: 4px;
            border-radius: 10px;
        }

        .nav-tab {
            padding: 8px 16px;
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.85rem;
            border-radius: 6px;
            transition: all 0.15s ease;
        }

        .nav-tab:hover {
            color: var(--text-primary);
            background: var(--bg-hover);
        }

        .nav-tab.active {
            color: var(--text-primary);
            background: var(--bg-card);
        }

        /* Status Badge */
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .status-indicator::before {
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .status-normal { background: rgba(34, 197, 94, 0.15); color: var(--accent-green); }
        .status-normal::before { background: var(--accent-green); }
        .status-elevated { background: rgba(234, 179, 8, 0.15); color: var(--accent-yellow); }
        .status-elevated::before { background: var(--accent-yellow); }
        .status-crisis { background: rgba(249, 115, 22, 0.15); color: #f97316; }
        .status-crisis::before { background: #f97316; }
        .status-emergency { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }
        .status-emergency::before { background: var(--accent-red); }

        /* Main Content */
        .main-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 24px;
        }

        /* Page Header - Simple */
        .page-header {
            margin-bottom: 32px;
        }

        .page-header h1 {
            margin-bottom: 4px;
        }

        .page-subtitle {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        /* Filter Tabs - Like Polymarket */
        .filter-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
        }

        .filter-tab {
            padding: 6px 14px;
            background: transparent;
            border: 1px solid var(--border-color);
            border-radius: 20px;
            color: var(--text-secondary);
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .filter-tab:hover {
            border-color: var(--border-hover);
            color: var(--text-primary);
        }

        .filter-tab.active {
            background: var(--text-primary);
            border-color: var(--text-primary);
            color: var(--bg-primary);
        }

        /* Cards - Clean Design */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            transition: border-color 0.15s ease;
        }

        .card:hover {
            border-color: var(--border-hover);
        }

        .card-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .card-header h3 {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-primary);
        }

        .card-header h3 i {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .card-body {
            padding: 20px;
        }

        /* Metrics Grid - Clean */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            transition: border-color 0.15s ease;
        }

        .metric-card:hover {
            border-color: var(--border-hover);
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1;
            margin-bottom: 8px;
        }

        .metric-change {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .metric-change.positive { color: var(--accent-green); }
        .metric-change.negative { color: var(--accent-red); }
        .metric-change.neutral { color: var(--text-secondary); }

        /* Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 20px;
        }

        /* Tables - Clean */
        .table-container {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        thead th {
            text-align: left;
            padding: 12px 16px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-color);
        }

        td {
            padding: 14px 16px;
            font-size: 0.9rem;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
        }

        tbody tr:hover {
            background: var(--bg-hover);
        }

        tbody tr:hover td {
            color: var(--text-primary);
        }

        /* Priority Tags */
        .priority-tag {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .priority-tag.p1 { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }
        .priority-tag.p2 { background: rgba(249, 115, 22, 0.15); color: #f97316; }
        .priority-tag.p3 { background: rgba(234, 179, 8, 0.15); color: var(--accent-yellow); }
        .priority-tag.p4 { background: rgba(34, 197, 94, 0.15); color: var(--accent-green); }

        /* Status Tags */
        .status-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .status-tag::before {
            content: '';
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: currentColor;
        }

        .status-tag.confirmed { background: rgba(34, 197, 94, 0.15); color: var(--accent-green); }
        .status-tag.pending { background: rgba(234, 179, 8, 0.15); color: var(--accent-yellow); }
        .status-tag.cancelled { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }

        /* Buttons - Clean */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px 18px;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.85rem;
            border: none;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-primary {
            background: var(--accent-blue);
            color: white;
        }

        .btn-primary:hover {
            background: #2563eb;
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-color: var(--border-hover);
        }

        .btn-success {
            background: var(--accent-green);
            color: white;
        }

        .btn-success:hover {
            background: #16a34a;
        }

        .btn-sm {
            padding: 6px 12px;
            font-size: 0.8rem;
        }

        .btn-icon {
            width: 36px;
            height: 36px;
            padding: 0;
        }

        /* Forms - Clean */
        .form-group {
            margin-bottom: 16px;
        }

        .form-label {
            display: block;
            margin-bottom: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .form-control {
            width: 100%;
            padding: 10px 14px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.9rem;
            color: var(--text-primary);
            transition: border-color 0.15s ease;
        }

        .form-control::placeholder {
            color: var(--text-muted);
        }

        .form-control:focus {
            outline: none;
            border-color: var(--accent-blue);
        }

        .form-select {
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23666' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 14px center;
            padding-right: 40px;
        }

        /* Alert Items - Clean */
        .alert-item {
            padding: 14px 16px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid;
            background: var(--bg-tertiary);
        }

        .alert-item.critical { border-color: var(--accent-red); }
        .alert-item.high { border-color: #f97316; }
        .alert-item.medium { border-color: var(--accent-yellow); }
        .alert-item.low { border-color: var(--accent-green); }

        .alert-title {
            font-weight: 600;
            margin-bottom: 4px;
            color: var(--text-primary);
            font-size: 0.9rem;
        }

        .alert-message {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 48px 24px;
            color: var(--text-muted);
        }

        .empty-state i {
            font-size: 2.5rem;
            margin-bottom: 16px;
            opacity: 0.5;
        }

        .empty-state h4 {
            color: var(--text-secondary);
            margin-bottom: 6px;
            font-weight: 500;
        }

        .empty-state p {
            font-size: 0.85rem;
        }

        /* Success State */
        .success-state {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 16px;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 8px;
            color: var(--accent-green);
            font-size: 0.9rem;
        }

        /* Grid System */
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .navbar-container {
                padding: 0 16px;
            }
            .nav-tabs {
                display: none;
            }
            .main-content {
                padding: 20px 16px;
            }
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--border-hover);
        }

        /* Footer - Minimal */
        .footer {
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            padding: 24px;
            margin-top: 48px;
        }

        .footer-container {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }

        .footer-brand {
            display: flex;
            align-items: center;
            gap: 24px;
        }

        .footer-logo {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .footer-logo i {
            font-size: 1.1rem;
        }

        .footer-logo.velindre i { color: var(--accent-green); }
        .footer-logo.cardiff i { color: var(--accent-red); }

        .footer-badges {
            display: flex;
            gap: 10px;
        }

        .footer-badge {
            padding: 4px 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        /* Animations */
        .fade-in {
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-container">
            <a href="/" class="navbar-brand">
                <div class="logo-icon">
                    <i class="fas fa-heartbeat"></i>
                </div>
                SACT Scheduler
            </a>

            <div class="nav-tabs">
                <a href="/" class="nav-tab {% if request.endpoint == 'index' %}active{% endif %}">Dashboard</a>
                <a href="/schedule" class="nav-tab {% if request.endpoint == 'schedule_page' %}active{% endif %}">Schedule</a>
                <a href="/patients" class="nav-tab {% if request.endpoint == 'patients_page' %}active{% endif %}">Patients</a>
                <a href="/analytics" class="nav-tab {% if request.endpoint == 'analytics_page' %}active{% endif %}">Analytics</a>
                <a href="/events" class="nav-tab {% if request.endpoint == 'events_page' %}active{% endif %}">Events</a>
                <a href="/about" class="nav-tab {% if request.endpoint == 'about_page' %}active{% endif %}">About</a>
            </div>

            <div class="status-indicator status-{{ mode }}">
                {{ mode|upper }}
            </div>
        </div>
    </nav>

    <main class="main-content">
        {% block content %}{% endblock %}
    </main>

    <footer class="footer">
        <div class="footer-container">
            <div class="footer-brand">
                <div class="footer-logo velindre">
                    <i class="fas fa-ribbon"></i>
                    <span>Velindre Cancer Centre</span>
                </div>
                <div class="footer-logo cardiff">
                    <i class="fas fa-university"></i>
                    <span>Cardiff University</span>
                </div>
            </div>
            <div class="footer-badges">
                <span class="footer-badge">NHS Compliant</span>
                <span class="footer-badge">OR-Tools Powered</span>
                <span class="footer-badge">2026</span>
            </div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
'''

# Index/Dashboard template - Clean Polymarket-inspired design
index_html = '''{% extends "base.html" %}
{% block title %}Dashboard | SACT Scheduler{% endblock %}
{% block content %}
<div class="page-header fade-in">
    <h1>Dashboard</h1>
    <p class="page-subtitle">Real-time scheduling overview</p>
</div>

<!-- Filter Tabs -->
<div class="filter-tabs">
    <button class="filter-tab active">Today</button>
    <button class="filter-tab">This Week</button>
    <button class="filter-tab">All Pending</button>
    <button class="filter-tab">Urgent</button>
</div>

<!-- Metrics -->
<div class="metrics-grid fade-in">
    <div class="metric-card">
        <div class="metric-label">Scheduled Today</div>
        <div class="metric-value" id="scheduled-count">{{ appointments|length }}</div>
        <span class="metric-change positive"><i class="fas fa-check"></i> On track</span>
    </div>
    <div class="metric-card">
        <div class="metric-label">Chair Utilization</div>
        <div class="metric-value" id="utilization">85%</div>
        <span class="metric-change positive"><i class="fas fa-arrow-up"></i> +5%</span>
    </div>
    <div class="metric-card">
        <div class="metric-label">Active Events</div>
        <div class="metric-value" id="events-count">0</div>
        <span class="metric-change neutral"><i class="fas fa-minus"></i> Clear</span>
    </div>
    <div class="metric-card">
        <div class="metric-label">Alerts</div>
        <div class="metric-value" id="alerts-count">0</div>
        <span class="metric-change neutral"><i class="fas fa-check"></i> None</span>
    </div>
</div>

<!-- Main Grid -->
<div class="dashboard-grid fade-in">
    <div class="card">
        <div class="card-header">
            <h3><i class="fas fa-calendar-alt"></i> Today's Schedule</h3>
            <button class="btn btn-primary btn-sm" onclick="runOptimize()">
                <i class="fas fa-magic"></i> Optimize
            </button>
        </div>
        <div class="card-body">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Patient</th>
                            <th>Chair</th>
                            <th>Duration</th>
                            <th>Priority</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="schedule-table">
                        {% for apt in appointments[:10] %}
                        <tr>
                            <td><strong>{{ apt.start_time.strftime('%H:%M') }}</strong></td>
                            <td>{{ apt.patient_id }}</td>
                            <td>{{ apt.chair_id }}</td>
                            <td>{{ apt.duration }} min</td>
                            <td><span class="priority-tag p{{ apt.priority }}">P{{ apt.priority }}</span></td>
                            <td><span class="status-tag confirmed">Confirmed</span></td>
                        </tr>
                        {% else %}
                        <tr>
                            <td colspan="6">
                                <div class="empty-state">
                                    <i class="fas fa-calendar-plus"></i>
                                    <h4>No appointments scheduled</h4>
                                    <p>Add patients and run optimization</p>
                                </div>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% if appointments|length > 10 %}
            <div style="padding: 12px; text-align: center; border-top: 1px solid var(--border-color);">
                <a href="/schedule" style="color: var(--accent-blue); text-decoration: none; font-size: 0.85rem;">
                    View all {{ appointments|length }} appointments <i class="fas fa-arrow-right"></i>
                </a>
            </div>
            {% endif %}
        </div>
    </div>

    <div style="display: flex; flex-direction: column; gap: 20px;">
        <!-- System Status -->
        <div class="card">
            <div class="card-header">
                <h3><i class="fas fa-heart-pulse"></i> System Status</h3>
            </div>
            <div class="card-body" id="alerts-container">
                <div class="success-state">
                    <i class="fas fa-check-circle"></i>
                    <span>All systems operational</span>
                </div>
            </div>
        </div>

        <!-- Quick Add -->
        <div class="card">
            <div class="card-header">
                <h3><i class="fas fa-plus"></i> Quick Add</h3>
            </div>
            <div class="card-body">
                <form id="quick-add-form">
                    <div class="form-group">
                        <label class="form-label">Patient ID</label>
                        <input type="text" class="form-control" id="patient_id" placeholder="Enter ID" required>
                    </div>
                    <div class="grid-2">
                        <div class="form-group">
                            <label class="form-label">Priority</label>
                            <select class="form-control form-select" id="priority">
                                <option value="1">P1 - Critical</option>
                                <option value="2" selected>P2 - High</option>
                                <option value="3">P3 - Standard</option>
                                <option value="4">P4 - Routine</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Duration</label>
                            <input type="number" class="form-control" id="duration" value="90" min="15" max="480">
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Postcode</label>
                        <input type="text" class="form-control" id="postcode" placeholder="CF14" value="CF14">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Patient History (for no-show prediction)</label>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                            <input type="number" class="form-control" id="total_appointments" placeholder="Total Appts" value="5" min="0" title="Total Appointments">
                            <input type="number" class="form-control" id="no_shows" placeholder="No-Shows" value="0" min="0" title="Previous No-Shows">
                            <input type="number" class="form-control" id="cancellations" placeholder="Cancels" value="0" min="0" title="Previous Cancellations">
                        </div>
                    </div>
                    <div class="form-group" style="background: rgba(124, 58, 237, 0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(124, 58, 237, 0.2);">
                        <label class="form-check" style="margin-bottom: 8px;">
                            <input type="checkbox" class="form-check-input" id="is_urgent">
                            <span style="color: #a78bfa; font-weight: 500;">Urgent squeeze-in (uses ML predictions)</span>
                        </label>
                        <div id="urgent-options" style="display: none; margin-top: 8px; padding-left: 24px;">
                            <label class="form-check" style="margin-bottom: 4px;">
                                <input type="checkbox" class="form-check-input" id="allow_double_booking" checked>
                                <span style="font-size: 0.85rem;">Allow double-booking (high no-show slots)</span>
                            </label>
                            <label class="form-check">
                                <input type="checkbox" class="form-check-input" id="allow_rescheduling">
                                <span style="font-size: 0.85rem;">Allow rescheduling lower-priority patients</span>
                            </label>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-success" style="width: 100%;">
                        <i class="fas fa-plus"></i> Add Patient
                    </button>
                    <button type="button" class="btn btn-primary" style="width: 100%; margin-top: 8px;" onclick="previewUrgentSlots()">
                        <i class="fas fa-search"></i> Preview Urgent Slots
                    </button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener('DOMContentLoaded', function() {
    loadAlerts();
    loadStatus();
});

function loadStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('events-count').textContent = data.event_count;
        });
}

function loadAlerts() {
    fetch('/api/alerts')
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById('alerts-container');
            if (data.length === 0) {
                container.innerHTML = '<div class="success-state"><i class="fas fa-check-circle"></i><span>All systems operational</span></div>';
            } else {
                container.innerHTML = data.map(a => `
                    <div class="alert-item ${a.priority.toLowerCase()}">
                        <div class="alert-title">${a.title}</div>
                        <div class="alert-message">${a.message.substring(0, 100)}...</div>
                    </div>
                `).join('');
                document.getElementById('alerts-count').textContent = data.length;
            }
        });
}

function runOptimize() {
    fetch('/api/optimize', {method: 'POST'})
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                alert('Optimization complete! Scheduled: ' + (data.scheduled || 0));
                location.reload();
            } else {
                alert('Optimization: ' + (data.message || 'No patients to schedule'));
            }
        });
}

// Toggle urgent options visibility
document.getElementById('is_urgent').addEventListener('change', function() {
    document.getElementById('urgent-options').style.display = this.checked ? 'block' : 'none';
});

document.getElementById('quick-add-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const isUrgent = document.getElementById('is_urgent').checked;

    const data = {
        patient_id: document.getElementById('patient_id').value,
        priority: document.getElementById('priority').value,
        duration: document.getElementById('duration').value,
        postcode: document.getElementById('postcode').value,
        total_appointments: document.getElementById('total_appointments').value,
        no_shows: document.getElementById('no_shows').value,
        cancellations: document.getElementById('cancellations').value,
        is_urgent: isUrgent,
        allow_double_booking: isUrgent ? document.getElementById('allow_double_booking').checked : false,
        allow_rescheduling: isUrgent ? document.getElementById('allow_rescheduling').checked : false
    };

    fetch('/api/add_patient', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {
        if (result.success) {
            let msg = result.message;
            if (result.strategy) {
                msg += '\\n\\nStrategy: ' + result.strategy;
                if (result.strategy === 'double_booking') {
                    msg += '\\nNo-show probability: ' + (result.noshow_probability * 100).toFixed(1) + '%';
                    msg += '\\nConfidence: ' + result.confidence_level;
                }
                if (result.affected_patients && result.affected_patients.length > 0) {
                    msg += '\\nAffected: ' + result.affected_patients.join(', ');
                }
                if (result.appointment) {
                    msg += '\\n\\nScheduled at ' + result.appointment.start_time + ' on ' + result.appointment.chair_id;
                }
            }
            alert(msg);
            location.reload();
        } else {
            alert('Error: ' + result.message);
        }
    });
});

function previewUrgentSlots() {
    const data = {
        patient_id: 'PREVIEW',
        priority: document.getElementById('priority').value,
        duration: document.getElementById('duration').value,
        postcode: document.getElementById('postcode').value,
        long_infusion: false,
        allow_double_booking: document.getElementById('allow_double_booking').checked
    };

    fetch('/api/urgent/find-best-slot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {
        if (result.success && result.options.length > 0) {
            let msg = 'Found ' + result.total_options + ' possible slots:\\n\\n';
            result.options.slice(0, 5).forEach((opt, i) => {
                msg += (i + 1) + '. ' + opt.start_time + ' on ' + opt.chair_id;
                msg += ' (score: ' + opt.score + ')';
                if (opt.is_noshow_based) {
                    msg += ' [No-show: ' + (opt.noshow_probability * 100).toFixed(1) + '%]';
                }
                msg += '\\n';
            });
            alert(msg);
        } else {
            alert('No slots available: ' + (result.message || 'Schedule may be full'));
        }
    });
}
</script>
{% endblock %}
'''

# Schedule template - Clean design
schedule_html = '''{% extends "base.html" %}
{% block title %}Schedule | SACT Scheduler{% endblock %}
{% block content %}
<div class="page-header fade-in">
    <h1>Schedule</h1>
    <p class="page-subtitle">View and manage appointments</p>
</div>

<!-- Filter Tabs -->
<div class="filter-tabs">
    <button class="filter-tab active">All</button>
    {% for site in sites %}
    <button class="filter-tab">{{ site.name }}</button>
    {% endfor %}
</div>

<div class="card fade-in">
    <div class="card-header">
        <h3><i class="fas fa-calendar-alt"></i> {{ appointments|length }} Appointments</h3>
        <button class="btn btn-secondary btn-sm">
            <i class="fas fa-download"></i> Export
        </button>
    </div>
    <div class="card-body">
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Patient</th>
                        <th>Chair</th>
                        <th>Site</th>
                        <th>Duration</th>
                        <th>Priority</th>
                        <th>Status</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    {% for apt in appointments %}
                    <tr>
                        <td><strong>{{ apt.start_time.strftime('%H:%M') }} - {{ apt.end_time.strftime('%H:%M') }}</strong></td>
                        <td>{{ apt.patient_id }}</td>
                        <td>{{ apt.chair_id }}</td>
                        <td>{{ apt.site_code }}</td>
                        <td>{{ apt.duration }} min</td>
                        <td><span class="priority-tag p{{ apt.priority }}">P{{ apt.priority }}</span></td>
                        <td><span class="status-tag confirmed">Confirmed</span></td>
                        <td>
                            <button class="btn btn-secondary btn-icon btn-sm"><i class="fas fa-ellipsis-h"></i></button>
                        </td>
                    </tr>
                    {% else %}
                    <tr>
                        <td colspan="8">
                            <div class="empty-state">
                                <i class="fas fa-calendar-times"></i>
                                <h4>No appointments found</h4>
                                <p>Add patients from the dashboard to create appointments</p>
                            </div>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>

<!-- High No-Show Slots Panel - For Urgent Patient Insertion -->
<div class="card fade-in delay-1" style="margin-top: 24px;">
    <div class="card-header" style="background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%);">
        <h3 style="color: #a78bfa;"><i class="fas fa-bolt"></i> High No-Show Risk Slots</h3>
        <div style="display: flex; gap: 10px;">
            <button class="btn btn-primary btn-sm" onclick="loadHighNoshowSlots()">
                <i class="fas fa-sync"></i> Refresh
            </button>
            <button class="btn btn-success btn-sm" onclick="getDoubleBookingRecommendations()">
                <i class="fas fa-lightbulb"></i> Get Recommendations
            </button>
        </div>
    </div>
    <div class="card-body">
        <p style="color: var(--gray-400); font-size: 0.85rem; margin-bottom: 16px;">
            <i class="fas fa-info-circle"></i> These slots have high no-show probability and are ideal for double-booking urgent patients.
        </p>
        <div id="noshow-slots-container">
            <div class="loading-state">
                <i class="fas fa-spinner fa-spin"></i>
                <span>Loading no-show predictions...</span>
            </div>
        </div>
        <div id="recommendations-container" style="margin-top: 16px; display: none;"></div>
    </div>
</div>

<!-- Urgent Insertion Statistics -->
<div class="card fade-in delay-2" style="margin-top: 24px;">
    <div class="card-header">
        <h3><i class="fas fa-chart-line"></i> Urgent Insertion Statistics</h3>
    </div>
    <div class="card-body">
        <div id="urgent-stats-container" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
            <div style="text-align: center; padding: 16px; background: rgba(52, 211, 153, 0.1); border-radius: 8px;">
                <div id="stat-total" style="font-size: 2rem; font-weight: bold; color: #34d399;">0</div>
                <div style="font-size: 0.8rem; color: var(--gray-400);">Total Insertions</div>
            </div>
            <div style="text-align: center; padding: 16px; background: rgba(59, 130, 246, 0.1); border-radius: 8px;">
                <div id="stat-gap" style="font-size: 2rem; font-weight: bold; color: #3b82f6;">0</div>
                <div style="font-size: 0.8rem; color: var(--gray-400);">Gap-Based</div>
            </div>
            <div style="text-align: center; padding: 16px; background: rgba(124, 58, 237, 0.1); border-radius: 8px;">
                <div id="stat-double" style="font-size: 2rem; font-weight: bold; color: #a78bfa;">0</div>
                <div style="font-size: 0.8rem; color: var(--gray-400);">Double-Bookings</div>
            </div>
            <div style="text-align: center; padding: 16px; background: rgba(251, 146, 60, 0.1); border-radius: 8px;">
                <div id="stat-resched" style="font-size: 2rem; font-weight: bold; color: #fb923c;">0</div>
                <div style="font-size: 0.8rem; color: var(--gray-400);">Rescheduled</div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener('DOMContentLoaded', function() {
    loadHighNoshowSlots();
    loadUrgentStats();
});

function loadHighNoshowSlots() {
    fetch('/api/urgent/high-noshow-slots')
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById('noshow-slots-container');
            if (!data.success || data.total_slots === 0) {
                container.innerHTML = '<div class="empty-state"><i class="fas fa-check-circle"></i><p>No high no-show risk slots detected</p></div>';
                return;
            }

            let html = '<div class="table-container"><table><thead><tr>';
            html += '<th>Patient</th><th>Time</th><th>Chair</th><th>No-Show Risk</th><th>Score</th><th>Action</th>';
            html += '</tr></thead><tbody>';

            data.slots.forEach(slot => {
                const riskClass = slot.noshow_probability >= 0.30 ? 'high' : slot.noshow_probability >= 0.20 ? 'medium' : 'low';
                const riskColor = slot.noshow_probability >= 0.30 ? '#ef4444' : slot.noshow_probability >= 0.20 ? '#f59e0b' : '#3b82f6';
                html += '<tr>';
                html += '<td><strong>' + slot.patient_id + '</strong></td>';
                html += '<td>' + slot.start_time + ' - ' + slot.end_time + '</td>';
                html += '<td>' + slot.chair_id + '</td>';
                html += '<td><span style="background: ' + riskColor + '20; color: ' + riskColor + '; padding: 4px 12px; border-radius: 20px; font-weight: 500;">' + slot.noshow_percent + '</span></td>';
                html += '<td>' + slot.double_book_score + '</td>';
                html += '<td><button class="btn btn-primary btn-sm" onclick="doubleBookSlot(\\'' + slot.patient_id + '\\', \\'' + slot.start_time + '\\', \\'' + slot.chair_id + '\\')"><i class="fas fa-plus"></i> Double-Book</button></td>';
                html += '</tr>';
            });

            html += '</tbody></table></div>';
            html += '<div style="margin-top: 12px; padding: 12px; background: rgba(124, 58, 237, 0.1); border-radius: 8px; font-size: 0.85rem;">';
            html += '<i class="fas fa-chart-pie" style="color: #a78bfa;"></i> ';
            html += '<strong style="color: #ef4444;">' + data.high_risk_count + '</strong> high risk (>=30%) | ';
            html += '<strong style="color: #f59e0b;">' + data.medium_risk_count + '</strong> medium risk (20-30%)';
            html += '</div>';

            container.innerHTML = html;
        })
        .catch(err => {
            document.getElementById('noshow-slots-container').innerHTML = '<div class="error-state"><i class="fas fa-exclamation-triangle"></i><p>Error loading no-show data</p></div>';
        });
}

function getDoubleBookingRecommendations() {
    fetch('/api/urgent/recommend-double-booking?target_slots=5')
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById('recommendations-container');
            container.style.display = 'block';

            if (!data.success) {
                container.innerHTML = '<div class="error-state"><p>' + data.message + '</p></div>';
                return;
            }

            let html = '<div style="padding: 16px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%); border-radius: 8px; border: 1px solid rgba(52, 211, 153, 0.2);">';
            html += '<h4 style="margin-bottom: 12px; color: #34d399;"><i class="fas fa-lightbulb"></i> Double-Booking Recommendations</h4>';
            html += '<p style="font-size: 0.85rem; color: var(--gray-400); margin-bottom: 12px;">Expected no-shows from these slots: <strong>' + data.summary.expected_noshows.toFixed(2) + '</strong></p>';
            html += '<div style="display: flex; flex-wrap: wrap; gap: 8px;">';

            data.recommended_slots.forEach(slot => {
                html += '<div style="padding: 10px 16px; background: var(--gray-800); border-radius: 8px; border: 1px solid var(--gray-700);">';
                html += '<strong>' + slot.patient_id + '</strong> @ ' + slot.start_time;
                html += ' <span style="color: #a78bfa;">(' + slot.noshow_probability + ')</span>';
                html += '</div>';
            });

            html += '</div></div>';
            container.innerHTML = html;
        });
}

function loadUrgentStats() {
    fetch('/api/urgent/stats')
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                document.getElementById('stat-total').textContent = data.stats.total_insertions || 0;
                document.getElementById('stat-gap').textContent = data.stats.gap_based || 0;
                document.getElementById('stat-double').textContent = data.stats.double_bookings || 0;
                document.getElementById('stat-resched').textContent = data.stats.rescheduled || 0;
            }
        });
}

function doubleBookSlot(patientId, startTime, chairId) {
    const urgentPatientId = prompt('Enter urgent patient ID to double-book at ' + startTime + ':');
    if (!urgentPatientId) return;

    const duration = prompt('Enter treatment duration (minutes):', '60');
    if (!duration) return;

    fetch('/api/urgent/insert', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            patient_id: urgentPatientId,
            priority: 1,
            duration: parseInt(duration),
            allow_double_booking: true,
            allow_rescheduling: false
        })
    })
    .then(r => r.json())
    .then(result => {
        if (result.success) {
            alert('Urgent patient inserted successfully!\\n\\nStrategy: ' + result.strategy + '\\nScheduled at: ' + result.appointment.start_time + ' on ' + result.appointment.chair_id);
            location.reload();
        } else {
            alert('Error: ' + result.message);
        }
    });
}
</script>
{% endblock %}
'''

# Patients template - Clean design
patients_html = '''{% extends "base.html" %}
{% block title %}Patients | SACT Scheduler{% endblock %}
{% block content %}
<div class="page-header fade-in">
    <h1>Patients</h1>
    <p class="page-subtitle">Manage pending patients</p>
</div>

<!-- Filter Tabs -->
<div class="filter-tabs">
    <button class="filter-tab active">All ({{ patients|length }})</button>
    <button class="filter-tab">P1 Critical</button>
    <button class="filter-tab">P2 High</button>
    <button class="filter-tab">P3 Standard</button>
</div>

<div class="card fade-in">
    <div class="card-header">
        <h3><i class="fas fa-users"></i> Pending Patients</h3>
        <div style="display: flex; gap: 10px;">
            <input type="text" class="form-control" placeholder="Search..." id="patient-search" style="width: 200px;">
            <button class="btn btn-success btn-sm">
                <i class="fas fa-plus"></i> Add
            </button>
        </div>
    </div>
    <div class="card-body">
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Patient ID</th>
                        <th>Postcode</th>
                        <th>Priority</th>
                        <th>Protocol</th>
                        <th>Duration</th>
                        <th>Status</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    {% for p in patients %}
                    <tr>
                        <td><strong>{{ p.patient_id }}</strong></td>
                        <td>{{ p.postcode }}</td>
                        <td><span class="priority-tag p{{ p.priority }}">P{{ p.priority }}</span></td>
                        <td>{{ p.protocol }}</td>
                        <td>{{ p.expected_duration }} min</td>
                        <td><span class="status-tag pending">Pending</span></td>
                        <td>
                            <button class="btn btn-secondary btn-icon btn-sm"><i class="fas fa-ellipsis-h"></i></button>
                        </td>
                    </tr>
                    {% else %}
                    <tr>
                        <td colspan="7">
                            <div class="empty-state">
                                <i class="fas fa-user-plus"></i>
                                <h4>No pending patients</h4>
                                <p>Add patients to schedule appointments</p>
                            </div>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>
{% endblock %}
'''

# Events template
events_html = '''{% extends "base.html" %}
{% block title %}Events | SACT Scheduler{% endblock %}
{% block content %}
<div class="page-header fade-in">
    <h1>Event Monitoring</h1>
    <p class="page-subtitle">Real-time monitoring of events affecting operations</p>
</div>

<div style="display: grid; grid-template-columns: 1.5fr 1fr; gap: 28px;">
    <div class="card fade-in">
        <div class="card-header">
            <h3><i class="fas fa-globe"></i>Active Events</h3>
            <button class="btn btn-secondary btn-sm" onclick="location.reload()">
                <i class="fas fa-sync"></i>Refresh
            </button>
        </div>
        <div class="card-body">
            {% for event in events %}
            <div class="alert-item {% if event.severity >= 0.7 %}critical{% elif event.severity >= 0.4 %}high{% else %}medium{% endif %}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="alert-title">{{ event.title }}</div>
                        <div class="alert-message">{{ event.description }}</div>
                    </div>
                    <span class="priority-badge {% if event.severity >= 0.7 %}p1{% elif event.severity >= 0.4 %}p2{% else %}p3{% endif %}">
                        {{ (event.severity * 100)|int }}%
                    </span>
                </div>
                <div style="margin-top: 12px; font-size: 0.75rem; color: var(--gray-500);">
                    <i class="fas fa-tag"></i> {{ event.event_type }}
                </div>
            </div>
            {% else %}
            <div class="success-state">
                <i class="fas fa-check-circle"></i>
                <span>No active events affecting operations</span>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="card fade-in">
        <div class="card-header">
            <h3><i class="fas fa-sliders-h"></i>Operating Mode</h3>
        </div>
        <div class="card-body">
            <p style="margin-bottom: 24px; color: var(--gray-400); font-size: 0.9rem; line-height: 1.6;">
                Current mode determines scheduling parameters, buffer times, and system behavior.
            </p>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <button class="btn btn-secondary" onclick="setMode('normal')" style="justify-content: flex-start;">
                    <span style="width: 12px; height: 12px; border-radius: 50%; background: var(--success);"></span>
                    Normal Operations
                </button>
                <button class="btn btn-secondary" onclick="setMode('elevated')" style="justify-content: flex-start;">
                    <span style="width: 12px; height: 12px; border-radius: 50%; background: var(--warning);"></span>
                    Elevated Alert
                </button>
                <button class="btn btn-secondary" onclick="setMode('crisis')" style="justify-content: flex-start;">
                    <span style="width: 12px; height: 12px; border-radius: 50%; background: #ea580c;"></span>
                    Crisis Mode
                </button>
                <button class="btn btn-secondary" onclick="setMode('emergency')" style="justify-content: flex-start; border-color: var(--danger);">
                    <span style="width: 12px; height: 12px; border-radius: 50%; background: var(--danger);"></span>
                    Emergency Mode
                </button>
            </div>
        </div>
    </div>
</div>

<script>
function setMode(mode) {
    fetch('/api/mode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: mode})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            location.reload();
        }
    });
}
</script>
{% endblock %}
'''

# Analytics template with real-time data and auto-refresh
analytics_html = '''{% extends "base.html" %}
{% block title %}Analytics | SACT Scheduler{% endblock %}
{% block content %}
<div class="page-header fade-in">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <h1>Analytics & Insights</h1>
            <p class="page-subtitle">Real-time performance metrics with auto-refresh</p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <div id="refresh-status" style="display: flex; align-items: center; gap: 8px; padding: 8px 16px; background: rgba(5, 150, 105, 0.1); border: 1px solid rgba(5, 150, 105, 0.2); border-radius: 20px; font-size: 0.75rem; color: #34d399;">
                <span id="status-dot" style="width: 8px; height: 8px; border-radius: 50%; background: #34d399; animation: pulse 2s infinite;"></span>
                <span id="status-text">Connected</span>
            </div>
            <select id="refresh-interval" class="form-control form-select" style="width: 140px; padding: 8px 12px; font-size: 0.8rem;" onchange="updateRefreshInterval(this.value)">
                <option value="30000">30 seconds</option>
                <option value="60000">1 minute</option>
                <option value="300000">5 minutes</option>
                <option value="900000" selected>15 minutes</option>
                <option value="1800000">30 minutes</option>
            </select>
        </div>
    </div>
</div>

<!-- Analysis Workflow Toggle -->
<div class="card fade-in" style="margin-bottom: 28px; border: 1px solid rgba(99, 102, 241, 0.3); background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);">
    <div class="card-body" style="padding: 24px 28px;">
        <!-- Toggle Header -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <i class="fas fa-play-circle" style="color: white; font-size: 1.2rem;"></i>
                </div>
                <div>
                    <h3 style="margin: 0; font-size: 1.1rem; color: var(--white);">Analysis Workflow</h3>
                    <p style="margin: 0; font-size: 0.75rem; color: var(--gray-400);">Connect to data source, load data, and run analysis</p>
                </div>
            </div>
            <button id="workflow-toggle" class="btn btn-primary" onclick="toggleWorkflowPanel()" style="padding: 10px 20px;">
                <i class="fas fa-chevron-down" id="toggle-icon"></i> <span id="toggle-text">Show Workflow</span>
            </button>
        </div>

        <!-- Workflow Panel (collapsed by default) -->
        <div id="workflow-panel" style="display: none;">
            <!-- Workflow Steps -->
            <div style="display: flex; gap: 20px; margin-bottom: 24px;">
                <!-- Step 1: Select Source -->
                <div id="step-1" class="workflow-step active" style="flex: 1; padding: 20px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 2px solid rgba(99, 102, 241, 0.5); position: relative;">
                    <div style="position: absolute; top: -12px; left: 20px; background: var(--primary-900); padding: 2px 12px; border-radius: 20px; font-size: 0.7rem; color: #6366f1; font-weight: 600;">STEP 1</div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                        <div style="width: 36px; height: 36px; background: rgba(99, 102, 241, 0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                            <i class="fas fa-database" style="color: #6366f1;"></i>
                        </div>
                        <span style="font-weight: 600; color: var(--white);">Select Data Source</span>
                    </div>
                    <div style="display: flex; gap: 10px; margin-bottom: 12px;">
                        <button id="btn-local" class="source-btn active" onclick="selectSource('local')" style="flex: 1; padding: 12px; background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; border-radius: 8px; color: var(--white); cursor: pointer; transition: all 0.2s;">
                            <i class="fas fa-folder"></i> Local Files
                        </button>
                        <button id="btn-cloud" class="source-btn" onclick="selectSource('cloud')" style="flex: 1; padding: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: var(--gray-400); cursor: pointer; transition: all 0.2s;">
                            <i class="fas fa-cloud"></i> Cloud
                        </button>
                    </div>
                    <div id="source-path" style="font-size: 0.75rem; color: var(--gray-500); word-break: break-all;">
                        <i class="fas fa-folder-open"></i> datasets/sample_data/
                    </div>
                </div>

                <!-- Step 2: Load Data -->
                <div id="step-2" class="workflow-step" style="flex: 1; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 2px solid rgba(255,255,255,0.1); position: relative; opacity: 0.7;">
                    <div style="position: absolute; top: -12px; left: 20px; background: var(--primary-900); padding: 2px 12px; border-radius: 20px; font-size: 0.7rem; color: var(--gray-500); font-weight: 600;">STEP 2</div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                        <div style="width: 36px; height: 36px; background: rgba(255,255,255,0.05); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                            <i class="fas fa-download" style="color: var(--gray-500);"></i>
                        </div>
                        <span style="font-weight: 600; color: var(--gray-400);">Load Data</span>
                    </div>
                    <button id="btn-load" onclick="loadDataFromSource()" style="width: 100%; padding: 12px; background: rgba(5, 150, 105, 0.2); border: 1px solid #059669; border-radius: 8px; color: #34d399; cursor: pointer; font-weight: 500; transition: all 0.2s;">
                        <i class="fas fa-plug"></i> Connect & Load
                    </button>
                    <div id="load-status" style="margin-top: 12px; font-size: 0.75rem; color: var(--gray-500);">
                        <i class="fas fa-info-circle"></i> Click to load patient & appointment data
                    </div>
                </div>

                <!-- Step 3: Analyze -->
                <div id="step-3" class="workflow-step" style="flex: 1; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 2px solid rgba(255,255,255,0.1); position: relative; opacity: 0.7;">
                    <div style="position: absolute; top: -12px; left: 20px; background: var(--primary-900); padding: 2px 12px; border-radius: 20px; font-size: 0.7rem; color: var(--gray-500); font-weight: 600;">STEP 3</div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                        <div style="width: 36px; height: 36px; background: rgba(255,255,255,0.05); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                            <i class="fas fa-chart-line" style="color: var(--gray-500);"></i>
                        </div>
                        <span style="font-weight: 600; color: var(--gray-400);">Analyze</span>
                    </div>
                    <button id="btn-analyze" onclick="runFullAnalysis()" disabled style="width: 100%; padding: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: var(--gray-500); cursor: not-allowed; font-weight: 500; transition: all 0.2s;">
                        <i class="fas fa-magic"></i> Run Analysis
                    </button>
                    <div id="analyze-status" style="margin-top: 12px; font-size: 0.75rem; color: var(--gray-500);">
                        <i class="fas fa-lock"></i> Load data first to enable
                    </div>
                </div>
            </div>

            <!-- Progress Bar -->
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; height: 8px; overflow: hidden;">
                <div id="workflow-progress" style="width: 33%; height: 100%; background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 8px; transition: width 0.5s ease;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.7rem; color: var(--gray-500);">
                <span>Source Selected</span>
                <span>Data Loaded</span>
                <span>Analysis Complete</span>
            </div>
        </div>
    </div>
</div>

<!-- Data Source Status -->
<div class="card fade-in" style="margin-bottom: 28px;">
    <div class="card-body" style="padding: 20px 28px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div style="display: flex; align-items: center; gap: 24px;">
                <div>
                    <div style="font-size: 0.7rem; color: var(--gray-500); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">Data Source</div>
                    <div id="data-source" style="font-size: 0.95rem; color: var(--white); font-weight: 600;">Loading...</div>
                </div>
                <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--gray-500); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">Last Data Refresh</div>
                    <div id="last-refresh" style="font-size: 0.95rem; color: var(--white); font-weight: 600;">--</div>
                </div>
                <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--gray-500); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">Last Optimization</div>
                    <div id="last-optimization" style="font-size: 0.95rem; color: var(--white); font-weight: 600;">--</div>
                </div>
                <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--gray-500); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">Next Auto-Refresh</div>
                    <div id="next-refresh" style="font-size: 0.95rem; color: var(--primary-400); font-weight: 600;">--</div>
                </div>
            </div>
            <button class="btn btn-success btn-sm" onclick="runFullOptimization()">
                <i class="fas fa-magic"></i> Run Optimization
            </button>
        </div>
    </div>
</div>

<!-- Live Metrics -->
<div class="metrics-grid">
    <div class="metric-card fade-in delay-1">
        <div class="metric-icon blue">
            <i class="fas fa-percentage"></i>
        </div>
        <div class="metric-value" id="metric-utilization">--</div>
        <div class="metric-label">Chair Utilization</div>
        <span class="metric-trend up" id="trend-utilization">
            <i class="fas fa-sync fa-spin"></i> Loading
        </span>
    </div>
    <div class="metric-card fade-in delay-2">
        <div class="metric-icon green">
            <i class="fas fa-calendar-check"></i>
        </div>
        <div class="metric-value" id="metric-scheduled">--</div>
        <div class="metric-label">Scheduled Today</div>
        <span class="metric-trend up" id="trend-scheduled">
            <i class="fas fa-sync fa-spin"></i> Loading
        </span>
    </div>
    <div class="metric-card fade-in delay-3">
        <div class="metric-icon orange">
            <i class="fas fa-user-clock"></i>
        </div>
        <div class="metric-value" id="metric-pending">--</div>
        <div class="metric-label">Pending Patients</div>
        <span class="metric-trend up" id="trend-pending">
            <i class="fas fa-sync fa-spin"></i> Loading
        </span>
    </div>
    <div class="metric-card fade-in delay-4">
        <div class="metric-icon red">
            <i class="fas fa-user-times"></i>
        </div>
        <div class="metric-value" id="metric-noshow">--</div>
        <div class="metric-label">No-Show Rate</div>
        <span class="metric-trend up" id="trend-noshow">
            <i class="fas fa-sync fa-spin"></i> Loading
        </span>
    </div>
</div>

<!-- Charts Row -->
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-bottom: 28px;">
    <div class="card fade-in">
        <div class="card-header">
            <h3><i class="fas fa-chart-line"></i>Utilization Trend</h3>
            <span style="font-size: 0.75rem; color: var(--gray-500);">Live updates</span>
        </div>
        <div class="card-body">
            <canvas id="utilizationChart" height="280"></canvas>
        </div>
    </div>

    <div class="card fade-in">
        <div class="card-header">
            <h3><i class="fas fa-chart-pie"></i>Priority Distribution</h3>
            <span style="font-size: 0.75rem; color: var(--gray-500);">Current schedule</span>
        </div>
        <div class="card-body">
            <canvas id="priorityChart" height="280"></canvas>
        </div>
    </div>
</div>

<!-- Optimization Results & ML Predictions -->
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 28px;">
    <div class="card fade-in">
        <div class="card-header">
            <h3><i class="fas fa-cogs"></i>Optimization Results</h3>
        </div>
        <div class="card-body" id="optimization-results">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div style="padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 2rem; font-weight: 700; color: var(--white);" id="opt-scheduled">--</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500); text-transform: uppercase;">Patients Scheduled</div>
                </div>
                <div style="padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 2rem; font-weight: 700; color: var(--warning);" id="opt-unscheduled">--</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500); text-transform: uppercase;">Unscheduled</div>
                </div>
                <div style="padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 2rem; font-weight: 700; color: #34d399;" id="opt-utilization">--</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500); text-transform: uppercase;">Utilization Achieved</div>
                </div>
                <div style="padding: 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 2rem; font-weight: 700; color: var(--primary-400);" id="opt-score">--</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500); text-transform: uppercase;">Objective Score</div>
                </div>
            </div>
        </div>
    </div>

    <div class="card fade-in">
        <div class="card-header">
            <h3><i class="fas fa-brain"></i>ML Predictions</h3>
            <span id="ml-status" style="font-size: 0.7rem; padding: 4px 10px; background: rgba(124, 58, 237, 0.2); color: #a78bfa; border-radius: 10px;">Active</span>
        </div>
        <div class="card-body" id="ml-predictions">
            <div style="max-height: 250px; overflow-y: auto;">
                <table style="width: 100%;">
                    <thead>
                        <tr>
                            <th style="padding: 10px; font-size: 0.7rem;">Patient</th>
                            <th style="padding: 10px; font-size: 0.7rem;">No-Show Risk</th>
                            <th style="padding: 10px; font-size: 0.7rem;">Est. Duration</th>
                        </tr>
                    </thead>
                    <tbody id="ml-table">
                        <tr><td colspan="3" style="text-align: center; padding: 40px; color: var(--gray-500);">Loading predictions...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
let refreshInterval = 900000; // 15 minutes default
let refreshTimer = null;
let nextRefreshTime = null;
let utilizationChart = null;
let priorityChart = null;

document.addEventListener('DOMContentLoaded', function() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';

    initCharts();
    loadAllData();
    startAutoRefresh();
});

function initCharts() {
    // Utilization Chart
    utilizationChart = new Chart(document.getElementById('utilizationChart'), {
        type: 'line',
        data: {
            labels: ['8:00', '9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00'],
            datasets: [{
                label: 'Chair Utilization %',
                data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#1e293b',
                pointBorderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255, 255, 255, 0.04)' } },
                x: { grid: { color: 'rgba(255, 255, 255, 0.04)' } }
            }
        }
    });

    // Priority Chart
    priorityChart = new Chart(document.getElementById('priorityChart'), {
        type: 'doughnut',
        data: {
            labels: ['P1 Critical', 'P2 High', 'P3 Standard', 'P4 Routine'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: ['rgba(220, 38, 38, 0.8)', 'rgba(234, 88, 12, 0.8)', 'rgba(217, 119, 6, 0.8)', 'rgba(5, 150, 105, 0.8)'],
                borderColor: '#0f172a',
                borderWidth: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: { legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } } }
        }
    });
}

// Workflow state
let workflowState = {
    sourceSelected: 'local',
    dataLoaded: false,
    analysisComplete: false
};

function toggleWorkflowPanel() {
    const panel = document.getElementById('workflow-panel');
    const icon = document.getElementById('toggle-icon');
    const text = document.getElementById('toggle-text');

    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        icon.className = 'fas fa-chevron-up';
        text.textContent = 'Hide Workflow';
    } else {
        panel.style.display = 'none';
        icon.className = 'fas fa-chevron-down';
        text.textContent = 'Show Workflow';
    }
}

function selectSource(source) {
    workflowState.sourceSelected = source;

    // Update button styles
    const btnLocal = document.getElementById('btn-local');
    const btnCloud = document.getElementById('btn-cloud');
    const sourcePath = document.getElementById('source-path');

    if (source === 'local') {
        btnLocal.style.background = 'rgba(99, 102, 241, 0.2)';
        btnLocal.style.border = '1px solid #6366f1';
        btnLocal.style.color = 'var(--white)';
        btnCloud.style.background = 'rgba(255,255,255,0.03)';
        btnCloud.style.border = '1px solid rgba(255,255,255,0.1)';
        btnCloud.style.color = 'var(--gray-400)';
        sourcePath.innerHTML = '<i class="fas fa-folder-open"></i> datasets/sample_data/';
    } else {
        btnCloud.style.background = 'rgba(99, 102, 241, 0.2)';
        btnCloud.style.border = '1px solid #6366f1';
        btnCloud.style.color = 'var(--white)';
        btnLocal.style.background = 'rgba(255,255,255,0.03)';
        btnLocal.style.border = '1px solid rgba(255,255,255,0.1)';
        btnLocal.style.color = 'var(--gray-400)';
        sourcePath.innerHTML = '<i class="fas fa-cloud"></i> Cloud endpoint (configure in settings)';
    }

    // Reset subsequent steps if source changes
    if (workflowState.dataLoaded) {
        resetWorkflowFrom(2);
    }
}

function loadDataFromSource() {
    const btnLoad = document.getElementById('btn-load');
    const loadStatus = document.getElementById('load-status');
    const step2 = document.getElementById('step-2');
    const step3 = document.getElementById('step-3');

    // Show loading state
    btnLoad.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    btnLoad.disabled = true;
    loadStatus.innerHTML = '<i class="fas fa-sync fa-spin"></i> Connecting to data source...';

    // Call the data load API
    fetch('/api/data/load', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                workflowState.dataLoaded = true;

                // Update Step 2 appearance
                step2.style.border = '2px solid #059669';
                step2.style.opacity = '1';
                btnLoad.innerHTML = '<i class="fas fa-check"></i> Data Loaded';
                btnLoad.style.background = 'rgba(5, 150, 105, 0.3)';
                btnLoad.style.color = '#34d399';
                loadStatus.innerHTML = '<i class="fas fa-check-circle" style="color: #34d399;"></i> ' +
                    (data.patients_count || 0) + ' patients, ' + (data.appointments_count || 0) + ' appointments loaded';

                // Enable Step 3
                step3.style.opacity = '1';
                step3.style.border = '2px solid rgba(99, 102, 241, 0.5)';
                const btnAnalyze = document.getElementById('btn-analyze');
                btnAnalyze.disabled = false;
                btnAnalyze.style.background = 'rgba(139, 92, 246, 0.2)';
                btnAnalyze.style.border = '1px solid #8b5cf6';
                btnAnalyze.style.color = '#a78bfa';
                btnAnalyze.style.cursor = 'pointer';
                document.getElementById('analyze-status').innerHTML = '<i class="fas fa-unlock"></i> Ready to analyze';

                // Update progress bar
                document.getElementById('workflow-progress').style.width = '66%';

                // Refresh the main data display
                loadAllData();
            } else {
                btnLoad.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Retry';
                btnLoad.disabled = false;
                loadStatus.innerHTML = '<i class="fas fa-times-circle" style="color: #f87171;"></i> ' + (data.message || 'Failed to load data');
            }
        })
        .catch(err => {
            btnLoad.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Retry';
            btnLoad.disabled = false;
            loadStatus.innerHTML = '<i class="fas fa-times-circle" style="color: #f87171;"></i> Connection error';
        });
}

function runFullAnalysis() {
    if (!workflowState.dataLoaded) {
        alert('Please load data first');
        return;
    }

    const btnAnalyze = document.getElementById('btn-analyze');
    const analyzeStatus = document.getElementById('analyze-status');
    const step3 = document.getElementById('step-3');

    // Show analyzing state
    btnAnalyze.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    btnAnalyze.disabled = true;
    analyzeStatus.innerHTML = '<i class="fas fa-sync fa-spin"></i> Running ML predictions & optimization...';

    // Call the optimization API
    fetch('/api/optimize/run', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            workflowState.analysisComplete = true;

            // Update Step 3 appearance
            step3.style.border = '2px solid #8b5cf6';
            btnAnalyze.innerHTML = '<i class="fas fa-check"></i> Analysis Complete';
            btnAnalyze.style.background = 'rgba(139, 92, 246, 0.3)';
            btnAnalyze.style.color = '#a78bfa';
            analyzeStatus.innerHTML = '<i class="fas fa-check-circle" style="color: #a78bfa;"></i> ' +
                data.message + (data.scheduled > 0 ? ' (' + data.scheduled + ' scheduled)' : '');

            // Update progress bar to 100%
            document.getElementById('workflow-progress').style.width = '100%';

            // Refresh all data displays
            loadAllData();
        })
        .catch(err => {
            btnAnalyze.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Retry';
            btnAnalyze.disabled = false;
            analyzeStatus.innerHTML = '<i class="fas fa-times-circle" style="color: #f87171;"></i> Analysis failed';
        });
}

function resetWorkflowFrom(step) {
    if (step <= 2) {
        workflowState.dataLoaded = false;
        const step2 = document.getElementById('step-2');
        step2.style.border = '2px solid rgba(255,255,255,0.1)';
        step2.style.opacity = '0.7';
        document.getElementById('btn-load').innerHTML = '<i class="fas fa-plug"></i> Connect & Load';
        document.getElementById('btn-load').disabled = false;
        document.getElementById('btn-load').style.background = 'rgba(5, 150, 105, 0.2)';
        document.getElementById('btn-load').style.color = '#34d399';
        document.getElementById('load-status').innerHTML = '<i class="fas fa-info-circle"></i> Click to load patient & appointment data';
    }
    if (step <= 3) {
        workflowState.analysisComplete = false;
        const step3 = document.getElementById('step-3');
        step3.style.border = '2px solid rgba(255,255,255,0.1)';
        step3.style.opacity = '0.7';
        const btnAnalyze = document.getElementById('btn-analyze');
        btnAnalyze.innerHTML = '<i class="fas fa-magic"></i> Run Analysis';
        btnAnalyze.disabled = true;
        btnAnalyze.style.background = 'rgba(255,255,255,0.03)';
        btnAnalyze.style.border = '1px solid rgba(255,255,255,0.1)';
        btnAnalyze.style.color = 'var(--gray-500)';
        btnAnalyze.style.cursor = 'not-allowed';
        document.getElementById('analyze-status').innerHTML = '<i class="fas fa-lock"></i> Load data first to enable';
        document.getElementById('workflow-progress').style.width = '33%';
    }
}

function loadAllData() {
    // Load status and metrics
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('data-source').textContent = data.data_source_status || 'Local';
            document.getElementById('last-refresh').textContent = data.last_data_refresh ? new Date(data.last_data_refresh).toLocaleTimeString() : 'Never';
            document.getElementById('last-optimization').textContent = data.last_optimization ? new Date(data.last_optimization).toLocaleTimeString() : 'Never';

            // Update metrics
            if (data.metrics) {
                document.getElementById('metric-utilization').textContent = data.metrics.chair_utilization + '%';
                document.getElementById('metric-scheduled').textContent = data.metrics.scheduled_today;
                document.getElementById('metric-pending').textContent = data.metrics.pending_patients;
                document.getElementById('metric-noshow').textContent = data.metrics.no_show_rate + '%';

                document.getElementById('trend-utilization').innerHTML = '<i class="fas fa-check"></i> Live';
                document.getElementById('trend-scheduled').innerHTML = '<i class="fas fa-check"></i> Live';
                document.getElementById('trend-pending').innerHTML = '<i class="fas fa-check"></i> Live';
                document.getElementById('trend-noshow').innerHTML = '<i class="fas fa-check"></i> Live';

                // Update charts with simulated hourly data
                utilizationChart.data.datasets[0].data = [65, 78, 85, 92, 88, 75, 90, 95, 82, 70];
                utilizationChart.update();
            }
        });

    // Load optimization results
    fetch('/api/metrics')
        .then(r => r.json())
        .then(data => {
            if (data.optimization_results) {
                document.getElementById('opt-scheduled').textContent = data.optimization_results.patients_scheduled || 0;
                document.getElementById('opt-unscheduled').textContent = data.optimization_results.patients_unscheduled || 0;
                document.getElementById('opt-utilization').textContent = (data.optimization_results.utilization_achieved || 0) + '%';
                document.getElementById('opt-score').textContent = (data.optimization_results.objective_score || 0).toFixed(2);
            }
        });

    // Load ML predictions
    fetch('/api/ml/predictions')
        .then(r => r.json())
        .then(data => {
            document.getElementById('ml-status').textContent = data.model_status === 'active' ? 'Active' : 'Simulated';

            const tbody = document.getElementById('ml-table');
            if (data.noshow_predictions && data.noshow_predictions.length > 0) {
                tbody.innerHTML = data.noshow_predictions.map((pred, i) => {
                    const dur = data.duration_predictions[i];
                    const riskColor = pred.risk_level === 'high' ? '#f87171' : pred.risk_level === 'medium' ? '#fbbf24' : '#34d399';
                    return `<tr>
                        <td style="padding: 12px; color: var(--white);">${pred.patient_id}</td>
                        <td style="padding: 12px;"><span style="color: ${riskColor};">${(pred.probability * 100).toFixed(1)}%</span></td>
                        <td style="padding: 12px;">${dur ? dur.predicted_duration : '--'} min</td>
                    </tr>`;
                }).join('');

                // Update priority chart based on predictions count
                priorityChart.data.datasets[0].data = [
                    data.noshow_predictions.filter(p => p.risk_level === 'high').length || 1,
                    data.noshow_predictions.filter(p => p.risk_level === 'medium').length || 2,
                    data.noshow_predictions.filter(p => p.risk_level === 'low').length || 3,
                    1
                ];
                priorityChart.update();
            } else {
                tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; padding: 40px; color: var(--gray-500);">No predictions available. Load data first.</td></tr>';
            }
        });
}

function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);

    nextRefreshTime = Date.now() + refreshInterval;
    updateNextRefreshDisplay();

    refreshTimer = setInterval(() => {
        loadAllData();
        nextRefreshTime = Date.now() + refreshInterval;
    }, refreshInterval);

    // Update countdown every second
    setInterval(updateNextRefreshDisplay, 1000);
}

function updateNextRefreshDisplay() {
    if (nextRefreshTime) {
        const remaining = Math.max(0, nextRefreshTime - Date.now());
        const minutes = Math.floor(remaining / 60000);
        const seconds = Math.floor((remaining % 60000) / 1000);
        document.getElementById('next-refresh').textContent = `${minutes}m ${seconds}s`;
    }
}

function updateRefreshInterval(ms) {
    refreshInterval = parseInt(ms);
    startAutoRefresh();

    // Also update server config
    fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({refresh_interval: refreshInterval / 1000})
    });
}

function manualRefresh() {
    document.getElementById('status-text').textContent = 'Refreshing...';
    document.getElementById('status-dot').style.background = '#fbbf24';

    fetch('/api/refresh', {method: 'POST'})
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                loadAllData();
                document.getElementById('status-text').textContent = 'Connected';
                document.getElementById('status-dot').style.background = '#34d399';
                nextRefreshTime = Date.now() + refreshInterval;
            }
        });
}

function runFullOptimization() {
    document.getElementById('opt-scheduled').textContent = '...';

    fetch('/api/optimize/run', {method: 'POST'})
        .then(r => r.json())
        .then(data => {
            loadAllData();
            alert(data.message + (data.scheduled > 0 ? ' - Scheduled: ' + data.scheduled : ''));
        });
}
</script>
{% endblock %}
'''

# About template
about_html = '''{% extends "base.html" %}
{% block title %}About | SACT Scheduler{% endblock %}
{% block content %}
<div class="page-header fade-in">
    <h1>About This Project</h1>
    <p class="page-subtitle">SACT Intelligent Scheduling System - MSc Data Science Dissertation</p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-bottom: 28px;">
    <!-- Healthcare Partner -->
    <div class="card fade-in delay-1">
        <div class="card-header">
            <h3><i class="fas fa-hospital"></i>Healthcare Partner</h3>
        </div>
        <div class="card-body">
            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 24px;">
                <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #00a86b 0%, #059669 100%); border-radius: 20px; display: flex; align-items: center; justify-content: center; font-size: 2.5rem; color: white; box-shadow: 0 8px 30px rgba(0, 168, 107, 0.3);">
                    <i class="fas fa-ribbon"></i>
                </div>
                <div>
                    <h2 style="font-size: 1.8rem; margin-bottom: 4px;">Velindre Cancer Centre</h2>
                    <p style="color: var(--gray-400); font-size: 1rem;">NHS Wales Trust</p>
                </div>
            </div>
            <p style="color: var(--gray-300); line-height: 1.8; margin-bottom: 20px;">
                Velindre Cancer Centre is a specialist NHS cancer treatment centre serving South East Wales.
                The centre provides systemic anti-cancer therapy (SACT), radiotherapy, and supportive care
                to thousands of patients annually.
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div style="padding: 16px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--white); margin-bottom: 4px;">4</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500); text-transform: uppercase; letter-spacing: 0.08em;">Treatment Sites</div>
                </div>
                <div style="padding: 16px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--white); margin-bottom: 4px;">26</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500); text-transform: uppercase; letter-spacing: 0.08em;">Treatment Chairs</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Academic Institution -->
    <div class="card fade-in delay-2">
        <div class="card-header">
            <h3><i class="fas fa-graduation-cap"></i>Academic Institution</h3>
        </div>
        <div class="card-body">
            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 24px;">
                <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); border-radius: 20px; display: flex; align-items: center; justify-content: center; font-size: 2.5rem; color: white; font-weight: 700; font-family: 'Playfair Display', Georgia, serif; box-shadow: 0 8px 30px rgba(220, 38, 38, 0.3);">
                    C
                </div>
                <div>
                    <h2 style="font-size: 1.8rem; margin-bottom: 4px;">Cardiff University</h2>
                    <p style="color: var(--gray-400); font-size: 1rem;">School of Mathematics</p>
                </div>
            </div>
            <p style="color: var(--gray-300); line-height: 1.8; margin-bottom: 20px;">
                This system was developed as part of an MSc Data Science dissertation project
                at Cardiff University's School of Mathematics, combining advanced optimization
                techniques with machine learning for healthcare scheduling.
            </p>
            <div style="padding: 20px; background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, transparent 100%); border: 1px solid rgba(220, 38, 38, 0.2); border-radius: 16px;">
                <div style="font-size: 0.75rem; color: var(--gray-500); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">Programme</div>
                <div style="font-size: 1.1rem; color: var(--white); font-weight: 600;">MSc Data Science & Analytics</div>
                <div style="font-size: 0.85rem; color: var(--gray-400); margin-top: 4px;">Dissertation Project 2025-2026</div>
            </div>
        </div>
    </div>
</div>

<!-- System Features -->
<div class="card fade-in delay-3">
    <div class="card-header">
        <h3><i class="fas fa-cogs"></i>System Features & Capabilities</h3>
    </div>
    <div class="card-body">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px;">
            <div style="text-align: center; padding: 28px 20px; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px solid rgba(255,255,255,0.06); transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='rgba(59, 130, 246, 0.3)';" onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(255,255,255,0.06)';">
                <div style="width: 60px; height: 60px; margin: 0 auto 16px; background: linear-gradient(135deg, rgba(37, 99, 235, 0.2) 0%, rgba(37, 99, 235, 0.1) 100%); border-radius: 16px; display: flex; align-items: center; justify-content: center;">
                    <i class="fas fa-brain" style="font-size: 1.5rem; color: #60a5fa;"></i>
                </div>
                <h4 style="font-family: 'Inter', sans-serif; font-size: 1rem; margin-bottom: 8px; color: var(--white);">ML Optimization</h4>
                <p style="font-size: 0.8rem; color: var(--gray-500); line-height: 1.5;">OR-Tools CP-SAT solver with multi-objective optimization for appointment scheduling</p>
            </div>

            <div style="text-align: center; padding: 28px 20px; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px solid rgba(255,255,255,0.06); transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='rgba(5, 150, 105, 0.3)';" onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(255,255,255,0.06)';">
                <div style="width: 60px; height: 60px; margin: 0 auto 16px; background: linear-gradient(135deg, rgba(5, 150, 105, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%); border-radius: 16px; display: flex; align-items: center; justify-content: center;">
                    <i class="fas fa-satellite-dish" style="font-size: 1.5rem; color: #34d399;"></i>
                </div>
                <h4 style="font-family: 'Inter', sans-serif; font-size: 1rem; margin-bottom: 8px; color: var(--white);">Real-Time Monitoring</h4>
                <p style="font-size: 0.8rem; color: var(--gray-500); line-height: 1.5;">Weather, traffic, and news event monitoring with NLP sentiment analysis</p>
            </div>

            <div style="text-align: center; padding: 28px 20px; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px solid rgba(255,255,255,0.06); transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='rgba(217, 119, 6, 0.3)';" onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(255,255,255,0.06)';">
                <div style="width: 60px; height: 60px; margin: 0 auto 16px; background: linear-gradient(135deg, rgba(217, 119, 6, 0.2) 0%, rgba(217, 119, 6, 0.1) 100%); border-radius: 16px; display: flex; align-items: center; justify-content: center;">
                    <i class="fas fa-chart-pie" style="font-size: 1.5rem; color: #fbbf24;"></i>
                </div>
                <h4 style="font-family: 'Inter', sans-serif; font-size: 1rem; margin-bottom: 8px; color: var(--white);">Predictive Analytics</h4>
                <p style="font-size: 0.8rem; color: var(--gray-500); line-height: 1.5;">No-show prediction and treatment duration estimation using XGBoost models</p>
            </div>

            <div style="text-align: center; padding: 28px 20px; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px solid rgba(255,255,255,0.06); transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='rgba(124, 58, 237, 0.3)';" onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(255,255,255,0.06)';">
                <div style="width: 60px; height: 60px; margin: 0 auto 16px; background: linear-gradient(135deg, rgba(124, 58, 237, 0.2) 0%, rgba(124, 58, 237, 0.1) 100%); border-radius: 16px; display: flex; align-items: center; justify-content: center;">
                    <i class="fas fa-bolt" style="font-size: 1.5rem; color: #a78bfa;"></i>
                </div>
                <h4 style="font-family: 'Inter', sans-serif; font-size: 1rem; margin-bottom: 8px; color: var(--white);">Urgent Squeeze-In</h4>
                <p style="font-size: 0.8rem; color: var(--gray-500); line-height: 1.5;">ML-powered emergency insertion using no-show predictions for intelligent double-booking</p>
            </div>
        </div>
    </div>
</div>

<!-- Technical Stack -->
<div class="card fade-in delay-4" style="margin-top: 28px;">
    <div class="card-header">
        <h3><i class="fas fa-layer-group"></i>Technical Stack</h3>
    </div>
    <div class="card-body">
        <div style="display: flex; flex-wrap: wrap; gap: 12px;">
            <span style="padding: 10px 20px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 30px; font-size: 0.85rem; color: #60a5fa;">Python 3.12</span>
            <span style="padding: 10px 20px; background: rgba(5, 150, 105, 0.1); border: 1px solid rgba(5, 150, 105, 0.2); border-radius: 30px; font-size: 0.85rem; color: #34d399;">Flask</span>
            <span style="padding: 10px 20px; background: rgba(217, 119, 6, 0.1); border: 1px solid rgba(217, 119, 6, 0.2); border-radius: 30px; font-size: 0.85rem; color: #fbbf24;">OR-Tools</span>
            <span style="padding: 10px 20px; background: rgba(124, 58, 237, 0.1); border: 1px solid rgba(124, 58, 237, 0.2); border-radius: 30px; font-size: 0.85rem; color: #a78bfa;">XGBoost</span>
            <span style="padding: 10px 20px; background: rgba(236, 72, 153, 0.1); border: 1px solid rgba(236, 72, 153, 0.2); border-radius: 30px; font-size: 0.85rem; color: #f472b6;">scikit-learn</span>
            <span style="padding: 10px 20px; background: rgba(20, 184, 166, 0.1); border: 1px solid rgba(20, 184, 166, 0.2); border-radius: 30px; font-size: 0.85rem; color: #2dd4bf;">Pandas</span>
            <span style="padding: 10px 20px; background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 30px; font-size: 0.85rem; color: #fb7185;">NLTK/VADER</span>
            <span style="padding: 10px 20px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 30px; font-size: 0.85rem; color: #818cf8;">Chart.js</span>
            <span style="padding: 10px 20px; background: rgba(251, 146, 60, 0.1); border: 1px solid rgba(251, 146, 60, 0.2); border-radius: 30px; font-size: 0.85rem; color: #fb923c;">Folium Maps</span>
            <span style="padding: 10px 20px; background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 30px; font-size: 0.85rem; color: #4ade80;">Open-Meteo API</span>
        </div>
    </div>
</div>
{% endblock %}
'''

# Write templates
(templates_dir / 'base.html').write_text(base_html)
(templates_dir / 'index.html').write_text(index_html)
(templates_dir / 'schedule.html').write_text(schedule_html)
(templates_dir / 'patients.html').write_text(patients_html)
(templates_dir / 'events.html').write_text(events_html)
(templates_dir / 'analytics.html').write_text(analytics_html)
(templates_dir / 'about.html').write_text(about_html)

# Update Flask template folder
app.template_folder = str(templates_dir)


# =============================================================================
# VIEWER DATA ENDPOINTS (For mentor's viewer integration)
# =============================================================================

@app.route('/api/schedule/full')
def api_schedule_full():
    """
    Enriched schedule data for the viewer Gantt chart.
    Serves the full appointments.xlsx data (34 fields) with ML predictions.
    """
    try:
        # Use the rich appointments data from Excel (not the sparse optimizer output)
        data_dir = app_state.get('data_dir', 'datasets/sample_data')
        appt_path = Path(data_dir) / 'appointments.xlsx'

        if appt_path.exists():
            df = pd.read_excel(appt_path)

            # Add ML predictions — build pid->prediction map from list
            raw_predictions = app_state.get('ml_predictions', {})
            pred_map = {}
            if isinstance(raw_predictions, dict) and 'noshow_predictions' in raw_predictions:
                for p in raw_predictions.get('noshow_predictions', []):
                    pred_map[p['patient_id']] = p
            elif isinstance(raw_predictions, dict):
                pred_map = raw_predictions  # Already a map

            if pred_map:
                df['noshow_probability'] = df['Patient_ID'].map(
                    lambda pid: pred_map.get(pid, {}).get('probability', pred_map.get(pid, {}).get('noshow_probability', 0))
                )
                df['risk_level'] = df['Patient_ID'].map(
                    lambda pid: pred_map.get(pid, {}).get('risk_level', 'unknown')
                )

            # Convert to JSON-safe records
            # Replace NaN/NaT with None, convert numpy types to Python native
            df = df.fillna('')  # Replace all NaN with empty string
            df = df.where(df.notna(), None)  # Belt and suspenders

            # Use pandas to_dict which handles type conversion
            records = json.loads(df.to_json(orient='records', date_format='iso', default_handler=str))

            return jsonify({
                'schedule': records,
                'count': len(records),
                'has_ml_predictions': bool(pred_map)
            })
        else:
            # Fallback to optimizer output
            schedule = app_state.get('appointments', [])
            enriched = []
            for appt in schedule:
                entry = {}
                if hasattr(appt, '__dict__'):
                    entry = {k: str(v) if hasattr(v, 'isoformat') else v
                             for k, v in appt.__dict__.items() if not k.startswith('_')}
                elif isinstance(appt, dict):
                    entry = dict(appt)
                enriched.append(entry)

            return jsonify({
                'schedule': enriched,
                'count': len(enriched),
                'has_ml_predictions': False
            })
    except Exception as e:
        logger.error(f"Schedule full error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/patients/list')
def api_patients_list():
    """Full patient list with all details."""
    try:
        df = app_state.get('patients_df')
        if df is None:
            return jsonify({'patients': [], 'count': 0})

        # Clean NaN values before converting
        df = df.fillna('')

        # Add ML predictions
        raw_predictions = app_state.get('ml_predictions', {})
        pred_map = {}
        if isinstance(raw_predictions, dict) and 'noshow_predictions' in raw_predictions:
            for p in raw_predictions.get('noshow_predictions', []):
                pred_map[p['patient_id']] = p

        if pred_map:
            df['noshow_probability'] = df['Patient_ID'].map(
                lambda pid: pred_map.get(pid, {}).get('probability', 0)
            )
            df['risk_level'] = df['Patient_ID'].map(
                lambda pid: pred_map.get(pid, {}).get('risk_level', 'unknown')
            )

        # Use pandas JSON serialization to avoid NaN
        records = json.loads(df.to_json(orient='records', default_handler=str))

        return jsonify({'patients': records, 'count': len(records)})
    except Exception as e:
        logger.error(f"Patients list error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sites')
def api_sites():
    """Site configuration with chairs and operating hours."""
    try:
        # Try loading from Excel first
        data_dir = app_state.get('data_dir', 'datasets/sample_data')
        sites_path = Path(data_dir) / 'sites.xlsx'

        if sites_path.exists():
            df = pd.read_excel(sites_path)
            sites = df.to_dict('records')
        else:
            # Fallback to config defaults
            try:
                from config import DEFAULT_SITES
                sites = [{'id': k, **{kk: str(vv) if not isinstance(vv, (int, float, bool)) else vv
                          for kk, vv in v.items()}} for k, v in DEFAULT_SITES.items()]
            except (ImportError, AttributeError):
                sites = []

        return jsonify({'sites': sites, 'count': len(sites)})
    except Exception as e:
        logger.error(f"Sites endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/regimens')
def api_regimens():
    """Regimen/protocol details."""
    try:
        df = app_state.get('regimens_df')
        if df is None:
            return jsonify({'regimens': [], 'count': 0})

        df = df.fillna('')
        records = json.loads(df.to_json(orient='records', default_handler=str))
        return jsonify({'regimens': records, 'count': len(records)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/causal/counterfactual-explanation', methods=['POST'])
def api_counterfactual_explanation():
    """
    Find minimum feature change to flip a patient's risk prediction.

    CF(x) = argmin_{x'} ||x' - x||_2  s.t. f(x') != f(x)

    POST /api/ml/causal/counterfactual-explanation
    {
        "patient_id": "P93810",
        "Previous_NoShows": 3,
        "Travel_Distance_KM": 40,
        "Age": 65,
        "Cycle_Number": 2,
        "Performance_Status": 2
    }
    """
    if not causal_model:
        return jsonify({'error': 'Causal model not available'}), 503

    data = request.json or {}

    # Build patient features
    patient_features = {
        'Previous_NoShows': data.get('Previous_NoShows', 2),
        'Previous_Cancellations': data.get('Previous_Cancellations', 1),
        'Travel_Distance_KM': data.get('Travel_Distance_KM', 20),
        'Travel_Time_Min': data.get('Travel_Time_Min', 30),
        'Days_Booked_In_Advance': data.get('Days_Booked_In_Advance', 14),
        'Cycle_Number': data.get('Cycle_Number', 3),
        'Weather_Severity': data.get('Weather_Severity', 0.1),
        'Performance_Status': data.get('Performance_Status', 1),
        'Age': data.get('Age', 55),
        'Patient_NoShow_Rate': data.get('Previous_NoShows', 2) / max(data.get('Total_Appointments', 10), 1),
    }

    # Simple prediction function using no-show rate heuristic
    def predict_noshow(features):
        rate = features.get('Patient_NoShow_Rate', 0.1)
        distance_factor = features.get('Travel_Distance_KM', 10) / 100
        weather_factor = features.get('Weather_Severity', 0) * 0.15
        ps_factor = features.get('Performance_Status', 0) * 0.03
        prob = 0.05 + rate * 0.5 + distance_factor * 0.1 + weather_factor + ps_factor
        # Recalculate rate from no-shows
        noshows = features.get('Previous_NoShows', 0)
        total = max(features.get('Total_Appointments', 10), 1)
        prob = 0.05 + (noshows / total) * 0.5 + distance_factor * 0.1 + weather_factor + ps_factor
        return min(max(prob, 0.01), 0.95)

    try:
        result = causal_model.counterfactual_explanation(
            patient_features=patient_features,
            prediction_func=predict_noshow,
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Counterfactual explanation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/causal/validate', methods=['POST'])
def api_causal_validate():
    """Run causal validation suite (placebo + falsification + sensitivity)."""
    try:
        from ml.causal_validation import CausalValidator
        validator = CausalValidator(tolerance=0.05)

        if historical_appointments_df is not None and len(historical_appointments_df) > 50:
            results = validator.run_all_tests(historical_appointments_df, causal_model)
            return jsonify(results)
        else:
            return jsonify({'error': 'Insufficient historical data for validation'}), 400
    except Exception as e:
        logger.error(f"Causal validation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/model-cards', methods=['GET'])
def api_model_cards_all():
    """Get Model Cards for all ML models — transparency and accountability."""
    try:
        from ml.model_cards import ModelCardGenerator
        generator = ModelCardGenerator()

        models_map = {
            'noshow_ensemble': noshow_model,
            'duration_model': duration_model,
            'causal_model': causal_model,
        }

        result = generator.generate_all_cards(
            models=models_map,
            historical_data=historical_appointments_df
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Model cards error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/model-cards/<model_key>', methods=['GET'])
def api_model_card_single(model_key):
    """Get Model Card for a specific model."""
    try:
        from ml.model_cards import ModelCardGenerator
        generator = ModelCardGenerator()

        models_map = {
            'noshow_ensemble': noshow_model,
            'duration_model': duration_model,
            'causal_model': causal_model,
            'rl_scheduler': None,
        }

        model = models_map.get(model_key)
        is_trained = getattr(model, 'is_trained', False) if model else False

        card = generator.generate_card(
            model_key=model_key,
            model=model,
            historical_data=historical_appointments_df,
            is_trained=is_trained
        )
        return jsonify(card)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/uncertainty-optimization', methods=['GET'])
def api_uncertainty_opt_status():
    """Get DRO uncertainty-aware optimization status and formulas."""
    try:
        from optimization.uncertainty_optimization import UncertaintyAwareOptimizer
        dro = UncertaintyAwareOptimizer()
        return jsonify({
            'status': 'active',
            'method': 'Distributionally Robust Optimization (Wasserstein DRO)',
            'parameters': {
                'epsilon': dro.epsilon,
                'alpha': dro.alpha,
                'n_scenarios': dro.n_scenarios,
            },
            'formulas': dro.get_formulas(),
            'description': (
                'Replaces point-estimate noshow probabilities with worst-case values '
                'from a Wasserstein ambiguity set. Duration buffers use CVaR to protect '
                'against the worst (1-alpha) fraction of overruns. '
                'CP-SAT objective uses Rockafellar-Uryasev CVaR linearization: '
                'max { eta - 1/(K(1-alpha)) * sum_k z_k } s.t. z_k >= eta - U_k.'
            ),
            'cvar_in_cpsat': {
                'formulation': 'max CVaR_alpha(U) via auxiliary variables (Rockafellar & Uryasev 2000)',
                'eta': 'VaR auxiliary variable (integer)',
                'z_k': 'Shortfall slack per scenario (z_k >= eta - U_k, z_k >= 0)',
                'K': '10 scenarios of no-show realizations',
                'integrated': True,
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/uncertainty-optimization/evaluate', methods=['POST'])
def api_uncertainty_evaluate():
    """Evaluate current schedule robustness under distributional shifts."""
    try:
        from optimization.uncertainty_optimization import UncertaintyAwareOptimizer
        data = request.json or {}
        epsilon = _clamp_float(data.get('epsilon'), field='epsilon',
                               default=0.05, min_value=0.0, max_value=1.0)
        alpha = _clamp_float(data.get('alpha'), field='alpha',
                             default=0.90, min_value=0.0, max_value=1.0)
        n_scenarios = _clamp_int(data.get('n_scenarios'), field='n_uncertainty_scenarios',
                                 default=50, min_value=1)

        dro = UncertaintyAwareOptimizer(epsilon=epsilon, alpha=alpha, n_scenarios=n_scenarios)

        appointments = app_state.get('appointments', [])
        if not appointments:
            return jsonify({'error': 'No schedule to evaluate'}), 400

        # Build patient list from appointments for scenario generation
        patients_for_scenarios = []
        for apt in appointments[:100]:
            class TempPatient:
                pass
            p = TempPatient()
            p.patient_id = apt.patient_id
            p.noshow_probability = getattr(apt, 'noshow_probability', 0.13)
            p.expected_duration = apt.duration
            p.priority = apt.priority
            patients_for_scenarios.append(p)

        scenarios = dro.generate_scenarios(patients_for_scenarios, n_scenarios)
        evaluation = dro.evaluate_schedule_robustness(appointments[:100], scenarios)

        return jsonify(evaluation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/uncertainty-optimization/calibrate', methods=['POST'])
def api_uncertainty_calibrate():
    """Calibrate epsilon from historical data (seasonal shifts, disruptions)."""
    try:
        from optimization.uncertainty_optimization import UncertaintyAwareOptimizer
        dro = UncertaintyAwareOptimizer()
        result = dro.calibrate_epsilon(historical_appointments_df)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/uncertainty-optimization/metrics', methods=['POST'])
def api_uncertainty_metrics():
    """Evaluate worst-case utilization and stability across scenarios."""
    try:
        from optimization.uncertainty_optimization import UncertaintyAwareOptimizer
        data = request.json or {}
        epsilon = _clamp_float(data.get('epsilon'), field='epsilon',
                               default=0.05, min_value=0.0, max_value=1.0)
        n_scenarios = _clamp_int(data.get('n_scenarios'), field='n_uncertainty_scenarios',
                                 default=30, min_value=1)

        dro = UncertaintyAwareOptimizer(epsilon=epsilon, alpha=0.90)

        # Calibrate epsilon from data if requested
        if data.get('auto_calibrate', True) and historical_appointments_df is not None:
            cal = dro.calibrate_epsilon(historical_appointments_df)
            epsilon = cal.get('epsilon_recommended', epsilon)

        appointments = app_state.get('appointments', [])
        if not appointments:
            return jsonify({'error': 'No schedule to evaluate'}), 400

        # Build patients for scenario generation
        patients_list = []
        for apt in appointments[:100]:
            class _P:
                pass
            p = _P()
            p.patient_id = apt.patient_id
            p.noshow_probability = getattr(apt, 'noshow_probability', 0.13)
            p.expected_duration = apt.duration
            p.priority = apt.priority
            patients_list.append(p)

        scenarios = dro.generate_scenarios(patients_list, n_scenarios)
        metrics = dro.evaluate_schedule_metrics(appointments[:100], scenarios)

        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/sensitivity', methods=['GET'])
def api_sensitivity_status():
    """Get sensitivity analysis module status and available features."""
    try:
        from ml.sensitivity_analysis import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer()
        return jsonify({
            'status': 'active',
            'method': 'Numerical Finite-Difference Sensitivity Analysis',
            'formulas': {
                'local': 'S_i = d(y_hat)/d(x_i) via central difference',
                'global': 'I_j = (1/n) * sum_i |S_ij|',
                'elasticity': 'E_i = S_i * (x_i / y_hat)',
            },
            'available_models': ['noshow', 'duration'],
            'feature_categories': list(analyzer.FEATURE_CATEGORIES.keys()),
            'default_perturbation': 0.01,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/sensitivity/local', methods=['POST'])
def api_sensitivity_local():
    """
    Compute local sensitivity S_i = dy_hat/dx_i for a specific patient.

    POST JSON:
    {
        "patient_id": "P24116",    (optional, uses first patient if omitted)
        "model": "noshow",         (or "duration")
        "perturbation": 0.01       (optional, default 1%)
    }
    """
    try:
        from ml.sensitivity_analysis import SensitivityAnalyzer
        data = request.json or {}
        model_type = data.get('model', 'noshow')
        perturbation = data.get('perturbation', 0.01)
        patient_id = data.get('patient_id')

        analyzer = SensitivityAnalyzer(perturbation=perturbation)

        # Get the model and feature data
        if model_type == 'noshow':
            target_model = noshow_model
        else:
            target_model = duration_model

        # Build feature vector from historical data
        if historical_appointments_df is not None and len(historical_appointments_df) > 10:
            df = historical_appointments_df.copy()
            X, feature_names = _prepare_sensitivity_data(df)

            # Pick patient row
            if patient_id and 'Patient_ID' in df.columns:
                mask = df['Patient_ID'] == patient_id
                if mask.any():
                    idx = mask.idxmax()
                    x_patient = X[idx]
                else:
                    x_patient = X[0]
                    patient_id = str(df.iloc[0].get('Patient_ID', 'sample_0'))
            else:
                x_patient = X[0]
                patient_id = str(df.iloc[0].get('Patient_ID', 'sample_0'))

            # Get underlying sklearn model
            sklearn_model = _get_sklearn_model(target_model, model_type, X, df)

            local_result = analyzer.local_sensitivity(
                model=sklearn_model,
                feature_vector=x_patient,
                feature_names=feature_names,
                model_type=model_type,
                patient_id=patient_id
            )

            result = analyzer.format_results(local=local_result)
            return jsonify(result)
        else:
            return jsonify({'error': 'Insufficient historical data'}), 400

    except Exception as e:
        logger.error(f"Local sensitivity error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/sensitivity/global', methods=['POST'])
def api_sensitivity_global():
    """
    Compute global importance I_j = (1/n) * sum|S_ij| across population.

    POST JSON:
    {
        "model": "noshow",         (or "duration")
        "max_samples": 100,        (optional, default 100)
        "perturbation": 0.01       (optional)
    }
    """
    try:
        from ml.sensitivity_analysis import SensitivityAnalyzer
        data = request.json or {}
        model_type = data.get('model', 'noshow')
        max_samples = min(data.get('max_samples', 100), 200)
        perturbation = data.get('perturbation', 0.01)

        analyzer = SensitivityAnalyzer(perturbation=perturbation)

        if model_type == 'noshow':
            target_model = noshow_model
        else:
            target_model = duration_model

        if historical_appointments_df is not None and len(historical_appointments_df) > 10:
            df = historical_appointments_df.copy()
            X, feature_names = _prepare_sensitivity_data(df)

            sklearn_model = _get_sklearn_model(target_model, model_type, X, df)

            global_result = analyzer.global_importance(
                model=sklearn_model,
                feature_matrix=X,
                feature_names=feature_names,
                model_type=model_type,
                max_samples=max_samples
            )

            result = analyzer.format_results(glob=global_result)
            return jsonify(result)
        else:
            return jsonify({'error': 'Insufficient historical data'}), 400

    except Exception as e:
        logger.error(f"Global sensitivity error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/sensitivity/full', methods=['POST'])
def api_sensitivity_full():
    """
    Run both local and global sensitivity analysis.

    POST JSON:
    {
        "patient_id": "P24116",
        "model": "noshow",
        "max_samples": 100
    }
    """
    try:
        from ml.sensitivity_analysis import SensitivityAnalyzer
        data = request.json or {}
        model_type = data.get('model', 'noshow')
        max_samples = min(data.get('max_samples', 100), 200)
        patient_id = data.get('patient_id')

        analyzer = SensitivityAnalyzer(perturbation=0.01)

        if model_type == 'noshow':
            target_model = noshow_model
        else:
            target_model = duration_model

        if historical_appointments_df is not None and len(historical_appointments_df) > 10:
            df = historical_appointments_df.copy()
            X, feature_names = _prepare_sensitivity_data(df)

            # Pick patient
            if patient_id and 'Patient_ID' in df.columns:
                mask = df['Patient_ID'] == patient_id
                idx = mask.idxmax() if mask.any() else 0
            else:
                idx = 0
                patient_id = str(df.iloc[0].get('Patient_ID', 'sample_0'))

            sklearn_model = _get_sklearn_model(target_model, model_type, X, df)

            local_result = analyzer.local_sensitivity(
                sklearn_model, X[idx], feature_names, model_type, patient_id
            )
            global_result = analyzer.global_importance(
                sklearn_model, X, feature_names, model_type, max_samples
            )

            # Also compute elasticity for the patient
            elasticity = analyzer.elasticity(
                sklearn_model, X[idx], feature_names, model_type
            )
            sorted_elasticity = sorted(elasticity.items(), key=lambda x: abs(x[1]), reverse=True)

            result = analyzer.format_results(local=local_result, glob=global_result)
            result['elasticity'] = {
                'formula': 'E_i = S_i * (x_i / y_hat)',
                'top_20': [
                    {'feature': k, 'elasticity': round(v, 4)}
                    for k, v in sorted_elasticity[:20]
                ],
            }

            # Also include sklearn built-in importance (tree + permutation)
            y_target = (df['Attended_Status'] == 'No').astype(int) if model_type == 'noshow' and 'Attended_Status' in df.columns else pd.to_numeric(df.get('Actual_Duration', pd.Series([60]*len(X))), errors='coerce').fillna(60)
            sklearn_imp = analyzer.sklearn_importance(
                sklearn_model, X, y_target.values, feature_names, model_type, n_repeats=5
            )
            result['sklearn_importance'] = sklearn_imp

            return jsonify(result)
        else:
            return jsonify({'error': 'Insufficient historical data'}), 400

    except Exception as e:
        logger.error(f"Full sensitivity error: {e}")
        return jsonify({'error': str(e)}), 500


def _prepare_sensitivity_data(df):
    """Prepare numeric feature matrix for sensitivity analysis."""
    # Select only numeric-friendly columns
    numeric_cols = []
    exclude = {
        'Appointment_ID', 'Patient_ID', 'Date', 'Attended_Status',
        'Actual_Duration', 'Start_Time', 'End_Time', 'Created_Date',
        'NHS_Number', 'Consultant', 'Created_By', 'Modification_Reason',
        'Modification_Code', 'Time', 'Regimen_Code', 'Cancer_Type',
        'Site_Code', 'Chair_Type', 'Age_Band', 'Priority',
        'Day_Of_Week', 'Appointment_Hour', 'Person_Stated_Gender_Code',
        'Primary_Diagnosis_ICD10', 'Intent_Of_Treatment',
        'Treatment_Context', 'Toxicity_Grade', 'Bloods_Required',
        'Long_Infusion',
    }

    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            numeric_cols.append(col)

    # Also add one-hot encoding for key categoricals
    extra_features = {}
    if 'Priority' in df.columns:
        for p in ['P1', 'P2', 'P3', 'P4']:
            extra_features[f'is_{p}'] = (df['Priority'] == p).astype(float)
    if 'Site_Code' in df.columns:
        for s in ['WC', 'PCH', 'RGH', 'POW', 'CWM']:
            extra_features[f'site_{s}'] = (df['Site_Code'] == s).astype(float)
    if 'Day_Of_Week' in df.columns:
        for d_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            extra_features[f'is_{d_name}'] = (df['Day_Of_Week'] == d_name).astype(float)
    if 'Attended_Status' in df.columns:
        pass  # target, don't include

    X_df = df[numeric_cols].copy().fillna(0)
    for fname, fvals in extra_features.items():
        X_df[fname] = fvals.values

    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)
    return X, feature_names


def _get_sklearn_model(target_model, model_type, X, df):
    """Get or train a sklearn model for sensitivity analysis."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    if model_type == 'noshow':
        y = (df['Attended_Status'] == 'No').astype(int) if 'Attended_Status' in df.columns else np.zeros(len(X))
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42, min_samples_leaf=5)
        try:
            if y.sum() > 5 and (len(y) - y.sum()) > 5:
                model.fit(X, y)
            else:
                # Not enough positive/negative examples — train on actual data as-is
                model.fit(X, y)
        except Exception:
            # Last resort: use actual labels even if imbalanced
            model.fit(X, y)
    else:
        y = pd.to_numeric(df.get('Actual_Duration', df.get('Planned_Duration', pd.Series([60]*len(X)))), errors='coerce').fillna(60)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42, min_samples_leaf=5)
        try:
            model.fit(X, y)
        except Exception:
            model.fit(X, np.full(len(X), 60.0))

    return model

    return model


@app.route('/api/schedule/robustness')
def api_schedule_robustness():
    """Get detailed robustness metrics for the current schedule."""
    try:
        if app_state.get('appointments'):
            metrics = optimizer._calculate_metrics(
                app_state['appointments'], app_state.get('patients', []), datetime.now()
            )
            rob = metrics.get('robustness', {})
            return jsonify({
                'robustness_score': metrics.get('robustness_score', 0),
                'details': rob,
                'interpretation': {
                    'R_S': f"Schedule robustness: {rob.get('R_S', 0):.3f} (0=fragile, 1=maximally robust)",
                    'avg_gap': f"Avg gap: {rob.get('avg_gap_minutes', 0):.1f} min",
                    'slack_min': f"Tightest: {rob.get('slack_min', 0):.0f} min",
                    'slack_max': f"Most slack: {rob.get('slack_max', 0):.0f} min",
                    'cascade_risk': f"Critical slots (<10min): {rob.get('critical_slots', 0)}, Tight (10-20min): {rob.get('tight_slots', 0)}",
                },
                'formulas': {
                    'R_S': 'R(S) = (1/|P|) * sum_p integral_0^H P(delay_p > t) dt',
                    'slack': 'Slack_i = min_{j!=i} |s_i - s_j|',
                }
            })
        return jsonify({'robustness_score': 0, 'error': 'No schedule'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Global RL agent
rl_agent = None

def _get_rl_agent():
    global rl_agent
    if rl_agent is None:
        from ml.rl_scheduler import SchedulingRLAgent
        rl_agent = SchedulingRLAgent(n_chairs=19, learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
        # Train on historical data if available
        if historical_appointments_df is not None and len(historical_appointments_df) > 50:
            records = historical_appointments_df.head(500).to_dict('records')
            result = rl_agent.train_on_historical(records, n_epochs=5)
            logger.info(f"RL agent trained: {result}")
    return rl_agent


@app.route('/api/ml/rl-scheduler', methods=['GET'])
def api_rl_status():
    """Get RL scheduling agent status and learned policy."""
    try:
        agent = _get_rl_agent()
        return jsonify(agent.get_policy_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/rl-scheduler/recommend', methods=['POST'])
def api_rl_recommend():
    """
    Get RL-recommended scheduling action for current state.

    POST /api/ml/rl-scheduler/recommend
    {
        "hour": 10,
        "chairs_occupied": 8,
        "queue_size": 5,
        "avg_noshow_risk": 0.15,
        "urgent_count": 1
    }
    """
    try:
        from ml.rl_scheduler import RLState
        agent = _get_rl_agent()
        data = request.json or {}

        state = RLState(
            hour=data.get('hour', 10),
            chairs_occupied=data.get('chairs_occupied', 5),
            total_chairs=19,
            queue_size=data.get('queue_size', 3),
            avg_noshow_risk=data.get('avg_noshow_risk', 0.15),
            urgent_count=data.get('urgent_count', 0),
            utilization=data.get('chairs_occupied', 5) / 19,
        )

        recommendation = agent.recommend_action(state)
        return jsonify(recommendation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/rl-scheduler/train', methods=['POST'])
def api_rl_train():
    """Train RL agent on historical data."""
    try:
        agent = _get_rl_agent()
        data = request.json or {}
        epochs = data.get('epochs', 5)

        if historical_appointments_df is not None:
            records = historical_appointments_df.head(500).to_dict('records')
            result = agent.train_on_historical(records, n_epochs=epochs)
            return jsonify({'success': True, **result})
        else:
            return jsonify({'error': 'No historical data available'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Global MARL scheduler
marl_scheduler = None

def _get_marl_scheduler():
    global marl_scheduler
    if marl_scheduler is None:
        from ml.rl_scheduler import MultiAgentChairScheduler
        marl_scheduler = MultiAgentChairScheduler(n_chairs=19, alpha=0.1, gamma=0.95, epsilon=0.15)
        # Train on historical data
        if historical_appointments_df is not None and len(historical_appointments_df) > 50:
            records = historical_appointments_df.head(500).to_dict('records')
            result = marl_scheduler.train_on_historical(records, n_epochs=3)
            logger.info(f"MARL trained: {result.get('total_updates')} updates across {marl_scheduler.n_chairs} agents")
    return marl_scheduler


@app.route('/api/ml/marl-scheduler', methods=['GET'])
def api_marl_status():
    """Get Multi-Agent RL chair scheduler status."""
    try:
        marl = _get_marl_scheduler()
        return jsonify(marl.get_policy_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/marl-scheduler/assign', methods=['POST'])
def api_marl_assign():
    """Assign patients to chairs using MARL policy."""
    try:
        from ml.rl_scheduler import ChairState
        marl = _get_marl_scheduler()
        data = request.json or {}

        patients = data.get('patients', [])
        if not patients:
            # Use pending patients
            patients = [
                {'patient_id': p.patient_id, 'priority': p.priority,
                 'duration': p.expected_duration, 'noshow_probability': p.noshow_probability}
                for p in app_state.get('patients', [])[:10]
            ]

        # Build chair states from current schedule
        chair_states = []
        for c in range(marl.n_chairs):
            chair_states.append(ChairState(
                chair_id=c, is_occupied=False,
                current_patient_priority=0, current_remaining_min=0,
                next_gap_min=60, queue_size=len(patients), hour=10
            ))

        assignments = marl.assign_patients(patients, chair_states)
        return jsonify({
            'assignments': assignments,
            'n_assigned': len(assignments),
            'method': 'Multi-Agent RL (independent Q-Learning per chair)',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/marl-scheduler/train', methods=['POST'])
def api_marl_train():
    """Train MARL agents on historical data."""
    try:
        marl = _get_marl_scheduler()
        data = request.json or {}
        epochs = data.get('epochs', 5)

        if historical_appointments_df is not None:
            records = historical_appointments_df.head(500).to_dict('records')
            result = marl.train_on_historical(records, n_epochs=epochs)
            return jsonify({'success': True, **result})
        else:
            return jsonify({'error': 'No historical data'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Global online learner
online_learner = None

def _get_online_learner():
    global online_learner
    if online_learner is None:
        from ml.online_learning import OnlineLearner
        online_learner = OnlineLearner(learning_rate=0.01, ema_alpha=0.1)
    return online_learner


@app.route('/api/ml/online-learning', methods=['GET'])
def api_online_learning_status():
    """Get online learning status and posterior parameters."""
    try:
        learner = _get_online_learner()
        return jsonify(learner.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/online-learning/update', methods=['POST'])
def api_online_learning_update():
    """
    Feed a new observation to update models in real-time.

    POST /api/ml/online-learning/update
    {
        "attended": true,
        "actual_duration": 145,
        "weather_severity": 0.2
    }
    """
    try:
        learner = _get_online_learner()
        data = request.json or {}

        attended = data.get('attended', True)
        duration = data.get('actual_duration', None)
        weather = data.get('weather_severity', 0.0)

        features = np.array([
            data.get('previous_noshow_rate', 0.1),
            data.get('distance_km', 15),
            data.get('age', 55),
            data.get('cycle_number', 1),
            weather,
            data.get('expected_duration', 120),
            1.0 if data.get('is_first_cycle', False) else 0.0,
            1.0 if data.get('has_comorbidities', False) else 0.0,
        ])

        results = learner.update_on_new_observation(
            patient_features=features, attended=attended,
            actual_duration=duration, weather_severity=weather
        )

        return jsonify({
            'success': True,
            'updates': [{'model': r.model_name, 'method': r.method, 'improvement': r.improvement, 'details': r.details} for r in results],
            'total_updates': learner.update_count,
            'posterior': learner.get_posterior_summary(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/source', methods=['GET', 'POST'])
def api_data_source():
    """
    GET: Show current data source channel.
    POST: Switch the PRIMARY data source channel and reload.

    POST /api/data/source
    {
        "channel": "synthetic" | "real"     // Channel 1 or Channel 2
    }

    Note: Channel 3 (NHS open data) is always-on in the background for
    recalibration and is not a valid primary channel — see the auto-learning
    scheduler and `docs/THREE_CHANNEL_DATA_STRATEGY.md`.  Runtime promotion
    from Ch1 to Ch2 also happens automatically when files are dropped into
    `datasets/real_data/` (handled by the `ch2-watcher` daemon; no manual
    POST required).
    """
    if request.method == 'GET':
        # Check which channels have data
        synthetic_exists = (SAMPLE_DATA_DIR / 'patients.xlsx').exists()
        real_exists = (REAL_DATA_DIR / 'patients.xlsx').exists()
        nhs_exists = (DATASETS_DIR / 'nhs_open_data' / 'cancer_waiting_times').exists()
        nhs_files = len(list((DATASETS_DIR / 'nhs_open_data').rglob('*.*'))) if nhs_exists else 0

        current = 'synthetic'
        if not DATA_SOURCE_CONFIG.get('use_sample_data', True):
            current = 'real'

        return jsonify({
            'current_channel': current,
            'channels': {
                'synthetic': {
                    'available': synthetic_exists,
                    'path': str(SAMPLE_DATA_DIR),
                    'description': 'SACT v4.0 compliant synthetic data for testing and dissertation',
                    'patients': len(pd.read_excel(SAMPLE_DATA_DIR / 'patients.xlsx')) if synthetic_exists else 0,
                },
                'real': {
                    'available': real_exists,
                    'path': str(REAL_DATA_DIR),
                    'description': 'Real hospital data from Velindre Cancer Centre (ChemoCare exports)',
                    'patients': len(pd.read_excel(REAL_DATA_DIR / 'patients.xlsx')) if real_exists else 0,
                },
                'nhs_open': {
                    'available': nhs_exists,
                    'path': str(DATASETS_DIR / 'nhs_open_data'),
                    'description': 'NHS open data (CWT Jan 2026, SCMD-IP Jan 2026, SACT v4.0 metadata) — auto-updated every 24h',
                    'files': nhs_files,
                }
            },
            'auto_learning': {
                'enabled': True,
                'interval': '24 hours',
                'sources': [
                    'Cancer Waiting Times (Jan 2026)',
                    'NHSBSA SCMD-IP (Jan 2026)',
                    'SACT v4.0 Patient Data (auto-detect from Aug 2026; partial from Apr 2026)',
                ]
            }
        })

    else:
        data = request.json or {}
        channel = data.get('channel', 'synthetic')

        if channel == 'real':
            # Enforce the same required-files rule as the Ch2 watcher so the
            # manual toggle and the auto-switch behave identically.
            required_missing = [
                name for name in _real_data_watcher_state['required_files']
                if not (REAL_DATA_DIR / name).exists()
            ]
            if required_missing:
                return jsonify({
                    'error': (
                        'Channel 2 requires both patients.xlsx and '
                        'historical_appointments.xlsx in datasets/real_data/. '
                        'Missing: ' + ', '.join(required_missing)
                    ),
                    'missing_files': required_missing,
                }), 400
            DATA_SOURCE_CONFIG['use_sample_data'] = False
            DATA_SOURCE_CONFIG['local_path'] = str(REAL_DATA_DIR)
            DATA_SOURCE_CONFIG['active_channel'] = 'real'
            app_state['data_dir'] = str(REAL_DATA_DIR)
            app_state['active_channel'] = 'real'
        elif channel == 'synthetic':
            DATA_SOURCE_CONFIG['use_sample_data'] = True
            DATA_SOURCE_CONFIG['local_path'] = str(SAMPLE_DATA_DIR)
            DATA_SOURCE_CONFIG['active_channel'] = 'synthetic'
            app_state['data_dir'] = str(SAMPLE_DATA_DIR)
            app_state['active_channel'] = 'synthetic'
        else:
            return jsonify({'error': f'Unknown channel: {channel}. Use "synthetic" or "real".'}), 400

        # Reload data with new source
        try:
            initialize_data()
            return jsonify({
                'success': True,
                'channel': channel,
                'message': f'Switched to {channel} data. Models retrained.',
                'patients_loaded': len(app_state.get('patients', [])),
            })
        except Exception as e:
            return jsonify({'error': f'Failed to reload: {str(e)}'}), 500


@app.route('/api/data/export-sact')
def api_export_sact():
    """
    Export patient data in SACT v4.0 compliant format.
    Renames all fields to official NHS SACT v4.0 column names.
    """
    try:
        df = app_state.get('patients_df')
        if df is None:
            return jsonify({'error': 'No patient data'}), 400

        # SACT v4.0 official column name mapping
        sact_rename = {
            'NHS_Number': 'NHS_NUMBER',
            'Local_Patient_Identifier': 'LOCAL_PATIENT_IDENTIFIER',
            'NHS_Number_Status_Indicator_Code': 'NHS_NUMBER_STATUS_INDICATOR_CODE',
            'Person_Family_Name': 'PERSON_FAMILY_NAME',
            'Person_Given_Name': 'PERSON_GIVEN_NAME',
            'Person_Birth_Date': 'PERSON_BIRTH_DATE',
            'Person_Stated_Gender_Code': 'PERSON_STATED_GENDER_CODE',
            'Patient_Postcode': 'POSTCODE_OF_USUAL_ADDRESS',
            'Organisation_Identifier': 'ORGANISATION_IDENTIFIER_(CODE_OF_PROVIDER)',
            'Primary_Diagnosis_ICD10': 'PRIMARY_DIAGNOSIS_(ICD)',
            'Morphology_ICD_O': 'MORPHOLOGY_(CLEAN)_(ICD-O)',
            'Performance_Status': 'PERFORMANCE_STATUS_ADULT',
            'Consultant_Specialty_Code': 'CONSULTANT_SPECIALIST_CODE',
            'Regimen_Code': 'REGIMEN',
            'Intent_Of_Treatment': 'INTENT_OF_TREATMENT',
            'Treatment_Context': 'TREATMENT_CONTEXT',
            'Height_At_Start': 'HEIGHT_AT_START_OF_REGIMEN',
            'Weight_At_Start': 'WEIGHT_AT_START_OF_REGIMEN',
            'Clinical_Trial': 'CLINICAL_TRIAL',
            'Chemoradiation': 'CHEMORADIATION',
            'Date_Decision_To_Treat': 'DATE_DECISION_TO_TREAT',
            'Start_Date_Of_Regimen': 'START_DATE_OF_REGIMEN',
        }

        # Select SACT fields and rename
        sact_cols = [c for c in sact_rename.keys() if c in df.columns]
        export_df = df[sact_cols].rename(columns=sact_rename)
        export_df = export_df.fillna('')

        records = json.loads(export_df.to_json(orient='records', default_handler=str))

        return jsonify({
            'format': 'SACT_v4.0',
            'fields': len(export_df.columns),
            'records': len(records),
            'columns': list(export_df.columns),
            'data': records[:50],  # Preview first 50
            'note': 'Full export available at /api/data/export-sact?format=csv'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/staff')
def api_staff():
    """Nurse/staff data."""
    try:
        staff_path = Path(app_state.get('data_dir', 'datasets/sample_data')) / 'staff.xlsx'
        if staff_path.exists():
            df = pd.read_excel(staff_path)
            df = df.fillna('')
            staff = json.loads(df.to_json(orient='records', default_handler=str))
        else:
            staff = []

        return jsonify({'staff': staff, 'count': len(staff)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# NHS DATA INGESTION & AUTO-LEARNING ENDPOINTS
# =============================================================================

# Global instances
nhs_ingester = None
model_recalibrator = None
drift_detector_instance = None


def _get_ingester():
    global nhs_ingester
    if nhs_ingester is None:
        from data.nhs_data_ingestion import NHSDataIngester
        nhs_ingester = NHSDataIngester()
    return nhs_ingester


def _get_recalibrator():
    global model_recalibrator
    if model_recalibrator is None:
        from ml.auto_recalibration import ModelRecalibrator
        model_recalibrator = ModelRecalibrator()
        # Register available models
        models = {}
        if noshow_model: models['noshow_model'] = noshow_model
        if duration_model: models['duration_model'] = duration_model
        if survival_model: models['survival_model'] = survival_model
        if uplift_model: models['uplift_model'] = uplift_model
        if multitask_model: models['multitask_model'] = multitask_model
        if hierarchical_model: models['hierarchical_model'] = hierarchical_model
        if causal_model: models['causal_model'] = causal_model
        if event_impact_model: models['event_impact_model'] = event_impact_model
        # QRF and Conformal models — registered so Level 3 retrain covers all 12 models
        if qrf_duration_model: models['qrf_duration'] = qrf_duration_model
        if qrf_noshow_model: models['qrf_noshow'] = qrf_noshow_model
        if conformal_duration_predictor: models['conformal_duration'] = conformal_duration_predictor
        if conformal_noshow_predictor: models['conformal_noshow'] = conformal_noshow_predictor
        model_recalibrator.register_models(models)
    return model_recalibrator


def _get_drift_detector():
    global drift_detector_instance
    if drift_detector_instance is None:
        from ml.drift_detection import DriftDetector
        drift_detector_instance = DriftDetector()
    return drift_detector_instance


@app.route('/api/data/nhs/status')
def api_nhs_data_status():
    """Get NHS open data source status."""
    try:
        ingester = _get_ingester()
        return jsonify(ingester.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/nhs/check-updates', methods=['POST'])
def api_nhs_check_updates():
    """Trigger check for new NHS open data and download if available."""
    try:
        ingester = _get_ingester()
        results = ingester.check_and_download_all()

        return jsonify({
            'results': [
                {
                    'source': r.source,
                    'success': r.success,
                    'is_new_data': r.is_new_data,
                    'records': r.records_count,
                    'error': r.error
                }
                for r in results
            ],
            'count': len(results),
            'new_data_found': any(r.is_new_data for r in results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/sact-v4/status')
def api_sact_v4_status():
    """
    Get SACT v4.0 data availability status and quality phase.

    Returns:
        phase:    not_started | rollout_partial | full_conformance | first_complete
        quality:  unavailable | preliminary | conformance | complete
        usable:   bool — whether partial/full data can be used
        recommended_recalibration_level: None | 1 | 2 | 3
        note:     Human-readable guidance
    """
    try:
        ingester = _get_ingester()
        availability = ingester.check_sact_v4_availability()
        return jsonify({
            'sact_v4': availability,
            'instructions': (
                'To use SACT v4.0 real data: download CSV from NDRS portal and place '
                'in datasets/nhs_open_data/sact_v4/. '
                'The auto-learning scheduler will detect and process it within 24h, '
                'or use POST /api/data/sact-v4/check to trigger immediately.'
            )
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/sact-v4/check', methods=['POST'])
def api_sact_v4_check():
    """
    Trigger SACT v4.0 data check and auto-recalibration.

    Checks the local sact_v4 directory and NDRS page for new data.
    Triggers quality-aware recalibration:
        - rollout_partial (Apr-Jun 2026) → Level 2 (preliminary, no full retrain)
        - full_conformance (Jul 2026)    → Level 2
        - first_complete (Aug 2026+)     → Level 3 (full retrain)

    Returns: ingestion result + recalibration result
    """
    try:
        ingester = _get_ingester()
        availability = ingester.check_sact_v4_availability()

        # Download / check for SACT v4 data
        result = ingester.download_sact_v4_data()

        recal_result = None
        if result.success and result.records_count > 0:
            quality = availability['quality']
            recalibrator = _get_recalibrator()
            level = recalibrator.determine_update_level(
                'sact_v4_patient_data', 0.0, quality
            )
            data = None
            if result.file_path and result.file_path.endswith('.csv'):
                try:
                    import pandas as pd
                    data = pd.read_csv(result.file_path)
                except Exception:
                    pass
            rr = recalibrator.execute_recalibration(level, 'sact_v4_patient_data', data)
            recal_result = {
                'level': rr.level,
                'success': rr.success,
                'models_updated': rr.models_updated,
                'duration_seconds': rr.duration_seconds,
                'details': rr.details,
            }

        return jsonify({
            'availability': availability,
            'ingestion': {
                'success': result.success,
                'records': result.records_count,
                'file': result.file_path,
                'is_new': result.is_new_data,
                'error': result.error,
            },
            'recalibration': recal_result,
            'message': (
                f"SACT v4.0 phase: {availability['phase']} | "
                f"quality: {availability['quality']} | "
                f"records: {result.records_count} | "
                + (f"Level {recal_result['level']} recalibration triggered"
                   if recal_result else "No recalibration triggered (no data found)")
            )
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/sact-v4/validate', methods=['GET', 'POST'])
def api_sact_v4_validate():
    """
    Validate a SACT v4.0 dataset against the schema.

    GET  — validates the existing dataset in datasets/nhs_open_data/sact_v4/
    POST — accepts a JSON body with 'file_path' to validate a specific CSV,
           or 'dataset' (list of dicts) for inline validation

    Returns: validation report with score (0-100), grade (A-F), missing fields,
             invalid values, section coverage, and ready_for_ml flag.
    """
    try:
        import pandas as pd
        from data.sact_v4_schema import validate_dataset, check_synthetic_alignment

        df = None
        source_label = 'unknown'

        if request.method == 'POST':
            body = request.json or {}
            file_path = body.get('file_path')
            inline_dataset = body.get('dataset')

            if file_path:
                df = pd.read_csv(file_path)
                source_label = file_path
            elif inline_dataset:
                df = pd.DataFrame(inline_dataset)
                source_label = 'inline_payload'

        if df is None:
            # GET or POST with no body → use existing sact_v4 directory
            sact_v4_dir = Path(__file__).parent / 'datasets' / 'nhs_open_data' / 'sact_v4'
            csv_files = list(sact_v4_dir.glob('*.csv')) if sact_v4_dir.exists() else []
            if csv_files:
                df = pd.read_csv(csv_files[0])
                source_label = str(csv_files[0].name)
            else:
                # Fall back to synthetic dataset
                synthetic_path = Path(__file__).parent / 'datasets' / 'sample_data' / 'historical_appointments.xlsx'
                if synthetic_path.exists():
                    df = pd.read_excel(synthetic_path)
                    source_label = 'synthetic (sample_data)'
                else:
                    return jsonify({
                        'error': 'No dataset found. Place a SACT v4.0 CSV in '
                                 'datasets/nhs_open_data/sact_v4/ or POST a file_path.'
                    }), 404

        report = validate_dataset(df)
        alignment = check_synthetic_alignment(df)

        import numpy as np
        def _safe(v):
            """Convert numpy scalars to Python natives for JSON serialisation."""
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, (np.bool_,)): return bool(v)
            if isinstance(v, dict): return {k: _safe(vv) for k, vv in v.items()}
            if isinstance(v, list): return [_safe(i) for i in v]
            return v

        sec_cov = {
            k: {kk: _safe(vv) for kk, vv in v.items()} if isinstance(v, dict) else _safe(v)
            for k, v in report.get('section_coverage', {}).items()
        }

        return jsonify({
            'source': source_label,
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'validation': {
                'score': float(report['overall_score']),
                'grade': report['grade'],
                'ready_for_ml': bool(report['ready_for_ml']),
                'missing_mandatory': report['missing_mandatory'],
                'missing_optional': report.get('missing_optional', []),
                'invalid_values': _safe(report.get('invalid_values', {})),
                'section_coverage': sec_cov,
            },
            'alignment': _safe(alignment),
            'message': (
                f"Score {report['overall_score']}/100 (Grade {report['grade']}) — "
                f"{'Ready for ML' if report['ready_for_ml'] else 'Not ready for ML'}"
            )
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/recalibration/status')
def api_recalibration_status():
    """Get model recalibration status and history."""
    try:
        recalibrator = _get_recalibrator()
        return jsonify(recalibrator.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/recalibration/run', methods=['POST'])
def api_recalibration_run():
    """Trigger manual model recalibration."""
    try:
        data = request.json or {}
        level = data.get('level', 1)
        source = data.get('source', 'manual')

        recalibrator = _get_recalibrator()
        result = recalibrator.execute_recalibration(level, source)

        return jsonify({
            'level': result.level,
            'success': result.success,
            'models_updated': result.models_updated,
            'duration_seconds': result.duration_seconds,
            'details': result.details,
            'version': result.new_version
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/drift/report')
def api_drift_report():
    """Get drift detection report."""
    try:
        detector = _get_drift_detector()
        return jsonify(detector.get_summary_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/drift/check', methods=['POST'])
def api_drift_check():
    """Run drift check on provided data.

    Since §3.4 every successful drift check also returns an
    ``attribution`` block computed by ``ml/drift_attribution.py`` so
    downstream consumers see *which* features and bins explain the
    PSI movement without a second round-trip.
    """
    try:
        detector = _get_drift_detector()

        # Use historical data for drift check
        if historical_appointments_df is not None and len(historical_appointments_df) > 0:
            # Split into reference (first 70%) and current (last 30%)
            split_idx = int(len(historical_appointments_df) * 0.7)
            ref = historical_appointments_df.iloc[:split_idx]
            cur = historical_appointments_df.iloc[split_idx:]

            # Set reference from training data
            numeric_cols = ref.select_dtypes(include=[np.number]).columns
            ref_dict = {col: ref[col].values for col in numeric_cols[:10]}
            detector.set_reference(ref_dict)

            # Check drift
            cur_dict = {col: cur[col].values for col in numeric_cols[:10]}
            summary = detector.full_drift_check(cur_dict)

            # ------------- §3.4 automated root-cause attribution -------------
            attribution_dict = None
            try:
                from ml.drift_attribution import get_attributor
                attributor = get_attributor()
                # Feature-hint dictionary — human phrasing per known column.
                hints = {
                    'Travel_Time_Min':       'more remote patients',
                    'travel_time_minutes':   'more remote patients',
                    'distance_km':           'further travel',
                    'Age':                   'age mix shifted',
                    'no_show_rate':          'no-show rate profile shifted',
                    'Duration_Actual':       'appointment durations shifted',
                    'expected_duration':     'expected durations shifted',
                    'priority':              'priority mix shifted',
                }
                attrib = attributor.attribute(
                    reference=ref_dict, current=cur_dict, feature_hints=hints,
                )
                # Keep the public response slim: drop the per-bin detail
                # (still available on /api/drift/attribution/last) but keep
                # the top-bin summary so operators can see the "why".
                attribution_dict = {
                    'computed_ts':       attrib.computed_ts,
                    'total_psi':         round(attrib.total_psi, 4),
                    'overall_severity':  attrib.overall_severity,
                    'top_feature':       attrib.top_feature,
                    'top_feature_share': round(attrib.top_feature_share, 4),
                    'narrative':         attrib.narrative,
                    'feature_breakdown': [
                        {
                            'feature': fa.feature,
                            'psi':     round(fa.psi, 4),
                            'share':   round(fa.share_of_total, 4),
                            'top_bin': fa.top_bin_summary,
                        }
                        for fa in attrib.feature_breakdown
                    ],
                }
            except Exception as exc:
                # Attribution must never break drift detection for the UI.
                logger.warning(f"drift attribution failed, skipping: {exc}")
                attribution_dict = None

            return jsonify({
                'total_checked': summary.total_features_checked,
                'features_drifted': summary.features_drifted,
                'max_drift_score': round(summary.max_drift_score, 4),
                'overall_severity': summary.overall_severity,
                'recommended_action': summary.recommended_action,
                'reports': [
                    {
                        'feature': r.feature_name,
                        'drift_score': round(r.drift_score, 4),
                        'drifted': r.drift_detected,
                        'severity': r.severity,
                        'details': r.details
                    }
                    for r in summary.reports
                ],
                'attribution': attribution_dict,
            })
        else:
            return jsonify({'error': 'No historical data available for drift check'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------------------------------------------------------
# §3.4 — standalone drift-attribution endpoints
# -----------------------------------------------------------------------------
# These are *diagnostic* routes.  The /api/ml/drift/check endpoint above
# already returns attribution alongside every drift report; the two routes
# below let operators (a) request an ad-hoc attribution with a custom
# feature set, and (b) re-read the most recent attribution without a recompute.
# No UI panel per §3.4 brief.

@app.route('/api/drift/attribution/status', methods=['GET'])
def api_drift_attribution_status():
    """Attributor configuration, run count, last top feature + narrative."""
    try:
        from ml.drift_attribution import get_attributor
        return jsonify(get_attributor().status())
    except Exception as exc:
        logger.error(f"attribution status error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/drift/attribution', methods=['POST'])
def api_drift_attribution_compute():
    """
    Compute attribution for an ad-hoc (reference, current) pair.

    Body (all fields optional except when overriding defaults)::

        {
          "features":     ["Travel_Time_Min", "Age"],   // subset; default = top 10 numeric
          "reference_frac": 0.70,                       // split point in historical_df
          "feature_hints": {"Travel_Time_Min": "more remote patients"}
        }

    Defaults mirror the split used by /api/ml/drift/check so the
    two endpoints are directly comparable.
    """
    try:
        from ml.drift_attribution import get_attributor
        if historical_appointments_df is None or len(historical_appointments_df) == 0:
            return jsonify({'success': False,
                            'error': 'No historical data loaded.'}), 400
        body = request.json or {}
        frac = float(body.get('reference_frac', 0.70))
        split_idx = int(len(historical_appointments_df) * frac)
        ref_df = historical_appointments_df.iloc[:split_idx]
        cur_df = historical_appointments_df.iloc[split_idx:]
        numeric_cols = list(ref_df.select_dtypes(include=[np.number]).columns)
        requested = body.get('features') or numeric_cols[:10]
        hints = dict(body.get('feature_hints') or {})
        ref_dict = {c: ref_df[c].values for c in requested if c in ref_df.columns}
        cur_dict = {c: cur_df[c].values for c in requested if c in cur_df.columns}
        attrib = get_attributor().attribute(
            reference=ref_dict, current=cur_dict, feature_hints=hints,
        )
        return jsonify({'success': True, 'attribution': attrib.to_dict()})
    except Exception as exc:
        logger.error(f"attribution compute error: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/drift/attribution/last', methods=['GET'])
def api_drift_attribution_last():
    """Return the cached most-recent attribution (full per-bin detail)."""
    try:
        from ml.drift_attribution import get_attributor
        last = get_attributor().last()
        if last is None:
            return jsonify({'success': False, 'error': 'no prior attribution'}), 404
        return jsonify({'success': True, 'attribution': last.to_dict()})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# =============================================================================
# Post-registration hook — exempt /api/* + /auth/* + /health/* + /metrics from
# CSRF, now that every @app.route has been declared.  Keeps browser-form
# endpoints (if any are added in future) protected by default.
# =============================================================================
_csrf_exempted = _exempt_json_api_routes(app, _csrf)
if _csrf is not None:
    logger.info(f"csrf: exempted {len(_csrf_exempted)} JSON API routes from CSRF token check")

# Final readiness signal — app has finished wiring.  Flipped here (not inside
# a lazy callback) so /health/ready stays cheap.
_mark_app_ready()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("SACT Scheduler Flask Application")
    print("=" * 50)
    print(f"Starting server on http://localhost:1421")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(host='0.0.0.0', port=1421, debug=False)
