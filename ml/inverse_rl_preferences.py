"""
Inverse RL Preference Learner (Dissertation §1.4)
=================================================

Learns clinician Pareto weights θ over the 6 CP-SAT objectives
(priority, utilization, noshow_risk, waiting_time, robustness, travel)
from historical manual overrides of optimizer-proposed schedules.

Choice model (pairwise softmax / Bradley–Terry):

    P(manual ≻ proposal | θ) = σ( θ · ( Z(manual) − Z(proposal) ) )

where Z(·) ∈ R^6 is the per-objective feature vector of a schedule
(same expressions as the scalarised CP-SAT objective, without weights).

Training objective (maximum likelihood):

    θ* = argmax_{θ ≥ 0}   Σ_i  log σ( θ · ΔZ_i )  −  λ‖θ‖²

Non-negativity preserves monotonicity of each objective; an L2 prior
keeps θ close to OPTIMIZATION_WEIGHTS when data are scarce.  The raw
θ is then rescaled by the per-feature stddev σ_Z and normalised to sum
to 1 so it is interchangeable with the existing fixed-weight vector.

Persistence: JSONL override log + pickled model state so the Flask
process can refit incrementally without retaining schedule payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import pickle

import numpy as np

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_CACHE_DIR,
    MODELS_DIR,
    OPTIMIZATION_WEIGHTS,
    get_logger,
)

logger = get_logger(__name__)


# The canonical ordering used everywhere — MUST match the optimiser's
# scalarised objective order in optimization/optimizer.py.
OBJECTIVE_KEYS: Tuple[str, ...] = (
    'priority',
    'utilization',
    'noshow_risk',
    'waiting_time',
    'robustness',
    'travel',
)

IRL_OVERRIDE_LOG: Path = DATA_CACHE_DIR / 'irl_overrides.jsonl'
IRL_MODEL_FILE: Path = MODELS_DIR / 'inverse_rl_preferences.pkl'
IRL_HISTORY_FILE: Path = DATA_CACHE_DIR / 'irl_weights_history.jsonl'

# Number of synthetic overrides to auto-seed when the real log is empty.
# Chosen so the IRL fit is well-posed without dominating real data once
# actual clinician overrides start streaming in.
BOOTSTRAP_N_DEFAULT: int = 200

# L2 prior strength pulling θ toward uniform — tuned so that with 20
# real overrides the prior contributes ≈ 1 effective sample.
L2_LAMBDA_DEFAULT: float = 0.05


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveFeatures:
    """Feature vector Z ∈ R^6 for a candidate schedule."""
    priority: float = 0.0
    utilization: float = 0.0
    noshow_risk: float = 0.0
    waiting_time: float = 0.0
    robustness: float = 0.0
    travel: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in OBJECTIVE_KEYS], dtype=float)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> 'ObjectiveFeatures':
        return cls(**{k: float(arr[i]) for i, k in enumerate(OBJECTIVE_KEYS)})


@dataclass
class OverrideRecord:
    """
    A single clinician override event (used as IRL training pair).

    Two orthogonal tags — channel numbering follows README.md §4:
      source  — provenance of the OVERRIDE ROW itself:
                 'real'      = logged by a clinician acting on live data
                              (valid training signal regardless of which
                              data channel is active — Channel 1 synthetic
                              IS the operational data until Channel 2 real
                              hospital data arrives in datasets/real_data/)
                 'synthetic' = produced by the bootstrap seed for cold-start
      channel — which of the 3 data channels was active at log time:
                 'synthetic' → Channel 1 (datasets/sample_data/, default today)
                 'real'      → Channel 2 (datasets/real_data/, hospital export,
                               promoted automatically by the Ch2 watcher when
                               patients.xlsx is dropped into the folder)
                 'nhs'       → Channel 3 (datasets/nhs_open_data/, NHS open
                               aggregates — partially available, always
                               running in background for recalibration)
    """
    ts: str
    z_proposed: List[float]
    z_manual: List[float]
    site_code: Optional[str] = None
    reason: Optional[str] = None
    source: str = 'real'              # 'real' | 'synthetic'
    channel: Optional[str] = None     # 'synthetic' | 'real' | 'nhs' | None

    def delta(self) -> np.ndarray:
        return np.asarray(self.z_manual, dtype=float) - np.asarray(self.z_proposed, dtype=float)


@dataclass
class IRLFitResult:
    """Summary of a single IRL training run."""
    theta_raw: List[float]                 # learned weights in standardised feature space
    theta_weights: Dict[str, float]        # normalised weights in objective space (sum to 1)
    prior_weights: Dict[str, float]        # what the optimiser was using before fitting
    n_samples: int
    n_real: int
    n_synthetic: int
    log_likelihood: float
    mean_agreement: float                  # P(θ·ΔZ > 0) on training data
    converged: bool
    l2_lambda: float
    fit_ts: str
    feature_std: List[float]               # per-feature stddev used for scaling
    training_mode: str = 'bootstrap'       # 'bootstrap' | 'real_only' | 'mixed'
    channel_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction — same expressions as the CP-SAT objective
# ---------------------------------------------------------------------------


def compute_objective_features(
    patients: Sequence,
    assignments: Dict[str, Dict],
    robust_noshow_penalties: Optional[Dict[str, float]] = None,
) -> ObjectiveFeatures:
    """
    Compute Z(schedule) matching the scalarised CP-SAT objective terms
    (see optimization/optimizer.py lines ~615-678).

    Args:
        patients: iterable of Patient-like objects (Patient dataclass OR dict)
                  with attributes: patient_id, priority, expected_duration,
                  travel_time_minutes (or travel_time), noshow_probability,
                  days_waiting (optional).
        assignments: mapping patient_id → {'start': int (minutes-from-day-start),
                     'chair_id': str, 'assigned': bool}.  Unassigned patients
                     may be omitted or have assigned=False.
        robust_noshow_penalties: optional DRO-robust per-patient penalty
                     (already in integer percentage points).

    Returns:
        ObjectiveFeatures whose components have the same sign as the
        optimiser objective (higher = better for every component).
    """
    z = {k: 0.0 for k in OBJECTIVE_KEYS}
    robust_noshow_penalties = robust_noshow_penalties or {}

    for p in patients:
        pid = _field(p, 'patient_id')
        assign = assignments.get(pid, {}) if pid else {}
        is_assigned = bool(assign.get('assigned', bool(assign))) if assign else False

        priority = int(_field(p, 'priority', 3))
        duration = int(_field(p, 'expected_duration', 90))
        days_waiting = int(_field(p, 'days_waiting', 14))
        noshow_prob = float(_field(p, 'noshow_probability', 0.15))
        travel_minutes = float(
            _field(p, 'travel_time_minutes', _field(p, 'travel_time', 30))
        )

        # Objective 1: priority-weighted assignment (only counts if assigned)
        if is_assigned:
            z['priority'] += (5 - priority) * 100.0

        # Objective 2: utilisation = negative start time (prefer earlier)
        start = float(assign.get('start', 0)) if is_assigned else 0.0
        z['utilization'] += -start

        # Objective 3: no-show risk (negative penalty, only if assigned)
        if is_assigned:
            if pid in robust_noshow_penalties:
                noshow_penalty = float(robust_noshow_penalties[pid])
            else:
                noshow_penalty = float(int(noshow_prob * 100))
            z['noshow_risk'] += -noshow_penalty

        # Objective 4: waiting-time bonus (only if assigned)
        if is_assigned:
            waiting_bonus = min(days_waiting, 62) * 5.0
            z['waiting_time'] += waiting_bonus

        # Objective 5: robustness = negative over-length penalty (global)
        duration_risk = max(0, duration - 120) // 30
        z['robustness'] += -float(duration_risk)

        # Objective 6: travel penalty (global — matches optimiser)
        travel_penalty = int(travel_minutes) // 10
        z['travel'] += -float(travel_penalty)

    return ObjectiveFeatures(**z)


def _field(obj, key: str, default=None):
    """Uniform attribute/dict accessor."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# IRL learner
# ---------------------------------------------------------------------------


class InverseRLPreferenceLearner:
    """
    Learns Pareto weights θ ∈ R^6 from clinician overrides.

    Usage:
        learner = InverseRLPreferenceLearner()
        learner.log_override(z_proposed=[...], z_manual=[...])
        result  = learner.fit(bootstrap_if_empty=True)
        weights = learner.to_optimizer_weights()   # dict sum=1, for ScheduleOptimizer
    """

    def __init__(
        self,
        override_log_path: Path = IRL_OVERRIDE_LOG,
        model_path: Path = IRL_MODEL_FILE,
        history_path: Path = IRL_HISTORY_FILE,
        prior_weights: Optional[Dict[str, float]] = None,
        l2_lambda: float = L2_LAMBDA_DEFAULT,
    ) -> None:
        self.override_log_path = Path(override_log_path)
        self.model_path = Path(model_path)
        self.history_path = Path(history_path)
        self.prior_weights = dict(prior_weights or OPTIMIZATION_WEIGHTS)
        self.l2_lambda = float(l2_lambda)

        self.theta_raw: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.last_fit: Optional[IRLFitResult] = None

        self.override_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self._try_load_state()

    # -------- override log I/O --------

    def log_override(
        self,
        z_proposed: Sequence[float],
        z_manual: Sequence[float],
        site_code: Optional[str] = None,
        reason: Optional[str] = None,
        source: str = 'real',
        channel: Optional[str] = None,
    ) -> OverrideRecord:
        """Append a clinician-override event to the JSONL log."""
        record = OverrideRecord(
            ts=datetime.utcnow().isoformat(timespec='seconds'),
            z_proposed=[float(x) for x in z_proposed],
            z_manual=[float(x) for x in z_manual],
            site_code=site_code,
            reason=reason,
            source=source,
            channel=channel,
        )
        with self.override_log_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(asdict(record)) + '\n')
        return record

    def load_overrides(self) -> List[OverrideRecord]:
        if not self.override_log_path.exists():
            return []
        out: List[OverrideRecord] = []
        with self.override_log_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(OverrideRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(f"Skipping bad IRL log line: {exc}")
        return out

    def clear_overrides(self) -> None:
        """Wipe the override log (used on reset)."""
        if self.override_log_path.exists():
            self.override_log_path.unlink()

    # -------- synthetic bootstrap --------

    @staticmethod
    def _bootstrap_overrides(
        n: int = BOOTSTRAP_N_DEFAULT,
        true_theta: Optional[Sequence[float]] = None,
        seed: int = 42,
    ) -> List[OverrideRecord]:
        """
        Generate n synthetic (z_proposed, z_manual) pairs consistent with a
        latent "truth" θ* drawn from a plausible Velindre clinician profile
        (leaning toward no-show aversion and waiting-time sensitivity).

        The pairs are constructed so that under θ* the manual schedule is
        preferred with probability ≥ 0.5 — i.e. real clinician overrides
        improve the objective as perceived through θ*.
        """
        rng = np.random.RandomState(seed)
        true = (
            np.asarray(true_theta, dtype=float)
            if true_theta is not None
            else np.array([0.35, 0.10, 0.25, 0.20, 0.05, 0.05])
        )
        true = true / true.sum()

        # Feature scale roughly proportional to real Z magnitudes encountered
        # in a single-day Velindre instance (30–60 patients).
        scale = np.array([2000.0, 400.0, 800.0, 300.0, 20.0, 60.0])

        out: List[OverrideRecord] = []
        for i in range(n):
            z_prop = rng.normal(0.0, 1.0, size=6) * scale
            # Δ in a direction that (on expectation) increases θ*·Δ
            delta = rng.normal(0.0, 0.4, size=6) * scale
            # Push Δ toward the θ* gradient half the time so manual ≻ proposal
            if rng.rand() > 0.15:
                delta += true * scale * abs(rng.normal(0.6, 0.3))
            z_man = z_prop + delta
            out.append(OverrideRecord(
                ts=datetime.utcnow().isoformat(timespec='seconds'),
                z_proposed=z_prop.tolist(),
                z_manual=z_man.tolist(),
                site_code=None,
                reason='synthetic_bootstrap',
                source='synthetic',
                channel='synthetic',
            ))
        return out

    def seed_bootstrap(
        self,
        n: int = BOOTSTRAP_N_DEFAULT,
        true_theta: Optional[Sequence[float]] = None,
    ) -> int:
        """Append synthetic bootstrap overrides to the log. Returns count."""
        synth = self._bootstrap_overrides(n=n, true_theta=true_theta)
        with self.override_log_path.open('a', encoding='utf-8') as fh:
            for rec in synth:
                fh.write(json.dumps(asdict(rec)) + '\n')
        return len(synth)

    # -------- fitting --------

    def fit(
        self,
        bootstrap_if_empty: bool = True,
        min_real_overrides: int = 20,
        prefer_real: bool = True,
    ) -> IRLFitResult:
        """
        Fit θ via MLE on pairwise softmax.

        Training-mix logic (critical):
          * If real overrides  >= min_real_overrides AND prefer_real=True:
                train on REAL records only — synthetic bootstrap pairs are
                ignored (they remain on disk for audit traceability).
                training_mode = 'real_only'
          * Else if real overrides < min_real_overrides AND bootstrap_if_empty:
                seed synthetic pairs up to the minimum, then train on the
                combined pool.  training_mode = 'bootstrap' or 'mixed'
          * Else:
                train on whatever is present.
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for InverseRLPreferenceLearner.fit()")

        records_all = self.load_overrides()
        n_real_all = sum(1 for r in records_all if r.source == 'real')
        n_synth_all = sum(1 for r in records_all if r.source == 'synthetic')

        if prefer_real and n_real_all > 0 and n_real_all >= min_real_overrides:
            # Real clinician overrides dominate — drop synthetic bootstrap.
            records = [r for r in records_all if r.source == 'real']
            training_mode = 'real_only'
        else:
            if bootstrap_if_empty and n_real_all < min_real_overrides:
                n_needed = max(BOOTSTRAP_N_DEFAULT, min_real_overrides - n_real_all)
                if n_synth_all < n_needed:
                    self.seed_bootstrap(n=n_needed - n_synth_all)
                    records_all = self.load_overrides()
                    n_real_all = sum(1 for r in records_all if r.source == 'real')
                    n_synth_all = sum(1 for r in records_all if r.source == 'synthetic')
            records = list(records_all)
            training_mode = 'bootstrap' if n_real_all == 0 else 'mixed'

        if len(records) < 2:
            raise ValueError("Need at least 2 override records to fit IRL.")

        # Per-channel counts for traceability + UI
        channel_counts: Dict[str, int] = {}
        for r in records:
            key = r.channel or 'unknown'
            channel_counts[key] = channel_counts.get(key, 0) + 1

        deltas = np.vstack([r.delta() for r in records])  # (N, 6)

        # Per-feature standardisation so the optimiser sees comparable scales.
        std = deltas.std(axis=0)
        std[std < 1e-9] = 1.0
        dz = deltas / std

        lam = self.l2_lambda

        def neg_log_lik(theta: np.ndarray) -> float:
            s = dz @ theta
            # log σ(s) = -log(1 + exp(-s)); stable form
            nll = float(np.sum(np.logaddexp(0.0, -s)))
            reg = lam * float(np.sum(theta ** 2))
            return nll + reg

        def grad(theta: np.ndarray) -> np.ndarray:
            s = dz @ theta
            # d/dθ log σ(s) = σ(-s) * z  (positive-log version: minimise NLL)
            # NLL gradient: Σ -σ(-s) * dz_i   →   Σ (σ(s) - 1) * dz_i
            sig = 1.0 / (1.0 + np.exp(-s))
            g = -dz.T @ (1.0 - sig)
            g += 2.0 * lam * theta
            return g

        theta0 = np.ones(6)
        bounds = [(0.0, None)] * 6  # θ ≥ 0 keeps objectives monotone
        try:
            res = minimize(
                neg_log_lik, theta0, jac=grad, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'gtol': 1e-7},
            )
            converged = bool(res.success)
            theta_star = np.asarray(res.x, dtype=float)
        except Exception as exc:  # pragma: no cover
            logger.error(f"IRL fit failed: {exc}")
            converged = False
            theta_star = np.ones(6)

        # Convert from standardised space back to objective space: θ_raw = θ/σ
        theta_obj = theta_star / std
        if theta_obj.sum() <= 0:
            theta_obj = np.asarray([self.prior_weights[k] for k in OBJECTIVE_KEYS])
        theta_norm = theta_obj / theta_obj.sum()

        # Metrics
        ll = -neg_log_lik(theta_star)
        agreement = float(np.mean((dz @ theta_star) > 0.0))

        self.theta_raw = theta_star
        self.feature_std = std
        n_real_train = sum(1 for r in records if r.source == 'real')
        fit = IRLFitResult(
            theta_raw=theta_star.tolist(),
            theta_weights={k: float(theta_norm[i]) for i, k in enumerate(OBJECTIVE_KEYS)},
            prior_weights=dict(self.prior_weights),
            n_samples=len(records),
            n_real=n_real_train,
            n_synthetic=len(records) - n_real_train,
            log_likelihood=float(ll),
            mean_agreement=agreement,
            converged=converged,
            l2_lambda=lam,
            fit_ts=datetime.utcnow().isoformat(timespec='seconds'),
            feature_std=std.tolist(),
            training_mode=training_mode,
            channel_counts=channel_counts,
        )
        self.last_fit = fit
        self._save_state()
        self._append_history(fit)
        logger.info(
            f"IRL fit [{training_mode}]: N={fit.n_samples} "
            f"({fit.n_real} real / {fit.n_synthetic} synth, "
            f"channels={channel_counts}) LL={fit.log_likelihood:.2f} "
            f"agree={fit.mean_agreement:.3f} θ={fit.theta_weights}"
        )
        return fit

    # -------- use --------

    def to_optimizer_weights(self) -> Dict[str, float]:
        """Return learned weights as a dict consumable by ScheduleOptimizer."""
        if self.last_fit is None:
            return dict(self.prior_weights)
        return dict(self.last_fit.theta_weights)

    def predict_preference(
        self,
        z_a: Sequence[float],
        z_b: Sequence[float],
    ) -> float:
        """
        P(schedule_a ≻ schedule_b | θ) under the learned choice model.
        Returns 0.5 if not yet fit.
        """
        if self.theta_raw is None or self.feature_std is None:
            return 0.5
        dz = (np.asarray(z_a, dtype=float) - np.asarray(z_b, dtype=float)) / self.feature_std
        s = float(dz @ self.theta_raw)
        return 1.0 / (1.0 + np.exp(-s))

    # -------- persistence --------

    def _save_state(self) -> None:
        payload = {
            'theta_raw': None if self.theta_raw is None else self.theta_raw.tolist(),
            'feature_std': None if self.feature_std is None else self.feature_std.tolist(),
            'last_fit': asdict(self.last_fit) if self.last_fit else None,
            'prior_weights': self.prior_weights,
            'l2_lambda': self.l2_lambda,
        }
        # T2.3: SHA-256 sidecar — refuses tampered IRL preference vectors.
        from safe_loader import safe_save
        safe_save(payload, self.model_path)

    def _try_load_state(self) -> None:
        if not self.model_path.exists():
            return
        try:
            from safe_loader import safe_load
            payload = safe_load(self.model_path)
            self.theta_raw = (
                np.asarray(payload['theta_raw'], dtype=float)
                if payload.get('theta_raw') is not None else None
            )
            self.feature_std = (
                np.asarray(payload['feature_std'], dtype=float)
                if payload.get('feature_std') is not None else None
            )
            if payload.get('last_fit'):
                self.last_fit = IRLFitResult(**payload['last_fit'])
            if 'prior_weights' in payload:
                self.prior_weights = dict(payload['prior_weights'])
            if 'l2_lambda' in payload:
                self.l2_lambda = float(payload['l2_lambda'])
        except Exception as exc:
            logger.warning(f"Could not load IRL state: {exc}")

    def _append_history(self, fit: IRLFitResult) -> None:
        """Append a compact fit row to irl_weights_history.jsonl for R analysis."""
        row = {
            'ts': fit.fit_ts,
            'n_samples': fit.n_samples,
            'n_real': fit.n_real,
            'n_synthetic': fit.n_synthetic,
            'log_likelihood': fit.log_likelihood,
            'mean_agreement': fit.mean_agreement,
            'converged': fit.converged,
            'training_mode': fit.training_mode,
            **{f'ch_{k}': v for k, v in fit.channel_counts.items()},
            **{f'w_{k}': fit.theta_weights[k] for k in OBJECTIVE_KEYS},
            **{f'prior_{k}': fit.prior_weights[k] for k in OBJECTIVE_KEYS},
        }
        try:
            with self.history_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(row) + '\n')
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not append IRL history: {exc}")


__all__ = [
    'OBJECTIVE_KEYS',
    'ObjectiveFeatures',
    'OverrideRecord',
    'IRLFitResult',
    'InverseRLPreferenceLearner',
    'compute_objective_features',
]
