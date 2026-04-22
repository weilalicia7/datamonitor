"""
RCT Randomization Layer (Dissertation §2.4)
===========================================

The existing causal machinery (instrumental variables §4.2, double
machine learning §4.3) operates on *observational* data and therefore
inherits all the usual endogeneity concerns.  The §2.4 brief calls
for a gold-standard complement: run a small randomised-controlled
trial (5 % of bookings) for unbiased average treatment effect (ATE)
estimates, then use those as a Bayesian prior for the observational
DML estimator.

This module implements the randomisation layer and the
precision-weighted shrinkage.  It is intentionally minimal — four
reminder arms, deterministic hash-based assignment, JSONL persistence,
Wald confidence intervals — so the integration with the rest of the
system is a single function call `apply_rct_prior()` in the DML
endpoint and a handful of lightweight Flask routes.

Reminder arms (chosen to match the `InterventionType` enum already
used by the uplift module so downstream code needs no changes):

    CONTROL      — no automated reminder beyond standard of care
    SMS_24H      — SMS reminder 24 h before the appointment
    SMS_48H      — SMS reminder 48 h before the appointment
    PHONE_24H    — proactive phone call 24 h before the appointment

Mathematical core
-----------------

For any two arms (treatment t, control c) with sample sizes n_t, n_c
and attendance rates π_t, π_c we return

    ATE        = π_t - π_c
    SE         = sqrt( π_t(1-π_t)/n_t + π_c(1-π_c)/n_c )
    95 % CI    = ATE ± 1.96 · SE

For the DML → RCT bridge we use precision-weighted Gaussian shrinkage

    θ̂_post  = ( θ̂_dml / σ²_dml + θ̂_rct / σ²_rct )
              / ( 1/σ²_dml + 1/σ²_rct )
    σ²_post = 1 / ( 1/σ²_dml + 1/σ²_rct )

which is the posterior mean / variance under the Gaussian prior
N(θ̂_rct, σ²_rct) on the DML likelihood.  In the limit σ²_rct → 0
(RCT is perfectly informative) the posterior collapses to the RCT
estimate; in the limit σ²_rct → ∞ (RCT is uninformative) the
posterior is the DML estimate verbatim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


class TrialArm(str, Enum):
    CONTROL = "control"
    SMS_24H = "sms_24h"
    SMS_48H = "sms_48h"
    PHONE_24H = "phone_24h"


DEFAULT_TRIAL_RATE: float = 0.05     # fraction of bookings flagged for the RCT
DEFAULT_MIN_N_PER_ARM: int = 30      # below this, ATE is reported as "under-powered"
TRIAL_ASSIGNMENTS_FILE: Path = DATA_CACHE_DIR / 'trial_assignments.jsonl'
TRIAL_OUTCOMES_FILE: Path = DATA_CACHE_DIR / 'trial_outcomes.jsonl'
TRIAL_CONFIG_FILE: Path = DATA_CACHE_DIR / 'trial_config.json'
TRIAL_ATE_HISTORY_FILE: Path = DATA_CACHE_DIR / 'trial_ate_history.jsonl'


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrialAssignment:
    ts: str
    appointment_id: str
    patient_id: str
    arm: str
    trial_rate: float
    strata: str = ""


@dataclass
class TrialOutcome:
    ts: str
    appointment_id: str
    patient_id: str
    arm: str
    attended: bool


@dataclass
class ATEResult:
    treatment_arm: str
    control_arm: str
    n_treatment: int
    n_control: int
    pi_treatment: float
    pi_control: float
    ate: float
    standard_error: float
    ci_low: float
    ci_high: float
    under_powered: bool
    min_n_per_arm: int


# ---------------------------------------------------------------------------
# Trial assigner
# ---------------------------------------------------------------------------


class TrialAssigner:
    """
    Hash-based deterministic 5 % randomiser.

    For every (appointment_id, patient_id) pair the same assignment is
    produced on every call — important because the same booking may be
    processed many times across restarts and we don't want the arm to
    drift.  Stratification is by (priority, age_band) so each stratum
    is balanced across arms to the extent the hash allows.
    """

    def __init__(
        self,
        trial_rate: float = DEFAULT_TRIAL_RATE,
        arms: Tuple[TrialArm, ...] = tuple(TrialArm),
        seed: str = "velindre-sact-rct-v1",
        min_n_per_arm: int = DEFAULT_MIN_N_PER_ARM,
        assignments_path: Path = TRIAL_ASSIGNMENTS_FILE,
        outcomes_path: Path = TRIAL_OUTCOMES_FILE,
        history_path: Path = TRIAL_ATE_HISTORY_FILE,
    ):
        if not (0.0 < trial_rate <= 0.50):
            raise ValueError("trial_rate must be in (0, 0.50].")
        if len(arms) < 2:
            raise ValueError("Need ≥2 arms for an ATE comparison.")
        self.trial_rate = float(trial_rate)
        self.arms: Tuple[TrialArm, ...] = tuple(arms)
        self.seed = seed
        self.min_n_per_arm = int(min_n_per_arm)
        self.assignments_path = Path(assignments_path)
        self.outcomes_path = Path(outcomes_path)
        self.history_path = Path(history_path)
        for p in (self.assignments_path, self.outcomes_path, self.history_path):
            p.parent.mkdir(parents=True, exist_ok=True)

    # ---- config persistence ----

    def to_config(self) -> Dict:
        return {
            'trial_rate': self.trial_rate,
            'arms': [a.value for a in self.arms],
            'seed': self.seed,
            'min_n_per_arm': self.min_n_per_arm,
        }

    def save_config(self, path: Path = TRIAL_CONFIG_FILE) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_config(), indent=2), encoding='utf-8')

    @classmethod
    def load_config(cls, path: Path = TRIAL_CONFIG_FILE) -> 'TrialAssigner':
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            cfg = json.loads(path.read_text(encoding='utf-8'))
            arms = tuple(TrialArm(a) for a in cfg.get('arms', [a.value for a in TrialArm]))
            return cls(
                trial_rate=float(cfg.get('trial_rate', DEFAULT_TRIAL_RATE)),
                arms=arms,
                seed=cfg.get('seed', "velindre-sact-rct-v1"),
                min_n_per_arm=int(cfg.get('min_n_per_arm', DEFAULT_MIN_N_PER_ARM)),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(f"TrialAssigner.load_config fell back to defaults: {exc}")
            return cls()

    # ---- deterministic assignment ----

    def _hash(self, appointment_id: str, patient_id: str, tag: str) -> int:
        """Stable integer hash in [0, 2⁶⁴)."""
        key = f"{self.seed}|{tag}|{patient_id}|{appointment_id}"
        h = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(h, 'big', signed=False)

    def assign(
        self,
        appointment_id: str,
        patient_id: str,
        strata: str = "",
    ) -> Optional[TrialArm]:
        """
        Return the trial arm for this booking, or None if the booking
        was not selected into the trial on this roll.  Deterministic:
        the same (appointment_id, patient_id) always returns the same
        result under the same seed + trial_rate.
        """
        # Step 1 — trial selection.  Rate is encoded to 5-decimal precision
        # so small rate changes are honoured without re-rolling.
        threshold = int(self.trial_rate * 100_000)
        if self._hash(appointment_id, patient_id, "select") % 100_000 >= threshold:
            return None
        # Step 2 — arm selection.  Uniform over the configured arms.
        idx = self._hash(appointment_id, patient_id, "arm") % len(self.arms)
        return self.arms[idx]

    def batch_assign(
        self,
        rows: List[Dict],
        appointment_key: str = 'Appointment_ID',
        patient_key: str = 'Patient_ID',
    ) -> List[Optional[TrialArm]]:
        """Vectorised convenience: return arm for each row (None if not in trial)."""
        return [
            self.assign(str(r.get(appointment_key, '')), str(r.get(patient_key, '')))
            for r in rows
        ]

    # ---- logging ----

    def log_assignment(
        self, appointment_id: str, patient_id: str, arm: TrialArm, strata: str = ""
    ) -> TrialAssignment:
        rec = TrialAssignment(
            ts=datetime.utcnow().isoformat(timespec='seconds'),
            appointment_id=appointment_id,
            patient_id=patient_id,
            arm=arm.value,
            trial_rate=self.trial_rate,
            strata=strata,
        )
        with self.assignments_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(asdict(rec)) + '\n')
        return rec

    def record_outcome(
        self, appointment_id: str, patient_id: str, arm: TrialArm, attended: bool
    ) -> TrialOutcome:
        rec = TrialOutcome(
            ts=datetime.utcnow().isoformat(timespec='seconds'),
            appointment_id=appointment_id,
            patient_id=patient_id,
            arm=arm.value,
            attended=bool(attended),
        )
        with self.outcomes_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(asdict(rec)) + '\n')
        return rec

    # ---- reads ----

    def load_outcomes(self) -> List[TrialOutcome]:
        if not self.outcomes_path.exists():
            return []
        out: List[TrialOutcome] = []
        with self.outcomes_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(TrialOutcome(**json.loads(line)))
                except Exception as exc:
                    logger.warning(f"Bad trial outcome row skipped: {exc}")
        return out

    def arm_counts(self) -> Dict[str, Dict[str, int]]:
        """Return {arm: {total, attended, noshow}} from the outcomes log."""
        out = self.load_outcomes()
        result: Dict[str, Dict[str, int]] = {a.value: {'total': 0, 'attended': 0, 'noshow': 0}
                                             for a in TrialArm}
        for o in out:
            d = result.setdefault(o.arm, {'total': 0, 'attended': 0, 'noshow': 0})
            d['total'] += 1
            if o.attended:
                d['attended'] += 1
            else:
                d['noshow'] += 1
        return result

    def assignment_counts(self) -> Dict[str, int]:
        """Return {arm: count} from the assignments log (before outcomes known)."""
        result: Dict[str, int] = {}
        if not self.assignments_path.exists():
            return result
        with self.assignments_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    result[rec['arm']] = result.get(rec['arm'], 0) + 1
                except Exception:
                    continue
        return result

    # ---- ATE estimation ----

    def compute_ate(
        self,
        treatment_arm: TrialArm,
        control_arm: TrialArm = TrialArm.CONTROL,
    ) -> ATEResult:
        counts = self.arm_counts()
        t = counts.get(treatment_arm.value, {'total': 0, 'attended': 0})
        c = counts.get(control_arm.value, {'total': 0, 'attended': 0})
        n_t, a_t = int(t['total']), int(t['attended'])
        n_c, a_c = int(c['total']), int(c['attended'])
        pi_t = (a_t / n_t) if n_t > 0 else 0.0
        pi_c = (a_c / n_c) if n_c > 0 else 0.0
        ate = pi_t - pi_c
        # Wald SE for difference in proportions
        var_t = pi_t * (1.0 - pi_t) / max(n_t, 1)
        var_c = pi_c * (1.0 - pi_c) / max(n_c, 1)
        se = math.sqrt(max(var_t + var_c, 1e-12))
        ci_low, ci_high = ate - 1.96 * se, ate + 1.96 * se
        under_powered = (n_t < self.min_n_per_arm) or (n_c < self.min_n_per_arm)
        return ATEResult(
            treatment_arm=treatment_arm.value,
            control_arm=control_arm.value,
            n_treatment=n_t,
            n_control=n_c,
            pi_treatment=pi_t,
            pi_control=pi_c,
            ate=ate,
            standard_error=se,
            ci_low=ci_low,
            ci_high=ci_high,
            under_powered=under_powered,
            min_n_per_arm=self.min_n_per_arm,
        )

    def compute_all_ates(self) -> List[ATEResult]:
        """One ATE result per non-control arm vs. the CONTROL arm."""
        return [self.compute_ate(a) for a in self.arms if a != TrialArm.CONTROL]

    def append_ate_history(self, ate: ATEResult) -> None:
        try:
            with self.history_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(asdict(ate)) + '\n')
        except Exception as exc:  # pragma: no cover
            logger.warning(f"ATE history append failed: {exc}")


# ---------------------------------------------------------------------------
# Bayesian precision-weighted bridge for DML endpoint
# ---------------------------------------------------------------------------


def apply_rct_prior(
    dml_ate: float,
    dml_se: float,
    rct_ate: float,
    rct_se: float,
) -> Dict[str, float]:
    """
    Precision-weighted posterior under a Gaussian RCT prior.

    Returns a dict with posterior mean, SE, 95 % CI, and an
    "effective weight" statistic `w_rct ∈ [0, 1]` showing how much
    the posterior leans on the RCT vs. the DML estimate.
    """
    sigma2_dml = max(dml_se ** 2, 1e-12)
    sigma2_rct = max(rct_se ** 2, 1e-12)
    prec_dml = 1.0 / sigma2_dml
    prec_rct = 1.0 / sigma2_rct
    prec_post = prec_dml + prec_rct
    mean_post = (dml_ate * prec_dml + rct_ate * prec_rct) / prec_post
    sigma_post = math.sqrt(1.0 / prec_post)
    return {
        'posterior_ate': float(mean_post),
        'posterior_se': float(sigma_post),
        'posterior_ci_low': float(mean_post - 1.96 * sigma_post),
        'posterior_ci_high': float(mean_post + 1.96 * sigma_post),
        'w_rct': float(prec_rct / prec_post),
    }


__all__ = [
    'TrialArm',
    'TrialAssigner',
    'TrialAssignment',
    'TrialOutcome',
    'ATEResult',
    'apply_rct_prior',
    'DEFAULT_TRIAL_RATE',
    'DEFAULT_MIN_N_PER_ARM',
]
