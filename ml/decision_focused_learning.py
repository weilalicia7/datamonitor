"""
Decision-Focused Learning for the No-Show Model (Dissertation §2.1)
===================================================================

Standard ML training minimises cross-entropy (statistical accuracy).
In SACT scheduling what actually matters is the *cost of the
scheduling decision* taken on the back of the prediction:

  * A probability that is a little too low but still below the
    "assign-alone" threshold τ produces the right decision at zero cost.
  * A probability that is a little too high and flips to "double-book"
    imposes a crowding cost even though the prediction is only slightly
    miscalibrated.

This module closes the gap without touching the base XGBoost ensemble.
It fits a two-parameter calibration head

    g(p) = σ( a · logit(p) + b )

to minimise a smooth surrogate of decision regret

    c̃(p, y, τ) = (1 − σ_τ(p)) · y · WASTE_COST
                + σ_τ(p)     · (1 − y) · CROWDING_COST
    with σ_τ(p) = σ(β · (p − τ))

where y ∈ {0, 1} is the observed attendance outcome, τ is the
double-booking threshold used by the optimiser, β is the sharpness
of the smooth decision, and WASTE_COST / CROWDING_COST come from
the scheduling weights.  The SPO+ style surrogate of Elmachtoub &
Grigas (2022) is equivalent to this smooth-cost form when the
downstream decision is a threshold rule on a single probability.

Gradient is exact w.r.t. (a, b) — no perturbation loop required for
the two-parameter head — which is why this version is *many* orders
of magnitude cheaper than calling CP-SAT inside the training loop
(Wilder et al. 2019's blackbox-perturbation route), while preserving
the decision-awareness that Wilder's method targets.

Usage
-----
    from ml.decision_focused_learning import DFLCalibrator
    dfl = DFLCalibrator()
    dfl.fit(p_raw=raw_probs, y_true=attendance_outcomes)
    p_star = dfl.calibrate(p_raw_new)           # vectorised
    dfl.save()                                  # pickle → MODELS_DIR

When attached to a NoShowModel, every prediction flows through
`dfl.calibrate()` before the optimiser sees it, so the entire
system — CP-SAT objective, squeeze-in, IRL — benefits automatically
with zero additional integration work.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import pickle

import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_CACHE_DIR,
    MODELS_DIR,
    OPTIMIZATION_WEIGHTS,
    NOSHOW_THRESHOLDS,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults — derived from the optimiser's own weights so the DFL objective
# is consistent with whatever the operator has configured (fixed prior, or
# learned θ from the IRL module).
# ---------------------------------------------------------------------------
DEFAULT_DOUBLE_BOOK_THRESHOLD: float = float(NOSHOW_THRESHOLDS.get('high', 0.40))
DEFAULT_SHARPNESS: float = 20.0  # σ(20·(p−τ)) approximates the hard rule well
DEFAULT_WASTE_COST: float = 100.0 * float(OPTIMIZATION_WEIGHTS.get('utilization', 0.25))
DEFAULT_CROWD_COST: float = 100.0 * float(OPTIMIZATION_WEIGHTS.get('robustness', 0.10))

DFL_MODEL_FILE: Path = MODELS_DIR / 'dfl_calibrator.pkl'
DFL_HISTORY_FILE: Path = DATA_CACHE_DIR / 'dfl_history.jsonl'


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def smooth_decision_cost(
    p: np.ndarray,
    y: np.ndarray,
    threshold: float = DEFAULT_DOUBLE_BOOK_THRESHOLD,
    sharpness: float = DEFAULT_SHARPNESS,
    waste_cost: float = DEFAULT_WASTE_COST,
    crowd_cost: float = DEFAULT_CROWD_COST,
) -> float:
    """
    Smooth surrogate of total decision cost (vectorised).

    Lower is better.  The cost decomposes into two terms:
      * false "safe"   — predicted low but patient no-shows → chair idle
      * false "risky"  — predicted high but patient attends → double-book crowding
    """
    sigma_tau = _sigmoid(sharpness * (p - threshold))
    per_example = (1.0 - sigma_tau) * y * waste_cost + sigma_tau * (1.0 - y) * crowd_cost
    return float(per_example.sum())


def regret(p_pred: np.ndarray, y_true: np.ndarray, **kw) -> float:
    """
    Decision regret: cost under predicted probabilities minus cost
    under an oracle that knows y_true.  Oracle sends p=1 when y=1 and
    p=0 when y=0 — i.e., the right decision every time, zero cost.
    Regret is therefore equal to the predicted cost itself — we keep
    the symbolic definition so the API stays honest.
    """
    return smooth_decision_cost(p_pred, y_true, **kw) - smooth_decision_cost(
        y_true.astype(float), y_true, **kw
    )


# ---------------------------------------------------------------------------
# DFL calibration head
# ---------------------------------------------------------------------------


@dataclass
class DFLFitResult:
    """Summary of a single DFL training run."""
    a: float                    # slope on logit scale
    b: float                    # bias on logit scale
    n_samples: int
    threshold: float
    sharpness: float
    waste_cost: float
    crowd_cost: float
    regret_before: float        # smooth cost with raw probabilities
    regret_after: float         # smooth cost after calibration
    regret_improvement_pct: float
    ce_before: float            # cross-entropy (for A/B monitoring)
    ce_after: float
    auc_before: Optional[float]
    auc_after: Optional[float]
    iterations: int
    converged: bool
    fit_ts: str


class DFLCalibrator:
    """
    Decision-focused calibration layer sitting on top of any probabilistic
    binary classifier (the NoShowModel ensemble).  Two parameters; fit
    with gradient descent on the smooth-decision cost.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_DOUBLE_BOOK_THRESHOLD,
        sharpness: float = DEFAULT_SHARPNESS,
        waste_cost: float = DEFAULT_WASTE_COST,
        crowd_cost: float = DEFAULT_CROWD_COST,
        model_path: Path = DFL_MODEL_FILE,
        history_path: Path = DFL_HISTORY_FILE,
    ) -> None:
        self.threshold = float(threshold)
        self.sharpness = float(sharpness)
        self.waste_cost = float(waste_cost)
        self.crowd_cost = float(crowd_cost)
        self.model_path = Path(model_path)
        self.history_path = Path(history_path)

        self.a: float = 1.0   # identity calibration on fresh instance
        self.b: float = 0.0
        self.last_fit: Optional[DFLFitResult] = None

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self._try_load_state()

    # -------- public API --------

    def is_fitted(self) -> bool:
        return self.last_fit is not None

    def calibrate(self, p_raw) -> np.ndarray:
        """
        Apply the calibration head.  Accepts scalars, lists, or arrays;
        always returns a numpy array so downstream code can treat the
        single-patient and batch cases uniformly.
        """
        arr = np.asarray(p_raw, dtype=float).ravel()
        logit = _logit(arr)
        return _sigmoid(self.a * logit + self.b)

    def calibrate_scalar(self, p_raw: float) -> float:
        """Convenience wrapper for the single-patient hot path."""
        return float(self.calibrate(p_raw)[0])

    def fit(
        self,
        p_raw: Sequence[float],
        y_true: Sequence[int],
        max_iterations: int = 500,
        learning_rate: float = 0.05,   # kept for API compatibility; unused by L-BFGS-B
        tol: float = 1e-7,
    ) -> DFLFitResult:
        """
        Fit (a, b) by L-BFGS-B on the smooth decision-cost surrogate.

        Non-negativity / monotonicity constraint: a ≥ 0 keeps the
        calibration monotone (a higher raw probability cannot produce a
        lower calibrated probability), which is required to preserve
        the no-show ordering used by the double-booking score.  L-BFGS-B
        handles the step-size selection robustly even when the smooth
        decision function saturates, which a hand-tuned GD loop does not.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scipy is required for DFLCalibrator.fit()") from exc

        p_raw = np.asarray(p_raw, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        if len(p_raw) != len(y_true) or len(p_raw) < 2:
            raise ValueError("DFLCalibrator.fit needs matching arrays of length ≥ 2.")

        lam_reg = 1e-3  # gentle L2 keeping (a, b) near identity
        logit_raw = _logit(p_raw)

        # Diagnostics before
        regret_before = smooth_decision_cost(
            p_raw, y_true,
            threshold=self.threshold, sharpness=self.sharpness,
            waste_cost=self.waste_cost, crowd_cost=self.crowd_cost,
        )
        ce_before = float(-np.mean(
            y_true * np.log(np.clip(p_raw, 1e-9, 1)) +
            (1 - y_true) * np.log(np.clip(1 - p_raw, 1e-9, 1))
        ))
        auc_before = _auc_safe(y_true, p_raw)

        def _objective_and_grad(params):
            a, b = float(params[0]), float(params[1])
            p_cal = _sigmoid(a * logit_raw + b)
            sigma_tau = _sigmoid(self.sharpness * (p_cal - self.threshold))
            cost = float(
                ((1.0 - sigma_tau) * y_true * self.waste_cost
                 + sigma_tau * (1.0 - y_true) * self.crowd_cost).sum()
            ) + lam_reg * ((a - 1.0) ** 2 + b ** 2)

            dc_dsig = (1.0 - y_true) * self.crowd_cost - y_true * self.waste_cost
            dsig_dp = self.sharpness * sigma_tau * (1.0 - sigma_tau)
            dp_da = p_cal * (1.0 - p_cal) * logit_raw
            dp_db = p_cal * (1.0 - p_cal)
            grad_a = float(np.sum(dc_dsig * dsig_dp * dp_da)) + 2.0 * lam_reg * (a - 1.0)
            grad_b = float(np.sum(dc_dsig * dsig_dp * dp_db)) + 2.0 * lam_reg * b
            return cost, np.array([grad_a, grad_b])

        result = minimize(
            _objective_and_grad,
            x0=np.array([1.0, 0.0]),             # identity warm-start
            jac=True,
            method='L-BFGS-B',
            bounds=[(1e-4, 20.0), (-20.0, 20.0)],  # monotonicity + saturation clip
            options={'maxiter': max_iterations, 'ftol': tol, 'gtol': tol},
        )
        self.a = float(result.x[0])
        self.b = float(result.x[1])
        converged = bool(result.success)
        it = int(result.nit)

        # Diagnostics after
        p_cal = self.calibrate(p_raw)
        regret_after = smooth_decision_cost(
            p_cal, y_true,
            threshold=self.threshold, sharpness=self.sharpness,
            waste_cost=self.waste_cost, crowd_cost=self.crowd_cost,
        )
        ce_after = float(-np.mean(
            y_true * np.log(np.clip(p_cal, 1e-9, 1)) +
            (1 - y_true) * np.log(np.clip(1 - p_cal, 1e-9, 1))
        ))
        auc_after = _auc_safe(y_true, p_cal)
        improvement_pct = (
            100.0 * (regret_before - regret_after) / max(regret_before, 1e-9)
        )

        fit = DFLFitResult(
            a=self.a, b=self.b,
            n_samples=len(p_raw),
            threshold=self.threshold, sharpness=self.sharpness,
            waste_cost=self.waste_cost, crowd_cost=self.crowd_cost,
            regret_before=regret_before, regret_after=regret_after,
            regret_improvement_pct=improvement_pct,
            ce_before=ce_before, ce_after=ce_after,
            auc_before=auc_before, auc_after=auc_after,
            iterations=it + 1, converged=converged,
            fit_ts=datetime.utcnow().isoformat(timespec='seconds'),
        )
        self.last_fit = fit
        self._save_state()
        self._append_history(fit)
        logger.info(
            f"DFL fit: N={fit.n_samples} a={fit.a:.3f} b={fit.b:.3f} "
            f"regret {fit.regret_before:.1f}→{fit.regret_after:.1f} "
            f"({fit.regret_improvement_pct:+.1f}%), "
            f"CE {fit.ce_before:.3f}→{fit.ce_after:.3f}"
        )
        return fit

    # -------- persistence --------

    def _save_state(self) -> None:
        payload = {
            'a': self.a, 'b': self.b,
            'threshold': self.threshold, 'sharpness': self.sharpness,
            'waste_cost': self.waste_cost, 'crowd_cost': self.crowd_cost,
            'last_fit': asdict(self.last_fit) if self.last_fit else None,
        }
        with self.model_path.open('wb') as fh:
            pickle.dump(payload, fh)

    def _try_load_state(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with self.model_path.open('rb') as fh:
                payload = pickle.load(fh)
            self.a = float(payload.get('a', 1.0))
            self.b = float(payload.get('b', 0.0))
            self.threshold = float(payload.get('threshold', self.threshold))
            self.sharpness = float(payload.get('sharpness', self.sharpness))
            self.waste_cost = float(payload.get('waste_cost', self.waste_cost))
            self.crowd_cost = float(payload.get('crowd_cost', self.crowd_cost))
            if payload.get('last_fit'):
                self.last_fit = DFLFitResult(**payload['last_fit'])
        except Exception as exc:
            logger.warning(f"Could not load DFL state: {exc}")

    def _append_history(self, fit: DFLFitResult) -> None:
        try:
            with self.history_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(asdict(fit)) + '\n')
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not append DFL history: {exc}")

    def reset(self) -> None:
        """Revert to identity calibration (a=1, b=0)."""
        self.a, self.b = 1.0, 0.0
        self.last_fit = None
        self._save_state()


def _auc_safe(y_true: np.ndarray, p: np.ndarray) -> Optional[float]:
    """AUC with graceful fallback if sklearn is unavailable or labels are degenerate."""
    if y_true.min() == y_true.max():
        return None  # all attended or all no-show — AUC undefined
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, p))
    except Exception:
        return None


__all__ = [
    'DFLCalibrator',
    'DFLFitResult',
    'smooth_decision_cost',
    'regret',
    'DEFAULT_DOUBLE_BOOK_THRESHOLD',
    'DEFAULT_SHARPNESS',
    'DEFAULT_WASTE_COST',
    'DEFAULT_CROWD_COST',
]
