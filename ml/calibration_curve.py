"""
Calibration curve + scoring for no-show probability predictions.

A high AUC says the predictor ranks patients in the right order; it does
NOT say the predicted probabilities are themselves trustworthy.  The
overbooking decision in the CP-SAT objective uses the probability VALUE,
not just the rank, so a well-calibrated model is operationally critical:

  - if the model says "this patient has a 0.40 no-show probability", the
    overbooker should expect ~40 % of such patients to actually no-show.

This module computes a reliability diagram (binned predicted vs. observed
rate) plus four scalar diagnostics:

  - Expected Calibration Error (ECE): bin-weighted mean |predicted - observed|.
  - Maximum Calibration Error (MCE): worst-bin |predicted - observed|.
  - Brier score: mean (predicted - observed)^2.
  - Brier skill score: 1 - Brier(model) / Brier(climatology).
    Climatology = predict the cohort base rate for every patient;
    skill > 0 means the model beats the base-rate predictor.

The output is consumed by the dissertation R analysis (fig 11 calibration
curve) and exposed via the diagnostic Flask endpoint
``GET /api/ml/calibration``.  Runs silently in the prediction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

import numpy as np


DEFAULT_N_BINS = 10


@dataclass
class CalibrationBin:
    bin_index:    int
    lower:        float
    upper:        float
    n:            int
    predicted_mean:  float
    observed_rate:   float
    abs_gap:         float


@dataclass
class CalibrationResult:
    n:                int
    n_bins:           int
    base_rate:        float
    ece:              float                       # Expected Calibration Error
    mce:              float                       # Maximum Calibration Error
    brier:            float                       # Brier score
    brier_climatology:float                       # Brier of base-rate predictor
    brier_skill_score:float                       # 1 - Brier/Brier_climatology
    bins:             List[CalibrationBin] = field(default_factory=list)
    method:           str = (
        "Equal-frequency binning (deciles by predicted probability); "
        "ECE = sum_b (n_b/N) * |mean_pred_b - obs_rate_b|; "
        "MCE = max_b |mean_pred_b - obs_rate_b|; "
        "Brier = mean (p - y)^2; "
        "Brier skill = 1 - Brier / Brier_climatology."
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def compute_calibration(y_true: np.ndarray, y_proba: np.ndarray,
                        *, n_bins: int = DEFAULT_N_BINS,
                        equal_frequency: bool = True
                        ) -> Optional[CalibrationResult]:
    """
    Reliability diagram + scoring for a probabilistic binary predictor.

    Equal-frequency binning is preferred over equal-width because real
    no-show predictions are typically left-skewed (most patients have
    low risk); equal-width buckets would leave the rightmost bins empty
    and inflate the headline ECE.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)
    if y.shape != p.shape or y.size < n_bins * 5:
        return None
    if not np.all((y == 0) | (y == 1)):
        return None
    if not np.all((p >= 0) & (p <= 1)):
        # clamp into [0, 1] - some models can emit slightly out-of-range scores
        p = np.clip(p, 0.0, 1.0)

    n_total = int(y.size)
    base_rate = float(y.mean())

    # Bin assignments
    if equal_frequency:
        # Quantile cuts; tie-aware so every bin has at least one sample
        ranks = np.argsort(p, kind="mergesort")
        boundary_idx = np.linspace(0, n_total, n_bins + 1, dtype=int)
        bin_id = np.zeros(n_total, dtype=int)
        for b in range(n_bins):
            bin_id[ranks[boundary_idx[b]:boundary_idx[b + 1]]] = b
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_id = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    bins: List[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_id == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        pmean = float(p[mask].mean())
        obs   = float(y[mask].mean())
        gap   = abs(pmean - obs)
        ece  += (n_b / n_total) * gap
        mce   = max(mce, gap)
        bins.append(CalibrationBin(
            bin_index      = b,
            lower          = float(p[mask].min()),
            upper          = float(p[mask].max()),
            n              = n_b,
            predicted_mean = round(pmean, 4),
            observed_rate  = round(obs,   4),
            abs_gap        = round(gap,   4),
        ))

    brier             = float(np.mean((p - y) ** 2))
    brier_climatology = float(np.mean((base_rate - y) ** 2))
    brier_skill       = (
        float(1.0 - brier / brier_climatology) if brier_climatology > 0 else 0.0
    )

    return CalibrationResult(
        n                = n_total,
        n_bins            = n_bins,
        base_rate         = round(base_rate, 4),
        ece               = round(ece, 4),
        mce               = round(mce, 4),
        brier             = round(brier, 4),
        brier_climatology = round(brier_climatology, 4),
        brier_skill_score = round(brier_skill, 4),
        bins              = bins,
    )
