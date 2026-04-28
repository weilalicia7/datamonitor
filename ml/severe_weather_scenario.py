"""
Severe-weather scenario simulation.

Given the historical appointment cohort, computes the effective chair-waste
reduction expected when a calibrated overbooking + phone-confirmation
intervention is targeted at the highest-risk quartile of patients on a
severe-weather day.  Mirrors the R analysis in
``dissertation/dissertation_analysis.R`` Section 4b so that the production
pipeline and the dissertation report identical numbers from the same data.

Intervention effect sizes are taken from the operational literature:
- phone confirmation 2 h prior captures ~50 % of intent-to-no-show patients
  (midpoint of the 40-60 % range reported by Daggy 2010, Liu 2019);
- 12 % slot overbooking on top-quartile risk; pre-confirmed overflow patients
  fill ~85 % of the no-show events that survive phone confirmation.

The output is exposed as a status endpoint
(``GET /api/ml/scenario/severe-weather``) for diagnostics; it is not surfaced
in the viewer UI - the simulation runs silently as part of the model-training
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


PHONE_CAPTURE_RATE = 0.50    # midpoint of literature 40-60 %
OVERBOOK_SHARE     = 0.12    # 12 % of slots offered to overflow patients
OVERBOOK_FILL_RATE = 0.85    # pre-confirmed overflow availability


DEFAULT_BOOTSTRAP_B = 1000
DEFAULT_BOOTSTRAP_SEED = 2026


@dataclass
class SevereWeatherScenarioResult:
    """Container for the severe-weather intervention simulation."""

    # Cohort-average no-show rate (the all-day baseline, kept for context).
    cohort_baseline_pct:         float
    # Severe-day no-intervention rate = cohort baseline + weather ATE.
    # This is the proper counterfactual for the intervention comparison.
    baseline_waste_pct:          float
    # Severe-day with-intervention rate (top-quartile risk only).
    post_intervention_waste_pct: float
    # (severe_baseline - severe_post) / severe_baseline, as %.
    relative_reduction_pct:      float
    # Weather ATE used to lift cohort baseline → severe-day baseline.
    weather_ate:                 float
    top_quartile_share_pct:      float
    n_observations:              int
    phone_capture_rate:          float = PHONE_CAPTURE_RATE
    overbook_share:              float = OVERBOOK_SHARE
    overbook_fill_rate:          float = OVERBOOK_FILL_RATE
    # 95 % percentile bootstrap CIs over row resamples (display %)
    baseline_ci_lo:              Optional[float] = None
    baseline_ci_hi:              Optional[float] = None
    post_ci_lo:                  Optional[float] = None
    post_ci_hi:                  Optional[float] = None
    relative_ci_lo:              Optional[float] = None
    relative_ci_hi:              Optional[float] = None
    n_bootstrap:                 Optional[int]   = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _scenario_one_pass(df: pd.DataFrame, weather_col: str,
                       travel_col: str, noshow_col: str,
                       *, weather_ate: float
                       ) -> Optional[Dict[str, float]]:
    """Single-pass scenario computation; returns rounded display values.

    The proper severe-weather counterfactual:

      cohort_baseline  = mean no-show across all days (~14.5 %).
      severe_baseline  = cohort_baseline + weather_ate (~ 28.9 % when
                         ATE = 0.144).  Each patient's no-show
                         probability lifts by the ATE on a severe-
                         weather day.
      severe_post      = severe_baseline with the intervention
                         applied to the top-quartile risk only:
                         their elevated no-show probability is
                         attenuated by phone confirmation +
                         overbooking; the rest of the cohort retains
                         their severe-day rate (no intervention).

    The dissertation now compares severe_baseline → severe_post (the
    proper counterfactual), and reports cohort_baseline separately
    for context.
    """
    if len(df) < 30:
        return None
    ws_max = df[weather_col].max()
    td_max = df[travel_col].max()
    if ws_max == 0 or td_max == 0:
        return None
    ws_norm = df[weather_col] / ws_max
    td_norm = df[travel_col]  / td_max
    risk    = 0.6 * ws_norm + 0.4 * td_norm
    cutoff   = risk.quantile(0.75)
    is_top_q = risk >= cutoff

    top_q_baseline = df.loc[is_top_q, noshow_col].mean()
    rest_baseline  = df.loc[~is_top_q, noshow_col].mean()
    if pd.isna(top_q_baseline) or pd.isna(rest_baseline):
        return None

    # Severe-weather lifts every patient's no-show probability by the
    # ATE.  Probabilities are clipped to [0, 1] for safety.
    top_q_severe = min(1.0, max(0.0, top_q_baseline + weather_ate))
    rest_severe  = min(1.0, max(0.0, rest_baseline  + weather_ate))

    # Intervention applies only to the top quartile (phone confirmation
    # then overbooking with the configured pre-confirmed fill rate).
    top_q_after_phone  = top_q_severe * (1 - PHONE_CAPTURE_RATE)
    top_q_after_overbk = top_q_after_phone * (
        1 - OVERBOOK_SHARE * OVERBOOK_FILL_RATE
    )
    top_share = float(is_top_q.mean())

    cohort_baseline = float(df[noshow_col].mean())
    # Proper severe-day no-intervention waste: cohort baseline + ATE.
    severe_baseline = min(1.0, max(0.0, cohort_baseline + weather_ate))
    # Severe-day WITH intervention (top-q intervened; rest unchanged).
    severe_post = top_share * top_q_after_overbk + (1 - top_share) * rest_severe

    cohort_baseline_disp = round(100 * cohort_baseline, 1)
    baseline_disp        = round(100 * severe_baseline, 1)
    post_disp            = round(100 * severe_post,     1)
    if baseline_disp <= 0:
        return None
    rel_red_disp = round(
        (baseline_disp - post_disp) / baseline_disp * 100, 1
    )
    return {
        "cohort_baseline_pct": cohort_baseline_disp,
        "baseline_pct":        baseline_disp,
        "post_pct":            post_disp,
        "relative_pct":        rel_red_disp,
        "top_share_pct":       round(100 * top_share, 1),
    }


def simulate_severe_weather_intervention(
    historical_df: pd.DataFrame,
    weather_col: str = "Weather_Severity",
    travel_col:  str = "Travel_Distance_KM",
    noshow_col:  str = "is_noshow",
    *,
    weather_ate:    float = 0.1444,    # default to LPM ATE; override per cohort
    n_bootstrap:    int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Optional[SevereWeatherScenarioResult]:
    """
    Run the severe-weather scenario simulation on the given cohort.

    Returns ``None`` if any required column is missing.

    Display values are rounded to one decimal place; the relative reduction
    is computed from the rounded baseline and post-intervention numbers so
    that ``(baseline - post) / baseline`` taken from the published digits
    reproduces the reported relative reduction exactly.

    Each headline number additionally carries a 95 % percentile bootstrap
    CI over ``n_bootstrap`` row resamples (default 1 000); set
    ``n_bootstrap=0`` to skip the bootstrap.
    """
    required = (weather_col, travel_col, noshow_col)
    if any(c not in historical_df.columns for c in required):
        return None

    df = historical_df[list(required)].dropna().reset_index(drop=True)
    if len(df) < 50:
        return None

    point = _scenario_one_pass(df, weather_col, travel_col, noshow_col,
                                weather_ate=weather_ate)
    if point is None:
        return None
    if not (point["post_pct"] < point["baseline_pct"]):
        return None

    # --- Bootstrap CIs over row resamples ---
    ci = {
        "baseline_lo": None, "baseline_hi": None,
        "post_lo":     None, "post_hi":     None,
        "rel_lo":      None, "rel_hi":      None,
        "n":           None,
    }
    if n_bootstrap > 0:
        rng = np.random.RandomState(bootstrap_seed)
        n_rows = len(df)
        baseline_arr, post_arr, rel_arr = [], [], []
        for _ in range(n_bootstrap):
            boot_idx = rng.randint(0, n_rows, size=n_rows)
            b_df = df.iloc[boot_idx].reset_index(drop=True)
            b = _scenario_one_pass(b_df, weather_col, travel_col, noshow_col,
                                    weather_ate=weather_ate)
            if b is not None and b["baseline_pct"] > 0:
                baseline_arr.append(b["baseline_pct"])
                post_arr.append(b["post_pct"])
                rel_arr.append(b["relative_pct"])
        if len(baseline_arr) >= n_bootstrap // 2:
            ci["baseline_lo"] = round(float(np.percentile(baseline_arr, 2.5)), 1)
            ci["baseline_hi"] = round(float(np.percentile(baseline_arr, 97.5)), 1)
            ci["post_lo"]     = round(float(np.percentile(post_arr,     2.5)), 1)
            ci["post_hi"]     = round(float(np.percentile(post_arr,     97.5)), 1)
            ci["rel_lo"]      = round(float(np.percentile(rel_arr,      2.5)), 1)
            ci["rel_hi"]      = round(float(np.percentile(rel_arr,      97.5)), 1)
            ci["n"]           = int(len(baseline_arr))

    return SevereWeatherScenarioResult(
        cohort_baseline_pct         = point["cohort_baseline_pct"],
        baseline_waste_pct          = point["baseline_pct"],
        post_intervention_waste_pct = point["post_pct"],
        relative_reduction_pct      = point["relative_pct"],
        weather_ate                 = round(weather_ate, 4),
        top_quartile_share_pct      = point["top_share_pct"],
        n_observations              = len(df),
        baseline_ci_lo              = ci["baseline_lo"],
        baseline_ci_hi              = ci["baseline_hi"],
        post_ci_lo                  = ci["post_lo"],
        post_ci_hi                  = ci["post_hi"],
        relative_ci_lo              = ci["rel_lo"],
        relative_ci_hi              = ci["rel_hi"],
        n_bootstrap                 = ci["n"],
    )
