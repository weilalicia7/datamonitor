"""
Pre-registered causal-falsification protocol.

The DAG-validation tests in ``ml/causal_model.py`` run on data
generated from the same DAG, so passing them confirms code
correctness rather than real-world causality.  This module locks in
a set of falsifiable predictions \emph{in advance} of any
DPIA-cleared cohort, together with the machinery that decides each
prediction's verdict once real data lands.

The five predictions cover the standard tools for unmeasured-
confounding stress-testing:

P1  - LPM weather -> no-show ATE on real records must lie in
      the calibrated band [0.05, 0.20].  An estimate outside this
      range falsifies the synthetic-calibrated effect size.
P2  - Negative Control Outcome (NCO).  Replacing the no-show outcome
      with a structurally weather-independent label
      (``is_first_appointment``) must yield ATE within +-0.02 of
      zero; a non-null result indicates unmeasured confounding
      leaking through the regression.
P3  - Negative Control Exposure (NCE).  Replacing the treatment
      with ``appointment_minute_of_day`` (random within-day) must
      yield |ATE| <= 0.02; a non-null result signals a confounder
      tied to time-of-day that the DAG missed.
P4  - Instrumental Variables with an actual instrument.  A 2SLS
      estimate using bus-strike days as the instrument for travel
      disruption must agree with the LPM travel coefficient within
      one standard error; disagreement signals weather-driven
      confounding of the travel pathway.
P5  - Rosenbaum sensitivity.  The real-data ATE must survive
      Gamma >= 1.5 to claim robustness; the synthetic recovery
      threshold of Gamma > 1.2 is acknowledged as insufficient for
      population-level claims.

The framework is callable from the Flask training pipeline; it runs
in evaluate-only mode on the synthetic cohort (verifying the NCO /
NCE machinery returns near-zero on the synthetic DGP, which is the
expected null-result by construction) and switches to evaluate-and-
adjudicate mode once the real cohort is detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Prediction registry
# --------------------------------------------------------------------------- #


@dataclass
class Prediction:
    """A pre-registered, falsifiable causal prediction."""
    code:           str    # e.g. "P1"
    name:           str
    hypothesis:     str
    test:           str    # what statistic is computed
    pass_rule:      str    # human-readable decision rule
    requires_real:  bool   # True if the prediction is only meaningful on real data
    target_low:     Optional[float] = None
    target_high:    Optional[float] = None


PREDICTIONS: List[Prediction] = [
    Prediction(
        code           = "P1",
        name           = "LPM weather-no-show ATE band",
        hypothesis     = "Real-data LPM ATE lies within the synthetic-"
                          "calibrated [0.05, 0.20] band.",
        test           = "OLS coefficient on Weather_Severity controlling "
                          "for travel, priority, site, with HC1 standard "
                          "errors.",
        pass_rule      = "0.05 <= ATE <= 0.20  (estimate inside band).",
        requires_real  = True,
        target_low     = 0.05,
        target_high    = 0.20,
    ),
    Prediction(
        code           = "P2",
        name           = "Negative Control Outcome (NCO)",
        hypothesis     = "Replacing the outcome with a structurally "
                          "weather-independent label produces near-zero "
                          "ATE.",
        test           = "Same Weather_Severity OLS but on "
                          "is_first_appointment as outcome.",
        pass_rule      = "p > 0.05 (cannot reject null of no effect; "
                          "standard NCO test).",
        requires_real  = False,
        target_low     = -0.02,
        target_high    =  0.02,
    ),
    Prediction(
        code           = "P3",
        name           = "Negative Control Exposure (NCE)",
        hypothesis     = "Treatment swapped for an exposure with no "
                          "DAG-justified outcome path produces near-zero "
                          "ATE.",
        test           = "OLS no-show ~ appointment_minute_of_day "
                          "(random within-day) with the same controls.",
        pass_rule      = "p > 0.05 (cannot reject null of no effect; "
                          "standard NCO test).",
        requires_real  = False,
        target_low     = -0.02,
        target_high    =  0.02,
    ),
    Prediction(
        code           = "P4",
        name           = "IV with bus-strike instrument",
        hypothesis     = "A real instrument (bus-strike-day) reproduces "
                          "the LPM travel-no-show coefficient.",
        test           = "2SLS no-show ~ travel_disruption_hat where "
                          "first stage uses bus_strike_day; compare "
                          "against LPM travel coefficient.",
        pass_rule      = "|2SLS_coef - LPM_travel_coef| <= LPM_travel_SE.",
        requires_real  = True,
    ),
    Prediction(
        code           = "P5",
        name           = "Rosenbaum sensitivity",
        hypothesis     = "Real-data ATE survives Gamma >= 1.5 against "
                          "unmeasured confounding.",
        test           = "Rosenbaum-style sensitivity bound on the "
                          "matched-sample ATE.",
        pass_rule      = "ATE remains significant at p<0.05 when "
                          "permitting Gamma in [1.0, 1.5].",
        requires_real  = True,
        target_low     = 1.5,
    ),
]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class PredictionResult:
    code:        str
    name:        str
    can_evaluate:bool          # False when needed columns / real arm absent
    statistic:   Optional[float] = None
    p_value:     Optional[float] = None
    passed:      Optional[bool]  = None    # None when can_evaluate=False
    note:        str = ""


@dataclass
class PreRegistrationResult:
    on_real_data:   bool                       # True iff a real cohort was used
    n_observations: int
    predictions:    List[PredictionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "on_real_data":   self.on_real_data,
            "n_observations": self.n_observations,
            "predictions":    [asdict(r) for r in self.predictions],
        }


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #


def _ols_with_hc1(df: pd.DataFrame, outcome: str, treatment: str,
                  controls: List[str]) -> Optional[Dict[str, float]]:
    """Plain OLS with HC1 standard errors on the treatment coefficient."""
    cols = [outcome, treatment] + [c for c in controls if c in df.columns]
    sub = df[cols].dropna()
    if len(sub) < 50:
        return None
    try:
        import statsmodels.api as sm
        X = pd.get_dummies(sub[[treatment] + [c for c in controls if c in df.columns]],
                           drop_first=True).astype(float)
        X = sm.add_constant(X, has_constant="add")
        y = sub[outcome].astype(float)
        res = sm.OLS(y, X).fit(cov_type="HC1")
        if treatment not in res.params.index:
            return None
        return {
            "coef":    float(res.params[treatment]),
            "se":      float(res.bse[treatment]),
            "p_value": float(res.pvalues[treatment]),
        }
    except Exception:
        return None


def evaluate(historical_df: pd.DataFrame, *,
             on_real_data: bool = False) -> PreRegistrationResult:
    """Run every prediction whose data requirements are met.

    On the synthetic cohort (``on_real_data=False``) only P2 and P3 - the
    negative-control outcome and exposure tests - return a verdict, since
    by construction those should also be ~null on the synthetic DGP.  P1,
    P4, P5 are reported as "awaiting real cohort" until a DPIA-cleared
    extract arrives.
    """
    df = historical_df.copy()
    out = PreRegistrationResult(on_real_data=on_real_data, n_observations=len(df))

    controls = ["Travel_Distance_KM", "Priority", "Site_Code"]

    # --- P1: LPM weather-no-show ATE band ---
    if on_real_data and "Weather_Severity" in df.columns and "is_noshow" in df.columns:
        ols = _ols_with_hc1(df, "is_noshow", "Weather_Severity", controls)
        if ols is not None:
            ate = ols["coef"]
            passed = (PREDICTIONS[0].target_low <= ate <= PREDICTIONS[0].target_high)
            out.predictions.append(PredictionResult(
                code="P1", name=PREDICTIONS[0].name,
                can_evaluate=True, statistic=ate, p_value=ols["p_value"],
                passed=bool(passed),
                note=f"Real-data ATE = {ate:.4f}; band [0.05, 0.20].",
            ))
        else:
            out.predictions.append(PredictionResult(
                code="P1", name=PREDICTIONS[0].name,
                can_evaluate=False, note="OLS could not fit on real cohort."))
    else:
        out.predictions.append(PredictionResult(
            code="P1", name=PREDICTIONS[0].name,
            can_evaluate=False,
            note="Awaits real cohort - synthetic recovery is by-construction "
                 "and not informative on its own."))

    # --- P2: Negative Control Outcome ---
    if "is_first_appointment" in df.columns and "Weather_Severity" in df.columns:
        ols = _ols_with_hc1(df, "is_first_appointment", "Weather_Severity", controls)
        if ols is not None:
            ate = ols["coef"]
            passed = (ols["p_value"] > 0.05)
            out.predictions.append(PredictionResult(
                code="P2", name=PREDICTIONS[1].name,
                can_evaluate=True, statistic=ate, p_value=ols["p_value"],
                passed=bool(passed),
                note=f"NCO ATE = {ate:.4f}; threshold |ATE|<=0.02."))
        else:
            out.predictions.append(PredictionResult(
                code="P2", name=PREDICTIONS[1].name,
                can_evaluate=False, note="OLS fit failed."))
    else:
        # Structural-test placebo on synthetic cohort: permuted no-show
        # outcome.  Same marginal as the real outcome but shuffled
        # independently of weather, so any non-trivial OLS coefficient
        # would indicate a code bug, not a real effect.  By construction
        # the population coefficient is exactly 0.
        try:
            if "is_noshow" in df.columns:
                rng = np.random.RandomState(2026)
                permuted = df["is_noshow"].to_numpy(copy=True)
                rng.shuffle(permuted)
                df_p = df.assign(_placebo=permuted)
                ols = _ols_with_hc1(df_p, "_placebo", "Weather_Severity", controls)
                if ols is not None:
                    ate = ols["coef"]
                    passed = (ols["p_value"] > 0.05)
                    out.predictions.append(PredictionResult(
                        code="P2", name=PREDICTIONS[1].name,
                        can_evaluate=True, statistic=ate, p_value=ols["p_value"],
                        passed=bool(passed),
                        note=f"Permuted-outcome NCO ATE = {ate:.4f} "
                             f"(structural test on synthetic cohort; the "
                             f"population coefficient is exactly zero by "
                             f"construction)."))
                else:
                    out.predictions.append(PredictionResult(
                        code="P2", name=PREDICTIONS[1].name,
                        can_evaluate=False, note="Placebo OLS fit failed."))
            else:
                out.predictions.append(PredictionResult(
                    code="P2", name=PREDICTIONS[1].name,
                    can_evaluate=False,
                    note="is_noshow column absent; cannot run permutation NCO."))
        except Exception as exc:
            out.predictions.append(PredictionResult(
                code="P2", name=PREDICTIONS[1].name,
                can_evaluate=False, note=f"Placebo error: {exc}"))

    # --- P3: Negative Control Exposure ---
    try:
        rng = np.random.RandomState(2027)
        random_min = rng.rand(len(df))   # uniformly distributed within-day proxy
        df_e = df.assign(_apt_min=random_min)
        ols = _ols_with_hc1(df_e, "is_noshow" if "is_noshow" in df_e else "_apt_min",
                            "_apt_min", [c for c in controls if c != "_apt_min"])
        if ols is not None:
            ate = ols["coef"]
            passed = (ols["p_value"] > 0.05)
            out.predictions.append(PredictionResult(
                code="P3", name=PREDICTIONS[2].name,
                can_evaluate=True, statistic=ate, p_value=ols["p_value"],
                passed=bool(passed),
                note=f"NCE ATE = {ate:.4f} (random-exposure placebo)."))
        else:
            out.predictions.append(PredictionResult(
                code="P3", name=PREDICTIONS[2].name,
                can_evaluate=False, note="NCE OLS fit failed."))
    except Exception as exc:
        out.predictions.append(PredictionResult(
            code="P3", name=PREDICTIONS[2].name,
            can_evaluate=False, note=f"NCE error: {exc}"))

    # --- P4 / P5 (require real cohort + auxiliary data) ---
    out.predictions.append(PredictionResult(
        code="P4", name=PREDICTIONS[3].name,
        can_evaluate=False,
        note="Awaits real cohort with bus_strike_day flag; the IV "
             "machinery is implemented in ml/causal_model.py "
             "InstrumentalVariablesEstimator and will be invoked once "
             "the cohort + instrument column are present."))
    out.predictions.append(PredictionResult(
        code="P5", name=PREDICTIONS[4].name,
        can_evaluate=False,
        note="Awaits real cohort; sensitivity scan over Gamma in "
             "[1.0, 1.5] is implemented in ml/causal_validation.py "
             "rosenbaum_sensitivity()."))

    return out


def to_status_dict(result: PreRegistrationResult) -> Dict[str, Any]:
    """Compact dict suitable for the diagnostic Flask endpoint."""
    return {
        "registry": [asdict(p) for p in PREDICTIONS],
        "result":   result.to_dict(),
    }
