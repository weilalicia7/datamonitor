"""
Multi-method calibration benchmark for the no-show predictor.

Trains a fresh gradient-boosted no-show head on the synthetic cohort
(or the real cohort once the channel detector flips), evaluates FIVE
calibration regimes on the same held-out probe, breaks down the best
regime by protected subgroup, and runs a scheduler-impact sensitivity
analysis comparing each regime against an oracle-calibrated reference.
Writes one JSONL row to ``data_cache/calibration/results.jsonl``.

Calibration regimes
-------------------
* **raw**         - bare GBM probabilities (the historical baseline).
* **isotonic**    - 5-fold CV isotonic regression
                    (``sklearn.calibration.CalibratedClassifier\\-CV``).
* **sigmoid**     - Platt scaling (5-fold CV sigmoid).
* **beta**        - Beta calibration (Kull, Silva Filho & Flach 2017),
                    a 3-parameter logistic on (log p, log(1-p), 1)
                    fit by 5-fold CV.  Implemented inline because the
                    ``betacal`` package is not part of the standard
                    NHS-friendly pip stack.
* **temperature** - scalar temperature scaling on the logits
                    (T fit by minimising NLL on a held-out fold).

The §5.2.1 dissertation prose now picks the regime with the highest
Brier-skill score from this run as the headline production-time
calibrator, and reports the multi-regime comparison + per-subgroup
breakdown + scheduler-impact sensitivity in the appendix.

Scheduler-impact sensitivity
----------------------------
The CP-SAT objective uses the no-show probability VALUE for slot
overbooking (threshold 0.30 in the production config).  This benchmark
counts how often each regime's probabilities drive an overbook decision
that matches the OBSERVED outcome, against an "oracle" reference where
each test patient is given their bin's empirical no-show rate.  The
gap quantifies the operational cost of mis-calibrated probabilities.

CLI
---
    python -m ml.benchmark_calibration \\
        --n-patients 1500 --test-frac 0.20 --n-bins 10 --seed 42

Diagnostic-only - no UI panel, no email, no live-pipeline side effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# Production overbook threshold from optimization/optimizer.py defaults.
OVERBOOK_THRESHOLD = 0.30
# Numerical floor / ceiling for log-transforms.
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Cohort + feature loaders
# --------------------------------------------------------------------------- #

def _load_cohort(n: int):
    import pandas as pd
    path = _REPO_ROOT / "datasets" / "sample_data" / "historical_appointments.xlsx"
    df = pd.read_excel(path)
    df = df.head(n).reset_index(drop=True)
    return df


def _extract_xy(df):
    """Numeric feature matrix + binary no-show label."""
    import numpy as np
    import pandas as pd
    FEATURES = [
        "Age", "Cycle_Number", "Treatment_Day", "Planned_Duration",
        "Travel_Time_Min", "Day_Of_Week_Num",
        "Total_Appointments_Before", "Previous_NoShows",
    ]
    def _num(col):
        if col.dtype == object:
            col = col.astype(str).str.extract(r"(-?\d+\.?\d*)")[0]
        return pd.to_numeric(col, errors="coerce").fillna(0).values
    n = len(df)
    X = np.zeros((n, len(FEATURES)), dtype=float)
    for j, c in enumerate(FEATURES):
        if c in df.columns:
            X[:, j] = _num(df[c])
    if "Showed_Up" in df.columns:
        y = (1 - df["Showed_Up"].astype(int)).values
    elif "Attended_Status" in df.columns:
        y = df["Attended_Status"].astype(str).str.lower().map(
            {"yes": 0, "no": 1, "cancelled": 1, "attended": 0}
        ).fillna(0).astype(int).values
    elif "no_show" in df.columns:
        y = df["no_show"].astype(int).values
    elif "is_noshow" in df.columns:
        y = df["is_noshow"].astype(int).values
    else:
        y = None
    return X, y


def _extract_subgroups(df):
    """Per-row protected-attribute labels for the per-subgroup calibration."""
    import numpy as np
    import pandas as pd
    sg = {}
    if "Person_Stated_Gender_Code" in df.columns:
        gc = pd.to_numeric(df["Person_Stated_Gender_Code"],
                           errors="coerce").fillna(0).astype(int)
        sg["gender"] = np.where(gc == 1, "M",
                        np.where(gc == 2, "F", "Unknown"))
    if "Age_Band" in df.columns:
        sg["age_band"] = df["Age_Band"].astype(str).fillna("unknown").values
    elif "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce").fillna(0).astype(int)
        sg["age_band"] = np.where(age < 40, "0-39",
                          np.where(age < 65, "40-64", "65+"))
    if "Site_Code" in df.columns:
        sg["site"] = df["Site_Code"].astype(str).fillna("unknown").values
    if "Priority" in df.columns:
        sg["priority"] = df["Priority"].astype(str).fillna("unknown").values
    return sg


# --------------------------------------------------------------------------- #
# Calibration methods
# --------------------------------------------------------------------------- #


def _fit_predict_calibrated(method: str, base_estimator, X_tr, y_tr, X_te):
    """sklearn-backed regimes: isotonic, sigmoid (Platt)."""
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(base_estimator, method=method, cv=5)
    cal.fit(X_tr, y_tr)
    return cal.predict_proba(X_te)[:, 1]


def _beta_calibrate_fit(p_tr, y_tr):
    """
    Beta calibration (Kull, Silva Filho, Flach 2017).

    Maps raw probabilities through a 3-parameter logistic:

        logit(p_cal) = a * log(p) + b * log(1 - p) + c

    Fit by L-BFGS minimisation of binomial NLL.  Returns a callable
    that maps raw probabilities -> calibrated probabilities.
    """
    import numpy as np
    from scipy.optimize import minimize
    p = np.clip(np.asarray(p_tr, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y_tr, dtype=float)
    log_p   = np.log(p)
    log_1mp = np.log(1 - p)

    def nll(theta):
        a, b, c = theta
        z = a * log_p + b * log_1mp + c
        # Numerically-stable log-sigmoid
        # log(sigmoid(z)) = -log(1 + exp(-z))
        log_sig = np.where(z >= 0, -np.log1p(np.exp(-z)), z - np.log1p(np.exp(z)))
        log_one_minus = np.where(z >= 0, -z - np.log1p(np.exp(-z)),
                                 -np.log1p(np.exp(z)))
        return -float(np.mean(y * log_sig + (1 - y) * log_one_minus))

    init = np.array([1.0, -1.0, 0.0])
    res = minimize(nll, init, method="L-BFGS-B")
    a, b, c = res.x

    def predict(p_in):
        p_clip = np.clip(np.asarray(p_in, dtype=float), EPS, 1 - EPS)
        z = a * np.log(p_clip) + b * np.log(1 - p_clip) + c
        return 1.0 / (1.0 + np.exp(-z))

    return predict, {"a": float(a), "b": float(b), "c": float(c),
                      "nll": float(res.fun)}


def _temperature_scale_fit(p_tr, y_tr):
    """
    Scalar temperature scaling.  Treats raw probabilities as already
    derived from logits and learns a single multiplicative T that
    rescales those logits before re-applying sigmoid:

        logit(p_cal) = logit(p_raw) / T

    Fit by 1-D Brent search minimising NLL on the training fold.
    T > 1 softens (pulls toward 0.5); T < 1 sharpens.
    """
    import numpy as np
    from scipy.optimize import minimize_scalar
    p = np.clip(np.asarray(p_tr, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y_tr, dtype=float)
    z = np.log(p / (1 - p))   # logits

    def nll(T):
        if T <= 0:
            return 1e9
        zt = z / T
        log_sig = np.where(zt >= 0, -np.log1p(np.exp(-zt)),
                           zt - np.log1p(np.exp(zt)))
        log_one_minus = np.where(zt >= 0, -zt - np.log1p(np.exp(-zt)),
                                 -np.log1p(np.exp(zt)))
        return -float(np.mean(y * log_sig + (1 - y) * log_one_minus))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_hat = float(res.x)

    def predict(p_in):
        p_clip = np.clip(np.asarray(p_in, dtype=float), EPS, 1 - EPS)
        z_in = np.log(p_clip / (1 - p_clip))
        return 1.0 / (1.0 + np.exp(-z_in / T_hat))

    return predict, {"T": T_hat, "nll": float(res.fun)}


def _crossval_predict(p_tr_full, y_tr_full, p_te, fit_fn, k=5,
                       seed=42):
    """K-fold CV: fit calibrator on k-1 folds, predict on the k-th, then
    re-fit on full train and predict on test.  Returns just the test
    predictions (the standard "production" path used by sklearn's
    CalibratedClassifierCV)."""
    import numpy as np
    from sklearn.model_selection import KFold
    # CalibratedClassifierCV averages five fold-models; we mimic that
    # by fitting 5 calibrators on rotating 4/5 train sub-folds and
    # averaging their test predictions.
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    p_te_acc = np.zeros_like(p_te, dtype=float)
    fit_meta = []
    for train_idx, _ in kf.split(p_tr_full):
        predictor, meta = fit_fn(p_tr_full[train_idx], y_tr_full[train_idx])
        p_te_acc += predictor(p_te)
        fit_meta.append(meta)
    return p_te_acc / k, fit_meta


# --------------------------------------------------------------------------- #
# Per-subgroup calibration
# --------------------------------------------------------------------------- #


def _per_subgroup_calibration(y_te, p_te, subgroups, *, n_bins, min_n=20):
    """
    Per-protected-attribute ECE / Brier / Brier-skill on the test split.

    Returns ``{subgroup_name: {group_value: {n, ece, brier, ...}}}``.
    Skips group cells with < ``min_n`` patients (binomial CI too wide
    to draw conclusions; reported separately as ``too_small`` cells).
    """
    from ml.calibration_curve import compute_calibration
    out: Dict[str, Dict] = {}
    for sg_name, sg_values in subgroups.items():
        per_group: Dict[str, Dict] = {}
        sg_arr = sg_values
        # Restrict to test indices
        for g in sorted(set(sg_arr)):
            mask = sg_arr == g
            if int(mask.sum()) < min_n:
                per_group[str(g)] = {"too_small": True, "n": int(mask.sum())}
                continue
            cal = compute_calibration(y_te[mask], p_te[mask],
                                      n_bins=min(n_bins, max(2, int(mask.sum()) // 10)))
            if cal is None:
                per_group[str(g)] = {"too_small": True, "n": int(mask.sum())}
                continue
            per_group[str(g)] = {
                "n":                   cal.n,
                "base_rate":           cal.base_rate,
                "ece":                 cal.ece,
                "mce":                 cal.mce,
                "brier":               cal.brier,
                "brier_skill_score":   cal.brier_skill_score,
            }
        out[sg_name] = per_group
    return out


# --------------------------------------------------------------------------- #
# Scheduler-impact sensitivity
# --------------------------------------------------------------------------- #


def _oracle_calibrated_probs(y_te, p_te, n_bins=10):
    """
    Oracle reference: bin the predicted probabilities and replace each
    patient's probability with the bin's empirical no-show rate.  This
    is the post-hoc "perfectly calibrated" reference -- the same model's
    ranking, but with the probability values aligned with observation.
    """
    import numpy as np
    p = np.asarray(p_te, dtype=float)
    y = np.asarray(y_te, dtype=float)
    n = len(p)
    ranks = np.argsort(p, kind="mergesort")
    boundary_idx = np.linspace(0, n, n_bins + 1, dtype=int)
    out = np.zeros_like(p)
    for b in range(n_bins):
        idx = ranks[boundary_idx[b]:boundary_idx[b + 1]]
        if len(idx):
            out[idx] = float(y[idx].mean())
    return out


def _scheduler_impact(y_te, p_te, *, threshold=OVERBOOK_THRESHOLD):
    """
    How many overbook decisions does this calibration regime produce,
    and how often is the underlying patient actually a no-show?

    The CP-SAT objective overbooks when the predicted no-show
    probability exceeds the configured threshold; the precision of
    that decision (= recovered_no_shows / overbooks_proposed) is the
    operational quality the scheduler cares about.
    """
    import numpy as np
    p = np.asarray(p_te, dtype=float)
    y = np.asarray(y_te, dtype=float)
    overbook_mask = p >= threshold
    n_overbook = int(overbook_mask.sum())
    if n_overbook == 0:
        # No probabilities clear the threshold; no overbooks proposed.
        return {
            "threshold":             round(threshold, 4),
            "n_test":                int(len(y)),
            "n_overbook_proposed":   0,
            "n_recovered_noshows":   0,
            "precision":             None,
            "recall_vs_baseline":    None,
            "n_actual_noshows":      int(y.sum()),
        }
    n_recovered = int(y[overbook_mask].sum())
    precision = float(n_recovered) / n_overbook
    n_actual = int(y.sum())
    recall = (float(n_recovered) / n_actual) if n_actual > 0 else None
    return {
        "threshold":             round(threshold, 4),
        "n_test":                int(len(y)),
        "n_overbook_proposed":   n_overbook,
        "n_recovered_noshows":   n_recovered,
        "precision":             round(precision, 4),
        "recall_vs_baseline":    round(recall, 4) if recall is not None else None,
        "n_actual_noshows":      n_actual,
    }


# --------------------------------------------------------------------------- #
# Main benchmark
# --------------------------------------------------------------------------- #


def benchmark(*, n_patients: int, test_frac: float, n_bins: int, seed: int,
              output_path: Path) -> Dict:
    """
    Train a fresh GBM no-show head, evaluate **five** calibration regimes
    + per-subgroup breakdown + scheduler-impact sensitivity, and append
    one JSONL row per run.
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    from ml.calibration_curve import compute_calibration

    df = _load_cohort(n_patients)
    X, y = _extract_xy(df)
    if y is None or len(y) < 200:
        raise SystemExit("Cohort too small or no label column")
    subgroups_full = _extract_subgroups(df)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1.0 - test_frac))
    tr, te = idx[:cut], idx[cut:]

    if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
        raise SystemExit("Single-class split; cannot evaluate calibration")

    # Test-row subgroup labels
    subgroups_te = {k: v[te] for k, v in subgroups_full.items()}

    base = lambda: GradientBoostingClassifier(           # noqa: E731
        n_estimators=200, max_depth=3,
        random_state=seed, learning_rate=0.05,
    )

    # --- Train the base GBM once for raw + post-hoc regimes ---
    gbm = base(); gbm.fit(X[tr], y[tr])
    p_tr_raw = gbm.predict_proba(X[tr])[:, 1]
    p_te_raw = gbm.predict_proba(X[te])[:, 1]
    auc_raw  = float(roc_auc_score(y[te], p_te_raw))

    regimes: Dict[str, np.ndarray] = {"raw": p_te_raw}

    # Isotonic via sklearn (5-fold CV)
    regimes["isotonic"] = _fit_predict_calibrated("isotonic", base(),
                                                    X[tr], y[tr], X[te])
    # Sigmoid / Platt via sklearn (5-fold CV)
    regimes["sigmoid"] = _fit_predict_calibrated("sigmoid", base(),
                                                   X[tr], y[tr], X[te])
    # Beta calibration via 5-fold CV on raw probabilities
    p_tr_for_beta = p_tr_raw
    p_te_beta, beta_meta = _crossval_predict(
        p_tr_for_beta, y[tr], p_te_raw, _beta_calibrate_fit, k=5, seed=seed,
    )
    regimes["beta"] = p_te_beta
    # Temperature scaling via 5-fold CV
    p_te_temp, temp_meta = _crossval_predict(
        p_tr_for_beta, y[tr], p_te_raw, _temperature_scale_fit, k=5, seed=seed,
    )
    regimes["temperature"] = p_te_temp
    # Oracle reference: bin + observed-rate (post-hoc 'perfectly
    # calibrated' on the held-out probe).
    regimes["oracle"] = _oracle_calibrated_probs(y[te], p_te_raw, n_bins=n_bins)

    # ----- Per-regime metrics -----
    results: Dict[str, Dict] = {}
    aucs: Dict[str, float] = {}
    for name, p_te in regimes.items():
        cal = compute_calibration(y[te], p_te, n_bins=n_bins)
        if cal is None:
            results[name] = {"error": "compute_calibration returned None"}
            continue
        results[name] = cal.to_dict()
        # AUC unchanged within sampling noise for regimes that preserve
        # rank (isotonic preserves monotonically; sigmoid is monotone).
        aucs[name] = float(roc_auc_score(y[te], p_te))

    # ----- Pick best regime by Brier-skill -----
    candidate_names = ["isotonic", "sigmoid", "beta", "temperature"]
    best_name = max(
        candidate_names,
        key=lambda n: results[n].get("brier_skill_score", -1e9),
    )

    # ----- Per-subgroup calibration for the best regime -----
    per_subgroup = _per_subgroup_calibration(
        y[te], regimes[best_name], subgroups_te, n_bins=n_bins,
    )

    # ----- Scheduler-impact sensitivity for every regime -----
    impact = {name: _scheduler_impact(y[te], regimes[name])
              for name in regimes.keys()}

    # ----- Backwards-compat: keep legacy "calibration" / "calibration_isotonic"
    # at the top level so the existing R analysis path keeps working. -----
    cal_raw = results["raw"]
    cal_iso = results["isotonic"]
    delta_brier_skill = round(
        cal_iso.get("brier_skill_score", 0)
        - cal_raw.get("brier_skill_score", 0),
        4,
    )

    print(
        f"\n=== Calibration benchmark (n={len(df)}, seed={seed}) ===\n"
        f"  n_train={len(tr)}, n_test={len(te)}, base rate test={cal_raw['base_rate']:.3f}",
        flush=True,
    )
    for name in ["raw", "isotonic", "sigmoid", "beta", "temperature", "oracle"]:
        r = results[name]
        a = aucs.get(name, float("nan"))
        marker = "  <-- best (non-oracle)" if name == best_name else ""
        print(f"  [{name:>11s}] AUC={a:.3f}  Brier={r['brier']:.4f}  "
              f"skill={r['brier_skill_score']:+.4f}  ECE={r['ece']:.4f}  "
              f"MCE={r['mce']:.4f}{marker}")
    print()
    print(f"  Best non-oracle regime: {best_name}")
    print(f"  Scheduler impact at threshold {OVERBOOK_THRESHOLD}:")
    for name in ["raw", "isotonic", "sigmoid", "beta", "temperature", "oracle"]:
        i = impact[name]
        prec = "n/a" if i["precision"] is None else f"{i['precision']:.3f}"
        rec  = "n/a" if i["recall_vs_baseline"] is None else f"{i['recall_vs_baseline']:.3f}"
        print(f"  [{name:>11s}] proposed={i['n_overbook_proposed']:3d}  "
              f"recovered={i['n_recovered_noshows']:3d}  "
              f"precision={prec}  recall={rec}")

    row = {
        "ts":           datetime.utcnow().isoformat(timespec="seconds"),
        "n":            len(df),
        "n_train":      int(len(tr)),
        "n_test":       int(len(te)),
        "seed":         int(seed),
        "auc":          round(auc_raw, 4),
        "auc_isotonic": round(aucs.get("isotonic", float("nan")), 4),
        "auc_per_regime": {k: round(v, 4) for k, v in aucs.items()},
        "calibration":            cal_raw,
        "calibration_isotonic":   cal_iso,
        "delta_brier_skill_isotonic_minus_raw": delta_brier_skill,
        # New: full multi-regime block
        "calibration_regimes":          results,
        "best_regime":                  best_name,
        "best_regime_brier_skill":      round(
            results[best_name].get("brier_skill_score", 0), 4
        ),
        "best_regime_brier_skill_delta_vs_raw": round(
            results[best_name].get("brier_skill_score", 0)
            - results["raw"].get("brier_skill_score", 0),
            4,
        ),
        "per_subgroup_best_regime":     per_subgroup,
        "scheduler_impact":             impact,
        "scheduler_overbook_threshold": OVERBOOK_THRESHOLD,
        "beta_fit_meta":                beta_meta,
        "temperature_fit_meta":         temp_meta,
        "method":      "GradientBoostingClassifier (n=200, depth=3, lr=0.05) "
                       "on the eight tabular features intersected with the "
                       "production no-show ensemble's input vector.",
        "method_isotonic": (
            "5-fold cross-validated isotonic regression "
            "(sklearn.calibration.CalibratedClassifierCV, method='isotonic', "
            "cv=5) wrapped around the same GradientBoostingClassifier "
            "configuration."
        ),
        "method_calibration_regimes": (
            "Five regimes evaluated on the same held-out test split: "
            "raw GBM probabilities; sklearn-CV isotonic and sigmoid "
            "(Platt) wrappers; inline-implemented Beta calibration "
            "(Kull et al. 2017) and scalar temperature scaling, both "
            "fit by 5-fold CV on the training-fold raw probabilities. "
            "An oracle reference -- bin + observed-rate on the test "
            "probe -- bounds what perfectly calibrated probabilities "
            "would deliver from the same ranking."
        ),
        "method_per_subgroup": (
            "After picking the best non-oracle regime by Brier-skill, "
            "compute ECE/MCE/Brier/skill per protected subgroup "
            "(Gender, Age_Band, Site_Code, Priority).  Group cells "
            "with n < 20 are flagged too_small and excluded from the "
            "headline."
        ),
        "method_scheduler_impact": (
            f"At the production overbook threshold "
            f"({OVERBOOK_THRESHOLD}), count overbook proposals and "
            "compare against observed no-shows to derive precision + "
            "recall.  Comparing each regime against the oracle "
            "quantifies the operational cost of mis-calibrated "
            "probabilities at the CP-SAT decision boundary."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients", type=int, default=1500)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--n-bins",    type=int, default=10)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",
                        default="data_cache/calibration/results.jsonl")
    args = parser.parse_args()
    benchmark(
        n_patients=args.n_patients,
        test_frac=args.test_frac,
        n_bins=args.n_bins,
        seed=args.seed,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
