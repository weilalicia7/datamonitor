"""
Table 5.5 Statistical Significance aggregator.

Joins:
  * Paired t-tests (util & wait) from data_cache/100days/results.jsonl
  * DeLong tests (Stacked vs LR / DT / GB) from data_cache/baselines/results.jsonl
  * Weather LPM ATE on historical_appointments.xlsx (HC1 robust SE)
  * Chair_Number -> no-show Pearson correlation + p (falsification placebo)

Applies Benjamini-Hochberg FDR correction at q=0.05 across the full row set.
Writes one row per run to data_cache/table55/results.jsonl.

Mirrors the R-side LPM computation in dissertation/dissertation_analysis.R
Section 4 so the dissertation macro \\ateweather and the §5.6 Table 5.5
agree from the same point estimate.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR   = _REPO_ROOT / "data_cache" / "table55"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_OUT_PATH  = _OUT_DIR / "results.jsonl"

_HUNDRED_PATH  = _REPO_ROOT / "data_cache" / "100days"   / "results.jsonl"
_BASELINE_PATH = _REPO_ROOT / "data_cache" / "baselines" / "results.jsonl"
_HIST_PATH     = _REPO_ROOT / "datasets"   / "sample_data" / "historical_appointments.xlsx"


def _load_last_jsonl(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _bh_fdr(rows: List[Dict], q: float = 0.05) -> None:
    """Mutate rows in place: add 'bh_significant' (bool) and
    'bh_threshold' (the BH critical value at the row's rank) columns.

    Rows missing 'p' are skipped (kept as None).
    """
    indexed = [(i, r["p"]) for i, r in enumerate(rows) if r.get("p") is not None]
    indexed.sort(key=lambda t: t[1])
    m = len(indexed)
    if m == 0:
        return
    # Find largest rank k such that p_(k) <= q*k/m
    cutoff_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        thr = q * rank / m
        if p <= thr:
            cutoff_rank = rank
    for rank, (orig_i, p) in enumerate(indexed, start=1):
        thr = q * rank / m
        rows[orig_i]["bh_threshold"] = round(thr, 6)
        rows[orig_i]["bh_rank"] = rank
        rows[orig_i]["bh_m"] = m
        rows[orig_i]["bh_significant"] = bool(rank <= cutoff_rank)


def _weather_ate_lpm() -> Dict:
    """Linear Probability Model (LPM) with HC1 robust SE.

    is_noshow ~ Weather_Severity + Travel_Distance_KM
                + C(Priority) + C(Site_Code)

    Mirrors dissertation_analysis.R Section 4 so the value reproduces
    the macro \\ateweather.
    """
    df = pd.read_excel(_HIST_PATH)
    df = df.copy()
    df["is_noshow"] = (df["Attended_Status"] == "No").astype(float)
    cols_required = ("Weather_Severity", "Travel_Distance_KM", "Priority", "Site_Code")
    df = df.dropna(subset=list(cols_required) + ["is_noshow"])

    # Build design matrix manually so we don't pull in patsy/statsmodels
    # if they aren't installed; fall back to statsmodels if available
    # for HC1 SE (more reliable than rolling our own sandwich estimator).
    try:
        import statsmodels.api as sm
        priority_d = pd.get_dummies(df["Priority"], prefix="Priority", drop_first=True)
        site_d     = pd.get_dummies(df["Site_Code"], prefix="Site",   drop_first=True)
        X = pd.concat([df[["Weather_Severity", "Travel_Distance_KM"]],
                       priority_d.astype(float), site_d.astype(float)], axis=1)
        X = sm.add_constant(X)
        X = X.astype(float)
        y = df["is_noshow"].astype(float)
        model = sm.OLS(y, X).fit(cov_type="HC1")
        coef    = float(model.params["Weather_Severity"])
        se      = float(model.bse["Weather_Severity"])
        p_val   = float(model.pvalues["Weather_Severity"])
        ci_lo, ci_hi = (float(x) for x in model.conf_int().loc["Weather_Severity"])
        n       = int(model.nobs)
        return {"ate": coef, "se": se, "p": p_val,
                "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n,
                "method": "OLS-LPM with HC1 robust SE (statsmodels)"}
    except ImportError:
        # Degraded fallback: OLS without robust SE.  Documented as such
        # so the dissertation can flag the fallback in the caption.
        priority_d = pd.get_dummies(df["Priority"], prefix="Priority", drop_first=True)
        site_d     = pd.get_dummies(df["Site_Code"], prefix="Site",   drop_first=True)
        X = pd.concat([df[["Weather_Severity", "Travel_Distance_KM"]],
                       priority_d.astype(float), site_d.astype(float)], axis=1)
        X.insert(0, "const", 1.0)
        X = X.values.astype(float)
        y = df["is_noshow"].values.astype(float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        n, k = X.shape
        sigma2 = float((resid @ resid) / (n - k))
        XtX_inv = np.linalg.inv(X.T @ X)
        var_beta = sigma2 * XtX_inv
        ix = 1   # Weather_Severity is column 1 after constant
        coef = float(beta[ix])
        se   = float(np.sqrt(var_beta[ix, ix]))
        t_stat = coef / se
        p_val  = float(2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=n - k)))
        crit   = float(sp_stats.t.ppf(0.975, df=n - k))
        return {"ate": coef, "se": se, "p": p_val,
                "ci_lo": coef - crit * se, "ci_hi": coef + crit * se,
                "n": n,
                "method": "OLS-LPM (homoskedastic SE; statsmodels unavailable)"}


def _chair_noshow_placebo() -> Dict:
    """Pearson correlation + p-value: Chair_Number -> no-show.

    Falsification test: the chair a patient is assigned to should not
    causally drive their attendance.  Significant p would suggest
    confounding (e.g., sicker patients consistently routed to one chair).
    """
    df = pd.read_excel(_HIST_PATH)
    if "Chair_Number" not in df.columns or "Attended_Status" not in df.columns:
        return {"r": None, "p": None, "n": 0,
                "method": "missing Chair_Number / Attended_Status"}
    chair = pd.to_numeric(df["Chair_Number"], errors="coerce")
    noshow = (df["Attended_Status"] == "No").astype(float)
    valid = chair.notna() & noshow.notna()
    if valid.sum() < 50:
        return {"r": None, "p": None, "n": int(valid.sum()),
                "method": "n<50, skipped"}
    r, p = sp_stats.pearsonr(chair[valid], noshow[valid])
    return {"r": float(r), "p": float(p), "n": int(valid.sum()),
            "method": "Pearson correlation, two-sided"}


def main() -> Dict:
    h = _load_last_jsonl(_HUNDRED_PATH)
    b = _load_last_jsonl(_BASELINE_PATH)
    if h is None:
        raise RuntimeError(f"No 100-day benchmark found at {_HUNDRED_PATH}.  Run benchmark_100days.py first.")
    if b is None:
        raise RuntimeError(f"No baselines benchmark found at {_BASELINE_PATH}.  Run benchmark_simple_baselines.py first.")
    if "paired_tests" not in h:
        raise RuntimeError(
            "100-day benchmark is missing 'paired_tests'; re-run benchmark_100days.py "
            "after the per-seed paired-test extension.")

    # === Build rows ===
    rows: List[Dict] = []

    # AUC: Stacked vs LR / DT / GB (DeLong)
    delong = b.get("delong_stacked_vs", {})
    auc_models = b.get("models", {})
    auc_stacked = auc_models.get("Stacked", {}).get("auc_roc")
    for label, key in [("AUC: Stacked vs LR baseline",      "LogisticRegression"),
                        ("AUC: Stacked vs Decision Tree",    "DecisionTree"),
                        ("AUC: Stacked vs Gradient Boosting","GradientBoosting")]:
        if key not in delong:
            continue
        d = delong[key]
        auc_other = auc_models.get(key, {}).get("auc_roc")
        diff = (round(auc_stacked - auc_other, 4)
                if (auc_stacked is not None and auc_other is not None) else None)
        rows.append({
            "test": label,
            "statistic_kind": "z",
            "statistic_value": d.get("z"),
            "p": d.get("p"),
            "ci_lo": None, "ci_hi": None,
            "delta": diff,
            "delta_unit": "AUC",
            "source": "DeLong, ml/benchmark_simple_baselines.py",
        })

    # Utilisation: paired t-test (CP-SAT - Patrick) across 100 days
    pt_util = h["paired_tests"]["util_cpsat_minus_patrick_pp"]
    rows.append({
        "test": r"Utilisation: \system{} vs Patrick baseline",
        "statistic_kind": "t",
        "statistic_value": pt_util["t_stat"],
        "p": pt_util["p_value"],
        "ci_lo": pt_util["ci_lo"], "ci_hi": pt_util["ci_hi"],
        "delta": pt_util["mean_diff"],
        "delta_unit": "pp",
        "source": "paired t, n=" + str(pt_util["n"]) + " days, ml/benchmark_100days.py",
    })

    # Wait time: paired t-test (CP-SAT - Patrick) across 100 days
    pt_wait = h["paired_tests"]["wait_cpsat_minus_patrick_min"]
    rows.append({
        "test": r"Wait time: \system{} vs Patrick baseline",
        "statistic_kind": "t",
        "statistic_value": pt_wait["t_stat"],
        "p": pt_wait["p_value"],
        "ci_lo": pt_wait["ci_lo"], "ci_hi": pt_wait["ci_hi"],
        "delta": pt_wait["mean_diff"],
        "delta_unit": "min",
        "source": "paired t, n=" + str(pt_wait["n"]) + " days, ml/benchmark_100days.py",
    })

    # Weather ATE (LPM with HC1)
    w = _weather_ate_lpm()
    rows.append({
        "test": r"ATE (weather $\to$ no-show)",
        "statistic_kind": "LPM",
        "statistic_value": round(w["ate"], 4),
        "p": w["p"],
        "ci_lo": round(w["ci_lo"], 4), "ci_hi": round(w["ci_hi"], 4),
        "delta": round(w["ate"], 4),
        "delta_unit": "pp",
        "source": w["method"] + ", n=" + str(w["n"]),
    })

    # Causal placebo: Chair_Number -> no-show
    c = _chair_noshow_placebo()
    rows.append({
        "test": r"Falsification: Chair $\to$ no-show",
        "statistic_kind": "r",
        "statistic_value": round(c["r"], 4) if c["r"] is not None else None,
        "p": c["p"],
        "ci_lo": None, "ci_hi": None,
        "delta": None, "delta_unit": None,
        "source": c["method"] + ", n=" + str(c["n"]),
    })

    # Apply BH-FDR at q=0.05
    _bh_fdr(rows, q=0.05)

    out = {
        "ts": datetime.utcnow().isoformat(),
        "fdr_q": 0.05,
        "fdr_method": "Benjamini-Hochberg",
        "n_hypotheses": sum(1 for r in rows if r.get("p") is not None),
        "rows": rows,
    }

    # Print a console summary
    print("\n===== Table 5.5 Statistical Significance =====")
    print(f"BH-FDR at q=0.05 across m={out['n_hypotheses']} hypotheses\n")
    print(f"{'Test':<45}  {'Stat':>10}  {'p':>10}  {'BH thr':>10}  {'BH Y':>5}")
    for r in rows:
        sval = (f"{r['statistic_kind']}={r['statistic_value']:+.3f}"
                if r["statistic_value"] is not None else "—")
        pstr = f"{r['p']:.4g}" if r["p"] is not None else "—"
        thr  = f"{r['bh_threshold']:.4f}" if r.get("bh_threshold") is not None else "—"
        sig  = "Yes" if r.get("bh_significant") else "No"
        print(f"{r['test']:<45}  {sval:>10}  {pstr:>10}  {thr:>10}  {sig:>5}")

    with _OUT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, default=float) + "\n")
    print(f"\nWrote {_OUT_PATH}")
    return out


if __name__ == "__main__":
    main()
