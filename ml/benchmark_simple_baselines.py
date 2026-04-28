"""
Simple baselines benchmark — Logistic Regression + Decision Tree
=================================================================

Trains the two simple-baseline classifiers used in dissertation Table 5.1
on the same training data, train/test split, scaling, and feature set
used by the production no-show ensemble.  Reports AUC, precision, recall,
F1, and 5-fold cross-validated AUC.

Also computes the DeLong test for paired AUC comparison between the
stacked ensemble and each baseline (resolves the §3.6.1 DeLong claim).

Output: ``data_cache/baselines/results.jsonl`` (one row per run).
Mirrors the patient-feature pipeline from ``flask_app.py::train_advanced_ml_models``.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
)
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score
)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "data_cache" / "baselines"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_OUT_PATH = _OUT_DIR / "results.jsonl"

SEED = 42
TEST_SIZE = 0.20


def _safe_get(row, key, default):
    val = row.get(key, default)
    if pd.isna(val):
        return default
    return val


def _build_features():
    """Mirror train_advanced_ml_models feature construction."""
    df = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" / "historical_appointments.xlsx")
    patients_list = []
    noshow_labels = []
    for _, row in df.iterrows():
        patients_list.append({
            "expected_duration":   _safe_get(row, "Planned_Duration", 120),
            "cycle_number":        _safe_get(row, "Cycle_Number", 1),
            "age":                 _safe_get(row, "Age", 55),
            "noshow_rate":         _safe_get(row, "Patient_NoShow_Rate", 0.1),
            "distance_km":         _safe_get(row, "Travel_Distance_KM", 10),
            "complexity_factor":   _safe_get(row, "Complexity_Factor", 0.5),
            "comorbidity_count":   _safe_get(row, "Comorbidity_Count", 0),
            "duration_variance":   _safe_get(row, "Duration_Variance", 0.15),
            "appointment_hour":    _safe_get(row, "Appointment_Hour", 10),
            "day_of_week":         _safe_get(row, "Day_Of_Week_Num", 2),
            "is_first_cycle":      int(bool(_safe_get(row, "Is_First_Cycle", False))),
        })
        noshow_labels.append(1 if row.get("Attended_Status") == "No" else 0)
    X = pd.DataFrame(patients_list).astype(float).fillna(0)
    y = pd.Series(noshow_labels)
    return X, y


def _evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        "auc_roc":   roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "y_prob":    y_prob.tolist(),
    }


def _cv_auc(model, X, y):
    """5-fold stratified CV AUC (manual to avoid XGB sklearn-API edge cases)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in cv.split(X, y):
        m = type(model)(**model.get_params())
        m.fit(np.asarray(X)[tr], np.asarray(y)[tr])
        prob = m.predict_proba(np.asarray(X)[te])[:, 1]
        scores.append(roc_auc_score(np.asarray(y)[te], prob))
    return float(np.mean(scores)), float(np.std(scores))


def _delong_test(y_true, prob_a, prob_b):
    """
    DeLong's test for two correlated AUCs (DeLong, DeLong & Clarke-Pearson 1988).
    Returns z-statistic and two-sided p-value.
    """
    y_true = np.asarray(y_true, dtype=int)
    pos = y_true == 1
    neg = ~pos
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    def _midrank(x):
        x = np.asarray(x, dtype=float)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        i = 0
        n = len(x)
        while i < n:
            j = i
            while j < n - 1 and x[order[j + 1]] == x[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return ranks

    aucs = []
    v01_list, v10_list = [], []
    for prob in (prob_a, prob_b):
        prob = np.asarray(prob, dtype=float)
        tx, ty = prob[pos], prob[neg]
        tz = np.concatenate([tx, ty])
        rk = _midrank(tz)
        rkx = _midrank(tx)
        rky = _midrank(ty)
        auc = (rk[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        v01 = (rk[:n_pos] - rkx) / n_neg
        v10 = 1.0 - (rk[n_pos:] - rky) / n_pos
        aucs.append(auc)
        v01_list.append(v01)
        v10_list.append(v10)

    s01 = np.cov(np.vstack(v01_list))
    s10 = np.cov(np.vstack(v10_list))
    s = s01 / n_pos + s10 / n_neg
    auc_diff = aucs[0] - aucs[1]
    var = s[0, 0] + s[1, 1] - 2.0 * s[0, 1]
    if var <= 0:
        return float("nan"), float("nan")
    z = auc_diff / np.sqrt(var)
    p = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z)))
    return float(z), float(p)


def main():
    np.random.seed(SEED)
    X, y = _build_features()
    print(f"Loaded {len(X)} samples; positives = {int(y.sum())} ({y.mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    out = {"ts": datetime.utcnow().isoformat(), "seed": SEED, "test_size": TEST_SIZE,
           "n_train": int(len(X_train)), "n_test": int(len(X_test)),
           "models": {}}

    base_specs = [
        ("LogisticRegression", LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)),
        ("DecisionTree",       DecisionTreeClassifier(random_state=SEED)),
        ("RandomForest",       RandomForestClassifier(n_estimators=100, max_depth=10,
                                                       min_samples_split=5,
                                                       min_samples_leaf=2,
                                                       random_state=SEED, n_jobs=-1)),
        ("GradientBoosting",   GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                           learning_rate=0.1,
                                                           random_state=SEED)),
    ]
    if XGB_AVAILABLE:
        base_specs.append(
            ("XGBoost", xgb.XGBClassifier(n_estimators=100, max_depth=6,
                                          learning_rate=0.1, random_state=SEED,
                                          use_label_encoder=False,
                                          eval_metric="logloss"))
        )

    base_oof = {}
    for name, model in base_specs:
        ev = _evaluate(model, Xtr, Xte, y_train, y_test)
        cv_mean, cv_std = _cv_auc(
            type(model)(**model.get_params()), Xtr, y_train
        )
        out["models"][name] = {
            "auc_roc":     round(ev["auc_roc"], 3),
            "precision":   round(ev["precision"], 2),
            "recall":      round(ev["recall"], 2),
            "f1":          round(ev["f1"], 2),
            "cv_auc_mean": round(cv_mean, 3),
            "cv_auc_std":  round(cv_std, 3),
            "y_prob":      ev["y_prob"],
        }
        print(f"{name:>20}  AUC={ev['auc_roc']:.3f}  P={ev['precision']:.2f}  "
              f"R={ev['recall']:.2f}  F1={ev['f1']:.2f}  "
              f"CV={cv_mean:.3f}+/-{cv_std:.3f}")

    # Stacked ensemble: train RF + GB + XGB on OOF, meta = LR
    print()
    print("Training stacked ensemble (OOF + LR meta-learner)...")
    stack_models = [m for n, m in base_specs if n in ("RandomForest", "GradientBoosting", "XGBoost")]
    oof_train = np.zeros((len(Xtr), len(stack_models)))
    oof_test = np.zeros((len(Xte), len(stack_models)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for j, m in enumerate(stack_models):
        oof_train[:, j] = cross_val_predict(
            type(m)(**m.get_params()), Xtr, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        m.fit(Xtr, y_train)
        oof_test[:, j] = m.predict_proba(Xte)[:, 1]
    meta = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    meta.fit(oof_train, y_train)
    y_prob_stk = meta.predict_proba(oof_test)[:, 1]
    y_pred_stk = meta.predict(oof_test)
    cv_meta_mean, cv_meta_std = _cv_auc(
        LogisticRegression(C=1.0, random_state=SEED, max_iter=1000), oof_train, y_train
    )
    out["models"]["Stacked"] = {
        "auc_roc":     round(roc_auc_score(y_test, y_prob_stk), 3),
        "precision":   round(precision_score(y_test, y_pred_stk, zero_division=0), 2),
        "recall":      round(recall_score(y_test, y_pred_stk, zero_division=0), 2),
        "f1":          round(f1_score(y_test, y_pred_stk, zero_division=0), 2),
        "cv_auc_mean": round(cv_meta_mean, 3),
        "cv_auc_std":  round(cv_meta_std, 3),
        "y_prob":      y_prob_stk.tolist(),
    }
    s = out["models"]["Stacked"]
    print(f"{'Stacked':>20}  AUC={s['auc_roc']:.3f}  P={s['precision']:.2f}  "
          f"R={s['recall']:.2f}  F1={s['f1']:.2f}  "
          f"CV={cv_meta_mean:.3f}+/-{cv_meta_std:.3f}")

    # DeLong test: Stacked vs each baseline
    print()
    print("DeLong test (Stacked vs each model):")
    delong_results = {}
    for name in ("LogisticRegression", "DecisionTree", "GradientBoosting"):
        if name in out["models"]:
            z, p = _delong_test(np.asarray(y_test),
                                 np.asarray(y_prob_stk),
                                 np.asarray(out["models"][name]["y_prob"]))
            delong_results[name] = {"z": round(z, 3), "p": round(p, 4)}
            print(f"  Stacked vs {name:>20}: z={z:+.3f}  p={p:.4f}")
    out["delong_stacked_vs"] = delong_results

    out_path = _OUT_PATH
    with out_path.open("a", encoding="utf-8") as fh:
        printable = {k: v for k, v in out.items() if k != "models"}
        printable["models"] = {n: {k: v for k, v in d.items() if k != "y_prob"}
                              for n, d in out["models"].items()}
        fh.write(json.dumps(printable) + "\n")
    print(f"\nWrote {out_path}")
    return out


if __name__ == "__main__":
    main()
