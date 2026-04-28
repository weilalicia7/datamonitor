"""
SHAP feature-importance benchmark for the stacked ensemble used in
Table 5.2.  Mirrors the production train pipeline, computes Tree-SHAP
on each base learner, aggregates by mean of mean-absolute SHAP across
RF/GB/XGB, reports top-10 features.

Output: ``data_cache/shap_importance/results.jsonl``
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "data_cache" / "shap_importance"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_OUT_PATH = _OUT_DIR / "results.jsonl"

SEED = 42


def _safe_get(row, key, default):
    val = row.get(key, default)
    return default if pd.isna(val) else val


def _build_features():
    df = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" / "historical_appointments.xlsx")
    rows, labels = [], []
    for _, row in df.iterrows():
        rows.append({
            "expected_duration":   _safe_get(row, "Planned_Duration", 120),
            "cycle_number":        _safe_get(row, "Cycle_Number", 1),
            "age":                 _safe_get(row, "Age", 55),
            "prev_noshow_rate":    _safe_get(row, "Patient_NoShow_Rate", 0.1),
            "travel_distance_km":  _safe_get(row, "Travel_Distance_KM", 10),
            "treatment_complexity":_safe_get(row, "Complexity_Factor", 0.5),
            "comorbidity_count":   _safe_get(row, "Comorbidity_Count", 0),
            "duration_variance":   _safe_get(row, "Duration_Variance", 0.15),
            "appointment_hour":    _safe_get(row, "Appointment_Hour", 10),
            "day_of_week":         _safe_get(row, "Day_Of_Week_Num", 2),
            "is_first_cycle":      int(bool(_safe_get(row, "Is_First_Cycle", False))),
        })
        labels.append(1 if row.get("Attended_Status") == "No" else 0)
    return pd.DataFrame(rows).astype(float).fillna(0), pd.Series(labels)


def main():
    np.random.seed(SEED)
    X, y = _build_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    Xtr = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    Xte = pd.DataFrame(scaler.transform(X_test),  columns=X.columns)

    base_specs = {
        "RandomForest":     RandomForestClassifier(n_estimators=100, max_depth=10,
                                                    min_samples_split=5, min_samples_leaf=2,
                                                    random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                        learning_rate=0.1, random_state=SEED),
    }
    if XGB_AVAILABLE:
        base_specs["XGBoost"] = xgb.XGBClassifier(n_estimators=100, max_depth=6,
                                                   learning_rate=0.1, random_state=SEED,
                                                   use_label_encoder=False,
                                                   eval_metric="logloss")

    feature_names = list(X.columns)
    mean_abs = np.zeros(len(feature_names))
    feature_outcome_corr = {}
    for j, fname in enumerate(feature_names):
        feature_outcome_corr[fname] = float(np.corrcoef(Xte.iloc[:, j], y_test)[0, 1])

    for name, model in base_specs.items():
        model.fit(Xtr, y_train)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(Xte)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]                # positive class
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]          # newer shap API: (n, p, 2)
        mean_abs += np.abs(shap_vals).mean(axis=0)
        print(f"{name} done")

    mean_abs /= len(base_specs)
    ranked = sorted(zip(feature_names, mean_abs.tolist()),
                    key=lambda x: -x[1])

    def _direction(fname, corr):
        if corr > 0.05:
            return "Positive"
        if corr < -0.05:
            return "Negative"
        return "Mixed"

    print("\nTop-10 features by mean |SHAP|, with feature-outcome correlation direction:")
    for f, v in ranked[:10]:
        c = feature_outcome_corr[f]
        print(f"  {f:>22}  {v:.4f}  corr={c:+.3f}  ({_direction(f, c)})")

    out = {
        "ts": datetime.utcnow().isoformat(),
        "seed": SEED,
        "n_test": int(len(Xte)),
        "ranked": ranked,
        "feature_outcome_corr": feature_outcome_corr,
    }
    with _OUT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"\nWrote {_OUT_PATH}")
    return out


if __name__ == "__main__":
    main()
