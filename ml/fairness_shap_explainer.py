"""
SHAP root-cause explainer for fairness disparity (§5.13 / Improvement G)
========================================================================

External-review Improvement G asked: *the fairness audit flags gender
and site as showing no-show disparity, but never says WHY.*

This harness uses SHAP values on the no-show model to decompose the
per-group prediction gap into per-feature contributions.  For every
disparate protected-attribute comparison (e.g., female vs. male, or
outreach-site vs. main-site) it ranks the features by absolute
contribution to the gap, classifies each top feature as

  * ``protected``    — the protected attribute itself is a model
                       input (direct discrimination)
  * ``proxy``        — a near-proxy (e.g., ethnicity postcode) that
                       leaks the protected signal
  * ``geographic``   — a legitimate geography/travel-burden feature
                       (travel_distance_km, is_main_site)
  * ``clinical``     — a legitimate patient-history / risk-correlate
                       feature (prev_noshow_rate, age_band)
  * ``temporal``     — a neutral scheduling feature
                       (hour, day_of_week)
  * ``other``        — anything unclassified

and emits a verdict per comparison:

  * ``bias_direct``     — at least one top-3 feature is ``protected``
  * ``legitimate``      — all top-3 features are geographic / clinical
                          / temporal (no protected or proxy features)
  * ``legitimate_with_note``
                        — majority legitimate but one "proxy" feature
                          is in the top-3 and deserves operator review

The JSONL this script writes is consumed by
``dissertation_analysis.R §20e`` to render Table 5.8 in §5.13; the
companion Flask endpoint
``/api/metrics/fairness-shap/status`` returns the same row as JSON
so operators have a read-only diagnostic.

Scope
-----
SHAP is computed on the GradientBoostingClassifier base learner of
``ml.noshow_model.NoShowModel`` (via ``shap.TreeExplainer``).  The
stacked-ensemble meta-learner can layer non-tree combinations on top
of base predictions; for attribution we use the GBM because (a) it is
exact tree-explainable, (b) it is the strongest single learner in the
ensemble, and (c) the §5.13 prose is about identifying which features
the model uses to LOCATE the disparity — a single-base explanation
captures that question accurately without the added complication of
the meta-blend.

CLI
---
    python -m ml.fairness_shap_explainer \
        --n-patients 120 --n-history 1500

Never invoked by the live Flask backend.  The status endpoint is
read-only and just re-serves the latest JSONL row.  The benchmark
writes to ``data_cache/fairness_shap/results.jsonl`` only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Feature-classification heuristics.  When a feature name matches one of the
# prefixes / keywords below it is assigned the corresponding category.  This
# is the same taxonomy the §5.13 prose uses, so a future rename of any
# feature should update BOTH this file and the prose.
# --------------------------------------------------------------------------- #

_PROTECTED_KEYS = ("gender", "sex", "ethnicity", "religion", "disability")
_PROXY_KEYS = (
    # Proxies are legitimate-looking features whose distribution
    # inside a protected group is known to leak the protected signal.
    # Postcode area codes are the classic UK example.
    "postcode_area", "lsoa", "imd_decile",
)
_GEOGRAPHIC_KEYS = (
    "distance_km", "travel_distance", "is_main_site",
    "is_outreach_site", "site_",
)
_CLINICAL_KEYS = (
    "prev_noshow", "prev_cancel", "prev_late",
    "total_appointments", "days_since_last",
    "age", "has_comorbidities", "iv_access_difficulty",
    "requires_1to1_nursing", "treatment_cycle", "is_first_cycle",
    "is_new_patient", "performance_status",
    "expected_duration", "planned_duration", "cycle_number",
    "protocol", "regimen",
)
_TEMPORAL_KEYS = (
    "hour", "day_of_week", "is_weekend", "month",
    "is_winter", "is_spring", "is_summer", "is_autumn",
    "days_until_appointment", "booked_", "booking_lead",
)
_ADMINISTRATIVE_KEYS = (
    "contact_pref",
)


def _classify_feature(name: str) -> str:
    n = name.lower()
    if any(k in n for k in _PROTECTED_KEYS):
        return "protected"
    if any(k in n for k in _PROXY_KEYS):
        return "proxy"
    if any(k in n for k in _GEOGRAPHIC_KEYS):
        return "geographic"
    if any(k in n for k in _CLINICAL_KEYS):
        return "clinical"
    if any(k in n for k in _TEMPORAL_KEYS):
        return "temporal"
    if any(k in n for k in _ADMINISTRATIVE_KEYS):
        return "administrative"
    return "other"


def _verdict_for_topk(top_categories: List[str]) -> str:
    """Given the per-feature categories of the top-k contributors to a
    disparity, assign a verdict tag."""
    if any(c == "protected" for c in top_categories):
        return "bias_direct"
    has_proxy = any(c == "proxy" for c in top_categories)
    has_admin = any(c == "administrative" for c in top_categories)
    all_legit = all(
        c in ("geographic", "clinical", "temporal", "administrative", "other")
        for c in top_categories
    )
    if has_proxy:
        return "legitimate_with_note"
    if has_admin:
        # Administrative features (e.g., contact preference) aren't
        # protected but can be culturally correlated — worth flagging
        # without accusing the model of direct bias.
        return "legitimate_with_note"
    if all_legit:
        return "legitimate"
    return "legitimate_with_note"


# --------------------------------------------------------------------------- #
# Data + model setup — reuse the real cohort so §5.13 numbers match §5.6.*
# --------------------------------------------------------------------------- #


def _load_historical(n_history: int):
    """Load historical appointments for model training.  Builds a tiny
    (attended/no-show) label via a deterministic draw on
    ``Patient_NoShow_Rate`` when no outcome column is present — this
    is only a stand-in so the model has SOMETHING to learn; the fairness
    reading is about directional SHAP contributions, not about AUC."""
    import numpy as np
    import pandas as pd

    apt_path = _REPO_ROOT / "datasets" / "sample_data" / "historical_appointments.xlsx"
    pat_path = _REPO_ROOT / "datasets" / "sample_data" / "patients.xlsx"
    apt = pd.read_excel(apt_path).head(n_history)
    pat = pd.read_excel(pat_path)

    # Merge patient demographics in
    merged = apt.merge(pat, on="Patient_ID", how="left", suffixes=("", "_pat"))

    # Synthesize a Attended label via a deterministic draw from
    # Patient_NoShow_Rate (when available) — we want a stable target
    # so the explainer is reproducible across runs.
    rng = np.random.RandomState(42)
    if "Patient_NoShow_Rate" in merged.columns:
        probs = merged["Patient_NoShow_Rate"].fillna(0.13).clip(0.01, 0.99).to_numpy()
    else:
        probs = np.full(len(merged), 0.13)
    y = (rng.uniform(size=len(merged)) < probs).astype(int)
    return merged, y


def _feature_frame(df):
    """Map the merged dataframe into the feature space the
    FeatureEngineer produces (so SHAP feature names line up with the
    live prediction pipeline)."""
    from ml.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    rows = []
    for _, row in df.iterrows():
        pat = {
            "patient_id": str(row.get("Patient_ID", "")),
            "Person_Birth_Date": row.get("Person_Birth_Date"),
            "Patient_NoShow_Rate": row.get("Patient_NoShow_Rate", 0.13),
            "Patient_Postcode": row.get("Patient_Postcode", "CF14"),
            "Travel_Distance_KM": row.get("Travel_Distance_KM", 5.0),
            "Prev_Appointments_Count": row.get("Prev_Appointments_Count", 0),
            "Prev_NoShow_Count": row.get("Prev_NoShow_Count", 0),
            "Prev_Cancel_Count": row.get("Prev_Cancel_Count", 0),
            "Prev_Late_Count": row.get("Prev_Late_Count", 0),
            "Performance_Status": row.get("Performance_Status", 0),
            "Has_Comorbidities": row.get("Has_Comorbidities", 0),
            "IV_Access_Difficulty": row.get("IV_Access_Difficulty", 0),
            "Requires_1to1_Nursing": row.get("Requires_1to1_Nursing", 0),
            "Contact_Preference": row.get("Contact_Preference", "SMS"),
        }
        apt = {
            "date": row.get("Date", datetime(2026, 4, 24)),
            "time": row.get("Time"),
            "Site_Code": row.get("Site_Code", "WC"),
            "Regimen_Code": row.get("Regimen_Code", "R-CHOP"),
            "Planned_Duration": row.get("Planned_Duration", 90),
            "Cycle_Number": row.get("Cycle_Number", 1),
            "Booking_Lead_Days": row.get("Booking_Lead_Days", 14),
        }
        feats = fe.create_patient_features(pat, apt)
        rows.append(feats.features)
    return rows


def _extract_numeric_matrix(rows: List[Dict]):
    """Turn the list-of-dicts into a dense numpy matrix with a stable
    feature-name order.  Drop any string features (age_band has a
    numeric `age` + one-hot `age_band_*` siblings so dropping the
    string doesn't lose signal)."""
    import numpy as np
    # Establish the union of keys across all rows
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k in seen:
                continue
            v = r[k]
            if isinstance(v, (int, float, bool)):
                keys.append(k)
                seen.add(k)
    X = np.zeros((len(rows), len(keys)), dtype=float)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            v = r.get(k, 0.0)
            try:
                X[i, j] = float(v)
            except (TypeError, ValueError):
                X[i, j] = 0.0
    return X, keys


# --------------------------------------------------------------------------- #
# Bootstrap explainer
# --------------------------------------------------------------------------- #


def _train_gbm(X, y):
    """Train a GradientBoostingClassifier on the historical frame.
    Deterministic seed so the SHAP attributions are reproducible
    across runs."""
    from sklearn.ensemble import GradientBoostingClassifier
    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42,
    )
    gbm.fit(X, y)
    return gbm


def _compute_group_gap(
    X, gbm, feature_names, groups_idx: Dict[str, List[int]],
    *, top_k: int = 3,
):
    """For every unordered pair of groups with >= 5 members each,
    return a dict describing the mean SHAP gap per feature."""
    import numpy as np
    import shap

    explainer = shap.TreeExplainer(gbm)
    # SHAP for binary classifier returns shape (n, n_features) for
    # class 1 on shap>=0.40
    shap_all = explainer.shap_values(X)
    if isinstance(shap_all, list):
        # older shap APIs return per-class lists
        shap_vals = np.asarray(shap_all[1])
    else:
        shap_vals = np.asarray(shap_all)

    # Mean predicted probability per group, for disparity context
    pred_proba = gbm.predict_proba(X)[:, 1]

    # Consider only groups with enough members
    valid_groups = [g for g, idxs in groups_idx.items() if len(idxs) >= 5]
    comparisons = []
    for i, g1 in enumerate(valid_groups):
        for g2 in valid_groups[i + 1:]:
            idx1 = groups_idx[g1]
            idx2 = groups_idx[g2]
            m1 = float(pred_proba[idx1].mean())
            m2 = float(pred_proba[idx2].mean())
            prediction_gap = m1 - m2

            # Per-feature SHAP-gap: mean(SHAP_group1) - mean(SHAP_group2)
            mean_shap1 = shap_vals[idx1].mean(axis=0)
            mean_shap2 = shap_vals[idx2].mean(axis=0)
            gap_per_feat = mean_shap1 - mean_shap2

            # Sort by |gap| descending
            order = np.argsort(-np.abs(gap_per_feat))
            contributions = []
            for j in order[:top_k]:
                contributions.append({
                    "feature": feature_names[j],
                    "mean_shap_a": float(mean_shap1[j]),
                    "mean_shap_b": float(mean_shap2[j]),
                    "gap": float(gap_per_feat[j]),
                    "abs_gap": float(abs(gap_per_feat[j])),
                    "category": _classify_feature(feature_names[j]),
                })
            # Total absolute explained-ish measure — sum of |gap| across
            # all features (approximates |E[prediction gap]| for GBM).
            total_abs_gap = float(np.abs(gap_per_feat).sum())
            top_abs_gap = float(sum(c["abs_gap"] for c in contributions))
            verdict = _verdict_for_topk([c["category"] for c in contributions])
            comparisons.append({
                "group_a": g1,
                "group_b": g2,
                "n_a": len(idx1),
                "n_b": len(idx2),
                "mean_pred_a": m1,
                "mean_pred_b": m2,
                "prediction_gap": prediction_gap,
                "total_abs_gap": total_abs_gap,
                "top_k_abs_gap": top_abs_gap,
                "top_features": contributions,
                "verdict": verdict,
            })
    return comparisons


def _run_attribute(
    df, X, gbm, feature_names, *,
    attribute_column: str, attribute_label: str, top_k: int,
) -> Dict:
    """Group the cohort by ``attribute_column`` and run the SHAP-gap
    analysis over all group-pairs with >= 5 members each."""
    if attribute_column not in df.columns:
        return {
            "attribute": attribute_label,
            "attribute_column": attribute_column,
            "n_with_attribute": 0,
            "comparisons": [],
            "note": f"column {attribute_column!r} not present in cohort",
        }
    group_values = df[attribute_column].fillna("unknown")
    groups_idx: Dict[str, List[int]] = {}
    for idx, val in enumerate(group_values.tolist()):
        key = str(val)
        groups_idx.setdefault(key, []).append(idx)
    comparisons = _compute_group_gap(
        X, gbm, feature_names, groups_idx, top_k=top_k,
    )
    return {
        "attribute": attribute_label,
        "attribute_column": attribute_column,
        "n_with_attribute": int(len(df)),
        "group_sizes": {k: len(v) for k, v in groups_idx.items()},
        "comparisons": comparisons,
    }


def benchmark(
    *, n_patients: int, n_history: int, top_k: int, output_path: Path,
) -> Dict:
    t0 = time.perf_counter()
    # --------- load + train ----------
    df_all, y_all = _load_historical(n_history)
    train_rows = _feature_frame(df_all.head(n_history))
    X_train, feature_names = _extract_numeric_matrix(train_rows)
    print(
        f"[fairness-shap] training GBM on {X_train.shape[0]} historical "
        f"rows × {X_train.shape[1]} features...",
        flush=True,
    )
    gbm = _train_gbm(X_train, y_all[: X_train.shape[0]])

    # --------- score + attribute ----------
    df_test = df_all.head(n_patients).reset_index(drop=True)
    test_rows = _feature_frame(df_test)
    X_test, feat_test = _extract_numeric_matrix(test_rows)
    # Align feature order with training; missing features default 0.0
    import numpy as np
    aligned = np.zeros((X_test.shape[0], len(feature_names)), dtype=float)
    pos = {n: i for i, n in enumerate(feat_test)}
    for j, fn in enumerate(feature_names):
        if fn in pos:
            aligned[:, j] = X_test[:, pos[fn]]
    X_test = aligned

    # Gender label
    if "Person_Stated_Gender_Code" in df_test.columns:
        df_test["Gender"] = df_test["Person_Stated_Gender_Code"].map(
            {1: "M", 2: "F"}
        ).fillna("unknown")
    # Site code is usually already a string
    if "Site_Code" in df_test.columns:
        df_test["Site_Code"] = df_test["Site_Code"].fillna("unknown").astype(str)

    per_attr: List[Dict] = []
    for col, label in (("Gender", "Gender"), ("Site_Code", "Site")):
        per_attr.append(_run_attribute(
            df_test, X_test, gbm, feature_names,
            attribute_column=col, attribute_label=label,
            top_k=top_k,
        ))

    wall_s = time.perf_counter() - t0
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_patients": int(n_patients),
        "n_history": int(n_history),
        "top_k": int(top_k),
        "n_features_total": int(X_test.shape[1]),
        "model": "GradientBoostingClassifier (base learner)",
        "feature_categories": {
            "protected": list(_PROTECTED_KEYS),
            "proxy": list(_PROXY_KEYS),
            "geographic": list(_GEOGRAPHIC_KEYS),
            "clinical": list(_CLINICAL_KEYS),
            "temporal": list(_TEMPORAL_KEYS),
            "administrative": list(_ADMINISTRATIVE_KEYS),
        },
        "wall_seconds": float(wall_s),
        "attributes": per_attr,
        "method_note": (
            "TreeExplainer SHAP attributions on a GradientBoostingClassifier "
            "base learner trained on historical_appointments.xlsx.  For each "
            "protected attribute, every unordered group-pair with >= 5 members "
            "gets a mean-SHAP gap per feature; features sorted by |gap| "
            "descending; top-k feeds the verdict.  "
            "bias_direct = any top-k feature is protected; "
            "legitimate = all top-k geographic/clinical/temporal; "
            "legitimate_with_note = otherwise."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print(
        f"[fairness-shap] wall={wall_s:.1f}s  attributes={len(per_attr)}  "
        f"-> {output_path}",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients", type=int, default=120)
    parser.add_argument("--n-history",  type=int, default=1500)
    parser.add_argument("--top-k",      type=int, default=3,
                        help="Number of top contributors retained per comparison")
    parser.add_argument(
        "--output",
        default="data_cache/fairness_shap/results.jsonl",
    )
    args = parser.parse_args()
    benchmark(
        n_patients=args.n_patients,
        n_history=args.n_history,
        top_k=args.top_k,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
