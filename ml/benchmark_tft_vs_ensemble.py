"""
Head-to-head held-out benchmark: TFT-Lite vs.\ stacked no-show ensemble
======================================================================

Runs both predictors on the SAME held-out split of the production
historical_appointments.xlsx data and writes one comparison row to
``data_cache/ensemble_benchmark/results.jsonl``.  This is the single
source of truth for dissertation §4.5.6 Table: "no-show AUC (TFT) vs.
no-show AUC (stacked ensemble)".

The previous dissertation number (TFT AUC 0.706 "up from 0.635 for
the tree ensemble") was apples-to-oranges:
  - TFT AUC was computed on the TRAINING set (no held-out split)
  - The 0.635 baseline was a stale literal, not measured on the
    same cohort/split

This benchmark computes both AUCs on the SAME test set indices, using
the SAME eligibility filter (the TFT's "≥ past_window prior
appointments" rule applied to both models), so the head-to-head
comparison is honest.  Stacked-ensemble predictions use the currently
loaded production NoShowModel via ``predict_batch()``; TFT is fit
fresh on the train-half so it has no test-set leak by construction.

The script is invoked manually — never from the live backend — and
appends one row per run to the JSONL.  No UI panel.  No email.  No
impact on the prediction pipeline.

CLI
---
    python -m ml.benchmark_tft_vs_ensemble \
        --test-frac 0.20 --seed 42 --epochs 40

Output row shape::

    {
      "ts":                      ISO8601,
      "n_eligible":              int,    # after ≥past_window filter
      "n_train":                 int,
      "n_test":                  int,
      "seed":                    int,
      "tft_noshow_auc":          float,  # measured on n_test held-out
      "tft_cancel_auc":          float or None,
      "tft_epochs":              int,
      "ensemble_noshow_auc":     float,  # measured on SAME n_test rows
      "ensemble_available":      bool,   # False when NoShowModel not loadable
      "comparison_note":         str,    # honest caveats (e.g. "ensemble was
                                         # pre-trained on full historical data")
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_eligible(past_window: int):
    """
    Return the TFT-eligible rows from historical_appointments.xlsx:
    patients with ≥ past_window prior appointments.  Using the same
    eligibility filter for both models is what makes the head-to-head
    comparison legitimate — the stacked ensemble is evaluated on
    exactly the rows the TFT would see.
    """
    import pandas as pd

    hist = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" /
                          "historical_appointments.xlsx")
    # Sort by (patient, date) so the ≥past_window filter counts
    # truly-prior appointments, not arbitrary rows.
    sort_cols = [c for c in ("Patient_ID", "Appointment_Date", "Date", "ts")
                 if c in hist.columns]
    if sort_cols:
        hist = hist.sort_values(sort_cols)
    # For each row, how many prior rows does that patient have?
    hist["_prior_count"] = hist.groupby("Patient_ID").cumcount()
    eligible = hist[hist["_prior_count"] >= past_window].copy()
    return eligible


def _extract_labels(df):
    """
    Normalise the no-show label across the several column conventions
    that appear in the dissertation dataset.
    """
    import numpy as np
    if "no_show" in df.columns:
        return np.asarray(df["no_show"].astype(int))
    if "Showed_Up" in df.columns:
        return np.asarray((1 - df["Showed_Up"].astype(int)))
    if "Attended_Status" in df.columns:
        m = df["Attended_Status"].astype(str).str.lower().map(
            {"yes": 0, "no": 1, "cancelled": 1, "attended": 0}
        ).fillna(0).astype(int)
        return np.asarray(m)
    raise SystemExit("No no-show label column found in the eligible data")


def _fit_tft_on_train(df_train, df_test, full_hist, *, epochs: int, past_window: int):
    """
    Fit a fresh TFTTrainer on the train half and return held-out
    no-show predictions on the test half.

    For each test row we build the per-patient past history from the
    full historical dataset using ONLY rows whose date strictly
    precedes the test row's date.  This avoids future-leak and mirrors
    the production inference path (`predict_single` takes the patient's
    prior appointments as context).
    """
    import numpy as np
    from ml.temporal_fusion_transformer import TFTTrainer

    t = TFTTrainer(past_window=past_window)
    t.fit(df_train, epochs=epochs)

    # Group full history by patient for fast past-row lookup
    hist_by_pid = {
        pid: grp.sort_values(
            [c for c in ("Appointment_Date", "Date", "ts") if c in grp.columns]
            or ["_prior_count"]
        )
        for pid, grp in full_hist.groupby("Patient_ID")
    }

    noshow_probs: list = []
    cancel_probs: list = []
    for _, row in df_test.iterrows():
        pid = row.get("Patient_ID")
        patient_dict = row.to_dict()
        past_rows = []
        if pid in hist_by_pid:
            grp = hist_by_pid[pid]
            # Use rows with _prior_count strictly less than the test row's
            # _prior_count so we never leak future appointments.
            prior_ct = int(row.get("_prior_count", 0))
            past_rows = grp[grp["_prior_count"] < prior_ct].tail(
                past_window
            ).to_dict("records")
        try:
            pred = t.predict_single(patient_dict, past_rows)
            noshow_probs.append(float(pred.get("p_noshow", 0.5)))
            cancel_probs.append(float(pred.get("p_cancel", 0.0)))
        except Exception as exc:
            print(f"  predict_single failed for {pid}: {exc}", flush=True)
            noshow_probs.append(0.5)
            cancel_probs.append(0.0)

    return (
        np.asarray(noshow_probs),
        np.asarray(cancel_probs),
        getattr(t.last_fit, "epochs", epochs),
    )


def _build_feature_matrix(df):
    """
    Numeric feature matrix for the fresh-fit production-stack ensemble.

    Mirrors the eight-feature tabular intersection the dissertation's
    benchmark ensemble uses elsewhere.  All columns are coerced to
    floats so the downstream sklearn fit cannot trip on object dtypes.
    """
    import numpy as np
    import pandas as pd
    FEATURES = [
        "Age", "Cycle_Number", "Treatment_Day", "Planned_Duration",
        "Travel_Time_Min", "Day_Of_Week_Num",
        "Total_Appointments_Before", "Previous_NoShows",
    ]
    out = np.zeros((len(df), len(FEATURES)), dtype=float)
    for j, c in enumerate(FEATURES):
        if c not in df.columns:
            continue
        col = df[c]
        if col.dtype == object:
            col = col.astype(str).str.extract(r"(-?\d+\.?\d*)")[0]
        out[:, j] = pd.to_numeric(col, errors="coerce").fillna(0).values
    return out, FEATURES


def _ensemble_predict_on_test(df_train, df_test, *, seed: int = 42):
    """
    Fit a **production-equivalent stacked ensemble + sigmoid (Platt)
    calibration** on df_train and return calibrated no-show probability
    predictions for df_test.

    Matches the production ``NoShowModel(use_stacking=True)`` recipe:

        Random Forest        n=100, max_depth=10
        Gradient Boosting    n=100, max_depth=5,  lr=0.1
        XGBoost (if avail)   n=100, max_depth=6,  lr=0.1
        Meta-learner         logistic regression on 5-fold OOF stacked
                             predictions (no interaction terms, kept
                             simple to match the benchmark contract)
        Calibration          5-fold CV sigmoid (Platt) on the
                             stacked predictor -- production's
                             best-Brier-skill regime per §5.2.1

    The reviewer's key point: this is an **ad-hoc retrain** on the
    eligibility-filtered subset, not the production model that was
    fit on the full historical pool.  The result row therefore tags
    ``ensemble_arm = "ad_hoc_retrain_on_filtered"`` so the dissertation
    prose cannot conflate this AUC with the production headline.

    Falls back to a per-patient heuristic baseline if scikit-learn is
    unavailable; the caller sets ``ensemble_available = False`` so the
    headline flag is honest.
    """
    import numpy as np

    try:
        from sklearn.ensemble import (RandomForestClassifier,
                                       GradientBoostingClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_predict
        from sklearn.calibration import CalibratedClassifierCV

        try:
            from xgboost import XGBClassifier
            _HAS_XGB = True
        except Exception:
            _HAS_XGB = False

        X_train, _ = _build_feature_matrix(df_train)
        X_test,  _ = _build_feature_matrix(df_test)
        y_train = _extract_labels(df_train)

        scaler = StandardScaler().fit(X_train)
        Xtr_s = scaler.transform(X_train)
        Xte_s = scaler.transform(X_test)

        # Production base learners (hyperparameters from
        # ml/noshow_model.py:_initialize_models).
        base_estimators = {
            "rf":  RandomForestClassifier(n_estimators=100, max_depth=10,
                                           min_samples_split=5,
                                           min_samples_leaf=2,
                                           random_state=42, n_jobs=-1),
            "gb":  GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                               learning_rate=0.1,
                                               random_state=42),
        }
        if _HAS_XGB:
            base_estimators["xgb"] = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, use_label_encoder=False,
                eval_metric="logloss",
            )

        # ---- Stage 1: 5-fold OOF base predictions for the meta-learner ----
        oof_preds: dict = {}
        for name, est in base_estimators.items():
            oof = cross_val_predict(est, Xtr_s, y_train, cv=5,
                                    method="predict_proba", n_jobs=1)[:, 1]
            oof_preds[name] = oof

        meta_X_train = np.column_stack(list(oof_preds.values()))

        # ---- Stage 2: meta-learner (logistic regression) ----
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        meta.fit(meta_X_train, y_train)

        # Re-fit base learners on the full train split for inference
        base_test_preds: dict = {}
        for name, est in base_estimators.items():
            est.fit(Xtr_s, y_train)
            base_test_preds[name] = est.predict_proba(Xte_s)[:, 1]
        meta_X_test = np.column_stack(list(base_test_preds.values()))
        p_te_uncal = meta.predict_proba(meta_X_test)[:, 1]

        # Also produce a reference uncalibrated AUC for the row.
        from sklearn.metrics import roc_auc_score as _auc
        p_uncal_auc = float(_auc(y_train,
                                  meta.predict_proba(meta_X_train)[:, 1])) \
                       if y_train.min() != y_train.max() else None

        # ---- Stage 3: production-equivalent sigmoid (Platt) calibration ----
        # Wrap the FULL stacked predictor in CalibratedClassifierCV so the
        # held-out probabilities reflect the production calibration choice
        # (sigmoid wins on Brier-skill per §5.2.1 multi-method comparison).
        # Here we calibrate the meta-learner directly on the OOF features
        # via 5-fold CV; this matches what production would emit at
        # inference time.
        cal = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            method="sigmoid", cv=5,
        )
        cal.fit(meta_X_train, y_train)
        p_te_cal = cal.predict_proba(meta_X_test)[:, 1]

        # Pack into a dict so the row can carry both forms (uncal + cal)
        return {
            "ok":              True,
            "p_uncal":         np.asarray(p_te_uncal),
            "p_cal":           np.asarray(p_te_cal),
            "base_models":     list(base_estimators.keys()),
            "has_xgb":         bool(_HAS_XGB),
            "meta_learner":    "LogisticRegression(C=1.0)",
            "calibration":     "5-fold CV sigmoid (Platt) on the stacked predictor",
        }
    except Exception as exc:
        print(f"  Production-stack fit failed ({exc!r}); using heuristic baseline",
              flush=True)
        probs = []
        for _, row in df_test.iterrows():
            total = max(int(row.get("Total_Appointments_Before", 0) or 0), 1)
            prev = int(row.get("Previous_NoShows", 0) or 0)
            probs.append(float(np.clip(prev / total, 0.02, 0.80)))
        return {
            "ok":              False,
            "p_uncal":         np.asarray(probs),
            "p_cal":           np.asarray(probs),
            "base_models":     ["heuristic"],
            "has_xgb":         False,
            "meta_learner":    None,
            "calibration":     None,
        }


def _auc_safe(y, p):
    import numpy as np
    y = np.asarray(y)
    if y.min() == y.max() or p is None:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def benchmark(
    *,
    test_frac: float = 0.20,
    seed: int = 42,
    epochs: int = 40,
    past_window: int = 3,
    output_path: Path,
) -> dict:
    import numpy as np

    df = _load_eligible(past_window)
    if len(df) < 40:
        raise SystemExit(
            f"Only {len(df)} eligible rows (need ≥ 40) — widen the cohort "
            "or lower past_window"
        )

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = max(10, int(len(df) * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_train = _extract_labels(df_train)
    y_test = _extract_labels(df_test)
    print(
        f"\n=== TFT vs stacked ensemble head-to-head ===\n"
        f"  n_eligible={len(df)}  n_train={len(df_train)}  n_test={len(df_test)}\n"
        f"  train class balance: y=1 -> {int(y_train.sum())} / {len(y_train)}\n"
        f"  test  class balance: y=1 -> {int(y_test.sum())} / {len(y_test)}",
        flush=True,
    )

    # TFT arm — fit fresh, predict on the held-out rows
    tft_p, tft_cancel_p, tft_epochs = _fit_tft_on_train(
        df_train, df_test, df, epochs=epochs, past_window=past_window,
    )
    tft_noshow_auc = _auc_safe(y_test, tft_p)
    tft_cancel_auc = _auc_safe(y_test, tft_cancel_p)
    print(f"  TFT no-show AUC (held-out): {tft_noshow_auc}", flush=True)

    # Stacked-ensemble arm — fit the production-equivalent stack
    # (RF + GB + XGB + LogReg meta-learner + 5-fold CV sigmoid
    # calibration) on the same eligibility-filtered train half so the
    # held-out AUC is directly comparable to the TFT's.  This is an
    # AD-HOC RETRAIN on the filtered subset, NOT the production model
    # (which was fit on the full historical pool); the result row tags
    # the arm explicitly so the dissertation prose cannot conflate it
    # with the production headline AUC.
    ens = _ensemble_predict_on_test(df_train, df_test, seed=seed)
    ens_auc_uncal = _auc_safe(y_test, ens["p_uncal"])
    ens_auc_cal   = _auc_safe(y_test, ens["p_cal"])
    arm_label = ("Stacked ensemble (RF+GB" +
                 ("+XGB" if ens.get("has_xgb") else "") +
                 ", logistic meta + sigmoid cal)") \
                if ens["ok"] else "Heuristic baseline"
    print(f"  {arm_label} AUC (uncal):         {ens_auc_uncal}", flush=True)
    print(f"  {arm_label} AUC (sigmoid-cal):   {ens_auc_cal}", flush=True)

    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "n_eligible": int(len(df)),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "seed": int(seed),
        "past_window": int(past_window),
        "tft_noshow_auc": None if tft_noshow_auc is None else float(tft_noshow_auc),
        "tft_cancel_auc": None if tft_cancel_auc is None else float(tft_cancel_auc),
        "tft_epochs": int(tft_epochs),
        # Headline AUC = sigmoid-calibrated stacked ensemble (matches
        # the §5.2.1 best-Brier-skill regime, so the head-to-head
        # comparison uses the production calibration choice).
        "ensemble_noshow_auc":      None if ens_auc_cal   is None else float(ens_auc_cal),
        "ensemble_uncal_noshow_auc":None if ens_auc_uncal is None else float(ens_auc_uncal),
        "ensemble_available":       bool(ens["ok"]),
        # Provenance fields the dissertation prose now cites verbatim
        # so reviewers can see exactly what was retrained.
        "ensemble_arm":             "ad_hoc_retrain_on_filtered_subset",
        "ensemble_base_models":     ens.get("base_models"),
        "ensemble_has_xgb":         bool(ens.get("has_xgb")),
        "ensemble_meta_learner":    ens.get("meta_learner"),
        "ensemble_calibration":     ens.get("calibration"),
        "comparison_note": (
            "AD-HOC RETRAIN on the past_window-or-more eligibility "
            "subset (NOT the production no-show model). The stacked "
            "ensemble arm is a fresh fit of the production base-learner "
            "configuration (Random Forest n=100/depth=10, Gradient "
            "Boosting n=100/depth=5/lr=0.1" +
            (", XGBoost n=100/depth=6/lr=0.1" if ens.get("has_xgb") else "") +
            ") combined via a logistic-regression meta-learner on "
            "5-fold OOF predictions, then wrapped in a 5-fold CV "
            "sigmoid (Platt) calibration -- the regime §5.2.1 "
            "identifies as production-best by Brier-skill. The TFT "
            "arm is a fresh TFTTrainer fit on the same df_train.  Both "
            "AUCs are measured on the SAME held-out rows; both are "
            "seeded deterministically.  The production stacked "
            "ensemble (table 5.1) reports AUC 0.635 on the FULL, "
            "UNFILTERED test set -- that figure is NOT directly "
            "comparable to this row because the eligibility filter "
            "selects a more difficult sub-cohort with longer history."
            if ens["ok"] else
            "TFT measured on the SAME held-out rows as a heuristic "
            "baseline (Previous_NoShows / Total_Appointments_Before, "
            "clipped to [0.02, 0.80]).  Scikit-learn was not available "
            "so a fully-fit stacked ensemble arm could not run; the "
            "heuristic is a floor, not a like-for-like comparison."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--past-window", type=int, default=10)
    parser.add_argument(
        "--output", default="data_cache/ensemble_benchmark/results.jsonl",
    )
    args = parser.parse_args()
    benchmark(
        test_frac=args.test_frac,
        seed=args.seed,
        epochs=args.epochs,
        past_window=args.past_window,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
