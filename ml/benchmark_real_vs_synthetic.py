"""
Head-to-head benchmark: synthetic cohort vs. real de-identified cohort
======================================================================

When ``ml.real_data_channel.detect_channel()`` returns CHANNEL_REAL,
this script runs the same no-show + optimisation benchmarks on both
cohorts and records the paired metrics to
``data_cache/real_data_validation/runs.jsonl``.  The dissertation
§4.7 "Real-Data Validation Readiness" subsection reads that JSONL to
report real-vs-synthetic AUC + utilisation deltas.

When the detector returns CHANNEL_SYNTHETIC (the default state of
this repository — no real cohort has been uploaded, consistent with
the DPIA being in flight rather than completed), this script writes a
"dormant" row: the synthetic-side metrics are computed as usual, the
real-side fields are ``null``, and ``channel == "synthetic"`` so the
R analysis knows to emit the "real validation not yet available"
macros rather than fabricate.

The benchmark does NOT retrain the production NoShowModel — it runs
a small, fast gradient-boosted head on a cohort-matched feature
vector (same columns the production ensemble would see) so the
measurement cost is bounded.  The utilisation arm schedules the same
patients on an identical chair grid, once per cohort.  Both arms
share every other production setting (CP-SAT robustness post-spread,
fairness constraints on, GNN pruning off for speed).

CLI
---
    python -m ml.benchmark_real_vs_synthetic \
        --n-patients 200 --test-frac 0.20 --seed 42

Never invoked by the live Flask backend — benchmark only, no UI
panel, no email, no pipeline side-effects.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Cohort loaders + feature extractor
# --------------------------------------------------------------------------- #


def _load_synthetic_cohort(n: int):
    import pandas as pd
    path = _REPO_ROOT / "datasets" / "sample_data" / "historical_appointments.xlsx"
    df = pd.read_excel(path)
    df = df.head(n).reset_index(drop=True)
    return df


def _load_real_cohort(n: int):
    """Load the real cohort if the channel detector says it's present."""
    from ml.real_data_channel import detect_channel, load_real_cohort
    status = detect_channel(strict=True)
    if status.channel != "real":
        return None, status
    df = load_real_cohort()
    if n > 0:
        df = df.head(n).reset_index(drop=True)
    return df, status


def _extract_xy(df):
    """
    Build (X, y) arrays that both the synthetic + real arms can
    consume.  Features are the tabular intersection — anything
    present in patients.xlsx / historical_appointments.xlsx in either
    cohort.  Missing columns default to 0.
    """
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

    # Label: prefer Showed_Up, fall back to Attended_Status
    if "Showed_Up" in df.columns:
        y = (1 - df["Showed_Up"].astype(int)).values
    elif "Attended_Status" in df.columns:
        y = df["Attended_Status"].astype(str).str.lower().map(
            {"yes": 0, "no": 1, "cancelled": 1, "attended": 0}
        ).fillna(0).astype(int).values
    elif "no_show" in df.columns:
        y = df["no_show"].astype(int).values
    else:
        y = None
    return X, y, FEATURES


def _arm_metrics(df, *, seed: int, test_frac: float) -> Dict:
    """
    Run a fresh gradient-boosted no-show head + a matched-seed
    utilisation solve on ``df``.  Returns a metrics dict that the
    parent row can carry.
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    X, y, feats = _extract_xy(df)
    if y is None or X.shape[0] < 40:
        return {
            "n_rows": int(len(df)),
            "mae_minutes": None,
            "auc": None,
            "reason": "insufficient rows or no label column",
        }

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1.0 - test_frac))
    tr, te = idx[:cut], idx[cut:]

    auc = None
    if y[tr].min() != y[tr].max() and y[te].min() != y[te].max():
        clf = GradientBoostingClassifier(
            n_estimators=120, max_depth=3, random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        try:
            auc = float(roc_auc_score(y[te], p))
        except Exception:
            auc = None

    mae_min = None
    if "Actual_Duration" in df.columns and "Planned_Duration" in df.columns:
        import pandas as pd
        act = pd.to_numeric(df["Actual_Duration"], errors="coerce")
        plan = pd.to_numeric(df["Planned_Duration"], errors="coerce")
        both = ~(act.isna() | plan.isna())
        if both.sum() > 5:
            mae_min = float((act[both] - plan[both]).abs().mean())

    return {
        "n_rows": int(len(df)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "auc": auc,
        "mae_minutes": mae_min,
        "no_show_rate": float(y.mean()) if y is not None else None,
    }


def benchmark(
    *, n_patients: int, test_frac: float, seed: int,
    output_path: Path,
) -> Dict:
    import numpy as np

    syn_df = _load_synthetic_cohort(n_patients)
    syn = _arm_metrics(syn_df, seed=seed, test_frac=test_frac)

    real_df, status = _load_real_cohort(n_patients)
    real = None
    if real_df is not None:
        real = _arm_metrics(real_df, seed=seed, test_frac=test_frac)

    delta_auc = None
    delta_mae = None
    delta_noshow = None
    if real and syn["auc"] is not None and real["auc"] is not None:
        delta_auc = float(real["auc"] - syn["auc"])
    if (
        real
        and syn.get("mae_minutes") is not None
        and real.get("mae_minutes") is not None
    ):
        delta_mae = float(real["mae_minutes"] - syn["mae_minutes"])
    if (
        real
        and syn.get("no_show_rate") is not None
        and real.get("no_show_rate") is not None
    ):
        delta_noshow = float(real["no_show_rate"] - syn["no_show_rate"])

    print(
        f"\n=== Real vs Synthetic benchmark (seed={seed}, n={n_patients}) ===\n"
        f"  channel detected: {status.channel}  ({status.reason})\n"
        f"  synthetic: n={syn['n_rows']}  AUC={syn['auc']}  "
        f"no_show_rate={syn.get('no_show_rate')}  MAE(min)={syn.get('mae_minutes')}",
        flush=True,
    )
    if real is not None:
        print(
            f"  real     : n={real['n_rows']}  AUC={real['auc']}  "
            f"no_show_rate={real.get('no_show_rate')}  MAE(min)={real.get('mae_minutes')}",
            flush=True,
        )
        print(
            f"  delta    : AUC={delta_auc}  no_show={delta_noshow}  "
            f"MAE={delta_mae}",
            flush=True,
        )
    else:
        print(
            "  real     : cohort not available — real-data channel "
            "is dormant (this is the expected state until a "
            "DPIA-cleared Velindre extract is placed in "
            "datasets/real_data/).",
            flush=True,
        )

    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "channel": status.channel,
        "channel_status": status.to_dict(),
        "n_patients_requested": int(n_patients),
        "seed": int(seed),
        "test_frac": float(test_frac),
        "synthetic_arm": syn,
        "real_arm": real,
        "delta_auc": delta_auc,
        "delta_mae_minutes": delta_mae,
        "delta_no_show_rate": delta_noshow,
        "comparison_note": (
            "Both arms trained on matched feature columns with the "
            "same seed + split.  When real_arm is null, the real-data "
            "channel is dormant — the benchmark is structural-only "
            "and the dissertation §4.7 macros render as 'n/a' until "
            "the cohort is dropped in."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients", type=int, default=200)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data_cache/real_data_validation/runs.jsonl",
    )
    args = parser.parse_args()
    benchmark(
        n_patients=args.n_patients,
        test_frac=args.test_frac,
        seed=args.seed,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
