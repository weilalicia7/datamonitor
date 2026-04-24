"""
Conformal duration benchmark — measures MAE, interval width + width heterogeneity
=================================================================================

Regression for dissertation §4.5 external-review note (mistake 15):
the earlier prose claimed "MAE = 24.3 min, mean interval width 87 min"
as static literals with no cohort provenance or heteroscedasticity
decomposition.  The reviewer suggested adding a depth discussion of
*why* conformal intervals are ~3.6× the MAE — heteroscedasticity +
long-tailed duration distribution.

This script runs a real head-to-head on the
``historical_appointments.xlsx`` cohort:

1. Extract (features, actual_duration) pairs for rows with both
   ``Planned_Duration`` (or similar) and ``Actual_Duration`` present.
2. Random 80/20 train/test split (seeded).
3. Fit the production ``ConformalDurationPredictor`` (CQR, alpha=0.10)
   on the train split.
4. Predict on the test split → per-row point estimate + 90 % interval.
5. Compute:
     - MAE, RMSE on the point estimates (sanity vs Table 4.2-ish claim)
     - mean / median interval width
     - heteroscedasticity proxy: CV(widths) = std/mean of widths
     - long-tail proxy: fraction of widths > 2 × median width
     - width-vs-predicted-duration Spearman correlation
     - observed empirical coverage on the test split
6. Append one row to
   ``data_cache/conformal_benchmark/duration_results.jsonl``.

The R analysis (``dissertation_analysis.R §22`` — new) reads the
JSONL and emits seven macros that the §4.5 "Treatment Duration
Prediction" subsection consumes — all from the same benchmark row,
so the dissertation cells / prose / depth paragraph cannot drift.

Runs manually; no UI panel; no impact on the prediction pipeline.

CLI
---
    python -m ml.benchmark_conformal_duration \
        --test-frac 0.20 --seed 42 --alpha 0.10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_cohort():
    """Build (patient_dicts, actual_duration_array) from the historical data."""
    import numpy as np
    import pandas as pd

    hist_path = (
        _REPO_ROOT / "datasets" / "sample_data" / "historical_appointments.xlsx"
    )
    df = pd.read_excel(hist_path)
    # Keep only rows with a valid actual duration (>0) and planned duration
    for col in ("Actual_Duration", "Planned_Duration"):
        if col not in df.columns:
            raise SystemExit(
                f"historical_appointments.xlsx missing required column {col!r}"
            )
    df = df.dropna(subset=["Actual_Duration", "Planned_Duration"])
    df = df[df["Actual_Duration"] > 0].reset_index(drop=True)
    if len(df) < 40:
        raise SystemExit(
            f"Need >=40 appointments with actual duration; only {len(df)} present"
        )

    # Build patient dicts that satisfy ConformalDurationPredictor._extract_features
    patients = []
    for _, row in df.iterrows():
        patients.append({
            "patient_id": str(row.get("Patient_ID", f"P{len(patients):04d}")),
            "expected_duration": float(row.get("Planned_Duration", 120)),
            "cycle_number": int(row.get("Cycle_Number", 1) or 1),
            "age": int(row.get("Age", 55) or 55),
            "complexity_factor": 0.5,  # Not present in historical_appointments.xlsx
            "weight": float(row.get("Weight", 70) or 70),
        })
    return patients, np.asarray(df["Actual_Duration"].values, dtype=float)


def _spearman(a, b):
    """Spearman rank correlation without scipy (avoid extra dep)."""
    import numpy as np

    def _rank(x):
        order = np.argsort(x, kind="stable")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float) + 1
        return ranks

    a_rank, b_rank = _rank(np.asarray(a)), _rank(np.asarray(b))
    ma, mb = a_rank.mean(), b_rank.mean()
    num = float(((a_rank - ma) * (b_rank - mb)).sum())
    den = math.sqrt(
        float(((a_rank - ma) ** 2).sum()) * float(((b_rank - mb) ** 2).sum())
    )
    return float(num / den) if den > 0 else 0.0


def benchmark(
    *, test_frac: float, seed: int, alpha: float, output_path: Path,
) -> Dict:
    import numpy as np

    patients, y = _load_cohort()
    n = len(patients)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(10, int(n * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    p_train = [patients[i] for i in train_idx]
    y_train = y[train_idx]
    p_test = [patients[i] for i in test_idx]
    y_test = y[test_idx]

    print(
        f"\n=== Conformal duration benchmark ===\n"
        f"  n_total={n}  n_train={len(p_train)}  n_test={len(p_test)}\n"
        f"  alpha={alpha} (target coverage {(1 - alpha) * 100:.0f}%)",
        flush=True,
    )

    from ml.conformal_prediction import ConformalDurationPredictor

    predictor = ConformalDurationPredictor(alpha=alpha)
    predictor.fit(p_train, y_train)

    preds = [predictor.predict(p) for p in p_test]
    point = np.asarray([pr.point_estimate for pr in preds], dtype=float)
    widths = np.asarray([pr.interval_width for pr in preds], dtype=float)
    lowers = np.asarray([pr.lower_bound for pr in preds], dtype=float)
    uppers = np.asarray([pr.upper_bound for pr in preds], dtype=float)

    # Point accuracy
    errs = np.abs(point - y_test)
    mae = float(errs.mean())
    rmse = float(math.sqrt(((point - y_test) ** 2).mean()))

    # Interval statistics
    width_mean = float(widths.mean())
    width_median = float(np.median(widths))
    width_std = float(widths.std())
    width_p90 = float(np.quantile(widths, 0.90))
    # Heteroscedasticity proxies
    cv_width = float(width_std / max(width_mean, 1e-9))
    tail_frac_2x = float((widths > 2 * width_median).mean())
    # Correlation of width with predicted point — if hetero is driven
    # by predicted mean, this is non-zero
    width_vs_point_spearman = _spearman(widths, point)

    # Empirical coverage on the held-out split
    covered = np.logical_and(y_test >= lowers, y_test <= uppers)
    empirical_coverage = float(covered.mean())

    # Ratio the depth paragraph cites
    width_over_mae = float(width_mean / max(mae, 1e-9))

    print(
        f"  MAE={mae:.1f}  RMSE={rmse:.1f}\n"
        f"  mean width={width_mean:.1f}  median={width_median:.1f}  "
        f"p90={width_p90:.1f}  std={width_std:.1f}\n"
        f"  CV(widths)={cv_width:.3f}  tail_frac_2x_median={tail_frac_2x:.3f}\n"
        f"  Spearman(width, predicted_duration)={width_vs_point_spearman:+.3f}\n"
        f"  empirical coverage={empirical_coverage:.3f}  "
        f"(target {(1 - alpha):.2f})\n"
        f"  width/MAE={width_over_mae:.2f}x",
        flush=True,
    )

    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "n_total": int(n),
        "n_train": int(len(p_train)),
        "n_test": int(len(p_test)),
        "seed": int(seed),
        "alpha": float(alpha),
        "target_coverage": float(1 - alpha),
        "mae": mae,
        "rmse": rmse,
        "width_mean": width_mean,
        "width_median": width_median,
        "width_p90": width_p90,
        "width_std": width_std,
        "cv_width": cv_width,
        "tail_frac_2x_median": tail_frac_2x,
        "width_vs_point_spearman": width_vs_point_spearman,
        "empirical_coverage": empirical_coverage,
        "width_over_mae": width_over_mae,
        "comparison_note": (
            "MAE is the mean of |predicted - actual| at the point estimate; "
            "the conformal interval width is 2 * conformal-quantile at the "
            "alpha-level target coverage, so it bounds the FULL residual "
            "distribution at (1 - alpha), not just the first moment.  "
            "CV(widths) and the 2x-median tail fraction quantify "
            "heteroscedasticity: larger values mean some patients receive "
            "much wider intervals than others, which is the depth the "
            "dissertation §4.5 discussion now cites."
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
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Miscoverage level (0.10 -> 90% intervals)")
    parser.add_argument(
        "--output",
        default="data_cache/conformal_benchmark/duration_results.jsonl",
    )
    args = parser.parse_args()
    benchmark(
        test_frac=args.test_frac,
        seed=args.seed,
        alpha=args.alpha,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
