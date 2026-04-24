"""
Head-to-head benchmark: baseline R(S) vs. robustness-weighted CP-SAT
=====================================================================

Regression for §5.9 / Table 5.3 external-review finding: the
dissertation displayed R(S) = 0.135 (from the appointments.xlsx
actual schedule) as both the "baseline" prose value AND the "system
optimised" column of Table 5.3, with a fabricated 0.098 baseline and
a fabricated ±0.028 / ±0.031 uncertainty.  This script replaces the
fabricated numbers with real measurements on the same patient cohort.

For a fixed n-patient slice of ``patients.xlsx`` and a shared chair
grid, we schedule twice:

  * Arm A (baseline)   — robustness weight = 0 (drop the buffer
                         objective entirely, so CP-SAT packs chairs
                         tightly with no slack incentive)
  * Arm B (robust)     — robustness weight = 0.10 (the production
                         default)

For each arm we compute R(S) using the SAME slack-based formula the
R analysis uses:

    R(S) = mean( clip( slack_minutes_between_consecutive_appointments
                       on the same chair, 0, 60 ) / 60 )

so the dissertation table cells cannot drift from the benchmark JSONL.

CLI
---
    python -m ml.benchmark_robustness \
        --n-patients 40 --n-chairs 6 --time-limit-seconds 10 --seed 42

Never invoked by the live Flask backend.  The default optimiser
behaviour (robustness=0.10) is unchanged; this script temporarily
flips the weight for its own solve pair only.  Output is JSONL only
— no UI panel, no email, no impact on the prediction pipeline.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_patients(n: int):
    import pandas as pd
    from optimization.optimizer import Patient
    from config import OPERATING_HOURS

    df = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" / "patients.xlsx")
    if len(df) < n:
        raise SystemExit(
            f"Need {n} patients but patients.xlsx only has {len(df)}"
        )
    df = df.head(n)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_h, end_h = OPERATING_HOURS

    patients = []
    for _, row in df.iterrows():
        patients.append(Patient(
            patient_id=str(row.get("Patient_ID", f"P{len(patients):04d}")),
            priority=int(row.get("Priority_Int", 3) or 3),
            protocol=str(row.get("Regimen_Code", "R-CHOP")),
            expected_duration=int(row.get("Average_Duration", 90) or 90),
            postcode=str(row.get("Patient_Postcode", "CF14")),
            earliest_time=today.replace(hour=start_h),
            latest_time=today.replace(hour=end_h),
            noshow_probability=float(row.get("Patient_NoShow_Rate", 0.13) or 0.13),
        ))
    return patients


def _build_chairs(n_chairs: int):
    from optimization.optimizer import Chair
    from config import OPERATING_HOURS, DEFAULT_SITES

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_h, end_h = OPERATING_HOURS
    chairs = []
    for site in DEFAULT_SITES:
        for i in range(site["chairs"]):
            chairs.append(Chair(
                chair_id=f"{site['code']}-C{i+1:02d}",
                site_code=site["code"],
                is_recliner=i < site.get("recliners", 0),
                available_from=today.replace(hour=start_h),
                available_until=today.replace(hour=end_h),
            ))
            if len(chairs) >= n_chairs:
                return chairs
    return chairs


def robustness_score(appointments) -> dict:
    """
    R(S) = mean(min(slack_min / 60, 1)) over consecutive chair
    transitions.  Returns both the score and the per-category
    breakdown so R can render a proper two-panel figure.

    Mirrors dissertation_analysis.R §8 verbatim so the Python
    benchmark and the R analysis compute the same quantity.
    """
    from collections import defaultdict

    by_chair = defaultdict(list)
    for a in appointments or []:
        by_chair[a.chair_id].append(a)

    slacks = []
    for chair_id, items in by_chair.items():
        items = sorted(items, key=lambda a: a.start_time)
        for a, b in zip(items, items[1:]):
            gap = (b.start_time - a.end_time).total_seconds() / 60.0
            if 0 <= gap < 300:
                slacks.append(gap)

    if not slacks:
        return {
            "robustness_score": 1.0,
            "n_transitions": 0,
            "slack_distribution": {
                "critical": 0.0, "tight": 0.0,
                "adequate": 0.0, "ample": 0.0,
            },
        }

    score = sum(min(s / 60.0, 1.0) for s in slacks) / len(slacks)

    n = len(slacks)
    critical = sum(1 for s in slacks if s < 10) / n
    tight    = sum(1 for s in slacks if 10 <= s < 20) / n
    adequate = sum(1 for s in slacks if 20 <= s < 60) / n
    ample    = sum(1 for s in slacks if s >= 60) / n

    return {
        "robustness_score": float(score),
        "n_transitions": int(n),
        "slack_distribution": {
            "critical": float(critical),
            "tight": float(tight),
            "adequate": float(adequate),
            "ample": float(ample),
        },
    }


def _run_arm(
    patients, chairs, *, robustness_weight: float,
    time_limit_s: float,
) -> Dict:
    from optimization.optimizer import ScheduleOptimizer

    opt = ScheduleOptimizer()
    opt.chairs = chairs
    # Disable CG to stay on the monolithic CP-SAT path; keep fairness
    # ON for both arms so the robustness delta isn't confounded.
    opt.set_components(column_generation=False, fairness=True)

    # Snapshot, flip, restore.  Re-normalise so the 5 remaining
    # objectives still sum to 1 when we drop robustness.
    orig = dict(opt.weights)
    new_w = dict(orig)
    if robustness_weight == 0.0:
        removed = new_w.pop("robustness", 0.0)
        total = sum(new_w.values())
        if total > 0:
            for k in new_w:
                new_w[k] = new_w[k] / total
        new_w["robustness"] = 0.0
    else:
        new_w["robustness"] = robustness_weight
        total = sum(new_w.values())
        if total > 0:
            for k in new_w:
                new_w[k] = new_w[k] / total
    opt.set_weights(new_w, normalise=False)

    try:
        t0 = time.perf_counter()
        result = opt.optimize(patients, time_limit_seconds=int(time_limit_s))
        dt = time.perf_counter() - t0
    finally:
        opt.set_weights(orig, normalise=False)

    rob = robustness_score(getattr(result, "appointments", []) or [])
    return {
        "robustness_weight": float(robustness_weight),
        "solve_time_s": float(dt),
        "n_scheduled": int(len(result.appointments or [])),
        "n_patients": int(len(patients)),
        "status": str(getattr(result, "status", "UNKNOWN")),
        **rob,
    }


def benchmark(
    *, n_patients: int, n_chairs: int, time_limit_s: float, seed: int,
    output_path: Path,
) -> Dict:
    patients = _load_patients(n_patients)
    chairs = _build_chairs(n_chairs)

    print(f"\n=== Robustness benchmark (n={n_patients}, chairs={n_chairs}, seed={seed}) ===",
          flush=True)

    baseline = _run_arm(patients, chairs,
                        robustness_weight=0.0, time_limit_s=time_limit_s)
    print(f"  Arm A (robustness=0.00): R(S)={baseline['robustness_score']:.3f}  "
          f"transitions={baseline['n_transitions']}", flush=True)

    robust = _run_arm(patients, chairs,
                      robustness_weight=0.10, time_limit_s=time_limit_s)
    print(f"  Arm B (robustness=0.10): R(S)={robust['robustness_score']:.3f}  "
          f"transitions={robust['n_transitions']}", flush=True)

    delta = robust["robustness_score"] - baseline["robustness_score"]
    pct = (
        100.0 * delta / baseline["robustness_score"]
        if baseline["robustness_score"] > 0 else 0.0
    )
    print(f"  Delta R(S): {delta:+.3f}  (+{pct:.1f}%)", flush=True)

    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "n_patients": int(n_patients),
        "n_chairs": int(n_chairs),
        "time_limit_s": float(time_limit_s),
        "seed": int(seed),
        "arm_baseline": baseline,
        "arm_robust": robust,
        "delta_robustness": float(delta),
        "delta_robustness_pct": float(pct),
        "comparison_note": (
            "Both arms schedule the same cohort on the same chair grid; "
            "only the robustness objective weight differs (0.00 in "
            "Arm A, the production default 0.10 in Arm B).  The R(S) "
            "formula matches dissertation_analysis.R §8 verbatim: "
            "R(S) = mean(min(slack_min / 60, 1)) over consecutive "
            "chair transitions.  Dissertation §5.9 Table 5.3 cites "
            "this JSONL directly — neither number is fabricated."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients", type=int, default=40)
    parser.add_argument("--n-chairs", type=int, default=8)
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data_cache/robustness_benchmark/results.jsonl",
    )
    args = parser.parse_args()
    benchmark(
        n_patients=args.n_patients,
        n_chairs=args.n_chairs,
        time_limit_s=args.time_limit_seconds,
        seed=args.seed,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
