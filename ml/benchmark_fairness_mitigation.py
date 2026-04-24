"""
Head-to-head benchmark: fairness constraints OFF vs. ON
=======================================================

Runs the same cohort through ``ScheduleOptimizer`` twice:

  * Arm A (baseline)   — `_fairness_constraints_enabled = False`
  * Arm B (mitigation) — `_fairness_constraints_enabled = True`

For each arm the script measures:

  * per-group scheduling rates (Age_Band, Gender, Site_Code)
  * pairwise Four-Fifths worst ratio per group column
  * overall utilisation (scheduled / total)

Then writes one comparison row to
``data_cache/fairness_benchmark/results.jsonl`` that the R analysis
pipeline reads as the single source of truth for dissertation §5.6.2.

This addresses the external-review finding that the fairness audit
was showing failures (gender 0.766; site 0.659) without a paired
mitigation evaluation — a world-class system should demonstrate
that the DRO-style fairness constraints already in the CP-SAT
objective actually move the disparity ratios.

CLI
---
    python -m ml.benchmark_fairness_mitigation \
        --n-patients 60 --time-limit-seconds 15 --seed 42

Never invoked by the live Flask backend — benchmark only.  No UI
panel; no email; the only side-effect is a JSONL append.  The
optimiser's default remains `_fairness_constraints_enabled = True`
so this script cannot affect the prediction pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Data + group helpers
# --------------------------------------------------------------------------- #


def _load_patients(n: int):
    """Load n patients from the real patients.xlsx with their protected
    attributes attached so the audit can group by Gender / Site_Code /
    Age_Band."""
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
    audit_rows = []
    for _, row in df.iterrows():
        pid = str(row.get("Patient_ID", f"P{len(patients):04d}"))
        p = Patient(
            patient_id=pid,
            priority=int(row.get("Priority_Int", 3) or 3),
            protocol=str(row.get("Regimen_Code", "R-CHOP")),
            expected_duration=int(row.get("Average_Duration", 90) or 90),
            postcode=str(row.get("Patient_Postcode", "CF14")),
            earliest_time=today.replace(hour=start_h),
            latest_time=today.replace(hour=end_h),
            noshow_probability=float(row.get("Patient_NoShow_Rate", 0.13) or 0.13),
        )
        # Attach audit attributes used by the FairnessAuditor
        age_v = row.get("Age")
        try:
            age_n = int(age_v) if age_v is not None else None
        except Exception:
            age_n = None
        if age_n is not None:
            age_band = (
                "0-39" if age_n < 40 else
                "40-64" if age_n < 65 else
                "65+"
            )
        else:
            age_band = "unknown"
        p.age_band = age_band
        gender = str(row.get("Gender", row.get("Person_Stated_Gender_Code",
                                               "unknown"))).strip() or "unknown"
        # Normalise numeric codes 1/2/0/9 to M/F/U
        if gender in {"1"}:
            gender = "M"
        elif gender in {"2"}:
            gender = "F"
        elif gender in {"0", "9", ""}:
            gender = "unknown"
        p.gender = gender
        patients.append(p)
        audit_rows.append({
            "Patient_ID": pid,
            "Age_Band": age_band,
            "Gender": gender,
            "Site_Code": str(row.get("Home_Site_Code", row.get("Site_Code", "WC"))),
        })
    return patients, audit_rows


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


# --------------------------------------------------------------------------- #
# Audit driver
# --------------------------------------------------------------------------- #


def _worst_ratio(auditor, audit_rows, scheduled_ids, group_column):
    """Run the fairness auditor over ``group_column`` and return the
    worst (smallest) Four-Fifths ratio across all group pairs.  A
    ratio of 1.0 means perfect parity; < 0.80 fails the rule."""
    report = auditor.audit_schedule(audit_rows, scheduled_ids, group_column=group_column)
    ratios = [m.ratio for m in report.metrics if m.ratio is not None]
    return min(ratios) if ratios else 1.0


def _run_arm(
    patients, audit_rows, chairs, *, fairness_enabled: bool,
    time_limit_s: float,
) -> Dict:
    """One scheduling pass with fairness constraints on/off."""
    from optimization.optimizer import ScheduleOptimizer
    from ml.fairness_audit import FairnessAuditor

    opt = ScheduleOptimizer()
    opt.chairs = chairs
    opt._fairness_constraints_enabled = fairness_enabled
    # Don't route to column generation — we want the monolithic CP-SAT
    # path which is where the fairness penalties live.
    opt._cg_enabled = False

    t0 = time.perf_counter()
    result = opt.optimize(patients, time_limit_seconds=int(time_limit_s))
    dt = time.perf_counter() - t0

    scheduled_ids = {a.patient_id for a in (result.appointments or [])}
    utilisation = len(scheduled_ids) / max(len(patients), 1)

    auditor = FairnessAuditor()
    ratios = {
        col: _worst_ratio(auditor, audit_rows, scheduled_ids, col)
        for col in ("Age_Band", "Gender", "Site_Code")
    }

    return {
        "fairness_enabled": bool(fairness_enabled),
        "solve_time_s": float(dt),
        "utilisation": float(utilisation),
        "n_scheduled": int(len(scheduled_ids)),
        "n_patients": int(len(patients)),
        "status": str(getattr(result, "status", "UNKNOWN")),
        "worst_four_fifths_ratio": {k: float(v) for k, v in ratios.items()},
    }


def benchmark(
    *, n_patients: int, n_chairs: int, time_limit_s: float, seed: int,
    output_path: Path,
) -> Dict:
    patients, audit_rows = _load_patients(n_patients)
    chairs = _build_chairs(n_chairs)

    print(f"\n=== Fairness mitigation head-to-head (n={n_patients}, seed={seed}) ===",
          flush=True)

    arm_off = _run_arm(patients, audit_rows, chairs,
                       fairness_enabled=False, time_limit_s=time_limit_s)
    print(f"  OFF: util={arm_off['utilisation']:.3f}  "
          f"worst ratios = {arm_off['worst_four_fifths_ratio']}",
          flush=True)
    arm_on = _run_arm(patients, audit_rows, chairs,
                      fairness_enabled=True, time_limit_s=time_limit_s)
    print(f"  ON : util={arm_on['utilisation']:.3f}  "
          f"worst ratios = {arm_on['worst_four_fifths_ratio']}",
          flush=True)

    deltas = {
        col: arm_on["worst_four_fifths_ratio"][col]
             - arm_off["worst_four_fifths_ratio"][col]
        for col in arm_off["worst_four_fifths_ratio"]
    }
    util_delta = arm_on["utilisation"] - arm_off["utilisation"]
    print(f"  ratio deltas (ON - OFF): {deltas}", flush=True)
    print(f"  util delta:              {util_delta:+.4f}", flush=True)

    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "n_patients": int(n_patients),
        "n_chairs": int(n_chairs),
        "time_limit_s": float(time_limit_s),
        "seed": int(seed),
        "arm_off": arm_off,
        "arm_on": arm_on,
        "delta_worst_ratio": {k: float(v) for k, v in deltas.items()},
        "delta_utilisation": float(util_delta),
        "comparison_note": (
            "Both arms schedule the same cohort on the same chair grid; "
            "only ScheduleOptimizer._fairness_constraints_enabled differs. "
            "worst_four_fifths_ratio is the MIN over pairwise "
            "min(rate_a, rate_b) / max(rate_a, rate_b) across the "
            "group column's pairs — 1.0 = perfect parity, < 0.80 fails "
            "the Four-Fifths Rule."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients", type=int, default=60)
    parser.add_argument("--n-chairs", type=int, default=12)
    parser.add_argument("--time-limit-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data_cache/fairness_benchmark/results.jsonl",
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
