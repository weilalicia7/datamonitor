"""
Real benchmark: monolithic CP-SAT vs.\ Column Generation
========================================================

Drives ScheduleOptimizer.optimize() against the production patients
data for a sequence of cohort sizes and writes one timed row per
(n_patients, solver) pair to ``data_cache/cg_benchmark/results.jsonl``.

Single source of truth for dissertation §4.5.X Table 4.4.  Reading
this benchmark instead of an in-R simulation answers the brief's
"needs real data to test" requirement and guarantees the table's
speedup column matches the CP-SAT-time / CG-time ratio by construction
(both are measured by the same wall-clock).

The script is invoked manually from a clean Python process, NOT from
the live Flask backend, so it never interferes with the prediction
pipeline.  No UI panel; no email; the only side-effect is the JSONL
write.

CLI
---
    python -m ml.benchmark_column_generation \
        --patient-counts 30,50,75,100,150 \
        --time-limit-seconds 30 \
        --chairs 45 \
        --output data_cache/cg_benchmark/results.jsonl

Each row in the output JSONL has shape::

    {
      "ts":               ISO8601,
      "n_patients":       int,
      "n_chairs":         int,
      "time_limit_s":     float,   # the wall-clock cap passed to both solvers
      "cpsat_time_s":     float,   # measured solve time for monolithic CP-SAT
      "cpsat_status":     str,     # OPTIMAL / FEASIBLE / TIMEOUT / FAILED
      "cpsat_n_scheduled": int,
      "cg_time_s":        float,   # measured solve time for column generation
      "cg_status":        str,     # CG_OPTIMAL / CG_TIME_LIMIT / CG_MAX_ITER / etc.
      "cg_n_scheduled":   int,
      "speedup":          float,   # cpsat_time_s / cg_time_s, computed AT THE
                                   # MEASUREMENT SITE so dissertation table
                                   # cell == column ratio by construction
      "cpsat_timed_out":  bool     # True iff cpsat_time_s >= time_limit_s
    }

The R analysis in dissertation_analysis.R §16 prefers this JSONL when
present and falls back to the calibrated empirical model only when
the file is missing or empty.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Defer heavy imports until after argparse so --help is fast
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _build_patients(n: int):
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


def _solve(opt, patients, time_limit_s: float, *, use_cg: bool):
    """Run a single solve; return (wall_seconds, status, n_scheduled)."""
    opt._cg_enabled = use_cg
    t0 = time.perf_counter()
    result = opt.optimize(patients, time_limit_seconds=int(time_limit_s))
    dt = time.perf_counter() - t0
    status = str(getattr(result, "status", "UNKNOWN"))
    n_sched = len(getattr(result, "appointments", []) or [])
    return dt, status, n_sched


def benchmark(
    patient_counts: List[int],
    *,
    n_chairs: int = 45,
    time_limit_s: float = 30.0,
    output_path: Path,
) -> List[dict]:
    from optimization.optimizer import ScheduleOptimizer

    rows: List[dict] = []
    chairs = _build_chairs(n_chairs)

    for n in patient_counts:
        print(f"\n=== n_patients={n} ===", flush=True)
        patients = _build_patients(n)

        # Fresh optimiser per cohort so warm-start cache from a previous
        # solve cannot bias the CP-SAT timing.  We also clear the
        # optimiser's chair list and re-set it to the deterministic
        # benchmark chairs (45 by default).
        opt_cpsat = ScheduleOptimizer()
        opt_cpsat.chairs = chairs
        cpsat_dt, cpsat_status, cpsat_sched = _solve(
            opt_cpsat, patients, time_limit_s, use_cg=False,
        )
        print(
            f"  CP-SAT : {cpsat_dt:7.3f} s  status={cpsat_status:14s} "
            f"scheduled={cpsat_sched}/{n}",
            flush=True,
        )

        opt_cg = ScheduleOptimizer()
        opt_cg.chairs = chairs
        cg_dt, cg_status, cg_sched = _solve(
            opt_cg, patients, time_limit_s, use_cg=True,
        )
        print(
            f"  CG     : {cg_dt:7.3f} s  status={cg_status:14s} "
            f"scheduled={cg_sched}/{n}",
            flush=True,
        )

        cpsat_timed_out = cpsat_dt >= time_limit_s - 0.5  # leave half-second slop
        speedup = cpsat_dt / max(cg_dt, 1e-6)
        print(
            f"  ratio  : {speedup:.2f}x"
            + (f"  (CP-SAT hit time-limit)" if cpsat_timed_out else ""),
            flush=True,
        )

        rows.append({
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "n_patients": int(n),
            "n_chairs": int(n_chairs),
            "time_limit_s": float(time_limit_s),
            "cpsat_time_s": float(cpsat_dt),
            "cpsat_status": cpsat_status,
            "cpsat_n_scheduled": int(cpsat_sched),
            "cg_time_s": float(cg_dt),
            "cg_status": cg_status,
            "cg_n_scheduled": int(cg_sched),
            "speedup": float(speedup),
            "cpsat_timed_out": bool(cpsat_timed_out),
        })

    # Append (don't overwrite) so multiple benchmark runs accumulate as evidence
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nAppended {len(rows)} rows to {output_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--patient-counts", default="30,50,75,100,150",
        help="Comma-separated cohort sizes to benchmark",
    )
    parser.add_argument(
        "--time-limit-seconds", type=float, default=30.0,
        help="Wall-clock cap passed to both CP-SAT and CG (s)",
    )
    parser.add_argument("--chairs", type=int, default=45)
    parser.add_argument(
        "--output", default="data_cache/cg_benchmark/results.jsonl",
    )
    args = parser.parse_args()

    counts = [int(x) for x in args.patient_counts.split(",")]
    benchmark(
        counts,
        n_chairs=args.chairs,
        time_limit_s=args.time_limit_seconds,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
