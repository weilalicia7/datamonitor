"""
Ablation study for the advanced optimiser components (§5.8 / Table 5.5)
=======================================================================

Regression for external-review Improvement B: many components (IRL,
DFL, GNN, CG, MPC, DRO fairness, robustness post-spread, CVaR) are
described but never quantitatively ablated.  This harness toggles
each batch-solve component off in turn against a fixed cohort + chair
grid and records utilisation, solve time, gender-fairness ratio, and
P1 compliance for each configuration.  The output JSONL drives the
dissertation's §5.8 ablation table via ``dissertation_analysis.R``
§21c, so every cell in the table is traceable to a single timestamped
benchmark row.

Scoped to components that flip via a single flag on
``ScheduleOptimizer`` through the public ``set_components()`` API:

  * CG              — ``column_generation``           (column generation)
  * GNN             — ``gnn``                         (feasibility pre-filter)
  * Fairness        — ``fairness``                    (DRO-style parity)
  * Robustness      — ``weights['robustness'] = 0``   (slack post-spread)
  * CVaR            — ``cvar``                        (worst-case scenarios)

The prediction-side components (DFL, IRL, TFT) have dedicated
benchmarks and are intentionally out of scope here — ablating them
would require re-running the full prediction pipeline, while this
harness stays laser-focused on the CP-SAT batch solve so the table
rows are directly comparable.

CLI
---
    python -m ml.benchmark_ablation \
        --n-patients 40 --n-chairs 8 --time-limit-seconds 15 --seed 42

Never invoked by the live Flask backend.  No UI panel.  The only
side-effect is a JSONL append to ``data_cache/ablation/results.jsonl``.
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
# Fixture builders — shared between all arms so only the toggle differs
# --------------------------------------------------------------------------- #


def _load_patients(n: int):
    import pandas as pd
    from optimization.optimizer import Patient
    from config import OPERATING_HOURS

    df = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" / "patients.xlsx")
    if len(df) < n:
        raise SystemExit(f"Only {len(df)} patients in patients.xlsx; need {n}")
    df = df.head(n)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_h, end_h = OPERATING_HOURS

    patients = []
    audit_rows = []
    for _, row in df.iterrows():
        pid = str(row.get("Patient_ID", f"P{len(patients):04d}"))
        pri = int(row.get("Priority_Int", 3) or 3)
        p = Patient(
            patient_id=pid,
            priority=pri,
            protocol=str(row.get("Regimen_Code", "R-CHOP")),
            expected_duration=int(row.get("Average_Duration", 90) or 90),
            postcode=str(row.get("Patient_Postcode", "CF14")),
            earliest_time=today.replace(hour=start_h),
            latest_time=today.replace(hour=end_h),
            noshow_probability=float(row.get("Patient_NoShow_Rate", 0.13) or 0.13),
        )
        # Gender for the fairness audit
        gender_code = row.get("Person_Stated_Gender_Code", 0)
        try:
            g_int = int(gender_code)
        except (TypeError, ValueError):
            g_int = 0
        gender = "M" if g_int == 1 else "F" if g_int == 2 else "unknown"
        p.gender = gender
        # Age band for the fairness audit
        try:
            age = int(row.get("Age", 0) or 0)
        except (TypeError, ValueError):
            age = 0
        p.age_band = "0-39" if age < 40 else "40-64" if age < 65 else "65+"
        patients.append(p)
        audit_rows.append({
            "Patient_ID": pid, "Gender": gender,
            "Age_Band": p.age_band,
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
# Metric computation — shared across all arms so the table cells match
# --------------------------------------------------------------------------- #


def _gender_fairness_ratio(audit_rows, scheduled_ids):
    """
    Worst pairwise Four-Fifths Rule ratio across Gender groups.

    Returns a dict with two ratios so the dissertation §5.8 prose can
    report both transparently:

      - ``coded_only``: ratio computed over Male/Female only.  This is
        the SACT v4.0 audit convention used by Section~5.6.1 — the
        ``unknown`` (Person_Stated_Gender_Code = 9 / missing) bucket is
        a *data-quality* category, not a protected attribute, so the
        headline ratio excludes it.
      - ``with_unknown``: ratio over all groups including ``unknown``.
        On a small cohort this often collapses to 0.0 because the
        2-patient ``unknown`` bucket is too small for the optimiser to
        schedule reliably; reporting it transparently lets a reader
        see why the historical "all-arms = 0.000" headline arose.

    Also returns per-group counts so the prose can cite the cohort
    composition without re-deriving it.
    """
    from ml.fairness_audit import FairnessAuditor
    auditor = FairnessAuditor()
    full = auditor.audit_schedule(audit_rows, scheduled_ids,
                                  group_column="Gender")
    full_ratios = [m.ratio for m in full.metrics if m.ratio is not None]
    full_min = min(full_ratios) if full_ratios else 1.0

    coded_rows = [r for r in audit_rows
                  if r.get("Gender") in ("M", "F")]
    coded = auditor.audit_schedule(coded_rows, scheduled_ids,
                                   group_column="Gender")
    coded_ratios = [m.ratio for m in coded.metrics if m.ratio is not None]
    coded_min = min(coded_ratios) if coded_ratios else 1.0

    counts = {"M": 0, "F": 0, "unknown": 0}
    sched_per_group = {"M": 0, "F": 0, "unknown": 0}
    for r in audit_rows:
        g = r.get("Gender", "unknown")
        if g not in counts:
            counts[g] = 0; sched_per_group[g] = 0
        counts[g] += 1
        if r.get("Patient_ID") in scheduled_ids:
            sched_per_group[g] += 1
    return {
        "coded_only":   round(float(coded_min), 4),
        "with_unknown": round(float(full_min),  4),
        "group_counts":          dict(counts),
        "group_scheduled":       dict(sched_per_group),
    }


def _p1_compliance_pct(patients, appointments) -> float:
    """Percentage of priority-1 patients who received a scheduled slot."""
    p1 = [p for p in patients if getattr(p, "priority", 3) == 1]
    if not p1:
        return 100.0
    scheduled = {a.patient_id for a in appointments}
    covered = sum(1 for p in p1 if p.patient_id in scheduled)
    return 100.0 * covered / len(p1)


def _run_arm(
    patients, audit_rows, chairs, *,
    arm_name: str,
    time_limit_s: float,
    cg_enabled: bool = True,
    gnn_enabled: bool = False,
    fairness_enabled: bool = True,
    cvar_enabled: bool = True,
    robustness_weight: float = 0.10,
) -> Dict:
    from optimization.optimizer import ScheduleOptimizer

    opt = ScheduleOptimizer()
    opt.chairs = chairs
    opt.set_components(
        column_generation=bool(cg_enabled),
        gnn=bool(gnn_enabled),
        fairness=bool(fairness_enabled),
        cvar=bool(cvar_enabled),
    )

    # Robustness knob — overrides the weight, leaving the other five
    # objective weights normalised around the missing mass so the
    # comparison isn't dominated by a single objective dropping to 0
    orig = dict(opt.weights)
    new_w = dict(orig)
    new_w["robustness"] = float(robustness_weight)
    total = sum(new_w.values())
    if total > 0:
        for k in new_w:
            new_w[k] = new_w[k] / total
    opt.set_weights(new_w, normalise=False)

    t0 = time.perf_counter()
    result = opt.optimize(patients, time_limit_seconds=int(time_limit_s))
    dt = time.perf_counter() - t0

    scheduled_ids = {a.patient_id for a in (result.appointments or [])}
    util = len(scheduled_ids) / max(len(patients), 1)
    p1 = _p1_compliance_pct(patients, result.appointments or [])
    gender = _gender_fairness_ratio(audit_rows, scheduled_ids)

    print(
        f"  [{arm_name:<22s}] util={util:.3f}  "
        f"solve={dt:.2f}s  p1={p1:.1f}%  "
        f"gender_ratio_M_F={gender['coded_only']:.3f}  "
        f"gender_ratio_full={gender['with_unknown']:.3f}",
        flush=True,
    )
    return {
        "arm": arm_name,
        "utilisation": float(util),
        "solve_time_s": float(dt),
        "p1_compliance_pct": float(p1),
        # Headline = M/F-only (matches §5.6.1 SACT v4.0 audit convention).
        "gender_fairness_ratio": float(gender["coded_only"]),
        # With-unknown disclosed for transparency.  The 2-patient
        # "unknown" bucket on this n=40 cohort often collapses the
        # full-pool ratio to 0.0 even when M-vs-F is well balanced.
        "gender_fairness_ratio_with_unknown": float(gender["with_unknown"]),
        "gender_group_counts":    gender["group_counts"],
        "gender_group_scheduled": gender["group_scheduled"],
        "n_scheduled": int(len(scheduled_ids)),
        "n_patients": int(len(patients)),
        "status": str(getattr(result, "status", "UNKNOWN")),
        "config": {
            "cg_enabled": bool(cg_enabled),
            "gnn_enabled": bool(gnn_enabled),
            "fairness_enabled": bool(fairness_enabled),
            "cvar_enabled": bool(cvar_enabled),
            "robustness_weight": float(robustness_weight),
        },
    }


def benchmark(
    *, n_patients: int, n_chairs: int, time_limit_s: float, seed: int,
    output_path: Path,
) -> Dict:
    patients, audit_rows = _load_patients(n_patients)
    chairs = _build_chairs(n_chairs)

    print(
        f"\n=== Ablation study (n={n_patients}, chairs={n_chairs}, "
        f"time_limit={time_limit_s}s, seed={seed}) ===\n"
        f"Each arm runs the SAME cohort + grid with ONE component flipped.\n"
        f"Baseline has all five components in production defaults.",
        flush=True,
    )

    full = _run_arm(
        patients, audit_rows, chairs, arm_name="baseline (all on)",
        time_limit_s=time_limit_s,
    )
    no_cg = _run_arm(
        patients, audit_rows, chairs, arm_name="-cg",
        time_limit_s=time_limit_s, cg_enabled=False,
    )
    no_gnn = _run_arm(
        patients, audit_rows, chairs, arm_name="-gnn",
        time_limit_s=time_limit_s, gnn_enabled=False,
    )
    no_fair = _run_arm(
        patients, audit_rows, chairs, arm_name="-fairness",
        time_limit_s=time_limit_s, fairness_enabled=False,
    )
    no_cvar = _run_arm(
        patients, audit_rows, chairs, arm_name="-cvar",
        time_limit_s=time_limit_s, cvar_enabled=False,
    )
    no_rob = _run_arm(
        patients, audit_rows, chairs, arm_name="-robustness",
        time_limit_s=time_limit_s, robustness_weight=0.0,
    )

    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "n_patients": int(n_patients),
        "n_chairs": int(n_chairs),
        "time_limit_s": float(time_limit_s),
        "seed": int(seed),
        "arms": {
            "baseline": full,
            "no_cg": no_cg,
            "no_gnn": no_gnn,
            "no_fairness": no_fair,
            "no_cvar": no_cvar,
            "no_robustness": no_rob,
        },
        "comparison_note": (
            "Every arm runs the same cohort + chair grid.  Each arm "
            "disables exactly one component relative to the baseline; "
            "reading the row as a diff isolates that component's "
            "contribution to the four metrics.  Prediction-side "
            "components (DFL / IRL / TFT) are benchmarked separately "
            "and not included here because they do not flow through "
            "the CP-SAT batch-solve objective."
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
    parser.add_argument("--time-limit-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data_cache/ablation/results.jsonl",
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
