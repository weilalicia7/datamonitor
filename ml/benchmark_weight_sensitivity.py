"""
Weight-sensitivity sweep (§5.11 / Improvement E)
================================================

External-review Improvement E asked for a Pareto frontier plot
showing trade-offs between pairs of optimisation objectives — e.g.,
utilisation vs. waiting time, no-show risk vs. robustness — and
evidence that the production default weights are near-optimal on
those frontiers.

This harness sweeps two Pareto frontiers by interpolating between
two weights while keeping the other four frozen at the production
defaults (``OPTIMIZATION_WEIGHTS`` from ``config.py``).  For each
sweep point it runs the solver once and records six scalar
measurements of the resulting schedule:

    utilisation                 (fraction of patients scheduled)
    mean_wait_min               (mean minutes between earliest_time and start)
    mean_scheduled_noshow_rate  (mean no-show probability averaged over scheduled patients only)
    robustness_score            (= ml/benchmark_robustness.robustness_score)
    p1_compliance               (fraction of priority-1 patients scheduled)
    solve_time_s

The R analysis (``dissertation_analysis.R §21c``) reads the JSONL,
generates two figures (fig27 utilisation-vs-wait, fig28 noshow-vs-
robustness) with the default-weight point overlaid as a larger
marker, and emits near-optimality macros (nearest-distance of the
default point to its own Pareto front, fraction of sweep points
dominated by the default, etc.).

Scope
-----
To keep the benchmark under a minute we:
  * use 20 patients and a 6-chair grid
  * disable the expensive DRO+CVaR branch via _use_cvar_objective=False
  * disable GNN + column generation
  * use an 8-second CP-SAT time limit per point (most points converge
    in well under a second on this size)

The sweep therefore measures the objective trade-off attributable to
the weight change, not the interplay with DRO / CVaR / CG / GNN —
those components have their own benchmarks (§4.5.1, §5.6.2, §5.8
already cover their individual contributions).

CLI
---
    python -m ml.benchmark_weight_sensitivity --n-patients 20 --n-chairs 6 \
        --points-per-frontier 11 --time-limit-seconds 8
"""
from __future__ import annotations

import argparse
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
    df = df.head(n).reset_index(drop=True)
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


def _score_point(appts, patients) -> Dict:
    from ml.benchmark_robustness import robustness_score

    if not appts:
        return {
            "utilisation": 0.0, "mean_wait_min": 0.0,
            "mean_scheduled_noshow_rate": 0.0, "robustness_score": 1.0,
            "p1_compliance": 100.0, "n_scheduled": 0,
        }
    by_id = {p.patient_id: p for p in patients}
    scheduled_ids = {a.patient_id for a in appts}
    waits, ns = [], []
    for a in appts:
        p = by_id.get(a.patient_id)
        if p is None:
            continue
        w = (a.start_time - p.earliest_time).total_seconds() / 60.0
        if w >= 0:
            waits.append(w)
        ns.append(float(p.noshow_probability))
    util = len(scheduled_ids) / max(len(patients), 1)
    mean_wait = sum(waits) / max(len(waits), 1)
    mean_ns = sum(ns) / max(len(ns), 1)
    rob = robustness_score(appts)["robustness_score"]
    p1 = [p for p in patients if getattr(p, "priority", 3) == 1]
    p1_ok = sum(1 for p in p1 if p.patient_id in scheduled_ids)
    p1_pct = 100.0 * p1_ok / max(len(p1), 1) if p1 else 100.0
    return {
        "utilisation": float(util),
        "mean_wait_min": float(mean_wait),
        "mean_scheduled_noshow_rate": float(mean_ns),
        "robustness_score": float(rob),
        "p1_compliance": float(p1_pct),
        "n_scheduled": int(len(scheduled_ids)),
    }


def _solve_with_weights(patients, chairs, weights, time_limit_s: float):
    from optimization.optimizer import ScheduleOptimizer
    opt = ScheduleOptimizer()
    opt.chairs = chairs
    # Keep the sweep fast + isolated — the trade-off is driven by the
    # weight change, not by the heavy components which already have
    # their own benchmarks.
    opt._cg_enabled = False
    opt._gnn_enabled = False
    opt._use_cvar_objective = False
    opt._fairness_constraints_enabled = False
    opt.set_weights(weights, normalise=False)
    t0 = time.perf_counter()
    result = opt.optimize(patients, time_limit_seconds=int(time_limit_s))
    dt = time.perf_counter() - t0
    return result, dt


def _sweep_frontier(
    patients, chairs, *,
    defaults: Dict, axis_a: str, axis_b: str,
    n_points: int, time_limit_s: float,
) -> List[Dict]:
    """Interpolate the pair (axis_a, axis_b) across its combined mass
    while keeping the other 4 weights frozen at their defaults.  Each
    interpolation step sets axis_a_weight = t and axis_b_weight =
    (default_a + default_b) - t for t ∈ linspace(0, default_a+default_b,
    n_points).  Total mass stays invariant so the non-swept weights
    don't dilute."""
    mass_ab = defaults[axis_a] + defaults[axis_b]
    results = []
    for i in range(n_points):
        t = mass_ab * i / (n_points - 1) if n_points > 1 else defaults[axis_a]
        w = dict(defaults)
        w[axis_a] = t
        w[axis_b] = mass_ab - t
        # defensive: avoid negative weights from rounding
        w[axis_a] = max(0.0, w[axis_a])
        w[axis_b] = max(0.0, w[axis_b])
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        res, dt = _solve_with_weights(patients, chairs, w, time_limit_s)
        score = _score_point(res.appointments or [], patients)
        score.update({
            "frontier": f"{axis_a}_vs_{axis_b}",
            "axis_a_weight": float(w[axis_a]),
            "axis_b_weight": float(w[axis_b]),
            "t_fraction": float(i / (n_points - 1)) if n_points > 1 else 0.0,
            "solve_time_s": float(dt),
        })
        results.append(score)
    return results


def benchmark(
    *, n_patients: int, n_chairs: int, points_per_frontier: int,
    time_limit_s: float, output_path: Path,
) -> Dict:
    from config import OPTIMIZATION_WEIGHTS, OPERATING_HOURS

    patients = _load_patients(n_patients)
    chairs = _build_chairs(n_chairs)
    start_h, end_h = OPERATING_HOURS
    horizon_min = (end_h - start_h) * 60
    defaults = dict(OPTIMIZATION_WEIGHTS)

    print(
        f"\n=== Weight sensitivity sweep (n={n_patients}, chairs={n_chairs}, "
        f"points/frontier={points_per_frontier}, budget={time_limit_s}s) ===",
        flush=True,
    )
    print(f"Default weights: {defaults}", flush=True)

    # Measure the default (production) point once with the SAME solver
    # configuration as the sweep arms (DRO/CVaR/GNN/fairness all off)
    # so the default marker is apples-to-apples comparable with the
    # frontier traces on fig27/fig28.
    res_def, dt_def = _solve_with_weights(
        patients, chairs, defaults, time_limit_s,
    )
    default_score = _score_point(res_def.appointments or [], patients)
    default_score.update({
        "frontier": "default",
        "solve_time_s": float(dt_def),
        "weights": defaults,
    })
    print(
        f"  [DEFAULT] util={default_score['utilisation']:.3f}  "
        f"wait={default_score['mean_wait_min']:.1f}m  "
        f"noshow={default_score['mean_scheduled_noshow_rate']:.3f}  "
        f"rob={default_score['robustness_score']:.3f}",
        flush=True,
    )

    print(f"\nFrontier A: utilization vs waiting_time", flush=True)
    frontier_a = _sweep_frontier(
        patients, chairs, defaults=defaults,
        axis_a="utilization", axis_b="waiting_time",
        n_points=points_per_frontier, time_limit_s=time_limit_s,
    )
    for p in frontier_a:
        print(
            f"  t={p['t_fraction']:.2f}  w_util={p['axis_a_weight']:.3f}  "
            f"w_wait={p['axis_b_weight']:.3f}  "
            f"util={p['utilisation']:.3f}  wait={p['mean_wait_min']:.1f}m",
            flush=True,
        )

    print(f"\nFrontier B: noshow_risk vs robustness", flush=True)
    frontier_b = _sweep_frontier(
        patients, chairs, defaults=defaults,
        axis_a="noshow_risk", axis_b="robustness",
        n_points=points_per_frontier, time_limit_s=time_limit_s,
    )
    for p in frontier_b:
        print(
            f"  t={p['t_fraction']:.2f}  w_noshow={p['axis_a_weight']:.3f}  "
            f"w_rob={p['axis_b_weight']:.3f}  "
            f"noshow_exp={p['mean_scheduled_noshow_rate']:.3f}  "
            f"R(S)={p['robustness_score']:.3f}",
            flush=True,
        )

    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "n_patients": int(n_patients),
        "n_chairs": int(n_chairs),
        "points_per_frontier": int(points_per_frontier),
        "time_limit_s": float(time_limit_s),
        "horizon_min": int(horizon_min),
        "default_weights": defaults,
        "default_point": default_score,
        "frontier_util_vs_wait": frontier_a,
        "frontier_noshow_vs_robust": frontier_b,
        "comparison_note": (
            "Two Pareto frontiers per run: (A) utilisation vs. waiting "
            "time by sweeping the pair of weights while keeping the "
            "other four frozen at production defaults; (B) no-show "
            "risk vs. robustness, same technique.  The default point "
            "is measured once and logged separately so R can overlay "
            "it on the plots and compute its distance to the nearest "
            "swept frontier point."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients", type=int, default=20)
    parser.add_argument("--n-chairs", type=int, default=6)
    parser.add_argument("--points-per-frontier", type=int, default=11)
    parser.add_argument("--time-limit-seconds", type=float, default=8.0)
    parser.add_argument(
        "--output", default="data_cache/weight_sensitivity/results.jsonl",
    )
    args = parser.parse_args()
    benchmark(
        n_patients=args.n_patients,
        n_chairs=args.n_chairs,
        points_per_frontier=args.points_per_frontier,
        time_limit_s=args.time_limit_seconds,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
