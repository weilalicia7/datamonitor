"""
Robustness weight-sweep benchmark (§5.9 Figure 5.9 right panel / Improvement I)
==============================================================================

External-review Improvement I asked that Figure 5.9 (slack distribution
histogram of the real schedule) be accompanied by a second panel
showing how R(S) changes under different ``robustness`` objective
weights --- e.g., ``weight_robust = 0.1`` vs ``0.3``.  The existing
§5.11 weight-sensitivity harness sweeps *pairs* of weights (noshow vs
robustness); this harness isolates w_robust alone, so the right
panel of Figure 5.9 reads as a clean response curve.

Scheme
------
  for w in W := [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
      * override OPTIMIZATION_WEIGHTS["robustness"] = w and
        renormalise the other five so the total remains 1.0
      * solve a fixed real cohort (patients.xlsx head(n_patients))
        once with deterministic CP-SAT
      * compute R(S) on the assigned schedule via
        ml.benchmark_robustness.robustness_score
      * bucket slack into Critical / Tight / Adequate / Ample and
        record the fraction in each

The output JSONL is consumed by ``dissertation_analysis.R §7`` which
builds the TWO-panel combined Figure 5.9 with the existing slack
histogram on the left and an R(S)-vs-w_robust line plot on the right.
The default point (w_robust = 0.10 from
``config.py::OPTIMIZATION_WEIGHTS``) is highlighted as a black
diamond so readers can locate the production weight on the curve.

Determinism
-----------
Reuses the deterministic CP-SAT monkey-patch from
``ml.benchmark_component_significance`` --- single-threaded search,
fixed random_seed --- so the curve is reproducible across runs.
Without it, OR-Tools' parallel search emits non-deterministic
incumbents on time-limited solves and the curve would be noisy.

CLI
---
    python -m ml.benchmark_robustness_weight_sweep \
        --n-patients 20 --n-chairs 6 --time-limit-seconds 4
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_WEIGHTS: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


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


def _renormalise_with_robustness(base_weights: Dict[str, float],
                                 w_rob: float) -> Dict[str, float]:
    """Set robustness = w_rob; keep the other five objectives in their
    relative proportions so they sum to 1 - w_rob.  Matches the
    approach in benchmark_ablation.py so §5.9 and §5.8 are on the
    same footing."""
    keys = ("priority", "utilization", "noshow_risk",
            "waiting_time", "travel")
    other_sum = sum(base_weights.get(k, 0.0) for k in keys)
    out = {}
    for k in keys:
        if other_sum > 0:
            out[k] = base_weights.get(k, 0.0) * (1.0 - w_rob) / other_sum
        else:
            out[k] = (1.0 - w_rob) / len(keys)
    out["robustness"] = float(w_rob)
    return out


def _slack_categories_from_appointments(appointments) -> Dict[str, float]:
    """Replicate dissertation_analysis.R §7's category breakdown:
    Critical <10 min, Tight 10-20, Adequate 20-60, Ample >=60."""
    if not appointments:
        return {"critical": 0.0, "tight": 0.0, "adequate": 0.0,
                "ample": 0.0, "n_slacks": 0,
                "mean_slack_min": 0.0, "median_slack_min": 0.0}
    # Compute consecutive-chair slack in minutes
    per_chair: Dict[str, List] = {}
    for a in appointments:
        per_chair.setdefault(a.chair_id, []).append(a)
    slacks: List[float] = []
    for chair_id, appts in per_chair.items():
        appts = sorted(appts, key=lambda x: x.start_time)
        for i in range(len(appts) - 1):
            gap = (appts[i + 1].start_time - appts[i].end_time
                   ).total_seconds() / 60.0
            if gap >= 0:
                slacks.append(gap)
    if not slacks:
        return {"critical": 0.0, "tight": 0.0, "adequate": 0.0,
                "ample": 0.0, "n_slacks": 0,
                "mean_slack_min": 0.0, "median_slack_min": 0.0}
    n = len(slacks)
    n_crit = sum(1 for s in slacks if s < 10)
    n_tight = sum(1 for s in slacks if 10 <= s < 20)
    n_adeq = sum(1 for s in slacks if 20 <= s < 60)
    n_ample = sum(1 for s in slacks if s >= 60)
    slacks_sorted = sorted(slacks)
    median = slacks_sorted[n // 2] if n % 2 else (
        (slacks_sorted[n // 2 - 1] + slacks_sorted[n // 2]) / 2
    )
    return {
        "critical":        float(n_crit / n),
        "tight":           float(n_tight / n),
        "adequate":        float(n_adeq / n),
        "ample":           float(n_ample / n),
        "n_slacks":        int(n),
        "mean_slack_min":  float(sum(slacks) / n),
        "median_slack_min": float(median),
    }


def _solve_at_weight(patients, chairs, *, w_rob: float, time_limit_s: float):
    from optimization.optimizer import ScheduleOptimizer
    from config import OPTIMIZATION_WEIGHTS
    from ml.benchmark_robustness import robustness_score
    # Fresh patient copies so the DRO in-place mutation doesn't leak
    # across weights (same bug as benchmark_component_significance).
    p_copy = [copy.copy(p) for p in patients]
    weights = _renormalise_with_robustness(dict(OPTIMIZATION_WEIGHTS), w_rob)
    opt = ScheduleOptimizer()
    opt.chairs = chairs
    opt.set_components(column_generation=False, fairness=True)
    opt.set_weights(weights, normalise=False)
    t0 = time.perf_counter()
    result = opt.optimize(
        p_copy,
        time_limit_seconds=max(1, int(round(time_limit_s))),
    )
    dt = time.perf_counter() - t0
    appts = result.appointments or []
    rs = robustness_score(appts)["robustness_score"]
    cats = _slack_categories_from_appointments(appts)
    return {
        "w_robust":        float(w_rob),
        "weights":         weights,
        "robustness_score": float(rs),
        "n_scheduled":     int(len(appts)),
        "n_patients_in":   int(len(patients)),
        "solve_time_s":    float(dt),
        **cats,
    }


def benchmark(
    *, n_patients: int, n_chairs: int, time_limit_s: float,
    weights: Sequence[float], seed: int, output_path: Path,
) -> Dict:
    # Deterministic CP-SAT — reuse the monkey-patch from the
    # component-significance harness so single-threaded search +
    # fixed random_seed give reproducible curves.
    try:
        from ml.benchmark_component_significance import _patch_cpsat_for_determinism
        patched = _patch_cpsat_for_determinism(seed=seed)
    except Exception:                                   # pragma: no cover
        patched = False

    patients = _load_patients(n_patients)
    chairs = _build_chairs(n_chairs)
    print(
        f"\n=== Robustness weight sweep (n={n_patients}, chairs={n_chairs}, "
        f"budget={time_limit_s}s, weights={list(weights)}, "
        f"cpsat_det={patched}) ===",
        flush=True,
    )

    sweep: List[Dict] = []
    t0 = time.perf_counter()
    for w in weights:
        arm = _solve_at_weight(
            patients, chairs, w_rob=w, time_limit_s=time_limit_s,
        )
        sweep.append(arm)
        print(
            f"  w_robust={w:.2f}  R(S)={arm['robustness_score']:.3f}  "
            f"crit={arm['critical']:.2f}  tight={arm['tight']:.2f}  "
            f"adeq={arm['adequate']:.2f}  ample={arm['ample']:.2f}  "
            f"solve={arm['solve_time_s']:.2f}s",
            flush=True,
        )

    wall_s = time.perf_counter() - t0
    from config import OPTIMIZATION_WEIGHTS
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_patients": int(n_patients),
        "n_chairs": int(n_chairs),
        "time_limit_s": float(time_limit_s),
        "seed": int(seed),
        "production_default_robustness_weight":
            float(OPTIMIZATION_WEIGHTS.get("robustness", 0.10)),
        "weights_swept": list(map(float, weights)),
        "wall_seconds": float(wall_s),
        "sweep": sweep,
        "method_note": (
            "Sweep of the `robustness` objective weight over "
            "{0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}, with the "
            "other five objectives rescaled to sum to (1 - w_robust) "
            "in their relative proportions.  CP-SAT run with "
            "num_search_workers=1 and random_seed=42 for reproducible "
            "curves.  Feeds the right panel of Figure 5.9 in the "
            "dissertation."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}  (wall={wall_s:.1f}s)")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients",         type=int,   default=20)
    parser.add_argument("--n-chairs",           type=int,   default=6)
    parser.add_argument("--time-limit-seconds", type=float, default=4.0)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument(
        "--weights",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WEIGHTS),
        help="Comma-separated list of w_robust values to sweep",
    )
    parser.add_argument(
        "--output",
        default="data_cache/robustness_weight_sweep/results.jsonl",
    )
    args = parser.parse_args()
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    benchmark(
        n_patients=args.n_patients,
        n_chairs=args.n_chairs,
        time_limit_s=args.time_limit_seconds,
        weights=weights,
        seed=args.seed,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
