"""
Paired-bootstrap significance benchmark (§5.12 / Improvement F)
===============================================================

External-review Improvement F asked for statistical significance
testing of each advanced solver component, not just the main outcome
headline.  This harness answers: *for each of the five CP-SAT
components (column generation, GNN pre-filter, CVaR, fairness
constraints, robustness), is the measured effect on utilisation /
gender-fairness / solve-time distinguishable from zero at a 100-replica
bootstrap?*

Design
------
For each component C ∈ {column_generation, gnn, cvar, fairness,
robustness}:

  repeat B times (default B = 100):
    * draw a bootstrap cohort of n patients with replacement from
      the real patients.xlsx head(n_patients) sample
    * solve the SAME cohort twice:
        baseline  — all five components ON at production defaults
        off-C     — C flipped off, the other four unchanged
      using the public ``ScheduleOptimizer.set_components`` API
    * record Δ_util   = util_baseline      - util_offC
             Δ_fair   = fairness_baseline  - fairness_offC
             Δ_time   = solve_baseline     - solve_offC

  then for each of the three metrics report:
    * mean Δ           — point estimate of the component's effect
    * 95 % percentile bootstrap confidence interval
    * two-sided bootstrap p-value
         p = 2 · min(P(Δ ≤ 0), P(Δ ≥ 0))
      floor-clipped at 1 / (B + 1) so it never reports exactly 0
      (bootstrap p cannot distinguish finer than ~1/B)

A metric with Δ-mean far from 0 and a CI that excludes 0 gives
p ≈ 0.01 (the floor for B = 100); a component with no measurable
effect gives p ≈ 1.0.

Scope + fidelity
----------------
The column-generation component auto-routes only above
``COLUMN_GEN_THRESHOLD`` (typically 50 patients).  On the ≤ 30-patient
bootstrap cohorts used here, CG would not engage with the default
threshold — so the CG arm also sets ``_cg_threshold = 1`` via the
benchmark.  This matches what ``tests/test_column_generation.py`` does
to exercise the CG path at small n; the comparison is therefore
"forced-CG vs forced-CP-SAT" rather than "auto-routed CG".  This is
documented in the §5.12 prose.

To keep the run under ~10 minutes we default to:
  * 10 patients / 3 chairs (enough variation for effects to surface)
  * 1-second CP-SAT budget per solve
  * 100 bootstrap replicas
Expected wall time: 100 × 5 × 2 × ~0.3 s ≈ 5 minutes.

CLI
---
    python -m ml.benchmark_component_significance \
        --n-patients 10 --n-chairs 3 \
        --n-bootstrap 100 --time-limit-seconds 1

Never invoked by the live Flask backend — CLI-only, benchmark-only.
The companion status endpoint ``/api/metrics/component-significance/status``
is a read-only diagnostic; it reads the JSONL and returns the latest
row but does not trigger a run.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _patch_cpsat_for_determinism(seed: int = 42) -> bool:
    """Monkey-patch OR-Tools ``CpSolver`` so every instantiation in
    this process runs single-threaded with a fixed ``random_seed``.

    Rationale: without this, CP-SAT's parallel search produces
    non-deterministic incumbents on time-limited solves, which means
    a paired bootstrap would pick up SOLVER variance as if it were
    COMPONENT variance.  Benchmarked: identical inputs across three
    solves gave utilisations (0.750, 0.500, 0.500) without the patch.
    With the patch, paired solves on the same cohort are reproducible,
    so the Δ reported per replica is a genuine component effect.

    Returns True if the patch was applied, False if OR-Tools is not
    importable (e.g., stub install).
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return False
    orig = cp_model.CpSolver
    class _DeterministicCpSolver(orig):  # type: ignore[valid-type, misc]
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.parameters.num_search_workers = 1
            self.parameters.random_seed = int(seed)
    cp_model.CpSolver = _DeterministicCpSolver
    return True


# --------------------------------------------------------------------------- #
# Cohort + chairs (mirrors benchmark_ablation.py so §5.12 and §5.8 cells come
# from the same data source)
# --------------------------------------------------------------------------- #


def _load_patients(n: int):
    import pandas as pd
    from optimization.optimizer import Patient
    from config import OPERATING_HOURS

    df = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" / "patients.xlsx")
    df = df.head(n).reset_index(drop=True)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_h, end_h = OPERATING_HOURS
    patients, audit_rows = [], []
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
        g_code = row.get("Person_Stated_Gender_Code", 0)
        try:
            g_int = int(g_code)
        except (TypeError, ValueError):
            g_int = 0
        gender = "M" if g_int == 1 else "F" if g_int == 2 else "unknown"
        p.gender = gender
        try:
            age = int(row.get("Age", 0) or 0)
        except (TypeError, ValueError):
            age = 0
        p.age_band = "0-39" if age < 40 else "40-64" if age < 65 else "65+"
        patients.append(p)
        audit_rows.append({
            "Patient_ID": pid, "Gender": gender, "Age_Band": p.age_band,
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
# Scoring helpers — identical formulas to benchmark_ablation.py so §5.12
# values are apples-to-apples comparable with Table 5.5
# --------------------------------------------------------------------------- #


def _gender_fairness_ratio(audit_rows, scheduled_ids) -> float:
    from ml.fairness_audit import FairnessAuditor
    auditor = FairnessAuditor()
    report = auditor.audit_schedule(
        audit_rows, scheduled_ids, group_column="Gender",
    )
    ratios = [m.ratio for m in report.metrics if m.ratio is not None]
    return float(min(ratios)) if ratios else 1.0


# --------------------------------------------------------------------------- #
# Single arm + paired solve
# --------------------------------------------------------------------------- #


def _bootstrap_cohort(patients, audit_rows, rng):
    """Draw a size-n bootstrap sample with replacement.  Suffix each
    patient's ID and audit row key so duplicates don't collide inside
    the solver."""
    from optimization.optimizer import Patient
    n = len(patients)
    idx = [rng.randrange(n) for _ in range(n)]
    boot_p, boot_audit = [], []
    for draw, i in enumerate(idx):
        src = patients[i]
        src_audit = audit_rows[i]
        # Shallow-copy the Patient so mutating id doesn't disturb the
        # original cohort for the next replica.
        p = copy.copy(src)
        new_id = f"{src.patient_id}__b{draw:03d}"
        p.patient_id = new_id
        # gender + age_band are attribute assignments; copy preserves them
        boot_p.append(p)
        boot_audit.append({
            "Patient_ID": new_id,
            "Gender": src_audit["Gender"],
            "Age_Band": src_audit["Age_Band"],
        })
    return boot_p, boot_audit


def _solve_and_score(
    patients, audit_rows, chairs, *,
    time_limit_s: float,
    component_flags: Dict[str, bool],
    force_cg_threshold: Optional[int] = None,
    robustness_weight: Optional[float] = None,
) -> Dict[str, float]:
    from optimization.optimizer import ScheduleOptimizer
    # Deep-copy patients per call: ScheduleOptimizer.optimize mutates
    # expected_duration (DRO buffering) and noshow_probability (event
    # adjustment) in place.  Without a fresh copy the paired second
    # solve sees inflated durations from the first, corrupting the
    # bootstrap Δ.  Audit rows are plain dicts so we don't need to
    # touch them.
    patients = [copy.copy(p) for p in patients]
    opt = ScheduleOptimizer()
    opt.chairs = chairs
    opt.set_components(
        column_generation=component_flags.get("column_generation", True),
        gnn=component_flags.get("gnn", False),
        cvar=component_flags.get("cvar", True),
        fairness=component_flags.get("fairness", True),
    )
    if force_cg_threshold is not None:
        opt._cg_threshold = int(force_cg_threshold)
    if robustness_weight is not None:
        w = dict(opt.weights)
        w["robustness"] = float(robustness_weight)
        tot = sum(w.values())
        if tot > 0:
            w = {k: v / tot for k, v in w.items()}
        opt.set_weights(w, normalise=False)

    t0 = time.perf_counter()
    # API declares int; round-up to 1 so sub-second CLI values don't
    # collapse to 0 and bail the CP-SAT solver before any work.
    result = opt.optimize(patients, time_limit_seconds=max(1, int(round(time_limit_s))))
    dt = time.perf_counter() - t0

    appts = result.appointments or []
    scheduled_ids = {a.patient_id for a in appts}
    util = len(scheduled_ids) / max(len(patients), 1)
    gender_ratio = _gender_fairness_ratio(audit_rows, scheduled_ids)

    return {
        "utilisation": float(util),
        "gender_fairness_ratio": float(gender_ratio),
        "solve_time_s": float(dt),
        "n_scheduled": int(len(scheduled_ids)),
    }


# Component definitions: key ↔ (off-baseline override, extras)
# Production defaults: CG=True, GNN=False, CVaR=True, fairness=True,
# robustness_weight=0.10 (from OPTIMIZATION_WEIGHTS).
#
# Each entry describes what "off" means.  Baseline is always all-on.
_COMPONENTS = {
    "column_generation": {
        "label": "Column generation (CG)",
        "baseline": {"column_generation": True},
        "off":      {"column_generation": False},
        # Force CG to actually engage on the small bootstrap cohort —
        # matches tests/test_column_generation.py.  Without this the
        # default threshold (>= 50) skips CG and CG-on == CG-off == 0.
        "force_cg_threshold": 1,
    },
    "gnn": {
        "label": "GNN feasibility pre-filter",
        # Production baseline has GNN OFF (it's opt-in via
        # enable_gnn_pruning).  We therefore flip GNN ON for the
        # "baseline" of this component and keep OFF as the "off" arm,
        # reversing the usual sense.  The Δ reported is still
        # baseline-minus-off, so positive Δ means "GNN on improves".
        "baseline": {"gnn": True},
        "off":      {"gnn": False},
    },
    "cvar": {
        "label": "CVaR worst-case objective",
        "baseline": {"cvar": True},
        "off":      {"cvar": False},
    },
    "fairness": {
        "label": "DRO fairness penalties",
        "baseline": {"fairness": True},
        "off":      {"fairness": False},
    },
    "robustness": {
        "label": "Slack-post-spread robustness",
        # Robustness is driven by the `robustness` weight, not a flag.
        # Baseline uses the production weight (read from config at run
        # time); "off" drops it to 0.
        "baseline": {},  # no component-flag changes
        "off":      {},
        "robustness_weight_on": None,   # populated at run time
        "robustness_weight_off": 0.0,
    },
}


def _run_paired(
    patients, audit_rows, chairs, *,
    component_key: str,
    time_limit_s: float,
    production_robustness_weight: float,
) -> Dict[str, float]:
    """One bootstrap replica: solve baseline-arm + off-arm, return Δs."""
    cfg = _COMPONENTS[component_key]
    on_flags = cfg.get("baseline", {})
    off_flags = cfg.get("off", {})
    thr = cfg.get("force_cg_threshold")

    if component_key == "robustness":
        w_on = production_robustness_weight
        w_off = 0.0
    else:
        w_on = None
        w_off = None

    s_on = _solve_and_score(
        patients, audit_rows, chairs,
        time_limit_s=time_limit_s,
        component_flags=on_flags,
        force_cg_threshold=thr,
        robustness_weight=w_on,
    )
    s_off = _solve_and_score(
        patients, audit_rows, chairs,
        time_limit_s=time_limit_s,
        component_flags=off_flags,
        force_cg_threshold=thr,
        robustness_weight=w_off,
    )
    return {
        "delta_utilisation":          s_on["utilisation"]          - s_off["utilisation"],
        "delta_gender_fairness_ratio": s_on["gender_fairness_ratio"] - s_off["gender_fairness_ratio"],
        "delta_solve_time_s":         s_on["solve_time_s"]         - s_off["solve_time_s"],
    }


# --------------------------------------------------------------------------- #
# Bootstrap statistics: mean, 95% CI, two-sided p-value
# --------------------------------------------------------------------------- #


def _percentile(xs: List[float], q: float) -> float:
    """Linear-interpolation percentile for q ∈ [0, 1]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _bootstrap_stats(deltas: List[float]) -> Dict[str, float]:
    """Point estimate + 95 % percentile CI + two-sided bootstrap p-value."""
    xs = [d for d in deltas if d is not None]
    n = len(xs)
    if n == 0:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "p_value": 1.0, "n_valid": 0}
    mean = sum(xs) / n
    lo = _percentile(xs, 0.025)
    hi = _percentile(xs, 0.975)
    # Two-sided bootstrap p:  if the effect is on the positive side,
    # count how many replicas are ≤ 0; otherwise count ≥ 0.
    if mean >= 0:
        tail = sum(1 for d in xs if d <= 0) / n
    else:
        tail = sum(1 for d in xs if d >= 0) / n
    p_raw = 2.0 * tail
    p = max(min(p_raw, 1.0), 1.0 / (n + 1))
    return {
        "mean_delta": float(mean),
        "ci_low":     float(lo),
        "ci_high":    float(hi),
        "p_value":    float(p),
        "n_valid":    int(n),
    }


# --------------------------------------------------------------------------- #
# Main benchmark
# --------------------------------------------------------------------------- #


def benchmark(
    *, n_patients: int, n_chairs: int, n_bootstrap: int,
    time_limit_s: float, seed: int, output_path: Path,
) -> Dict:
    from config import OPTIMIZATION_WEIGHTS

    # Pin CP-SAT to single-threaded deterministic search before any
    # solve happens.  Without this the multi-threaded default produces
    # non-deterministic incumbents on time-limited solves, which would
    # corrupt the paired bootstrap (see _patch_cpsat_for_determinism).
    patched = _patch_cpsat_for_determinism(seed=seed)

    patients, audit_rows = _load_patients(n_patients)
    chairs = _build_chairs(n_chairs)
    rng = random.Random(seed)
    prod_rob_w = float(OPTIMIZATION_WEIGHTS.get("robustness", 0.10))

    print(
        f"\n=== Component-significance bootstrap (n={n_patients}, "
        f"chairs={n_chairs}, B={n_bootstrap}, budget={time_limit_s}s, "
        f"seed={seed}, cpsat_deterministic={patched}) ===",
        flush=True,
    )
    print(
        f"Components sweeping: {', '.join(_COMPONENTS.keys())}",
        flush=True,
    )

    components_out: Dict[str, Dict] = {}
    t_start = time.perf_counter()
    for ckey in _COMPONENTS:
        label = _COMPONENTS[ckey]["label"]
        print(f"\n[{label}]", flush=True)
        delta_u, delta_f, delta_t = [], [], []
        for b in range(n_bootstrap):
            boot_p, boot_audit = _bootstrap_cohort(patients, audit_rows, rng)
            deltas = _run_paired(
                boot_p, boot_audit, chairs,
                component_key=ckey,
                time_limit_s=time_limit_s,
                production_robustness_weight=prod_rob_w,
            )
            delta_u.append(deltas["delta_utilisation"])
            delta_f.append(deltas["delta_gender_fairness_ratio"])
            delta_t.append(deltas["delta_solve_time_s"])
            if (b + 1) % 20 == 0 or b == n_bootstrap - 1:
                print(
                    f"  replica {b+1:3d}/{n_bootstrap} "
                    f"d_util~{sum(delta_u)/len(delta_u):+.3f} "
                    f"d_fair~{sum(delta_f)/len(delta_f):+.3f} "
                    f"d_time~{sum(delta_t)/len(delta_t):+.3f}s",
                    flush=True,
                )
        components_out[ckey] = {
            "label": label,
            "metrics": {
                "utilisation":            _bootstrap_stats(delta_u),
                "gender_fairness_ratio":  _bootstrap_stats(delta_f),
                "solve_time_s":           _bootstrap_stats(delta_t),
            },
        }

    wall_s = time.perf_counter() - t_start
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_patients": int(n_patients),
        "n_chairs":   int(n_chairs),
        "n_bootstrap": int(n_bootstrap),
        "time_limit_s": float(time_limit_s),
        "seed": int(seed),
        "production_robustness_weight": float(prod_rob_w),
        "wall_seconds": float(wall_s),
        "components": components_out,
        "metric_keys": ["utilisation", "gender_fairness_ratio", "solve_time_s"],
        "method_note": (
            "Paired bootstrap: each replica draws a size-n cohort with "
            "replacement, solves the same cohort twice (baseline / "
            "component-off), and records the Δ per metric.  p is "
            "two-sided bootstrap: 2·min(P(Δ≤0), P(Δ≥0)), floor-clipped "
            "at 1/(B+1).  CI is 2.5%/97.5% percentile of the Δ "
            "distribution across replicas."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}  (wall={wall_s:.1f}s)")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients",   type=int,   default=10)
    parser.add_argument("--n-chairs",     type=int,   default=3)
    parser.add_argument("--n-bootstrap",  type=int,   default=100)
    parser.add_argument("--time-limit-seconds", type=float, default=1.0,
                        help="Per-solve CP-SAT budget (rounded up to the "
                             "nearest second, minimum 1).")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--output",
        default="data_cache/component_significance/results.jsonl",
    )
    args = parser.parse_args()
    benchmark(
        n_patients=args.n_patients,
        n_chairs=args.n_chairs,
        n_bootstrap=args.n_bootstrap,
        time_limit_s=args.time_limit_seconds,
        seed=args.seed,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
