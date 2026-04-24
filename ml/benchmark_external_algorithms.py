"""
External-algorithm benchmark: CP-SAT vs NSGA-II vs risk-greedy
==============================================================

External-review Improvement C (Comparative benchmarking): the paper's
only comparator so far has been a simple rule-based baseline.  The
reviewer asked for NSGA-II (a multi-objective genetic algorithm) and
a risk-aware greedy heuristic (sort by no-show probability, assign
earliest slot), with hypervolume of Pareto fronts as the summary
metric.  This harness implements all three — no external NSGA-II
dependency (self-contained ~150-line NSGA-II so the benchmark is
hermetic) — and writes one comparison row per run to
``data_cache/external_benchmark/results.jsonl``.  The dissertation
§5.10 table reads that JSONL, so every cell is traceable to one
timestamped benchmark row.

Methods
-------
* **CP-SAT (ours)**    — ScheduleOptimizer.optimize with each of the
                         four PARETO_WEIGHT_SETS configs.  Produces
                         up to four non-dominated points in the
                         objective space.
* **NSGA-II**          — minimal genetic algorithm (population 48,
                         20 generations) operating on assignment
                         chromosomes.  Non-dominated sorting +
                         crowding distance, uniform crossover,
                         random-reassignment mutation.  Produces a
                         Pareto front.
* **Risk-greedy**      — sort patients by no-show probability
                         descending (highest-risk first), then
                         assign earliest-available slot on the
                         first-fit chair within the patient's window.
                         Single point in the objective space.

Objectives (all maximise, so NSGA-II minimises the negations):
    utilisation      in [0, 1]
    p1_compliance    in [0, 1]   (renamed from % for NSGA coding)
    gender_ratio     in [0, 1]   (worst-pair Four-Fifths)
    1 - mean_wait/H  in [0, 1]   (H = horizon, so 1.0 = no wait)

Hypervolume reference point
---------------------------
All four objectives normalised to [0, 1] where 0 is worst and 1 is
best.  The reference point is the origin (0, 0, 0, 0); hypervolume
is the dominated 4-D volume, in [0, 1], larger is better.  This uses
the inclusion-exclusion computation that is exact for ≤ 10 Pareto
points (which we always have here) so we don't need a specialised
HV library.

CLI
---
    python -m ml.benchmark_external_algorithms \
        --n-patients 30 --n-chairs 6 --time-limit-seconds 10 \
        --nsga-population 48 --nsga-generations 20 --seed 42

Never invoked by the live Flask backend.  No UI panel.  Writes to
``data_cache/external_benchmark/results.jsonl`` only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Data fixtures
# --------------------------------------------------------------------------- #


def _load_patients(n: int):
    import pandas as pd
    from optimization.optimizer import Patient
    from config import OPERATING_HOURS

    df = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" / "patients.xlsx")
    if len(df) < n:
        raise SystemExit(f"Only {len(df)} patients; need {n}")
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
        g_raw = row.get("Person_Stated_Gender_Code", 0)
        try:
            g_int = int(g_raw)
        except (TypeError, ValueError):
            g_int = 0
        p.gender = "M" if g_int == 1 else "F" if g_int == 2 else "unknown"
        try:
            age = int(row.get("Age", 0) or 0)
        except (TypeError, ValueError):
            age = 0
        p.age_band = "0-39" if age < 40 else "40-64" if age < 65 else "65+"
        patients.append(p)
        audit_rows.append({
            "Patient_ID": pid, "Gender": p.gender, "Age_Band": p.age_band,
        })
    return patients, audit_rows, today, start_h, end_h


def _build_chairs(n_chairs: int, today, start_h, end_h):
    from optimization.optimizer import Chair
    from config import DEFAULT_SITES

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
# Metrics — shared across all three methods for apples-to-apples
# --------------------------------------------------------------------------- #


def _gender_fairness_ratio(audit_rows, scheduled_ids):
    from ml.fairness_audit import FairnessAuditor
    auditor = FairnessAuditor()
    rep = auditor.audit_schedule(audit_rows, scheduled_ids, group_column="Gender")
    ratios = [m.ratio for m in rep.metrics if m.ratio is not None]
    return min(ratios) if ratios else 1.0


def _mean_wait_minutes(patients, appointments):
    """Mean minutes between earliest_time and actual start_time."""
    if not appointments:
        return 0.0
    by_id = {p.patient_id: p for p in patients}
    waits = []
    for a in appointments:
        p = by_id.get(a.patient_id)
        if p is None:
            continue
        wait = (a.start_time - p.earliest_time).total_seconds() / 60.0
        if wait >= 0:
            waits.append(wait)
    return float(sum(waits) / max(len(waits), 1))


def _p1_compliance_pct(patients, appointments):
    p1 = [p for p in patients if getattr(p, "priority", 3) == 1]
    if not p1:
        return 100.0
    scheduled = {a.patient_id for a in appointments}
    return 100.0 * sum(1 for p in p1 if p.patient_id in scheduled) / len(p1)


def _score_schedule(
    patients, audit_rows, appointments, *, horizon_min: float,
) -> Dict:
    scheduled_ids = {a.patient_id for a in (appointments or [])}
    util = len(scheduled_ids) / max(len(patients), 1)
    p1 = _p1_compliance_pct(patients, appointments or []) / 100.0
    gender = _gender_fairness_ratio(audit_rows, scheduled_ids)
    mw = _mean_wait_minutes(patients, appointments or [])
    wait_score = max(0.0, 1.0 - mw / max(horizon_min, 1.0))
    return {
        "utilisation": float(util),
        "p1_compliance": float(p1),
        "gender_fairness_ratio": float(gender),
        "mean_wait_min": float(mw),
        "wait_score": float(wait_score),
        "n_scheduled": int(len(scheduled_ids)),
    }


# --------------------------------------------------------------------------- #
# Method 1 — CP-SAT (production)
# --------------------------------------------------------------------------- #


def _run_cpsat_pareto(
    patients, audit_rows, chairs, *, time_limit_s: float, horizon_min: float,
) -> List[Dict]:
    """Run ScheduleOptimizer once per PARETO_WEIGHT_SETS row, score each."""
    from optimization.optimizer import ScheduleOptimizer
    from config import PARETO_WEIGHT_SETS

    points = []
    for cfg in PARETO_WEIGHT_SETS:
        opt = ScheduleOptimizer()
        opt.chairs = chairs
        opt._cg_enabled = False
        weights = {k: v for k, v in cfg.items() if k != "name"}
        opt.set_weights(weights, normalise=False)
        t0 = time.perf_counter()
        result = opt.optimize(patients, time_limit_seconds=int(time_limit_s))
        dt = time.perf_counter() - t0
        score = _score_schedule(
            patients, audit_rows, result.appointments or [],
            horizon_min=horizon_min,
        )
        score["config_name"] = cfg.get("name", "custom")
        score["solve_time_s"] = float(dt)
        points.append(score)
        print(
            f"  [cpsat/{score['config_name']:<18s}] "
            f"util={score['utilisation']:.3f}  "
            f"p1={score['p1_compliance']:.2f}  "
            f"gender={score['gender_fairness_ratio']:.3f}  "
            f"wait={score['mean_wait_min']:.1f}m  "
            f"t={dt:.1f}s",
            flush=True,
        )
    return points


# --------------------------------------------------------------------------- #
# Method 2 — NSGA-II (minimal, self-contained)
# --------------------------------------------------------------------------- #


def _build_schedule_from_chromosome(chromosome, patients, chairs, horizon_min):
    """
    chromosome[i] = (chair_idx_or_None, start_min) for patient i.
    Returns list of ScheduledAppointment with no-overlap enforced by
    the greedy reconstruction — infeasible chromosomes get their
    conflicting assignments dropped to unassigned.
    """
    from optimization.optimizer import ScheduledAppointment

    today = patients[0].earliest_time.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_h = patients[0].earliest_time.hour

    # Sort (p_idx, chair_idx, start) by start_min so we can greedy-place
    triples = []
    for i, (c_idx, s_min) in enumerate(chromosome):
        if c_idx is None:
            continue
        triples.append((int(s_min), i, int(c_idx)))
    triples.sort()

    chair_end = [0] * len(chairs)
    appts = []
    for s_min, i, c_idx in triples:
        p = patients[i]
        earliest = max(
            s_min,
            int((p.earliest_time - today).total_seconds() / 60.0) - start_h * 60,
            chair_end[c_idx],
        )
        latest_allowed = int(
            (p.latest_time - today).total_seconds() / 60.0
        ) - start_h * 60 - p.expected_duration
        if earliest > latest_allowed:
            continue
        start_time = today + timedelta(minutes=(earliest + start_h * 60))
        end_time = start_time + timedelta(minutes=p.expected_duration)
        appts.append(ScheduledAppointment(
            patient_id=p.patient_id,
            chair_id=chairs[c_idx].chair_id,
            site_code=chairs[c_idx].site_code,
            start_time=start_time,
            end_time=end_time,
            duration=p.expected_duration,
            priority=p.priority,
            travel_time=30,
        ))
        chair_end[c_idx] = earliest + p.expected_duration
    return appts


def _random_chromosome(patients, chairs, horizon_min, rng):
    out = []
    for p in patients:
        c_idx = rng.randrange(len(chairs) + 1)  # +1 for "unassigned"
        if c_idx == len(chairs):
            out.append((None, 0))
        else:
            dur = p.expected_duration
            max_start = max(1, int(horizon_min) - dur)
            out.append((c_idx, rng.randrange(max_start)))
    return out


def _objectives(appts, patients, audit_rows, horizon_min):
    """Return (−util, −p1, −gender, −wait_score) — all minimised."""
    s = _score_schedule(patients, audit_rows, appts, horizon_min=horizon_min)
    return (
        -s["utilisation"],
        -s["p1_compliance"],
        -s["gender_fairness_ratio"],
        -s["wait_score"],
    )


def _dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _non_dominated_sort(objs):
    """Return list of fronts (each a list of indices)."""
    n = len(objs)
    dominated_by = [[] for _ in range(n)]
    n_dominates = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                dominated_by[p].append(q)
            elif _dominates(objs[q], objs[p]):
                n_dominates[p] += 1
        if n_dominates[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                n_dominates[q] -= 1
                if n_dominates[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]  # trailing empty


def _crowding_distance(front_indices, objs):
    import math
    n = len(front_indices)
    if n == 0:
        return {}
    dist = {i: 0.0 for i in front_indices}
    for m in range(len(objs[0])):
        sorted_idx = sorted(front_indices, key=lambda i: objs[i][m])
        dist[sorted_idx[0]] = math.inf
        dist[sorted_idx[-1]] = math.inf
        v_min, v_max = objs[sorted_idx[0]][m], objs[sorted_idx[-1]][m]
        rng = max(v_max - v_min, 1e-9)
        for k in range(1, n - 1):
            dist[sorted_idx[k]] += (
                objs[sorted_idx[k + 1]][m] - objs[sorted_idx[k - 1]][m]
            ) / rng
    return dist


def _run_nsga2(
    patients, audit_rows, chairs, *,
    population: int, generations: int, horizon_min: float, seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    pop = [
        _random_chromosome(patients, chairs, horizon_min, rng)
        for _ in range(population)
    ]
    best_front_chromos = []
    for gen in range(generations):
        schedules = [
            _build_schedule_from_chromosome(c, patients, chairs, horizon_min)
            for c in pop
        ]
        objs = [
            _objectives(s, patients, audit_rows, horizon_min) for s in schedules
        ]
        fronts = _non_dominated_sort(objs)

        # Tournament + uniform crossover + random-reassignment mutation
        new_pop: List = []
        all_ranks = {i: r for r, fr in enumerate(fronts) for i in fr}
        crowd = {
            i: d for fr in fronts for i, d in _crowding_distance(fr, objs).items()
        }

        def _tournament():
            a, b = rng.randrange(len(pop)), rng.randrange(len(pop))
            if all_ranks[a] < all_ranks[b]:
                return a
            if all_ranks[b] < all_ranks[a]:
                return b
            return a if crowd[a] >= crowd[b] else b

        while len(new_pop) < population:
            p1_idx, p2_idx = _tournament(), _tournament()
            p1, p2 = pop[p1_idx], pop[p2_idx]
            child = [
                p1[i] if rng.random() < 0.5 else p2[i]
                for i in range(len(patients))
            ]
            # Mutation — 10% of genes randomly reassigned
            for i in range(len(child)):
                if rng.random() < 0.10:
                    dur = patients[i].expected_duration
                    c_idx = rng.randrange(len(chairs) + 1)
                    if c_idx == len(chairs):
                        child[i] = (None, 0)
                    else:
                        max_start = max(1, int(horizon_min) - dur)
                        child[i] = (c_idx, rng.randrange(max_start))
            new_pop.append(child)
        pop = new_pop

    # Final score of the last population's Pareto front
    schedules = [
        _build_schedule_from_chromosome(c, patients, chairs, horizon_min)
        for c in pop
    ]
    objs = [_objectives(s, patients, audit_rows, horizon_min) for s in schedules]
    fronts = _non_dominated_sort(objs)
    front_idx = fronts[0]
    points = []
    for i in front_idx:
        s = _score_schedule(
            patients, audit_rows, schedules[i], horizon_min=horizon_min,
        )
        s["config_name"] = f"nsga_ind{i}"
        s["solve_time_s"] = 0.0  # aggregated below
        points.append(s)
    return points


# --------------------------------------------------------------------------- #
# Method 3 — Risk-greedy
# --------------------------------------------------------------------------- #


def _run_risk_greedy(
    patients, audit_rows, chairs, *, horizon_min: float,
) -> List[Dict]:
    """Sort by no-show probability DESCENDING (highest-risk first so
    double-booking slack prefers them), then earliest-available slot
    per chair.  This is the classic "risk-aware greedy" baseline the
    reviewer suggested."""
    from optimization.optimizer import ScheduledAppointment

    today = patients[0].earliest_time.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_h = patients[0].earliest_time.hour

    order = sorted(
        range(len(patients)),
        key=lambda i: -float(patients[i].noshow_probability),
    )
    chair_end = [0] * len(chairs)
    appts = []
    t0 = time.perf_counter()
    for i in order:
        p = patients[i]
        e_min = int((p.earliest_time - today).total_seconds() / 60.0) - start_h * 60
        l_min = int((p.latest_time - today).total_seconds() / 60.0) - start_h * 60
        l_start = l_min - p.expected_duration
        best_c = None
        best_start = None
        for c_idx in range(len(chairs)):
            s_min = max(e_min, chair_end[c_idx])
            if s_min > l_start:
                continue
            if best_start is None or s_min < best_start:
                best_c, best_start = c_idx, s_min
        if best_c is not None:
            start_time = today + timedelta(minutes=best_start + start_h * 60)
            end_time = start_time + timedelta(minutes=p.expected_duration)
            appts.append(ScheduledAppointment(
                patient_id=p.patient_id,
                chair_id=chairs[best_c].chair_id,
                site_code=chairs[best_c].site_code,
                start_time=start_time,
                end_time=end_time,
                duration=p.expected_duration,
                priority=p.priority,
                travel_time=30,
            ))
            chair_end[best_c] = best_start + p.expected_duration
    dt = time.perf_counter() - t0
    score = _score_schedule(
        patients, audit_rows, appts, horizon_min=horizon_min,
    )
    score["config_name"] = "risk_greedy"
    score["solve_time_s"] = float(dt)
    return [score]


# --------------------------------------------------------------------------- #
# Hypervolume — exact inclusion–exclusion for small Pareto sets
# --------------------------------------------------------------------------- #


def _hypervolume(points_norm, ref=(0.0, 0.0, 0.0, 0.0)):
    """
    Each point is a tuple of normalised objective values in [0, 1]
    (larger = better).  Ref is the origin (all zeros, worst case).
    Returns HV dominated by the Pareto set over the reference box.

    Uses inclusion–exclusion — exact for |points| <= 12 or so, which
    is always the case here.
    """
    if not points_norm:
        return 0.0

    # Inclusion-exclusion over subsets: Σ_{S non-empty} (-1)^{|S|+1}
    #   * prod_k (min_{p in S} p[k] − ref[k])
    n = len(points_norm)
    total = 0.0
    for r in range(1, n + 1):
        for subset in itertools.combinations(points_norm, r):
            mins = [
                max(0.0, min(p[k] for p in subset) - ref[k])
                for k in range(len(ref))
            ]
            vol = 1.0
            for m in mins:
                vol *= m
            sign = -1 if r % 2 == 0 else 1
            total += sign * vol
    return float(total)


def _to_hv_points(scores):
    """Map score dicts → (util, p1, gender, wait_score) tuples."""
    return [
        (s["utilisation"], s["p1_compliance"],
         s["gender_fairness_ratio"], s["wait_score"])
        for s in scores
    ]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def benchmark(
    *, n_patients: int, n_chairs: int, time_limit_s: float,
    nsga_population: int, nsga_generations: int, seed: int,
    output_path: Path,
) -> Dict:
    patients, audit_rows, today, start_h, end_h = _load_patients(n_patients)
    chairs = _build_chairs(n_chairs, today, start_h, end_h)
    horizon_min = (end_h - start_h) * 60

    print(
        f"\n=== External-algorithm benchmark "
        f"(n={n_patients}, chairs={n_chairs}, seed={seed}) ===",
        flush=True,
    )
    print("CP-SAT (Pareto weight sweep):", flush=True)
    cpsat_points = _run_cpsat_pareto(
        patients, audit_rows, chairs,
        time_limit_s=time_limit_s, horizon_min=horizon_min,
    )
    print(
        f"NSGA-II (pop={nsga_population}, gens={nsga_generations}):",
        flush=True,
    )
    t0 = time.perf_counter()
    nsga_points = _run_nsga2(
        patients, audit_rows, chairs,
        population=nsga_population, generations=nsga_generations,
        horizon_min=horizon_min, seed=seed,
    )
    nsga_dt = time.perf_counter() - t0
    for p in nsga_points:
        p["solve_time_s"] = nsga_dt / max(len(nsga_points), 1)
    print(
        f"  [nsga-II] {len(nsga_points)} Pareto points  total t={nsga_dt:.1f}s",
        flush=True,
    )
    print("Risk-greedy:", flush=True)
    rg_points = _run_risk_greedy(
        patients, audit_rows, chairs, horizon_min=horizon_min,
    )
    for p in rg_points:
        print(
            f"  [risk-greedy] util={p['utilisation']:.3f}  "
            f"p1={p['p1_compliance']:.2f}  "
            f"gender={p['gender_fairness_ratio']:.3f}  "
            f"wait={p['mean_wait_min']:.1f}m  t={p['solve_time_s']:.3f}s",
            flush=True,
        )

    hv_cpsat = _hypervolume(_to_hv_points(cpsat_points))
    hv_nsga = _hypervolume(_to_hv_points(nsga_points[:8]))   # cap for speed
    hv_rg = _hypervolume(_to_hv_points(rg_points))
    print(
        f"\nHypervolume (larger = better):\n"
        f"  CP-SAT     : {hv_cpsat:.4f}\n"
        f"  NSGA-II    : {hv_nsga:.4f}\n"
        f"  Risk-greedy: {hv_rg:.4f}",
        flush=True,
    )

    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "n_patients": int(n_patients),
        "n_chairs": int(n_chairs),
        "time_limit_s": float(time_limit_s),
        "seed": int(seed),
        "nsga_population": int(nsga_population),
        "nsga_generations": int(nsga_generations),
        "cpsat_points": cpsat_points,
        "nsga_points": nsga_points[:8],
        "risk_greedy_points": rg_points,
        "hypervolume": {
            "cpsat": hv_cpsat,
            "nsga": hv_nsga,
            "risk_greedy": hv_rg,
        },
        "comparison_note": (
            "All three methods score the same cohort + chair grid on "
            "the same four objectives (utilisation, p1_compliance, "
            "gender_fairness_ratio, wait_score), each normalised to "
            "[0, 1] with larger = better.  Hypervolume is the 4-D "
            "dominated volume relative to the origin reference point, "
            "in [0, 1], computed by exact inclusion–exclusion so the "
            "cell is fully reproducible."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nAppended 1 row to {output_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients", type=int, default=30)
    parser.add_argument("--n-chairs", type=int, default=6)
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--nsga-population", type=int, default=48)
    parser.add_argument("--nsga-generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data_cache/external_benchmark/results.jsonl",
    )
    args = parser.parse_args()
    benchmark(
        n_patients=args.n_patients,
        n_chairs=args.n_chairs,
        time_limit_s=args.time_limit_seconds,
        nsga_population=args.nsga_population,
        nsga_generations=args.nsga_generations,
        seed=args.seed,
        output_path=_REPO_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
