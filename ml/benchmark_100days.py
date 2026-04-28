"""
100-day in-sample benchmark — CP-SAT vs Patrick rule-based baseline
====================================================================

Reproduces dissertation Table 5.3.  For each of 100 random seeds,
samples 40 patients from ``patients.xlsx``, builds 45 chairs across
all five Velindre sites, then runs:

  * CP-SAT (production ScheduleOptimizer with default 'balanced' weights)
  * Patrick et al. (2008) priority-first-fit baseline (sort P1->P4,
    earliest-available slot within window)

Records per-day metrics: chair-time utilisation, mean waiting time,
P1 compliance, mean travel distance (scheduled), solve time.
Aggregates mean +/- SD across 100 days.

Output: ``data_cache/100days/results.jsonl``
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
_OUT_DIR = _REPO_ROOT / "data_cache" / "100days"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY_HOURS = 9
HORIZON_MIN = DAY_HOURS * 60
N_CHAIRS = 45
DEFAULT_N_PATIENTS_PER_DAY = 150  # Velindre-scale contested-capacity regime


_PRI_STR2INT = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def _load_patient_pool():
    from config import OPERATING_HOURS
    # Use historical_appointments which has Planned_Duration + Priority + Postcode
    df = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" / "historical_appointments.xlsx")
    # Need postcode + gender from patients.xlsx — join on Patient_ID/Local_Patient_Identifier
    pat = pd.read_excel(_REPO_ROOT / "datasets" / "sample_data" / "patients.xlsx")
    pat_map = pat.set_index("Local_Patient_Identifier")[
        ["Patient_Postcode", "Person_Stated_Gender_Code"]
    ].to_dict("index")
    df["__postcode"] = df["Patient_ID"].map(lambda p: pat_map.get(p, {}).get(
        "Patient_Postcode", "CF14"))
    df["__gender_code"] = df["Patient_ID"].map(lambda p: pat_map.get(p, {}).get(
        "Person_Stated_Gender_Code", 0))
    return df, OPERATING_HOURS


def _build_day(df_pool: pd.DataFrame, seed: int):
    """Sample N patients with deterministic seeded jitter."""
    from optimization.optimizer import Patient
    rng = random.Random(seed)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    sub = df_pool.sample(n=_N_PATIENTS_PER_DAY, random_state=seed,
                          replace=_N_PATIENTS_PER_DAY > len(df_pool))  # noqa: E501

    patients = []
    audit_rows = []
    for idx, (_, row) in enumerate(sub.iterrows()):
        pid_base = str(row.get("Patient_ID", f"P{rng.randint(0, 99999):05d}"))
        # Parse priority "P1"/"P2"/"P3"/"P4" -> int
        pri_str = str(row.get("Priority", "P3"))
        pri = _PRI_STR2INT.get(pri_str, 3)
        try:
            dur = int(row.get("Planned_Duration", 90) or 90)
        except (TypeError, ValueError):
            dur = 90
        # Use unique patient_id per (seed, idx) so duplicates from sampling don't collide
        p = Patient(
            patient_id=f"{pid_base}_d{seed}_{idx}",
            priority=pri,
            protocol=str(row.get("Regimen_Code", "R-CHOP")),
            expected_duration=dur,
            postcode=str(row.get("__postcode", "CF14")),
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=8 + DAY_HOURS),
            noshow_probability=float(row.get("Patient_NoShow_Rate",
                                              row.get("Historical_NoShow_Rate", 0.13))
                                       or 0.13),
        )
        gd = row.get("__gender_code", 0)
        try:
            gi = int(gd)
        except (TypeError, ValueError):
            gi = 0
        p.gender = "M" if gi == 1 else "F" if gi == 2 else "unknown"
        patients.append(p)
        audit_rows.append({"Patient_ID": p.patient_id, "Gender": p.gender})
    return patients, audit_rows, today


def _build_chairs(today):
    from optimization.optimizer import Chair
    from config import DEFAULT_SITES
    chairs = []
    for site in DEFAULT_SITES:
        for i in range(site["chairs"]):
            chairs.append(Chair(
                chair_id=f"{site['code']}-C{i+1:02d}",
                site_code=site["code"],
                is_recliner=i < site.get("recliners", 0),
                available_from=today.replace(hour=8),
                available_until=today.replace(hour=8 + DAY_HOURS),
            ))
            if len(chairs) >= N_CHAIRS:
                return chairs
    return chairs


def _haversine_km(p_pc: str, c_lat: float, c_lon: float) -> float:
    from config import POSTCODE_COORDINATES
    from math import radians, sin, cos, sqrt, atan2
    pc_root = (p_pc or "").upper().split()[0][:4]
    coord = POSTCODE_COORDINATES.get(pc_root) or POSTCODE_COORDINATES.get(pc_root[:3])
    if not coord:
        return 10.0
    lat1, lon1, lat2, lon2 = map(radians, [coord["lat"], coord["lon"], c_lat, c_lon])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 6371.0 * 2 * atan2(sqrt(a), sqrt(1-a))


def _metrics(patients, chairs, appointments, *, solve_time_s: float) -> Dict:
    from config import DEFAULT_SITES
    n_scheduled = len(appointments)
    chair_minutes_used = sum(
        (a.end_time - a.start_time).total_seconds() / 60.0 for a in appointments
    )
    chair_minutes_available = N_CHAIRS * HORIZON_MIN
    util = chair_minutes_used / chair_minutes_available if chair_minutes_available else 0.0

    by_id = {p.patient_id: p for p in patients}
    waits = []
    travels = []
    for a in appointments:
        p = by_id.get(a.patient_id)
        if p is None:
            continue
        wait = (a.start_time - p.earliest_time).total_seconds() / 60.0
        if wait >= 0:
            waits.append(wait)
        chair_obj = next((c for c in chairs if c.chair_id == a.chair_id), None)
        if chair_obj:
            site = next((s for s in DEFAULT_SITES if s["code"] == chair_obj.site_code), None)
            if site:
                travels.append(_haversine_km(p.postcode, site["lat"], site["lon"]))

    p1 = [p for p in patients if getattr(p, "priority", 3) == 1]
    p1_sched = {a.patient_id for a in appointments}
    p1_compl = (sum(1 for p in p1 if p.patient_id in p1_sched) / len(p1) * 100.0
                if p1 else 100.0)

    return {
        "n_scheduled": n_scheduled,
        "chair_util_pct": util * 100.0,
        "mean_wait_min": float(np.mean(waits)) if waits else 0.0,
        "p1_compliance_pct": p1_compl,
        "mean_travel_km": float(np.mean(travels)) if travels else 0.0,
        "solve_time_s": solve_time_s,
    }


def _run_cpsat(patients, audit_rows, chairs, *, time_limit_s: float):
    from optimization.optimizer import ScheduleOptimizer
    opt = ScheduleOptimizer()
    opt.chairs = chairs
    opt.set_components(column_generation=False)
    opt.set_weights({"priority": 0.30, "utilization": 0.25, "noshow_risk": 0.15,
                     "waiting_time": 0.15, "robustness": 0.10, "travel": 0.05},
                    normalise=False)
    t0 = time.perf_counter()
    result = opt.optimize(patients, time_limit_seconds=int(time_limit_s))
    return result.appointments or [], time.perf_counter() - t0


def _run_patrick(patients, chairs, *, time_limit_s: float):
    """Priority-first-fit: sort P1->P4 then earliest-time, assign to earliest chair slot."""
    from optimization.optimizer import ScheduledAppointment
    t0 = time.perf_counter()
    sorted_patients = sorted(patients, key=lambda p: (getattr(p, "priority", 3),
                                                       p.earliest_time))
    chair_next_free = {c.chair_id: c.available_from for c in chairs}
    appts = []
    for p in sorted_patients:
        best_chair, best_start = None, None
        for c in chairs:
            free = max(chair_next_free[c.chair_id], p.earliest_time)
            end = free + timedelta(minutes=p.expected_duration)
            if end > p.latest_time or end > c.available_until:
                continue
            if best_start is None or free < best_start:
                best_chair, best_start = c, free
        if best_chair is None:
            continue
        end = best_start + timedelta(minutes=p.expected_duration)
        appts.append(ScheduledAppointment(patient_id=p.patient_id,
                                           chair_id=best_chair.chair_id,
                                           site_code=best_chair.site_code,
                                           start_time=best_start,
                                           end_time=end,
                                           duration=p.expected_duration,
                                           priority=getattr(p, "priority", 3),
                                           travel_time=0))
        chair_next_free[best_chair.chair_id] = end
    return appts, time.perf_counter() - t0


def main(n_days: int = 100, time_limit_s: float = 5.0,
         n_patients_per_day: int = DEFAULT_N_PATIENTS_PER_DAY):
    global _N_PATIENTS_PER_DAY
    _N_PATIENTS_PER_DAY = n_patients_per_day
    df_pool, _ = _load_patient_pool()
    print(f"Pool: {len(df_pool)} patients; running {n_days} days at "
          f"{_N_PATIENTS_PER_DAY}p/{N_CHAIRS}ch with {time_limit_s}s solve limit")

    cpsat_runs, patrick_runs = [], []
    t_start = time.perf_counter()
    for seed in range(n_days):
        patients, audit_rows, today = _build_day(df_pool, seed)
        chairs = _build_chairs(today)

        appts_p, dt_p = _run_patrick(patients, chairs, time_limit_s=time_limit_s)
        m_p = _metrics(patients, chairs, appts_p, solve_time_s=dt_p)
        m_p["arm"] = "patrick"
        m_p["seed"] = seed
        patrick_runs.append(m_p)

        appts_c, dt_c = _run_cpsat(patients, audit_rows, chairs, time_limit_s=time_limit_s)
        m_c = _metrics(patients, chairs, appts_c, solve_time_s=dt_c)
        m_c["arm"] = "cpsat"
        m_c["seed"] = seed
        cpsat_runs.append(m_c)

        if (seed + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            est_total = elapsed / (seed + 1) * n_days
            print(f"  seed {seed+1}/{n_days}  elapsed {elapsed:.0f}s  "
                  f"(est total {est_total:.0f}s)  "
                  f"cpsat util={m_c['chair_util_pct']:.1f}%  "
                  f"patrick util={m_p['chair_util_pct']:.1f}%")

    def _mean_sd(runs, key):
        vals = np.array([r[key] for r in runs], dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1))

    summary = {"ts": datetime.utcnow().isoformat(), "n_days": n_days,
               "n_patients_per_day": _N_PATIENTS_PER_DAY,
               "n_chairs": N_CHAIRS,
               "horizon_min": HORIZON_MIN,
               "time_limit_per_solve_s": time_limit_s}
    for arm, runs in [("cpsat", cpsat_runs), ("patrick", patrick_runs)]:
        block = {}
        for k in ("chair_util_pct", "mean_wait_min", "p1_compliance_pct",
                  "mean_travel_km", "solve_time_s", "n_scheduled"):
            m, s = _mean_sd(runs, k)
            block[f"{k}_mean"] = round(m, 3)
            block[f"{k}_sd"]   = round(s, 3)
        summary[arm] = block

    # Per-seed pairs + paired t-tests + bootstrap CIs for util & wait_min.
    # Used downstream by benchmark_table_55.py to populate the §5.6
    # "Statistical Significance" table without re-running the long benchmark.
    from scipy import stats as _sp_stats
    pairs = [{"seed": cr["seed"],
              "util_cpsat":   cr["chair_util_pct"],
              "util_patrick": pr["chair_util_pct"],
              "wait_cpsat":   cr["mean_wait_min"],
              "wait_patrick": pr["mean_wait_min"]}
             for cr, pr in zip(cpsat_runs, patrick_runs)]
    summary["per_seed"] = pairs

    def _paired_test(diffs: list[float], n_boot: int = 1000,
                      seed: int = 42) -> dict:
        d = np.asarray(diffs, dtype=float)
        n = len(d)
        mean_diff = float(d.mean())
        sd_diff   = float(d.std(ddof=1))
        se        = sd_diff / np.sqrt(n)
        t_stat    = mean_diff / se if se > 0 else float("nan")
        # two-sided p
        p_val     = float(2.0 * (1.0 - _sp_stats.t.cdf(abs(t_stat), df=n-1)))
        # 95% percentile bootstrap CI on the mean difference
        rng = np.random.RandomState(seed)
        boot = np.array([d[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        return {"n": n, "mean_diff": round(mean_diff, 4),
                "sd_diff": round(sd_diff, 4),
                "t_stat":   round(t_stat, 4),
                "p_value":  p_val,
                "ci_lo":    round(ci_lo, 4),
                "ci_hi":    round(ci_hi, 4),
                "n_bootstrap": n_boot}

    util_diffs = [p["util_cpsat"] - p["util_patrick"] for p in pairs]
    wait_diffs = [p["wait_cpsat"] - p["wait_patrick"] for p in pairs]
    summary["paired_tests"] = {
        "util_cpsat_minus_patrick_pp": _paired_test(util_diffs),
        "wait_cpsat_minus_patrick_min": _paired_test(wait_diffs),
    }

    out_path = _OUT_DIR / "results.jsonl"
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")

    print("\n===== Summary =====")
    for arm in ("patrick", "cpsat"):
        b = summary[arm]
        print(f"{arm:>8}: util={b['chair_util_pct_mean']:.1f}±{b['chair_util_pct_sd']:.1f}  "
              f"wait={b['mean_wait_min_mean']:.1f}±{b['mean_wait_min_sd']:.1f}min  "
              f"p1={b['p1_compliance_pct_mean']:.1f}±{b['p1_compliance_pct_sd']:.1f}%  "
              f"travel={b['mean_travel_km_mean']:.1f}±{b['mean_travel_km_sd']:.1f}km  "
              f"solve={b['solve_time_s_mean']:.2f}±{b['solve_time_s_sd']:.2f}s")
    print(f"\nWrote {out_path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=100)
    ap.add_argument("--time-limit-s", type=float, default=5.0)
    ap.add_argument("--n-patients", type=int, default=DEFAULT_N_PATIENTS_PER_DAY)
    args = ap.parse_args()
    main(n_days=args.n_days, time_limit_s=args.time_limit_s,
         n_patients_per_day=args.n_patients)
