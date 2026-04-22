"""
Digital Twin for What-If Simulation (Dissertation §3.3)
=======================================================

Maintains a parallel simulated environment that mirrors live
scheduler state and advances virtual time using historical arrival
patterns.  The goal is *safe* evaluation of aggressive policies —
for example, lowering the double-book threshold to 0.05 — without
ever touching the live schedule.

The design follows the §3.3 brief:

    simulator.step()                             # virtual time tick
    evaluate(new_policy, horizon=7_days)          # policy rollout

Key properties
--------------

* **State isolation.**  ``snapshot_live_state`` deep-copies the
  mutable state (appointments, pending patients, chair usage, the
  patient_data_map).  Nothing the twin does can leak back into
  ``app_state``.
* **Empirical arrival model.**  Arrival rates λ(h, d) come from
  ``historical_appointments.xlsx`` — hour-of-day and day-of-week are
  fitted as Poisson rates in events per hour.  Urgent fraction is
  measured from the same file.
* **Production primitive re-use.**  The twin does *not* reimplement
  the scheduler.  It dependency-injects the same ``squeeze_fn`` /
  ``optimize_fn`` / ``noshow_fn`` callables Flask holds, so the
  numbers we report would be identical to what live ops would see
  given the same arrivals.
* **Determinism.**  Every rollout uses an ``rng_seed`` for
  reproducibility; the same (snapshot, policy, seed) triplet must
  produce byte-identical metrics.
* **Persistence.**  Evaluations are written to
  ``data_cache/digital_twin/evaluations/*.json`` so the dissertation
  R script and the status endpoint can read historical runs.
* **Invisible integration.**  No UI panel — Flask calls
  ``twin.evaluate_policy(...)`` as a pre-commit gate when operators
  propose a threshold change, and ``/api/twin/*`` surfaces
  diagnostics.

Brief reference
---------------
* §3.3 of the dissertation specifies horizon=7 days and double-book
  threshold 0.05 as the motivating aggressive-policy example.
* §9b.3 of MATH_LOGIC.md captures the arrival-rate estimator and
  rollout loop.
"""

from __future__ import annotations

import copy
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults straight from the §3.3 brief
# ---------------------------------------------------------------------------
DEFAULT_HORIZON_DAYS: int = 7
DEFAULT_STEP_HOURS: int = 1
DEFAULT_RNG_SEED: int = 42
DEFAULT_DOUBLE_BOOK_THRESHOLD: float = 0.30
AGGRESSIVE_DOUBLE_BOOK_THRESHOLD: float = 0.05
TWIN_DIR: Path = DATA_CACHE_DIR / "digital_twin"
ARRIVAL_MODEL_FILE: Path = TWIN_DIR / "arrival_model.json"
EVALUATIONS_DIR: Path = TWIN_DIR / "evaluations"
EVENT_LOG_FILE: Path = TWIN_DIR / "twin_events.jsonl"


# ---------------------------------------------------------------------------
# Lightweight dataclasses — the twin's public types
# ---------------------------------------------------------------------------


@dataclass
class ArrivalModel:
    """Empirical Poisson arrival model: λ(hour_of_day, day_of_week)."""
    lambda_hd: Dict[str, float] = field(default_factory=dict)  # "{dow}:{hod}" -> events/hour
    urgent_fraction: float = 0.0
    historical_days_observed: int = 0
    total_events: int = 0
    fitted_ts: Optional[str] = None
    source_file: Optional[str] = None

    def rate(self, ts: datetime) -> float:
        """Return λ (events/hour) for a given timestamp."""
        key = f"{ts.weekday()}:{ts.hour}"
        return float(self.lambda_hd.get(key, 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArrivalModel":
        return cls(**d)


@dataclass
class PolicySpec:
    """A scheduling policy the twin is asked to evaluate."""
    name: str = "baseline"
    double_book_threshold: float = DEFAULT_DOUBLE_BOOK_THRESHOLD
    allow_rescheduling: bool = False
    allow_double_booking: bool = True
    optimization_weights: Optional[Dict[str, float]] = None
    slow_path_every_n_steps: int = 24  # call full reopt every N steps; None disables
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicySpec":
        # Strip unknown keys to stay forward-compatible.
        allowed = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**allowed)


@dataclass
class TwinState:
    """A frozen snapshot of the live scheduler state."""
    snapshot_ts: str
    virtual_time: str
    appointments: List[Dict[str, Any]]             # plain dicts — decoupled from optimizer classes
    patients_pending: List[Dict[str, Any]]
    patient_data_map: Dict[str, Dict[str, Any]]
    operating_weights: Dict[str, float]
    chair_capacity: Dict[str, int] = field(default_factory=dict)
    mode: str = "NORMAL"
    notes: str = ""

    def clone(self) -> "TwinState":
        return copy.deepcopy(self)


@dataclass
class TwinStepResult:
    """Per-step outcome."""
    virtual_time: str
    arrivals_sampled: int
    arrivals_accepted: int
    arrivals_rejected: int
    double_bookings_created: int
    reschedules_performed: int
    noshows_realised: int
    appointments_completed: int
    utilization_pct: float
    step_latency_ms: float


@dataclass
class TwinEvaluation:
    """Aggregate result of a full horizon rollout."""
    policy: Dict[str, Any]
    horizon_days: int
    step_hours: int
    num_steps: int
    rng_seed: int
    total_arrivals: int
    total_accepted: int
    total_rejected: int
    total_double_bookings: int
    total_reschedules: int
    total_noshows_realised: int
    accept_rate: float
    noshow_rate_realised: float
    mean_utilization_pct: float
    p50_step_latency_ms: float
    p95_step_latency_ms: float
    policy_score: float
    guardrail_violations: List[str]
    runtime_s: float
    evaluated_ts: str
    snapshot_ts: str
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Arrival-model fitter
# ---------------------------------------------------------------------------


def fit_arrival_model(
    historical_df: "pd.DataFrame",
    urgent_col: Optional[str] = None,
    min_days: int = 7,
) -> ArrivalModel:
    """
    Fit Poisson arrival rates λ(hour_of_day, day_of_week) from
    historical appointments.

    Args:
        historical_df: DataFrame with at minimum a Date or
            Appointment_Date column (datetime-like) OR Date + Appointment_Hour.
        urgent_col: Optional name of a boolean-or-priority column used
            to measure urgent fraction.  If None we try 'is_urgent',
            'Is_Urgent', 'priority'/'Priority'<=1, else 0.
        min_days: Minimum distinct days required to fit;
            below this we fall back to uniform 0.5 events/hour (keeps
            tests green even with 1-day fixtures).

    Returns:
        ArrivalModel with λ_hd, urgent_fraction, metadata.
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required for fit_arrival_model")

    if historical_df is None or len(historical_df) == 0:
        logger.warning("fit_arrival_model: empty historical_df — uniform fallback")
        return _uniform_fallback_model(n_events=0)

    df = historical_df.copy()

    # Normalise the date column
    date_col = None
    for cand in ("Date", "date", "Appointment_Date", "appointment_date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        logger.warning("fit_arrival_model: no Date column — uniform fallback")
        return _uniform_fallback_model(n_events=len(df))

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Derive hour-of-day
    if "Appointment_Hour" in df.columns:
        hod = df["Appointment_Hour"].astype(int)
    else:
        hod = df[date_col].dt.hour
    df["_hod"] = hod.clip(0, 23).astype(int)
    df["_dow"] = df[date_col].dt.weekday.astype(int)
    df["_day"] = df[date_col].dt.date

    n_days = df["_day"].nunique()
    if n_days < min_days:
        logger.warning(
            f"fit_arrival_model: only {n_days} distinct days (< {min_days}) — "
            "still fitting but using smoothed rates"
        )

    n_days_observed = max(int(n_days), 1)
    counts = df.groupby(["_dow", "_hod"]).size()

    # Estimate rate per (dow, hod); since each dow appears roughly
    # n_days / 7 times in the window, divide by that.
    lambda_hd: Dict[str, float] = {}
    for dow in range(7):
        dow_days = max(n_days_observed / 7.0, 1.0 / 7.0)
        for hod in range(24):
            c = float(counts.get((dow, hod), 0))
            lam = c / dow_days   # events per hour on this (dow, hod)
            lambda_hd[f"{dow}:{hod}"] = lam

    # Urgent fraction
    urgent_fraction = 0.0
    if urgent_col is None:
        for cand in ("is_urgent", "Is_Urgent"):
            if cand in df.columns:
                urgent_col = cand
                break
    if urgent_col and urgent_col in df.columns:
        urgent_fraction = float(
            pd.Series(df[urgent_col]).astype(bool).mean()
        )
    elif "priority" in df.columns or "Priority" in df.columns:
        pcol = "priority" if "priority" in df.columns else "Priority"
        try:
            urgent_fraction = float(
                (pd.to_numeric(df[pcol], errors="coerce") <= 1).mean()
            )
        except Exception:
            urgent_fraction = 0.0

    model = ArrivalModel(
        lambda_hd=lambda_hd,
        urgent_fraction=float(max(0.0, min(1.0, urgent_fraction))),
        historical_days_observed=n_days_observed,
        total_events=int(len(df)),
        fitted_ts=datetime.utcnow().isoformat(timespec="seconds"),
        source_file="historical_appointments",
    )
    return model


def _uniform_fallback_model(n_events: int) -> ArrivalModel:
    """Return a flat λ=0.5 arrivals/hour model.  Only used when
    fitting cannot proceed (empty df, no Date column)."""
    lam = {f"{d}:{h}": 0.5 for d in range(7) for h in range(24)}
    return ArrivalModel(
        lambda_hd=lam,
        urgent_fraction=0.10,
        historical_days_observed=0,
        total_events=int(n_events),
        fitted_ts=datetime.utcnow().isoformat(timespec="seconds"),
        source_file="fallback",
    )


# ---------------------------------------------------------------------------
# Policy guardrails  (safety: refuse obviously bad policies)
# ---------------------------------------------------------------------------


GUARDRAIL_MAX_DOUBLE_BOOK_RATE = 0.25   # > 25% of accepted = violation
GUARDRAIL_MIN_ACCEPT_RATE = 0.50        # < 50% accepted = violation
GUARDRAIL_MAX_NOSHOW_RATE = 0.35        # > 35% realised no-shows = violation


# ---------------------------------------------------------------------------
# DigitalTwin coordinator
# ---------------------------------------------------------------------------


class DigitalTwin:
    """
    Parallel simulated environment mirroring live scheduler state.

    Parameters
    ----------
    squeeze_fn:
        Callable ``squeeze_fn(patient_dict, state, policy) -> dict``
        with keys ``success``, ``strategy`` ('gap'|'double_booking'|
        'reschedule'|'rejected'), ``chair_id``, ``start_time``.  The
        Flask wiring passes a wrapper around
        ``SqueezeInHandler.squeeze_in_with_noshow``.  If None, a
        deterministic built-in fallback (gap-first / double-book-if-
        threshold-allows) is used — this keeps unit tests hermetic
        and lets the twin run even when OR-Tools isn't installed.
    optimize_fn:
        Callable ``optimize_fn(state, policy) -> dict`` that runs the
        CP-SAT optimiser and returns updated appointments.  Optional;
        if None, slow-path invocations are no-ops.
    noshow_fn:
        Callable ``noshow_fn(patient_dict, appointment_dict) -> float``
        returning no-show probability in [0, 1].  If None, uses
        per-patient ``no_show_rate`` from patient_data_map or 0.15.
    """

    def __init__(
        self,
        *,
        squeeze_fn: Optional[Callable[[Dict, "TwinState", PolicySpec], Dict]] = None,
        optimize_fn: Optional[Callable[["TwinState", PolicySpec], Dict]] = None,
        noshow_fn: Optional[Callable[[Dict, Dict], float]] = None,
        storage_dir: Path = TWIN_DIR,
    ):
        self.squeeze_fn = squeeze_fn or _fallback_squeeze_fn
        self.optimize_fn = optimize_fn
        self.noshow_fn = noshow_fn
        self.storage_dir = Path(storage_dir)
        self.evaluations_dir = self.storage_dir / "evaluations"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        self.arrival_model_path = self.storage_dir / "arrival_model.json"

        self._lock = threading.Lock()
        self.arrival_model: Optional[ArrivalModel] = self._load_arrival_model()
        self.last_snapshot: Optional[TwinState] = None
        self.last_snapshot_ts: Optional[datetime] = None
        self.last_evaluation: Optional[TwinEvaluation] = None
        self.total_snapshots: int = 0
        self.total_evaluations: int = 0
        self.total_steps: int = 0

    # ----------------------------------------------------------------- #
    # Arrival model persistence
    # ----------------------------------------------------------------- #

    def _load_arrival_model(self) -> Optional[ArrivalModel]:
        if not self.arrival_model_path.exists():
            return None
        try:
            with open(self.arrival_model_path) as f:
                return ArrivalModel.from_dict(json.load(f))
        except Exception as exc:  # pragma: no cover — corrupt JSON
            logger.warning(f"Failed to load arrival model: {exc}")
            return None

    def fit_and_save_arrival_model(self, historical_df) -> ArrivalModel:
        model = fit_arrival_model(historical_df)
        with open(self.arrival_model_path, "w") as f:
            json.dump(model.to_dict(), f, indent=2)
        self.arrival_model = model
        logger.info(
            f"Arrival model fitted: {model.total_events} events over "
            f"{model.historical_days_observed} days, urgent_fraction="
            f"{model.urgent_fraction:.3f}"
        )
        return model

    # ----------------------------------------------------------------- #
    # Snapshot
    # ----------------------------------------------------------------- #

    def snapshot_live_state(
        self,
        app_state: Dict[str, Any],
        operating_weights: Optional[Dict[str, float]] = None,
        note: str = "",
    ) -> TwinState:
        """Take a deep-copy snapshot of the live Flask ``app_state``."""
        now = datetime.utcnow()
        apps = _serialise_appointments(app_state.get("appointments", []))
        patients = _serialise_patients(app_state.get("patients", []))
        pdata = copy.deepcopy(app_state.get("patient_data_map", {}) or {})
        weights = dict(operating_weights or {})
        chair_capacity = _compute_chair_capacity(app_state)
        mode = str(app_state.get("mode", "NORMAL"))

        state = TwinState(
            snapshot_ts=now.isoformat(timespec="seconds"),
            virtual_time=now.isoformat(timespec="seconds"),
            appointments=apps,
            patients_pending=patients,
            patient_data_map=pdata,
            operating_weights=weights,
            chair_capacity=chair_capacity,
            mode=mode,
            notes=note,
        )
        with self._lock:
            self.last_snapshot = state
            self.last_snapshot_ts = now
            self.total_snapshots += 1
        _append_event(EVENT_LOG_FILE, {
            "event": "snapshot",
            "ts": now.isoformat(timespec="seconds"),
            "appointments": len(apps),
            "patients": len(patients),
            "note": note,
        })
        return state

    def reset(self) -> TwinState:
        """Return a fresh clone of the last snapshot (useful between runs)."""
        if self.last_snapshot is None:
            raise RuntimeError("No snapshot exists — call snapshot_live_state first")
        return self.last_snapshot.clone()

    # ----------------------------------------------------------------- #
    # Rollout primitives
    # ----------------------------------------------------------------- #

    def step(
        self,
        state: TwinState,
        policy: PolicySpec,
        rng: random.Random,
        step_hours: int = DEFAULT_STEP_HOURS,
    ) -> Tuple[TwinState, TwinStepResult]:
        """
        Advance the twin by ``step_hours`` hours.

        1. Sample arrivals using fitted Poisson rate.
        2. For each arrival, ask the policy to accept/reject via
           ``squeeze_fn``.
        3. For scheduled appointments whose start_time falls in this
           step, realise no-shows using Bernoulli(p_noshow).
        4. Update utilisation and return step metrics.

        The state object is mutated in-place (on its own deep copy).
        """
        start = time.perf_counter()
        t0 = datetime.fromisoformat(state.virtual_time)
        t1 = t0 + timedelta(hours=step_hours)

        # --- 1. arrivals ------------------------------------------------
        arrivals = self._sample_arrivals(t0, t1, rng)

        accepted = 0
        rejected = 0
        double_bookings = 0
        reschedules = 0

        for arrival in arrivals:
            outcome = self.squeeze_fn(arrival, state, policy)
            if outcome.get("success"):
                accepted += 1
                strategy = outcome.get("strategy", "gap")
                if strategy == "double_booking":
                    double_bookings += 1
                elif strategy == "reschedule":
                    reschedules += 1
                # Append scheduled appointment to the twin state
                if outcome.get("appointment"):
                    state.appointments.append(outcome["appointment"])
            else:
                rejected += 1

        # --- 2. realise no-shows + completions for appointments in [t0, t1) --
        noshows = 0
        completed = 0
        remaining = []
        for appt in state.appointments:
            astart = _parse_appt_time(appt.get("start_time"), t0)
            if astart is None or not (t0 <= astart < t1):
                remaining.append(appt)
                continue
            p_ns = self._noshow_prob(appt, state)
            if rng.random() < p_ns:
                noshows += 1
            else:
                completed += 1
            # Appointment is consumed in either case (visited or no-show)

        state.appointments = remaining

        # --- 3. utilisation ------------------------------------------------
        # Proxy: consumed_slots / available_slots where available slots =
        # chairs × step_hours and consumed_slots = completed + noshows + new
        # accepts still in the queue.  This mirrors "what share of the chair
        # capacity was in use this hour" better than counting leftovers.
        total_cap = max(sum(state.chair_capacity.values()), 1)
        consumed = completed + noshows + accepted
        util = 100.0 * (consumed / float(total_cap * max(step_hours, 1)))
        util = float(max(0.0, min(100.0, util)))

        # --- 4. advance clock + bump counters ----------------------------
        state.virtual_time = t1.isoformat(timespec="seconds")
        latency_ms = (time.perf_counter() - start) * 1000.0
        with self._lock:
            self.total_steps += 1

        return state, TwinStepResult(
            virtual_time=state.virtual_time,
            arrivals_sampled=len(arrivals),
            arrivals_accepted=accepted,
            arrivals_rejected=rejected,
            double_bookings_created=double_bookings,
            reschedules_performed=reschedules,
            noshows_realised=noshows,
            appointments_completed=completed,
            utilization_pct=util,
            step_latency_ms=latency_ms,
        )

    def _sample_arrivals(
        self, t0: datetime, t1: datetime, rng: random.Random
    ) -> List[Dict[str, Any]]:
        """Poisson-sample arrivals for [t0, t1)."""
        if self.arrival_model is None:
            return []
        lam = self.arrival_model.rate(t0)
        hours = max((t1 - t0).total_seconds() / 3600.0, 0.0)
        expected = lam * hours
        n = _poisson_sample(expected, rng)
        arrivals = []
        for i in range(n):
            offset_s = rng.random() * hours * 3600.0
            arr_ts = t0 + timedelta(seconds=offset_s)
            is_urgent = rng.random() < self.arrival_model.urgent_fraction
            pid = f"TWINPAT_{int(arr_ts.timestamp()) % 100000:05d}_{i}"
            arrivals.append({
                "patient_id": pid,
                "arrival_ts": arr_ts.isoformat(timespec="seconds"),
                "expected_duration": 60,
                "priority": 1 if is_urgent else 3,
                "is_urgent": is_urgent,
                "earliest_time": arr_ts.isoformat(timespec="seconds"),
                "latest_time": (arr_ts + timedelta(hours=8)).isoformat(timespec="seconds"),
            })
        return arrivals

    def _noshow_prob(self, appt: Dict[str, Any], state: TwinState) -> float:
        pid = appt.get("patient_id")
        if self.noshow_fn is not None:
            try:
                return float(self.noshow_fn(state.patient_data_map.get(pid, {}), appt))
            except Exception as exc:  # pragma: no cover — injected callback failures
                logger.warning(f"noshow_fn failed for {pid}: {exc}")
        # Fallback: patient-specific rate from map, else 0.15
        rec = state.patient_data_map.get(pid, {})
        ns = rec.get("no_show_rate")
        if ns is None:
            apps = rec.get("total_appointments", 0)
            nss = rec.get("no_shows", 0)
            if apps:
                ns = float(nss) / float(apps)
        if ns is None:
            ns = 0.15
        return float(max(0.0, min(1.0, ns)))

    # ----------------------------------------------------------------- #
    # Full-horizon evaluation
    # ----------------------------------------------------------------- #

    def evaluate_policy(
        self,
        snapshot: Optional[TwinState] = None,
        policy: Optional[PolicySpec] = None,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        step_hours: int = DEFAULT_STEP_HOURS,
        rng_seed: int = DEFAULT_RNG_SEED,
    ) -> TwinEvaluation:
        """
        Run a policy over ``horizon_days`` days of virtual time and
        return aggregated metrics.

        Uses ``snapshot`` if provided, else the last snapshot, else
        a synthetic empty snapshot.
        """
        policy = policy or PolicySpec()
        snap = snapshot or self.last_snapshot
        if snap is None:
            snap = _empty_snapshot()
        state = snap.clone()

        if self.arrival_model is None:
            self.arrival_model = _uniform_fallback_model(0)

        rng = random.Random(int(rng_seed))
        start = time.perf_counter()

        n_steps = int(horizon_days) * 24 // max(int(step_hours), 1)
        step_results: List[TwinStepResult] = []
        for i in range(n_steps):
            state, sr = self.step(state, policy, rng, step_hours=step_hours)
            step_results.append(sr)
            # Optional slow-path reopt at policy's cadence
            if (
                policy.slow_path_every_n_steps
                and self.optimize_fn is not None
                and (i + 1) % int(policy.slow_path_every_n_steps) == 0
            ):
                try:
                    self.optimize_fn(state, policy)
                except Exception as exc:  # pragma: no cover — optimizer failures
                    logger.warning(f"optimize_fn failed at step {i}: {exc}")

        runtime_s = time.perf_counter() - start

        # Aggregate
        total_arr = sum(s.arrivals_sampled for s in step_results)
        total_acc = sum(s.arrivals_accepted for s in step_results)
        total_rej = sum(s.arrivals_rejected for s in step_results)
        total_db = sum(s.double_bookings_created for s in step_results)
        total_rs = sum(s.reschedules_performed for s in step_results)
        total_ns = sum(s.noshows_realised for s in step_results)
        total_comp = sum(s.appointments_completed for s in step_results)
        util_mean = (
            sum(s.utilization_pct for s in step_results) / max(len(step_results), 1)
        )
        acc_rate = total_acc / max(total_arr, 1)
        noshow_rate = total_ns / max(total_ns + total_comp, 1)
        latencies = sorted(s.step_latency_ms for s in step_results)
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

        # Policy score — higher is better
        # 0.45 * accept + 0.30 * utilization - 0.15 * db_rate - 0.10 * noshow_rate
        db_rate = total_db / max(total_acc, 1)
        policy_score = (
            0.45 * acc_rate
            + 0.30 * (util_mean / 100.0)
            - 0.15 * db_rate
            - 0.10 * noshow_rate
        )

        # Guardrails
        violations: List[str] = []
        if db_rate > GUARDRAIL_MAX_DOUBLE_BOOK_RATE:
            violations.append(
                f"double_book_rate={db_rate:.3f} > {GUARDRAIL_MAX_DOUBLE_BOOK_RATE}"
            )
        if acc_rate < GUARDRAIL_MIN_ACCEPT_RATE and total_arr > 0:
            violations.append(
                f"accept_rate={acc_rate:.3f} < {GUARDRAIL_MIN_ACCEPT_RATE}"
            )
        if noshow_rate > GUARDRAIL_MAX_NOSHOW_RATE:
            violations.append(
                f"noshow_rate={noshow_rate:.3f} > {GUARDRAIL_MAX_NOSHOW_RATE}"
            )

        ev = TwinEvaluation(
            policy=policy.to_dict(),
            horizon_days=int(horizon_days),
            step_hours=int(step_hours),
            num_steps=len(step_results),
            rng_seed=int(rng_seed),
            total_arrivals=total_arr,
            total_accepted=total_acc,
            total_rejected=total_rej,
            total_double_bookings=total_db,
            total_reschedules=total_rs,
            total_noshows_realised=total_ns,
            accept_rate=float(acc_rate),
            noshow_rate_realised=float(noshow_rate),
            mean_utilization_pct=float(util_mean),
            p50_step_latency_ms=float(p50),
            p95_step_latency_ms=float(p95),
            policy_score=float(policy_score),
            guardrail_violations=violations,
            runtime_s=float(runtime_s),
            evaluated_ts=datetime.utcnow().isoformat(timespec="seconds"),
            snapshot_ts=snap.snapshot_ts,
            steps=[asdict(s) for s in step_results],
        )
        with self._lock:
            self.last_evaluation = ev
            self.total_evaluations += 1
        self._persist_evaluation(ev)
        _append_event(EVENT_LOG_FILE, {
            "event": "evaluation",
            "ts": ev.evaluated_ts,
            "policy": policy.name,
            "horizon_days": horizon_days,
            "score": ev.policy_score,
            "violations": len(violations),
        })
        return ev

    def compare_policies(
        self,
        policies: List[PolicySpec],
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        step_hours: int = DEFAULT_STEP_HOURS,
        rng_seed: int = DEFAULT_RNG_SEED,
        snapshot: Optional[TwinState] = None,
    ) -> Dict[str, Any]:
        """Run each policy from the same snapshot + seed; rank by score."""
        snap = snapshot or self.last_snapshot or _empty_snapshot()
        results: List[TwinEvaluation] = []
        for p in policies:
            results.append(
                self.evaluate_policy(
                    snapshot=snap,
                    policy=p,
                    horizon_days=horizon_days,
                    step_hours=step_hours,
                    rng_seed=rng_seed,
                )
            )
        ranked = sorted(results, key=lambda e: e.policy_score, reverse=True)
        return {
            "policies": [e.to_dict() for e in results],
            "ranked_names": [e.policy["name"] for e in ranked],
            "best_policy": ranked[0].policy if ranked else None,
            "best_score": ranked[0].policy_score if ranked else None,
            "evaluated_ts": datetime.utcnow().isoformat(timespec="seconds"),
        }

    # ----------------------------------------------------------------- #
    # Persistence / diagnostics
    # ----------------------------------------------------------------- #

    def _persist_evaluation(self, ev: TwinEvaluation) -> Path:
        name = ev.policy.get("name", "policy").replace(" ", "_")
        stem = f"{ev.evaluated_ts.replace(':', '-')}_{name}"
        path = self.evaluations_dir / f"{stem}.json"
        slim = ev.to_dict()
        # Keep only summary step telemetry (accept + latency) to keep file small
        slim["steps"] = [
            {
                "virtual_time": s["virtual_time"],
                "arrivals_sampled": s["arrivals_sampled"],
                "arrivals_accepted": s["arrivals_accepted"],
                "double_bookings_created": s["double_bookings_created"],
                "noshows_realised": s["noshows_realised"],
                "utilization_pct": s["utilization_pct"],
                "step_latency_ms": s["step_latency_ms"],
            }
            for s in slim["steps"]
        ]
        with open(path, "w") as f:
            json.dump(slim, f, indent=2)
        return path

    def list_evaluations(self, limit: int = 50) -> List[Dict[str, Any]]:
        files = sorted(
            self.evaluations_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        out = []
        for p in files:
            try:
                with open(p) as f:
                    d = json.load(f)
                out.append({
                    "file": p.name,
                    "evaluated_ts": d.get("evaluated_ts"),
                    "policy_name": d.get("policy", {}).get("name"),
                    "horizon_days": d.get("horizon_days"),
                    "policy_score": d.get("policy_score"),
                    "accept_rate": d.get("accept_rate"),
                    "mean_utilization_pct": d.get("mean_utilization_pct"),
                    "guardrail_violations": d.get("guardrail_violations", []),
                })
            except Exception as exc:  # pragma: no cover — corrupt eval JSON
                logger.warning(f"Skipping corrupt evaluation {p}: {exc}")
        return out

    def status(self) -> Dict[str, Any]:
        """Shape used by /api/twin/status."""
        am = self.arrival_model
        am_summary = None
        if am is not None:
            vals = list(am.lambda_hd.values())
            am_summary = {
                "total_events": am.total_events,
                "historical_days_observed": am.historical_days_observed,
                "urgent_fraction": am.urgent_fraction,
                "lambda_min": float(min(vals)) if vals else 0.0,
                "lambda_max": float(max(vals)) if vals else 0.0,
                "lambda_mean": float(sum(vals) / len(vals)) if vals else 0.0,
                "fitted_ts": am.fitted_ts,
                "source_file": am.source_file,
            }
        return {
            "arrival_model": am_summary,
            "last_snapshot_ts": self.last_snapshot.snapshot_ts if self.last_snapshot else None,
            "last_evaluation_policy": (
                self.last_evaluation.policy.get("name") if self.last_evaluation else None
            ),
            "last_evaluation_score": (
                self.last_evaluation.policy_score if self.last_evaluation else None
            ),
            "last_evaluation_violations": (
                self.last_evaluation.guardrail_violations if self.last_evaluation else []
            ),
            "total_snapshots": self.total_snapshots,
            "total_evaluations": self.total_evaluations,
            "total_steps": self.total_steps,
            "storage_dir": str(self.storage_dir),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialise_appointments(apps: List[Any]) -> List[Dict[str, Any]]:
    """Convert live ScheduledAppointment-or-dict list into plain dicts."""
    out: List[Dict[str, Any]] = []
    for a in apps:
        if isinstance(a, dict):
            d = copy.deepcopy(a)
        else:
            d = {
                "patient_id": getattr(a, "patient_id", None),
                "chair_id": getattr(a, "chair_id", None),
                "site_code": getattr(a, "site_code", None),
                "start_time": _ts_to_str(getattr(a, "start_time", None)),
                "end_time": _ts_to_str(getattr(a, "end_time", None)),
                "duration": getattr(a, "duration", None),
                "priority": getattr(a, "priority", None),
            }
        out.append(d)
    return out


def _serialise_patients(ps: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in ps:
        if isinstance(p, dict):
            out.append(copy.deepcopy(p))
        else:
            out.append({
                "patient_id": getattr(p, "patient_id", None),
                "priority": getattr(p, "priority", None),
                "protocol": getattr(p, "protocol", None),
                "expected_duration": getattr(p, "expected_duration", None),
                "postcode": getattr(p, "postcode", None),
                "earliest_time": _ts_to_str(getattr(p, "earliest_time", None)),
                "latest_time": _ts_to_str(getattr(p, "latest_time", None)),
                "is_urgent": getattr(p, "is_urgent", False),
            })
    return out


def _ts_to_str(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.isoformat(timespec="seconds")
    if isinstance(ts, str):
        return ts
    return str(ts)


def _parse_appt_time(raw: Any, fallback: datetime) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw))
    except Exception:
        return None


def _compute_chair_capacity(app_state: Dict[str, Any]) -> Dict[str, int]:
    """Best-effort summary of chairs-per-site from app_state or DEFAULT_SITES."""
    cap: Dict[str, int] = {}
    chairs = app_state.get("chairs") if isinstance(app_state, dict) else None
    if chairs:
        for c in chairs:
            sc = getattr(c, "site_code", None) or (c.get("site_code") if isinstance(c, dict) else None)
            if sc is None:
                continue
            cap[sc] = cap.get(sc, 0) + 1
    if cap:
        return cap
    # Fallback to DEFAULT_SITES
    try:
        from config import DEFAULT_SITES
        for s in DEFAULT_SITES:
            cap[s["code"]] = int(s.get("chairs", 0)) + int(s.get("recliners", 0))
    except Exception:
        pass
    if not cap:
        cap = {"UNKNOWN": 8}
    return cap


def _poisson_sample(lam: float, rng: random.Random) -> int:
    """Knuth's Poisson sampler — fine for small λ (< ~20 in our case)."""
    if lam <= 0:
        return 0
    if lam > 30.0:
        # Normal approximation for larger lambda
        x = rng.gauss(lam, math.sqrt(lam))
        return max(0, int(round(x)))
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _append_event(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:  # pragma: no cover — non-fatal
        pass


def _empty_snapshot() -> TwinState:
    now = datetime.utcnow().isoformat(timespec="seconds")
    return TwinState(
        snapshot_ts=now,
        virtual_time=now,
        appointments=[],
        patients_pending=[],
        patient_data_map={},
        operating_weights={},
        chair_capacity={"UNKNOWN": 8},
        mode="NORMAL",
        notes="empty-synthetic",
    )


# ---------------------------------------------------------------------------
# Fallback squeeze fn used when no production primitive is injected
# ---------------------------------------------------------------------------


def _fallback_squeeze_fn(
    arrival: Dict[str, Any], state: TwinState, policy: PolicySpec
) -> Dict[str, Any]:
    """
    Minimal heuristic for tests / dry-runs:
      * If any chair has spare capacity → 'gap', always accept.
      * Else if the policy's double-book threshold < p_noshow of
        the candidate collision → 'double_booking'.
      * Else if policy allows rescheduling → 'reschedule'.
      * Else reject.

    This is NOT used in production.  Flask injects a wrapper around
    ``SqueezeInHandler.squeeze_in_with_noshow``.
    """
    arr_ts = arrival.get("arrival_ts")
    # Count current load vs. capacity
    total_cap = max(sum(state.chair_capacity.values()), 1)
    load = len(state.appointments)

    if load < total_cap:
        return {
            "success": True,
            "strategy": "gap",
            "appointment": {
                "patient_id": arrival["patient_id"],
                "chair_id": list(state.chair_capacity.keys())[0] + "-01",
                "site_code": list(state.chair_capacity.keys())[0],
                "start_time": arr_ts,
                "end_time": arr_ts,
                "duration": arrival.get("expected_duration", 60),
                "priority": arrival.get("priority", 3),
            },
        }

    # Over capacity — decide whether to double-book.
    # Use a synthetic noshow prob equal to the arrival's urgency-inverse.
    p_ns = 0.30 if arrival.get("is_urgent") else 0.10
    if policy.allow_double_booking and p_ns >= policy.double_book_threshold:
        return {
            "success": True,
            "strategy": "double_booking",
            "appointment": {
                "patient_id": arrival["patient_id"],
                "chair_id": list(state.chair_capacity.keys())[0] + "-DB",
                "site_code": list(state.chair_capacity.keys())[0],
                "start_time": arr_ts,
                "end_time": arr_ts,
                "duration": arrival.get("expected_duration", 60),
                "priority": arrival.get("priority", 3),
            },
        }
    if policy.allow_rescheduling:
        return {
            "success": True,
            "strategy": "reschedule",
            "appointment": {
                "patient_id": arrival["patient_id"],
                "chair_id": list(state.chair_capacity.keys())[0] + "-RS",
                "site_code": list(state.chair_capacity.keys())[0],
                "start_time": arr_ts,
                "end_time": arr_ts,
                "duration": arrival.get("expected_duration", 60),
                "priority": arrival.get("priority", 3),
            },
        }
    return {"success": False, "strategy": "rejected"}


# ---------------------------------------------------------------------------
# Module-level convenience singletons (mirrors other §3.x modules)
# ---------------------------------------------------------------------------

_GLOBAL: Optional[DigitalTwin] = None


def get_digital_twin() -> DigitalTwin:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = DigitalTwin()
    return _GLOBAL


def set_digital_twin(twin: DigitalTwin) -> None:
    """Explicit injection used by flask_app.py at start-up."""
    global _GLOBAL
    _GLOBAL = twin
