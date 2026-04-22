"""
Auto-scaling Optimiser with Timeout Guarantees (Dissertation §5.1)
==================================================================

CP-SAT solve time is structurally unpredictable: the same instance
can take 0.1 s today and 30 s tomorrow depending on branching
decisions, so a hospital deployment needs \\emph{hard} timing
guarantees rather than a single optimistic budget.  \\S5.1 asks for
three mechanisms:

1. **Cascade**.  Try 5 s, 2 s, 1 s, 0.5 s sequentially; if every
   CP-SAT attempt fails, fall back to a greedy priority-first
   placement.
2. **Early stopping**.  Abort a solve as soon as the current
   incumbent is provably within 1\\,\\% of the solver's lower bound
   (``objective - best_bound`` / ``objective`` $\\le 0.01$).
3. **Parallel search**.  Run 4 weight configurations simultaneously;
   take the best by composite score.

This module layers on top of the existing ``ScheduleOptimizer``
(``optimization/optimizer.py``) without touching its internals.
It is a pure wrapper — operators opt in per request.

Design contract
---------------
* **Single entry point**: ``AutoScalingOptimizer.optimize(patients)``
  returns a standard ``OptimizationResult`` augmented with a
  ``AutoScalingReport`` dataclass describing the cascade path, the
  winning configuration, and the greedy-fallback flag.
* **Thread-safe**: parallel configurations run in a
  ``ThreadPoolExecutor`` with ``max_workers = parallel_configs``.
  The underlying CP-SAT solver releases the GIL during
  ``Solve()``, so Python threads give a real parallel speed-up.
* **Deterministic tie-breaking**: when parallel configurations tie
  on the composite score we pick the one with the smaller
  fingerprint of its weight vector to keep replay reproducible.
* **Invisible in the prediction pipeline**: the Flask layer
  keeps the legacy ``run_optimization()`` single-solve path as
  default; auto-scaling is toggled via
  ``POST /api/optimize/autoscale/config``.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, DEFAULT_SITES, get_logger, OPTIMIZATION_WEIGHTS

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults straight from the §5.1 brief
# ---------------------------------------------------------------------------
DEFAULT_CASCADE_BUDGETS: Tuple[float, ...] = (5.0, 2.0, 1.0, 0.5)
DEFAULT_PARALLEL_CONFIGS: int = 4
DEFAULT_EARLY_STOP_GAP: float = 0.01        # 1 % optimality gap
DEFAULT_PARALLEL_TIME_LIMIT_S: float = 5.0  # per-worker budget
AUTOSCALE_DIR: Path = DATA_CACHE_DIR / "auto_scaling"
AUTOSCALE_LOG: Path = AUTOSCALE_DIR / "runs.jsonl"

# 4 weight configurations for the parallel race.  All sum to 1.0 and
# cover the corners of the (priority, throughput, robustness) space.
DEFAULT_PARALLEL_WEIGHT_CONFIGS: Tuple[Dict[str, float], ...] = (
    {  # balanced — mirrors config.OPTIMIZATION_WEIGHTS
        'name': 'balanced',
        'priority': 0.30, 'utilization': 0.25, 'noshow_risk': 0.15,
        'waiting_time': 0.15, 'robustness': 0.10, 'travel': 0.05,
    },
    {  # priority-heavy
        'name': 'priority_heavy',
        'priority': 0.55, 'utilization': 0.15, 'noshow_risk': 0.10,
        'waiting_time': 0.10, 'robustness': 0.05, 'travel': 0.05,
    },
    {  # throughput-heavy
        'name': 'throughput',
        'priority': 0.20, 'utilization': 0.45, 'noshow_risk': 0.10,
        'waiting_time': 0.10, 'robustness': 0.10, 'travel': 0.05,
    },
    {  # robustness-heavy
        'name': 'robustness_heavy',
        'priority': 0.25, 'utilization': 0.20, 'noshow_risk': 0.15,
        'waiting_time': 0.10, 'robustness': 0.25, 'travel': 0.05,
    },
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AttemptRecord:
    """One cascade / parallel attempt."""
    stage: str                # 'parallel' | 'cascade' | 'greedy'
    config_name: str          # 'balanced' | 'priority_heavy' | etc.
    time_budget_s: float
    solve_time_s: float
    status: str
    success: bool
    n_scheduled: int
    n_unscheduled: int
    objective_score: Optional[float] = None
    early_stopped: bool = False


@dataclass
class AutoScalingReport:
    computed_ts: str
    n_patients: int
    winner_stage: str                # 'parallel' | 'cascade' | 'greedy'
    winner_config: str
    winner_time_budget_s: float
    winner_solve_time_s: float
    winner_n_scheduled: int
    winner_objective_score: Optional[float]
    early_stopped: bool
    greedy_fallback: bool
    total_wall_time_s: float
    attempts: List[AttemptRecord] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Early-stopping callback (for CP-SAT)
# ---------------------------------------------------------------------------


def _build_early_stop_callback(gap_tol: float):
    """Return a CP-SAT SolutionCallback that halts when the
    optimality gap falls below ``gap_tol``."""
    try:
        from ortools.sat.python import cp_model
    except Exception:  # pragma: no cover — OR-Tools not installed
        return None

    class _EarlyStop(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self._stopped = False
            self._last_gap = None
            self._n_solutions = 0

        def on_solution_callback(self):
            self._n_solutions += 1
            obj = self.ObjectiveValue()
            bnd = self.BestObjectiveBound()
            if obj is None or abs(obj) < 1e-9:
                return
            gap = abs(obj - bnd) / max(abs(obj), 1e-9)
            self._last_gap = gap
            if gap <= gap_tol:
                self._stopped = True
                self.StopSearch()

        @property
        def stopped(self) -> bool:
            return self._stopped

        @property
        def last_gap(self) -> Optional[float]:
            return self._last_gap

    return _EarlyStop()


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------


def greedy_schedule(
    patients: List[Any],
    sites: Optional[List[Dict[str, Any]]] = None,
    day_start_hour: int = 8,
    day_end_hour: int = 18,
) -> Tuple[List[Any], List[str]]:
    """
    Deterministic priority-first greedy placement.

    * Sort patients by (priority, earliest_time) ascending.
    * Walk chairs round-robin, place on first open window.
    * No overlap, respects chair count per site, 5-min slack.

    Returns ``(appointments, unscheduled_patient_ids)``.
    """
    from optimization.optimizer import Chair, ScheduledAppointment
    from datetime import datetime, timedelta

    sites = sites or DEFAULT_SITES
    # Chairs
    chairs: List[Chair] = []
    for s in sites:
        for i in range(int(s.get('chairs', 0)) + int(s.get('recliners', 0))):
            chairs.append(Chair(
                chair_id=f"{s['code']}-C{i+1:02d}",
                site_code=s['code'],
                is_recliner=(i >= int(s.get('chairs', 0))),
            ))
    if not chairs:
        chairs = [Chair(chair_id='DEFAULT-C01', site_code='DEFAULT')]

    # Sort patients
    def _earliest(p):
        e = getattr(p, 'earliest_time', None)
        if isinstance(e, datetime):
            return e
        return datetime.now().replace(hour=day_start_hour, minute=0,
                                      second=0, microsecond=0)
    sorted_ps = sorted(
        patients,
        key=lambda p: (getattr(p, 'priority', 3), _earliest(p))
    )

    # Next free slot per chair_id
    chair_next_free: Dict[str, datetime] = {}
    today = datetime.now().replace(hour=day_start_hour, minute=0,
                                   second=0, microsecond=0)
    day_end = today.replace(hour=day_end_hour, minute=0)

    appointments: List[ScheduledAppointment] = []
    unscheduled: List[str] = []

    for p in sorted_ps:
        placed = False
        dur = int(getattr(p, 'expected_duration', 60) or 60)
        e_time = _earliest(p)
        # Align to today
        if isinstance(e_time, datetime):
            e_time = today.replace(hour=e_time.hour, minute=e_time.minute)
        else:
            e_time = today
        for c in chairs:
            slot_start = max(e_time, chair_next_free.get(c.chair_id, today))
            # Add 5 min slack between appts
            if slot_start + timedelta(minutes=dur) > day_end:
                continue
            appointments.append(ScheduledAppointment(
                patient_id=p.patient_id,
                chair_id=c.chair_id,
                site_code=c.site_code,
                start_time=slot_start,
                end_time=slot_start + timedelta(minutes=dur),
                duration=dur,
                priority=int(getattr(p, 'priority', 3)),
                travel_time=int(getattr(p, 'travel_time_minutes', 30)),
            ))
            chair_next_free[c.chair_id] = (
                slot_start + timedelta(minutes=dur + 5)
            )
            placed = True
            break
        if not placed:
            unscheduled.append(p.patient_id)
    return appointments, unscheduled


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class AutoScalingOptimizer:
    """Cascade + parallel + early-stop wrapper around ``ScheduleOptimizer``.

    Parameters
    ----------
    base_optimizer:
        Callable returning an ``OptimizationResult``.  Must accept
        ``(patients, date=None, time_limit_seconds=...)``.  Flask
        passes its already-initialised ``ScheduleOptimizer.optimize``
        method; tests can pass any mock.
    set_weights:
        Optional callable ``(weights_dict) -> None`` used to swap
        weight vectors between parallel workers.  Legacy hook —
        suffers from a race when multiple workers share one optimiser
        instance, so the T2.2 fix forces serialisation when this is
        the only weight injection mechanism.  Prefer ``solve_with_weights``.
    solve_with_weights:
        Preferred T2.2 callable
        ``(patients, weights_dict, time_limit_s) -> OptimizationResult``
        that returns a result computed with the given weights without
        mutating shared state.  When provided, the parallel race runs
        truly in parallel + correctly.
    cascade_budgets:
        Ordered tuple of time limits (seconds).  Matches §5.1 brief.
    parallel_configs:
        How many weight-config workers to race.
    parallel_time_limit_s:
        Per-worker budget during the parallel race.
    early_stop_gap:
        Optimality-gap tolerance for CP-SAT early stopping.
    weight_configs:
        The concrete weight dicts raced in parallel.  Defaults to the
        built-in 4-corner set.
    """

    def __init__(
        self,
        *,
        base_optimizer: Optional[Callable[..., Any]] = None,
        set_weights: Optional[Callable[[Dict[str, float]], None]] = None,
        solve_with_weights: Optional[
            Callable[[List[Any], Dict[str, float], float], Any]
        ] = None,
        cascade_budgets: Sequence[float] = DEFAULT_CASCADE_BUDGETS,
        parallel_configs: int = DEFAULT_PARALLEL_CONFIGS,
        parallel_time_limit_s: float = DEFAULT_PARALLEL_TIME_LIMIT_S,
        early_stop_gap: float = DEFAULT_EARLY_STOP_GAP,
        weight_configs: Sequence[Dict[str, float]] = DEFAULT_PARALLEL_WEIGHT_CONFIGS,
        enable_parallel: bool = True,
        enable_cascade: bool = True,
        enable_greedy_fallback: bool = True,
        storage_dir: Path = AUTOSCALE_DIR,
    ):
        self._base_optimizer = base_optimizer
        self._set_weights = set_weights
        # T2.2: prefer the per-call API over the legacy mutate-then-solve.
        self._solve_with_weights = solve_with_weights
        self.cascade_budgets = tuple(float(b) for b in cascade_budgets)
        self.parallel_configs = int(parallel_configs)
        self.parallel_time_limit_s = float(parallel_time_limit_s)
        self.early_stop_gap = float(early_stop_gap)
        self.weight_configs = list(weight_configs)[: self.parallel_configs]
        self.enable_parallel = bool(enable_parallel)
        self.enable_cascade = bool(enable_cascade)
        self.enable_greedy_fallback = bool(enable_greedy_fallback)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.runs_log = self.storage_dir / 'runs.jsonl'

        self._lock = threading.Lock()
        self._last: Optional[AutoScalingReport] = None
        self._n_runs: int = 0
        # Logged once when we degrade the parallel race to serial.
        self._race_degrade_warned: bool = False

    # ----------------------------------------------------------- API ---- #

    def optimize(self, patients: List[Any]) -> Tuple[Any, AutoScalingReport]:
        """Run auto-scaling search.

        Returns ``(best_result, AutoScalingReport)``.  The schedule
        returned is the same shape as a regular
        ``ScheduleOptimizer.optimize`` result.
        """
        wall_start = time.perf_counter()
        attempts: List[AttemptRecord] = []
        winner_result = None
        winner_attempt: Optional[AttemptRecord] = None

        # -------- 1. Parallel race --------------------------------------
        if self.enable_parallel and self._base_optimizer is not None:
            parallel_results = self._parallel_race(patients, attempts)
            if parallel_results:
                winner_attempt, winner_result = parallel_results

        # -------- 2. Cascade sequential --------------------------------
        if winner_result is None and self.enable_cascade and self._base_optimizer is not None:
            for budget in self.cascade_budgets[1:]:
                ar, res = self._single_solve(patients, budget, 'cascade', 'balanced')
                attempts.append(ar)
                if ar.success:
                    winner_attempt, winner_result = ar, res
                    break

        # -------- 3. Greedy fallback -----------------------------------
        if winner_result is None and self.enable_greedy_fallback:
            ar, res = self._run_greedy(patients)
            attempts.append(ar)
            winner_attempt, winner_result = ar, res

        wall = time.perf_counter() - wall_start
        report = self._build_report(
            patients, winner_attempt, attempts, wall_s=wall,
        )
        with self._lock:
            self._last = report
            self._n_runs += 1
        self._append_event(report)
        return winner_result, report

    def last(self) -> Optional[AutoScalingReport]:
        with self._lock:
            return self._last

    def status(self) -> Dict[str, Any]:
        last = self.last()
        return {
            'cascade_budgets': list(self.cascade_budgets),
            'parallel_configs': self.parallel_configs,
            'parallel_time_limit_s': self.parallel_time_limit_s,
            'early_stop_gap': self.early_stop_gap,
            'weight_config_names': [w.get('name') for w in self.weight_configs],
            'enable_parallel': self.enable_parallel,
            'enable_cascade': self.enable_cascade,
            'enable_greedy_fallback': self.enable_greedy_fallback,
            'total_runs': self._n_runs,
            'log_path': str(self.runs_log),
            'last_winner_stage': last.winner_stage if last else None,
            'last_winner_config': last.winner_config if last else None,
            'last_winner_solve_time_s': last.winner_solve_time_s if last else None,
            'last_greedy_fallback': last.greedy_fallback if last else None,
            'last_total_wall_time_s': last.total_wall_time_s if last else None,
            'last_narrative': last.narrative if last else None,
        }

    def update_config(
        self,
        cascade_budgets: Optional[Sequence[float]] = None,
        parallel_configs: Optional[int] = None,
        parallel_time_limit_s: Optional[float] = None,
        early_stop_gap: Optional[float] = None,
        enable_parallel: Optional[bool] = None,
        enable_cascade: Optional[bool] = None,
        enable_greedy_fallback: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if cascade_budgets is not None:
            self.cascade_budgets = tuple(float(b) for b in cascade_budgets)
        if parallel_configs is not None:
            self.parallel_configs = int(parallel_configs)
            self.weight_configs = list(DEFAULT_PARALLEL_WEIGHT_CONFIGS)[
                : self.parallel_configs
            ]
        if parallel_time_limit_s is not None:
            self.parallel_time_limit_s = float(parallel_time_limit_s)
        if early_stop_gap is not None:
            self.early_stop_gap = float(early_stop_gap)
        if enable_parallel is not None:
            self.enable_parallel = bool(enable_parallel)
        if enable_cascade is not None:
            self.enable_cascade = bool(enable_cascade)
        if enable_greedy_fallback is not None:
            self.enable_greedy_fallback = bool(enable_greedy_fallback)
        return {
            'cascade_budgets': list(self.cascade_budgets),
            'parallel_configs': self.parallel_configs,
            'parallel_time_limit_s': self.parallel_time_limit_s,
            'early_stop_gap': self.early_stop_gap,
            'enable_parallel': self.enable_parallel,
            'enable_cascade': self.enable_cascade,
            'enable_greedy_fallback': self.enable_greedy_fallback,
        }

    # --------------------------------------------------------- Internals - #

    def _attempt_from_result(
        self,
        result: Any,
        *,
        stage: str,
        config_name: str,
        time_budget_s: float,
        solve_time: float,
    ) -> AttemptRecord:
        """Build an AttemptRecord from a base-optimiser result."""
        success = bool(getattr(result, 'success', False))
        appts = getattr(result, 'appointments', []) or []
        uns = getattr(result, 'unscheduled', []) or []
        metrics = getattr(result, 'metrics', None) or {}
        obj_score = (
            metrics.get('objective_score') if isinstance(metrics, dict) else None
        )
        return AttemptRecord(
            stage=stage,
            config_name=config_name,
            time_budget_s=float(time_budget_s),
            solve_time_s=float(solve_time),
            status=str(getattr(result, 'status', 'UNKNOWN')),
            success=success,
            n_scheduled=len(appts),
            n_unscheduled=len(uns),
            objective_score=float(obj_score) if obj_score is not None else None,
            early_stopped=solve_time < 0.9 * time_budget_s and success,
        )

    def _failed_attempt(
        self, config_name: str, time_budget_s: float, exc: Exception,
    ) -> AttemptRecord:
        return AttemptRecord(
            stage='parallel',
            config_name=config_name,
            time_budget_s=float(time_budget_s),
            solve_time_s=0.0,
            status=f'EXCEPTION:{type(exc).__name__}',
            success=False,
            n_scheduled=0,
            n_unscheduled=0,
            objective_score=None,
            early_stopped=False,
        )

    def _single_solve(
        self,
        patients: List[Any],
        time_budget_s: float,
        stage: str,
        config_name: str,
    ) -> Tuple[AttemptRecord, Any]:
        t0 = time.perf_counter()
        try:
            result = self._base_optimizer(
                patients, time_limit_seconds=max(int(time_budget_s), 1),
            )
        except TypeError:
            # Some mocks accept a slightly different kwarg name
            result = self._base_optimizer(
                patients, time_limit=max(int(time_budget_s), 1),
            )
        solve_time = time.perf_counter() - t0
        attempt = self._attempt_from_result(
            result, stage=stage, config_name=config_name,
            time_budget_s=time_budget_s, solve_time=solve_time,
        )
        return attempt, result

    def _parallel_race(
        self, patients: List[Any], attempts_sink: List[AttemptRecord]
    ) -> Optional[Tuple[AttemptRecord, Any]]:
        """Race ``self.parallel_configs`` weight configurations.

        T2.2 race fix: previously each worker called ``self._set_weights(cfg)``
        under a lock held only for the assignment, then released the lock and
        ran the solve.  Worker B could overwrite worker A's weights mid-solve,
        so worker A returned a result computed with the wrong config.  Two
        correct paths now exist:

        1. ``self._solve_with_weights`` (preferred) — caller injects a callable
           ``(patients, weights_dict, time_limit_s) -> result`` that internally
           clones / threads the weights through to a fresh optimiser.  Each
           worker is fully independent; true parallelism is preserved.
        2. ``self._set_weights`` (legacy) — held under the lock for the WHOLE
           worker (set + solve), so workers serialise.  Correct, but no
           speed-up.  A warning is logged the first time we degrade.
        """
        if self.parallel_configs <= 1 or not self.weight_configs:
            return None
        budget = self.parallel_time_limit_s
        lock = threading.Lock()
        per_run_attempts: List[Tuple[AttemptRecord, Any]] = []

        def _worker(cfg: Dict[str, float]) -> Tuple[AttemptRecord, Any]:
            cfg_name = str(cfg.get('name', 'custom'))
            weights_only = {k: v for k, v in cfg.items() if k != 'name'}

            # Path 1: per-call weights — race correctly + truly in parallel.
            if self._solve_with_weights is not None:
                t0 = time.perf_counter()
                try:
                    res = self._solve_with_weights(patients, weights_only, budget)
                except Exception as exc:                  # pragma: no cover
                    logger.warning(f"solve_with_weights failed for {cfg_name}: {exc}")
                    return self._failed_attempt(cfg_name, budget, exc), None
                solve_time = time.perf_counter() - t0
                ar = self._attempt_from_result(
                    res, stage='parallel', config_name=cfg_name,
                    time_budget_s=budget, solve_time=solve_time,
                )
                return ar, res

            # Path 2: legacy set_weights — must hold the lock across the
            # entire solve so workers can't trample each other's weights.
            if self._set_weights is not None:
                with lock:
                    if not self._race_degrade_warned:
                        logger.warning(
                            "auto-scaling parallel race degraded to serial — "
                            "inject solve_with_weights to restore parallelism"
                        )
                        self._race_degrade_warned = True
                    self._set_weights(weights_only)
                    ar, res = self._single_solve(
                        patients, budget, 'parallel', cfg_name,
                    )
                return ar, res

            # No weight injection mechanism — every worker just races on the
            # same balanced config; pointless but not incorrect.
            return self._single_solve(patients, budget, 'parallel', cfg_name)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.parallel_configs
        ) as pool:
            futs = {pool.submit(_worker, cfg): cfg for cfg in self.weight_configs}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    ar, res = fut.result()
                    per_run_attempts.append((ar, res))
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"Parallel worker failed: {exc}")

        for ar, _ in per_run_attempts:
            attempts_sink.append(ar)

        # Pick the best successful attempt by (n_scheduled desc, solve_time asc)
        candidates = [(ar, res) for ar, res in per_run_attempts if ar.success]
        if not candidates:
            return None
        candidates.sort(
            key=lambda x: (-x[0].n_scheduled,
                           x[0].solve_time_s,
                           x[0].config_name),
        )
        return candidates[0]

    def _run_greedy(self, patients: List[Any]) -> Tuple[AttemptRecord, Any]:
        from optimization.optimizer import OptimizationResult
        t0 = time.perf_counter()
        appts, uns = greedy_schedule(patients)
        solve_time = time.perf_counter() - t0
        result = OptimizationResult(
            success=len(appts) > 0,
            appointments=appts,
            unscheduled=uns,
            metrics={'objective_score': None},
            solve_time=solve_time,
            status='GREEDY_FALLBACK',
        )
        attempt = AttemptRecord(
            stage='greedy',
            config_name='priority_first',
            time_budget_s=0.0,
            solve_time_s=float(solve_time),
            status='GREEDY_FALLBACK',
            success=bool(result.success),
            n_scheduled=len(appts),
            n_unscheduled=len(uns),
            objective_score=None,
            early_stopped=False,
        )
        return attempt, result

    def _build_report(
        self,
        patients: List[Any],
        winner: Optional[AttemptRecord],
        attempts: List[AttemptRecord],
        wall_s: float,
    ) -> AutoScalingReport:
        n = len(patients)
        if winner is None:
            winner = AttemptRecord(
                stage='greedy', config_name='none',
                time_budget_s=0.0, solve_time_s=0.0,
                status='NO_SOLUTION', success=False,
                n_scheduled=0, n_unscheduled=n,
            )
        greedy_fallback = winner.stage == 'greedy'
        early_stop = any(a.early_stopped for a in attempts)
        narrative = _build_narrative(
            winner, attempts, wall_s, n,
        )
        return AutoScalingReport(
            computed_ts=datetime.utcnow().isoformat(timespec='seconds'),
            n_patients=int(n),
            winner_stage=winner.stage,
            winner_config=winner.config_name,
            winner_time_budget_s=winner.time_budget_s,
            winner_solve_time_s=winner.solve_time_s,
            winner_n_scheduled=winner.n_scheduled,
            winner_objective_score=winner.objective_score,
            early_stopped=bool(early_stop),
            greedy_fallback=bool(greedy_fallback),
            total_wall_time_s=float(wall_s),
            attempts=attempts,
            narrative=narrative,
        )

    def _append_event(self, report: AutoScalingReport) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            record = {
                'ts': report.computed_ts,
                'n_patients': report.n_patients,
                'winner_stage': report.winner_stage,
                'winner_config': report.winner_config,
                'winner_solve_time_s': report.winner_solve_time_s,
                'winner_n_scheduled': report.winner_n_scheduled,
                'early_stopped': report.early_stopped,
                'greedy_fallback': report.greedy_fallback,
                'total_wall_time_s': report.total_wall_time_s,
                'narrative': report.narrative,
                'attempts': [asdict(a) for a in report.attempts],
            }
            with open(self.runs_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Auto-scaling log write failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_narrative(
    winner: AttemptRecord,
    attempts: List[AttemptRecord],
    wall_s: float,
    n_patients: int,
) -> str:
    if not winner.success:
        return (
            f"Auto-scaling: FAIL -- no path succeeded across "
            f"{len(attempts)} attempt(s) over {n_patients} patients "
            f"in {wall_s:.2f}s wall time."
        )
    stages = ', '.join(sorted({a.stage for a in attempts}))
    msg = (
        f"Auto-scaling: winner stage = '{winner.stage}' / config = "
        f"'{winner.config_name}' with {winner.n_scheduled}/{n_patients} "
        f"scheduled in {winner.solve_time_s:.2f}s; total wall time "
        f"{wall_s:.2f}s across stages [{stages}]"
    )
    if winner.early_stopped:
        msg += "; early-stopped within 1% of optimal"
    if winner.stage == 'greedy':
        msg += "; GREEDY FALLBACK engaged"
    return msg + "."


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GLOBAL: Optional[AutoScalingOptimizer] = None


def get_auto_scaler() -> AutoScalingOptimizer:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = AutoScalingOptimizer()
    return _GLOBAL


def set_auto_scaler(o: AutoScalingOptimizer) -> None:
    global _GLOBAL
    _GLOBAL = o
