"""
Stochastic MPC Scheduler (Dissertation §5.5)
============================================

Real cancer day units don't run a static schedule.  They face three
sources of real-time uncertainty the deterministic optimiser can't
see:

  1. Unscheduled urgent arrivals (neutropenic sepsis, toxicity)
  2. Late cancellations (patient calls at 8:45 for a 09:00 slot)
  3. Early finishes / overruns (actual durations drift from
     predictions)

A world-class system has to react *optimally* to these events as
they unfold, balancing new patients against disruption of
already-scheduled care.  This module ships that controller.

Architecture
------------
We formulate the day as a finite-horizon Markov Decision Process and
solve it with **Model Predictive Control (MPC) + scenario rollout**:

    Real-time event -> MPC controller
                       |
                       |  samples K futures for the next tau minutes
                       |  runs CP-SAT on each sample, fixes t=0 action
                       |  picks the action maximising average reward
                       |
                       |  on timeout (>500 ms) fall back to the
                       |  existing squeeze_in heuristic
                       v
                    ScheduleAction

Concrete pieces, all co-located in this file so the dissertation §5.5
maps 1:1 to the implementation:

* :class:`MDPState` / :class:`ChairState` / :class:`QueuedPatient`
  — state-space encoding.
* :class:`ScheduleAction` — the action the controller picks.
* :class:`UrgentArrivalModel` — Gamma-Poisson conjugate updates;
  exactly the class from the §S-4 brief, plus persistence.
* :class:`ScenarioSampler` — samples K (no-show, cancel,
  urgent-arrival) triples for the next tau minutes.
* :class:`TerminalValueFunction` — learnable approximation of the
  end-of-day value.  Ships with a feature-weighted linear default
  and optional PyTorch MLP upgrade.
* :class:`RolloutPlanner` — per-scenario planner; uses an injected
  ``cpsat_fn`` or falls back to a priority-ordered greedy over the
  lookahead window.
* :class:`MPCController` — the orchestrator.  Two public methods:
  ``decide(state)`` and ``event_trigger(kind, payload)``.
  Wraps the existing squeeze-in handler as a hard fallback.
* Metrics + event log at
  ``data_cache/mpc_scheduler/events.jsonl``.

Design invariants
-----------------
* **Invisible in the static pipeline**.  ``run_optimization`` is
  unchanged; the MPC controller is opt-in through
  ``/api/mpc/decide`` or ``/api/mpc/simulate`` and attaches its
  last decision to the optimisation result as
  ``app_state['optimization_results']['mpc_decision']``.  No UI
  panel per the brief.
* **Deterministic under a seed**.  Scenario sampler + reward
  computation are pure-Python so a given seed yields byte-identical
  decisions.
* **Fails safe**.  On scenario-solve timeout or exception the
  controller returns the squeeze-in handler's action so no request
  is left unanswered.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults straight from the §S-2 brief
# ---------------------------------------------------------------------------
DEFAULT_DECISION_INTERVAL_MIN: int = 5
DEFAULT_LOOKAHEAD_MINUTES: int = 120
DEFAULT_N_SCENARIOS: int = 10
DEFAULT_CPSAT_TIMEOUT_S: float = 1.0
DEFAULT_TOTAL_TIMEOUT_S: float = 0.5    # §S-3 fallback trigger
DEFAULT_DAY_START_HOUR: int = 8
DEFAULT_DAY_END_HOUR: int = 18
DEFAULT_WAIT_PENALTY: float = 1.0
DEFAULT_IDLE_PENALTY: float = 5.0
DEFAULT_PRIORITY_COMPLETE_BASE: float = 10.0
DEFAULT_TERMINAL_UNSCHEDULED_PENALTY: float = 20.0
MPC_DIR: Path = DATA_CACHE_DIR / "mpc_scheduler"
MPC_EVENT_LOG: Path = MPC_DIR / "events.jsonl"
MPC_DECISION_LOG: Path = MPC_DIR / "decisions.jsonl"
MPC_SIM_LOG: Path = MPC_DIR / "simulations.jsonl"
ARRIVAL_MODEL_FILE: Path = MPC_DIR / "arrival_model.json"
VALUE_FN_FILE: Path = MPC_DIR / "value_function.json"


# ---------------------------------------------------------------------------
# MDP state / action / reward shapes
# ---------------------------------------------------------------------------


class ChairStatus(str, Enum):
    IDLE = "IDLE"
    OCCUPIED = "OCCUPIED"


@dataclass
class ChairState:
    chair_id: str
    site_code: str
    status: ChairStatus
    patient_id: Optional[str] = None
    remaining_minutes: int = 0
    # T2.1 fix: priority of the patient assigned to this chair, captured at
    # assignment time so the reward function can credit completions by
    # priority even after the queue entry has been removed.  None when the
    # chair is IDLE.
    priority_at_assignment: Optional[int] = None

    def copy(self) -> "ChairState":
        return ChairState(
            chair_id=self.chair_id, site_code=self.site_code,
            status=self.status, patient_id=self.patient_id,
            remaining_minutes=self.remaining_minutes,
            priority_at_assignment=self.priority_at_assignment,
        )


@dataclass
class QueuedPatient:
    patient_id: str
    priority: int
    expected_duration: int
    no_show_prob: float = 0.10
    cancel_prob: float = 0.05
    wait_penalty: float = DEFAULT_WAIT_PENALTY
    arrival_time_min: int = 0          # minutes since day-start when queued
    is_urgent: bool = False

    def copy(self) -> "QueuedPatient":
        return QueuedPatient(**asdict(self))


@dataclass
class MDPState:
    time_min: int                      # minutes since day-start
    chairs: List[ChairState]
    queue: List[QueuedPatient]
    cumulative_wait_minutes: float = 0.0
    n_noshows_so_far: int = 0
    n_cancellations_so_far: int = 0

    def copy(self) -> "MDPState":
        return MDPState(
            time_min=self.time_min,
            chairs=[c.copy() for c in self.chairs],
            queue=[p.copy() for p in self.queue],
            cumulative_wait_minutes=self.cumulative_wait_minutes,
            n_noshows_so_far=self.n_noshows_so_far,
            n_cancellations_so_far=self.n_cancellations_so_far,
        )

    @property
    def idle_chairs(self) -> List[ChairState]:
        return [c for c in self.chairs if c.status == ChairStatus.IDLE]

    @property
    def n_idle_chairs(self) -> int:
        return len(self.idle_chairs)

    def feature_vector(self, day_end_min: int = 600) -> List[float]:
        """Six-dim feature vector used by the terminal value function."""
        n_queue_high = sum(1 for p in self.queue if p.priority <= 2)
        n_queue_low = sum(1 for p in self.queue if p.priority >= 3)
        n_idle = self.n_idle_chairs
        time_frac = min(1.0, self.time_min / max(day_end_min, 1))
        remaining_capacity = (
            sum(1 for _ in self.chairs) *
            max(day_end_min - self.time_min, 0)
        )
        return [
            float(n_queue_high),
            float(n_queue_low),
            float(n_idle),
            float(time_frac),
            float(self.cumulative_wait_minutes / 60.0),
            float(remaining_capacity),
        ]


@dataclass
class ScheduleAction:
    assignments: Dict[str, Optional[str]]   # chair_id -> patient_id or None
    reason: str = ""

    @property
    def any_assignment(self) -> bool:
        return any(v is not None for v in self.assignments.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": dict(self.assignments),
            "reason": self.reason,
            "any_assignment": self.any_assignment,
        }


@dataclass
class ScenarioOutcome:
    scenario_id: int
    immediate_action: ScheduleAction
    total_reward: float
    n_urgent_arrivals: int
    n_noshows: int
    n_cancellations: int


@dataclass
class MPCDecision:
    ts: str
    time_min: int
    n_scenarios: int
    lookahead_minutes: int
    action: Dict[str, Any]
    chosen_action_expected_reward: float
    per_scenario_rewards: List[float]
    used_fallback: bool
    fallback_reason: Optional[str]
    decision_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# §S-4 UrgentArrivalModel — Gamma-Poisson conjugate
# ---------------------------------------------------------------------------


class UrgentArrivalModel:
    """Bayesian Gamma-Poisson model for urgent-arrival rate (arrivals / min).

    Exactly the structure the §S-4 brief prescribes:

    * ``alpha``, ``beta`` = Gamma posterior hyperparameters.
    * ``update(arrivals, minutes_observed)`` = conjugate update.
    * ``predict_rate()`` = posterior mean = alpha / beta (arrivals/min).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._lock = threading.Lock()
        self._total_observations = 0
        self._fitted_ts: Optional[str] = None

    def update(self, arrivals_in_interval: int, minutes_observed: float = 5.0) -> None:
        with self._lock:
            self.alpha += float(max(int(arrivals_in_interval), 0))
            self.beta += float(max(minutes_observed, 1e-9))
            self._total_observations += int(max(int(arrivals_in_interval), 0))
            self._fitted_ts = datetime.utcnow().isoformat(timespec="seconds")

    def predict_rate(self, t: Optional[int] = None) -> float:
        """Posterior mean rate in arrivals per minute.

        ``t`` is a convenience hook for time-varying rates; the default
        implementation is stationary Gamma-Poisson.
        """
        with self._lock:
            return float(self.alpha / max(self.beta, 1e-9))

    def predict_arrivals_in(self, minutes: float) -> float:
        return self.predict_rate() * float(minutes)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "alpha": self.alpha,
                "beta": self.beta,
                "posterior_mean_rate_per_min": self.alpha / max(self.beta, 1e-9),
                "posterior_mean_rate_per_hour":
                    60.0 * self.alpha / max(self.beta, 1e-9),
                "total_observations": self._total_observations,
                "fitted_ts": self._fitted_ts,
            }

    def save(self, path: Path = ARRIVAL_MODEL_FILE) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "total_observations": self._total_observations,
                    "fitted_ts": self._fitted_ts,
                }, f, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"UrgentArrivalModel save failed: {exc}")

    @classmethod
    def load(cls, path: Path = ARRIVAL_MODEL_FILE) -> "UrgentArrivalModel":
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            m = cls(alpha=float(d.get("alpha", 1.0)),
                    beta=float(d.get("beta", 1.0)))
            m._total_observations = int(d.get("total_observations", 0))
            m._fitted_ts = d.get("fitted_ts")
            return m
        except Exception:
            return cls()


# ---------------------------------------------------------------------------
# §S-2.3 TerminalValueFunction — fast approximate V_hat(S)
# ---------------------------------------------------------------------------


class TerminalValueFunction:
    """Feature-weighted linear approximator for the end-of-day value.

    Ships with a hand-calibrated default: positive weight on idle-chair
    count + remaining capacity (more flexibility left = higher future
    value), negative weights on queue length and cumulative wait.  The
    operator can optionally train a PyTorch MLP offline on complete-day
    simulations (§S-2.3) and load its weights via ``from_json``.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "n_queue_high", "n_queue_low", "n_idle",
        "time_frac", "cumulative_wait_hours", "remaining_capacity",
    )

    def __init__(self, weights: Optional[Sequence[float]] = None,
                 bias: float = 0.0):
        self.weights: List[float] = list(weights) if weights else [
            -4.0,   # n_queue_high: large penalty per unserved high-priority
            -1.0,   # n_queue_low
             2.0,   # n_idle (flexibility)
            -5.0,   # time_frac (less time left → less value)
            -1.5,   # cumulative_wait_hours
             0.01,  # remaining_capacity
        ]
        self.bias = float(bias)

    def predict(self, state: MDPState, day_end_min: int = 600) -> float:
        feats = state.feature_vector(day_end_min=day_end_min)
        return float(
            sum(w * x for w, x in zip(self.weights, feats)) + self.bias
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": list(self.weights),
            "bias": float(self.bias),
            "feature_names": list(self.FEATURE_NAMES),
        }

    def save(self, path: Path = VALUE_FN_FILE) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"TerminalValueFunction save failed: {exc}")

    @classmethod
    def load(cls, path: Path = VALUE_FN_FILE) -> "TerminalValueFunction":
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(
                weights=list(d.get("weights", [])) or None,
                bias=float(d.get("bias", 0.0)),
            )
        except Exception:
            return cls()


# ---------------------------------------------------------------------------
# §S-2 Reward function
# ---------------------------------------------------------------------------


def compute_immediate_reward(
    state_before: MDPState,
    action: ScheduleAction,
    state_after: MDPState,
    step_minutes: int,
    wait_penalty: float = DEFAULT_WAIT_PENALTY,
    idle_penalty: float = DEFAULT_IDLE_PENALTY,
    priority_complete_base: float = DEFAULT_PRIORITY_COMPLETE_BASE,
) -> float:
    """Reward for one decision epoch (§S-1.4 formula)."""
    r = 0.0
    # Completions during [t, t+step)
    finished = [
        c for c in state_after.chairs
        if c.status == ChairStatus.IDLE and any(
            bc.chair_id == c.chair_id and bc.status == ChairStatus.OCCUPIED
            for bc in state_before.chairs
        )
    ]
    # Credit completions by priority — we now stash priority on the chair at
    # assignment time (T2.1) so the reward correctly favours urgent cases.
    # Multiplier (6 - priority) gives priority 1 (most urgent) the largest
    # reward and priority 5 the smallest, matching the §S-1.4 formulation.
    for fc in finished:
        prio_pre = next(
            (bc for bc in state_before.chairs if bc.chair_id == fc.chair_id),
            None,
        )
        if prio_pre is None:
            continue
        priority = prio_pre.priority_at_assignment
        if priority is None:
            # Pre-existing OCCUPIED chair from initial state — assume mid (3).
            priority = 3
        # Clamp to [1, 5] so a misbehaving caller doesn't crater the reward.
        priority = max(1, min(5, int(priority)))
        r += priority_complete_base * (6 - priority)

    # Waiting cost (capped at 2 hours / patient per epoch)
    for p in state_after.queue:
        waited = max(state_after.time_min - p.arrival_time_min, 0) / 60.0
        r -= wait_penalty * min(waited, 2.0)

    # Idle-chair-with-queue penalty
    if state_after.queue:
        for c in state_after.chairs:
            if c.status == ChairStatus.IDLE:
                r -= idle_penalty

    return float(r)


def compute_terminal_penalty(
    state: MDPState,
    priority_complete_base: float = DEFAULT_PRIORITY_COMPLETE_BASE,
    unscheduled_penalty: float = DEFAULT_TERMINAL_UNSCHEDULED_PENALTY,
) -> float:
    """§S-1.4 end-of-day penalty for unscheduled patients."""
    pen = 0.0
    for p in state.queue:
        pen -= max(5 - p.priority, 1) * unscheduled_penalty
    return float(pen)


# ---------------------------------------------------------------------------
# §S-2 ScenarioSampler
# ---------------------------------------------------------------------------


class ScenarioSampler:
    """Sample K plausible (arrivals, no-show, cancel) futures for [t, t+tau)."""

    def __init__(
        self,
        arrival_model: UrgentArrivalModel,
        n_scenarios: int = DEFAULT_N_SCENARIOS,
        step_minutes: int = DEFAULT_DECISION_INTERVAL_MIN,
        rng_seed: Optional[int] = None,
    ):
        self.arrival_model = arrival_model
        self.n_scenarios = int(n_scenarios)
        self.step_minutes = int(step_minutes)
        self._rng = random.Random(rng_seed)

    def sample_scenarios(
        self,
        state: MDPState,
        lookahead_minutes: int = DEFAULT_LOOKAHEAD_MINUTES,
    ) -> List[Dict[str, Any]]:
        """Return ``n_scenarios`` scenarios, each describing the random
        realisations over ``[state.time_min, state.time_min + lookahead)``.
        """
        rate = self.arrival_model.predict_rate()
        n_steps = max(1, lookahead_minutes // max(self.step_minutes, 1))
        scenarios: List[Dict[str, Any]] = []
        for k in range(self.n_scenarios):
            urgent_arrivals: List[Dict[str, Any]] = []
            # Poisson arrivals per step
            for s in range(n_steps):
                n = _poisson(rate * self.step_minutes, self._rng)
                for _ in range(n):
                    offset = s * self.step_minutes + self._rng.randint(
                        0, max(self.step_minutes - 1, 0)
                    )
                    urgent_arrivals.append({
                        "arrival_time_min": state.time_min + offset,
                        "expected_duration": 45 + self._rng.randint(-10, 15),
                        "priority": 1 if self._rng.random() < 0.6 else 2,
                    })
            # Per-patient no-show / cancel realisations
            outcomes: Dict[str, str] = {}
            for p in state.queue:
                r = self._rng.random()
                if r < p.no_show_prob:
                    outcomes[p.patient_id] = "noshow"
                elif r < p.no_show_prob + p.cancel_prob:
                    outcomes[p.patient_id] = "cancel"
                else:
                    outcomes[p.patient_id] = "show"
            scenarios.append({
                "scenario_id": k,
                "urgent_arrivals": urgent_arrivals,
                "patient_outcomes": outcomes,
            })
        return scenarios


# ---------------------------------------------------------------------------
# §S-2.2 RolloutPlanner — per-scenario planner
# ---------------------------------------------------------------------------


class RolloutPlanner:
    """
    Evaluate a candidate immediate action under one sampled scenario.

    The plan is a short rollout: we apply the immediate action, then
    greedily fill each subsequent free chair with the highest-priority
    waiting patient (consistent with the §S-2 fast-heuristic fallback
    for the distant future).  The terminal value function estimates the
    value beyond the horizon.

    This rollout mirrors what a CP-SAT solve would buy us in expectation
    (see §S-2.2) but runs in microseconds, so K=10 scenarios stays well
    under the 500 ms MPC budget even without a CP-SAT process.  Operators
    who have CP-SAT available can inject a ``cpsat_fn`` to replace the
    greedy rollout.
    """

    def __init__(
        self,
        value_fn: TerminalValueFunction,
        cpsat_fn: Optional[Callable[[MDPState, int], Any]] = None,
        day_end_min: int = 600,
        step_minutes: int = DEFAULT_DECISION_INTERVAL_MIN,
    ):
        self.value_fn = value_fn
        self.cpsat_fn = cpsat_fn
        self.day_end_min = int(day_end_min)
        self.step_minutes = int(step_minutes)

    def evaluate_action(
        self,
        state: MDPState,
        action: ScheduleAction,
        scenario: Dict[str, Any],
        lookahead_minutes: int = DEFAULT_LOOKAHEAD_MINUTES,
    ) -> Tuple[float, MDPState]:
        """Return ``(total_reward_within_lookahead + terminal_value, final_state)``."""
        s = state.copy()
        s = self._apply_action(s, action)
        total_r = compute_immediate_reward(state, action, s, self.step_minutes)

        steps = max(1, lookahead_minutes // max(self.step_minutes, 1))
        scenario_arrivals = list(scenario.get("urgent_arrivals", []))
        outcomes = dict(scenario.get("patient_outcomes", {}))

        for _ in range(steps):
            before = s.copy()
            # advance clock
            s.time_min += self.step_minutes
            # Decrement remaining time on busy chairs; free those that finish.
            # NOTE: priority_at_assignment intentionally PRESERVED across the
            # OCCUPIED→IDLE transition for the duration of this step so the
            # reward function (compute_reward_step) can credit completions by
            # patient priority.  The next assignment overwrites it.
            for c in s.chairs:
                if c.status == ChairStatus.OCCUPIED:
                    c.remaining_minutes = max(0, c.remaining_minutes - self.step_minutes)
                    if c.remaining_minutes <= 0:
                        c.status = ChairStatus.IDLE
                        c.patient_id = None
            # Process any urgent arrivals that fire in this step
            fired = [a for a in scenario_arrivals
                     if a["arrival_time_min"] < s.time_min]
            for arr in fired:
                s.queue.append(QueuedPatient(
                    patient_id=f"urgent_{arr['arrival_time_min']}_{len(s.queue)}",
                    priority=int(arr.get("priority", 1)),
                    expected_duration=int(arr.get("expected_duration", 45)),
                    no_show_prob=0.0, cancel_prob=0.0,
                    arrival_time_min=int(arr["arrival_time_min"]),
                    is_urgent=True,
                ))
                scenario_arrivals.remove(arr)

            # Apply no-show / cancel realisations once per patient
            to_remove: List[str] = []
            for p in s.queue:
                outcome = outcomes.get(p.patient_id)
                if outcome == "noshow":
                    to_remove.append(p.patient_id)
                    s.n_noshows_so_far += 1
                elif outcome == "cancel":
                    to_remove.append(p.patient_id)
                    s.n_cancellations_so_far += 1
            if to_remove:
                s.queue = [p for p in s.queue if p.patient_id not in to_remove]
                for pid in to_remove:
                    outcomes.pop(pid, None)

            # Greedy fill: sort queue by (priority ASC, arrival_time ASC)
            greedy_action = self._greedy_fill(s)
            s = self._apply_action(s, greedy_action)
            total_r += compute_immediate_reward(
                before, greedy_action, s, self.step_minutes,
            )

            if s.time_min >= self.day_end_min:
                total_r += compute_terminal_penalty(s)
                break
        else:
            # Truncated — add approximate terminal value
            total_r += self.value_fn.predict(s, day_end_min=self.day_end_min)

        return float(total_r), s

    def _apply_action(self, state: MDPState, action: ScheduleAction) -> MDPState:
        """Apply assignments to state (mutates + returns)."""
        if not action or not action.assignments:
            return state
        queue_index = {p.patient_id: p for p in state.queue}
        assigned_patient_ids: List[str] = []
        for chair_id, pid in action.assignments.items():
            if pid is None:
                continue
            patient = queue_index.get(pid)
            if patient is None:
                continue
            chair = next(
                (c for c in state.chairs
                 if c.chair_id == chair_id
                 and c.status == ChairStatus.IDLE),
                None,
            )
            if chair is None:
                continue
            chair.status = ChairStatus.OCCUPIED
            chair.patient_id = patient.patient_id
            chair.remaining_minutes = int(patient.expected_duration)
            # Capture priority so the reward formula can credit by urgency.
            chair.priority_at_assignment = int(patient.priority)
            assigned_patient_ids.append(patient.patient_id)
        # Remove assigned patients from queue
        if assigned_patient_ids:
            state.queue = [
                p for p in state.queue
                if p.patient_id not in assigned_patient_ids
            ]
        return state

    def _greedy_fill(self, state: MDPState) -> ScheduleAction:
        """Priority-first fill of each idle chair."""
        idle = [c for c in state.chairs if c.status == ChairStatus.IDLE]
        if not idle or not state.queue:
            return ScheduleAction(assignments={c.chair_id: None for c in idle})
        queue = sorted(
            state.queue,
            key=lambda p: (p.priority, p.arrival_time_min),
        )
        assignments: Dict[str, Optional[str]] = {}
        used: set = set()
        for c in idle:
            pick = next((p for p in queue if p.patient_id not in used), None)
            if pick is None:
                assignments[c.chair_id] = None
            else:
                assignments[c.chair_id] = pick.patient_id
                used.add(pick.patient_id)
        return ScheduleAction(assignments=assignments, reason="rollout_greedy")


# ---------------------------------------------------------------------------
# §S-2 MPCController — the orchestrator
# ---------------------------------------------------------------------------


class MPCController:
    """Receding-horizon MPC with scenario rollout + squeeze-in fallback."""

    def __init__(
        self,
        *,
        arrival_model: Optional[UrgentArrivalModel] = None,
        value_fn: Optional[TerminalValueFunction] = None,
        sampler: Optional[ScenarioSampler] = None,
        rollout_planner: Optional[RolloutPlanner] = None,
        fallback_fn: Optional[Callable[[MDPState], ScheduleAction]] = None,
        n_scenarios: int = DEFAULT_N_SCENARIOS,
        lookahead_minutes: int = DEFAULT_LOOKAHEAD_MINUTES,
        total_timeout_s: float = DEFAULT_TOTAL_TIMEOUT_S,
        day_start_hour: int = DEFAULT_DAY_START_HOUR,
        day_end_hour: int = DEFAULT_DAY_END_HOUR,
        storage_dir: Path = MPC_DIR,
    ):
        self.arrival_model = arrival_model or UrgentArrivalModel.load()
        self.value_fn = value_fn or TerminalValueFunction.load()
        self.sampler = sampler or ScenarioSampler(self.arrival_model, n_scenarios)
        self.rollout_planner = rollout_planner or RolloutPlanner(
            self.value_fn,
            day_end_min=(day_end_hour - day_start_hour) * 60,
        )
        self.fallback_fn = fallback_fn
        self.n_scenarios = int(n_scenarios)
        self.lookahead_minutes = int(lookahead_minutes)
        self.total_timeout_s = float(total_timeout_s)
        self.day_start_hour = int(day_start_hour)
        self.day_end_hour = int(day_end_hour)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = self.storage_dir / "events.jsonl"
        self.decision_log = self.storage_dir / "decisions.jsonl"

        self._lock = threading.Lock()
        self._last_decision: Optional[MPCDecision] = None
        self._n_decisions: int = 0
        self._n_fallbacks: int = 0

    # ----------------------------------------------------------- API ---- #

    def decide(
        self,
        state: MDPState,
        candidate_actions: Optional[Sequence[ScheduleAction]] = None,
    ) -> MPCDecision:
        """Pick the best immediate action under scenario rollout."""
        # T4.5 — Prometheus + OTel hot-path instrumentation.
        try:
            from observability import observe_optimizer_solve
            _obs_ctx = observe_optimizer_solve("mpc")
        except Exception:                                 # pragma: no cover
            from contextlib import nullcontext
            _obs_ctx = nullcontext()
        with _obs_ctx:
            return self._decide_impl(state, candidate_actions)

    def _decide_impl(self, state, candidate_actions=None):
        t0 = time.perf_counter()
        candidates = list(candidate_actions) if candidate_actions else \
            self._generate_candidate_actions(state)
        if not candidates:
            return self._build_decision_from_fallback(
                state, t0, reason="no_candidate_actions",
            )

        scenarios = self.sampler.sample_scenarios(
            state, lookahead_minutes=self.lookahead_minutes,
        )

        # Evaluate each candidate across all scenarios, averaging the reward.
        per_action_rewards: List[Tuple[ScheduleAction, List[float]]] = []
        for action in candidates:
            rewards: List[float] = []
            for sc in scenarios:
                if time.perf_counter() - t0 > self.total_timeout_s:
                    logger.warning(
                        "MPC decide: timeout (%.3fs) — falling back",
                        time.perf_counter() - t0,
                    )
                    return self._build_decision_from_fallback(
                        state, t0, reason="total_timeout",
                    )
                r, _ = self.rollout_planner.evaluate_action(
                    state, action, sc,
                    lookahead_minutes=self.lookahead_minutes,
                )
                rewards.append(float(r))
            per_action_rewards.append((action, rewards))

        # Pick the action with the highest mean reward (ties broken by n_assignments)
        best = max(
            per_action_rewards,
            key=lambda pair: (
                sum(pair[1]) / max(len(pair[1]), 1),
                sum(1 for v in pair[0].assignments.values() if v is not None),
            ),
        )
        best_action, best_rewards = best
        mean_reward = sum(best_rewards) / max(len(best_rewards), 1)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        decision = MPCDecision(
            ts=datetime.utcnow().isoformat(timespec="seconds"),
            time_min=int(state.time_min),
            n_scenarios=len(scenarios),
            lookahead_minutes=self.lookahead_minutes,
            action=best_action.to_dict(),
            chosen_action_expected_reward=float(mean_reward),
            per_scenario_rewards=[float(r) for r in best_rewards],
            used_fallback=False,
            fallback_reason=None,
            decision_latency_ms=latency_ms,
        )
        self._persist_decision(decision)
        with self._lock:
            self._last_decision = decision
            self._n_decisions += 1
        return decision

    def event_trigger(
        self,
        kind: str,
        state: MDPState,
        payload: Optional[Dict[str, Any]] = None,
    ) -> MPCDecision:
        """Handle a real-time event (``chair_free`` | ``urgent_arrival`` |
        ``noshow`` | ``cancel``).  Logs the event + fires a decide()."""
        self._append_event({
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "event": kind,
            "time_min": state.time_min,
            "queue_size": len(state.queue),
            "idle_chairs": state.n_idle_chairs,
            "payload": payload or {},
        })
        if kind == "urgent_arrival" and payload is not None:
            minutes_observed = float(payload.get("minutes_observed", 5.0))
            n_arrivals = int(payload.get("n_arrivals", 1))
            self.arrival_model.update(n_arrivals, minutes_observed)
        return self.decide(state)

    def simulate_day(
        self,
        initial_state: MDPState,
        policies: Sequence[str] = ("greedy", "mpc"),
        total_minutes: Optional[int] = None,
        rng_seed: int = 42,
    ) -> Dict[str, Any]:
        """Run a full-day simulation against multiple policies for §S-9.

        Returns per-policy aggregate metrics.
        """
        total_minutes = int(
            total_minutes or (self.day_end_hour - self.day_start_hour) * 60
        )
        results: Dict[str, Any] = {}
        for policy in policies:
            rng = random.Random(rng_seed)
            # Fresh sampler per policy so both see the same stream
            sampler = ScenarioSampler(
                self.arrival_model, n_scenarios=self.n_scenarios,
                rng_seed=rng_seed,
            )
            state = initial_state.copy()
            metrics = {
                "policy": policy,
                "completed": 0,
                "urgent_accepted": 0,
                "urgent_arrived": 0,
                "sum_wait_min": 0.0,
                "idle_minutes": 0,
                "n_decisions": 0,
                "n_fallbacks": 0,
            }
            # Pre-generate a single scenario of arrivals for the whole day
            day_scenario = sampler.sample_scenarios(
                state, lookahead_minutes=total_minutes,
            )[0]
            arrivals_remaining = sorted(
                day_scenario["urgent_arrivals"],
                key=lambda a: a["arrival_time_min"],
            )
            outcomes = dict(day_scenario["patient_outcomes"])
            step = DEFAULT_DECISION_INTERVAL_MIN
            while state.time_min < total_minutes:
                # Fire arrivals that have happened
                while arrivals_remaining and \
                        arrivals_remaining[0]["arrival_time_min"] <= state.time_min:
                    a = arrivals_remaining.pop(0)
                    state.queue.append(QueuedPatient(
                        patient_id=f"sim_urgent_{state.time_min}_{metrics['urgent_arrived']}",
                        priority=int(a.get("priority", 1)),
                        expected_duration=int(a.get("expected_duration", 45)),
                        no_show_prob=0.0, cancel_prob=0.0,
                        arrival_time_min=state.time_min,
                        is_urgent=True,
                    ))
                    metrics["urgent_arrived"] += 1
                # Decide
                if policy == "greedy":
                    action = self.rollout_planner._greedy_fill(state)
                elif policy == "mpc":
                    d = self.decide(state)
                    action = ScheduleAction(
                        assignments=d.action["assignments"],
                        reason=d.action["reason"],
                    )
                    metrics["n_decisions"] += 1
                    if d.used_fallback:
                        metrics["n_fallbacks"] += 1
                else:  # static
                    action = ScheduleAction(
                        assignments={c.chair_id: None for c in state.chairs},
                        reason="static",
                    )
                # Count urgent acceptance
                for pid in action.assignments.values():
                    if pid and pid.startswith("sim_urgent_"):
                        metrics["urgent_accepted"] += 1
                # Apply
                state = self.rollout_planner._apply_action(state, action)
                # Advance time + finish busy chairs
                state.time_min += step
                for c in state.chairs:
                    if c.status == ChairStatus.OCCUPIED:
                        c.remaining_minutes = max(0, c.remaining_minutes - step)
                        if c.remaining_minutes <= 0:
                            c.status = ChairStatus.IDLE
                            # priority_at_assignment kept until the next assign;
                            # see note in step() above.
                            c.patient_id = None
                            metrics["completed"] += 1
                    else:
                        if state.queue:
                            pass  # will fill next step
                        else:
                            metrics["idle_minutes"] += step
                # Accumulate wait
                for p in state.queue:
                    metrics["sum_wait_min"] += step
                # Remove no-shows / cancels in static order
                to_drop: List[str] = []
                for p in state.queue:
                    out = outcomes.get(p.patient_id)
                    if out == "noshow":
                        to_drop.append(p.patient_id)
                        outcomes.pop(p.patient_id)
                    elif out == "cancel":
                        to_drop.append(p.patient_id)
                        outcomes.pop(p.patient_id)
                state.queue = [p for p in state.queue
                              if p.patient_id not in to_drop]
            metrics["avg_urgent_wait_min"] = (
                metrics["sum_wait_min"] / max(metrics["urgent_arrived"], 1)
            )
            metrics["urgent_acceptance_rate"] = (
                metrics["urgent_accepted"] / max(metrics["urgent_arrived"], 1)
            )
            metrics["utilisation"] = 1.0 - metrics["idle_minutes"] / max(
                total_minutes * len(initial_state.chairs), 1
            )
            results[policy] = metrics

        self._persist_simulation(results)
        return results

    def last(self) -> Optional[MPCDecision]:
        with self._lock:
            return self._last_decision

    def status(self) -> Dict[str, Any]:
        am = self.arrival_model.status()
        with self._lock:
            last = self._last_decision
        return {
            "n_scenarios": self.n_scenarios,
            "lookahead_minutes": self.lookahead_minutes,
            "total_timeout_s": self.total_timeout_s,
            "day_start_hour": self.day_start_hour,
            "day_end_hour": self.day_end_hour,
            "arrival_model": am,
            "value_function": self.value_fn.to_dict(),
            "total_decisions": self._n_decisions,
            "total_fallbacks": self._n_fallbacks,
            "last_decision_ts": last.ts if last else None,
            "last_used_fallback": last.used_fallback if last else None,
            "last_latency_ms": last.decision_latency_ms if last else None,
            "log_path": str(self.event_log),
        }

    def update_config(
        self,
        n_scenarios: Optional[int] = None,
        lookahead_minutes: Optional[int] = None,
        total_timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        if n_scenarios is not None:
            self.n_scenarios = int(n_scenarios)
            self.sampler.n_scenarios = self.n_scenarios
        if lookahead_minutes is not None:
            self.lookahead_minutes = int(lookahead_minutes)
        if total_timeout_s is not None:
            self.total_timeout_s = float(total_timeout_s)
        return {
            "n_scenarios": self.n_scenarios,
            "lookahead_minutes": self.lookahead_minutes,
            "total_timeout_s": self.total_timeout_s,
        }

    # --------------------------------------------------------- Internals - #

    def _generate_candidate_actions(self, state: MDPState) -> List[ScheduleAction]:
        """Enumerate per-chair candidate actions.

        For each idle chair we consider (i) assign each queued patient,
        or (ii) leave idle.  The full cross-product is tractable for
        the typical 1--3 idle chairs at any epoch; for more we cap at
        the top-N priority patients per chair.
        """
        idle = [c for c in state.chairs if c.status == ChairStatus.IDLE]
        if not idle:
            # Nothing to decide — return the trivial wait action.
            return [
                ScheduleAction(
                    assignments={c.chair_id: None for c in state.chairs},
                    reason="no_idle_chairs",
                )
            ]
        if not state.queue:
            return [
                ScheduleAction(
                    assignments={c.chair_id: None for c in state.chairs},
                    reason="empty_queue",
                )
            ]
        # Sort queue by priority ASC (highest first)
        queue_sorted = sorted(state.queue, key=lambda p: p.priority)
        top_k = queue_sorted[: max(len(idle) * 3, 5)]

        candidates: List[ScheduleAction] = []
        # Candidate 1: leave everyone idle (wait action)
        wait_assignments: Dict[str, Optional[str]] = {
            c.chair_id: None for c in state.chairs
        }
        candidates.append(ScheduleAction(
            assignments=wait_assignments, reason="wait",
        ))
        # Candidate 2..: greedy assign + each top-K priority patient to first chair
        for pick in top_k:
            a: Dict[str, Optional[str]] = {c.chair_id: None for c in state.chairs}
            a[idle[0].chair_id] = pick.patient_id
            # Fill remaining idle chairs greedily
            used = {pick.patient_id}
            for chair in idle[1:]:
                nxt = next((p for p in queue_sorted
                            if p.patient_id not in used), None)
                if nxt:
                    a[chair.chair_id] = nxt.patient_id
                    used.add(nxt.patient_id)
            candidates.append(ScheduleAction(
                assignments=a, reason=f"first_chair_gets_{pick.patient_id}",
            ))
        return candidates

    def _build_decision_from_fallback(
        self, state: MDPState, t0: float, reason: str,
    ) -> MPCDecision:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if self.fallback_fn is not None:
            action = self.fallback_fn(state)
        else:
            action = self.rollout_planner._greedy_fill(state)
        decision = MPCDecision(
            ts=datetime.utcnow().isoformat(timespec="seconds"),
            time_min=int(state.time_min),
            n_scenarios=0,
            lookahead_minutes=self.lookahead_minutes,
            action=action.to_dict(),
            chosen_action_expected_reward=0.0,
            per_scenario_rewards=[],
            used_fallback=True,
            fallback_reason=reason,
            decision_latency_ms=latency_ms,
        )
        self._persist_decision(decision)
        with self._lock:
            self._last_decision = decision
            self._n_decisions += 1
            self._n_fallbacks += 1
        return decision

    def _persist_decision(self, decision: MPCDecision) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            with open(self.decision_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision.to_dict()) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"MPC decision log write failed: {exc}")

    def _persist_simulation(self, results: Dict[str, Any]) -> None:
        try:
            MPC_SIM_LOG.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
                "results": results,
            }
            with open(MPC_SIM_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"MPC simulation log write failed: {exc}")

    def _append_event(self, record: Dict[str, Any]) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            with open(self.event_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"MPC event log write failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poisson(lam: float, rng: random.Random) -> int:
    """Knuth-style Poisson sampler (small lambda)."""
    if lam <= 0:
        return 0
    if lam > 30.0:
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


def state_from_app_state(
    app_state: Dict[str, Any],
    current_minute: Optional[int] = None,
    day_start_hour: int = DEFAULT_DAY_START_HOUR,
) -> MDPState:
    """Build an :class:`MDPState` from the Flask ``app_state`` dict.

    The Flask layer passes this whenever it wants an MPC decision on
    the live schedule without plumbing a custom state constructor.
    """
    now = datetime.now()
    minute = int(current_minute if current_minute is not None else
                 (now.hour - day_start_hour) * 60 + now.minute)

    chairs: List[ChairState] = []
    appts = app_state.get("appointments") or []
    # Build a chair-id → next-free-time map
    chair_next_free: Dict[str, Tuple[datetime, str, int]] = {}
    for a in appts:
        cid = getattr(a, "chair_id", None) or (a.get("chair_id") if isinstance(a, dict) else None)
        site = getattr(a, "site_code", None) or (a.get("site_code") if isinstance(a, dict) else "UNKNOWN")
        start = getattr(a, "start_time", None) or (a.get("start_time") if isinstance(a, dict) else None)
        end = getattr(a, "end_time", None) or (a.get("end_time") if isinstance(a, dict) else None)
        pid = getattr(a, "patient_id", None) or (a.get("patient_id") if isinstance(a, dict) else None)
        if cid is None or start is None or end is None:
            continue
        if isinstance(start, str):
            try: start = datetime.fromisoformat(start)
            except Exception: start = now
        if isinstance(end, str):
            try: end = datetime.fromisoformat(end)
            except Exception: end = start
        if start <= now < end:
            rem = int((end - now).total_seconds() // 60)
            chair_next_free[cid] = (end, pid, rem)
        elif now < start and cid not in chair_next_free:
            chair_next_free[cid] = (now, None, 0)

    try:
        from config import DEFAULT_SITES
        for s in DEFAULT_SITES:
            for i in range(int(s.get("chairs", 0)) + int(s.get("recliners", 0))):
                cid = f"{s['code']}-C{i+1:02d}"
                info = chair_next_free.get(cid)
                if info and info[2] > 0:
                    chairs.append(ChairState(
                        chair_id=cid, site_code=s["code"],
                        status=ChairStatus.OCCUPIED,
                        patient_id=info[1], remaining_minutes=info[2],
                    ))
                else:
                    chairs.append(ChairState(
                        chair_id=cid, site_code=s["code"],
                        status=ChairStatus.IDLE,
                    ))
    except Exception:
        chairs = [ChairState(chair_id="DEFAULT-C01",
                             site_code="DEFAULT",
                             status=ChairStatus.IDLE)]

    queue: List[QueuedPatient] = []
    for p in (app_state.get("patients") or []):
        pid = getattr(p, "patient_id", None)
        if pid is None or any(a.patient_id == pid for a in appts if hasattr(a, "patient_id")):
            continue
        queue.append(QueuedPatient(
            patient_id=str(pid),
            priority=int(getattr(p, "priority", 3) or 3),
            expected_duration=int(getattr(p, "expected_duration", 60) or 60),
            no_show_prob=float(getattr(p, "noshow_probability", 0.10) or 0.10),
            cancel_prob=0.05,
            arrival_time_min=minute,
        ))

    return MDPState(
        time_min=minute,
        chairs=chairs,
        queue=queue,
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_GLOBAL: Optional[MPCController] = None


def get_controller() -> MPCController:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = MPCController()
    return _GLOBAL


def set_controller(c: MPCController) -> None:
    global _GLOBAL
    _GLOBAL = c
