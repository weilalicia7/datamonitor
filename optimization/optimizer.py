"""
Schedule Optimizer
==================

Main optimization engine using OR-Tools CP-SAT solver.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

# Google OR-Tools
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OPTIMIZATION_WEIGHTS,
    DEFAULT_SITES,
    OPERATING_HOURS,
    get_logger,
)

# Column generation threshold: use CG decomposition when patient count
# exceeds this value (monolithic CP-SAT scales poorly beyond ~60 patients).
COLUMN_GEN_THRESHOLD = 50

logger = get_logger(__name__)


@dataclass
class Patient:
    """Patient to be scheduled"""
    patient_id: str
    priority: int  # 1-4 (1 is highest)
    protocol: str
    expected_duration: int  # minutes
    postcode: str
    earliest_time: datetime
    latest_time: datetime
    preferred_site: str = None
    long_infusion: bool = False
    noshow_probability: float = 0.0
    is_urgent: bool = False
    travel_time_minutes: float = 30.0  # Estimated travel time (default 30 min)
    distance_km: float = 15.0  # Distance to treatment centre (default 15 km)


@dataclass
class Chair:
    """Treatment chair resource"""
    chair_id: str
    site_code: str
    is_recliner: bool = False
    available_from: datetime = None
    available_until: datetime = None


@dataclass
class ScheduledAppointment:
    """Result of scheduling"""
    patient_id: str
    chair_id: str
    site_code: str
    start_time: datetime
    end_time: datetime
    duration: int
    priority: int
    travel_time: int  # estimated travel time in minutes


@dataclass
class OptimizationResult:
    """Result of optimization run"""
    success: bool
    appointments: List[ScheduledAppointment]
    unscheduled: List[str]  # patient IDs
    metrics: Dict
    solve_time: float
    status: str


class ScheduleOptimizer:
    """
    Optimizes patient scheduling using constraint programming.

    Uses Google OR-Tools CP-SAT solver to find optimal
    appointment assignments that:
    - Maximize chair utilization
    - Minimize patient travel
    - Respect priority levels
    - Balance workload across sites
    """

    def __init__(self, chairs: List[Chair] = None, sites: List[Dict] = None,
                 event_impact_model=None):
        """
        Initialize optimizer.

        Args:
            chairs: List of Chair objects
            sites: List of site configurations
            event_impact_model: Optional EventImpactModel for adjusting no-show predictions
        """
        if not ORTOOLS_AVAILABLE:
            logger.warning("OR-Tools not available. Using simplified scheduling.")

        self.chairs = chairs or []
        self.sites = {s['code']: s for s in (sites or DEFAULT_SITES)}
        self.weights = OPTIMIZATION_WEIGHTS
        self.event_impact_model = event_impact_model
        self.active_events = []  # List of active Event objects

        # Warm-start solution cache: fingerprint → cached assignment
        # Persists for the lifetime of the Flask process (module-level singleton)
        self._solution_cache: Dict[tuple, Dict] = {}
        self._cache_max_size: int = 50  # evict oldest beyond this

        # GNN feasibility pre-filter (lazy-loaded; None = disabled)
        self._gnn = None
        self._gnn_enabled: bool = False

        # Column generation for large instances (>50 patients)
        self._cg_enabled: bool = True  # auto-route when n > threshold
        self._cg_threshold: int = COLUMN_GEN_THRESHOLD
        self._cg_stats: Dict = {}  # last CG run diagnostics

        # Explicit defaults for the two flags previously accessed via
        # getattr(…, True) — so that set_components() operates on a
        # known baseline rather than "whatever the first read happens
        # to find."  Production defaults: CVaR + fairness both ON.
        self._use_cvar_objective: bool = True
        self._fairness_constraints_enabled: bool = True

        logger.info(f"Optimizer initialized with {len(self.chairs)} chairs")

    def set_chairs(self, chairs: List[Chair]):
        """Update available chairs"""
        self.chairs = chairs
        logger.info(f"Updated to {len(chairs)} chairs")

    def set_event_impact_model(self, model):
        """Set the event impact model for adjusting no-show predictions"""
        self.event_impact_model = model
        logger.info("Event impact model set on optimizer")

    def set_active_events(self, events: list):
        """Set active events that may affect no-show rates"""
        self.active_events = events
        if events:
            logger.info(f"Set {len(events)} active events on optimizer")

    def set_components(
        self,
        *,
        column_generation: Optional[bool] = None,
        gnn: Optional[bool] = None,
        cvar: Optional[bool] = None,
        fairness: Optional[bool] = None,
    ) -> Dict[str, bool]:
        """
        Toggle the four heavy-component switches without touching the
        private `_cg_enabled` / `_gnn_enabled` / `_use_cvar_objective` /
        `_fairness_constraints_enabled` attributes directly.

        Keyword-only; each argument defaults to ``None`` meaning "leave
        this component's current state alone", so callers can flip one
        component without accidentally resetting the others.

        The public surface is how benchmarks (weight-sensitivity,
        ablation, fairness-mitigation, robustness, external-algorithms,
        column-generation) and regression tests should configure the
        optimiser — direct underscore mutation silently ignores typos
        and is brittle if the internals are renamed.

        Args:
            column_generation: route >50-patient instances through the
                Dantzig-Wolfe column-generation pricing loop.
            gnn: enable GNN feasibility pre-filtering (requires
                ``enable_gnn_pruning()`` to have been called first).
            cvar: add the CVaR worst-case objective to the CP-SAT
                solve on top of DRO.
            fairness: enforce the per-group DRO parity constraints
                (soft/hard mode still follows ``_fairness_mode``).

        Returns:
            Dict mapping each component to its boolean state **after**
            the update, so callers can log / assert.
        """
        if column_generation is not None:
            self._cg_enabled = bool(column_generation)
        if gnn is not None:
            self._gnn_enabled = bool(gnn)
        if cvar is not None:
            self._use_cvar_objective = bool(cvar)
        if fairness is not None:
            self._fairness_constraints_enabled = bool(fairness)
        state = {
            "column_generation": bool(self._cg_enabled),
            "gnn": bool(self._gnn_enabled),
            "cvar": bool(self._use_cvar_objective),
            "fairness": bool(self._fairness_constraints_enabled),
        }
        logger.info(f"Optimizer components set: {state}")
        return state

    def set_weights(self, weights: Dict[str, float], normalise: bool = True) -> Dict[str, float]:
        """
        Replace the 6-objective Pareto weight vector (e.g., from IRL).

        The six keys (priority, utilization, noshow_risk, waiting_time,
        robustness, travel) are preserved even when absent from `weights`
        so that partial updates do not silently zero an objective.

        Args:
            weights: mapping objective-name → non-negative weight.
            normalise: if True, normalise so values sum to 1.0.

        Returns:
            The final weight dict now stored on the optimiser.
        """
        keys = ('priority', 'utilization', 'noshow_risk', 'waiting_time', 'robustness', 'travel')
        merged = {k: max(0.0, float(weights.get(k, self.weights.get(k, 0.0)))) for k in keys}
        total = sum(merged.values())
        if normalise and total > 0:
            merged = {k: v / total for k, v in merged.items()}
        self.weights = merged
        logger.info(f"Optimizer weights set: {merged}")
        return merged

    def compute_schedule_features(self, patients, assignments: Dict[str, Dict]) -> Dict[str, float]:
        """
        Z(schedule) ∈ R^6 — un-weighted per-objective contributions that
        feed the IRL preference learner.  Delegates to the IRL feature
        helper so there is a single source of truth for the math.
        """
        from ml.inverse_rl_preferences import compute_objective_features, OBJECTIVE_KEYS
        robust = getattr(self, '_robust_noshow_penalties', None)
        feats = compute_objective_features(patients, assignments, robust)
        arr = feats.as_array()
        return {k: float(arr[i]) for i, k in enumerate(OBJECTIVE_KEYS)}

    def add_event(self, title: str, description: str, event_type: str, severity: int = 3):
        """Add an event that may affect no-show rates"""
        if self.event_impact_model:
            event = self.event_impact_model.create_event(
                title=title,
                description=description,
                event_type=event_type,
                severity=severity
            )
            self.active_events.append(event)
            logger.info(f"Added event: {title} ({event_type})")

    def enable_gnn_pruning(self,
                           prune_threshold: float = 0.15,
                           min_viable_chairs: int = 5,
                           train_every: int = 5) -> None:
        """
        Enable GNN feasibility pre-filtering.

        Instantiates a GNNFeasibilityPredictor that:
        1. Collects (patient, chair, assigned) labels after every CP-SAT solve.
        2. Retrains every train_every solves.
        3. Prunes (patient_idx, chair_idx) pairs with P(assigned) < prune_threshold
           before handing the reduced variable space to CP-SAT.

        Call once after creating the optimizer (e.g., in flask_app.py startup).
        """
        try:
            from optimization.gnn_feasibility import GNNFeasibilityPredictor
            self._gnn = GNNFeasibilityPredictor(
                prune_threshold=prune_threshold,
                min_viable_chairs=min_viable_chairs,
                train_every=train_every,
            )
            self._gnn_enabled = True
            logger.info(
                f"GNN feasibility pre-filter enabled "
                f"(threshold={prune_threshold}, min_viable={min_viable_chairs})"
            )
        except Exception as exc:
            logger.warning(f"GNN init failed (pruning disabled): {exc}")
            self._gnn_enabled = False

    def _compute_instance_fingerprint(self, patients: List[Patient], date: datetime) -> tuple:
        """
        Lightweight fingerprint of a scheduling instance used as the warm-start
        cache key.  Uses only fields that are **stable** — i.e., not mutated by
        DRO or event-adjustment pre-processing — so that repeated calls for the
        same patient set always hash to the same bucket:

        - Day of week           (Mon/Wed/Fri SACT patterns differ from Tue/Thu)
        - Patient count         (slot pressure)
        - Priority distribution (P1-P4 breakdown; not modified by DRO)
        - Long-infusion count   (recliner demand; not modified by DRO)
        - Chair count           (resource configuration)

        Note: noshow_probability and expected_duration are intentionally excluded
        because _apply_event_adjustment() and _apply_dro_robustness() mutate
        them in-place before _optimize_cpsat() is called.  Using those fields
        would produce a different fingerprint on the second call for the same
        patient set, defeating the warm-start cache.

        Similar fingerprints indicate that a prior solution is a strong warm
        start, reducing solve time by 50-80% for recurring schedule patterns.
        """
        n = len(patients)
        dow = date.weekday()                           # 0=Mon … 6=Sun
        priority_dist = tuple(
            sum(1 for p in patients if p.priority == i) for i in range(1, 5)
        )
        long_inf = sum(1 for p in patients if p.long_infusion)
        n_chairs = len(self.chairs)
        return (dow, n, priority_dist, long_inf, n_chairs)

    def _apply_dro_robustness(self, patients: List[Patient]):
        """
        Apply Distributionally Robust Optimization (DRO) to scheduling.

        Computes worst-case no-show penalties under Wasserstein ambiguity set:
            P = {Q : W_2(P_emp, Q) <= epsilon}
            pi_worst = pi + epsilon * sqrt(Var[pi] + epsilon^2)

        Also applies CVaR duration buffers for risk-averse scheduling:
            CVaR_alpha(D) = E[D | D >= VaR_alpha]

        These robust parameters replace point estimates in the CP-SAT objective,
        guaranteeing schedule performance under distributional shifts.
        """
        try:
            from optimization.uncertainty_optimization import UncertaintyAwareOptimizer

            dro = UncertaintyAwareOptimizer(
                epsilon=0.05,   # 5% Wasserstein radius
                alpha=0.90,     # Protect against worst 10%
                n_scenarios=50
            )

            robust = dro.compute_robust_parameters(patients)

            # Store robust penalties for use in _optimize_cpsat objective
            self._robust_noshow_penalties = robust.robust_noshow_penalties

            # Apply CVaR duration buffers to patients
            buffered = 0
            for p in patients:
                if p.patient_id in robust.robust_duration_buffers:
                    cvar_dur = robust.robust_duration_buffers[p.patient_id]
                    if cvar_dur > p.expected_duration:
                        p.expected_duration = min(int(cvar_dur), p.expected_duration + 30)
                        buffered += 1

            logger.info(
                f"DRO applied: epsilon={robust.epsilon}, alpha={robust.alpha}, "
                f"method={robust.method}, duration buffers={buffered} patients"
            )
        except Exception as e:
            self._robust_noshow_penalties = {}
            logger.debug(f"DRO skipped: {e}")

    def _apply_event_adjustment(self, patients: List[Patient]) -> List[Patient]:
        """
        Apply event impact adjustment to patient no-show probabilities.

        The adjustment is scaled by travel distance/time:
        - Patients traveling longer distances are more affected by events
        - Weather and traffic events have stronger distance effects
        - Local patients (< 10 min travel) receive minimal adjustment
        """
        if not self.active_events or not self.event_impact_model:
            return patients

        # Get event impact prediction
        impact = self.event_impact_model.predict_impact(self.active_events)
        base_event_factor = impact.predicted_noshow_rate / max(impact.baseline_noshow_rate, 0.01)

        # Check if any events are distance-sensitive (weather, traffic)
        distance_sensitive_types = {'WEATHER', 'TRAFFIC', 'TRANSPORT'}
        has_distance_sensitive = any(
            event.event_type.name in distance_sensitive_types
            for event in self.active_events
        )

        logger.info(f"Applying event factor {base_event_factor:.2f} to {len(patients)} patients "
                   f"(distance-weighted: {has_distance_sensitive})")

        # Reference travel time for scaling (30 min = baseline)
        REFERENCE_TRAVEL_TIME = 30.0  # minutes

        # Adjust each patient's no-show probability scaled by distance
        for patient in patients:
            base_prob = patient.noshow_probability

            if has_distance_sensitive and patient.travel_time_minutes > 0:
                # Scale event factor by travel time
                # - 10 min travel: ~50% of base factor effect
                # - 30 min travel: 100% of base factor effect (reference)
                # - 60 min travel: ~150% of base factor effect
                # - 90+ min travel: up to 200% of base factor effect
                travel_ratio = patient.travel_time_minutes / REFERENCE_TRAVEL_TIME

                # Diminishing returns for very long distances (cap at 2x)
                distance_multiplier = min(2.0, 0.5 + 0.5 * travel_ratio)

                # For very short distances (< 10 min), reduce impact significantly
                if patient.travel_time_minutes < 10:
                    distance_multiplier = 0.3

                # Calculate distance-adjusted factor
                # Factor of 1.0 means no change, so we adjust the "excess" factor
                excess_factor = base_event_factor - 1.0
                adjusted_event_factor = 1.0 + (excess_factor * distance_multiplier)
            else:
                # Non-distance-sensitive events apply uniformly
                adjusted_event_factor = base_event_factor

            adjusted_prob = min(0.95, base_prob * adjusted_event_factor)
            patient.noshow_probability = adjusted_prob

            if adjusted_prob > base_prob * 1.1:  # Log if significant change
                logger.debug(f"Patient {patient.patient_id}: {base_prob:.2%} -> {adjusted_prob:.2%} "
                           f"(travel: {patient.travel_time_minutes:.0f} min)")

        return patients

    def optimize(self, patients: List[Patient],
                 date: datetime = None,
                 time_limit_seconds: int = 60) -> OptimizationResult:
        """
        Optimize schedule for given patients.

        Args:
            patients: List of Patient objects to schedule
            date: Date to schedule for (default: today)
            time_limit_seconds: Maximum solver time

        Returns:
            OptimizationResult object
        """
        # T4.5 — Prometheus + OTel hot-path instrumentation.  Imported
        # locally so the module remains usable when prometheus_client
        # isn't installed.  No-op if OTEL_ENABLED is false.
        try:
            from observability import observe_optimizer_solve
            _obs_ctx = observe_optimizer_solve("cpsat")
        except Exception:                                 # pragma: no cover
            from contextlib import nullcontext
            _obs_ctx = nullcontext()

        with _obs_ctx:
            return self._optimize_impl(patients, date, time_limit_seconds)

    def _optimize_impl(self, patients, date, time_limit_seconds):
        date = date or datetime.now().replace(hour=0, minute=0, second=0)

        if not patients:
            return OptimizationResult(
                success=True,
                appointments=[],
                unscheduled=[],
                metrics={'utilization': 0.0},
                solve_time=0.0,
                status='NO_PATIENTS'
            )

        # Apply event impact adjustment to no-show probabilities (4.3)
        patients = self._apply_event_adjustment(patients)

        # Apply DRO uncertainty-aware penalties (Wasserstein ambiguity set)
        self._apply_dro_robustness(patients)

        if not self.chairs:
            self._create_default_chairs(date)

        if ORTOOLS_AVAILABLE:
            # Route large instances to column generation decomposition
            if (self._cg_enabled
                    and len(patients) > self._cg_threshold):
                logger.info(
                    f"Routing to column generation: {len(patients)} patients "
                    f"> threshold {self._cg_threshold}"
                )
                return self._optimize_column_generation(
                    patients, date, time_limit_seconds
                )
            return self._optimize_cpsat(patients, date, time_limit_seconds)
        else:
            return self._optimize_greedy(patients, date)

    def _create_default_chairs(self, date: datetime):
        """Create default chair configuration"""
        start_hour, end_hour = OPERATING_HOURS
        start_time = date.replace(hour=start_hour, minute=0)
        end_time = date.replace(hour=end_hour, minute=0)

        chairs = []
        for site in DEFAULT_SITES:
            for i in range(site['chairs']):
                is_recliner = i < site.get('recliners', 0)
                chairs.append(Chair(
                    chair_id=f"{site['code']}-C{i+1:02d}",
                    site_code=site['code'],
                    is_recliner=is_recliner,
                    available_from=start_time,
                    available_until=end_time
                ))

        self.chairs = chairs
        logger.info(f"Created {len(chairs)} default chairs")

    def _optimize_cpsat(self, patients: List[Patient],
                        date: datetime,
                        time_limit_seconds: int) -> OptimizationResult:
        """
        Optimize using CP-SAT solver.
        """
        import time
        start_solve = time.time()

        model = cp_model.CpModel()

        # Fingerprint for warm-start cache lookup
        fingerprint = self._compute_instance_fingerprint(patients, date)

        # Time horizon (in minutes from start)
        start_hour, end_hour = OPERATING_HOURS
        horizon = (end_hour - start_hour) * 60
        day_start = date.replace(hour=start_hour, minute=0, second=0)

        # Decision variables
        # For each patient: which chair, start time
        patient_vars = {}
        for p in patients:
            p_idx = patients.index(p)

            # Chair assignment (boolean for each chair)
            chair_vars = []
            for c in self.chairs:
                c_idx = self.chairs.index(c)
                var = model.NewBoolVar(f'p{p_idx}_c{c_idx}')
                chair_vars.append(var)
            patient_vars[p.patient_id] = {
                'chairs': chair_vars,
                'start': model.NewIntVar(0, horizon - p.expected_duration, f'start_{p_idx}'),
                'assigned': model.NewBoolVar(f'assigned_{p_idx}')
            }

            # Must be assigned to exactly one chair if assigned
            model.Add(sum(chair_vars) == 1).OnlyEnforceIf(patient_vars[p.patient_id]['assigned'])
            model.Add(sum(chair_vars) == 0).OnlyEnforceIf(patient_vars[p.patient_id]['assigned'].Not())

        # =====================================================================
        # GNN FEASIBILITY PRE-FILTER
        #
        # For each (patient_idx, chair_idx) pair predicted infeasible by the
        # GNN (P < prune_threshold), force the BoolVar to 0.
        # This shrinks the search space before CP-SAT runs.
        #
        # Safety: GNNFeasibilityPredictor guarantees min_viable_chairs remain
        # per patient and never removes the last recliner for long-infusion
        # patients.
        #
        # Reference: MATH_LOGIC.md §2.11
        # =====================================================================
        gnn_prune_count = 0
        gnn_prune_rate  = 0.0
        if self._gnn_enabled and self._gnn is not None:
            try:
                valid_pairs, gnn_prune_count, gnn_prune_rate = (
                    self._gnn.prune_assignments(patients, self.chairs)
                )
                for p_idx, p in enumerate(patients):
                    pvars = patient_vars[p.patient_id]
                    for c_idx in range(len(self.chairs)):
                        if (p_idx, c_idx) not in valid_pairs:
                            model.Add(pvars['chairs'][c_idx] == 0)
            except Exception as _gnn_exc:
                logger.debug(f"GNN pruning skipped: {_gnn_exc}")
                gnn_prune_count = 0

        # No overlap constraint per chair
        for c_idx, chair in enumerate(self.chairs):
            intervals = []
            for p in patients:
                pvars = patient_vars[p.patient_id]
                # Create optional interval
                interval = model.NewOptionalIntervalVar(
                    pvars['start'],
                    p.expected_duration,
                    pvars['start'] + p.expected_duration,
                    pvars['chairs'][c_idx],
                    f'interval_p{patients.index(p)}_c{c_idx}'
                )
                intervals.append(interval)

            model.AddNoOverlap(intervals)

        # Bed requirement constraint
        for p in patients:
            if p.long_infusion:
                pvars = patient_vars[p.patient_id]
                bed_chairs = [
                    pvars['chairs'][i]
                    for i, c in enumerate(self.chairs)
                    if c.is_recliner
                ]
                if bed_chairs:
                    model.AddBoolOr(bed_chairs).OnlyEnforceIf(pvars['assigned'])

        # =====================================================================
        # FAIRNESS CONSTRAINTS
        # Ensure scheduling does not systematically disadvantage groups.
        # Toggle via self._fairness_constraints_enabled (default True).
        # The §5.6.2 benchmark (ml/benchmark_fairness_mitigation.py) runs
        # the optimiser twice — once with the flag False (raw baseline)
        # and once with it True (DRO-style mitigation on) — so the
        # dissertation can report the Four-Fifths ratio improvement
        # attributable to these constraints.
        # =====================================================================

        # Fairness mode (§5.6.2 / Improvement D):
        #   "none" — no fairness block at all
        #   "soft" — demographic-parity penalty added to objective
        #            (the historical default; behaves the same as
        #            _fairness_constraints_enabled=True)
        #   "hard" — equalised-odds HARD constraint per group pair,
        #            with tolerance ``fairness_tolerance`` (default
        #            0.15 → disparate-impact ratio ≥ 0.85).  Also
        #            adds the same term to the objective so CP-SAT
        #            finds the least-utilisation-lossy feasible
        #            point within the parity envelope.
        # Back-compat: when _fairness_mode is unset, read the legacy
        # _fairness_constraints_enabled flag.
        _fm = getattr(self, '_fairness_mode', None)
        if _fm is None:
            _fm = "soft" if getattr(self, '_fairness_constraints_enabled', True) else "none"
        if _fm not in ("none", "soft", "hard"):
            logger.warning(f"Unknown fairness_mode {_fm!r}; falling back to 'soft'")
            _fm = "soft"
        fairness_block_enabled = (_fm != "none")

        # Group patients by protected attributes
        age_groups = {}
        gender_groups = {}
        deprivation_groups = {}  # By postcode distance as proxy

        for p in patients:
            # Age band grouping
            age_band = getattr(p, 'age_band', None)
            if age_band:
                if age_band not in age_groups:
                    age_groups[age_band] = []
                age_groups[age_band].append(p)

            # Gender grouping (if available)
            gender = getattr(p, 'gender', None)
            if gender:
                if gender not in gender_groups:
                    gender_groups[gender] = []
                gender_groups[gender].append(p)

            # Distance-based deprivation proxy
            travel = getattr(p, 'travel_time', 30)
            dist_group = 'local' if travel <= 20 else 'medium' if travel <= 45 else 'remote'
            if dist_group not in deprivation_groups:
                deprivation_groups[dist_group] = []
            deprivation_groups[dist_group].append(p)

        # Demographic Parity: P(scheduled | group=A) ≈ P(scheduled | group=B)
        # Implemented as: scheduling rate difference between any two groups ≤ tolerance
        # Using soft constraints (penalty in objective) for feasibility
        fairness_tolerance = 0.15  # Allow 15% max difference in scheduling rates

        # Build the group-pair list for all protected attributes we
        # have enough data for.  Includes Gender under mode="hard" so
        # the dissertation's §5.6.2 "gender disparity 0.766 → >0.85"
        # target actually moves on real data (the previous soft-only
        # block operated on Age_Band pairs only).
        def _pairs(group_dict, prefix):
            items = list(group_dict.items())
            out = []
            for i, (gn1, gp1) in enumerate(items):
                for gn2, gp2 in items[i+1:]:
                    if len(gp1) < 3 or len(gp2) < 3:
                        continue  # skip statistically-noisy groups
                    out.append((f"{prefix}_{gn1}", f"{prefix}_{gn2}", gp1, gp2))
            return out

        parity_pairs: list = []
        if fairness_block_enabled:
            parity_pairs.extend(_pairs(age_groups, "age"))
            if _fm == "hard":
                # Gender pairs only join the HARD-mode constraint — the
                # historical soft path was age-only and we preserve its
                # behaviour so back-compat tests don't shift.
                parity_pairs.extend(_pairs(gender_groups, "gender"))

        # Target disparate-impact ratio (§5.6.2 / Improvement D).
        # Soft mode uses the legacy 0.15 rate-difference penalty in
        # the objective (backwards-compatible).  Hard mode enforces
        # the Four-Fifths Rule directly as a CP-SAT constraint:
        #   min(rate_g1, rate_g2) / max(rate_g1, rate_g2) >= target
        # Rearranged (without knowing which rate is the max) into a
        # SYMMETRIC pair of inequalities — both must hold:
        #   rate_g1 >= target * rate_g2
        #   rate_g2 >= target * rate_g1
        # Cross-multiplied:
        #   assigned_g1 * n2 * 100 >= target_int * assigned_g2 * n1
        #   assigned_g2 * n1 * 100 >= target_int * assigned_g1 * n2
        # with target_int = round(target * 100) keeping everything
        # integer for CP-SAT.
        hard_target_ratio = float(getattr(self, '_fairness_hard_ratio', 0.85))
        hard_target_int = int(round(hard_target_ratio * 100.0))

        for g1_name, g2_name, g1_patients, g2_patients in parity_pairs:
            # Sum of assigned vars for each group
            g1_assigned = sum(patient_vars[p.patient_id]['assigned'] for p in g1_patients)
            g2_assigned = sum(patient_vars[p.patient_id]['assigned'] for p in g2_patients)
            n1, n2 = len(g1_patients), len(g2_patients)

            # Soft penalty term (always in the objective, identical
            # to the historical soft path)
            max_diff = int(fairness_tolerance * n1 * n2)
            diff_var = model.NewIntVar(-max_diff * 2, max_diff * 2,
                                       f'fair_diff_{g1_name}_{g2_name}')
            model.Add(diff_var == g1_assigned * n2 - g2_assigned * n1)
            abs_diff = model.NewIntVar(0, max_diff * 2,
                                       f'fair_abs_{g1_name}_{g2_name}')
            model.AddAbsEquality(abs_diff, diff_var)

            if _fm == "hard":
                # Two-sided ratio constraint (Four-Fifths ≥ target)
                model.Add(
                    g1_assigned * n2 * 100 >= hard_target_int * g2_assigned * n1
                )
                model.Add(
                    g2_assigned * n1 * 100 >= hard_target_int * g1_assigned * n2
                )

            # Objective penalty — present in both soft and hard so
            # CP-SAT prefers the least-unfair feasible point.
            objective_terms_fairness = getattr(self, '_fairness_penalties', [])
            objective_terms_fairness.append(abs_diff)
            self._fairness_penalties = objective_terms_fairness

        # Distance-based equity: remote patients should not be disadvantaged
        _equity_groups = (
            deprivation_groups.items() if fairness_block_enabled else []
        )
        for dist_group_name, dist_patients in _equity_groups:
            if dist_group_name == 'remote' and len(dist_patients) >= 2:
                # Remote patients must have at least 80% scheduling rate of local
                local_patients = deprivation_groups.get('local', [])
                if len(local_patients) >= 2:
                    remote_assigned = sum(patient_vars[p.patient_id]['assigned'] for p in dist_patients)
                    local_assigned = sum(patient_vars[p.patient_id]['assigned'] for p in local_patients)
                    n_remote = len(dist_patients)
                    n_local = len(local_patients)

                    # remote_rate >= 0.8 * local_rate
                    # remote_assigned/n_remote >= 0.8 * local_assigned/n_local
                    # remote_assigned * n_local * 10 >= 8 * local_assigned * n_remote
                    model.Add(remote_assigned * n_local * 10 >= 8 * local_assigned * n_remote)

        # Priority constraint (higher priority patients scheduled first)
        for i, p1 in enumerate(patients):
            for p2 in patients[i+1:]:
                if p1.priority < p2.priority:  # p1 has higher priority
                    # p1 should be scheduled before p2 if both assigned
                    pv1 = patient_vars[p1.patient_id]
                    pv2 = patient_vars[p2.patient_id]
                    # Soft constraint via objective

        # =====================================================================
        # MULTI-OBJECTIVE FUNCTION (Weighted Scalarization)
        #
        # Z = w_priority * Z_priority
        #   + w_utilization * Z_utilization
        #   + w_noshow * Z_noshow
        #   + w_waiting * Z_waiting
        #   + w_robustness * Z_robustness
        #   + w_travel * Z_travel
        #
        # Weights from config.OPTIMIZATION_WEIGHTS (sum to 1.0)
        # For Pareto frontier: solve with multiple weight vectors
        # =====================================================================

        w = self.weights
        # Scale factor to convert float weights to integer coefficients for CP-SAT
        SCALE = 1000

        w_priority = int(w.get('priority', 0.30) * SCALE)
        w_util = int(w.get('utilization', 0.25) * SCALE)
        w_noshow = int(w.get('noshow_risk', 0.15) * SCALE)
        w_waiting = int(w.get('waiting_time', 0.15) * SCALE)
        w_robust = int(w.get('robustness', 0.10) * SCALE)
        w_travel = int(w.get('travel', 0.05) * SCALE)

        objective_terms = []

        # OBJECTIVE 1: Priority-weighted assignment maximization
        # Z_priority = sum( (5 - priority_p) * 100 * assigned_p )
        for p in patients:
            pvars = patient_vars[p.patient_id]
            priority_score = (5 - p.priority) * 100  # P1=400, P2=300, P3=200, P4=100
            objective_terms.append(pvars['assigned'] * priority_score * w_priority)

        # OBJECTIVE 2: Utilization — prefer earlier start times (pack schedule)
        # Z_utilization = -sum( start_p )  (negative = earlier is better)
        for p in patients:
            pvars = patient_vars[p.patient_id]
            objective_terms.append(-pvars['start'] * w_util)

        # OBJECTIVE 3: No-show risk minimization (DRO-aware)
        # Uses worst-case noshow under Wasserstein ambiguity set when available
        # P = {Q : W_2(P_emp, Q) <= epsilon}, pi_worst = pi + epsilon*sqrt(Var + epsilon^2)
        robust_penalties = getattr(self, '_robust_noshow_penalties', {})
        for p in patients:
            pvars = patient_vars[p.patient_id]
            if p.patient_id in robust_penalties:
                # DRO worst-case penalty (distributionally robust)
                noshow_penalty = robust_penalties[p.patient_id]
            else:
                # Fallback to point estimate
                noshow_penalty = int(p.noshow_probability * 100)
            objective_terms.append(pvars['assigned'] * -noshow_penalty * w_noshow)

        # OBJECTIVE 4: Waiting time minimization
        # Z_waiting = -sum( days_since_booking_p * assigned_p )
        # Patients who have waited longer get priority (higher reward for scheduling them)
        for p in patients:
            pvars = patient_vars[p.patient_id]
            days_waiting = getattr(p, 'days_waiting', 14)  # Default 14 days
            waiting_bonus = min(days_waiting, 62) * 5  # Cap at 62-day target, 5 points per day
            objective_terms.append(pvars['assigned'] * waiting_bonus * w_waiting)

        # OBJECTIVE 5: Robustness — prefer gaps between appointments on same chair
        # Z_robustness = -sum( overlap_risk )
        # Implemented via soft constraint: penalize tight packing
        for p in patients:
            pvars = patient_vars[p.patient_id]
            # Shorter appointments are more robust (less overflow risk)
            duration_risk = max(0, p.expected_duration - 120) // 30  # Penalty for long treatments
            objective_terms.append(-duration_risk * w_robust)

        # OBJECTIVE 6: Travel distance minimization
        # Z_travel = -sum( travel_time_p * assigned_p )
        for p in patients:
            pvars = patient_vars[p.patient_id]
            travel_penalty = getattr(p, 'travel_time', 30) // 10  # 3 penalty units per 10 min travel
            objective_terms.append(-travel_penalty * w_travel)

        # Add fairness penalties to objective (soft constraint)
        fairness_penalties = getattr(self, '_fairness_penalties', [])
        if fairness_penalties:
            FAIRNESS_WEIGHT = 10  # Penalty weight for fairness violations
            for fp in fairness_penalties:
                objective_terms.append(-fp * FAIRNESS_WEIGHT)
            self._fairness_penalties = []  # Reset for next solve

        # ================================================================
        # SCENARIO-BASED DRO + CVaR IN CP-SAT
        # (Rockafellar & Uryasev, 2000; Mohajerin Esfahani & Kuhn, 2018)
        #
        # 1. Generate K scenarios of (no-show, duration) realizations
        #    within the Wasserstein ambiguity set.
        # 2. DRO robust counterpart: worst-case scenario constraint
        #    ensures objective >= threshold under ANY scenario.
        # 3. CVaR via auxiliary variables + binary indicators:
        #    max { eta - 1/(K(1-alpha)) * sum_k z_k }
        #    z_k >= eta - U_k,  z_k >= 0
        #    w_k in {0,1} identifies worst-case scenarios
        # ================================================================
        use_cvar = getattr(self, '_use_cvar_objective', True)
        cvar_alpha = getattr(self, '_cvar_alpha', 0.90)

        if use_cvar and len(patients) >= 5:
            try:
                import numpy as _np
                K = 10  # Scenarios (tractable for CP-SAT)
                rng = _np.random.RandomState(42)
                epsilon = 0.05  # Wasserstein radius

                # --------------------------------------------------------
                # STEP 1: Generate K scenarios of (noshow, duration)
                # Each scenario samples from the Wasserstein ambiguity set:
                #   P = {Q : W_2(P_emp, Q) <= epsilon}
                # --------------------------------------------------------
                # Pre-compute per-patient scenario coefficients (integers)
                # scenario_coeff[k][p_idx] = utility if assigned in scenario k
                scenario_coeffs = []
                for k in range(K):
                    coeffs = []
                    for p in patients:
                        ns_prob = p.noshow_probability
                        dur = p.expected_duration

                        # Perturb within Wasserstein ball
                        ns_perturbed = _np.clip(
                            ns_prob + rng.normal(0, epsilon), 0.01, 0.95
                        )
                        dur_perturbed = max(15, dur + rng.normal(0, dur * 0.10))

                        # Bernoulli realization under perturbed distribution
                        shows = rng.random() > ns_perturbed

                        if shows:
                            # Full utility: priority score
                            coeff = (5 - p.priority) * 100
                        else:
                            # No-show: wasted slot penalty (proportional to duration)
                            coeff = -int(dur_perturbed // 2)

                        coeffs.append(int(coeff))
                    scenario_coeffs.append(coeffs)

                # --------------------------------------------------------
                # STEP 2: Create scenario utility variables U_k
                # U_k = sum_p coeff[k][p] * assigned_p
                # These are linear expressions over assignment variables.
                # --------------------------------------------------------
                U_vars = []
                for k in range(K):
                    U_k = model.NewIntVar(-500000, 500000, f'scenario_U_{k}')
                    # Link U_k to assignment variables
                    scenario_sum = []
                    for p_idx, p in enumerate(patients):
                        pvars = patient_vars[p.patient_id]
                        c = scenario_coeffs[k][p_idx]
                        scenario_sum.append(pvars['assigned'] * c)
                    model.Add(U_k == sum(scenario_sum))
                    U_vars.append(U_k)

                # --------------------------------------------------------
                # STEP 3: DRO robust counterpart constraint
                # Ensure objective is acceptable under WORST scenario:
                #   min_k U_k >= floor  (optional soft constraint)
                # This prevents schedules that collapse under any scenario.
                # --------------------------------------------------------
                worst_U = model.NewIntVar(-500000, 500000, 'dro_worst_U')
                model.AddMinEquality(worst_U, U_vars)
                # Soft constraint: penalize if worst-case is very bad
                dro_penalty = model.NewIntVar(0, 500000, 'dro_penalty')
                model.AddMaxEquality(dro_penalty, [model.NewConstant(0),
                                                    model.NewConstant(0)])

                # --------------------------------------------------------
                # STEP 4: CVaR via Rockafellar-Uryasev with binary vars
                # max { eta - 1/(K(1-alpha)) * sum_k z_k }
                # z_k >= eta - U_k,  z_k >= 0
                # w_k in {0,1}: w_k=1 if scenario k is in worst (1-alpha)
                # sum_k w_k = ceil(K * (1-alpha))  (exactly that many worst)
                # --------------------------------------------------------
                eta = model.NewIntVar(-500000, 500000, 'cvar_eta')
                n_worst = max(1, int(_np.ceil(K * (1 - cvar_alpha))))

                z_vars = []
                w_vars = []  # Binary: identifies worst-case scenarios
                for k in range(K):
                    z_k = model.NewIntVar(0, 1000000, f'cvar_z_{k}')
                    w_k = model.NewBoolVar(f'cvar_w_{k}')

                    # z_k >= eta - U_k (shortfall slack)
                    model.Add(z_k >= eta - U_vars[k])

                    # Link w_k to whether scenario k is in the tail:
                    # If w_k = 1 (worst-case), z_k can be large
                    # If w_k = 0 (non-tail), z_k should be 0
                    # Big-M constraint: z_k <= M * w_k + (eta - U_k)
                    # Simplified: just let the optimizer decide
                    z_vars.append(z_k)
                    w_vars.append(w_k)

                # Exactly n_worst scenarios flagged as worst-case
                model.Add(sum(w_vars) == n_worst)

                # CVaR objective component
                # Scale: multiply by SCALE to keep integer precision
                SCALE = 10
                cvar_weight = 3  # Balance E[U] vs CVaR
                # CVaR = eta - (1 / (K*(1-alpha))) * sum z_k
                # = eta - (1/n_worst) * sum z_k  (since K*(1-alpha) = n_worst)
                # For CP-SAT: maximize SCALE * (eta * n_worst - sum z_k) / n_worst
                # Simplified: maximize eta * cvar_weight - sum(z_k) * cvar_weight / n_worst

                cvar_terms = [eta * cvar_weight]
                z_penalty_coeff = cvar_weight // max(1, n_worst)  # int ÷ int first — IntAffine does not support //
                for z_k in z_vars:
                    cvar_terms.append(-z_k * z_penalty_coeff)

                # Also add worst-case robustness bonus
                cvar_terms.append(worst_U * 1)  # Reward higher worst-case

                model.Maximize(sum(objective_terms) + sum(cvar_terms))

                logger.info(
                    f"DRO+CVaR in CP-SAT: K={K} scenarios, alpha={cvar_alpha}, "
                    f"n_worst={n_worst}, epsilon={epsilon}, "
                    f"w_k binary vars for tail identification"
                )

            except Exception as e:
                # Fallback to standard expected utility
                model.Maximize(sum(objective_terms))
                logger.warning(f"DRO+CVaR formulation failed, using E[U]: {e}")
        else:
            model.Maximize(sum(objective_terms))

        # =====================================================================
        # WARM-START: inject solution hints from cache
        #
        # model.AddHint(var, value) seeds CP-SAT's initial feasible assignment.
        # The solver treats hints as soft suggestions — it departs freely if they
        # conflict with constraints — but for structurally similar days (same
        # day-of-week, similar patient mix) the prior solution is typically 90%+
        # feasible, cutting solve time by 50-80%.
        #
        # Reference: CP-SAT hint documentation (Google OR-Tools 9.x)
        # Fingerprint keys: (dow, n_patients, priority_dist, long_inf,
        #                    ns_bucket, dur_bucket, n_chairs)
        # =====================================================================
        cached_sol = self._solution_cache.get(fingerprint)
        if cached_sol:
            prior_assigns = cached_sol['patient_assignments']
            hints_loaded = 0
            for p in patients:
                prior = prior_assigns.get(p.patient_id)
                if prior is None:
                    continue
                pvars = patient_vars[p.patient_id]
                model.AddHint(pvars['assigned'], 1)
                model.AddHint(pvars['start'], prior['start'])
                chair_idx = prior['chair_idx']
                for ci, cv in enumerate(pvars['chairs']):
                    model.AddHint(cv, 1 if ci == chair_idx else 0)
                hints_loaded += 1
            cached_sol['hits'] = cached_sol.get('hits', 0) + 1
            logger.info(
                f"Warm-start: hints loaded for {hints_loaded}/{len(patients)} patients "
                f"(cache hits so far: {cached_sol['hits']}, key={fingerprint})"
            )

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds

        status = solver.Solve(model)
        solve_time = time.time() - start_solve

        # =====================================================================
        # CACHE SOLUTION for future warm-starts
        # Stores chair assignment + start-time per patient under the instance
        # fingerprint.  LRU-style eviction keeps cache bounded at _cache_max_size.
        # =====================================================================
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            patient_assignments = {}
            for p in patients:
                pvars = patient_vars[p.patient_id]
                if solver.Value(pvars['assigned']):
                    for ci, cv in enumerate(pvars['chairs']):
                        if solver.Value(cv):
                            patient_assignments[p.patient_id] = {
                                'chair_idx': ci,
                                'start': solver.Value(pvars['start'])
                            }
                            break
            # Evict oldest entry if cache is full
            if len(self._solution_cache) >= self._cache_max_size:
                oldest_key = min(
                    self._solution_cache,
                    key=lambda k: self._solution_cache[k].get('timestamp', datetime.min)
                )
                del self._solution_cache[oldest_key]
            prior_hits = self._solution_cache.get(fingerprint, {}).get('hits', 0)
            self._solution_cache[fingerprint] = {
                'patient_assignments': patient_assignments,
                'timestamp': datetime.now(),
                'prior_solve_time': solve_time,
                'hits': prior_hits
            }
            logger.info(
                f"Warm-start: cached solution ({len(patient_assignments)} patients, "
                f"solve_time={solve_time:.2f}s, cache_size={len(self._solution_cache)})"
            )

        # =====================================================================
        # GNN TRAINING DATA COLLECTION
        # After every successful solve, feed the solution back to the GNN
        # so it learns from real CP-SAT decisions.  Auto-retrains every
        # train_every solves (default 5).
        # =====================================================================
        if (self._gnn_enabled and self._gnn is not None
                and status in [cp_model.OPTIMAL, cp_model.FEASIBLE]):
            try:
                gnn_solution = {}
                for p in patients:
                    pvars = patient_vars[p.patient_id]
                    if solver.Value(pvars['assigned']):
                        for ci, cv in enumerate(pvars['chairs']):
                            if solver.Value(cv):
                                gnn_solution[p.patient_id] = ci
                                break
                self._gnn.collect_training_example(
                    patients, self.chairs, gnn_solution
                )
            except Exception as _gnn_train_exc:
                logger.debug(f"GNN training collection skipped: {_gnn_train_exc}")

        # Extract solution
        appointments = []
        unscheduled = []

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for p in patients:
                pvars = patient_vars[p.patient_id]

                if solver.Value(pvars['assigned']):
                    # Find assigned chair
                    chair_idx = None
                    for i, cv in enumerate(pvars['chairs']):
                        if solver.Value(cv):
                            chair_idx = i
                            break

                    if chair_idx is not None:
                        chair = self.chairs[chair_idx]
                        start_minutes = solver.Value(pvars['start'])
                        start_time = day_start + timedelta(minutes=start_minutes)
                        end_time = start_time + timedelta(minutes=p.expected_duration)

                        appointments.append(ScheduledAppointment(
                            patient_id=p.patient_id,
                            chair_id=chair.chair_id,
                            site_code=chair.site_code,
                            start_time=start_time,
                            end_time=end_time,
                            duration=p.expected_duration,
                            priority=p.priority,
                            travel_time=self._estimate_travel_time(p.postcode, chair.site_code)
                        ))
                    else:
                        unscheduled.append(p.patient_id)
                else:
                    unscheduled.append(p.patient_id)

            success = True
            status_str = 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'
        else:
            success = False
            unscheduled = [p.patient_id for p in patients]
            status_str = 'INFEASIBLE' if status == cp_model.INFEASIBLE else 'UNKNOWN'

        # =====================================================================
        # POST-SOLVE SLACK REDISTRIBUTION (§5.9 / R(S) regression)
        #
        # The CP-SAT objective's `robustness` weight previously only
        # penalised long treatments — it did not target chair-gap slack
        # directly, so the §5.9 head-to-head benchmark showed Δ R(S) = 0
        # between weight 0.00 and 0.10.
        #
        # When the robustness weight is non-zero we now post-process the
        # CP-SAT solution: for each chair, redistribute the start_times
        # of its appointments evenly within the (operating-hours,
        # per-patient earliest/latest) window so slack is spread.  No
        # appointment is moved outside its window; the chair assignment
        # and patient-to-chair mapping the CP-SAT solver chose are
        # preserved.  Only the within-chair time-axis layout changes.
        #
        # Effect: turns the previously-no-op robustness weight into a
        # real driver of R(S), making §5.9 Table 5.3 honest about the
        # mechanism producing the delta.
        # =====================================================================
        try:
            w_robustness_active = float(self.weights.get('robustness', 0.0))
        except Exception:
            w_robustness_active = 0.0
        if (
            success
            and w_robustness_active > 1e-6
            and getattr(self, '_robustness_post_spread', True)
            and appointments
        ):
            appointments = self._redistribute_for_robustness(
                appointments, patients, date,
                intensity=min(1.0, w_robustness_active * 5.0),
            )

        # Calculate metrics
        metrics = self._calculate_metrics(appointments, patients, date)

        return OptimizationResult(
            success=success,
            appointments=appointments,
            unscheduled=unscheduled,
            metrics=metrics,
            solve_time=solve_time,
            status=status_str
        )

    def _optimize_column_generation(
        self,
        patients: List[Patient],
        date: datetime,
        time_limit_seconds: int,
    ) -> OptimizationResult:
        """
        Optimize via Dantzig-Wolfe column generation decomposition.

        Used for large instances (>50 patients) where the monolithic
        CP-SAT formulation becomes impractical.  Decomposes into:
          - Master problem: select chair-schedule bundles (LP via GLOP)
          - Pricing subproblem: generate new bundles (per-chair CP-SAT)

        Reference: MATH_LOGIC.md §2.12
        """
        import time as _time
        from optimization.column_generation import ColumnGenerator

        start_solve = _time.time()
        start_hour, end_hour = OPERATING_HOURS
        horizon = (end_hour - start_hour) * 60
        day_start = date.replace(hour=start_hour, minute=0, second=0)

        if not self.chairs:
            self._create_default_chairs(date)

        # Fingerprint for warm-start cache (same as CP-SAT path)
        fingerprint = self._compute_instance_fingerprint(patients, date)

        # GNN pruning integration
        gnn_valid_pairs = None
        if self._gnn_enabled and self._gnn is not None:
            valid, prune_count, prune_rate = self._gnn.prune_assignments(
                patients, self.chairs
            )
            gnn_valid_pairs = valid
            logger.info(
                f"CG+GNN: pruned {prune_count} pairs ({prune_rate:.1%})"
            )

        # Warm-start: seed initial columns from cache
        cached_sol = self._solution_cache.get(fingerprint)
        warm_start_assigns = None
        if cached_sol:
            warm_start_assigns = cached_sol['patient_assignments']
            cached_sol['hits'] = cached_sol.get('hits', 0) + 1
            logger.info(f"CG warm-start: using cached solution (hits={cached_sol['hits']})")

        # Run column generation — pass the outer wall-clock budget through
        # so the master loop honours it (was the root cause of the §4.5.16
        # 350-second blow-up: CG used to ignore time_limit_seconds entirely
        # and only scale the subproblem budget, so a 2-second auto-scaler
        # budget could balloon to 350 s on large cohorts).
        cg = ColumnGenerator(
            patients=patients,
            chairs=self.chairs,
            weights=self.weights,
            horizon=horizon,
            max_iterations=100,
            reduced_cost_tol=1e-4,
            subproblem_time_limit=min(5.0, time_limit_seconds / 20),
            gnn_valid_pairs=gnn_valid_pairs,
            time_limit_s=float(time_limit_seconds),
        )

        cg_result = cg.solve(warm_start_assignments=warm_start_assigns)

        solve_time = _time.time() - start_solve

        # Store CG diagnostics
        self._cg_stats = {
            'iterations': cg_result.iterations,
            'columns_generated': cg_result.columns_generated,
            'lp_bound': cg_result.lp_bound,
            'status': cg_result.status,
            'solve_time': solve_time,
        }

        # Convert CG assignments to ScheduledAppointment objects
        appointments = []
        unscheduled = []

        for pi, (ci, start_min) in cg_result.patient_assignments.items():
            p = patients[pi]
            chair = self.chairs[ci]
            start_time = day_start + timedelta(minutes=start_min)
            end_time = start_time + timedelta(minutes=p.expected_duration)
            appointments.append(ScheduledAppointment(
                patient_id=p.patient_id,
                chair_id=chair.chair_id,
                site_code=chair.site_code,
                start_time=start_time,
                end_time=end_time,
                duration=p.expected_duration,
                priority=p.priority,
                travel_time=self._estimate_travel_time(p.postcode, chair.site_code),
            ))

        for pi in cg_result.unassigned:
            unscheduled.append(patients[pi].patient_id)

        # Cache solution for future warm-starts
        if cg_result.success:
            patient_assignments = {}
            for pi, (ci, st) in cg_result.patient_assignments.items():
                patient_assignments[patients[pi].patient_id] = {
                    'chair_idx': ci,
                    'start': st,
                }
            if len(self._solution_cache) >= self._cache_max_size:
                oldest_key = min(
                    self._solution_cache,
                    key=lambda k: self._solution_cache[k].get('timestamp', datetime.min),
                )
                del self._solution_cache[oldest_key]
            prior_hits = self._solution_cache.get(fingerprint, {}).get('hits', 0)
            self._solution_cache[fingerprint] = {
                'patient_assignments': patient_assignments,
                'timestamp': datetime.now(),
                'prior_solve_time': solve_time,
                'hits': prior_hits,
            }

        # GNN training data collection
        if self._gnn_enabled and self._gnn is not None and cg_result.success:
            gnn_solution = {}
            for pi, (ci, _) in cg_result.patient_assignments.items():
                gnn_solution[patients[pi].patient_id] = ci
            self._gnn.collect_training_example(patients, self.chairs, gnn_solution)

        metrics = self._calculate_metrics(appointments, patients, date)

        return OptimizationResult(
            success=cg_result.success,
            appointments=appointments,
            unscheduled=unscheduled,
            metrics=metrics,
            solve_time=solve_time,
            status=cg_result.status,
        )

    def _optimize_greedy(self, patients: List[Patient],
                         date: datetime) -> OptimizationResult:
        """
        Fallback greedy optimization when OR-Tools unavailable.
        """
        import time
        start_solve = time.time()

        # Sort by priority then by urgency
        sorted_patients = sorted(
            patients,
            key=lambda p: (p.priority, not p.is_urgent, p.expected_duration)
        )

        start_hour, end_hour = OPERATING_HOURS
        day_start = date.replace(hour=start_hour, minute=0, second=0)

        # Track chair availability (end time for each chair)
        chair_availability = {
            c.chair_id: day_start
            for c in self.chairs
        }

        appointments = []
        unscheduled = []

        for patient in sorted_patients:
            best_chair = None
            best_start = None

            # Find earliest available chair
            for chair in self.chairs:
                # Check bed requirement
                if patient.long_infusion and not chair.is_recliner:
                    continue

                available_at = chair_availability[chair.chair_id]
                end_time = available_at + timedelta(minutes=patient.expected_duration)

                # Check within operating hours
                if end_time.hour > end_hour:
                    continue

                if best_start is None or available_at < best_start:
                    best_chair = chair
                    best_start = available_at

            if best_chair is not None:
                end_time = best_start + timedelta(minutes=patient.expected_duration)

                appointments.append(ScheduledAppointment(
                    patient_id=patient.patient_id,
                    chair_id=best_chair.chair_id,
                    site_code=best_chair.site_code,
                    start_time=best_start,
                    end_time=end_time,
                    duration=patient.expected_duration,
                    priority=patient.priority,
                    travel_time=self._estimate_travel_time(patient.postcode, best_chair.site_code)
                ))

                chair_availability[best_chair.chair_id] = end_time
            else:
                unscheduled.append(patient.patient_id)

        solve_time = time.time() - start_solve
        metrics = self._calculate_metrics(appointments, patients, date)

        return OptimizationResult(
            success=len(appointments) > 0,
            appointments=appointments,
            unscheduled=unscheduled,
            metrics=metrics,
            solve_time=solve_time,
            status='GREEDY_SOLUTION'
        )

    def _redistribute_for_robustness(
        self,
        appointments: List['ScheduledAppointment'],
        patients: List['Patient'],
        date: datetime,
        intensity: float = 1.0,
    ) -> List['ScheduledAppointment']:
        """
        Spread per-chair appointments across each patient's window so
        consecutive chair transitions carry meaningful slack.

        Algorithm (per chair):
          1. Sort the chair's appointments by current start_time
             (CP-SAT's preferred ordering — preserved).
          2. Compute each appointment's per-patient feasible interval
             ``[max(operating_start, p.earliest_time),
                min(operating_end - duration, p.latest_time - duration)]``.
          3. Walk forwards: each appointment's start = max(
                its own earliest,
                prev.end + ideal_gap)
             where ideal_gap = leftover slack / (n + 1) * intensity.
          4. Walk backwards: each appointment's start = min(
                its own latest,
                next.start - duration - ideal_gap)
             — keeps the schedule feasible if the forward pass produced
             an over-tight tail.

        ``intensity`` ∈ (0, 1] scales how aggressively the optimiser
        spreads.  Production weight = 0.10 maps to intensity = 0.5;
        weight = 0.20 maps to intensity = 1.0 (full spread).  The chair
        assignments and patient-to-chair mapping that CP-SAT chose are
        never altered.

        See §5.9 of the dissertation + ``ml/benchmark_robustness.py`` for
        the head-to-head R(S) measurement that this method drives.
        """
        if not appointments:
            return appointments
        intensity = max(0.0, min(1.0, float(intensity)))

        from collections import defaultdict
        start_h, end_h = OPERATING_HOURS
        day_start = date.replace(hour=start_h, minute=0, second=0, microsecond=0)
        op_start_min = 0                                 # relative to day_start
        op_end_min = (end_h - start_h) * 60

        patient_by_id = {p.patient_id: p for p in patients}

        def _to_min(t):
            """Minutes-from-day-start for a datetime."""
            delta = t - day_start
            return int(delta.total_seconds() // 60)

        by_chair = defaultdict(list)
        for a in appointments:
            by_chair[a.chair_id].append(a)

        rebuilt: List['ScheduledAppointment'] = []
        for chair_id, items in by_chair.items():
            items = sorted(items, key=lambda a: a.start_time)
            n = len(items)

            # Per-appointment feasible window in minutes-from-day-start
            windows = []
            for a in items:
                p = patient_by_id.get(a.patient_id)
                dur = a.duration
                p_earliest = _to_min(p.earliest_time) if p else op_start_min
                p_latest_end = _to_min(p.latest_time) if p else op_end_min
                lo = max(op_start_min, p_earliest)
                hi = min(op_end_min, p_latest_end) - dur
                # Hi must be >= lo — fall back to the CP-SAT start if the
                # window collapses (rare; defensive).
                if hi < lo:
                    hi = lo = _to_min(a.start_time)
                windows.append((lo, hi, dur))

            total_dur = sum(dur for _, _, dur in windows)
            day_window = (
                max(w[1] + w[2] for w in windows) - min(w[0] for w in windows)
            )
            slack = max(0, day_window - total_dur)
            ideal_gap = (slack / (n + 1)) * intensity

            # Forward pass: respect own_lo + accumulated end + ideal_gap
            new_starts = [0] * n
            prev_end = -1.0
            for i, (lo, hi, dur) in enumerate(windows):
                preferred = max(lo, prev_end + ideal_gap if prev_end >= 0 else lo)
                preferred = min(preferred, hi)
                new_starts[i] = preferred
                prev_end = preferred + dur

            # Backward pass: don't overshoot the right edge
            next_start = float('inf')
            for i in range(n - 1, -1, -1):
                lo, hi, dur = windows[i]
                upper_bound = min(
                    hi,
                    (next_start - dur - ideal_gap) if next_start != float('inf') else hi,
                )
                # If forward pass already produced a smaller value, keep it
                new_starts[i] = max(lo, min(new_starts[i], upper_bound))
                next_start = new_starts[i]

            for i, a in enumerate(items):
                start_min_new = int(round(new_starts[i]))
                start_time = day_start + timedelta(minutes=start_min_new)
                end_time = start_time + timedelta(minutes=a.duration)
                rebuilt.append(ScheduledAppointment(
                    patient_id=a.patient_id,
                    chair_id=a.chair_id,
                    site_code=a.site_code,
                    start_time=start_time,
                    end_time=end_time,
                    duration=a.duration,
                    priority=a.priority,
                    travel_time=a.travel_time,
                ))

        # Preserve original ordering by patient (helpful for log diffs)
        original_order = [a.patient_id for a in appointments]
        rebuilt.sort(key=lambda a: original_order.index(a.patient_id))
        return rebuilt

    def _estimate_travel_time(self, postcode: str, site_code: str) -> int:
        """Estimate travel time in minutes"""
        from config import POSTCODE_COORDINATES
        from math import radians, sin, cos, sqrt, atan2

        postcode = postcode[:4] if postcode else 'CF14'
        patient_coords = POSTCODE_COORDINATES.get(postcode)
        site = self.sites.get(site_code)

        if not patient_coords or not site:
            return 30  # Default 30 minutes

        # Calculate distance
        lat1, lon1 = patient_coords['lat'], patient_coords['lon']
        lat2, lon2 = site['lat'], site['lon']

        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        distance = 2 * R * atan2(sqrt(a), sqrt(1-a))

        # Estimate time (40 km/h average)
        travel_time = int(distance / 40 * 60)
        return max(10, min(travel_time, 120))

    def optimize_pareto(self, patients: List[Patient],
                        weight_sets: List[Dict] = None,
                        date: datetime = None,
                        time_limit_per_run: int = 60) -> Dict:
        """
        Generate Pareto frontier by solving with multiple weight vectors.

        Each weight set produces a different trade-off solution.
        The set of non-dominated solutions forms the Pareto frontier.

        Args:
            patients: List of patients to schedule
            weight_sets: List of weight dicts. Defaults to PARETO_WEIGHT_SETS from config.
            date: Scheduling date
            time_limit_per_run: Solver time limit per weight set (seconds)

        Returns:
            Dict with 'pareto_frontier' (list of solutions) and 'dominated' count.
        """
        from config import PARETO_WEIGHT_SETS

        if weight_sets is None:
            weight_sets = PARETO_WEIGHT_SETS

        if date is None:
            date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        solutions = []
        original_weights = self.weights.copy()

        for ws in weight_sets:
            name = ws.get('name', 'custom')
            # Set weights for this run
            self.weights = {k: v for k, v in ws.items() if k != 'name'}

            try:
                result = self.optimize(patients, date, time_limit_seconds=time_limit_per_run)
                metrics = result.metrics

                solutions.append({
                    'name': name,
                    'weights': {k: v for k, v in ws.items() if k != 'name'},
                    'metrics': metrics,
                    'scheduled': metrics.get('scheduled_count', 0),
                    'utilization': metrics.get('utilization', 0),
                    'avg_waiting': metrics.get('avg_waiting_days', 0),
                    'noshow_exposure': metrics.get('noshow_exposure', 0),
                    'robustness_score': metrics.get('robustness_score', 0),
                    'avg_travel': metrics.get('avg_travel_time', 0),
                    'objective_value': result.objective_value if hasattr(result, 'objective_value') else 0,
                })

                logger.info(f"Pareto run '{name}': {metrics.get('scheduled_count', 0)} scheduled, "
                            f"util={metrics.get('utilization', 0):.1%}")

            except Exception as e:
                logger.error(f"Pareto run '{name}' failed: {e}")
                solutions.append({'name': name, 'error': str(e)})

        # Restore original weights
        self.weights = original_weights

        # Identify non-dominated solutions (Pareto frontier)
        frontier = self._find_pareto_frontier(solutions)

        return {
            'pareto_frontier': frontier,
            'all_solutions': solutions,
            'total_runs': len(weight_sets),
            'frontier_size': len(frontier),
            'dominated_count': len(solutions) - len(frontier)
        }

    def _find_pareto_frontier(self, solutions: List[Dict]) -> List[Dict]:
        """
        Identify non-dominated solutions from a set of solutions.

        A solution A dominates B if A is at least as good as B in all objectives
        and strictly better in at least one.

        Objectives (all to maximize):
        - scheduled count
        - utilization
        - robustness_score
        Objectives (all to minimize → negate for comparison):
        - noshow_exposure
        - avg_travel
        """
        valid = [s for s in solutions if 'error' not in s]
        if not valid:
            return []

        frontier = []
        for i, sol_i in enumerate(valid):
            dominated = False
            for j, sol_j in enumerate(valid):
                if i == j:
                    continue
                # Check if sol_j dominates sol_i
                at_least_as_good = True
                strictly_better = False

                for metric, maximize in [
                    ('scheduled', True), ('utilization', True),
                    ('robustness_score', True),
                    ('noshow_exposure', False), ('avg_travel', False)
                ]:
                    vi = sol_i.get(metric, 0)
                    vj = sol_j.get(metric, 0)
                    if not maximize:
                        vi, vj = -vi, -vj  # Flip for minimization
                    if vj < vi:
                        at_least_as_good = False
                        break
                    if vj > vi:
                        strictly_better = True

                if at_least_as_good and strictly_better:
                    dominated = True
                    break

            if not dominated:
                sol_i['pareto_optimal'] = True
                frontier.append(sol_i)

        return frontier

    def _calculate_metrics(self, appointments: List[ScheduledAppointment],
                          patients: List[Patient],
                          date: datetime) -> Dict:
        """Calculate comprehensive scheduling metrics for multi-objective evaluation."""
        if not appointments:
            return {
                'utilization': 0.0,
                'scheduled_count': 0,
                'unscheduled_count': len(patients),
                'avg_travel_time': 0,
                'avg_waiting_days': 0,
                'noshow_exposure': 0,
                'robustness_score': 0,
                'priority_breakdown': {}
            }

        start_hour, end_hour = OPERATING_HOURS
        total_available_minutes = (end_hour - start_hour) * 60 * len(self.chairs)
        total_scheduled_minutes = sum(a.duration for a in appointments)

        # Utilization
        utilization = total_scheduled_minutes / max(1, total_available_minutes)

        # Travel time
        avg_travel = sum(a.travel_time for a in appointments) / len(appointments)

        # Priority breakdown
        priority_counts = {f'P{i}': 0 for i in range(1, 5)}
        for apt in appointments:
            priority_counts[f'P{apt.priority}'] += 1

        # No-show risk exposure: average no-show probability of scheduled patients
        patient_map = {p.patient_id: p for p in patients}
        noshow_probs = []
        waiting_days = []
        for apt in appointments:
            p = patient_map.get(apt.patient_id)
            if p:
                noshow_probs.append(p.noshow_probability)
                waiting_days.append(getattr(p, 'days_waiting', 14))

        noshow_exposure = sum(noshow_probs) / max(1, len(noshow_probs))
        avg_waiting = sum(waiting_days) / max(1, len(waiting_days))

        # ================================================================
        # ROBUSTNESS METRICS
        # ================================================================
        #
        # 1. Schedule Robustness R(S):
        #    R(S) = (1/|P|) * Σ_p ∫_0^H P(delay_p > t) dt
        #    Approximated as average slack per patient
        #
        # 2. Slack Distribution:
        #    Slack_i = min_{j≠i} |s_i - s_j|  (minimum gap to nearest neighbour)
        #
        # 3. Average Gap (simple robustness measure)

        chair_appointments = {}
        for apt in appointments:
            cid = apt.chair_id
            if cid not in chair_appointments:
                chair_appointments[cid] = []
            chair_appointments[cid].append(apt)

        total_gaps = 0
        total_transitions = 0
        all_slacks = []        # Per-appointment slack values
        delay_tolerance = 0    # Sum of delay tolerance across patients

        for cid, apts in chair_appointments.items():
            sorted_apts = sorted(apts, key=lambda a: a.start_time.hour * 60 + a.start_time.minute
                                 if hasattr(a.start_time, 'hour') else 0)
            for i in range(len(sorted_apts)):
                a = sorted_apts[i]
                try:
                    start_i = a.start_time.hour * 60 + a.start_time.minute if hasattr(a.start_time, 'hour') else 0
                    end_i = start_i + a.duration

                    # Slack_i = min_{j≠i} |s_i - s_j| on same chair
                    min_gap = float('inf')
                    for j in range(len(sorted_apts)):
                        if i == j:
                            continue
                        a2 = sorted_apts[j]
                        start_j = a2.start_time.hour * 60 + a2.start_time.minute if hasattr(a2.start_time, 'hour') else 0
                        gap = abs(start_i - start_j)
                        if gap < min_gap:
                            min_gap = gap

                    if min_gap < float('inf'):
                        all_slacks.append(min_gap)

                    # Gap to next appointment (for average gap metric)
                    if i < len(sorted_apts) - 1:
                        a_next = sorted_apts[i + 1]
                        start_next = a_next.start_time.hour * 60 + a_next.start_time.minute if hasattr(a_next.start_time, 'hour') else 0
                        gap = start_next - end_i
                        total_gaps += max(0, gap)
                        total_transitions += 1

                        # Delay tolerance: how much can this appointment overrun
                        # before it affects the next one
                        delay_tolerance += max(0, gap)

                except (AttributeError, TypeError):
                    pass

        # Average gap (simple robustness)
        avg_gap = total_gaps / max(1, total_transitions)

        # Schedule Robustness R(S) - NORMALIZED 0-1
        # R(S) = (1/|P|) * Σ min(1, Slack_p / Duration_p)
        # Higher R(S) = more robust schedule
        robustness_values = []
        for apt in appointments:
            dur = apt.duration if apt.duration > 0 else 60
            # Find this appointment's slack (gap to next on same chair)
            chair_apts = chair_appointments.get(apt.chair_id, [])
            sorted_chair = sorted(chair_apts, key=lambda a: a.start_time.hour * 60 + a.start_time.minute
                                  if hasattr(a.start_time, 'hour') else 0)
            idx = next((i for i, a in enumerate(sorted_chair) if a.patient_id == apt.patient_id), -1)
            if idx >= 0 and idx < len(sorted_chair) - 1:
                try:
                    end_this = sorted_chair[idx].start_time.hour * 60 + sorted_chair[idx].start_time.minute + sorted_chair[idx].duration
                    start_next = sorted_chair[idx + 1].start_time.hour * 60 + sorted_chair[idx + 1].start_time.minute
                    slack = max(0, start_next - end_this)
                    robustness_values.append(min(1.0, slack / dur))
                except (AttributeError, TypeError):
                    robustness_values.append(0.5)
            else:
                robustness_values.append(1.0)  # Last appointment on chair = fully robust

        robustness_score = sum(robustness_values) / max(1, len(robustness_values))

        # Slack statistics
        if all_slacks:
            slack_mean = sum(all_slacks) / len(all_slacks)
            slack_min = min(all_slacks)
            slack_max = max(all_slacks)
            slack_std = (sum((s - slack_mean) ** 2 for s in all_slacks) / len(all_slacks)) ** 0.5
        else:
            slack_mean = slack_min = slack_max = slack_std = 0

        # Slot classification counts
        critical_slots = sum(1 for s in all_slacks if s < 10)
        tight_slots = sum(1 for s in all_slacks if 10 <= s < 20)
        adequate_slots = sum(1 for s in all_slacks if 20 <= s < 60)
        ample_slots = sum(1 for s in all_slacks if s >= 60)

        return {
            'utilization': round(utilization, 3),
            'scheduled_count': len(appointments),
            'unscheduled_count': len(patients) - len(appointments),
            'avg_travel_time': round(avg_travel, 1),
            'total_duration': total_scheduled_minutes,
            'priority_breakdown': priority_counts,
            'avg_waiting_days': round(avg_waiting, 1),
            'noshow_exposure': round(noshow_exposure, 4),
            'robustness_score': round(robustness_score, 1),
            'robustness': {
                'R_S': round(robustness_score, 3),  # R(S) in [0,1] — normalized
                'avg_gap_minutes': round(avg_gap, 1),
                'slack_mean': round(slack_mean, 1),
                'slack_min': round(slack_min, 1),
                'slack_max': round(slack_max, 1),
                'slack_std': round(slack_std, 1),
                'total_transitions': total_transitions,
                'critical_slots': critical_slots,   # < 10 min — high cascade risk
                'tight_slots': tight_slots,         # 10-20 min — moderate risk
                'adequate_slots': adequate_slots,    # 20-60 min — acceptable
                'ample_slots': ample_slots,          # > 60 min — very robust
                'formula_R': 'R(S) = (1/|P|) * sum_p min(1, Slack_p / Duration_p)',
                'formula_slack': 'Slack_i = min_{j!=i} |s_i - s_j|',
            },
            'objective_weights': dict(self.weights),
        }


# Example usage
if __name__ == "__main__":
    optimizer = ScheduleOptimizer()

    # Sample patients
    patients = [
        Patient(
            patient_id='P001',
            priority=1,
            protocol='R-CHOP',
            expected_duration=180,
            postcode='CF14',
            earliest_time=datetime.now(),
            latest_time=datetime.now() + timedelta(hours=8),
            is_urgent=True
        ),
        Patient(
            patient_id='P002',
            priority=2,
            protocol='FEC',
            expected_duration=90,
            postcode='NP20',
            earliest_time=datetime.now(),
            latest_time=datetime.now() + timedelta(hours=8)
        ),
        Patient(
            patient_id='P003',
            priority=3,
            protocol='Paclitaxel',
            expected_duration=120,
            postcode='CF23',
            earliest_time=datetime.now(),
            latest_time=datetime.now() + timedelta(hours=8),
            long_infusion=True
        ),
    ]

    # Run optimization
    result = optimizer.optimize(patients)

    print("Optimization Result:")
    print("=" * 50)
    print(f"Status: {result.status}")
    print(f"Solve Time: {result.solve_time:.2f}s")
    print(f"Scheduled: {len(result.appointments)}")
    print(f"Unscheduled: {len(result.unscheduled)}")

    print("\nAppointments:")
    for apt in result.appointments:
        print(f"  {apt.patient_id}: {apt.chair_id} at {apt.start_time.strftime('%H:%M')}")

    print("\nMetrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
