"""
Column Generation for Large-Scale SACT Scheduling
===================================================

Decomposes the monolithic CP-SAT formulation into:
  - **Master problem (RMP)**: selects which chair-schedule "bundles" to use
  - **Pricing subproblem**: generates new bundles via reduced-cost pricing

Handles 100+ patients/day where the monolithic formulation scales poorly
(O(n_patients × n_chairs) binary variables).

Mathematical reference: MATH_LOGIC.md §2.12

Architecture
------------
1.  Generate initial columns via greedy first-fit-decreasing.
2.  Solve LP relaxation of the master (set-partitioning) with GLOP.
3.  Extract dual prices π_p (patient coverage) and μ_c (chair usage).
4.  For each chair c, solve a pricing subproblem:
        max  Σ_p (cost_p − π_p) · x_p  −  μ_c
    subject to no-overlap, bed-requirement, operating-hours.
5.  If any column has positive reduced cost → add to master, goto 2.
6.  Otherwise LP is solved; round to integer via a restricted CP-SAT
    over the LP-active columns only.

References
----------
- Desaulniers, Desrosiers, Solomon (2005) "Column Generation"
- Lübbecke & Desrosiers (2005) "Selected Topics in Column Generation"
- Barnhart et al. (1998) "Branch-and-Price"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OR-Tools availability (same guard as optimizer.py)
# ---------------------------------------------------------------------------
try:
    from ortools.sat.python import cp_model
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    logger.warning("OR-Tools not available — column generation disabled")

from config import OPERATING_HOURS, OPTIMIZATION_WEIGHTS


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Column:
    """
    A single feasible chair schedule ("bundle").

    Each column represents one chair's complete day plan: which patients
    are assigned, their start times, and the aggregate cost used in the
    master objective.

    Attributes
    ----------
    column_id : int
        Unique identifier within the CG iteration.
    chair_idx : int
        Index into the global chairs list.
    chair_id : str
        Human-readable chair identifier (e.g. 'WC-C03').
    patient_indices : list[int]
        Indices into the patient list for patients in this bundle.
    start_times : list[int]
        Start time (minutes from day start) for each patient.
    cost : float
        Weighted objective value of this bundle (used in master LP).
    reduced_cost : float
        Most recent reduced cost from the pricing subproblem.
    """
    column_id: int
    chair_idx: int
    chair_id: str
    patient_indices: List[int] = field(default_factory=list)
    start_times: List[int] = field(default_factory=list)
    cost: float = 0.0
    reduced_cost: float = 0.0


@dataclass
class CGResult:
    """Result container for the column generation solver."""
    success: bool
    columns_selected: List[Column]
    patient_assignments: Dict[int, Tuple[int, int]]  # p_idx → (chair_idx, start)
    unassigned: List[int]                             # p_idx list
    total_cost: float
    iterations: int
    columns_generated: int
    solve_time: float
    lp_bound: float
    status: str  # 'CG_OPTIMAL', 'CG_FEASIBLE', 'CG_MAX_ITER', 'CG_FAILED'


# ═══════════════════════════════════════════════════════════════════════════
# Column Generator
# ═══════════════════════════════════════════════════════════════════════════

class ColumnGenerator:
    """
    Dantzig-Wolfe column generation for SACT chair scheduling.

    Parameters
    ----------
    patients : list
        Patient objects (from optimizer.Patient).
    chairs : list
        Chair objects (from optimizer.Chair).
    weights : dict
        Objective weights (priority, utilization, etc.).
    horizon : int
        Day length in minutes (default derived from OPERATING_HOURS).
    max_iterations : int
        Maximum CG iterations before stopping (default 100).
    reduced_cost_tol : float
        Stop when best reduced cost < this (default 1e-4).
    subproblem_time_limit : float
        CP-SAT time limit per subproblem in seconds (default 5).
    gnn_valid_pairs : set or None
        If provided, restricts (patient_idx, chair_idx) pairs in subproblems.
    """

    def __init__(
        self,
        patients,
        chairs,
        weights: Dict[str, float] = None,
        horizon: int = None,
        max_iterations: int = 100,
        reduced_cost_tol: float = 1e-4,
        subproblem_time_limit: float = 5.0,
        gnn_valid_pairs: Optional[Set[Tuple[int, int]]] = None,
    ):
        self.patients = patients
        self.chairs = chairs
        self.n_patients = len(patients)
        self.n_chairs = len(chairs)
        self.weights = weights or OPTIMIZATION_WEIGHTS
        start_h, end_h = OPERATING_HOURS
        self.horizon = horizon or (end_h - start_h) * 60
        self.max_iterations = max_iterations
        self.reduced_cost_tol = reduced_cost_tol
        self.subproblem_time_limit = subproblem_time_limit
        self.gnn_valid_pairs = gnn_valid_pairs

        # All generated columns
        self._columns: List[Column] = []
        self._next_col_id = 0

        # Precompute per-patient objective cost (same formula as optimizer.py)
        self._patient_costs = self._compute_patient_costs()

    # ------------------------------------------------------------------
    # Patient cost (mirrors optimizer.py objective terms)
    # ------------------------------------------------------------------
    def _compute_patient_costs(self) -> np.ndarray:
        """
        Compute a scalar cost for scheduling each patient.

        Combines priority score, no-show penalty, and waiting bonus,
        scaled by the objective weights — exactly matching the terms
        in ScheduleOptimizer._optimize_cpsat().
        """
        SCALE = 1000
        w_pri = self.weights.get('priority', 0.30) * SCALE
        w_ns = self.weights.get('noshow_risk', 0.15) * SCALE
        w_wait = self.weights.get('waiting_time', 0.15) * SCALE

        costs = np.zeros(self.n_patients)
        for i, p in enumerate(self.patients):
            pri_score = (5 - p.priority) * 100 * w_pri
            ns_penalty = -getattr(p, 'noshow_probability', 0.1) * 100 * w_ns
            wait_bonus = min(getattr(p, 'days_waiting', 0), 62) * 5 * w_wait
            costs[i] = pri_score + ns_penalty + wait_bonus
        return costs

    # ------------------------------------------------------------------
    # Cost of a bundle
    # ------------------------------------------------------------------
    def _bundle_cost(self, patient_indices: List[int]) -> float:
        """Sum of per-patient costs for a given bundle."""
        return float(sum(self._patient_costs[pi] for pi in patient_indices))

    # ==================================================================
    # INITIAL COLUMN GENERATION  (greedy first-fit-decreasing)
    # ==================================================================
    def generate_initial_columns(
        self,
        warm_start_assignments: Optional[Dict[str, Dict]] = None,
    ) -> List[Column]:
        """
        Build an initial set of feasible columns.

        Strategy:
        1. If warm_start_assignments provided, reconstruct columns from
           the cached solution (one column per chair that has patients).
        2. Otherwise, greedy first-fit-decreasing by priority:
           sort patients by (priority ASC, duration DESC), then assign
           each patient to the earliest-available feasible chair.

        Returns list of generated Column objects.
        """
        columns = []

        if warm_start_assignments:
            columns = self._columns_from_warm_start(warm_start_assignments)
            if columns:
                logger.info(
                    f"CG: seeded {len(columns)} initial columns from warm-start cache"
                )
                return columns

        # Greedy first-fit-decreasing
        sorted_pats = sorted(
            range(self.n_patients),
            key=lambda i: (
                self.patients[i].priority,
                not self.patients[i].is_urgent,
                -self.patients[i].expected_duration,
            ),
        )

        # Track next-available time per chair (minutes from day start)
        chair_avail = [0] * self.n_chairs
        # Accumulate patients per chair
        chair_patients: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_chairs)]

        for pi in sorted_pats:
            p = self.patients[pi]
            dur = p.expected_duration
            best_ci = None
            best_start = None

            for ci, c in enumerate(self.chairs):
                # GNN pruning
                if self.gnn_valid_pairs is not None and (pi, ci) not in self.gnn_valid_pairs:
                    continue
                # Bed requirement
                if p.long_infusion and not c.is_recliner:
                    continue
                start = chair_avail[ci]
                if start + dur > self.horizon:
                    continue
                if best_start is None or start < best_start:
                    best_ci = ci
                    best_start = start

            if best_ci is not None:
                chair_patients[best_ci].append((pi, best_start))
                chair_avail[best_ci] = best_start + dur

        # Convert to Column objects
        for ci in range(self.n_chairs):
            if chair_patients[ci]:
                col = self._make_column(
                    ci,
                    [pi for pi, _ in chair_patients[ci]],
                    [st for _, st in chair_patients[ci]],
                )
                columns.append(col)

        logger.info(
            f"CG: generated {len(columns)} initial greedy columns "
            f"covering {sum(len(c.patient_indices) for c in columns)}/{self.n_patients} patients"
        )
        return columns

    def _columns_from_warm_start(
        self, assignments: Dict[str, Dict]
    ) -> List[Column]:
        """Reconstruct columns from a warm-start cache entry."""
        # Group by chair_idx
        chair_bundles: Dict[int, List[Tuple[int, int]]] = {}
        pid_to_idx = {p.patient_id: i for i, p in enumerate(self.patients)}

        for pid, info in assignments.items():
            pi = pid_to_idx.get(pid)
            if pi is None:
                continue
            ci = info['chair_idx']
            start = info['start']
            chair_bundles.setdefault(ci, []).append((pi, start))

        columns = []
        for ci, bundle in chair_bundles.items():
            bundle.sort(key=lambda x: x[1])  # sort by start time
            col = self._make_column(
                ci,
                [pi for pi, _ in bundle],
                [st for _, st in bundle],
            )
            columns.append(col)
        return columns

    def _make_column(
        self, chair_idx: int, patient_indices: List[int], start_times: List[int]
    ) -> Column:
        """Create a Column object and register it."""
        col = Column(
            column_id=self._next_col_id,
            chair_idx=chair_idx,
            chair_id=self.chairs[chair_idx].chair_id,
            patient_indices=list(patient_indices),
            start_times=list(start_times),
            cost=self._bundle_cost(patient_indices),
        )
        self._next_col_id += 1
        self._columns.append(col)
        return col

    # ==================================================================
    # MASTER PROBLEM  (LP relaxation via GLOP)
    # ==================================================================
    def _solve_master(self) -> Tuple[Optional[List[float]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solve the LP relaxation of the restricted master problem.

        Master formulation (set-partitioning relaxation):

            max   Σ_k  cost_k · λ_k
            s.t.  Σ_k  a_{p,k} · λ_k  ≤  1     ∀p  (patient coverage)
                  Σ_{k∈K_c}  λ_k       ≤  1     ∀c  (one schedule per chair)
                  0 ≤ λ_k ≤ 1

        Returns
        -------
        (lambda_vals, pi_duals, mu_duals) or (None, None, None) on failure.
            lambda_vals : list[float]  — LP values for each column
            pi_duals    : ndarray[n_patients]  — dual prices for coverage
            mu_duals    : ndarray[n_chairs]    — dual prices for chair usage
        """
        if not ORTOOLS_AVAILABLE:
            return None, None, None

        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            logger.error("CG: GLOP solver not available")
            return None, None, None

        n_cols = len(self._columns)

        # Decision variables: λ_k ∈ [0, 1]
        lam = [solver.NumVar(0.0, 1.0, f'lam_{k}') for k in range(n_cols)]

        # Patient coverage constraints: Σ_k a_{p,k} · λ_k ≤ 1
        pi_constraints = []
        for pi in range(self.n_patients):
            ct = solver.Constraint(0.0, 1.0, f'cover_p{pi}')
            for k, col in enumerate(self._columns):
                if pi in col.patient_indices:
                    ct.SetCoefficient(lam[k], 1.0)
            pi_constraints.append(ct)

        # Chair usage constraints: Σ_{k∈K_c} λ_k ≤ 1
        mu_constraints = []
        for ci in range(self.n_chairs):
            ct = solver.Constraint(0.0, 1.0, f'chair_c{ci}')
            for k, col in enumerate(self._columns):
                if col.chair_idx == ci:
                    ct.SetCoefficient(lam[k], 1.0)
            mu_constraints.append(ct)

        # Objective: max Σ_k cost_k · λ_k
        objective = solver.Objective()
        for k, col in enumerate(self._columns):
            objective.SetCoefficient(lam[k], col.cost)
        objective.SetMaximization()

        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            logger.warning(f"CG master LP status: {status}")
            return None, None, None

        lambda_vals = [lam[k].solution_value() for k in range(n_cols)]

        # Extract dual values
        pi_duals = np.array([ct.dual_value() for ct in pi_constraints])
        mu_duals = np.array([ct.dual_value() for ct in mu_constraints])

        return lambda_vals, pi_duals, mu_duals

    # ==================================================================
    # PRICING SUBPROBLEM  (per-chair CP-SAT)
    # ==================================================================
    def _solve_pricing(
        self, chair_idx: int, pi_duals: np.ndarray, mu_dual: float
    ) -> Optional[Column]:
        """
        Solve the pricing subproblem for a single chair.

        Finds the bundle of patients for chair `chair_idx` with maximum
        reduced cost:

            rc = Σ_p (cost_p − π_p) · x_p  −  μ_c

        Subject to:
            - No overlap on the chair
            - Bed requirement (long_infusion → recliner)
            - Operating hours [0, horizon]

        Returns a Column if reduced cost > tolerance, else None.
        """
        if not ORTOOLS_AVAILABLE:
            return None

        chair = self.chairs[chair_idx]
        model = cp_model.CpModel()

        # Candidate patients for this chair
        candidates = []
        for pi in range(self.n_patients):
            p = self.patients[pi]
            # Bed requirement filter
            if p.long_infusion and not chair.is_recliner:
                continue
            # GNN filter
            if self.gnn_valid_pairs is not None and (pi, chair_idx) not in self.gnn_valid_pairs:
                continue
            candidates.append(pi)

        if not candidates:
            return None

        # Decision variables
        x = {}       # x[pi] ∈ {0,1}: patient pi assigned to this chair
        s = {}       # s[pi] ∈ [0, horizon - dur]: start time
        intervals = []

        for pi in candidates:
            p = self.patients[pi]
            dur = p.expected_duration
            x[pi] = model.NewBoolVar(f'x_{pi}')
            s[pi] = model.NewIntVar(0, max(0, self.horizon - dur), f's_{pi}')
            interval = model.NewOptionalIntervalVar(
                s[pi], dur, s[pi] + dur, x[pi], f'intv_{pi}'
            )
            intervals.append(interval)

        # No-overlap
        model.AddNoOverlap(intervals)

        # Objective: maximise reduced cost
        # rc = Σ_pi (cost_pi - π_pi) * x_pi - μ_c
        # (μ_c is a constant; we maximise the variable part)
        obj_terms = []
        for pi in candidates:
            # Scale to integer for CP-SAT: multiply by 100
            net_cost = int((self._patient_costs[pi] - pi_duals[pi]) * 100)
            obj_terms.append(x[pi] * net_cost)

        model.Maximize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.subproblem_time_limit

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        # Extract solution
        selected = []
        starts = []
        for pi in candidates:
            if solver.Value(x[pi]):
                selected.append(pi)
                starts.append(solver.Value(s[pi]))

        if not selected:
            return None

        # Compute actual reduced cost (unscaled)
        rc = sum(self._patient_costs[pi] - pi_duals[pi] for pi in selected) - mu_dual

        if rc <= self.reduced_cost_tol:
            return None

        col = self._make_column(chair_idx, selected, starts)
        col.reduced_cost = rc
        return col

    # ==================================================================
    # INTEGER ROUNDING  (restricted CP-SAT on LP-active columns)
    # ==================================================================
    def _round_to_integer(
        self, lambda_vals: List[float]
    ) -> Dict[int, Tuple[int, int]]:
        """
        Convert fractional LP solution to integer assignments.

        Strategy:
        1. Fix columns with λ_k ≈ 1 (within 0.01).
        2. For remaining patients, solve a small CP-SAT using only
           columns with λ_k > 0.01 as candidate assignments.

        Returns dict mapping patient_idx → (chair_idx, start_minutes).
        """
        assignments: Dict[int, Tuple[int, int]] = {}
        assigned_patients: Set[int] = set()
        used_chairs: Set[int] = set()

        # Phase 1: fix near-integer columns
        for k, lv in enumerate(lambda_vals):
            col = self._columns[k]
            if lv > 0.99 and col.chair_idx not in used_chairs:
                conflict = any(pi in assigned_patients for pi in col.patient_indices)
                if not conflict:
                    for pi, st in zip(col.patient_indices, col.start_times):
                        assignments[pi] = (col.chair_idx, st)
                        assigned_patients.add(pi)
                    used_chairs.add(col.chair_idx)

        # Phase 2: collect fractional candidates
        remaining = [pi for pi in range(self.n_patients) if pi not in assigned_patients]
        if not remaining:
            return assignments

        # Gather candidate (patient, chair, start) triples from fractional columns
        candidates: Dict[int, List[Tuple[int, int, int]]] = {pi: [] for pi in remaining}
        for k, lv in enumerate(lambda_vals):
            if lv < 0.01:
                continue
            col = self._columns[k]
            if col.chair_idx in used_chairs:
                continue
            for pi, st in zip(col.patient_indices, col.start_times):
                if pi in candidates:
                    candidates[pi].append((col.chair_idx, st, k))

        # Phase 3: small CP-SAT for remaining patients
        if ORTOOLS_AVAILABLE and remaining:
            rounding_assignments = self._solve_rounding_cpsat(
                remaining, candidates, used_chairs
            )
            assignments.update(rounding_assignments)

        return assignments

    def _solve_rounding_cpsat(
        self,
        remaining: List[int],
        candidates: Dict[int, List[Tuple[int, int, int]]],
        used_chairs: Set[int],
    ) -> Dict[int, Tuple[int, int]]:
        """
        Small CP-SAT to assign remaining patients using LP-suggested slots.
        """
        model = cp_model.CpModel()

        # Collect all candidate chair indices
        avail_chairs = set()
        for pi in remaining:
            for ci, st, _ in candidates.get(pi, []):
                if ci not in used_chairs:
                    avail_chairs.add(ci)

        # If no candidates from LP, try all available chairs (greedy fallback)
        if not avail_chairs:
            avail_chairs = {ci for ci in range(self.n_chairs) if ci not in used_chairs}

        # Decision variables
        assign = {}   # assign[pi] ∈ {0,1}
        p_chairs = {} # p_chairs[pi][ci] ∈ {0,1}
        starts = {}   # starts[pi] ∈ [0, horizon]
        intervals_per_chair: Dict[int, list] = {ci: [] for ci in avail_chairs}

        for pi in remaining:
            p = self.patients[pi]
            dur = p.expected_duration
            assign[pi] = model.NewBoolVar(f'a_{pi}')
            starts[pi] = model.NewIntVar(0, max(0, self.horizon - dur), f's_{pi}')
            p_chairs[pi] = {}

            feasible_chairs = []
            for ci in avail_chairs:
                c = self.chairs[ci]
                if p.long_infusion and not c.is_recliner:
                    continue
                if self.gnn_valid_pairs is not None and (pi, ci) not in self.gnn_valid_pairs:
                    continue
                cv = model.NewBoolVar(f'pc_{pi}_{ci}')
                p_chairs[pi][ci] = cv
                feasible_chairs.append(cv)

                interval = model.NewOptionalIntervalVar(
                    starts[pi], dur, starts[pi] + dur, cv, f'intv_{pi}_{ci}'
                )
                intervals_per_chair[ci].append(interval)

            if feasible_chairs:
                model.Add(sum(feasible_chairs) == 1).OnlyEnforceIf(assign[pi])
                model.Add(sum(feasible_chairs) == 0).OnlyEnforceIf(assign[pi].Not())
            else:
                model.Add(assign[pi] == 0)

        # No-overlap per chair
        for ci in avail_chairs:
            if intervals_per_chair[ci]:
                model.AddNoOverlap(intervals_per_chair[ci])

        # Objective: maximise assignments weighted by patient cost
        obj = []
        for pi in remaining:
            obj.append(assign[pi] * int(self._patient_costs[pi]))
        model.Maximize(sum(obj))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10

        status = solver.Solve(model)
        result = {}

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for pi in remaining:
                if solver.Value(assign[pi]):
                    for ci, cv in p_chairs[pi].items():
                        if solver.Value(cv):
                            result[pi] = (ci, solver.Value(starts[pi]))
                            break
        return result

    # ==================================================================
    # MAIN CG LOOP
    # ==================================================================
    def solve(
        self,
        warm_start_assignments: Optional[Dict[str, Dict]] = None,
    ) -> CGResult:
        """
        Run the full column generation procedure.

        Parameters
        ----------
        warm_start_assignments : dict or None
            Cached patient assignments from a prior solve
            (maps patient_id → {chair_idx, start}).

        Returns
        -------
        CGResult with assignments, metrics, and diagnostics.
        """
        t0 = time.time()

        if not ORTOOLS_AVAILABLE:
            return CGResult(
                success=False, columns_selected=[], patient_assignments={},
                unassigned=list(range(self.n_patients)), total_cost=0,
                iterations=0, columns_generated=0, solve_time=0,
                lp_bound=0, status='CG_FAILED',
            )

        # Step 1: initial columns
        initial_cols = self.generate_initial_columns(warm_start_assignments)
        if not initial_cols:
            logger.warning("CG: no initial columns generated")
            return CGResult(
                success=False, columns_selected=[], patient_assignments={},
                unassigned=list(range(self.n_patients)), total_cost=0,
                iterations=0, columns_generated=0,
                solve_time=time.time() - t0, lp_bound=0, status='CG_FAILED',
            )

        # Step 2-5: iterate
        best_lp_obj = 0.0
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Solve master LP
            lambda_vals, pi_duals, mu_duals = self._solve_master()
            if lambda_vals is None:
                logger.warning(f"CG: master LP failed at iteration {iterations}")
                break

            lp_obj = sum(
                lv * col.cost
                for lv, col in zip(lambda_vals, self._columns)
            )
            best_lp_obj = max(best_lp_obj, lp_obj)

            # Pricing: generate new columns
            new_cols = 0
            for ci in range(self.n_chairs):
                col = self._solve_pricing(ci, pi_duals, mu_duals[ci])
                if col is not None:
                    new_cols += 1

            if new_cols == 0:
                logger.info(
                    f"CG converged at iteration {iterations}: "
                    f"no improving columns (LP obj={lp_obj:.1f})"
                )
                break

            logger.debug(
                f"CG iter {iterations}: {new_cols} new columns, "
                f"LP obj={lp_obj:.1f}, total columns={len(self._columns)}"
            )

        # Step 6: integer rounding
        lambda_vals, _, _ = self._solve_master()
        if lambda_vals is None:
            return CGResult(
                success=False, columns_selected=[], patient_assignments={},
                unassigned=list(range(self.n_patients)), total_cost=0,
                iterations=iterations,
                columns_generated=len(self._columns),
                solve_time=time.time() - t0, lp_bound=best_lp_obj,
                status='CG_FAILED',
            )

        assignments = self._round_to_integer(lambda_vals)
        unassigned = [pi for pi in range(self.n_patients) if pi not in assignments]

        # Identify selected columns
        selected_cols = [
            col for k, col in enumerate(self._columns)
            if lambda_vals[k] > 0.5
        ]

        total_cost = sum(self._patient_costs[pi] for pi in assignments)
        solve_time = time.time() - t0

        converged = iterations < self.max_iterations
        status = 'CG_OPTIMAL' if converged and not unassigned else (
            'CG_FEASIBLE' if assignments else 'CG_FAILED'
        )
        if not converged and assignments:
            status = 'CG_MAX_ITER'

        logger.info(
            f"CG complete: {len(assignments)}/{self.n_patients} assigned, "
            f"{iterations} iterations, {len(self._columns)} columns, "
            f"{solve_time:.2f}s, status={status}"
        )

        return CGResult(
            success=len(assignments) > 0,
            columns_selected=selected_cols,
            patient_assignments=assignments,
            unassigned=unassigned,
            total_cost=total_cost,
            iterations=iterations,
            columns_generated=len(self._columns),
            solve_time=solve_time,
            lp_bound=best_lp_obj,
            status=status,
        )

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    def get_stats(self) -> Dict:
        """Return diagnostic statistics about the CG state."""
        return {
            'n_patients': self.n_patients,
            'n_chairs': self.n_chairs,
            'columns_generated': len(self._columns),
            'horizon_minutes': self.horizon,
            'max_iterations': self.max_iterations,
            'reduced_cost_tol': self.reduced_cost_tol,
            'gnn_pruning_active': self.gnn_valid_pairs is not None,
        }
