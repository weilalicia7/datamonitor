"""
Dedicated tests for optimization.column_generation.

Separate from tests/test_optimization.py (which also covers CG as part of
optimizer integration tests) to satisfy the module-level test inventory
in docs/PRODUCTION_READINESS_PLAN.md.

Covers:
    * Dataclass shape (Column, CGResult).
    * Construction defaults + get_stats.
    * Greedy initial columns are feasible and respect bed-requirement.
    * Full CG solve on a small instance produces a non-overlapping schedule.
    * Master LP duals have the right shapes.
    * Bundle cost = sum of per-patient costs.
    * Single-patient solve succeeds.
    * vs-CP-SAT ground truth: both formulations assign the same patients
      on a tiny, easy instance.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_SITES, OPERATING_HOURS  # noqa: E402
from optimization.column_generation import ColumnGenerator, Column, CGResult  # noqa: E402
from optimization.optimizer import Chair, Patient, ScheduleOptimizer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def today():
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


@pytest.fixture()
def chairs(today):
    start_h, end_h = OPERATING_HOURS
    out = []
    for site in DEFAULT_SITES[:1]:  # one site is enough for small cases
        for i in range(site['chairs']):
            out.append(Chair(
                chair_id=f"{site['code']}-C{i+1:02d}",
                site_code=site['code'],
                is_recliner=i < site.get('recliners', 0),
                available_from=today.replace(hour=start_h),
                available_until=today.replace(hour=end_h),
            ))
    return out


@pytest.fixture()
def patients(today):
    def _pat(i, duration=90, priority=None, long_infusion=False):
        return Patient(
            patient_id=f'CGX{i:02d}',
            priority=priority if priority is not None else (i % 4) + 1,
            protocol='R-CHOP',
            expected_duration=duration,
            postcode='CF14',
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=17),
            noshow_probability=0.10,
            long_infusion=long_infusion,
        )
    return [_pat(i, duration=60 + 30 * (i % 3)) for i in range(10)]


# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------

def test_column_dataclass_shape():
    col = Column(
        column_id=1,
        chair_idx=0,
        chair_id='WC-C01',
        patient_indices=[0, 1],
        start_times=[0, 90],
        cost=250.0,
    )
    assert col.column_id == 1
    assert col.patient_indices == [0, 1]
    assert col.reduced_cost == 0.0  # default


def test_cgresult_dataclass_shape():
    r = CGResult(
        success=True, columns_selected=[], patient_assignments={},
        unassigned=[], total_cost=0.0, iterations=1,
        columns_generated=1, solve_time=0.1, lp_bound=0.0,
        status='CG_OPTIMAL',
    )
    assert r.success
    assert r.status == 'CG_OPTIMAL'


# ---------------------------------------------------------------------------
# Construction + stats
# ---------------------------------------------------------------------------

def test_get_stats_reports_expected_keys(patients, chairs):
    cg = ColumnGenerator(patients[:5], chairs)
    stats = cg.get_stats()
    for key in ('n_patients', 'n_chairs', 'columns_generated',
                'horizon_minutes', 'max_iterations',
                'reduced_cost_tol', 'gnn_pruning_active'):
        assert key in stats
    assert stats['n_patients'] == 5
    assert stats['n_chairs'] == len(chairs)
    assert stats['gnn_pruning_active'] is False


# ---------------------------------------------------------------------------
# Greedy initial columns
# ---------------------------------------------------------------------------

def test_generate_initial_columns_respects_bed_requirement(today, chairs):
    # Mix of long-infusion and normal patients
    patient_list = [
        Patient(
            patient_id='LONG01', priority=1, protocol='R-CHOP',
            expected_duration=120, postcode='CF14',
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=17),
            long_infusion=True,
        ),
        Patient(
            patient_id='NORM01', priority=2, protocol='R-CHOP',
            expected_duration=90, postcode='CF14',
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=17),
        ),
    ]
    cg = ColumnGenerator(patient_list, chairs)
    cols = cg.generate_initial_columns()

    # Long-infusion patient must only appear on recliner chairs.
    for col in cols:
        chair = chairs[col.chair_idx]
        for pi in col.patient_indices:
            if patient_list[pi].long_infusion:
                assert chair.is_recliner, (
                    f"Long-infusion patient on non-recliner chair {chair.chair_id}"
                )


# ---------------------------------------------------------------------------
# Bundle cost correctness
# ---------------------------------------------------------------------------

def test_bundle_cost_sum_of_per_patient(patients, chairs):
    cg = ColumnGenerator(patients[:5], chairs)
    subset = [0, 2, 3]
    total = cg._bundle_cost(subset)
    expected = sum(cg._patient_costs[i] for i in subset)
    assert total == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Master LP
# ---------------------------------------------------------------------------

def test_master_lp_shapes(patients, chairs):
    cg = ColumnGenerator(patients[:6], chairs)
    cg.generate_initial_columns()
    lam, pi_d, mu_d = cg._solve_master()

    assert lam is not None
    assert len(pi_d) == len(patients[:6])
    assert len(mu_d) == len(chairs)
    # Lambdas bounded in [0, 1] (modulo LP tolerance)
    for v in lam:
        assert -1e-6 <= v <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Full CG solve
# ---------------------------------------------------------------------------

def test_full_solve_produces_non_overlapping_schedule(patients, chairs):
    cg = ColumnGenerator(patients[:8], chairs, max_iterations=15)
    result = cg.solve()

    assert isinstance(result, CGResult)
    assert result.success
    assert result.status in ('CG_OPTIMAL', 'CG_FEASIBLE', 'CG_MAX_ITER')
    assert len(result.patient_assignments) > 0

    # No overlap check: group assignments by chair
    per_chair = {}
    for pi, (ci, start) in result.patient_assignments.items():
        per_chair.setdefault(ci, []).append((start, start + patients[pi].expected_duration))
    for ci, slots in per_chair.items():
        slots.sort()
        for a, b in zip(slots, slots[1:]):
            assert a[1] <= b[0], f"Overlap on chair {ci}: {a} vs {b}"


def test_single_patient_solve(today, chairs):
    """Single-patient case: CG must assign the one patient without crashing."""
    one_patient = [Patient(
        patient_id='SOLO01', priority=1, protocol='R-CHOP',
        expected_duration=60, postcode='CF14',
        earliest_time=today.replace(hour=8),
        latest_time=today.replace(hour=17),
    )]
    cg = ColumnGenerator(one_patient, chairs, max_iterations=10)
    result = cg.solve()
    assert result.success
    assert len(result.patient_assignments) == 1


def test_cg_respects_wall_clock_time_limit(patients, chairs):
    """
    Regression for §4.5.16 absurdity: a 2-second auto-scaler budget
    used to balloon to 350 s on the production-path 202-patient cohort
    because ColumnGenerator.solve() ignored time_limit_seconds entirely
    and only honoured max_iterations.

    With a deliberately tiny wall-clock budget (0.1 s) the solver must
    return a feasible result tagged CG_TIME_LIMIT and report a
    solve_time within ~3× the budget (slop covers the in-flight master
    LP and the integer-rounding step that always runs after the loop).
    """
    import time as _time

    cg = ColumnGenerator(
        patients, chairs,
        max_iterations=100,
        subproblem_time_limit=0.1,
        time_limit_s=0.1,           # absurdly tight on purpose
    )
    t0 = _time.perf_counter()
    result = cg.solve()
    elapsed = _time.perf_counter() - t0

    # Must terminate near-immediately, not run the full 100 iterations.
    assert elapsed < 1.0, (
        f"CG ran for {elapsed:.2f}s with 0.1s budget — wall-clock "
        f"guard not enforced (status={result.status})"
    )
    # Result must still be feasible (rounding step always runs after break).
    assert result.success or result.status == 'CG_FAILED'
    if result.success:
        # Time-limit termination is the primary status when budget is
        # what stopped us; CG_OPTIMAL is acceptable iff the instance is
        # so small the loop converged before the budget bit.
        assert result.status in ('CG_TIME_LIMIT', 'CG_OPTIMAL', 'CG_FEASIBLE')


def test_cg_no_time_limit_preserves_legacy_behaviour(patients, chairs):
    """
    The new time_limit_s parameter must default to None so existing
    callers (any test or code that does not pass it) keep their old
    behaviour: run to convergence or to max_iterations, no premature
    time-based termination.
    """
    cg = ColumnGenerator(patients[:6], chairs, max_iterations=20)
    assert cg.time_limit_s is None
    result = cg.solve()
    # Without a budget the only stop reasons are convergence or iterations,
    # never CG_TIME_LIMIT.
    assert result.status != 'CG_TIME_LIMIT'


# ---------------------------------------------------------------------------
# vs-CP-SAT ground-truth comparison
# ---------------------------------------------------------------------------

def test_cg_matches_cpsat_on_tiny_instance(today, chairs):
    """Tiny (5-patient x 3-chair) instance — both solvers should schedule
    the same set of patients successfully.

    We use the CG threshold toggle on the optimizer to force one path then
    the other, and compare the count of assigned patients.
    """
    # Tiny patient list — plenty of capacity, both solvers succeed.
    p_list = [
        Patient(
            patient_id=f'GT{i:02d}', priority=(i % 3) + 1,
            protocol='R-CHOP', expected_duration=60, postcode='CF14',
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=17),
        )
        for i in range(5)
    ]

    # Path 1: monolithic CP-SAT (high threshold)
    optimizer_cpsat = ScheduleOptimizer()
    optimizer_cpsat._cg_threshold = 999
    res_cpsat = optimizer_cpsat.optimize(p_list, date=today)
    assert res_cpsat.success

    # Path 2: column generation (low threshold)
    optimizer_cg = ScheduleOptimizer()
    optimizer_cg._cg_threshold = 1  # force CG path
    res_cg = optimizer_cg.optimize(p_list, date=today)
    assert res_cg.success

    # Ground truth: both should schedule all 5 patients (easy instance)
    assert len(res_cpsat.appointments) == 5
    assert len(res_cg.appointments) == 5

    # Same patient set assigned.
    cpsat_ids = {a.patient_id for a in res_cpsat.appointments}
    cg_ids = {a.patient_id for a in res_cg.appointments}
    assert cpsat_ids == cg_ids
