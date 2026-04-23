"""
Dedicated tests for optimization.gnn_feasibility.

Covers:
    * Construction defaults.
    * Untrained predictor: pruning still honours hard-infeasibility rule
      (long-infusion → non-recliner always pruned).
    * Safety invariant: every patient keeps >= min_viable_chairs options.
    * Training collection + train() returns True with enough examples.
    * get_stats keys.
    * **Soundness check** — for a small instance CP-SAT can solve, the GNN
      pruner never removes any (patient, chair) pair that CP-SAT actually
      uses in its solution.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.gnn_feasibility import GNNFeasibilityPredictor  # noqa: E402
from optimization.optimizer import Chair, Patient, ScheduleOptimizer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def today():
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def _make_patients(today, n=5):
    """Small mixed cohort: some long-infusion, some normal."""
    return [
        Patient(
            patient_id=f'GP{i:02d}',
            priority=(i % 4) + 1,
            protocol='R-CHOP',
            expected_duration=60 + (i % 3) * 30,
            postcode='CF14',
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=17),
            noshow_probability=0.10 + 0.05 * (i % 3),
            long_infusion=(i == 0),   # patient 0 is long-infusion
            is_urgent=(i == 0),
        )
        for i in range(n)
    ]


def _make_chairs(n_recliners=1, n_standard=2):
    """Small chair pool — 1 recliner + 2 standard = 3 chairs total."""
    chairs = [
        Chair(chair_id=f'VCC-R{i+1:02d}', site_code='VCC', is_recliner=True)
        for i in range(n_recliners)
    ]
    chairs += [
        Chair(chair_id=f'VCC-S{i+1:02d}', site_code='VCC', is_recliner=False)
        for i in range(n_standard)
    ]
    return chairs


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_default_construction():
    gnn = GNNFeasibilityPredictor()
    assert gnn._is_trained is False
    assert gnn._n_solves_seen == 0
    assert gnn.prune_threshold == 0.15
    assert gnn.min_viable_chairs >= 1
    assert gnn.train_every == 5


def test_hard_rule_prunes_long_infusion_from_non_recliners(today):
    """Long-infusion patient: NEVER keep a non-recliner chair."""
    patients = _make_patients(today, n=5)  # patient 0 is long_infusion
    chairs = _make_chairs()

    gnn = GNNFeasibilityPredictor(min_viable_chairs=1)
    valid, prune_count, prune_rate = gnn.prune_assignments(patients, chairs)

    for ci, c in enumerate(chairs):
        if not c.is_recliner:
            assert (0, ci) not in valid, (
                f"Hard-rule violation: patient 0 (long_infusion) paired with "
                f"non-recliner chair {c.chair_id}"
            )

    assert prune_count >= 0
    assert 0.0 <= prune_rate <= 1.0


def test_safety_invariant_min_viable_chairs(today):
    """Every patient must retain at least min_viable_chairs options."""
    patients = _make_patients(today, n=5)
    chairs = _make_chairs(n_recliners=1, n_standard=4)  # 5 chairs total

    gnn = GNNFeasibilityPredictor(prune_threshold=0.5, min_viable_chairs=2)
    valid, _, _ = gnn.prune_assignments(patients, chairs)

    for pi in range(len(patients)):
        opts = sum(1 for (p2, _) in valid if p2 == pi)
        assert opts >= gnn.min_viable_chairs, (
            f"Patient {pi} has only {opts} options, min={gnn.min_viable_chairs}"
        )


def test_collect_training_example_and_train(today):
    """Collecting 3 examples and calling train(min_examples=2) must succeed."""
    pytest.importorskip("sklearn")
    patients = _make_patients(today, n=4)
    chairs = _make_chairs()

    gnn = GNNFeasibilityPredictor(train_every=999)  # disable auto-retrain
    for _ in range(3):
        assignments = {p.patient_id: (i % len(chairs)) for i, p in enumerate(patients)}
        gnn.collect_training_example(patients, chairs, assignments)
    assert gnn._n_solves_seen == 3

    ok = gnn.train(min_examples=2)
    assert ok is True
    assert gnn._is_trained


def test_get_stats_keys():
    gnn = GNNFeasibilityPredictor()
    stats = gnn.get_stats()
    for key in ('is_trained', 'n_solves_seen', 'prune_threshold',
                'min_viable_chairs', 'lifetime_prune_rate',
                'train_every', 'n_mp_rounds'):
        assert key in stats


# ---------------------------------------------------------------------------
# Soundness: pruning must not delete any pair CP-SAT ultimately uses.
# ---------------------------------------------------------------------------

def test_pruner_does_not_remove_pairs_cpsat_uses(today):
    """Small 5-patient x 3-chair instance.

    1. Run CP-SAT (via ScheduleOptimizer with CG disabled by high threshold)
       to find a feasible schedule.
    2. Map each ScheduledAppointment to (patient_idx, chair_idx).
    3. Run the (untrained) GNN pruner on the same (patients, chairs).
    4. Assert that every pair CP-SAT used is present in the pruner's
       valid_pairs set — i.e., the pruner is sound by construction.

    This is a stronger-than-baseline test: it directly verifies the contract
    "pruner must NEVER prune a pair CP-SAT could schedule".
    """
    patients = [
        Patient(
            patient_id=f'SN{i:02d}',
            priority=(i % 3) + 1,
            protocol='R-CHOP',
            expected_duration=60,          # short enough — 3 chairs handle 5 patients
            postcode='CF14',
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=17),
            # Mild: NO long-infusion here, so no hard-rule pruning affects the test.
            long_infusion=False,
        )
        for i in range(5)
    ]
    chairs = _make_chairs(n_recliners=1, n_standard=2)  # 3 chairs, mixed

    # Step 1: CP-SAT via optimizer
    optimizer = ScheduleOptimizer(chairs=chairs)
    optimizer._cg_threshold = 999  # force monolithic CP-SAT
    result = optimizer.optimize(patients, date=today)
    assert result.success
    assert len(result.appointments) > 0

    # Step 2: build CP-SAT's (p_idx, c_idx) set
    chair_id_to_idx = {c.chair_id: i for i, c in enumerate(chairs)}
    pid_to_idx = {p.patient_id: i for i, p in enumerate(patients)}
    cpsat_pairs = set()
    for appt in result.appointments:
        pi = pid_to_idx[appt.patient_id]
        ci = chair_id_to_idx[appt.chair_id]
        cpsat_pairs.add((pi, ci))

    # Step 3: run untrained pruner — still applies hard-rule + safety only.
    gnn = GNNFeasibilityPredictor(prune_threshold=0.15, min_viable_chairs=1)
    valid_pairs, _, _ = gnn.prune_assignments(patients, chairs)

    # Step 4: soundness — every CP-SAT pair must survive pruning.
    for pair in cpsat_pairs:
        assert pair in valid_pairs, (
            f"GNN pruner removed pair {pair} that CP-SAT actually used — "
            f"soundness violation."
        )
