"""
Tests for optimization.uncertainty_optimization — UncertaintyAwareOptimizer.

Covers:
    * Default construction.
    * compute_robust_parameters returns a RobustObjective with per-patient
      worst-case penalties and buffers.
    * Wasserstein worst-case no-show stays within [0.01, 0.95] band and
      exceeds the point estimate.
    * CVaR duration is >= mean when std > 1.
    * Scenario generation: returned list has the requested length and
      every patient is present in every scenario.
    * Schedule robustness evaluation with synthetic scenarios.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.uncertainty_optimization import (  # noqa: E402
    UncertaintyAwareOptimizer,
    UncertaintyProfile,
    RobustObjective,
)
from optimization.optimizer import Patient, ScheduledAppointment  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def today():
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def _make_patients(today, n=5):
    return [
        Patient(
            patient_id=f'UP{i:02d}',
            priority=(i % 4) + 1,
            protocol='R-CHOP',
            expected_duration=60 + 30 * (i % 3),
            postcode='CF14',
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=17),
            noshow_probability=0.05 + 0.02 * i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_construction():
    opt = UncertaintyAwareOptimizer()
    assert pytest.approx(opt.epsilon, rel=1e-12) == 0.05
    assert pytest.approx(opt.alpha, rel=1e-12) == 0.90
    assert opt.n_scenarios == 50


def test_compute_robust_parameters_returns_per_patient(today):
    patients = _make_patients(today, n=5)
    opt = UncertaintyAwareOptimizer(epsilon=0.05, alpha=0.90)

    result = opt.compute_robust_parameters(patients, historical_data=None)

    assert isinstance(result, RobustObjective)
    assert result.method == 'dro_wasserstein'
    assert result.epsilon == 0.05
    assert result.alpha == 0.90

    # One entry per patient in each dict
    for p in patients:
        assert p.patient_id in result.robust_noshow_penalties
        assert p.patient_id in result.robust_duration_buffers
        # Worst-case no-show > point estimate, and duration buffer >= mean
        point_penalty = int(p.noshow_probability * 100)
        assert result.robust_noshow_penalties[p.patient_id] >= point_penalty
        assert result.robust_duration_buffers[p.patient_id] >= p.expected_duration


def test_wasserstein_worst_case_bounds():
    opt = UncertaintyAwareOptimizer(epsilon=0.05)
    # Mean 0.1, std 0.1 → worst-case bounded by 0.95 and > mean.
    worst = opt._wasserstein_worst_case_noshow(mean=0.1, std=0.1)
    assert worst > 0.1
    assert 0.01 <= worst <= 0.95

    # Extreme mean saturates at 0.95 ceiling.
    saturated = opt._wasserstein_worst_case_noshow(mean=0.9, std=0.1)
    assert saturated <= 0.95


def test_cvar_duration_geq_mean_when_variable():
    opt = UncertaintyAwareOptimizer(alpha=0.90)
    # std < 1 → returns mean exactly (explicit branch).
    assert opt._cvar_duration(mean=60.0, std=0.5) == 60.0
    # std >= 1 → CVaR above mean.
    cvar = opt._cvar_duration(mean=120.0, std=20.0)
    assert cvar >= 120.0


def test_generate_scenarios_shape(today):
    patients = _make_patients(today, n=4)
    opt = UncertaintyAwareOptimizer(epsilon=0.05, n_scenarios=7)

    scenarios = opt.generate_scenarios(patients, n_scenarios=7)

    assert len(scenarios) == 7
    for scenario in scenarios:
        assert isinstance(scenario, dict)
        for p in patients:
            assert p.patient_id in scenario
            entry = scenario[p.patient_id]
            assert 'noshow' in entry
            assert 'duration' in entry
            assert 0.01 <= entry['noshow'] <= 0.95
            assert entry['duration'] > 0


def test_evaluate_schedule_robustness_reports_cvar_and_violations(today):
    patients = _make_patients(today, n=3)
    opt = UncertaintyAwareOptimizer(epsilon=0.05, alpha=0.90)

    # Build fake scheduled appointments for every patient.
    appointments = [
        ScheduledAppointment(
            patient_id=p.patient_id,
            chair_id=f'C{i:02d}',
            site_code='VCC',
            start_time=today.replace(hour=9),
            end_time=today.replace(hour=10),
            duration=p.expected_duration,
            priority=p.priority,
            travel_time=15,
        )
        for i, p in enumerate(patients)
    ]

    # Use a deterministic set of scenarios so the reported metrics are stable.
    scenarios = opt.generate_scenarios(patients, n_scenarios=10)
    metrics = opt.evaluate_schedule_robustness(appointments, scenarios)

    for key in ('mean_objective', 'std_objective', 'worst_case',
                'best_case', 'cvar', 'cvar_alpha',
                'violation_probability', 'n_scenarios',
                'epsilon', 'interpretation'):
        assert key in metrics
    assert metrics['n_scenarios'] == 10
    assert 0.0 <= metrics['violation_probability'] <= 1.0
    # Worst <= mean <= best (sanity)
    assert metrics['worst_case'] <= metrics['mean_objective'] <= metrics['best_case']
