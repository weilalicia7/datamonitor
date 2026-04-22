"""
Tests for ml/auto_scaling_optimizer.py (Dissertation §5.1)
==========================================================

Verify the cascade + parallel + early-stop + greedy-fallback wrapper
using a mockable base optimiser so the suite stays hermetic and fast.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from ml.auto_scaling_optimizer import (
    DEFAULT_CASCADE_BUDGETS,
    DEFAULT_PARALLEL_CONFIGS,
    DEFAULT_PARALLEL_WEIGHT_CONFIGS,
    AttemptRecord,
    AutoScalingOptimizer,
    AutoScalingReport,
    get_auto_scaler,
    greedy_schedule,
    set_auto_scaler,
)
from optimization.optimizer import (
    OptimizationResult,
    Patient,
    ScheduledAppointment,
)


# --------------------------------------------------------------------------- #
# Helpers + fixtures
# --------------------------------------------------------------------------- #


def _patient(pid: str, priority: int = 3, dur: int = 60) -> Patient:
    return Patient(
        patient_id=pid, priority=priority, protocol='X',
        expected_duration=dur, postcode='CF14',
        earliest_time=datetime(2026, 4, 22, 9, 0),
        latest_time=datetime(2026, 4, 22, 17, 0),
    )


def _good_result(patients, **kwargs) -> OptimizationResult:
    from datetime import timedelta
    appts = []
    for i, p in enumerate(patients):
        appts.append(ScheduledAppointment(
            patient_id=p.patient_id,
            chair_id=f'WC-C{i % 3 + 1:02d}', site_code='WC',
            start_time=datetime(2026, 4, 22, 9 + i // 6, (i % 6) * 10),
            end_time=datetime(2026, 4, 22, 9 + i // 6, (i % 6) * 10) + timedelta(minutes=60),
            duration=60, priority=p.priority, travel_time=15,
        ))
    return OptimizationResult(
        success=True, appointments=appts, unscheduled=[],
        metrics={'objective_score': 0.85},
        solve_time=0.2, status='OPTIMAL',
    )


def _fail_result(patients, **kwargs) -> OptimizationResult:
    return OptimizationResult(
        success=False, appointments=[],
        unscheduled=[p.patient_id for p in patients],
        metrics={}, solve_time=0.1, status='INFEASIBLE',
    )


@pytest.fixture
def tmp_autoscale_dir(tmp_path):
    return tmp_path / "autoscale"


@pytest.fixture
def patients():
    return [_patient(f'P{i:03d}', priority=(i % 3) + 1) for i in range(12)]


# --------------------------------------------------------------------------- #
# 1. Successful path
# --------------------------------------------------------------------------- #


class TestSuccessPath:
    def test_parallel_success_returns_report(self, patients, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            storage_dir=tmp_autoscale_dir,
        )
        result, report = opt.optimize(patients)
        assert result.success is True
        assert len(result.appointments) == len(patients)
        assert report.winner_stage == 'parallel'
        assert not report.greedy_fallback
        assert len(report.attempts) >= 1

    def test_winner_config_is_one_of_builtins(self, patients, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            storage_dir=tmp_autoscale_dir,
        )
        _, report = opt.optimize(patients)
        names = {w['name'] for w in DEFAULT_PARALLEL_WEIGHT_CONFIGS}
        assert report.winner_config in names


# --------------------------------------------------------------------------- #
# 2. Cascade fallback
# --------------------------------------------------------------------------- #


class TestCascade:
    def test_cascade_fires_when_parallel_disabled(self, patients,
                                                  tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            storage_dir=tmp_autoscale_dir,
            enable_parallel=False,
        )
        _, report = opt.optimize(patients)
        stages = {a.stage for a in report.attempts}
        assert 'cascade' in stages
        assert report.winner_stage == 'cascade'

    def test_cascade_budget_order_respected(self, patients, tmp_autoscale_dir):
        calls: list = []

        def _spy(patients, time_limit_seconds=60, **kwargs):
            calls.append(time_limit_seconds)
            return _fail_result(patients)

        opt = AutoScalingOptimizer(
            base_optimizer=_spy,
            storage_dir=tmp_autoscale_dir,
            enable_parallel=False,
            enable_greedy_fallback=False,
        )
        opt.optimize(patients)
        # Excludes the initial budget (used only in parallel) — cascade uses 1:
        # the first cascade call should use cascade_budgets[1] = 2s
        assert calls  # something was called
        assert calls[0] == max(int(DEFAULT_CASCADE_BUDGETS[1]), 1)


# --------------------------------------------------------------------------- #
# 3. Greedy fallback
# --------------------------------------------------------------------------- #


class TestGreedyFallback:
    def test_greedy_triggered_when_all_cpsat_fail(self, patients,
                                                  tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_fail_result,
            storage_dir=tmp_autoscale_dir,
        )
        result, report = opt.optimize(patients)
        assert report.greedy_fallback is True
        assert report.winner_stage == 'greedy'
        assert result.success is True  # greedy places at least some
        assert result.status == 'GREEDY_FALLBACK'

    def test_greedy_places_all_if_capacity_available(self, patients):
        appts, uns = greedy_schedule(patients)
        assert len(appts) + len(uns) == len(patients)
        assert all(
            a.start_time < a.end_time for a in appts
        )

    def test_greedy_respects_priority_ordering(self, patients):
        appts, _ = greedy_schedule(patients)
        # First chair should be occupied by a priority-1 patient
        by_start = sorted(appts, key=lambda a: a.start_time)
        if by_start:
            assert by_start[0].priority == 1


# --------------------------------------------------------------------------- #
# 4. Early stopping
# --------------------------------------------------------------------------- #


class TestEarlyStopping:
    def test_early_stop_flag_set_when_solve_time_short(self, patients,
                                                      tmp_autoscale_dir):
        # _good_result returns solve_time=0.2s with time_budget=5s
        # so early_stopped should be true (solve < 0.9 * budget AND success).
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            parallel_time_limit_s=5.0,
            storage_dir=tmp_autoscale_dir,
        )
        _, report = opt.optimize(patients)
        assert any(a.early_stopped for a in report.attempts)


# --------------------------------------------------------------------------- #
# 5. Parallel race
# --------------------------------------------------------------------------- #


class TestParallelRace:
    def test_four_configs_raced_by_default(self, patients, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            storage_dir=tmp_autoscale_dir,
        )
        _, report = opt.optimize(patients)
        parallel_attempts = [a for a in report.attempts if a.stage == 'parallel']
        assert len(parallel_attempts) == DEFAULT_PARALLEL_CONFIGS

    def test_best_scheduled_wins(self, patients, tmp_autoscale_dir):
        """When configs produce different n_scheduled, the one with
        the most scheduled must win."""
        call_counter = {'i': 0}

        def _variable(patients, time_limit_seconds=60, **kwargs):
            # Return 10, 11, 12, 9 scheduled for the four configs
            i = call_counter['i']; call_counter['i'] += 1
            n_sched = [10, 11, 12, 9][i % 4]
            r = _good_result(patients)
            r.appointments = r.appointments[:n_sched]
            r.unscheduled = [p.patient_id for p in patients[n_sched:]]
            return r

        opt = AutoScalingOptimizer(
            base_optimizer=_variable,
            storage_dir=tmp_autoscale_dir,
        )
        _, report = opt.optimize(patients)
        assert report.winner_n_scheduled == 12


# --------------------------------------------------------------------------- #
# T2.2 regression: parallel race must not cross-contaminate weights
# --------------------------------------------------------------------------- #


class TestParallelRaceWeightIsolation:
    """Pre-fix the parallel race called set_weights under a short lock and
    then released the lock for the solve, so worker A could be midway
    through a solve when worker B overwrote shared weights.  These tests
    pin the corrected behaviour:

      - solve_with_weights path: each worker SEES the weights it asked for,
        with no cross-contamination, even when the solve is slow.
      - legacy set_weights path: race fix forces serialisation so
        weight-vs-result correspondence holds.
    """

    def _make_solve_with_weights(self, observed):
        """Return a closure that records (cfg_name, observed_weights).

        We expose the weights it was given by stuffing them into the result's
        `metrics` dict, so the test can prove the worker that asked for
        config X actually solved with config X's weights.
        """
        import time as _time

        def _solve(patients, weights, time_limit_s):
            cfg_name = weights.get('__config_name__', 'unknown')
            # Force concurrent overlap: holds for ~50ms so all 4 workers
            # are in flight simultaneously, exposing any race.
            _time.sleep(0.05)
            r = _good_result(patients)
            r.metrics = dict(r.metrics or {})
            r.metrics['observed_weights_priority'] = weights.get('priority')
            r.metrics['cfg_name'] = cfg_name
            observed.append((cfg_name, weights.get('priority')))
            return r
        return _solve

    def test_per_call_weights_no_cross_contamination(
        self, patients, tmp_autoscale_dir,
    ):
        observed = []
        # Build configs whose `priority` weight is unique per cfg so we can
        # detect cross-contamination directly.
        configs = [
            {'name': 'A', '__config_name__': 'A', 'priority': 0.1, 'utilization': 0.9},
            {'name': 'B', '__config_name__': 'B', 'priority': 0.2, 'utilization': 0.8},
            {'name': 'C', '__config_name__': 'C', 'priority': 0.3, 'utilization': 0.7},
            {'name': 'D', '__config_name__': 'D', 'priority': 0.4, 'utilization': 0.6},
        ]
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,                    # cascade fallback
            solve_with_weights=self._make_solve_with_weights(observed),
            weight_configs=configs,
            parallel_configs=4,
            parallel_time_limit_s=2.0,
            enable_cascade=False,
            enable_greedy_fallback=False,
            storage_dir=tmp_autoscale_dir,
        )
        opt.optimize(patients)

        # Each cfg should have been observed exactly once with its OWN
        # priority weight — never another cfg's.
        expected = {('A', 0.1), ('B', 0.2), ('C', 0.3), ('D', 0.4)}
        assert set(observed) == expected, (
            f"weight cross-contamination detected: observed={observed}"
        )

    def test_legacy_set_weights_serialises_under_lock(
        self, patients, tmp_autoscale_dir,
    ):
        """Without solve_with_weights, the race must serialise — each
        worker holds the lock from set_weights through the solve."""
        import time as _time
        import threading as _t

        active = {'count': 0, 'max': 0, 'lock': _t.Lock()}
        weights_seen_by_solve: list = []
        current_weights = {'val': None}

        def _set_weights(w):
            current_weights['val'] = w.get('priority')

        def _slow_solve(patients, time_limit_seconds=60, **kwargs):
            with active['lock']:
                active['count'] += 1
                active['max'] = max(active['max'], active['count'])
            try:
                _time.sleep(0.04)
                weights_seen_by_solve.append(current_weights['val'])
                return _good_result(patients)
            finally:
                with active['lock']:
                    active['count'] -= 1

        configs = [
            {'name': 'A', 'priority': 0.1},
            {'name': 'B', 'priority': 0.2},
            {'name': 'C', 'priority': 0.3},
            {'name': 'D', 'priority': 0.4},
        ]
        opt = AutoScalingOptimizer(
            base_optimizer=_slow_solve,
            set_weights=_set_weights,
            weight_configs=configs,
            parallel_configs=4,
            parallel_time_limit_s=1.0,
            enable_cascade=False,
            enable_greedy_fallback=False,
            storage_dir=tmp_autoscale_dir,
        )
        opt.optimize(patients)

        # Critical: no two solves were active concurrently.  If max > 1,
        # the lock did not cover the solve and the race is still live.
        assert active['max'] == 1, (
            f"legacy path leaked concurrency: max active={active['max']}"
        )
        # Each solve saw a distinct priority value (no double-overwrite mid-solve).
        assert sorted(weights_seen_by_solve) == [0.1, 0.2, 0.3, 0.4]

    def test_solve_with_weights_truly_parallel(
        self, patients, tmp_autoscale_dir,
    ):
        """When solve_with_weights is provided, workers run truly in
        parallel — total wall time is dominated by ONE solve, not all four."""
        import time as _time
        import threading as _t

        active = {'count': 0, 'max': 0, 'lock': _t.Lock()}

        def _slow_solve(patients, weights, time_limit_s):
            with active['lock']:
                active['count'] += 1
                active['max'] = max(active['max'], active['count'])
            try:
                _time.sleep(0.05)
                return _good_result(patients)
            finally:
                with active['lock']:
                    active['count'] -= 1

        configs = [{'name': str(i), 'priority': 0.1 * i} for i in range(4)]
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            solve_with_weights=_slow_solve,
            weight_configs=configs,
            parallel_configs=4,
            parallel_time_limit_s=2.0,
            enable_cascade=False,
            enable_greedy_fallback=False,
            storage_dir=tmp_autoscale_dir,
        )
        t0 = _time.perf_counter()
        opt.optimize(patients)
        wall = _time.perf_counter() - t0

        # Sequential would be ~0.20 s (4 × 0.05); parallel ~0.05 s.  Allow
        # generous slack for thread-pool overhead but still well under serial.
        assert wall < 0.18, f"workers serialised — wall={wall:.3f}s"
        # And we DID see >1 worker active at once.
        assert active['max'] >= 2, (
            f"expected concurrent workers, max={active['max']}"
        )


# --------------------------------------------------------------------------- #
# 6. Persistence + status + config
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_log_written(self, patients, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            storage_dir=tmp_autoscale_dir,
        )
        opt.optimize(patients)
        log = tmp_autoscale_dir / "runs.jsonl"
        assert log.exists()
        rec = json.loads(log.read_text().strip().splitlines()[-1])
        assert 'winner_stage' in rec
        assert 'attempts' in rec

    def test_status_counters(self, patients, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            storage_dir=tmp_autoscale_dir,
        )
        before = opt.status()['total_runs']
        opt.optimize(patients)
        after = opt.status()['total_runs']
        assert after == before + 1

    def test_last_cached(self, patients, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            storage_dir=tmp_autoscale_dir,
        )
        assert opt.last() is None
        opt.optimize(patients)
        assert opt.last() is not None

    def test_update_config_round_trip(self, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(storage_dir=tmp_autoscale_dir)
        cfg = opt.update_config(
            cascade_budgets=[3.0, 1.0],
            parallel_configs=2,
            parallel_time_limit_s=2.0,
            early_stop_gap=0.02,
            enable_greedy_fallback=False,
        )
        assert cfg['cascade_budgets'] == [3.0, 1.0]
        assert cfg['parallel_configs'] == 2
        assert cfg['early_stop_gap'] == 0.02
        assert cfg['enable_greedy_fallback'] is False


class TestSingleton:
    def test_get_set(self, tmp_path):
        o = AutoScalingOptimizer(storage_dir=tmp_path / "a")
        set_auto_scaler(o)
        assert get_auto_scaler() is o


# --------------------------------------------------------------------------- #
# 7. JSON round-trip
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_report_to_dict_is_json_safe(self, patients, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=_good_result,
            storage_dir=tmp_autoscale_dir,
        )
        _, report = opt.optimize(patients)
        back = json.loads(json.dumps(report.to_dict(), default=str))
        assert isinstance(back['attempts'], list)
        assert back['winner_stage'] in {'parallel', 'cascade', 'greedy'}


# --------------------------------------------------------------------------- #
# 8. No base optimizer (degenerate)
# --------------------------------------------------------------------------- #


class TestNoBaseOptimizer:
    def test_falls_through_to_greedy(self, patients, tmp_autoscale_dir):
        opt = AutoScalingOptimizer(
            base_optimizer=None,
            storage_dir=tmp_autoscale_dir,
        )
        result, report = opt.optimize(patients)
        assert report.winner_stage == 'greedy'
        assert result.status == 'GREEDY_FALLBACK'
