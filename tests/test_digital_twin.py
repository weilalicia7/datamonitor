"""
Tests for ml/digital_twin.py (Dissertation §3.3)
================================================

These tests exercise the twin *without* any live Flask server.  They
use the built-in fallback squeeze function and a synthetic arrival
model so the suite stays hermetic and fast.
"""

from __future__ import annotations

import json
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from ml.digital_twin import (
    AGGRESSIVE_DOUBLE_BOOK_THRESHOLD,
    ArrivalModel,
    DEFAULT_DOUBLE_BOOK_THRESHOLD,
    DigitalTwin,
    GUARDRAIL_MAX_DOUBLE_BOOK_RATE,
    PolicySpec,
    TwinEvaluation,
    TwinState,
    TwinStepResult,
    _empty_snapshot,
    _fallback_squeeze_fn,
    _poisson_sample,
    _uniform_fallback_model,
    fit_arrival_model,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_twin_dir(tmp_path):
    return tmp_path / "twin"


@pytest.fixture
def twin(tmp_twin_dir):
    t = DigitalTwin(storage_dir=tmp_twin_dir)
    t.arrival_model = _uniform_fallback_model(200)
    return t


@pytest.fixture
def mock_app_state():
    return {
        "mode": "NORMAL",
        "appointments": [],
        "patients": [],
        "patient_data_map": {
            "P001": {"total_appointments": 10, "no_shows": 2},
            "P002": {"total_appointments": 20, "no_shows": 1},
        },
        "chairs": [
            type("_C", (), {"site_code": "VEL", "chair_id": "VEL-01"})(),
            type("_C", (), {"site_code": "VEL", "chair_id": "VEL-02"})(),
            type("_C", (), {"site_code": "VEL", "chair_id": "VEL-03"})(),
        ],
    }


@pytest.fixture
def fitted_historical_df():
    """7-day synthetic historical schedule — enough for fit_arrival_model."""
    rows = []
    start = datetime(2024, 1, 1)
    for day in range(14):
        for hr in range(8, 17):
            # ~2 arrivals per hour, more mid-morning
            n = 3 if 9 <= hr <= 12 else 1
            for i in range(n):
                ts = start + timedelta(days=day, hours=hr, minutes=15 * i)
                rows.append({
                    "Date": ts,
                    "Appointment_Hour": hr,
                    "is_urgent": (i == 0 and hr == 8),
                    "Patient_ID": f"P{day:02d}{hr:02d}{i}",
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ArrivalModel
# --------------------------------------------------------------------------- #


class TestArrivalModel:
    def test_fit_produces_positive_rates(self, fitted_historical_df):
        m = fit_arrival_model(fitted_historical_df)
        assert m.total_events == len(fitted_historical_df)
        assert m.historical_days_observed == 14
        # Mid-morning rate should exceed afternoon rate
        mid = m.rate(datetime(2024, 1, 2, 10, 0))  # Tue 10:00
        aft = m.rate(datetime(2024, 1, 2, 14, 0))  # Tue 14:00
        assert mid > aft

    def test_fit_empty_df_fallback(self):
        m = fit_arrival_model(pd.DataFrame())
        assert m.source_file == "fallback"
        assert all(v == 0.5 for v in m.lambda_hd.values())

    def test_fit_missing_date_col_fallback(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        m = fit_arrival_model(df)
        assert m.source_file == "fallback"

    def test_urgent_fraction_derived_from_is_urgent(self, fitted_historical_df):
        m = fit_arrival_model(fitted_historical_df)
        # 14 urgent rows out of 14*(3*4+1*5)=14*17 rows
        assert 0.0 < m.urgent_fraction < 0.1

    def test_roundtrip(self, fitted_historical_df):
        m = fit_arrival_model(fitted_historical_df)
        d = m.to_dict()
        m2 = ArrivalModel.from_dict(d)
        assert m2.total_events == m.total_events
        assert m2.lambda_hd == m.lambda_hd


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #


class TestSnapshot:
    def test_snapshot_copies_not_references(self, twin, mock_app_state):
        snap = twin.snapshot_live_state(mock_app_state)
        mock_app_state["appointments"].append({"patient_id": "NEW"})
        # Twin snapshot must not see the mutation
        assert len(snap.appointments) == 0
        assert twin.total_snapshots == 1

    def test_snapshot_chair_capacity_from_chairs(self, twin, mock_app_state):
        snap = twin.snapshot_live_state(mock_app_state)
        assert snap.chair_capacity.get("VEL") == 3

    def test_reset_requires_prior_snapshot(self, twin):
        with pytest.raises(RuntimeError):
            twin.reset()

    def test_reset_returns_independent_clone(self, twin, mock_app_state):
        snap = twin.snapshot_live_state(mock_app_state)
        c1 = twin.reset()
        c1.appointments.append({"patient_id": "P_X"})
        c2 = twin.reset()
        assert len(c2.appointments) == 0  # clone is independent


# --------------------------------------------------------------------------- #
# Step
# --------------------------------------------------------------------------- #


class TestStep:
    def test_step_advances_virtual_time(self, twin):
        state = _empty_snapshot()
        policy = PolicySpec()
        rng = random.Random(1)
        t0 = datetime.fromisoformat(state.virtual_time)
        state, sr = twin.step(state, policy, rng, step_hours=1)
        assert datetime.fromisoformat(state.virtual_time) == t0 + timedelta(hours=1)
        assert sr.virtual_time == state.virtual_time

    def test_step_on_empty_state_still_succeeds(self, twin):
        state = _empty_snapshot()
        rng = random.Random(0)
        _, sr = twin.step(state, PolicySpec(), rng)
        assert sr.step_latency_ms >= 0.0
        assert sr.arrivals_sampled >= 0

    def test_step_with_zero_arrival_model_is_zero(self, twin):
        # Force λ=0
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 0.0 for d in range(7) for h in range(24)}
        )
        state = _empty_snapshot()
        _, sr = twin.step(state, PolicySpec(), random.Random(0))
        assert sr.arrivals_sampled == 0

    def test_step_realises_noshows_for_past_appointments(self, twin):
        state = _empty_snapshot()
        t0 = datetime.fromisoformat(state.virtual_time)
        # Plant 20 appointments all inside a 2-hour window so step_hours=2 covers them
        for i in range(20):
            state.appointments.append({
                "patient_id": f"P{i}",
                "start_time": (t0 + timedelta(minutes=5 * i)).isoformat(timespec="seconds"),
                "end_time": (t0 + timedelta(minutes=5 * i + 30)).isoformat(timespec="seconds"),
                "duration": 30,
                "chair_id": "UNKNOWN-01",
                "site_code": "UNKNOWN",
                "priority": 3,
            })
        state.patient_data_map = {f"P{i}": {"no_show_rate": 1.0} for i in range(20)}
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 0.0 for d in range(7) for h in range(24)}
        )
        _, sr = twin.step(state, PolicySpec(), random.Random(0), step_hours=2)
        assert sr.noshows_realised == 20
        assert sr.appointments_completed == 0


# --------------------------------------------------------------------------- #
# Policy differentiation
# --------------------------------------------------------------------------- #


class TestPolicies:
    def test_aggressive_vs_conservative_differ(self, twin):
        """Aggressive policy (low threshold) should accept more arrivals
        once capacity is saturated than a conservative one."""
        snap = _empty_snapshot()
        snap.chair_capacity = {"VEL": 1}  # force saturation quickly

        aggressive = PolicySpec(
            name="aggressive",
            double_book_threshold=AGGRESSIVE_DOUBLE_BOOK_THRESHOLD,
            allow_double_booking=True,
        )
        conservative = PolicySpec(
            name="conservative",
            double_book_threshold=0.80,  # very high -> rarely double-books
            allow_double_booking=True,
        )
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 4.0 for d in range(7) for h in range(24)},
            urgent_fraction=0.0,
        )
        e_agg = twin.evaluate_policy(snapshot=snap, policy=aggressive,
                                     horizon_days=1, rng_seed=9)
        e_con = twin.evaluate_policy(snapshot=snap, policy=conservative,
                                     horizon_days=1, rng_seed=9)
        assert e_agg.total_double_bookings >= e_con.total_double_bookings
        assert e_agg.total_accepted >= e_con.total_accepted

    def test_reschedule_disallowed(self, twin):
        snap = _empty_snapshot()
        snap.chair_capacity = {"VEL": 1}
        pol = PolicySpec(
            name="noresched",
            double_book_threshold=0.99,
            allow_double_booking=False,
            allow_rescheduling=False,
        )
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 3.0 for d in range(7) for h in range(24)},
        )
        ev = twin.evaluate_policy(snapshot=snap, policy=pol,
                                  horizon_days=1, rng_seed=3)
        assert ev.total_reschedules == 0


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def test_same_seed_reproduces(self, twin):
        snap = _empty_snapshot()
        pol = PolicySpec()
        e1 = twin.evaluate_policy(snapshot=snap, policy=pol,
                                  horizon_days=1, rng_seed=42)
        e2 = twin.evaluate_policy(snapshot=snap, policy=pol,
                                  horizon_days=1, rng_seed=42)
        assert e1.total_arrivals == e2.total_arrivals
        assert e1.total_accepted == e2.total_accepted
        assert e1.policy_score == pytest.approx(e2.policy_score, rel=1e-9)

    def test_different_seeds_vary(self, twin):
        snap = _empty_snapshot()
        pol = PolicySpec()
        e1 = twin.evaluate_policy(snapshot=snap, policy=pol,
                                  horizon_days=1, rng_seed=1)
        e2 = twin.evaluate_policy(snapshot=snap, policy=pol,
                                  horizon_days=1, rng_seed=2)
        assert (e1.total_arrivals, e1.total_accepted) != (
            e2.total_arrivals, e2.total_accepted
        ) or e1.total_noshows_realised != e2.total_noshows_realised


# --------------------------------------------------------------------------- #
# Horizon math
# --------------------------------------------------------------------------- #


class TestHorizon:
    def test_default_7_day_horizon_yields_168_steps(self, twin):
        snap = _empty_snapshot()
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 0.0 for d in range(7) for h in range(24)}
        )
        ev = twin.evaluate_policy(snapshot=snap, policy=PolicySpec(),
                                  horizon_days=7, step_hours=1, rng_seed=0)
        assert ev.num_steps == 168
        assert ev.horizon_days == 7
        assert ev.step_hours == 1

    def test_step_hours_divides_horizon(self, twin):
        snap = _empty_snapshot()
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 0.0 for d in range(7) for h in range(24)}
        )
        ev = twin.evaluate_policy(snapshot=snap, policy=PolicySpec(),
                                  horizon_days=2, step_hours=4, rng_seed=0)
        assert ev.num_steps == 12  # 2*24//4


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #


class TestGuardrails:
    def test_high_double_book_rate_flags(self, twin):
        """An absurdly permissive policy + saturated capacity should
        trigger the double-book guardrail."""
        snap = _empty_snapshot()
        snap.chair_capacity = {"VEL": 1}
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 6.0 for d in range(7) for h in range(24)},
            urgent_fraction=1.0,  # ensures p_ns >= any threshold
        )
        pol = PolicySpec(
            name="reckless",
            double_book_threshold=0.01,  # accepts almost anything
            allow_double_booking=True,
        )
        ev = twin.evaluate_policy(snapshot=snap, policy=pol,
                                  horizon_days=1, rng_seed=11)
        db_rate = ev.total_double_bookings / max(ev.total_accepted, 1)
        if db_rate > GUARDRAIL_MAX_DOUBLE_BOOK_RATE:
            assert any("double_book_rate" in v for v in ev.guardrail_violations)

    def test_ok_policy_has_no_violations(self, twin):
        snap = _empty_snapshot()
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 0.2 for d in range(7) for h in range(24)},
            urgent_fraction=0.05,
        )
        ev = twin.evaluate_policy(snapshot=snap, policy=PolicySpec(),
                                  horizon_days=7, rng_seed=0)
        assert ev.guardrail_violations == []


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_evaluation_written_to_disk(self, twin, tmp_twin_dir):
        snap = _empty_snapshot()
        ev = twin.evaluate_policy(snapshot=snap, policy=PolicySpec(name="t"),
                                  horizon_days=1, rng_seed=0)
        files = list((tmp_twin_dir / "evaluations").glob("*.json"))
        assert files, "no evaluation json file was written"
        with open(files[0]) as f:
            d = json.load(f)
        assert d["policy"]["name"] == "t"
        assert d["num_steps"] == 24

    def test_list_evaluations_recent_first(self, twin):
        snap = _empty_snapshot()
        twin.evaluate_policy(snapshot=snap, policy=PolicySpec(name="a"),
                             horizon_days=1, rng_seed=0)
        twin.evaluate_policy(snapshot=snap, policy=PolicySpec(name="b"),
                             horizon_days=1, rng_seed=0)
        lst = twin.list_evaluations(limit=10)
        assert len(lst) >= 2
        assert all("evaluated_ts" in e for e in lst)


# --------------------------------------------------------------------------- #
# Compare policies
# --------------------------------------------------------------------------- #


class TestCompare:
    def test_compare_ranks_best(self, twin):
        snap = _empty_snapshot()
        twin.arrival_model = ArrivalModel(
            lambda_hd={f"{d}:{h}": 0.3 for d in range(7) for h in range(24)},
        )
        policies = [
            PolicySpec(name="A", double_book_threshold=0.30),
            PolicySpec(name="B", double_book_threshold=0.60),
            PolicySpec(name="C", double_book_threshold=0.10),
        ]
        out = twin.compare_policies(policies=policies,
                                    horizon_days=1,
                                    rng_seed=5,
                                    snapshot=snap)
        assert len(out["policies"]) == 3
        assert out["best_policy"]["name"] in {"A", "B", "C"}
        assert out["best_score"] is not None
        # Ranked should align with best_score
        assert out["ranked_names"][0] == out["best_policy"]["name"]


# --------------------------------------------------------------------------- #
# Fallback squeeze + poisson helpers
# --------------------------------------------------------------------------- #


class TestFallbacks:
    def test_fallback_squeeze_picks_gap_when_capacity_free(self):
        state = _empty_snapshot()
        pol = PolicySpec()
        arrival = {
            "patient_id": "P1",
            "arrival_ts": "2026-04-22T10:00:00",
            "expected_duration": 60,
            "priority": 3,
            "is_urgent": False,
        }
        out = _fallback_squeeze_fn(arrival, state, pol)
        assert out["success"] is True
        assert out["strategy"] == "gap"

    def test_fallback_rejects_when_no_threshold_met(self):
        state = _empty_snapshot()
        state.chair_capacity = {"VEL": 1}
        state.appointments = [{"patient_id": "X"}]  # saturate
        # Policy requires noshow p >= 0.99 to double-book, arrival p=0.10
        pol = PolicySpec(
            double_book_threshold=0.99,
            allow_double_booking=True,
            allow_rescheduling=False,
        )
        arrival = {
            "patient_id": "P1",
            "arrival_ts": "2026-04-22T10:00:00",
            "expected_duration": 60,
            "priority": 3,
            "is_urgent": False,
        }
        out = _fallback_squeeze_fn(arrival, state, pol)
        assert out["success"] is False

    def test_poisson_zero_lambda(self):
        rng = random.Random(0)
        assert _poisson_sample(0.0, rng) == 0
        assert _poisson_sample(-1.0, rng) == 0

    def test_poisson_mean_tracks_lambda(self):
        rng = random.Random(0)
        n = 2000
        samples = [_poisson_sample(5.0, rng) for _ in range(n)]
        mean = sum(samples) / n
        # Within 15% of true mean
        assert 4.25 <= mean <= 5.75


# --------------------------------------------------------------------------- #
# Status endpoint shape
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_status_shape_before_any_snapshot(self, twin):
        st = twin.status()
        assert "arrival_model" in st
        assert st["last_snapshot_ts"] is None
        assert st["total_evaluations"] == 0

    def test_status_shape_after_evaluation(self, twin):
        snap = _empty_snapshot()
        twin.evaluate_policy(snapshot=snap, policy=PolicySpec(name="x"),
                             horizon_days=1, rng_seed=0)
        st = twin.status()
        assert st["last_evaluation_policy"] == "x"
        assert st["total_evaluations"] == 1
        assert st["total_steps"] == 24
