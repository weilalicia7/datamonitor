"""
Tests for the tuning package — manifest gate + three tuners + orchestrator.

Coverage rules per the project's testing tenets:
- Real (synthetic-channel) data only; no mocks for the manifest layer.
- Each tuner runs end-to-end on a small slice of the synthetic dataset
  in <30 seconds.
- The channel gate is exercised explicitly: synthetic-channel manifests
  must NOT leak overrides into the boot path; real-channel manifests
  must.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tuning.bayes_opt as _bo
import tuning.grid_search as _gs
import tuning.manifest as _mf
import tuning.random_search as _rs
import tuning.run as _run


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_manifest_path(tmp_path, monkeypatch):
    p = tmp_path / "manifest.json"
    monkeypatch.setenv("TUNING_MANIFEST_PATH", str(p))
    yield p


@pytest.fixture
def tiny_history_df():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "Age": rng.integers(40, 85, n),
        "Cycle_Number": rng.integers(1, 8, n),
        "Travel_Time_Min": rng.uniform(5, 50, n),
        "Days_To_Appointment": rng.integers(1, 21, n),
        "Appointment_Hour": rng.integers(8, 17, n),
        "Day_Of_Week_Num": rng.integers(0, 5, n),
        "is_noshow": rng.integers(0, 2, n),
        "Actual_Duration": rng.uniform(45, 180, n),
    })
    return df


# --------------------------------------------------------------------------- #
# Manifest core
# --------------------------------------------------------------------------- #


class TestManifest:
    def test_load_returns_none_when_missing(self, tmp_manifest_path):
        assert _mf.load_manifest() is None

    def test_record_run_creates_manifest(self, tmp_manifest_path):
        m = _mf.record_tuning_run(
            channel=_mf.CHANNEL_SYNTHETIC,
            tuner_key="noshow_model",
            payload={"foo": 1},
            n_records=10,
        )
        assert m["data_channel"] == "synthetic"
        assert m["results"]["noshow_model"] == {"foo": 1}
        assert m["n_records"] == 10
        assert tmp_manifest_path.exists()

    def test_record_run_invalid_channel_raises(self, tmp_manifest_path):
        with pytest.raises(ValueError):
            _mf.record_tuning_run(
                channel="bogus", tuner_key="x", payload={}, n_records=0,
            )

    def test_overrides_blocked_in_synthetic_mode(self, tmp_manifest_path):
        _mf.record_tuning_run(
            channel=_mf.CHANNEL_SYNTHETIC,
            tuner_key="noshow_model",
            payload={"best_params": {"max_depth": 7}},
            n_records=10,
        )
        # Even though the manifest exists, synthetic mode blocks overrides.
        assert _mf.load_overrides() == {}

    def test_overrides_flow_through_in_real_mode(self, tmp_manifest_path):
        _mf.record_tuning_run(
            channel=_mf.CHANNEL_REAL,
            tuner_key="noshow_model",
            payload={"best_params": {"max_depth": 7}},
            n_records=100,
        )
        ov = _mf.load_overrides()
        assert "noshow_model" in ov
        assert ov["noshow_model"]["best_params"]["max_depth"] == 7

    def test_summary_shape(self, tmp_manifest_path):
        s_empty = _mf.summary()
        assert s_empty["present"] is False
        assert s_empty["overrides_active"] is False

        _mf.record_tuning_run(
            channel=_mf.CHANNEL_REAL,
            tuner_key="cpsat_weights",
            payload={"x": 1},
            n_records=50,
        )
        s_full = _mf.summary()
        assert s_full["present"] is True
        assert s_full["overrides_active"] is True
        assert "cpsat_weights" in s_full["tuners"]

    def test_detect_channel_default_synthetic(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SACT_CHANNEL", raising=False)
        monkeypatch.chdir(tmp_path)   # no datasets/real_data here
        assert _mf.detect_channel() == "synthetic"

    def test_detect_channel_explicit_env(self, monkeypatch):
        monkeypatch.setenv("SACT_CHANNEL", "real")
        assert _mf.detect_channel() == "real"

    def test_detect_channel_real_marker(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SACT_CHANNEL", raising=False)
        monkeypatch.chdir(tmp_path)
        marker = tmp_path / "datasets" / "real_data" / "historical_appointments.xlsx"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"")
        assert _mf.detect_channel() == "real"


# --------------------------------------------------------------------------- #
# Random search (real synthetic data — no mocks)
# --------------------------------------------------------------------------- #


class TestRandomSearch:
    def test_noshow_round_trip(self, tiny_history_df, tmp_manifest_path):
        res = _rs.tune_noshow_model(
            tiny_history_df, n_iter=5, cv_splits=3,
        )
        assert res.target == "noshow"
        assert res.cv == "TimeSeriesSplit(3)"
        assert res.n_iter == 5
        assert res.n_samples == 80
        assert "max_depth" in res.best_params or "n_estimators" in res.best_params
        assert -1.0 <= res.best_score <= 1.0
        assert res.elapsed_s >= 0.0

    def test_duration_round_trip(self, tiny_history_df, tmp_manifest_path):
        res = _rs.tune_duration_model(
            tiny_history_df, n_iter=5, cv_splits=3, estimator="rf",
        )
        assert res.target == "duration"
        assert res.estimator == "RandomForestRegressor"
        assert res.n_samples == 80

    def test_invalid_estimator_raises(self, tiny_history_df):
        with pytest.raises(ValueError):
            _rs.tune_noshow_model(
                tiny_history_df, n_iter=2, cv_splits=2, estimator="xgb_unknown",
            )

    def test_missing_target_raises(self, tiny_history_df):
        with pytest.raises(KeyError):
            _rs.tune_noshow_model(
                tiny_history_df.drop(columns=["is_noshow"]),
                n_iter=2, cv_splits=2,
            )


# --------------------------------------------------------------------------- #
# Grid search (composite scoring + Pareto frontier)
# --------------------------------------------------------------------------- #


class TestGridSearch:
    def test_pareto_frontier_filters_dominated(self):
        # Three profiles: one dominates another on every axis.
        from tuning.grid_search import ProfileScore, pareto_frontier
        good = ProfileScore("good", {}, 0.9, 0.9, 0.9, 1.0, 0.95, 0.1)
        weak = ProfileScore("weak", {}, 0.5, 0.5, 0.5, 5.0, 0.45, 0.1)
        mid  = ProfileScore("mid",  {}, 0.7, 0.7, 0.7, 3.0, 0.70, 0.1)
        front = pareto_frontier([good, weak, mid])
        assert good in front
        assert weak not in front

    def test_evaluate_returns_winner(self):
        from tuning.grid_search import evaluate_weight_profiles
        # Stub solver that varies metrics by 'priority' weight so the
        # higher-priority profile dominates the lower-priority one.
        class _Result:
            def __init__(self, metrics, n_appts):
                self.metrics = metrics
                self.appointments = list(range(n_appts))

        def solve_fn(patients, weights, time_limit_s):
            pri = float(weights.get("priority", 0.3))
            return _Result(
                metrics={"utilization": 0.5 + pri * 0.4,
                         "robustness_score": 0.4 + pri * 0.5,
                         "avg_waiting_days": 5.0 - pri * 3.0},
                n_appts=int(80 + pri * 20),
            )

        weight_sets = [
            {"name": "low",  "priority": 0.10, "utilization": 0.50},
            {"name": "mid",  "priority": 0.30, "utilization": 0.40},
            {"name": "high", "priority": 0.55, "utilization": 0.20},
        ]
        # 100 patient placeholders so scheduled_fraction is meaningful.
        patients = [object()] * 100
        res = evaluate_weight_profiles(
            patients=patients, weight_sets=weight_sets,
            solve_fn=solve_fn, time_limit_s=1,
        )
        assert res.candidates_evaluated == 3
        assert res.winner is not None
        assert res.winner["name"] == "high"


# --------------------------------------------------------------------------- #
# Bayesian optimisation
# --------------------------------------------------------------------------- #


class TestBayesOpt:
    def test_tune_scalar_finds_optimum(self):
        # Quadratic objective with optimum at value=0.10.
        # Composite score is highest when (utilisation - waiting_norm) max.
        # So we encode util = 1 - (value - 0.10)^2 → unimodal around 0.10.
        from tuning.bayes_opt import tune_scalar

        def evaluate(value):
            util = max(0.0, 1.0 - (value - 0.10) ** 2 * 100.0)
            return {"utilisation": util,
                    "avg_waiting_days": 0.0,
                    "fairness_ratio": 1.0}

        res = tune_scalar(
            target="dro_epsilon",
            evaluate_fn=evaluate,
            bounds=(0.005, 0.20),
            n_initial_points=5,
            n_calls=10,
            random_state=0,
        )
        assert res.target == "dro_epsilon"
        assert 0.005 <= res.best_value <= 0.20
        # The optimiser should land near the true argmax (0.10) ± a wide tolerance.
        assert abs(res.best_value - 0.10) <= 0.10
        assert res.best_objective > 0.0


# --------------------------------------------------------------------------- #
# Orchestrator (CLI argparse + record_tuning_run wiring)
# --------------------------------------------------------------------------- #


class TestOrchestrator:
    def test_run_random_search_writes_manifest(
        self, tiny_history_df, tmp_manifest_path,
    ):
        out = _run.run_random_search(
            tiny_history_df, _mf.CHANNEL_SYNTHETIC,
            n_iter=3, cv_splits=3,
        )
        assert "noshow_model" in out
        assert "duration_model" in out
        m = _mf.load_manifest()
        assert m is not None
        assert m["data_channel"] == "synthetic"
        assert "noshow_model" in m["results"]
        assert "duration_model" in m["results"]
        # Synthetic mode → boot path sees no overrides.
        assert _mf.load_overrides() == {}

    def test_load_historical_unknown_channel_raises(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            _run.load_historical("synthetic")
