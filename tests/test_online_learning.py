"""
Tests for ml/online_learning.py — Wave 3.2 (T3 coverage).

Covers:
- OnlineLearner construction + posterior initial state
- update_on_new_observation returns multiple update results
- Bayesian Beta-Bernoulli update for no-show rate (posterior mean monotone)
- EMA baseline moves toward observed value
- SGD classifier path end-to-end (skipped if sklearn missing)
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.online_learning import (
    OnlineLearner,
    OnlineUpdateResult,
    SKLEARN_AVAILABLE,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _features(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=8)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_default_state(self):
        learner = OnlineLearner()
        assert learner.update_count == 0
        summary = learner.get_posterior_summary()
        assert "noshow_rate" in summary
        assert summary["noshow_rate"]["distribution"] == "Beta"
        assert summary["duration_mean"]["distribution"] == "Normal"
        assert summary["total_updates"] == 0


# --------------------------------------------------------------------------- #
# Bayesian update
# --------------------------------------------------------------------------- #


class TestBayesianUpdate:
    def test_beta_update_pushes_rate_up_on_noshow(self):
        learner = OnlineLearner()
        before = learner.posterior["noshow_rate"]["alpha"] / (
            learner.posterior["noshow_rate"]["alpha"]
            + learner.posterior["noshow_rate"]["beta"]
        )
        # Simulate 5 consecutive no-shows; posterior mean should rise.
        for _ in range(5):
            learner.bayesian_update_noshow(attended=False)
        after = learner.posterior["noshow_rate"]["alpha"] / (
            learner.posterior["noshow_rate"]["alpha"]
            + learner.posterior["noshow_rate"]["beta"]
        )
        assert after > before

    def test_normal_update_duration(self):
        learner = OnlineLearner()
        mu_before = learner.posterior["duration_mean"]["mu"]
        result = learner.bayesian_update_duration(actual_duration=mu_before + 50.0)
        assert isinstance(result, OnlineUpdateResult)
        mu_after = learner.posterior["duration_mean"]["mu"]
        assert mu_after > mu_before
        # n should have incremented.
        assert learner.posterior["duration_mean"]["n"] > 10


# --------------------------------------------------------------------------- #
# EMA update
# --------------------------------------------------------------------------- #


class TestEMAUpdate:
    def test_ema_moves_toward_observation(self):
        learner = OnlineLearner(ema_alpha=0.5)
        base_before = learner.ema_baselines["noshow_rate"]
        # Force a sequence of no-shows with alpha=0.5 — should move up.
        for _ in range(3):
            learner.ema_update(attended=False)
        base_after = learner.ema_baselines["noshow_rate"]
        assert base_after > base_before


# --------------------------------------------------------------------------- #
# Combined update
# --------------------------------------------------------------------------- #


class TestUpdateOnNewObservation:
    def test_returns_multiple_results(self):
        learner = OnlineLearner()
        results = learner.update_on_new_observation(
            patient_features=_features(),
            attended=True,
            actual_duration=125.0,
            weather_severity=0.1,
        )
        assert isinstance(results, list)
        assert len(results) >= 2
        assert learner.update_count == 1
        assert all(isinstance(r, OnlineUpdateResult) for r in results)

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_sgd_classifier_updates(self):
        learner = OnlineLearner()
        # First call: learner is unfitted, partial_fit should run.
        r1 = learner.sgd_update_noshow(_features(seed=1), attended=True)
        assert r1 is not None
        assert r1.method == "sgd"
        # Second call: fitted model — partial_fit on second sample.
        r2 = learner.sgd_update_noshow(_features(seed=2), attended=False)
        assert r2 is not None
        assert learner._sgd_fitted is True
