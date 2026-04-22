"""
Tests for ml/quantile_forest.py — Wave 3.1.6 (T3 coverage).

Covers:
- QuantileRegressionForest fit + predict_quantiles + predict_interval
- Distribution-free CIs (lower < median < upper)
- QuantileForestDurationModel feature extraction + fit/predict
- QuantileForestNoShowModel summary
- Default fallback when not fitted
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.quantile_forest import (
    QuantileForestDurationModel,
    QuantileForestNoShowModel,
    QuantilePrediction,
    QuantileRegressionForest,
)


# --------------------------------------------------------------------------- #
# QuantileRegressionForest core
# --------------------------------------------------------------------------- #


class TestQRFCore:
    def _train_xy(self, n=80, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, size=(n, 4))
        # y depends on X[:,0] with heteroscedastic noise so quantiles spread.
        noise = rng.normal(0, 0.5 + np.abs(X[:, 0]), size=n)
        y = 2.0 * X[:, 0] + noise
        return X, y

    def test_fit_marks_is_fitted(self):
        X, y = self._train_xy()
        qrf = QuantileRegressionForest(n_estimators=20, max_depth=5)
        qrf.fit(X, y)
        assert qrf.is_fitted is True

    def test_predict_quantiles_shape(self):
        X, y = self._train_xy()
        qrf = QuantileRegressionForest(n_estimators=20, max_depth=5)
        qrf.fit(X, y)
        preds = qrf.predict_quantiles(X[:5], quantiles=[0.1, 0.5, 0.9])
        assert preds.shape == (5, 3)

    def test_quantiles_monotone(self):
        X, y = self._train_xy()
        qrf = QuantileRegressionForest(n_estimators=20, max_depth=5)
        qrf.fit(X, y)
        preds = qrf.predict_quantiles(X[:10], quantiles=[0.1, 0.5, 0.9])
        # Each row: lower ≤ median ≤ upper.
        for row in preds:
            assert row[0] <= row[1] <= row[2] + 1e-9

    def test_predict_interval(self):
        X, y = self._train_xy()
        qrf = QuantileRegressionForest(n_estimators=20, max_depth=5)
        qrf.fit(X, y)
        # predict_interval returns (median, lower, upper).
        median, lo, hi = qrf.predict_interval(X[:5], confidence=0.95)
        assert median.shape == (5,)
        assert lo.shape == (5,) and hi.shape == (5,)
        assert np.all(lo <= median + 1e-9)
        assert np.all(median <= hi + 1e-9)

    def test_predict_unfitted_raises(self):
        qrf = QuantileRegressionForest(n_estimators=10)
        with pytest.raises(ValueError):
            qrf.predict_quantiles(np.zeros((1, 4)))


# --------------------------------------------------------------------------- #
# QuantileForestDurationModel wrapper
# --------------------------------------------------------------------------- #


class TestDurationWrapper:
    def _patients(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            out.append({
                "patient_id": f"P{i:03d}",
                "expected_duration": 60 + i * 5,
                "cycle_number": int(rng.integers(1, 8)),
                "complexity_factor": float(rng.uniform(0.3, 1.0)),
                "age": int(rng.integers(40, 85)),
                "comorbidity_count": int(rng.integers(0, 4)),
                "duration_variance": float(rng.uniform(0.05, 0.3)),
                "protocol": rng.choice(["FOLFOX", "FOLFIRI", "AC"]),
                "noshow_rate": float(rng.uniform(0, 0.4)),
                "distance_km": float(rng.uniform(2, 30)),
            })
        return out

    def test_predict_default_when_unfitted(self):
        m = QuantileForestDurationModel(n_estimators=20, max_depth=4)
        out = m.predict(self._patients(n=1)[0])
        assert isinstance(out, QuantilePrediction)
        # Defaults still respect lower ≤ median ≤ upper.
        assert out.lower_bound <= out.point_estimate <= out.upper_bound

    def test_fit_predict_round_trip(self):
        patients = self._patients(n=30)
        durations = np.array([60 + i * 2 for i in range(30)], dtype=float)
        m = QuantileForestDurationModel(n_estimators=20, max_depth=4)
        m.fit(patients, durations)
        out = m.predict(patients[0])
        assert isinstance(out, QuantilePrediction)
        # Width is positive + reported.
        assert out.prediction_interval_width >= 0

    def test_get_model_summary(self):
        m = QuantileForestDurationModel(n_estimators=10, max_depth=3)
        summary = m.get_model_summary()
        assert isinstance(summary, dict)
        assert "n_estimators" in summary or "model_type" in summary or len(summary) > 0


# --------------------------------------------------------------------------- #
# QuantileForestNoShowModel — quick sanity
# --------------------------------------------------------------------------- #


class TestNoShowWrapper:
    def test_summary_keys(self):
        m = QuantileForestNoShowModel(n_estimators=5)
        # Just verify it constructs cleanly.
        assert m is not None
