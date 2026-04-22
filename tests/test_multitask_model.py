"""
Tests for ml/multitask_model.py — Wave 3.1.5 (T3 coverage).

Covers:
- construction (torch + fallback paths)
- _extract_features shape + padding
- fit + predict round-trip
- batch_predict + summary helpers
- fallback prediction path when not fitted
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.multitask_model import MultiTaskModel, MultiTaskPrediction


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _patient(pid="P001", **overrides):
    base = {
        "patient_id": pid,
        "noshow_rate": 0.15,
        "age": 60,
        "distance_km": 8.0,
        "is_first_appointment": False,
        "cycle_number": 3,
        "days_since_last": 28,
        "appointment_hour": 10,
        "day_of_week": 2,
        "expected_duration": 120,
        "complexity_factor": 0.7,
        "is_first_cycle": False,
        "comorbidity_count": 1,
        "duration_variance": 0.15,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_default_init(self):
        m = MultiTaskModel(input_dim=20)
        assert m.input_dim == 20
        assert m.is_fitted is False

    def test_numpy_fallback_init(self):
        m = MultiTaskModel(input_dim=20, use_torch=False)
        assert m.use_torch is False
        assert m.network is None


# --------------------------------------------------------------------------- #
# _extract_features
# --------------------------------------------------------------------------- #


class TestFeatureExtraction:
    def test_returns_input_dim_vector(self):
        m = MultiTaskModel(input_dim=20)
        feats = m._extract_features(_patient())
        assert isinstance(feats, np.ndarray)
        assert feats.shape == (20,)

    def test_pads_to_input_dim(self):
        m = MultiTaskModel(input_dim=30)
        feats = m._extract_features(_patient())
        assert feats.shape == (30,)
        # The trailing values should be the zero-padding.
        assert (feats[20:] == 0).all()


# --------------------------------------------------------------------------- #
# Fit + predict
# --------------------------------------------------------------------------- #


class TestFitPredict:
    def _training_set(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        patients = [_patient(f"P{i:03d}",
                             noshow_rate=float(rng.uniform(0, 0.5)))
                    for i in range(n)]
        noshow = (rng.uniform(0, 1, n) < 0.2).astype(int)
        durations = rng.uniform(60, 180, n)
        return patients, noshow, durations

    def test_predict_unfitted_uses_fallback(self):
        m = MultiTaskModel(input_dim=20, use_torch=False)
        out = m.predict(_patient())
        assert isinstance(out, MultiTaskPrediction)
        assert 0.0 <= out.noshow_probability <= 1.0
        assert out.predicted_duration > 0

    def test_fit_marks_is_fitted(self):
        patients, noshow, durations = self._training_set(n=20)
        m = MultiTaskModel(input_dim=20, use_torch=False)
        m.fit(patients, noshow, durations, epochs=3, batch_size=8)
        assert m.is_fitted is True

    def test_predict_after_fit_returns_dataclass(self):
        patients, noshow, durations = self._training_set(n=24)
        m = MultiTaskModel(input_dim=20)
        m.fit(patients, noshow, durations, epochs=3, batch_size=8)
        out = m.predict(_patient())
        assert isinstance(out, MultiTaskPrediction)
        assert 0.0 <= out.noshow_probability <= 1.0
        assert out.duration_lower <= out.predicted_duration <= out.duration_upper


# --------------------------------------------------------------------------- #
# Batch + summary
# --------------------------------------------------------------------------- #


class TestBatchAndSummary:
    def test_batch_predict_returns_one_per_patient(self):
        m = MultiTaskModel(input_dim=20, use_torch=False)
        patients = [_patient(f"P{i:03d}") for i in range(5)]
        results = m.batch_predict(patients)
        assert len(results) == 5
        assert all(isinstance(r, MultiTaskPrediction) for r in results)

    def test_get_model_summary(self):
        m = MultiTaskModel(input_dim=20, use_torch=False)
        s = m.get_model_summary()
        assert isinstance(s, dict)
        assert len(s) > 0
