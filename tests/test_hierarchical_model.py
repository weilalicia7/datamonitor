"""
Tests for ml/hierarchical_model.py — Wave 3.2 (T3 coverage).

Covers:
- HierarchicalBayesianModel construction + default parameters fallback
- Empirical-Bayes fit (no PyMC path) produces sensible estimates
- predict returns HierarchicalPrediction dataclass with bounded CI
- Known patient -> posterior mode, new patient -> prior_predictive
- Variance decomposition + patient-compare helpers
- Insufficient-data path uses defaults
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.hierarchical_model import (
    HierarchicalBayesianModel,
    HierarchicalModelSummary,
    HierarchicalPrediction,
    predict_hierarchical,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _patient_data(n_patients=6, obs_per_patient=5, seed=0):
    rng = np.random.default_rng(seed)
    true_alpha = rng.normal(0, 15, n_patients)
    patient_data = []
    durations = []
    patient_ids = []
    for i in range(n_patients):
        for j in range(obs_per_patient):
            pid = f"P{i:03d}"
            patient_ids.append(pid)
            patient_data.append({
                "patient_id": pid,
                "cycle_number": j + 1,
                "expected_duration": 90 + float(rng.normal(0, 15)),
                "complexity_factor": float(rng.uniform(0.3, 0.9)),
                "has_comorbidities": bool(rng.random() < 0.3),
                "appointment_hour": int(rng.integers(8, 17)),
                "day_of_week": int(rng.integers(0, 5)),
                "weather_severity": float(rng.uniform(0, 0.5)),
                "distance_km": float(rng.uniform(5, 30)),
            })
            y = 100 + true_alpha[i] + float(rng.normal(0, 15))
            durations.append(max(30.0, y))
    return patient_data, np.array(durations), patient_ids


# --------------------------------------------------------------------------- #
# Construction / defaults
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_default_state(self):
        m = HierarchicalBayesianModel(n_samples=50, n_chains=1, use_pymc=False)
        assert m.is_fitted is False
        assert m.is_initialized is False
        # Default tau/sigma/mu are seeded.
        assert m.mu == 100.0
        assert m.tau == 15.0
        assert m.sigma == 20.0

    def test_defaults_via_convenience_function(self):
        out = predict_hierarchical({"patient_id": "P999", "cycle_number": 1})
        assert isinstance(out, HierarchicalPrediction)
        assert out.prediction_type == "prior_predictive"
        # Unknown patient uses population prior => patient_effect == 0.
        assert out.patient_effect == 0.0


# --------------------------------------------------------------------------- #
# Empirical-Bayes fit path
# --------------------------------------------------------------------------- #


class TestEmpiricalBayesFit:
    def test_fit_sets_alphas_and_marks_fitted(self):
        patients, durations, ids = _patient_data(n_patients=5, obs_per_patient=4)
        m = HierarchicalBayesianModel(n_samples=50, n_chains=1, use_pymc=False)
        m.fit(patients, durations, patient_ids=ids)
        assert m.is_fitted is True
        assert m.alpha_mean is not None
        assert m.alpha_mean.shape == (5,)
        assert m.beta_mean is not None
        # sigma/tau are positive.
        assert m.sigma > 0
        assert m.tau > 0

    def test_predict_known_patient_uses_posterior(self):
        patients, durations, ids = _patient_data(n_patients=4, obs_per_patient=5)
        m = HierarchicalBayesianModel(n_samples=50, n_chains=1, use_pymc=False)
        m.fit(patients, durations, patient_ids=ids)
        # Pick a known patient.
        out = m.predict({"cycle_number": 2}, patient_id="P000")
        assert isinstance(out, HierarchicalPrediction)
        assert out.prediction_type == "posterior"
        lo, hi = out.credible_interval
        assert lo <= out.predicted_duration <= hi
        assert out.uncertainty >= 0.0


# --------------------------------------------------------------------------- #
# Insufficient-data fallback
# --------------------------------------------------------------------------- #


class TestFallback:
    def test_too_few_observations_uses_defaults(self):
        # Fewer than 10 observations triggers default-parameter path.
        patients, durations, ids = _patient_data(n_patients=2, obs_per_patient=2)
        m = HierarchicalBayesianModel(n_samples=50, n_chains=1, use_pymc=False)
        m.fit(patients, durations, patient_ids=ids)
        # Should be fitted via defaults, with empty patient list.
        assert m.is_fitted is True
        assert list(m.patient_to_idx.keys()) == []


# --------------------------------------------------------------------------- #
# Model summary / helpers
# --------------------------------------------------------------------------- #


class TestSummaryHelpers:
    def test_variance_decomposition(self):
        m = HierarchicalBayesianModel(use_pymc=False)
        m._use_default_parameters()
        out = m.get_variance_decomposition()
        assert "intraclass_correlation" in out
        assert 0.0 <= out["intraclass_correlation"] <= 1.0
        assert out["total_variance"] == out["between_patient_variance"] + out["within_patient_variance"]

    def test_model_summary_dataclass(self):
        patients, durations, ids = _patient_data(n_patients=4, obs_per_patient=4)
        m = HierarchicalBayesianModel(n_samples=50, n_chains=1, use_pymc=False)
        m.fit(patients, durations, patient_ids=ids)
        s = m.get_model_summary()
        assert isinstance(s, HierarchicalModelSummary)
        assert s.n_patients == 4
        assert len(s.fixed_effects) == len(m.FEATURE_NAMES)
