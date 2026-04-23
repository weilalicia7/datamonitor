"""
Tests for ml.sensitivity_analysis — SensitivityAnalyzer,
LocalSensitivity, GlobalImportance.

Covers:
    * Construction defaults (perturbation size).
    * Local sensitivity on a linear scikit-learn model — values match
      the known slope within numerical tolerance.
    * Global importance — mean |S_ij| matches absolute coefficient.
    * Fallback on a zero-output "model" (_get_prediction branch).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.sensitivity_analysis import (
    SensitivityAnalyzer,
    LocalSensitivity,
    GlobalImportance,
)


# ---------------------------------------------------------------------------
# Fixtures — lightweight sklearn-compatible stubs
# ---------------------------------------------------------------------------

class LinearModel:
    """f(x) = a . x (deterministic, used as ground truth for derivatives)."""

    def __init__(self, coefs):
        self.coefs = np.asarray(coefs, dtype=float)

    def predict(self, X):
        return X @ self.coefs


class ZeroModel:
    """Predicts zero — used for the fallback / empty-output branch."""

    def predict(self, X):
        return np.zeros(len(X))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_construction_default_perturbation():
    analyzer = SensitivityAnalyzer()
    assert pytest.approx(analyzer.perturbation, rel=1e-12) == 0.01
    custom = SensitivityAnalyzer(perturbation=0.05)
    assert pytest.approx(custom.perturbation, rel=1e-12) == 0.05
    # Feature categories shipped with expected top-level keys
    for cat in ('Patient History', 'Demographics', 'Temporal',
                'Geographic', 'Treatment', 'External'):
        assert cat in SensitivityAnalyzer.FEATURE_CATEGORIES


def test_local_sensitivity_matches_linear_slope():
    """For a linear model f(x)=a.x, dS_i/dx_i = a_i."""
    coefs = np.array([2.0, -1.5, 0.25])
    model = LinearModel(coefs)
    feats = np.array([1.0, 2.0, 3.0])
    names = ['a', 'b', 'c']

    analyzer = SensitivityAnalyzer(perturbation=0.01)
    local = analyzer.local_sensitivity(
        model, feats, names, model_type='duration', patient_id='LIN01'
    )

    assert isinstance(local, LocalSensitivity)
    assert local.patient_id == 'LIN01'
    assert local.model_type == 'duration'
    for name, expected in zip(names, coefs):
        # Central finite differences should recover the slope almost exactly.
        assert pytest.approx(local.sensitivities[name], abs=1e-3) == expected

    # Base prediction = a . x
    assert pytest.approx(local.base_prediction, abs=1e-6) == float(coefs @ feats)


def test_global_importance_returns_dataclass_and_ranking():
    """Over multiple samples, |coef| ranks feature importance."""
    coefs = np.array([3.0, 0.5, -2.0])  # |coef| order: 0, 2, 1
    model = LinearModel(coefs)
    names = ['a', 'b', 'c']

    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(30, 3))

    analyzer = SensitivityAnalyzer(perturbation=0.01)
    glob = analyzer.global_importance(
        model, X, names, model_type='duration', max_samples=20,
    )

    assert isinstance(glob, GlobalImportance)
    assert glob.n_samples == 20
    assert set(glob.importance.keys()) == set(names)
    # Rank list sorted descending by importance
    ordered = [n for n, _ in glob.importance_rank]
    assert ordered[0] == 'a'  # largest |coef|
    assert ordered[-1] == 'b'  # smallest |coef|


def test_get_prediction_fallback_on_zero_model():
    """Zero-output model path — base_prediction = 0, sensitivities = 0."""
    analyzer = SensitivityAnalyzer(perturbation=0.01)
    local = analyzer.local_sensitivity(
        ZeroModel(), np.array([1.0, 2.0]), ['x', 'y'],
        model_type='duration', patient_id='ZERO',
    )
    assert local.base_prediction == 0.0
    for v in local.sensitivities.values():
        assert v == 0.0

    # Elasticity with zero base prediction falls through the guard
    elasts = analyzer.elasticity(
        ZeroModel(), np.array([1.0, 2.0]), ['x', 'y'], model_type='duration'
    )
    assert all(v == 0.0 for v in elasts.values())
