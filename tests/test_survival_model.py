"""
Tests for ml/survival_model.py — Wave 3.2 (T3 coverage).

Covers:
- CoxProportionalHazards default coefficients + survival/hazard semantics
- SurvivalAnalysisModel initialization + predict round-trip
- SurvivalPrediction dataclass shape
- Optimal reminder timing ordering
- Risk categorization / batch predict
- Fit with empty data falls back to defaults
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.survival_model import (
    CoxProportionalHazards,
    SurvivalAnalysisModel,
    SurvivalPrediction,
    predict_noshow_timing,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _patient(pid="P001", *, age=65, noshow_rate=0.15, appointment_hour=14,
             day_of_week=4, cycle_number=2):
    return {
        "patient_id": pid,
        "age": age,
        "noshow_rate": noshow_rate,
        "appointment_hour": appointment_hour,
        "day_of_week": day_of_week,
        "cycle_number": cycle_number,
        "distance_km": 10.0,
        "days_since_last": 21,
        "weather_adverse": 0.0,
        "is_first_appointment": False,
    }


# --------------------------------------------------------------------------- #
# CoxProportionalHazards
# --------------------------------------------------------------------------- #


class TestCoxPH:
    def test_default_coefficients_after_use_defaults(self):
        cox = CoxProportionalHazards()
        cox._use_default_coefficients()
        assert cox.is_fitted is True
        assert len(cox.coefficients) == len(cox._default_coefficients)
        assert cox.baseline_hazard is not None
        assert cox.baseline_times is not None

    def test_hazard_is_non_negative(self):
        cox = CoxProportionalHazards()
        cox._use_default_coefficients()
        features = {name: 0.0 for name in cox.feature_names}
        for day in range(1, 15):
            h = cox.hazard(features, day)
            assert h >= 0
            assert np.isfinite(h)

    def test_survival_is_probability(self):
        cox = CoxProportionalHazards()
        cox._use_default_coefficients()
        features = {name: 0.0 for name in cox.feature_names}
        s = cox.survival(features, 0)
        assert 0.0 <= s <= 1.0

    def test_fit_with_empty_df_uses_defaults(self):
        cox = CoxProportionalHazards()
        cox.fit(pd.DataFrame())
        assert cox.is_fitted is True
        assert cox.coefficients is not None


# --------------------------------------------------------------------------- #
# SurvivalAnalysisModel
# --------------------------------------------------------------------------- #


class TestSurvivalAnalysisModel:
    def test_predict_returns_dataclass_with_valid_fields(self):
        model = SurvivalAnalysisModel()
        out = model.predict(_patient(), days_to_appointment=7)
        assert isinstance(out, SurvivalPrediction)
        assert out.patient_id == "P001"
        assert 0.0 <= out.survival_probability <= 1.0
        lo, hi = out.confidence_interval
        assert 0.0 <= lo <= hi <= 1.0
        assert out.risk_category in {"low", "medium", "high", "critical"}
        assert isinstance(out.optimal_reminder_days, list)

    def test_reminder_days_sorted_descending(self):
        model = SurvivalAnalysisModel()
        out = model.predict(_patient(), days_to_appointment=14)
        reminders = out.optimal_reminder_days
        # Sorted descending (furthest-out first).
        assert reminders == sorted(reminders, reverse=True)

    def test_batch_predict_one_per_patient(self):
        model = SurvivalAnalysisModel()
        patients = [_patient(pid=f"P{i:03d}") for i in range(4)]
        results = model.batch_predict(patients, days_to_appointment=7)
        assert len(results) == 4
        assert all(isinstance(r, SurvivalPrediction) for r in results)

    def test_convenience_function_returns_prediction(self):
        out = predict_noshow_timing(_patient(), days_to_appointment=7)
        assert isinstance(out, SurvivalPrediction)
        assert out.patient_id == "P001"
