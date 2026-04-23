"""
Tests for ml/uplift_model.py — Wave 3.2 (T3 coverage).

Covers:
- SLearner / TLearner construction + uplift prediction signs
- UpliftModel ensemble predict_uplift returns UpliftPrediction dataclass
- recommend_intervention returns InterventionRecommendation with best intervention
- NONE intervention yields zero uplift
- batch_recommend sorted by prioritization
- estimate_impact aggregate stats
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.uplift_model import (
    InterventionRecommendation,
    InterventionType,
    SLearner,
    TLearner,
    UpliftModel,
    UpliftPrediction,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _patient(pid="P001", *, age=55, noshow_rate=0.2, distance_km=15.0):
    return {
        "patient_id": pid,
        "age": age,
        "noshow_rate": noshow_rate,
        "distance_km": distance_km,
        "days_since_last": 30,
        "is_first_appointment": False,
        "cycle_number": 3,
    }


# --------------------------------------------------------------------------- #
# S-Learner / T-Learner
# --------------------------------------------------------------------------- #


class TestMetaLearners:
    def test_s_learner_default_fit(self):
        s = SLearner()
        s.fit(pd.DataFrame(), np.array([]), np.array([]))
        assert s.is_fitted is True

    def test_s_learner_sms_reduces_probability(self):
        s = SLearner()
        s.fit(pd.DataFrame(), np.array([]), np.array([]))
        features = {
            "previous_noshow_rate": 0.3,
            "age_risk": 0.4,
            "distance_km": 10.0,
            "days_since_last": 30,
            "is_first_appointment": 0.0,
            "cycle_number": 2,
        }
        uplift_sms = s.predict_uplift(features, InterventionType.SMS_REMINDER)
        # SMS reminder should have non-positive uplift (reduces no-show).
        assert uplift_sms <= 0.0

    def test_t_learner_none_intervention_zero_uplift(self):
        t = TLearner()
        t.fit(pd.DataFrame(), np.array([]), np.array([]))
        features = {
            "previous_noshow_rate": 0.2,
            "age_risk": 0.5,
            "distance_km": 10,
            "days_since_last": 14,
            "is_first_appointment": 0.0,
            "cycle_number": 1,
        }
        uplift = t.predict_uplift(features, InterventionType.NONE)
        assert uplift == 0.0


# --------------------------------------------------------------------------- #
# UpliftModel
# --------------------------------------------------------------------------- #


class TestUpliftModel:
    def test_predict_uplift_returns_dataclass(self):
        model = UpliftModel()
        model.initialize()
        pred = model.predict_uplift(_patient(), InterventionType.SMS_REMINDER)
        assert isinstance(pred, UpliftPrediction)
        assert 0.0 <= pred.baseline_noshow_prob <= 1.0
        assert 0.0 <= pred.treated_noshow_prob <= 1.0
        assert pred.intervention == InterventionType.SMS_REMINDER
        # Confidence is a reported scalar, not necessarily clamped.
        assert np.isfinite(pred.confidence)

    def test_recommend_intervention_picks_best(self):
        model = UpliftModel()
        rec = model.recommend_intervention(_patient())
        assert isinstance(rec, InterventionRecommendation)
        assert rec.best_intervention != InterventionType.NONE
        assert rec.all_interventions, "must evaluate at least one intervention"
        # all_interventions is sorted by uplift (most negative first), so best
        # should match first entry's intervention.
        assert rec.best_intervention == rec.all_interventions[0].intervention

    def test_batch_recommend_sorted_by_priority(self):
        model = UpliftModel()
        patients = [
            _patient(pid="LOW", noshow_rate=0.05),
            _patient(pid="HIGH", noshow_rate=0.45),
            _patient(pid="MID", noshow_rate=0.20),
        ]
        recs = model.batch_recommend(patients)
        scores = [r.prioritization_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_estimate_impact_aggregates(self):
        model = UpliftModel()
        patients = [_patient(pid=f"P{i}", noshow_rate=0.15 + 0.02 * i) for i in range(5)]
        impact = model.estimate_impact(patients)
        assert impact["n_patients"] == 5
        assert 0.0 <= impact["avg_baseline_noshow_rate"] <= 1.0
        assert "expected_impact" in impact
        assert "intervention_distribution" in impact
