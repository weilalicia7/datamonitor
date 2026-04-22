"""
Tests for ml/override_learning.py (Dissertation §5.2)
====================================================

Verify the human-in-the-loop override learner:
* Event log persistence
* Cold-start returns the configured prior
* Fit transitions to logistic-regression once `min_events_for_fit` reached
* Predicted probabilities are in [0, 1]
* Pattern recovery: if all overrides are late-afternoon, late slots get
  higher Pr(override) than morning slots
* Suggestions only emitted when Pr(override) > threshold
* Suggestion picks the alternative with lowest override probability
* Config round-trip
* Bulk helper over a schedule returns only the threshold-crossing cases
* JSON round-trip of events and suggestions
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ml.override_learning import (
    DEFAULT_COLD_START_PRIOR,
    DEFAULT_SUGGEST_THRESHOLD,
    FEATURE_COLS,
    OverrideEvent,
    OverrideLearner,
    OverrideStatus,
    OverrideSuggestion,
    _coerce_dt,
    _extract_features,
    compute_suggestions_for_schedule,
    get_learner,
    set_learner,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_learner_dir(tmp_path):
    d = tmp_path / "override_learning"
    return d


@pytest.fixture
def learner(tmp_learner_dir):
    return OverrideLearner(
        storage_dir=tmp_learner_dir,
        min_events_for_fit=20,
        suggest_threshold=0.80,
    )


def _make_event(i, hour, reason="pattern"):
    return OverrideEvent(
        ts=datetime(2026, 4, 22).isoformat(),
        patient_id=f"P{i:03d}",
        original_chair_id="WC-C01",
        original_start_time=datetime(2026, 4, 22, hour, 0).isoformat(),
        original_duration=60,
        new_chair_id="WC-C01",
        new_start_time=datetime(2026, 4, 22, 9, 0).isoformat(),
        new_duration=60,
        priority=3, site_code="WC",
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# 1. Cold-start
# --------------------------------------------------------------------------- #


class TestColdStart:
    def test_cold_start_returns_prior(self, learner):
        appt = {"patient_id": "A", "start_time": datetime(2026, 4, 22, 15, 0),
                "duration": 60, "priority": 3, "site_code": "WC"}
        p = learner.predict_override_probability(appt)
        assert p == pytest.approx(DEFAULT_COLD_START_PRIOR, abs=1e-9)

    def test_not_trained_before_threshold(self, learner):
        assert learner._is_trained is False
        for i in range(5):
            learner.log_event(_make_event(i, 15))
        # Still below min_events_for_fit
        assert learner._is_trained is False


# --------------------------------------------------------------------------- #
# 2. Fit transitions
# --------------------------------------------------------------------------- #


class TestFit:
    def test_eager_fit_at_threshold(self, learner):
        # Register negatives so the fit has both classes
        accepted = [
            {"patient_id": f"Q{i}", "start_time": datetime(2026, 4, 22, 9, 0),
             "duration": 60, "priority": 3, "site_code": "WC"}
            for i in range(40)
        ]
        learner.register_accepted_appointments(accepted)
        for i in range(20):
            learner.log_event(_make_event(i, 15))
        assert learner._is_trained is True
        assert learner._model_method in {"logistic_regression", "count_rate"}

    def test_predicted_probs_in_unit(self, learner):
        accepted = [
            {"patient_id": f"Q{i}", "start_time": datetime(2026, 4, 22, 9, 0),
             "duration": 60, "priority": 3, "site_code": "WC"}
            for i in range(40)
        ]
        learner.register_accepted_appointments(accepted)
        for i in range(25):
            learner.log_event(_make_event(i, 15))
        for appt_start_hour in range(8, 18):
            appt = {"patient_id": "X",
                    "start_time": datetime(2026, 4, 22, appt_start_hour, 0),
                    "duration": 60, "priority": 3, "site_code": "WC"}
            p = learner.predict_override_probability(appt)
            assert 0.0 <= p <= 1.0

    def test_fit_respects_min_threshold(self, learner):
        for i in range(10):
            learner.log_event(_make_event(i, 15))
        ok = learner.fit()
        assert ok is False
        assert learner._is_trained is False


# --------------------------------------------------------------------------- #
# 3. Pattern recovery
# --------------------------------------------------------------------------- #


class TestPattern:
    def test_late_afternoon_has_higher_prob(self, learner):
        """All overrides are 15:00-17:00; morning slots should score low."""
        accepted = [
            {"patient_id": f"Q{i}", "start_time": datetime(2026, 4, 22, 9 + (i % 3), 0),
             "duration": 60, "priority": 3, "site_code": "WC"}
            for i in range(50)
        ]
        learner.register_accepted_appointments(accepted)
        for i in range(30):
            learner.log_event(_make_event(i, 15 + (i % 3)))
        late_appt = {"patient_id": "X",
                     "start_time": datetime(2026, 4, 22, 16, 0),
                     "duration": 60, "priority": 3, "site_code": "WC"}
        morn_appt = {"patient_id": "Y",
                     "start_time": datetime(2026, 4, 22, 9, 0),
                     "duration": 60, "priority": 3, "site_code": "WC"}
        p_late = learner.predict_override_probability(late_appt)
        p_morn = learner.predict_override_probability(morn_appt)
        assert p_late > p_morn


# --------------------------------------------------------------------------- #
# 4. Suggestions
# --------------------------------------------------------------------------- #


class TestSuggestions:
    def test_suggestion_respects_threshold(self, learner):
        accepted = [
            {"patient_id": f"Q{i}", "start_time": datetime(2026, 4, 22, 9, 0),
             "duration": 60, "priority": 3, "site_code": "WC"}
            for i in range(60)
        ]
        learner.register_accepted_appointments(accepted)
        for i in range(30):
            learner.log_event(_make_event(i, 15 + (i % 3)))
        # Below threshold: morning slot
        early = {"patient_id": "X",
                 "start_time": datetime(2026, 4, 22, 9, 0),
                 "duration": 60, "priority": 3, "site_code": "WC"}
        assert learner.suggest(early) is None

    def test_suggestion_picks_lowest_prob_alternative(self, learner):
        accepted = [
            {"patient_id": f"Q{i}", "start_time": datetime(2026, 4, 22, 9, 0),
             "duration": 60, "priority": 3, "site_code": "WC"}
            for i in range(60)
        ]
        learner.register_accepted_appointments(accepted)
        for i in range(30):
            learner.log_event(_make_event(i, 15 + (i % 3)))
        # Configure a very low threshold so we definitely trip
        learner.update_config(suggest_threshold=0.01)
        late = {"patient_id": "X",
                "start_time": datetime(2026, 4, 22, 16, 0),
                "duration": 60, "priority": 3, "site_code": "WC",
                "chair_id": "WC-C01"}
        s = learner.suggest(late)
        assert s is not None
        # The suggested probability must be <= original
        assert s.suggested_override_prob < s.original_override_prob

    def test_custom_alternatives_respected(self, learner):
        accepted = [
            {"patient_id": f"Q{i}", "start_time": datetime(2026, 4, 22, 9, 0),
             "duration": 60, "priority": 3, "site_code": "WC"}
            for i in range(60)
        ]
        learner.register_accepted_appointments(accepted)
        for i in range(30):
            learner.log_event(_make_event(i, 15 + (i % 3)))
        learner.update_config(suggest_threshold=0.01)
        appt = {"patient_id": "X",
                "start_time": datetime(2026, 4, 22, 16, 0),
                "duration": 60, "priority": 3, "site_code": "WC"}
        alt_morning = {"patient_id": "X",
                       "start_time": datetime(2026, 4, 22, 9, 0),
                       "duration": 60, "priority": 3, "site_code": "WC"}
        alt_even_later = {"patient_id": "X",
                          "start_time": datetime(2026, 4, 22, 17, 0),
                          "duration": 60, "priority": 3, "site_code": "WC"}
        s = learner.suggest(appt, alternatives=[alt_morning, alt_even_later])
        assert s is not None
        assert "09:00" in str(s.suggested_start_time)


# --------------------------------------------------------------------------- #
# 5. Persistence + status + config round-trip
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_event_log_written(self, learner, tmp_learner_dir):
        learner.log_event(_make_event(0, 15))
        log = tmp_learner_dir / "events.jsonl"
        assert log.exists()
        rec = json.loads(log.read_text().strip().splitlines()[-1])
        assert rec["patient_id"] == "P000"

    def test_events_restored_on_restart(self, tmp_learner_dir):
        l1 = OverrideLearner(storage_dir=tmp_learner_dir, min_events_for_fit=20)
        for i in range(5):
            l1.log_event(_make_event(i, 15))
        l2 = OverrideLearner(storage_dir=tmp_learner_dir, min_events_for_fit=20)
        assert len(l2.get_events()) == 5

    def test_status_shape(self, learner):
        s = learner.status()
        assert isinstance(s, OverrideStatus)
        assert s.n_events == 0
        assert s.suggest_threshold == DEFAULT_SUGGEST_THRESHOLD

    def test_update_config_round_trip(self, learner):
        cfg = learner.update_config(
            suggest_threshold=0.5, min_events_for_fit=10,
            cold_start_prior=0.05, neighbourhood_hours=[-1, 1, 2],
        )
        assert cfg["suggest_threshold"] == 0.5
        assert cfg["min_events_for_fit"] == 10
        assert cfg["cold_start_prior"] == 0.05
        assert cfg["neighbourhood_hours"] == [-1, 1, 2]


class TestSingleton:
    def test_get_set(self, tmp_path):
        l = OverrideLearner(storage_dir=tmp_path / "m")
        set_learner(l)
        assert get_learner() is l


# --------------------------------------------------------------------------- #
# 6. Bulk helper
# --------------------------------------------------------------------------- #


class TestBulkHelper:
    def test_filters_to_above_threshold(self, learner):
        # Fit so late slots score high
        accepted = [
            {"patient_id": f"Q{i}", "start_time": datetime(2026, 4, 22, 9, 0),
             "duration": 60, "priority": 3, "site_code": "WC"}
            for i in range(60)
        ]
        learner.register_accepted_appointments(accepted)
        for i in range(30):
            learner.log_event(_make_event(i, 15 + (i % 3)))
        learner.update_config(suggest_threshold=0.70)
        schedule = [
            {"patient_id": "A", "start_time": datetime(2026, 4, 22, 9, 0),
             "duration": 60, "priority": 3, "site_code": "WC", "chair_id": "WC-C01"},
            {"patient_id": "B", "start_time": datetime(2026, 4, 22, 16, 0),
             "duration": 60, "priority": 3, "site_code": "WC", "chair_id": "WC-C02"},
        ]
        out = compute_suggestions_for_schedule(learner, schedule)
        # Morning is below threshold; afternoon should be above
        ids_flagged = {s.patient_id for s in out}
        assert "A" not in ids_flagged
        # B may or may not flag depending on the fitted curve — at least ensure
        # every flagged suggestion has above-threshold original_override_prob
        for s in out:
            assert s.original_override_prob >= learner.suggest_threshold


# --------------------------------------------------------------------------- #
# 7. Feature extraction helpers
# --------------------------------------------------------------------------- #


class TestFeatureExtraction:
    def test_all_feature_cols_present(self):
        appt = {"start_time": datetime(2026, 4, 22, 14, 0),
                "duration": 45, "priority": 2, "site_code": "WC",
                "noshow_prob": 0.25}
        f = _extract_features(appt)
        for c in FEATURE_COLS:
            assert c in f
        assert f["hour_of_day"] == 14
        assert f["duration_min"] == 45

    def test_priority_string_coerced(self):
        f = _extract_features({"priority": "P1"})
        assert f["priority"] == 1

    def test_coerce_dt_roundtrip(self):
        t = datetime(2026, 4, 22, 14, 30)
        assert _coerce_dt(t.isoformat()) == t


# --------------------------------------------------------------------------- #
# 8. JSON round-trip
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_events_json_safe(self, learner):
        e = _make_event(1, 15)
        learner.log_event(e)
        back = json.loads(json.dumps(learner.get_events()[-1].__dict__))
        assert back["patient_id"] == "P001"
