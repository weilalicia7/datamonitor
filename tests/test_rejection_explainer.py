"""
Tests for ml/rejection_explainer.py (Dissertation §5.3)
======================================================

Verifies the rejection-explainer produces the §5.3 brief's narrative
pattern from real schedule data, classifies blockers correctly,
finds alternative slots, and persists to JSONL.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ml.rejection_explainer import (
    BLOCKER_BUMP_HIGHER,
    BLOCKER_BUMP_SAME_OR_LOWER,
    BLOCKER_DURATION_TOO_LONG,
    BLOCKER_EARLIEST_EXCEEDED,
    BLOCKER_INSUFFICIENT_SLACK,
    BLOCKER_OUTSIDE_HOURS,
    BLOCKER_TRAVEL_EXCEEDED,
    AlternativeSlot,
    ChairBlocker,
    RejectionExplainer,
    RejectionExplanation,
    _friendly_day_phrase,
    _serialise_appt,
    get_explainer,
    set_explainer,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _appt(pid, chair, start_hour, dur=60, priority=3, start_minute=0):
    start = datetime(2026, 4, 22, start_hour, start_minute)
    return {
        "patient_id": pid,
        "chair_id": chair,
        "site_code": "WC",
        "start_time": start,
        "end_time": start + timedelta(minutes=dur),
        "duration": dur,
        "priority": priority,
    }


@pytest.fixture
def tmp_expl_dir(tmp_path):
    return tmp_path / "rejection"


@pytest.fixture
def explainer(tmp_expl_dir):
    return RejectionExplainer(storage_dir=tmp_expl_dir, look_ahead_days=3)


@pytest.fixture
def chairs():
    return [
        {"chair_id": "WC-C01", "site_code": "WC"},
        {"chair_id": "WC-C02", "site_code": "WC"},
        {"chair_id": "WC-C03", "site_code": "WC"},
    ]


@pytest.fixture
def today():
    return datetime(2026, 4, 22).replace(hour=0, minute=0,
                                         second=0, microsecond=0)


# --------------------------------------------------------------------------- #
# 1. Narrative matches the §5.3 brief
# --------------------------------------------------------------------------- #


class TestNarrativePattern:
    def test_brief_pattern_elements(self, explainer, chairs, today):
        # WC-C01 fully booked, WC-C02 has 45-min gap, WC-C03 has P1 collision
        schedule = [
            _appt("E01", "WC-C01", 9, dur=540),
            _appt("E02", "WC-C02", 9, dur=60),
            _appt("E03", "WC-C02", 10, start_minute=45, dur=60),
            _appt("E04", "WC-C02", 12, dur=360),
            _appt("X3", "WC-C03", 9, dur=60),
            _appt("VIP", "WC-C03", 10, dur=60, priority=1),
            _appt("X4", "WC-C03", 11, dur=420),
        ]
        patient = {
            "Patient_ID": "P12345", "priority": 2, "expected_duration": 60,
            "earliest_time": today.replace(hour=9), "no_shows": 3,
        }
        expl = explainer.explain(patient, schedule, chairs=chairs,
                                 current_date=today)
        narr = expl.narrative
        # Header
        assert "Patient P12345 (P2, previous no-shows: 3)" in narr
        assert "not scheduled because:" in narr
        # One of the three reported blockers must mention the 45-min slack
        assert "45 min slack (requires 60 min" in narr
        # P1 bump mentioned
        assert "priority-1 patient VIP" in narr or "priority-1" in narr
        # Alternative suggestion
        assert "Alternative: offer" in narr
        assert "[Accept]" in narr and "[Decline]" in narr


# --------------------------------------------------------------------------- #
# 2. Blocker type classification
# --------------------------------------------------------------------------- #


class TestBlockerClassification:
    def test_insufficient_slack(self, explainer, chairs, today):
        # WC-C01 has only 45-min gap; 60-min patient → INSUFFICIENT_SLACK
        schedule = [
            _appt("E1", "WC-C01", 9, dur=60),
            _appt("E2", "WC-C01", 10, start_minute=45, dur=60),
            _appt("E3", "WC-C01", 12, dur=360),
        ]
        patient = {
            "Patient_ID": "P1", "priority": 3, "expected_duration": 60,
            "earliest_time": today.replace(hour=9),
        }
        expl = explainer.explain(patient, schedule[:3] +  # only C01 has data
                                 [_appt("Z1", "WC-C02", 8, dur=600),
                                  _appt("Z2", "WC-C03", 8, dur=600)],
                                 chairs=chairs, current_date=today)
        types = {b.blocker_type for b in expl.blockers}
        assert BLOCKER_INSUFFICIENT_SLACK in types

    def test_bump_higher_priority(self, explainer, chairs, today):
        # WC-C01 fully booked with P1 throughout
        schedule = [_appt("VIP", "WC-C01", 8, dur=600, priority=1),
                    _appt("Z2", "WC-C02", 8, dur=600),
                    _appt("Z3", "WC-C03", 8, dur=600)]
        patient = {
            "Patient_ID": "P2AUTO", "priority": 2, "expected_duration": 60,
            "earliest_time": today.replace(hour=9),
        }
        expl = explainer.explain(patient, schedule, chairs=chairs,
                                 current_date=today)
        types = {b.blocker_type for b in expl.blockers}
        assert BLOCKER_BUMP_HIGHER in types

    def test_bump_same_or_lower(self, explainer, chairs, today):
        # All chairs fully booked with P3 patients; new patient is P2
        # With allow_reschedule=False → BUMP_SAME_OR_LOWER (since P3 < P2 priority-wise... wait, P2 is higher)
        # Actually: P2 (patient) vs P3 (collider) — collider has LOWER priority
        # (higher number). allow_reschedule=False → BUMP_SAME_OR_LOWER
        schedule = [_appt("X1", "WC-C01", 8, dur=600, priority=3),
                    _appt("X2", "WC-C02", 8, dur=600, priority=3),
                    _appt("X3", "WC-C03", 8, dur=600, priority=3)]
        patient = {
            "Patient_ID": "P2A", "priority": 2, "expected_duration": 60,
            "earliest_time": today.replace(hour=9),
        }
        expl = explainer.explain(patient, schedule, chairs=chairs,
                                 current_date=today)
        types = {b.blocker_type for b in expl.blockers}
        assert BLOCKER_BUMP_SAME_OR_LOWER in types

    def test_duration_too_long(self, explainer, chairs, today):
        # Patient needs 800 min — longer than the 8-18 day (600 min)
        patient = {
            "Patient_ID": "LONG", "priority": 3, "expected_duration": 800,
            "earliest_time": today.replace(hour=9),
        }
        expl = explainer.explain(patient, schedule=[],
                                 chairs=chairs, current_date=today)
        types = {b.blocker_type for b in expl.blockers}
        assert BLOCKER_DURATION_TOO_LONG in types

    def test_outside_hours_or_insufficient_slack_for_late_earliest(
        self, explainer, chairs, today
    ):
        # No items, patient earliest is 17:30 → only 30 min left before day-end,
        # needs 60 min.  Trailing-gap max_any will be 30 min so we expect
        # INSUFFICIENT_SLACK (the most specific classification available).
        patient = {
            "Patient_ID": "LATE", "priority": 3, "expected_duration": 60,
            "earliest_time": today.replace(hour=17, minute=30),
        }
        expl = explainer.explain(patient, schedule=[],
                                 chairs=chairs, current_date=today)
        types = {b.blocker_type for b in expl.blockers}
        assert (BLOCKER_INSUFFICIENT_SLACK in types
                or BLOCKER_OUTSIDE_HOURS in types
                or BLOCKER_EARLIEST_EXCEEDED in types)

    def test_earliest_exceeded(self, explainer, chairs, today):
        # Patient earliest 19:00 which is after the 18:00 day-end
        patient = {
            "Patient_ID": "VERYLATE", "priority": 3, "expected_duration": 60,
            "earliest_time": today.replace(hour=19),
        }
        expl = explainer.explain(patient, schedule=[],
                                 chairs=chairs, current_date=today)
        types = {b.blocker_type for b in expl.blockers}
        assert BLOCKER_EARLIEST_EXCEEDED in types

    def test_travel_exceeded(self, explainer, chairs, today):
        # Patient travel = 150 min > 120 min ceiling
        patient = {
            "Patient_ID": "REMOTE", "priority": 3, "expected_duration": 60,
            "earliest_time": today.replace(hour=9),
            "travel_time_minutes": 150,
        }
        expl = explainer.explain(patient, schedule=[],
                                 chairs=chairs, current_date=today)
        types = {b.blocker_type for b in expl.blockers}
        assert BLOCKER_TRAVEL_EXCEEDED in types


# --------------------------------------------------------------------------- #
# 3. Alternative-slot finder
# --------------------------------------------------------------------------- #


class TestAlternative:
    def test_alternative_next_day(self, explainer, chairs, today):
        # Today fully booked; alternative must land on a later day
        schedule = [_appt(f"X{c}", c, 8, dur=600)
                    for c in ("WC-C01", "WC-C02", "WC-C03")]
        patient = {
            "Patient_ID": "A", "priority": 3, "expected_duration": 60,
            "earliest_time": today.replace(hour=9),
        }
        expl = explainer.explain(patient, schedule, chairs=chairs,
                                 current_date=today)
        assert expl.alternative is not None
        alt_dt = datetime.fromisoformat(expl.alternative.start_time)
        assert alt_dt.date() > today.date()
        assert expl.alternative.wait_increase_minutes > 0

    def test_alternative_respects_earliest_across_days(self, explainer, chairs,
                                                      today):
        # Patient earliest=11:00 — alternative on later day must be >= 11:00
        schedule = [_appt(f"X{c}", c, 8, dur=600)
                    for c in ("WC-C01", "WC-C02", "WC-C03")]
        patient = {
            "Patient_ID": "A", "priority": 3, "expected_duration": 60,
            "earliest_time": today.replace(hour=11),
        }
        expl = explainer.explain(patient, schedule, chairs=chairs,
                                 current_date=today)
        assert expl.alternative is not None
        alt_dt = datetime.fromisoformat(expl.alternative.start_time)
        assert alt_dt.hour >= 11

    def test_no_alternative_when_all_days_full(self, explainer, chairs, today):
        # Saturate several days' worth of chairs
        look = explainer.look_ahead_days
        schedule = []
        for d in range(look):
            day = today + timedelta(days=d)
            for c in ("WC-C01", "WC-C02", "WC-C03"):
                a = _appt(f"X{d}_{c}", c, 8, dur=600)
                a["start_time"] = day.replace(hour=8)
                a["end_time"] = day.replace(hour=18)
                schedule.append(a)
        patient = {
            "Patient_ID": "A", "priority": 3, "expected_duration": 60,
            "earliest_time": today.replace(hour=9),
        }
        expl = explainer.explain(patient, schedule, chairs=chairs,
                                 current_date=today)
        assert expl.alternative is None
        assert "No alternative" in expl.narrative


# --------------------------------------------------------------------------- #
# 4. explain_all + bulk
# --------------------------------------------------------------------------- #


class TestExplainAll:
    def test_bulk_returns_one_per_patient(self, explainer, chairs, today):
        patients = [
            {"Patient_ID": f"U{i}", "priority": 3, "expected_duration": 60,
             "earliest_time": today.replace(hour=9)}
            for i in range(5)
        ]
        out = explainer.explain_all(patients, schedule=[], chairs=chairs,
                                    current_date=today)
        assert len(out) == 5
        assert all(isinstance(e, RejectionExplanation) for e in out)


# --------------------------------------------------------------------------- #
# 5. Persistence + status
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_log_written(self, explainer, tmp_expl_dir, chairs, today):
        explainer.explain(
            {"Patient_ID": "A", "priority": 3, "expected_duration": 60,
             "earliest_time": today.replace(hour=9)},
            schedule=[], chairs=chairs, current_date=today,
        )
        log = tmp_expl_dir / "explanations.jsonl"
        assert log.exists()
        rec = json.loads(log.read_text().strip().splitlines()[-1])
        assert rec["patient_id"] == "A"
        assert "n_blockers" in rec

    def test_status_shape(self, explainer):
        s = explainer.status()
        for k in (
            "day_start_hour", "day_end_hour", "max_travel_minutes",
            "look_ahead_days", "allow_reschedule", "total_runs",
        ):
            assert k in s

    def test_update_config_round_trip(self, explainer):
        cfg = explainer.update_config(
            day_start_hour=7, day_end_hour=19,
            max_travel_minutes=90, look_ahead_days=7,
            slack_buffer_min=10, allow_reschedule=True,
        )
        assert cfg["day_start_hour"] == 7
        assert cfg["day_end_hour"] == 19
        assert cfg["max_travel_minutes"] == 90
        assert cfg["look_ahead_days"] == 7
        assert cfg["slack_buffer_min"] == 10
        assert cfg["allow_reschedule"] is True


class TestSingleton:
    def test_get_set(self, tmp_path):
        e = RejectionExplainer(storage_dir=tmp_path / "r")
        set_explainer(e)
        assert get_explainer() is e


# --------------------------------------------------------------------------- #
# 6. JSON round-trip
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_to_dict_json_safe(self, explainer, chairs, today):
        expl = explainer.explain(
            {"Patient_ID": "A", "priority": 3, "expected_duration": 60,
             "earliest_time": today.replace(hour=9)},
            schedule=[], chairs=chairs, current_date=today,
        )
        back = json.loads(json.dumps(expl.to_dict(), default=str))
        assert back["patient_id"] == "A"
        assert isinstance(back["blockers"], list)


# --------------------------------------------------------------------------- #
# 7. Helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_friendly_day_phrase(self):
        slot = datetime(2026, 4, 27, 14, 0)        # Monday
        earliest = datetime(2026, 4, 22, 9, 0)
        s = _friendly_day_phrase(slot, earliest)
        assert "Monday" in s
        assert "2pm" in s

    def test_friendly_same_day_no_minutes(self):
        slot = datetime(2026, 4, 22, 9, 0)
        earliest = datetime(2026, 4, 22, 9, 0)
        s = _friendly_day_phrase(slot, earliest)
        # No "wait increase" suffix when delta is 0
        assert "wait increase" not in s

    def test_serialise_appt_dict(self):
        appt = {"patient_id": "P", "start_time": "2026-04-22T09:00:00",
                "end_time": "2026-04-22T10:00:00"}
        out = _serialise_appt(appt)
        assert out["start_time"] == datetime(2026, 4, 22, 9, 0)
        assert out["end_time"] == datetime(2026, 4, 22, 10, 0)


# --------------------------------------------------------------------------- #
# 8. Edge cases (Wave 3.8)
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Error-path / degenerate-input coverage per Wave 3.8.  The explainer
    is called from the Flask layer with whatever ``run_optimization``
    returns; it must gracefully handle empty and malformed inputs."""

    def test_empty_patient_list_returns_empty_explanations(
        self, explainer, chairs, today,
    ):
        """``explain_all([], ...)`` returns the empty list (not None)
        without raising.  Status counter still ticks — one run was made."""
        out = explainer.explain_all(
            [], schedule=[], chairs=chairs, current_date=today,
        )
        assert out == []
        assert isinstance(out, list)
        assert explainer.status()["total_runs"] == 1

    def test_malformed_result_without_unscheduled_key(
        self, explainer, chairs, today,
    ):
        """A caller passing an ``OptimizationResult``-shaped dict that
        lacks the ``unscheduled`` key must not crash the explainer.  The
        canonical pattern is ``result.get('unscheduled') or []``; applying
        that defensive read yields an empty list of explanations."""
        fake_result = {
            "success": True,
            "appointments": [],
            # 'unscheduled' key deliberately missing
        }
        # Direct dict access would raise KeyError
        with pytest.raises(KeyError):
            _ = fake_result["unscheduled"]
        # Defensive read + explain_all must not crash
        unscheduled = fake_result.get("unscheduled") or []
        out = explainer.explain_all(
            unscheduled, schedule=[], chairs=chairs, current_date=today,
        )
        assert out == []

    def test_all_scheduled_no_rejections_to_explain(
        self, explainer, chairs, today,
    ):
        """When every patient landed in ``appointments`` and
        ``unscheduled`` is empty, explain_all iterates over zero
        rejections and the status reports no narrative."""
        # Simulate "all scheduled" — explicitly pass an empty unscheduled
        # list.  The explainer must report 0 explanations and no last
        # narrative (nothing to say).
        out = explainer.explain_all(
            [], schedule=[_appt("P1", "WC-C01", 9)],
            chairs=chairs, current_date=today,
        )
        assert len(out) == 0
        status = explainer.status()
        assert status["last_n_explanations"] == 0
        # No narrative recorded because no rejection was explained
        assert status["last_narrative"] is None

    def test_explain_all_single_patient_empty_schedule(
        self, explainer, chairs, today,
    ):
        """One rejected patient, empty schedule, full chair availability
        — the explainer should emit one explanation (possibly still
        with blockers, e.g. travel) and NOT crash even with no
        existing appointments to reason about."""
        patient = {
            "Patient_ID": "SINGLE", "priority": 3,
            "expected_duration": 60,
            "earliest_time": today.replace(hour=9),
        }
        out = explainer.explain_all(
            [patient], schedule=[], chairs=chairs, current_date=today,
        )
        assert len(out) == 1
        assert out[0].patient_id == "SINGLE"

    def test_explain_all_preserves_order(
        self, explainer, chairs, today,
    ):
        """Explanations are returned in the same order as the input
        patient list — callers rely on this to correlate with the
        ``unscheduled`` sequence from the optimiser."""
        patients = [
            {"Patient_ID": f"O{i}", "priority": 3, "expected_duration": 60,
             "earliest_time": today.replace(hour=9)}
            for i in range(3)
        ]
        out = explainer.explain_all(
            patients, schedule=[], chairs=chairs, current_date=today,
        )
        assert [e.patient_id for e in out] == ["O0", "O1", "O2"]
