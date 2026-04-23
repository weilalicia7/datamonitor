"""
Tests for optimization.emergency_mode — EmergencyModeHandler.

Covers:
    * Default construction + mode defaults to NORMAL.
    * set_mode transitions and records history.
    * determine_mode returns NORMAL for low-severity inputs.
    * create_emergency_schedule defers low-priority patients under CRISIS.
    * get_mode_recommendations returns action list for each mode.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OperatingMode  # noqa: E402
from optimization.emergency_mode import (  # noqa: E402
    EmergencyModeHandler,
    EmergencySchedule,
    ModeAction,
)
from optimization.optimizer import Patient  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def today():
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def _patient(pid, priority, today, duration=60, postcode='CF14'):
    return Patient(
        patient_id=pid,
        priority=priority,
        protocol='Standard',
        expected_duration=duration,
        postcode=postcode,
        earliest_time=today.replace(hour=8),
        latest_time=today.replace(hour=17),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_construction_is_normal_mode():
    handler = EmergencyModeHandler()
    assert handler.current_mode == OperatingMode.NORMAL
    assert handler._mode_history == []
    # Settings lookup table has an entry for every mode
    for mode in OperatingMode:
        assert mode in handler.MODE_SETTINGS


def test_set_mode_transitions_and_history():
    handler = EmergencyModeHandler()
    handler.set_mode(OperatingMode.CRISIS, reason='test reason')
    assert handler.current_mode == OperatingMode.CRISIS
    assert len(handler._mode_history) == 1

    # Setting the same mode twice must not re-append history.
    handler.set_mode(OperatingMode.CRISIS)
    assert len(handler._mode_history) == 1

    handler.set_mode(OperatingMode.EMERGENCY)
    assert len(handler._mode_history) == 2


def test_determine_mode_returns_normal_for_low_severity():
    """Low severity (< 0.3) must map to NORMAL."""
    handler = EmergencyModeHandler()
    # Explicit check of the OperatingMode.NORMAL branch.
    assert handler.determine_mode(0.0) == OperatingMode.NORMAL
    assert handler.determine_mode(0.29) == OperatingMode.NORMAL


def test_crisis_mode_defers_low_priority_patients(today):
    """Under CRISIS, priority-4 patients must be deferred / cancelled."""
    handler = EmergencyModeHandler()
    handler.set_mode(OperatingMode.CRISIS)

    patients = [
        _patient('P1', priority=1, today=today),
        _patient('P2', priority=2, today=today),
        _patient('P3', priority=3, today=today),
        _patient('P4', priority=4, today=today),  # must be deferred
    ]

    schedule = handler.create_emergency_schedule(patients, existing_schedule=[], date=today)

    assert isinstance(schedule, EmergencySchedule)
    assert schedule.mode == OperatingMode.CRISIS
    # 'cancelled' collects deferred patients (priority > min_priority=3).
    assert 'P4' in schedule.cancelled
    assert 'P1' not in schedule.cancelled
    # At least one defer action was recorded
    assert any(a.action_type == 'defer' for a in schedule.actions_taken)
    assert all(isinstance(a, ModeAction) for a in schedule.actions_taken)


def test_get_mode_recommendations_for_normal_severity():
    """Low severity (below the 0.3 elevated threshold) must return a
    well-formed recommendation dict keyed by 'recommended_mode', 'actions',
    and 'summary' with the NORMAL-mode summary text."""
    handler = EmergencyModeHandler()
    rec = handler.get_mode_recommendations(
        severity_score=0.05,
        affected_postcodes=[],
        event_count=0,
    )
    for key in ('recommended_mode', 'current_mode', 'actions',
                'summary', 'active_events', 'severity_score'):
        assert key in rec
    assert isinstance(rec['actions'], list)
    # In NORMAL mode the actions list is empty and the summary is the
    # 'Continue normal operations' text.
    assert rec['summary'] == 'Continue normal operations'
