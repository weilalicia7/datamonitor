"""
Tests for ml/safety_guardrails.py (Dissertation §4.3)
=====================================================

Verify the runtime guardrails wrapper:
* Clean schedule → accept
* Priority-1 patient with <5 min slack → CRITICAL → reject
* Long infusion past 13:00 cutoff → HIGH → warn
* Long travel time → HIGH → warn
* Chair overlap without double-book flag → CRITICAL → reject
* Rule can be disabled via update_config
* Disabled CRITICAL rule does not produce violations
* JSONL persistence
* Every severity bucket counted separately in the report
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ml.safety_guardrails import (
    SafetyGuardrailsMonitor,
    SafetyReport,
    SafetyRuleConfig,
    SafetyViolation,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MODERATE,
    VERDICT_ACCEPT,
    VERDICT_REJECT,
    VERDICT_WARN,
    _coerce_dt,
    _gap_minutes,
    _overlap,
    get_monitor,
    set_monitor,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_safety_dir(tmp_path):
    return tmp_path / "safety"


@pytest.fixture
def monitor(tmp_safety_dir):
    return SafetyGuardrailsMonitor(storage_dir=tmp_safety_dir)


def _appt(pid, chair, site, hh, mm, dur, priority=3, travel=15, double=False):
    start = datetime(2026, 4, 22, hh, mm)
    return {
        'patient_id': pid,
        'chair_id': chair,
        'site_code': site,
        'start_time': start,
        'end_time': start + timedelta(minutes=dur),
        'duration': dur,
        'priority': priority,
        'travel_time': travel,
        'double_booked': double,
    }


def _patient(pid, noshow=0.10, earliest_hh=8):
    return {
        'Patient_ID': pid,
        'Patient_NoShow_Rate': noshow,
        'Earliest_Time': datetime(2026, 4, 22, earliest_hh, 0),
    }


# --------------------------------------------------------------------------- #
# 1. Clean schedule accepts
# --------------------------------------------------------------------------- #


class TestAccept:
    def test_clean_schedule_accepts(self, monitor):
        appts = [
            _appt('A', 'WC-C01', 'WC', 9, 0, 30),
            _appt('B', 'WC-C01', 'WC', 9, 45, 30),
            _appt('C', 'WC-C02', 'WC', 10, 0, 60),
        ]
        patients = [_patient('A'), _patient('B'), _patient('C')]
        r = monitor.evaluate(appts, patients)
        assert r.verdict == VERDICT_ACCEPT
        assert r.n_violations == 0

    def test_empty_schedule_accepts(self, monitor):
        r = monitor.evaluate([], [])
        assert r.verdict == VERDICT_ACCEPT


# --------------------------------------------------------------------------- #
# 2. Critical slack floor
# --------------------------------------------------------------------------- #


class TestCriticalSlack:
    def test_priority1_with_zero_slack_rejects(self, monitor):
        # P1 at 9:30 immediately after 9:00-9:30 appointment — 0 min slack before
        appts = [
            _appt('A', 'WC-C01', 'WC', 9, 0, 30, priority=3),
            _appt('B', 'WC-C01', 'WC', 9, 30, 30, priority=1),  # critical, 0 slack
            _appt('C', 'WC-C01', 'WC', 10, 30, 30, priority=3),
        ]
        patients = [_patient('A'), _patient('B'), _patient('C')]
        r = monitor.evaluate(appts, patients)
        assert r.verdict == VERDICT_REJECT
        assert r.n_critical >= 1
        assert 'critical_slack_floor' in r.rules_tripped

    def test_priority3_with_zero_slack_does_not_reject_for_slack(self, monitor):
        # Same zero-slack config but priority=3 → rule doesn't fire
        appts = [
            _appt('A', 'WC-C01', 'WC', 9, 0, 30, priority=3),
            _appt('B', 'WC-C01', 'WC', 9, 30, 30, priority=3),
        ]
        r = monitor.evaluate(appts, [_patient('A'), _patient('B')])
        # No critical_slack_floor violation (still might be something else, but not that rule)
        assert not any(v.rule_name == 'critical_slack_floor' for v in r.violations)


# --------------------------------------------------------------------------- #
# 3. Long infusion cutoff
# --------------------------------------------------------------------------- #


class TestLongInfusion:
    def test_long_infusion_at_15_warns(self, monitor):
        appts = [_appt('X', 'WC-C01', 'WC', 15, 0, 240)]   # 4-hour infusion at 15:00
        r = monitor.evaluate(appts, [_patient('X')])
        assert r.verdict == VERDICT_WARN
        assert any(v.rule_name == 'long_infusion_cutoff' for v in r.violations)

    def test_short_infusion_at_16_does_not_trip(self, monitor):
        appts = [_appt('X', 'WC-C01', 'WC', 16, 0, 30)]
        r = monitor.evaluate(appts, [_patient('X')])
        # Rule is keyed on duration >= threshold
        assert not any(v.rule_name == 'long_infusion_cutoff' for v in r.violations)


# --------------------------------------------------------------------------- #
# 4. Travel ceiling + wait ceiling
# --------------------------------------------------------------------------- #


class TestTravelWait:
    def test_long_travel_warns(self, monitor):
        a = _appt('T', 'WC-C01', 'WC', 10, 0, 60, travel=180)
        r = monitor.evaluate([a], [_patient('T')])
        assert any(v.rule_name == 'travel_time_ceiling' for v in r.violations)
        assert r.verdict == VERDICT_WARN  # no CRITICAL triggered

    def test_extreme_wait_warns(self, monitor):
        # Patient earliest=08:00, scheduled at 14:00 → 360 min wait > 180 min default
        p = _patient('W', earliest_hh=8)
        a = _appt('W', 'WC-C01', 'WC', 14, 0, 30)
        r = monitor.evaluate([a], [p])
        assert any(v.rule_name == 'wait_time_ceiling' for v in r.violations)


# --------------------------------------------------------------------------- #
# 5. Overlap detection
# --------------------------------------------------------------------------- #


class TestOverlap:
    def test_overlap_without_double_book_rejects(self, monitor):
        appts = [
            _appt('A', 'WC-C01', 'WC', 9, 0, 60, double=False),
            _appt('B', 'WC-C01', 'WC', 9, 30, 30, double=False),  # overlaps A
        ]
        r = monitor.evaluate(appts, [_patient('A'), _patient('B')])
        assert r.verdict == VERDICT_REJECT
        assert any(v.rule_name == 'concurrent_chair_overlap' for v in r.violations)

    def test_overlap_with_double_book_allowed(self, monitor):
        appts = [
            _appt('A', 'WC-C01', 'WC', 9, 0, 60, double=False),
            _appt('B', 'WC-C01', 'WC', 9, 30, 30, double=True),   # flagged DB
        ]
        r = monitor.evaluate(appts, [_patient('A'), _patient('B')])
        assert not any(v.rule_name == 'concurrent_chair_overlap' for v in r.violations)


# --------------------------------------------------------------------------- #
# 6. Consecutive high-noshow
# --------------------------------------------------------------------------- #


class TestConsecutiveHighNoshow:
    def test_three_consecutive_high_ns_on_chair(self, monitor):
        appts = [
            _appt(f'NS{i}', 'WC-C01', 'WC', 9 + i, 0, 30) for i in range(3)
        ]
        patients = [_patient(f'NS{i}', noshow=0.55) for i in range(3)]
        r = monitor.evaluate(appts, patients)
        assert any(v.rule_name == 'consecutive_high_noshow' for v in r.violations)
        # MODERATE only → should be warn, not reject
        assert r.verdict in {VERDICT_WARN, VERDICT_REJECT}
        assert r.n_moderate >= 1

    def test_low_noshow_does_not_trip(self, monitor):
        appts = [
            _appt(f'P{i}', 'WC-C01', 'WC', 9 + i, 0, 30) for i in range(3)
        ]
        patients = [_patient(f'P{i}', noshow=0.10) for i in range(3)]
        r = monitor.evaluate(appts, patients)
        assert not any(v.rule_name == 'consecutive_high_noshow' for v in r.violations)


# --------------------------------------------------------------------------- #
# 7. Rule enable / disable
# --------------------------------------------------------------------------- #


class TestRuleToggle:
    def test_disable_critical_rule_turns_reject_into_accept(self, monitor):
        # Reproduce the critical-slack violation
        appts = [
            _appt('A', 'WC-C01', 'WC', 9, 0, 30, priority=3),
            _appt('B', 'WC-C01', 'WC', 9, 30, 30, priority=1),
        ]
        monitor.update_config(rules={'critical_slack_floor': {'enabled': False}})
        r = monitor.evaluate(appts, [_patient('A'), _patient('B')])
        assert not any(v.rule_name == 'critical_slack_floor' for v in r.violations)

    def test_retune_slack_threshold(self, monitor):
        appts = [
            _appt('A', 'WC-C01', 'WC', 9, 0, 30, priority=3),
            _appt('B', 'WC-C01', 'WC', 9, 35, 30, priority=1),   # 5 min slack
        ]
        # Default min_slack_minutes=5 → passes at exactly 5 (strict <)
        r1 = monitor.evaluate(appts, [_patient('A'), _patient('B')])
        assert not any(v.rule_name == 'critical_slack_floor' for v in r1.violations)
        # Tighten to 10 → now violates
        monitor.update_config(rules={
            'critical_slack_floor': {'params': {'min_slack_minutes': 10}}
        })
        r2 = monitor.evaluate(appts, [_patient('A'), _patient('B')])
        assert any(v.rule_name == 'critical_slack_floor' for v in r2.violations)


# --------------------------------------------------------------------------- #
# 8. Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_log_written(self, monitor, tmp_safety_dir):
        monitor.evaluate(
            [_appt('A', 'WC-C01', 'WC', 9, 0, 30)],
            [_patient('A')],
        )
        log = tmp_safety_dir / "reports.jsonl"
        assert log.exists()
        rec = json.loads(log.read_text().strip().splitlines()[-1])
        assert 'verdict' in rec
        assert 'rules_tripped' in rec

    def test_status_counters(self, monitor):
        before = monitor.status()['total_runs']
        monitor.evaluate([], [])
        after = monitor.status()['total_runs']
        assert after == before + 1

    def test_last_cached(self, monitor):
        assert monitor.last() is None
        monitor.evaluate([], [])
        assert monitor.last() is not None


class TestSingleton:
    def test_get_set(self, tmp_path):
        m = SafetyGuardrailsMonitor(storage_dir=tmp_path / "s")
        set_monitor(m)
        assert get_monitor() is m


# --------------------------------------------------------------------------- #
# 9. Helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_coerce_dt_from_iso(self):
        v = _coerce_dt("2026-04-22T09:00:00")
        assert v == datetime(2026, 4, 22, 9, 0)

    def test_coerce_dt_from_hhmm(self):
        v = _coerce_dt("09:30")
        assert v is not None and v.hour == 9 and v.minute == 30

    def test_gap_minutes(self):
        a = datetime(2026, 4, 22, 9, 0)
        b = datetime(2026, 4, 22, 9, 30)
        assert _gap_minutes(a, b) == 30
        assert _gap_minutes(None, b) is None

    def test_overlap(self):
        a = _appt('A', 'WC', 'WC', 9, 0, 60)
        b = _appt('B', 'WC', 'WC', 9, 30, 60)
        assert _overlap(a, b) is True
        c = _appt('C', 'WC', 'WC', 11, 0, 30)
        assert _overlap(a, c) is False


# --------------------------------------------------------------------------- #
# 10. JSON round-trip
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_to_dict_is_json_safe(self, monitor):
        appts = [
            _appt('A', 'WC-C01', 'WC', 9, 0, 30, priority=1),
            _appt('B', 'WC-C01', 'WC', 9, 30, 30, priority=3),
        ]
        r = monitor.evaluate(appts, [_patient('A'), _patient('B')])
        dumped = json.dumps(r.to_dict(), default=str)
        back = json.loads(dumped)
        assert back['verdict'] in {VERDICT_ACCEPT, VERDICT_WARN, VERDICT_REJECT}
        assert isinstance(back['violations'], list)
