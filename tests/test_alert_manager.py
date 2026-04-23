"""
Tests for monitoring/alert_manager.py — production-readiness Wave 3.5.

All external monitors are stubbed via an EventAggregator double so tests do
not hit the network or touch the real weather/traffic/news caches.

Structure:
- TestAlertConstruction    — dataclass defaults, is_active flag, to_dict
- TestAlertManagerDefaults — constructor defaults, id sequence
- TestAlertLifecycle       — create -> acknowledge -> dismiss
- TestEventAlert           — event-to-alert conversion + severity gating
- TestCheckEvents          — check_events_and_update happy-path + mode change
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root on sys.path for top-level config/monitoring imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import EventType, OperatingMode  # noqa: E402
from monitoring.alert_manager import (  # noqa: E402
    Alert,
    AlertManager,
    AlertPriority,
    AlertStatus,
)
from monitoring.event_aggregator import Event  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _make_event(severity: float = 0.5, title: str = "Test event") -> Event:
    """Build a synthetic Event with the fields the alert manager reads."""
    return Event(
        event_id="EVT-TEST-001",
        event_types=[EventType.WEATHER_EVENT],
        source="weather",
        title=title,
        description="Stormy conditions across South Wales",
        severity=severity,
        sentiment=-severity,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=6),
        location="South Wales",
        affected_postcodes=["CF10", "CF14", "CF24"],
        affected_sites=["WC"],
        noshow_adjustment=0.1,
        duration_adjustment=15,
    )


def _stub_aggregator(
    events: list[Event] | None = None,
    mode: OperatingMode = OperatingMode.NORMAL,
) -> MagicMock:
    """Return a MagicMock standing in for EventAggregator."""
    agg = MagicMock()
    agg.get_active_events.return_value = events or []
    agg.determine_operating_mode.return_value = mode
    return agg


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect DATA_CACHE_DIR so each test writes to a fresh tmp directory."""
    monkeypatch.setattr("monitoring.alert_manager.DATA_CACHE_DIR", tmp_path)
    yield


# --------------------------------------------------------------------------- #
# 1. Alert dataclass
# --------------------------------------------------------------------------- #


class TestAlertConstruction:
    def test_defaults_and_is_active(self):
        alert = Alert(
            alert_id="ALT-1",
            priority=AlertPriority.HIGH,
            title="Test",
            message="Body",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert alert.status == AlertStatus.ACTIVE
        assert alert.is_active is True
        assert alert.actions == []
        d = alert.to_dict()
        # Dict uses ENUM NAME for priority
        assert d["priority"] == "HIGH"
        assert d["status"] == "active"
        assert d["is_active"] is True


# --------------------------------------------------------------------------- #
# 2. Manager defaults
# --------------------------------------------------------------------------- #


class TestAlertManagerDefaults:
    def test_initial_state(self):
        mgr = AlertManager(event_aggregator=_stub_aggregator())
        assert mgr.alerts == {}
        assert mgr.alert_history == []
        # _alert_counter starts at 0 on a fresh cache dir
        assert mgr._alert_counter == 0
        assert mgr._last_mode is OperatingMode.NORMAL

    def test_generate_alert_id_is_monotonic(self):
        mgr = AlertManager(event_aggregator=_stub_aggregator())
        id1 = mgr._generate_alert_id()
        id2 = mgr._generate_alert_id()
        assert id1.startswith("ALT-")
        # counter embedded in suffix, so id2 must compare greater lexically
        assert id2 > id1


# --------------------------------------------------------------------------- #
# 3. Lifecycle: create / acknowledge / dismiss
# --------------------------------------------------------------------------- #


class TestAlertLifecycle:
    def test_create_acknowledge_dismiss(self):
        mgr = AlertManager(event_aggregator=_stub_aggregator())
        alert = mgr.create_alert(
            priority=AlertPriority.MEDIUM,
            title="Rain",
            message="Heavy rain expected",
            expires_hours=2,
        )
        assert alert.alert_id in mgr.alerts

        assert mgr.acknowledge_alert(alert.alert_id, user="alice") is True
        assert mgr.alerts[alert.alert_id].acknowledged_by == "alice"
        assert mgr.alerts[alert.alert_id].status == AlertStatus.ACKNOWLEDGED

        assert mgr.dismiss_alert(alert.alert_id) is True
        assert alert.alert_id not in mgr.alerts
        assert any(a.alert_id == alert.alert_id for a in mgr.alert_history)

        # Dismiss / ack of unknown alerts return False
        assert mgr.dismiss_alert("NON-EXISTENT") is False
        assert mgr.acknowledge_alert("NON-EXISTENT") is False


# --------------------------------------------------------------------------- #
# 4. Event-to-alert conversion
# --------------------------------------------------------------------------- #


class TestEventAlert:
    def test_below_threshold_returns_none(self):
        mgr = AlertManager(event_aggregator=_stub_aggregator())
        low_event = _make_event(severity=0.1)
        assert mgr.create_event_alert(low_event) is None

    def test_critical_event_becomes_critical_alert(self):
        mgr = AlertManager(event_aggregator=_stub_aggregator())
        alert = mgr.create_event_alert(_make_event(severity=0.9, title="Storm"))
        assert alert is not None
        assert alert.priority == AlertPriority.CRITICAL
        assert "Storm" in alert.title
        # High severity should include Adjust Schedule action
        action_labels = [a["label"] for a in alert.actions]
        assert "Adjust Schedule" in action_labels


# --------------------------------------------------------------------------- #
# 5. check_events_and_update — aggregator happy path + mode change
# --------------------------------------------------------------------------- #


class TestCheckEvents:
    def test_creates_alerts_for_new_events_and_mode_change(self):
        ev = _make_event(severity=0.6)
        agg = _stub_aggregator(events=[ev], mode=OperatingMode.CRISIS)

        mgr = AlertManager(event_aggregator=agg)
        new_alerts = mgr.check_events_and_update()

        # Should produce at least one event alert + one mode-change alert
        assert len(new_alerts) >= 2
        sources = [a.source_event_id for a in new_alerts if a.source_event_id]
        assert ev.event_id in sources

        # Mode tracker was updated
        assert mgr._last_mode is OperatingMode.CRISIS

        # Second call with identical state should NOT duplicate the event alert
        # (existing_event_ids guard)
        new2 = mgr.check_events_and_update()
        new2_event_ids = [a.source_event_id for a in new2 if a.source_event_id]
        assert ev.event_id not in new2_event_ids
