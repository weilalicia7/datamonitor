"""
Tests for monitoring/event_aggregator.py — production-readiness Wave 3.5.

The aggregator composes three external monitors (weather/traffic/news);
we replace those dependencies on the INSTANCE so tests never hit the
network.

Structure:
- TestEventDataclass      — is_active, severity_level, to_dict
- TestIdGeneration        — hashlib md5 with usedforsecurity=False (T4.9 guard)
- TestImpactCalculation   — severity level -> EVENT_IMPACT_MATRIX lookup
- TestAggregateEvents     — get_active_events collects from all sources
- TestOperatingMode       — mode selection from aggregate severity
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import EventType, OperatingMode  # noqa: E402
from monitoring.event_aggregator import Event, EventAggregator  # noqa: E402
from monitoring.news_monitor import NewsItem  # noqa: E402
from monitoring.traffic_monitor import TrafficIncident  # noqa: E402
from monitoring.weather_monitor import WeatherCondition  # noqa: E402


def _make_aggregator_with_stubs(
    weather_condition: WeatherCondition | None = None,
    weather_alerts: list | None = None,
    traffic_incidents: list | None = None,
    news_items: list | None = None,
) -> EventAggregator:
    """Build an EventAggregator with all three monitors replaced by mocks."""
    agg = EventAggregator()
    agg.weather_monitor = MagicMock()
    agg.weather_monitor.get_current_conditions.return_value = weather_condition
    agg.weather_monitor.get_alerts.return_value = weather_alerts or []
    agg.traffic_monitor = MagicMock()
    agg.traffic_monitor.get_incidents.return_value = traffic_incidents or []
    agg.news_monitor = MagicMock()
    agg.news_monitor.get_relevant_items.return_value = news_items or []
    return agg


# --------------------------------------------------------------------------- #
# Event dataclass
# --------------------------------------------------------------------------- #


class TestEventDataclass:
    def test_severity_level_thresholds(self):
        base = {
            "event_id": "EV1",
            "event_types": [EventType.WEATHER_EVENT],
            "source": "weather",
            "title": "t",
            "description": "d",
            "sentiment": 0.0,
            "created_at": datetime.now(),
            "expires_at": None,
            "location": "",
        }
        assert Event(severity=0.9, **base).severity_level == "critical"
        assert Event(severity=0.6, **base).severity_level == "high"
        assert Event(severity=0.3, **base).severity_level == "medium"
        assert Event(severity=0.1, **base).severity_level == "low"

    def test_is_active_respects_expiry(self):
        past = datetime.now() - timedelta(hours=1)
        future = datetime.now() + timedelta(hours=1)
        ev_expired = Event(
            event_id="E1", event_types=[EventType.OTHER], source="manual",
            title="t", description="d", severity=0.5, sentiment=0.0,
            created_at=past, expires_at=past, location="",
        )
        ev_live = Event(
            event_id="E2", event_types=[EventType.OTHER], source="manual",
            title="t", description="d", severity=0.5, sentiment=0.0,
            created_at=past, expires_at=future, location="",
        )
        assert ev_expired.is_active is False
        assert ev_live.is_active is True
        # None expires -> always active
        ev_live.expires_at = None
        assert ev_live.is_active is True

        d = ev_live.to_dict()
        assert d["severity_level"] == "high"
        assert d["event_types"] == ["other"]


# --------------------------------------------------------------------------- #
# _generate_event_id — T4.9 guard: must NOT raise hashlib FIPS warning
# --------------------------------------------------------------------------- #


class TestIdGeneration:
    def test_id_is_deterministic_per_day_and_no_fips_warning(self):
        agg = _make_aggregator_with_stubs()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote warnings to exceptions
            id1 = agg._generate_event_id("weather", "abc")
            id2 = agg._generate_event_id("weather", "abc")
        # 12-char md5 prefix, stable within a calendar day
        assert id1 == id2
        assert len(id1) == 12
        # Different seed -> different id
        assert agg._generate_event_id("weather", "xyz") != id1


# --------------------------------------------------------------------------- #
# _calculate_impact — EVENT_IMPACT_MATRIX lookup correctness
# --------------------------------------------------------------------------- #


class TestImpactCalculation:
    def test_severity_level_maps_to_matrix(self):
        agg = _make_aggregator_with_stubs()
        # High severity weather -> (0.30, 30) per EVENT_IMPACT_MATRIX
        ns, dur = agg._calculate_impact([EventType.WEATHER_EVENT], 0.6)
        assert ns == pytest.approx(0.30)
        assert dur == 30

        # Unknown combination (e.g. LOW severity HEALTH_ALERT not in matrix)
        # should return (0, 0) fallback
        ns2, dur2 = agg._calculate_impact([EventType.HEALTH_ALERT], 0.1)
        assert ns2 == 0.0 and dur2 == 0


# --------------------------------------------------------------------------- #
# get_active_events — weather + traffic + news paths
# --------------------------------------------------------------------------- #


class TestAggregateEvents:
    def test_collects_events_from_all_sources_sorted_by_severity(self):
        wx = WeatherCondition(
            timestamp=datetime.now(),
            temperature=5.0,
            precipitation=3.0,
            precipitation_probability=90,
            weather_code=75,
            weather_description="Heavy snow",
            wind_speed=30.0,
            severity=0.75,
        )
        traf = TrafficIncident(
            incident_id="T1",
            incident_type="ACCIDENT",
            severity=0.55,
            description="Crash on M4",
            location="M4 J32",
            latitude=51.52,
            longitude=-3.18,
            road="M4",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2),
            delay_minutes=25,
            affected_postcodes=["CF14"],
        )
        news = NewsItem(
            item_id="N1",
            title="Flooding in Cardiff",
            description="Major flooding closes roads in Cardiff city centre",
            link="http://example.invalid",
            published=datetime.now(),
            source="BBC Wales",
            is_relevant=True,
            relevance_score=0.8,
            keywords_found=["flooding", "cardiff"],
        )

        agg = _make_aggregator_with_stubs(
            weather_condition=wx,
            traffic_incidents=[traf],
            news_items=[news],
        )
        events = agg.get_active_events()

        # Every source produced at least one event
        sources = {e.source for e in events}
        assert {"weather", "traffic", "news"}.issubset(sources)
        # Sort: severity descending
        severities = [e.severity for e in events]
        assert severities == sorted(severities, reverse=True)

    def test_low_severity_weather_is_skipped(self):
        # severity < 0.2 should NOT produce a weather event
        wx = WeatherCondition(
            timestamp=datetime.now(),
            temperature=15.0, precipitation=0.0, precipitation_probability=0,
            weather_code=1, weather_description="Mainly clear",
            wind_speed=5.0, severity=0.0,
        )
        agg = _make_aggregator_with_stubs(weather_condition=wx)
        events = agg.get_active_events()
        assert all(e.source != "weather" for e in events)


# --------------------------------------------------------------------------- #
# determine_operating_mode
# --------------------------------------------------------------------------- #


class TestOperatingMode:
    def test_empty_returns_normal(self):
        agg = _make_aggregator_with_stubs()
        assert agg.determine_operating_mode() is OperatingMode.NORMAL

    def test_high_severity_weather_drives_crisis(self):
        # High severity weather (0.75) -> above crisis threshold (0.5)
        wx = WeatherCondition(
            timestamp=datetime.now(),
            temperature=-2.0, precipitation=10.0, precipitation_probability=95,
            weather_code=75, weather_description="Heavy snow",
            wind_speed=40.0, severity=0.75,
        )
        agg = _make_aggregator_with_stubs(weather_condition=wx)
        mode = agg.determine_operating_mode()
        assert mode in (OperatingMode.CRISIS, OperatingMode.EMERGENCY)

    def test_get_aggregate_impact_on_empty(self):
        agg = _make_aggregator_with_stubs()
        impact = agg.get_aggregate_impact()
        assert impact == {"noshow_adjustment": 0.0, "duration_adjustment": 0}
