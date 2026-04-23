"""
Tests for monitoring/traffic_monitor.py — production-readiness Wave 3.5.

All TomTom HTTP calls go through requests.get in _fetch_api_incidents;
tests stub that function so the suite is fully hermetic.  Cache file is
redirected to tmp_path per test.

Structure:
- TestIncidentDataclass   — construction of TrafficIncident / RouteConditions
- TestHaversine           — geometric distance correctness (no HTTP)
- TestManualIncidents     — add / remove / clear_expired life-cycle
- TestApiFetchMocked      — requests.get stubbed with a canned TomTom payload
- TestApiFetchErrors      — RequestException -> falls back to cache
- TestGetRouteConditions  — route distance + rush-hour delay logic
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.traffic_monitor import (  # noqa: E402
    RouteConditions,
    TrafficIncident,
    TrafficMonitor,
)


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("monitoring.traffic_monitor.DATA_CACHE_DIR", tmp_path)
    # Keep the API key set so _fetch_api_incidents actually reaches the
    # requests.get call path (tests stub that call).
    monkeypatch.setattr("monitoring.traffic_monitor.TOMTOM_API_KEY", "TEST-KEY")
    yield


def _make_incident(**overrides) -> TrafficIncident:
    base = {
        "incident_id": "MANUAL-1",
        "incident_type": "ACCIDENT",
        "severity": 0.7,
        "description": "Multi-vehicle accident",
        "location": "M4 J32",
        "latitude": 51.52,
        "longitude": -3.18,
        "road": "M4",
        "start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(hours=2),
        "delay_minutes": 30,
        "affected_postcodes": ["CF14"],
    }
    base.update(overrides)
    return TrafficIncident(**base)


# --------------------------------------------------------------------------- #
# 1. Dataclass construction
# --------------------------------------------------------------------------- #


class TestIncidentDataclass:
    def test_manual_incident_fields(self):
        inc = _make_incident()
        assert inc.incident_id == "MANUAL-1"
        assert inc.delay_minutes == 30
        assert inc.affected_postcodes == ["CF14"]

    def test_route_conditions_default_incidents_is_empty_list(self):
        rc = RouteConditions(
            from_location="NP20", to_location="WC",
            normal_duration_minutes=25, current_duration_minutes=40,
            delay_minutes=15, severity=0.5,
        )
        assert rc.incidents == []


# --------------------------------------------------------------------------- #
# 2. Haversine
# --------------------------------------------------------------------------- #


class TestHaversine:
    def test_self_distance_is_zero(self):
        mon = TrafficMonitor(api_key="TEST")
        assert mon._haversine_distance(51.5, -3.2, 51.5, -3.2) == 0

    def test_cardiff_to_swansea_is_roughly_60km(self):
        mon = TrafficMonitor(api_key="TEST")
        # Cardiff ~51.48,-3.18 ; Swansea ~51.62,-3.94 -> great-circle ~56 km
        d = mon._haversine_distance(51.48, -3.18, 51.62, -3.94)
        assert 40 < d < 80


# --------------------------------------------------------------------------- #
# 3. Manual incident life-cycle
# --------------------------------------------------------------------------- #


class TestManualIncidents:
    def test_add_then_remove(self):
        mon = TrafficMonitor(api_key="TEST")
        inc = _make_incident()
        mon.add_manual_incident(inc)
        assert len(mon._manual_incidents) == 1
        assert mon.remove_manual_incident("MANUAL-1") is True
        assert len(mon._manual_incidents) == 0
        # Removing unknown returns False
        assert mon.remove_manual_incident("NOPE") is False

    def test_clear_expired_incidents(self):
        mon = TrafficMonitor(api_key="TEST")
        expired = _make_incident(
            incident_id="OLD",
            end_time=datetime.now() - timedelta(hours=1),
        )
        live = _make_incident(
            incident_id="NEW",
            end_time=datetime.now() + timedelta(hours=1),
        )
        mon.add_manual_incident(expired)
        mon.add_manual_incident(live)
        removed = mon.clear_expired_incidents()
        assert removed == 1
        ids = [i.incident_id for i in mon._manual_incidents]
        assert ids == ["NEW"]


# --------------------------------------------------------------------------- #
# 4. API fetch happy path — requests.get mocked
# --------------------------------------------------------------------------- #


class TestApiFetchMocked:
    def test_fetch_parses_tomtom_payload(self):
        mon = TrafficMonitor(api_key="TEST")
        payload = {
            "incidents": [
                {
                    "id": "abc-1",
                    "geometry": {"coordinates": [[-3.18, 51.52]]},
                    "properties": {
                        "iconCategory": "ACCIDENT",
                        "magnitudeOfDelay": 4,
                        "events": [{"description": "Crash M4 J32", "code": 1}],
                        "startTime": datetime.now().isoformat() + "Z",
                        "endTime": (datetime.now() + timedelta(hours=1)).isoformat() + "Z",
                    },
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()

        with patch(
            "monitoring.traffic_monitor.requests.get", return_value=mock_resp
        ) as g:
            incidents = mon.get_incidents()

        # The HTTP mock was invoked exactly once
        g.assert_called_once()
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.incident_type == "ACCIDENT"
        # delay_magnitude 4 -> bumps severity (+0.2 over ACCIDENT default 0.7, capped at 1.0)
        assert 0.7 <= inc.severity <= 1.0
        # delay_minutes = magnitude * 10
        assert inc.delay_minutes == 40


# --------------------------------------------------------------------------- #
# 5. API fetch error — falls back to cache (empty list)
# --------------------------------------------------------------------------- #


class TestApiFetchErrors:
    def test_request_exception_returns_cached_incidents(self):
        mon = TrafficMonitor(api_key="TEST")
        with patch(
            "monitoring.traffic_monitor.requests.get",
            side_effect=requests.RequestException("500 Server Error"),
        ):
            incidents = mon.get_incidents()
        # Fresh cache -> empty, and there were no manual incidents
        assert incidents == []


# --------------------------------------------------------------------------- #
# 6. get_route_conditions — uses haversine + rush-hour patterns
# --------------------------------------------------------------------------- #


class TestGetRouteConditions:
    def test_route_has_nonnegative_delay_and_duration(self):
        mon = TrafficMonitor(api_key="TEST")
        # Stub API so get_incidents returns only manual list
        with patch(
            "monitoring.traffic_monitor.requests.get",
            side_effect=requests.RequestException("down"),
        ):
            rc = mon.get_route_conditions("NP20", "WC")
        assert rc.current_duration_minutes >= rc.normal_duration_minutes
        assert rc.delay_minutes >= 0
        # Unknown site falls back to a defensive default
        rc2 = mon.get_route_conditions("ZZ99", "NOPE")
        assert rc2.normal_duration_minutes == 30
        assert rc2.severity == 0.0
