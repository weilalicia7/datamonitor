"""
Tests for monitoring/weather_monitor.py — production-readiness Wave 3.5.

Open-Meteo HTTP calls go through requests.get inside get_forecast; we
stub that function every test so the suite is fully offline.

Structure:
- TestWeatherCodeLookup   — severity mapping table + unknown-code fallback
- TestGetForecastMocked   — happy path: API returns 24h of hourly data
- TestGetCurrentConditions — parses current hour into a WeatherCondition
- TestApiErrorPath        — RequestException falls back to cached data
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.weather_monitor import (  # noqa: E402
    WeatherAlert,
    WeatherCondition,
    WeatherMonitor,
)


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("monitoring.weather_monitor.DATA_CACHE_DIR", tmp_path)
    yield


def _fake_forecast_payload() -> dict:
    """Build a minimal but valid Open-Meteo hourly forecast payload (48h)."""
    n = 48
    # Rotate a mix of codes so we get non-zero severity for some hours
    codes = [
        1,  2, 61, 63, 75, 61, 75, 82,
        95, 96, 99, 71,  0,  1,  3,  3,
        61, 63, 61, 71, 63, 65, 66, 67,
    ] * 2
    temps = [10.0 + (i % 10) for i in range(n)]
    precip = [0.5 * (i % 5) for i in range(n)]
    pprob = [(i * 5) % 101 for i in range(n)]
    wind = [10.0 + (i % 20) for i in range(n)]
    return {
        "hourly": {
            "temperature_2m": temps,
            "precipitation": precip,
            "precipitation_probability": pprob,
            "weathercode": codes,
            "windspeed_10m": wind,
        },
        "daily": {
            "weathercode": [63, 75],
            "precipitation_sum": [3.5, 2.1],
            "precipitation_probability_max": [90, 70],
            "temperature_2m_max": [14, 10],
            "temperature_2m_min": [5, 2],
        },
    }


# --------------------------------------------------------------------------- #
# 1. Weather-code severity map (no HTTP)
# --------------------------------------------------------------------------- #


class TestWeatherCodeLookup:
    def test_clear_sky_has_zero_severity(self):
        mon = WeatherMonitor()
        assert mon.get_weather_severity(0) == 0.0

    def test_thunder_with_hail_is_severe(self):
        mon = WeatherMonitor()
        # code 99: "Thunderstorm with heavy hail" -> 0.85
        assert mon.get_weather_severity(99) >= 0.8

    def test_unknown_code_returns_fallback(self):
        mon = WeatherMonitor()
        # Unknown code returns the ("Unknown", 0.3) fallback
        assert mon.get_weather_severity(-999) == 0.3


# --------------------------------------------------------------------------- #
# 2. get_forecast with requests.get mocked
# --------------------------------------------------------------------------- #


class TestGetForecastMocked:
    def test_api_call_returns_parsed_dict_and_caches(self):
        mon = WeatherMonitor()
        resp = MagicMock()
        resp.json.return_value = _fake_forecast_payload()
        resp.raise_for_status = MagicMock()
        with patch(
            "monitoring.weather_monitor.requests.get", return_value=resp
        ) as g:
            data = mon.get_forecast(days=1)
        g.assert_called_once()
        assert "hourly" in data
        assert len(data["hourly"]["weathercode"]) == 48
        # Cache file should have been written
        assert mon.cache_file.exists()


# --------------------------------------------------------------------------- #
# 3. get_current_conditions — end-to-end with the same mocked API
# --------------------------------------------------------------------------- #


class TestGetCurrentConditions:
    def test_returns_weather_condition_object(self):
        mon = WeatherMonitor()
        resp = MagicMock()
        resp.json.return_value = _fake_forecast_payload()
        resp.raise_for_status = MagicMock()
        with patch(
            "monitoring.weather_monitor.requests.get", return_value=resp
        ):
            cond = mon.get_current_conditions()
        assert isinstance(cond, WeatherCondition)
        # hour index is datetime.now().hour so values must sit inside the arrays
        assert cond.weather_code in _fake_forecast_payload()["hourly"]["weathercode"]
        assert 0.0 <= cond.severity <= 1.0


# --------------------------------------------------------------------------- #
# 4. API error path — RequestException -> empty/cached forecast
# --------------------------------------------------------------------------- #


class TestApiErrorPath:
    def test_request_exception_returns_cached_forecast(self):
        mon = WeatherMonitor()
        with patch(
            "monitoring.weather_monitor.requests.get",
            side_effect=requests.RequestException("down"),
        ):
            data = mon.get_forecast(days=1)
        # Fresh cache -> empty dict, but call did NOT raise
        assert data == {}
        # Current conditions must now be None (no hourly data)
        with patch(
            "monitoring.weather_monitor.requests.get",
            side_effect=requests.RequestException("down"),
        ):
            assert mon.get_current_conditions() is None
