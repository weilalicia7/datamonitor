"""
Weather Monitor
===============

Monitors weather conditions using Open-Meteo API (free, no key required).
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    WEATHER_API_URL,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DATA_CACHE_DIR,
    WEATHER_UPDATE_INTERVAL,
    get_logger
)

logger = get_logger(__name__)


@dataclass
class WeatherCondition:
    """Represents current weather conditions"""
    timestamp: datetime
    temperature: float
    precipitation: float
    precipitation_probability: int
    weather_code: int
    weather_description: str
    wind_speed: float
    severity: float  # 0-1 scale


@dataclass
class WeatherAlert:
    """Represents a weather alert"""
    alert_type: str
    severity: float
    start_time: datetime
    end_time: datetime
    description: str
    affected_area: str


class WeatherMonitor:
    """
    Monitors weather conditions using Open-Meteo API.

    Open-Meteo is a free, open-source weather API that doesn't require
    an API key and has no rate limits for reasonable usage.
    """

    # Weather code descriptions and severity mappings
    WEATHER_CODES = {
        0: ("Clear sky", 0.0),
        1: ("Mainly clear", 0.0),
        2: ("Partly cloudy", 0.0),
        3: ("Overcast", 0.05),
        45: ("Fog", 0.2),
        48: ("Depositing rime fog", 0.25),
        51: ("Light drizzle", 0.1),
        53: ("Moderate drizzle", 0.15),
        55: ("Dense drizzle", 0.2),
        56: ("Light freezing drizzle", 0.35),
        57: ("Dense freezing drizzle", 0.45),
        61: ("Slight rain", 0.15),
        63: ("Moderate rain", 0.25),
        65: ("Heavy rain", 0.4),
        66: ("Light freezing rain", 0.5),
        67: ("Heavy freezing rain", 0.7),
        71: ("Slight snow", 0.4),
        73: ("Moderate snow", 0.55),
        75: ("Heavy snow", 0.75),
        77: ("Snow grains", 0.3),
        80: ("Slight rain showers", 0.2),
        81: ("Moderate rain showers", 0.3),
        82: ("Violent rain showers", 0.5),
        85: ("Slight snow showers", 0.45),
        86: ("Heavy snow showers", 0.65),
        95: ("Thunderstorm", 0.6),
        96: ("Thunderstorm with slight hail", 0.7),
        99: ("Thunderstorm with heavy hail", 0.85),
    }

    def __init__(self, latitude: float = None, longitude: float = None):
        """
        Initialize weather monitor.

        Args:
            latitude: Latitude for weather data (default: Cardiff)
            longitude: Longitude for weather data (default: Cardiff)
        """
        self.latitude = latitude or DEFAULT_LATITUDE
        self.longitude = longitude or DEFAULT_LONGITUDE
        self.cache_file = DATA_CACHE_DIR / "weather_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cached weather data"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load weather cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save weather data to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, default=str)
        except Exception as e:
            logger.warning(f"Failed to save weather cache: {e}")

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if 'last_updated' not in self._cache:
            return False

        last_updated = datetime.fromisoformat(self._cache['last_updated'])
        age = (datetime.now() - last_updated).total_seconds()
        return age < WEATHER_UPDATE_INTERVAL

    def get_forecast(self, days: int = 7) -> Dict[str, Any]:
        """
        Get weather forecast from Open-Meteo API.

        Args:
            days: Number of days to forecast (max 16)

        Returns:
            Dict containing hourly and daily forecast data
        """
        # Check cache first
        if self._is_cache_valid():
            logger.debug("Using cached weather data")
            return self._cache.get('forecast', {})

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [
                "temperature_2m",
                "precipitation",
                "precipitation_probability",
                "weathercode",
                "windspeed_10m"
            ],
            "daily": [
                "weathercode",
                "precipitation_sum",
                "precipitation_probability_max",
                "temperature_2m_max",
                "temperature_2m_min"
            ],
            "timezone": "Europe/London",
            "forecast_days": min(days, 16)
        }

        try:
            response = requests.get(WEATHER_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Update cache
            self._cache['forecast'] = data
            self._cache['last_updated'] = datetime.now().isoformat()
            self._save_cache()

            logger.info("Weather data updated successfully")
            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch weather data: {e}")
            # Return cached data if available
            return self._cache.get('forecast', {})

    def get_current_conditions(self) -> Optional[WeatherCondition]:
        """
        Get current weather conditions.

        Returns:
            WeatherCondition object or None if unavailable
        """
        forecast = self.get_forecast(days=1)

        if not forecast or 'hourly' not in forecast:
            return None

        hourly = forecast['hourly']
        current_hour = datetime.now().hour

        try:
            weather_code = hourly['weathercode'][current_hour]
            description, severity = self.WEATHER_CODES.get(
                weather_code, ("Unknown", 0.3)
            )

            return WeatherCondition(
                timestamp=datetime.now(),
                temperature=hourly['temperature_2m'][current_hour],
                precipitation=hourly['precipitation'][current_hour],
                precipitation_probability=hourly['precipitation_probability'][current_hour],
                weather_code=weather_code,
                weather_description=description,
                wind_speed=hourly['windspeed_10m'][current_hour],
                severity=severity
            )
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing weather data: {e}")
            return None

    def get_weather_severity(self, weather_code: int) -> float:
        """
        Convert weather code to severity score.

        Args:
            weather_code: WMO weather code

        Returns:
            Severity score from 0 (clear) to 1 (severe)
        """
        _, severity = self.WEATHER_CODES.get(weather_code, ("Unknown", 0.3))
        return severity

    def get_hourly_severity(self, hours_ahead: int = 12) -> List[Dict]:
        """
        Get hourly severity forecast.

        Args:
            hours_ahead: Number of hours to forecast

        Returns:
            List of dicts with hour and severity
        """
        forecast = self.get_forecast()

        if not forecast or 'hourly' not in forecast:
            return []

        hourly = forecast['hourly']
        current_hour = datetime.now().hour

        results = []
        for i in range(min(hours_ahead, len(hourly['weathercode']) - current_hour)):
            idx = current_hour + i
            weather_code = hourly['weathercode'][idx]
            _, severity = self.WEATHER_CODES.get(weather_code, ("Unknown", 0.3))

            # Adjust severity based on precipitation probability
            precip_prob = hourly.get('precipitation_probability', [0] * 24)[idx]
            if precip_prob > 70:
                severity = min(1.0, severity + 0.1)

            # Adjust for wind speed
            wind_speed = hourly.get('windspeed_10m', [0] * 24)[idx]
            if wind_speed > 50:  # km/h
                severity = min(1.0, severity + 0.15)

            results.append({
                'hour': idx,
                'time': f"{idx:02d}:00",
                'weather_code': weather_code,
                'severity': round(severity, 2),
                'precipitation_prob': precip_prob,
                'wind_speed': wind_speed
            })

        return results

    def get_alerts(self) -> List[WeatherAlert]:
        """
        Get weather alerts based on forecast.

        Returns:
            List of WeatherAlert objects
        """
        alerts = []
        hourly_severity = self.get_hourly_severity(hours_ahead=24)

        # Find periods of high severity
        alert_start = None
        alert_severity = 0

        for entry in hourly_severity:
            if entry['severity'] >= 0.4:  # High severity threshold
                if alert_start is None:
                    alert_start = entry['hour']
                    alert_severity = entry['severity']
                else:
                    alert_severity = max(alert_severity, entry['severity'])
            else:
                if alert_start is not None:
                    # Create alert for the period
                    alerts.append(WeatherAlert(
                        alert_type="Weather Warning",
                        severity=alert_severity,
                        start_time=datetime.now().replace(hour=alert_start, minute=0),
                        end_time=datetime.now().replace(hour=entry['hour'], minute=0),
                        description=f"Adverse weather conditions expected",
                        affected_area="South Wales"
                    ))
                    alert_start = None
                    alert_severity = 0

        # Handle alert that extends to end of forecast
        if alert_start is not None:
            alerts.append(WeatherAlert(
                alert_type="Weather Warning",
                severity=alert_severity,
                start_time=datetime.now().replace(hour=alert_start, minute=0),
                end_time=datetime.now() + timedelta(hours=24),
                description=f"Adverse weather conditions expected",
                affected_area="South Wales"
            ))

        return alerts

    def get_severity_for_time(self, target_time: datetime) -> float:
        """
        Get weather severity for a specific time.

        Args:
            target_time: The datetime to get severity for

        Returns:
            Severity score (0-1)
        """
        forecast = self.get_forecast()

        if not forecast or 'hourly' not in forecast:
            return 0.2  # Default moderate uncertainty

        hourly = forecast['hourly']

        # Find the closest hour
        hours_from_now = int((target_time - datetime.now()).total_seconds() / 3600)
        hours_from_now = max(0, min(hours_from_now, len(hourly['weathercode']) - 1))

        weather_code = hourly['weathercode'][hours_from_now]
        return self.get_weather_severity(weather_code)


# Example usage and testing
if __name__ == "__main__":
    monitor = WeatherMonitor()

    # Get current conditions
    current = monitor.get_current_conditions()
    if current:
        print(f"Current weather: {current.weather_description}")
        print(f"Temperature: {current.temperature}°C")
        print(f"Severity: {current.severity}")

    # Get hourly severity
    print("\nHourly severity forecast:")
    for entry in monitor.get_hourly_severity(hours_ahead=6):
        print(f"  {entry['time']}: severity={entry['severity']}")

    # Get alerts
    alerts = monitor.get_alerts()
    if alerts:
        print(f"\nWeather alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  {alert.alert_type}: {alert.description}")
