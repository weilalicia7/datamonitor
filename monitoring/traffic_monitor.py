"""
Traffic Monitor
===============

Monitors traffic conditions and incidents affecting patient routes.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TOMTOM_API_KEY,
    TOMTOM_API_URL,
    DATA_CACHE_DIR,
    TRAFFIC_UPDATE_INTERVAL,
    POSTCODE_COORDINATES,
    DEFAULT_SITES,
    get_logger
)

logger = get_logger(__name__)


@dataclass
class TrafficIncident:
    """Represents a traffic incident"""
    incident_id: str
    incident_type: str
    severity: float  # 0-1 scale
    description: str
    location: str
    latitude: float
    longitude: float
    road: str
    start_time: datetime
    end_time: Optional[datetime]
    delay_minutes: int
    affected_postcodes: List[str] = field(default_factory=list)


@dataclass
class RouteConditions:
    """Represents traffic conditions on a route"""
    from_location: str
    to_location: str
    normal_duration_minutes: int
    current_duration_minutes: int
    delay_minutes: int
    severity: float
    incidents: List[TrafficIncident] = field(default_factory=list)


class TrafficMonitor:
    """
    Monitors traffic conditions using TomTom API and manual inputs.

    TomTom offers a free tier with 2,500 requests per day.
    Falls back to manual/historical patterns when API unavailable.
    """

    # Key routes to monitor in South Wales
    MONITORED_ROUTES = [
        {"name": "M4 Cardiff-Newport", "road": "M4", "junctions": "J29-J33"},
        {"name": "A470 Cardiff North", "road": "A470", "area": "Cardiff North"},
        {"name": "A48 Eastern Avenue", "road": "A48", "area": "Cardiff East"},
        {"name": "A4232 Link Road", "road": "A4232", "area": "Cardiff Bay"},
        {"name": "M4 Cardiff-Swansea", "road": "M4", "junctions": "J33-J47"},
    ]

    # Incident type severity mapping
    INCIDENT_SEVERITY = {
        'ACCIDENT': 0.7,
        'CONGESTION': 0.3,
        'ROAD_CLOSURE': 0.9,
        'ROADWORK': 0.4,
        'WEATHER': 0.5,
        'HAZARD': 0.4,
        'LANE_CLOSURE': 0.5,
        'DISABLED_VEHICLE': 0.3,
        'POLICE': 0.4,
        'OTHER': 0.3
    }

    # Rush hour patterns (hour: delay_factor)
    RUSH_HOUR_PATTERNS = {
        7: 1.3,
        8: 1.5,
        9: 1.4,
        16: 1.3,
        17: 1.5,
        18: 1.4,
    }

    def __init__(self, api_key: str = None):
        """
        Initialize traffic monitor.

        Args:
            api_key: TomTom API key (optional, falls back to manual mode)
        """
        self.api_key = api_key or TOMTOM_API_KEY
        self.cache_file = DATA_CACHE_DIR / "traffic_cache.json"
        self._cache = self._load_cache()
        self._manual_incidents: List[TrafficIncident] = []

    def _load_cache(self) -> Dict:
        """Load cached traffic data"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load traffic cache: {e}")
        return {'incidents': [], 'last_updated': None}

    def _save_cache(self) -> None:
        """Save traffic data to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, default=str)
        except Exception as e:
            logger.warning(f"Failed to save traffic cache: {e}")

    def _haversine_distance(self, lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points in kilometers.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def add_manual_incident(self, incident: TrafficIncident) -> None:
        """
        Add a manually reported traffic incident.

        Args:
            incident: TrafficIncident object
        """
        self._manual_incidents.append(incident)
        logger.info(f"Added manual incident: {incident.description}")

    def remove_manual_incident(self, incident_id: str) -> bool:
        """
        Remove a manual incident by ID.

        Args:
            incident_id: Incident ID to remove

        Returns:
            True if removed, False if not found
        """
        for i, incident in enumerate(self._manual_incidents):
            if incident.incident_id == incident_id:
                del self._manual_incidents[i]
                logger.info(f"Removed manual incident: {incident_id}")
                return True
        return False

    def clear_expired_incidents(self) -> int:
        """
        Remove expired manual incidents.

        Returns:
            Number of incidents removed
        """
        now = datetime.now()
        original_count = len(self._manual_incidents)

        self._manual_incidents = [
            inc for inc in self._manual_incidents
            if inc.end_time is None or inc.end_time > now
        ]

        removed = original_count - len(self._manual_incidents)
        if removed > 0:
            logger.info(f"Cleared {removed} expired incidents")

        return removed

    def _fetch_api_incidents(self) -> List[Dict]:
        """
        Fetch incidents from TomTom API.

        Returns:
            List of incident dictionaries
        """
        if not self.api_key:
            logger.debug("No API key configured, using manual mode only")
            return []

        # Check cache
        if self._cache.get('last_updated'):
            last_updated = datetime.fromisoformat(self._cache['last_updated'])
            age = (datetime.now() - last_updated).total_seconds()
            if age < TRAFFIC_UPDATE_INTERVAL:
                return self._cache.get('incidents', [])

        # South Wales bounding box
        bbox = "-4.5,51.3,-2.5,51.8"  # min_lon, min_lat, max_lon, max_lat

        params = {
            "key": self.api_key,
            "bbox": bbox,
            "fields": "{incidents{type,geometry{coordinates},properties{iconCategory,magnitudeOfDelay,events{description,code},startTime,endTime}}}",
            "language": "en-GB"
        }

        try:
            response = requests.get(TOMTOM_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            incidents = data.get('incidents', [])

            # Update cache
            self._cache['incidents'] = incidents
            self._cache['last_updated'] = datetime.now().isoformat()
            self._save_cache()

            logger.info(f"Fetched {len(incidents)} traffic incidents from API")
            return incidents

        except requests.RequestException as e:
            logger.error(f"Failed to fetch traffic data: {e}")
            return self._cache.get('incidents', [])

    def get_incidents(self) -> List[TrafficIncident]:
        """
        Get all active traffic incidents.

        Returns:
            List of TrafficIncident objects
        """
        incidents = []

        # Get API incidents
        api_data = self._fetch_api_incidents()
        for item in api_data:
            try:
                props = item.get('properties', {})
                coords = item.get('geometry', {}).get('coordinates', [[0, 0]])[0]

                incident_type = props.get('iconCategory', 'OTHER').upper()
                severity = self.INCIDENT_SEVERITY.get(incident_type, 0.3)

                # Adjust severity based on delay magnitude
                delay_magnitude = props.get('magnitudeOfDelay', 0)
                if delay_magnitude > 3:
                    severity = min(1.0, severity + 0.2)

                events = props.get('events', [])
                description = events[0].get('description', 'Traffic incident') if events else 'Traffic incident'

                incident = TrafficIncident(
                    incident_id=f"API-{item.get('id', datetime.now().timestamp())}",
                    incident_type=incident_type,
                    severity=severity,
                    description=description,
                    location="South Wales",
                    latitude=coords[1] if len(coords) > 1 else 0,
                    longitude=coords[0] if coords else 0,
                    road=self._determine_road(coords),
                    start_time=datetime.fromisoformat(props.get('startTime', datetime.now().isoformat()).replace('Z', '+00:00')),
                    end_time=datetime.fromisoformat(props['endTime'].replace('Z', '+00:00')) if props.get('endTime') else None,
                    delay_minutes=delay_magnitude * 10,  # Approximate
                    affected_postcodes=self._get_affected_postcodes(coords[1] if len(coords) > 1 else 0, coords[0] if coords else 0)
                )
                incidents.append(incident)

            except Exception as e:
                logger.warning(f"Error parsing incident: {e}")
                continue

        # Add manual incidents
        self.clear_expired_incidents()
        incidents.extend(self._manual_incidents)

        return incidents

    def _determine_road(self, coords: List) -> str:
        """Determine which road an incident is on based on coordinates"""
        if not coords or len(coords) < 2:
            return "Unknown"

        lat, lon = coords[1], coords[0]

        # Simple heuristics based on location
        if -3.3 < lon < -2.8 and 51.5 < lat < 51.7:
            return "M4"
        elif -3.3 < lon < -3.1 and 51.4 < lat < 51.55:
            return "A470"
        elif lon > -3.1 and 51.45 < lat < 51.55:
            return "A48"

        return "Local Road"

    def _get_affected_postcodes(self, lat: float, lon: float,
                                radius_km: float = 10) -> List[str]:
        """
        Get postcodes affected by an incident.

        Args:
            lat, lon: Incident coordinates
            radius_km: Radius to consider affected

        Returns:
            List of affected postcode districts
        """
        affected = []

        for postcode, info in POSTCODE_COORDINATES.items():
            distance = self._haversine_distance(
                lat, lon,
                info['lat'], info['lon']
            )
            if distance <= radius_km:
                affected.append(postcode)

        return affected

    def get_route_conditions(self, from_postcode: str,
                             to_site_code: str) -> RouteConditions:
        """
        Get traffic conditions for a specific route.

        Args:
            from_postcode: Origin postcode district (e.g., 'CF14')
            to_site_code: Destination site code (e.g., 'WC')

        Returns:
            RouteConditions object
        """
        # Get coordinates
        from_coords = POSTCODE_COORDINATES.get(from_postcode, {})
        to_coords = None
        for site in DEFAULT_SITES:
            if site['code'] == to_site_code:
                to_coords = {'lat': site['lat'], 'lon': site['lon']}
                break

        if not from_coords or not to_coords:
            return RouteConditions(
                from_location=from_postcode,
                to_location=to_site_code,
                normal_duration_minutes=30,
                current_duration_minutes=30,
                delay_minutes=0,
                severity=0.0
            )

        # Calculate base distance
        distance = self._haversine_distance(
            from_coords['lat'], from_coords['lon'],
            to_coords['lat'], to_coords['lon']
        )

        # Estimate normal duration (assume 40 km/h average)
        normal_duration = int(distance / 40 * 60)

        # Get incidents on route
        incidents = self.get_incidents()
        route_incidents = []
        total_delay = 0
        max_severity = 0.0

        for incident in incidents:
            # Check if incident is near the route
            inc_distance_from_origin = self._haversine_distance(
                from_coords['lat'], from_coords['lon'],
                incident.latitude, incident.longitude
            )
            inc_distance_from_dest = self._haversine_distance(
                to_coords['lat'], to_coords['lon'],
                incident.latitude, incident.longitude
            )

            # Incident is on route if it's closer to both points than the total distance
            if inc_distance_from_origin < distance and inc_distance_from_dest < distance:
                route_incidents.append(incident)
                total_delay += incident.delay_minutes
                max_severity = max(max_severity, incident.severity)

        # Apply rush hour factor
        current_hour = datetime.now().hour
        rush_factor = self.RUSH_HOUR_PATTERNS.get(current_hour, 1.0)
        rush_delay = int(normal_duration * (rush_factor - 1))
        total_delay += rush_delay

        current_duration = normal_duration + total_delay

        return RouteConditions(
            from_location=from_postcode,
            to_location=to_site_code,
            normal_duration_minutes=normal_duration,
            current_duration_minutes=current_duration,
            delay_minutes=total_delay,
            severity=max_severity,
            incidents=route_incidents
        )

    def get_overall_severity(self) -> float:
        """
        Get overall traffic severity score.

        Returns:
            Severity score from 0 to 1
        """
        incidents = self.get_incidents()

        if not incidents:
            # Check for rush hour
            current_hour = datetime.now().hour
            if current_hour in self.RUSH_HOUR_PATTERNS:
                return 0.2
            return 0.0

        # Weight by severity
        total_severity = sum(inc.severity for inc in incidents)
        avg_severity = total_severity / len(incidents)

        # Adjust for number of incidents
        count_factor = min(1.0, len(incidents) / 5)

        return min(1.0, avg_severity * 0.7 + count_factor * 0.3)

    def get_affected_postcodes(self) -> List[str]:
        """
        Get all postcodes affected by current incidents.

        Returns:
            List of unique postcode districts
        """
        affected = set()
        for incident in self.get_incidents():
            affected.update(incident.affected_postcodes)
        return list(affected)


# Example usage
if __name__ == "__main__":
    monitor = TrafficMonitor()

    # Add a test incident
    test_incident = TrafficIncident(
        incident_id="MANUAL-001",
        incident_type="ACCIDENT",
        severity=0.7,
        description="Multi-vehicle accident on M4 J32",
        location="M4 Junction 32",
        latitude=51.52,
        longitude=-3.18,
        road="M4",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=2),
        delay_minutes=30,
        affected_postcodes=["CF14", "CF15", "CF5"]
    )
    monitor.add_manual_incident(test_incident)

    # Get incidents
    print("Current incidents:")
    for inc in monitor.get_incidents():
        print(f"  - {inc.description} (severity: {inc.severity})")

    # Get route conditions
    conditions = monitor.get_route_conditions("NP20", "WC")
    print(f"\nRoute NP20 to Whitchurch:")
    print(f"  Normal: {conditions.normal_duration_minutes} min")
    print(f"  Current: {conditions.current_duration_minutes} min")
    print(f"  Delay: {conditions.delay_minutes} min")

    print(f"\nOverall severity: {monitor.get_overall_severity()}")
