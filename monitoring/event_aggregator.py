"""
Event Aggregator
================

Combines events from all monitoring sources into unified event objects.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    EventType,
    OperatingMode,
    MODE_THRESHOLDS,
    EVENT_IMPACT_MATRIX,
    POSTCODE_COORDINATES,
    get_logger
)

from .weather_monitor import WeatherMonitor, WeatherCondition, WeatherAlert
from .traffic_monitor import TrafficMonitor, TrafficIncident
from .news_monitor import NewsMonitor, NewsItem

logger = get_logger(__name__)


@dataclass
class Event:
    """
    Unified event object combining all event sources.

    Represents any event that may affect scheduling:
    weather, traffic, news, manual alerts, etc.
    """
    event_id: str
    event_types: List[EventType]
    source: str  # 'weather', 'traffic', 'news', 'manual'
    title: str
    description: str
    severity: float  # 0-1 scale
    sentiment: float  # -1 to 1 scale
    created_at: datetime
    expires_at: Optional[datetime]
    location: str
    affected_postcodes: List[str] = field(default_factory=list)
    affected_sites: List[str] = field(default_factory=list)
    noshow_adjustment: float = 0.0  # Add to no-show probability
    duration_adjustment: int = 0  # Minutes to add to expected duration
    raw_data: Dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if event is still active"""
        if self.expires_at is None:
            return True
        return datetime.now() < self.expires_at

    @property
    def severity_level(self) -> str:
        """Get severity level as string"""
        if self.severity >= 0.8:
            return 'critical'
        elif self.severity >= 0.5:
            return 'high'
        elif self.severity >= 0.3:
            return 'medium'
        else:
            return 'low'

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'event_types': [et.value for et in self.event_types],
            'source': self.source,
            'title': self.title,
            'description': self.description,
            'severity': self.severity,
            'sentiment': self.sentiment,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'location': self.location,
            'affected_postcodes': self.affected_postcodes,
            'affected_sites': self.affected_sites,
            'noshow_adjustment': self.noshow_adjustment,
            'duration_adjustment': self.duration_adjustment,
            'severity_level': self.severity_level,
            'is_active': self.is_active
        }


class EventAggregator:
    """
    Aggregates events from all monitoring sources.

    Combines weather, traffic, and news events into unified
    Event objects with calculated impacts on scheduling.
    """

    def __init__(self):
        """Initialize the event aggregator"""
        self.weather_monitor = WeatherMonitor()
        self.traffic_monitor = TrafficMonitor()
        self.news_monitor = NewsMonitor()
        self._manual_events: List[Event] = []
        self._event_cache: Dict[str, Event] = {}

    def _generate_event_id(self, source: str, identifier: str) -> str:
        """Generate unique event ID"""
        hash_input = f"{source}-{identifier}-{datetime.now().date()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _classify_news_event(self, news_item: NewsItem) -> List[EventType]:
        """Classify news item into event types"""
        types = []
        text = f"{news_item.title} {news_item.description}".lower()

        # Import keyword mappings from config
        from config import EVENT_KEYWORDS

        for event_type, keywords in EVENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    if event_type not in types:
                        types.append(event_type)
                    break

        return types if types else [EventType.OTHER]

    def _calculate_impact(self, event_types: List[EventType],
                          severity: float) -> tuple:
        """
        Calculate no-show and duration adjustments.

        Returns:
            Tuple of (noshow_adjustment, duration_adjustment)
        """
        max_noshow = 0.0
        max_duration = 0

        severity_level = 'low'
        if severity >= 0.8:
            severity_level = 'critical'
        elif severity >= 0.5:
            severity_level = 'high'
        elif severity >= 0.3:
            severity_level = 'medium'

        for event_type in event_types:
            key = (event_type, severity_level)
            if key in EVENT_IMPACT_MATRIX:
                noshow, duration = EVENT_IMPACT_MATRIX[key]
                max_noshow = max(max_noshow, noshow)
                max_duration = max(max_duration, duration)

        return max_noshow, max_duration

    def _get_affected_sites(self, postcodes: List[str]) -> List[str]:
        """Determine which sites are affected by postcodes"""
        # Simple mapping - in production would use geographic analysis
        site_postcode_map = {
            'WC': ['CF10', 'CF11', 'CF14', 'CF15', 'CF23', 'CF24', 'CF3', 'CF5'],
            'NP': ['NP10', 'NP19', 'NP20', 'NP44'],
            'SW': ['SA1', 'SA2', 'SA3', 'SA4']
        }

        affected_sites = []
        for site, site_postcodes in site_postcode_map.items():
            if any(pc in postcodes for pc in site_postcodes):
                affected_sites.append(site)

        return affected_sites

    def _convert_weather_event(self, condition: WeatherCondition) -> Optional[Event]:
        """Convert weather condition to Event"""
        if condition.severity < 0.2:
            return None

        event_id = self._generate_event_id('weather', str(condition.timestamp))

        noshow_adj, duration_adj = self._calculate_impact(
            [EventType.WEATHER_EVENT],
            condition.severity
        )

        return Event(
            event_id=event_id,
            event_types=[EventType.WEATHER_EVENT],
            source='weather',
            title=f"Weather: {condition.weather_description}",
            description=f"Current conditions: {condition.weather_description}, "
                        f"Temperature: {condition.temperature}°C, "
                        f"Wind: {condition.wind_speed} km/h",
            severity=condition.severity,
            sentiment=-condition.severity,  # Bad weather = negative sentiment
            created_at=condition.timestamp,
            expires_at=condition.timestamp + timedelta(hours=3),
            location="South Wales",
            affected_postcodes=list(POSTCODE_COORDINATES.keys()),
            affected_sites=['WC', 'NP', 'SW'],
            noshow_adjustment=noshow_adj,
            duration_adjustment=duration_adj,
            raw_data={'weather_code': condition.weather_code}
        )

    def _convert_traffic_incident(self, incident: TrafficIncident) -> Event:
        """Convert traffic incident to Event"""
        event_id = self._generate_event_id('traffic', incident.incident_id)

        noshow_adj, duration_adj = self._calculate_impact(
            [EventType.TRAFFIC_INCIDENT],
            incident.severity
        )

        affected_sites = self._get_affected_sites(incident.affected_postcodes)

        return Event(
            event_id=event_id,
            event_types=[EventType.TRAFFIC_INCIDENT],
            source='traffic',
            title=f"Traffic: {incident.incident_type} on {incident.road}",
            description=incident.description,
            severity=incident.severity,
            sentiment=-incident.severity,
            created_at=incident.start_time,
            expires_at=incident.end_time,
            location=incident.location,
            affected_postcodes=incident.affected_postcodes,
            affected_sites=affected_sites,
            noshow_adjustment=noshow_adj,
            duration_adjustment=max(duration_adj, incident.delay_minutes),
            raw_data={
                'incident_type': incident.incident_type,
                'road': incident.road,
                'delay_minutes': incident.delay_minutes
            }
        )

    def _convert_news_item(self, item: NewsItem) -> Optional[Event]:
        """Convert news item to Event"""
        if item.relevance_score < 0.4:
            return None

        event_types = self._classify_news_event(item)
        event_id = self._generate_event_id('news', item.item_id)

        # Estimate severity from relevance score
        severity = min(1.0, item.relevance_score * 1.2)

        noshow_adj, duration_adj = self._calculate_impact(event_types, severity)

        # Estimate affected postcodes based on keywords
        affected_postcodes = []
        text_lower = f"{item.title} {item.description}".lower()
        if 'cardiff' in text_lower:
            affected_postcodes.extend(['CF10', 'CF11', 'CF14', 'CF15'])
        if 'newport' in text_lower:
            affected_postcodes.extend(['NP10', 'NP19', 'NP20'])
        if 'swansea' in text_lower:
            affected_postcodes.extend(['SA1', 'SA2'])
        if not affected_postcodes:
            affected_postcodes = list(POSTCODE_COORDINATES.keys())

        affected_sites = self._get_affected_sites(affected_postcodes)

        return Event(
            event_id=event_id,
            event_types=event_types,
            source='news',
            title=item.title,
            description=item.description[:300],
            severity=severity,
            sentiment=-0.5 if severity > 0.5 else -0.2,  # Estimate
            created_at=item.published,
            expires_at=item.published + timedelta(hours=12),
            location=item.source,
            affected_postcodes=affected_postcodes,
            affected_sites=affected_sites,
            noshow_adjustment=noshow_adj,
            duration_adjustment=duration_adj,
            raw_data={
                'keywords': item.keywords_found,
                'link': item.link,
                'source': item.source
            }
        )

    def add_manual_event(self, event: Event) -> None:
        """Add a manually created event"""
        self._manual_events.append(event)
        logger.info(f"Added manual event: {event.title}")

    def remove_event(self, event_id: str) -> bool:
        """Remove an event by ID"""
        for i, event in enumerate(self._manual_events):
            if event.event_id == event_id:
                del self._manual_events[i]
                logger.info(f"Removed event: {event_id}")
                return True
        return False

    def get_all_events(self, include_expired: bool = False) -> List[Event]:
        """
        Get all events from all sources.

        Args:
            include_expired: Include expired events

        Returns:
            List of Event objects
        """
        events = []
        seen_ids: Set[str] = set()

        # Weather events
        try:
            weather = self.weather_monitor.get_current_conditions()
            if weather:
                event = self._convert_weather_event(weather)
                if event and event.event_id not in seen_ids:
                    events.append(event)
                    seen_ids.add(event.event_id)

            # Weather alerts
            for alert in self.weather_monitor.get_alerts():
                event_id = self._generate_event_id('weather_alert', str(alert.start_time))
                if event_id not in seen_ids:
                    noshow_adj, duration_adj = self._calculate_impact(
                        [EventType.WEATHER_EVENT], alert.severity
                    )
                    events.append(Event(
                        event_id=event_id,
                        event_types=[EventType.WEATHER_EVENT],
                        source='weather',
                        title=alert.alert_type,
                        description=alert.description,
                        severity=alert.severity,
                        sentiment=-alert.severity,
                        created_at=alert.start_time,
                        expires_at=alert.end_time,
                        location=alert.affected_area,
                        affected_postcodes=list(POSTCODE_COORDINATES.keys()),
                        affected_sites=['WC', 'NP', 'SW'],
                        noshow_adjustment=noshow_adj,
                        duration_adjustment=duration_adj
                    ))
                    seen_ids.add(event_id)
        except Exception as e:
            logger.error(f"Error getting weather events: {e}")

        # Traffic events
        try:
            for incident in self.traffic_monitor.get_incidents():
                event = self._convert_traffic_incident(incident)
                if event.event_id not in seen_ids:
                    events.append(event)
                    seen_ids.add(event.event_id)
        except Exception as e:
            logger.error(f"Error getting traffic events: {e}")

        # News events
        try:
            for item in self.news_monitor.get_relevant_items():
                event = self._convert_news_item(item)
                if event and event.event_id not in seen_ids:
                    events.append(event)
                    seen_ids.add(event.event_id)
        except Exception as e:
            logger.error(f"Error getting news events: {e}")

        # Manual events
        events.extend(self._manual_events)

        # Filter expired
        if not include_expired:
            events = [e for e in events if e.is_active]

        # Sort by severity (highest first)
        events.sort(key=lambda x: x.severity, reverse=True)

        return events

    def get_active_events(self) -> List[Event]:
        """Get only active (non-expired) events"""
        return self.get_all_events(include_expired=False)

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get events of a specific type"""
        return [
            e for e in self.get_active_events()
            if event_type in e.event_types
        ]

    def get_events_for_postcode(self, postcode: str) -> List[Event]:
        """Get events affecting a specific postcode"""
        return [
            e for e in self.get_active_events()
            if postcode in e.affected_postcodes
        ]

    def get_events_for_site(self, site_code: str) -> List[Event]:
        """Get events affecting a specific site"""
        return [
            e for e in self.get_active_events()
            if site_code in e.affected_sites
        ]

    def get_aggregate_severity(self) -> float:
        """
        Get aggregate severity from all active events.

        Returns:
            Maximum severity among all active events
        """
        events = self.get_active_events()
        if not events:
            return 0.0
        return max(e.severity for e in events)

    def get_aggregate_impact(self) -> Dict[str, float]:
        """
        Get aggregate impact from all active events.

        Returns:
            Dict with noshow_adjustment and duration_adjustment
        """
        events = self.get_active_events()

        if not events:
            return {'noshow_adjustment': 0.0, 'duration_adjustment': 0}

        # Use max adjustments (events don't compound additively)
        return {
            'noshow_adjustment': max(e.noshow_adjustment for e in events),
            'duration_adjustment': max(e.duration_adjustment for e in events)
        }

    def determine_operating_mode(self) -> OperatingMode:
        """
        Determine the appropriate operating mode.

        Returns:
            OperatingMode based on current events
        """
        events = self.get_active_events()

        if not events:
            return OperatingMode.NORMAL

        max_severity = max(e.severity for e in events)

        # Check for emergency-type events
        emergency_types = {EventType.EMERGENCY, EventType.HEALTH_ALERT}
        for event in events:
            if any(et in emergency_types for et in event.event_types):
                if event.severity >= 0.7:
                    return OperatingMode.EMERGENCY

        # Severity-based mode selection
        if max_severity >= MODE_THRESHOLDS['emergency']:
            return OperatingMode.EMERGENCY
        elif max_severity >= MODE_THRESHOLDS['crisis']:
            return OperatingMode.CRISIS
        elif max_severity >= MODE_THRESHOLDS['elevated']:
            return OperatingMode.ELEVATED
        else:
            return OperatingMode.NORMAL

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current event situation.

        Returns:
            Dict with event counts, severity, and mode
        """
        events = self.get_active_events()

        return {
            'total_events': len(events),
            'by_source': {
                'weather': len([e for e in events if e.source == 'weather']),
                'traffic': len([e for e in events if e.source == 'traffic']),
                'news': len([e for e in events if e.source == 'news']),
                'manual': len([e for e in events if e.source == 'manual'])
            },
            'max_severity': self.get_aggregate_severity(),
            'operating_mode': self.determine_operating_mode().value,
            'impact': self.get_aggregate_impact(),
            'events': [e.to_dict() for e in events[:10]]  # Top 10
        }


# Example usage
if __name__ == "__main__":
    aggregator = EventAggregator()

    print("Fetching events from all sources...")
    events = aggregator.get_active_events()
    print(f"Found {len(events)} active events")

    for event in events[:5]:
        print(f"\n[{event.source}] {event.title}")
        print(f"  Severity: {event.severity:.2f} ({event.severity_level})")
        print(f"  Types: {[et.value for et in event.event_types]}")
        print(f"  No-show adj: +{event.noshow_adjustment:.0%}")
        print(f"  Duration adj: +{event.duration_adjustment} min")

    summary = aggregator.get_summary()
    print(f"\nOperating mode: {summary['operating_mode']}")
    print(f"Aggregate severity: {summary['max_severity']:.2f}")
