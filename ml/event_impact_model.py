"""
Event Impact Model for Scheduling (4.3)

Models the causal effect of external events on patient no-shows:
- Local events (traffic disruptions, road closures)
- Regional events (weather emergencies, public health alerts)
- National events (public holidays, major news)
- Global events (pandemics, wars affecting mental health/travel)

Causal chain: Event -> Sentiment/Disruption -> No-Show

Uses sentiment analysis to quantify event severity and estimate
causal impact on appointment attendance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import re

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can affect no-shows."""
    WEATHER = "weather"
    TRAFFIC = "traffic"
    PUBLIC_HEALTH = "public_health"
    LOCAL_DISRUPTION = "local_disruption"
    PUBLIC_HOLIDAY = "public_holiday"
    MAJOR_NEWS = "major_news"
    EMERGENCY = "emergency"
    STRIKE = "strike"
    SPORTING_EVENT = "sporting_event"
    OTHER = "other"


class EventSeverity(Enum):
    """Severity levels for events."""
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    SEVERE = 5
    CRITICAL = 6


@dataclass
class Event:
    """Represents an external event that may affect appointments."""
    event_id: str
    event_type: EventType
    title: str
    description: str
    severity: EventSeverity
    start_time: datetime
    end_time: Optional[datetime]
    location: Optional[str]  # Geographic scope
    sentiment_score: float  # -1 (negative) to +1 (positive)
    relevance_score: float  # 0-1, how relevant to healthcare appointments
    source: str  # news, social media, official alert, etc.
    keywords: List[str] = field(default_factory=list)


@dataclass
class EventImpactPrediction:
    """Predicted impact of event(s) on no-show rate."""
    baseline_noshow_rate: float
    predicted_noshow_rate: float
    absolute_increase: float  # Percentage point increase
    relative_increase: float  # Relative % increase
    contributing_events: List[Dict[str, Any]]
    confidence_interval: Tuple[float, float]
    recommendations: List[str]


class SentimentAnalyzer:
    """
    Simple sentiment analyzer for event text.

    In production, this would use:
    - NLTK/spaCy for NLP
    - Pre-trained sentiment models (BERT, RoBERTa)
    - News APIs (NewsAPI, GDELT)
    """

    # Keywords indicating negative impact on attendance
    NEGATIVE_KEYWORDS = {
        'emergency': -0.8,
        'closure': -0.6,
        'cancelled': -0.5,
        'delayed': -0.4,
        'disruption': -0.5,
        'accident': -0.6,
        'incident': -0.4,
        'storm': -0.7,
        'flood': -0.8,
        'snow': -0.5,
        'strike': -0.7,
        'protest': -0.5,
        'road closed': -0.6,
        'traffic jam': -0.4,
        'power outage': -0.6,
        'evacuation': -0.9,
        'warning': -0.5,
        'alert': -0.4,
        'pandemic': -0.8,
        'outbreak': -0.7,
        'lockdown': -0.9,
        'war': -0.7,
        'conflict': -0.5,
        'crisis': -0.6,
    }

    # Keywords indicating positive/neutral context
    POSITIVE_KEYWORDS = {
        'clear': 0.3,
        'resolved': 0.4,
        'ended': 0.3,
        'improved': 0.4,
        'normal': 0.3,
        'reopened': 0.5,
        'celebration': 0.2,
        'festival': 0.1,
    }

    def analyze(self, text: str) -> Tuple[float, List[str]]:
        """
        Analyze text sentiment and extract keywords.

        Returns:
            Tuple of (sentiment_score, matched_keywords)
        """
        text_lower = text.lower()
        sentiment = 0.0
        matched = []

        # Check negative keywords
        for keyword, score in self.NEGATIVE_KEYWORDS.items():
            if keyword in text_lower:
                sentiment += score
                matched.append(keyword)

        # Check positive keywords
        for keyword, score in self.POSITIVE_KEYWORDS.items():
            if keyword in text_lower:
                sentiment += score
                matched.append(keyword)

        # Normalize to [-1, 1]
        sentiment = max(-1.0, min(1.0, sentiment))

        return sentiment, matched

    def estimate_relevance(self, text: str, event_type: EventType) -> float:
        """Estimate how relevant an event is to healthcare appointments."""
        text_lower = text.lower()

        # Healthcare-specific keywords
        healthcare_keywords = [
            'hospital', 'clinic', 'medical', 'health', 'patient',
            'appointment', 'treatment', 'cancer', 'nhs', 'velindre'
        ]

        # Transport keywords (affect ability to attend)
        transport_keywords = [
            'road', 'traffic', 'bus', 'train', 'transport', 'travel',
            'motorway', 'a48', 'm4', 'cardiff'
        ]

        relevance = 0.3  # Base relevance

        for kw in healthcare_keywords:
            if kw in text_lower:
                relevance += 0.15

        for kw in transport_keywords:
            if kw in text_lower:
                relevance += 0.1

        # Event type relevance
        type_relevance = {
            EventType.WEATHER: 0.8,
            EventType.TRAFFIC: 0.9,
            EventType.PUBLIC_HEALTH: 1.0,
            EventType.EMERGENCY: 0.9,
            EventType.STRIKE: 0.8,
            EventType.LOCAL_DISRUPTION: 0.7,
            EventType.PUBLIC_HOLIDAY: 0.4,
            EventType.MAJOR_NEWS: 0.3,
            EventType.SPORTING_EVENT: 0.5,
            EventType.OTHER: 0.3,
        }
        relevance *= type_relevance.get(event_type, 0.5)

        return min(1.0, relevance)


class EventImpactModel:
    """
    Model for estimating causal impact of events on no-shows.

    Causal structure:
        Event -> Disruption Level -> No-Show
        Event -> Patient Anxiety -> No-Show
        Event -> Transport Availability -> No-Show

    Uses historical data to estimate:
        P(No-Show | Event) vs P(No-Show | No Event)
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.is_fitted = False

        # Default impact coefficients (can be learned from data)
        # Impact per severity level on no-show probability
        self.severity_impact = {
            EventSeverity.MINIMAL: 0.01,
            EventSeverity.LOW: 0.02,
            EventSeverity.MODERATE: 0.05,
            EventSeverity.HIGH: 0.10,
            EventSeverity.SEVERE: 0.20,
            EventSeverity.CRITICAL: 0.35,
        }

        # Impact multipliers by event type
        self.type_multiplier = {
            EventType.WEATHER: 1.2,
            EventType.TRAFFIC: 1.0,
            EventType.PUBLIC_HEALTH: 1.5,
            EventType.EMERGENCY: 1.8,
            EventType.STRIKE: 1.3,
            EventType.LOCAL_DISRUPTION: 0.8,
            EventType.PUBLIC_HOLIDAY: 0.5,
            EventType.MAJOR_NEWS: 0.4,
            EventType.SPORTING_EVENT: 0.6,
            EventType.OTHER: 0.5,
        }

        # Baseline no-show rate
        self.baseline_noshow_rate = 0.12

        # Learned coefficients
        self.event_coefficients: Dict[str, float] = {}

        logger.info("EventImpactModel initialized")

    def fit(self, historical_data: pd.DataFrame, event_log: Optional[pd.DataFrame] = None) -> 'EventImpactModel':
        """
        Fit model on historical appointment data with event information.

        Parameters:
        -----------
        historical_data : pd.DataFrame
            Appointments with no-show outcomes
        event_log : pd.DataFrame, optional
            Historical events with dates and details
        """
        if len(historical_data) < 50:
            logger.warning("Insufficient data for event impact fitting")
            return self

        # Calculate baseline no-show rate
        if 'Attended_Status' in historical_data.columns:
            noshow = (historical_data['Attended_Status'] == 'No').mean()
        elif 'no_show' in historical_data.columns:
            noshow = historical_data['no_show'].mean()
        else:
            noshow = 0.12

        self.baseline_noshow_rate = noshow

        # If we have weather data, estimate weather impact
        if 'Weather_Severity' in historical_data.columns:
            self._fit_weather_impact(historical_data)

        # If we have event log, estimate event-specific impacts
        if event_log is not None and len(event_log) > 0:
            self._fit_event_impacts(historical_data, event_log)

        self.is_fitted = True
        logger.info(f"EventImpactModel fitted: baseline no-show={self.baseline_noshow_rate:.3f}")

        return self

    def _fit_weather_impact(self, data: pd.DataFrame):
        """Estimate impact of weather severity on no-shows."""
        df = data.copy()
        if 'no_show' not in df.columns:
            df['no_show'] = (df['Attended_Status'] == 'No').astype(int)

        # Bin weather severity
        df['weather_bin'] = pd.cut(
            df['Weather_Severity'],
            bins=[-0.01, 0.1, 0.2, 0.3, 1.0],
            labels=['clear', 'light', 'moderate', 'severe']
        )

        # Calculate no-show rate by weather
        weather_impact = df.groupby('weather_bin')['no_show'].mean()

        # Store coefficients
        baseline = weather_impact.get('clear', self.baseline_noshow_rate)
        for level, rate in weather_impact.items():
            impact = rate - baseline
            self.event_coefficients[f'weather_{level}'] = max(0, impact)

        logger.info(f"Weather impact coefficients: {self.event_coefficients}")

    def _fit_event_impacts(self, appointments: pd.DataFrame, events: pd.DataFrame):
        """Estimate impact of specific events on no-shows."""
        # This would match events to appointment dates and estimate impacts
        # For now, use default coefficients
        pass

    def create_event(
        self,
        title: str,
        description: str,
        event_type: Union[EventType, str],
        severity: Optional[Union[EventSeverity, int]] = None,
        start_time: Optional[datetime] = None
    ) -> Event:
        """
        Create an event from text description.

        Automatically estimates sentiment and severity if not provided.

        Args:
            title: Event title
            description: Event description
            event_type: EventType enum or string (e.g., 'WEATHER', 'TRAFFIC')
            severity: EventSeverity enum or int (1-6) or None for auto-detection
            start_time: Optional start time (defaults to now)
        """
        # Convert string event_type to enum
        if isinstance(event_type, str):
            try:
                event_type = EventType[event_type.upper()]
            except KeyError:
                event_type = EventType.OTHER

        # Convert int severity to enum
        if isinstance(severity, int):
            severity_map = {
                1: EventSeverity.MINIMAL,
                2: EventSeverity.LOW,
                3: EventSeverity.MODERATE,
                4: EventSeverity.HIGH,
                5: EventSeverity.SEVERE,
                6: EventSeverity.CRITICAL
            }
            severity = severity_map.get(severity, EventSeverity.MODERATE)

        # Analyze sentiment
        sentiment, keywords = self.sentiment_analyzer.analyze(f"{title} {description}")

        # Estimate severity from sentiment if not provided
        if severity is None:
            if sentiment <= -0.7:
                severity = EventSeverity.CRITICAL
            elif sentiment <= -0.5:
                severity = EventSeverity.SEVERE
            elif sentiment <= -0.3:
                severity = EventSeverity.HIGH
            elif sentiment <= -0.1:
                severity = EventSeverity.MODERATE
            elif sentiment < 0:
                severity = EventSeverity.LOW
            else:
                severity = EventSeverity.MINIMAL

        # Estimate relevance
        relevance = self.sentiment_analyzer.estimate_relevance(description, event_type)

        event = Event(
            event_id=f"EVT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            event_type=event_type,
            title=title,
            description=description,
            severity=severity,
            start_time=start_time or datetime.now(),
            end_time=None,
            location=None,
            sentiment_score=sentiment,
            relevance_score=relevance,
            source="manual",
            keywords=keywords
        )

        return event

    def predict_impact(
        self,
        events: List[Event],
        appointment_date: Optional[datetime] = None
    ) -> EventImpactPrediction:
        """
        Predict the impact of events on no-show rate.

        Parameters:
        -----------
        events : List[Event]
            Active events to consider
        appointment_date : datetime, optional
            Date of appointments (for time-based decay)

        Returns:
        --------
        EventImpactPrediction with predicted no-show increase
        """
        if not events:
            return EventImpactPrediction(
                baseline_noshow_rate=self.baseline_noshow_rate,
                predicted_noshow_rate=self.baseline_noshow_rate,
                absolute_increase=0.0,
                relative_increase=0.0,
                contributing_events=[],
                confidence_interval=(self.baseline_noshow_rate, self.baseline_noshow_rate),
                recommendations=[]
            )

        total_impact = 0.0
        contributing = []

        for event in events:
            # Base impact from severity
            base_impact = self.severity_impact.get(event.severity, 0.05)

            # Multiply by event type factor
            type_factor = self.type_multiplier.get(event.event_type, 1.0)

            # Multiply by relevance
            relevance_factor = event.relevance_score

            # Adjust for sentiment (more negative = higher impact)
            sentiment_factor = 1.0 + abs(min(0, event.sentiment_score))

            # Calculate event impact
            event_impact = base_impact * type_factor * relevance_factor * sentiment_factor

            # Time decay if appointment date provided
            if appointment_date and event.start_time:
                days_diff = (appointment_date - event.start_time).days
                if days_diff > 0:
                    # Impact decays over time
                    decay = np.exp(-0.1 * days_diff)
                    event_impact *= decay

            total_impact += event_impact

            contributing.append({
                'event_id': event.event_id,
                'title': event.title,
                'type': event.event_type.value,
                'severity': event.severity.value,
                'impact': round(event_impact, 4),
                'sentiment': round(event.sentiment_score, 2)
            })

        # Cap total impact at reasonable level
        total_impact = min(0.5, total_impact)

        predicted_rate = min(0.95, self.baseline_noshow_rate + total_impact)
        absolute_increase = predicted_rate - self.baseline_noshow_rate
        relative_increase = (absolute_increase / self.baseline_noshow_rate) * 100 if self.baseline_noshow_rate > 0 else 0

        # Confidence interval (wider with more uncertainty)
        ci_width = 0.02 + 0.01 * len(events)
        ci_lower = max(0, predicted_rate - ci_width)
        ci_upper = min(1, predicted_rate + ci_width)

        # Generate recommendations
        recommendations = self._generate_recommendations(events, total_impact)

        return EventImpactPrediction(
            baseline_noshow_rate=self.baseline_noshow_rate,
            predicted_noshow_rate=predicted_rate,
            absolute_increase=absolute_increase,
            relative_increase=relative_increase,
            contributing_events=contributing,
            confidence_interval=(ci_lower, ci_upper),
            recommendations=recommendations
        )

    def _generate_recommendations(self, events: List[Event], total_impact: float) -> List[str]:
        """Generate recommendations based on predicted impact."""
        recommendations = []

        if total_impact > 0.15:
            recommendations.append("Consider proactive phone calls to confirm attendance")
            recommendations.append("Prepare for potential rescheduling requests")

        if total_impact > 0.25:
            recommendations.append("Consider overbooking slots by 10-15%")
            recommendations.append("Alert clinical staff about potential disruption")

        if total_impact > 0.35:
            recommendations.append("Consider postponing non-urgent appointments")
            recommendations.append("Activate contingency scheduling protocol")

        # Event-specific recommendations
        for event in events:
            if event.event_type == EventType.WEATHER:
                recommendations.append("Monitor weather updates and communicate with patients")
            elif event.event_type == EventType.TRAFFIC:
                recommendations.append("Advise patients to allow extra travel time")
            elif event.event_type == EventType.PUBLIC_HEALTH:
                recommendations.append("Review infection control protocols")
            elif event.event_type == EventType.STRIKE:
                recommendations.append("Arrange alternative transport options")

        return list(set(recommendations))[:5]  # Unique, max 5

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary."""
        return {
            'model_type': 'Event Impact Model',
            'is_fitted': self.is_fitted,
            'baseline_noshow_rate': self.baseline_noshow_rate,
            'severity_levels': len(self.severity_impact),
            'event_types': len(self.type_multiplier),
            'learned_coefficients': len(self.event_coefficients),
            'coefficients': self.event_coefficients
        }


# Convenience functions
def analyze_event_impact(
    events: List[Dict[str, Any]],
    baseline_rate: float = 0.12
) -> EventImpactPrediction:
    """
    Analyze impact of events on no-shows.

    Parameters:
    -----------
    events : List[Dict]
        List of event dicts with 'title', 'description', 'type', 'severity'
    baseline_rate : float
        Baseline no-show rate

    Returns:
    --------
    EventImpactPrediction
    """
    model = EventImpactModel()
    model.baseline_noshow_rate = baseline_rate

    event_objects = []
    for e in events:
        event_type = EventType[e.get('type', 'OTHER').upper()]
        severity = EventSeverity(e.get('severity', 3))

        event = model.create_event(
            title=e.get('title', 'Unknown Event'),
            description=e.get('description', ''),
            event_type=event_type,
            severity=severity
        )
        event_objects.append(event)

    return model.predict_impact(event_objects)


def estimate_event_severity(text: str) -> Dict[str, Any]:
    """
    Estimate event severity from text description.

    Parameters:
    -----------
    text : str
        Event description text

    Returns:
    --------
    Dict with sentiment, severity, and keywords
    """
    analyzer = SentimentAnalyzer()
    sentiment, keywords = analyzer.analyze(text)

    # Map sentiment to severity
    if sentiment <= -0.7:
        severity = EventSeverity.CRITICAL
    elif sentiment <= -0.5:
        severity = EventSeverity.SEVERE
    elif sentiment <= -0.3:
        severity = EventSeverity.HIGH
    elif sentiment <= -0.1:
        severity = EventSeverity.MODERATE
    elif sentiment < 0:
        severity = EventSeverity.LOW
    else:
        severity = EventSeverity.MINIMAL

    return {
        'sentiment': sentiment,
        'severity': severity.name,
        'severity_value': severity.value,
        'keywords': keywords,
        'is_negative': sentiment < 0
    }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Event Impact Model (4.3)")
    print("=" * 60)

    model = EventImpactModel()

    # Create test events
    events = [
        model.create_event(
            title="Major Traffic Disruption",
            description="M4 motorway closed due to accident near Cardiff. Severe delays expected.",
            event_type=EventType.TRAFFIC,
        ),
        model.create_event(
            title="Storm Warning",
            description="Met Office yellow warning for heavy rain and strong winds in South Wales.",
            event_type=EventType.WEATHER,
        )
    ]

    print("\nActive Events:")
    for e in events:
        print(f"  - {e.title} (Severity: {e.severity.name}, Sentiment: {e.sentiment_score:.2f})")

    # Predict impact
    prediction = model.predict_impact(events)

    print(f"\nImpact Prediction:")
    print(f"  Baseline no-show rate: {prediction.baseline_noshow_rate:.1%}")
    print(f"  Predicted no-show rate: {prediction.predicted_noshow_rate:.1%}")
    print(f"  Absolute increase: {prediction.absolute_increase:.1%}")
    print(f"  Relative increase: {prediction.relative_increase:.1f}%")

    print(f"\nRecommendations:")
    for rec in prediction.recommendations:
        print(f"  - {rec}")
