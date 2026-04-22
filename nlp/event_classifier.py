"""
Event Classifier
================

Classifies events into categories and determines scheduling impact.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    EVENT_CLASSIFICATION_KEYWORDS,
    get_logger
)

logger = get_logger(__name__)


class EventCategory(Enum):
    """Categories of events that affect scheduling"""
    WEATHER = "weather"
    TRAFFIC = "traffic"
    TRANSPORT = "transport"
    EMERGENCY = "emergency"
    PLANNED = "planned"
    SOCIAL = "social"
    HEALTHCARE = "healthcare"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class ImpactLevel(Enum):
    """Level of impact on scheduling"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ClassificationResult:
    """Result of event classification"""
    text: str
    primary_category: EventCategory
    secondary_categories: List[EventCategory]
    impact_level: ImpactLevel
    confidence: float
    keywords_matched: List[str]
    noshow_impact: float  # Expected increase in no-show rate
    duration_impact: int  # Expected delay in minutes
    affected_radius_km: float  # Geographic impact radius


class EventClassifier:
    """
    Classifies events and determines their impact on scheduling.

    Uses keyword matching and rule-based classification
    to categorize events and estimate their impact.
    """

    # Category keywords
    CATEGORY_KEYWORDS = {
        EventCategory.WEATHER: [
            'weather', 'rain', 'snow', 'ice', 'storm', 'wind', 'fog',
            'flood', 'thunder', 'lightning', 'temperature', 'cold', 'hot',
            'freeze', 'hail', 'warning', 'met office', 'yellow', 'amber', 'red'
        ],
        EventCategory.TRAFFIC: [
            'traffic', 'road', 'accident', 'crash', 'collision', 'closure',
            'm4', 'a470', 'a48', 'motorway', 'junction', 'roadworks',
            'congestion', 'delays', 'diversion', 'lane'
        ],
        EventCategory.TRANSPORT: [
            'train', 'bus', 'rail', 'transport', 'arriva', 'tfw', 'station',
            'cancelled', 'delayed', 'strike', 'suspension', 'service'
        ],
        EventCategory.EMERGENCY: [
            'emergency', 'fire', 'police', 'ambulance', 'incident',
            'evacuation', 'bomb', 'threat', 'attack', 'explosion',
            'hazmat', 'chemical', 'gas leak', 'major incident'
        ],
        EventCategory.PLANNED: [
            'planned', 'scheduled', 'maintenance', 'works', 'upgrade',
            'improvement', 'construction', 'project', 'starting', 'begins'
        ],
        EventCategory.SOCIAL: [
            'event', 'concert', 'match', 'rugby', 'football', 'stadium',
            'principality', 'cardiff city', 'parade', 'protest', 'march',
            'festival', 'celebration', 'crowd'
        ],
        EventCategory.HEALTHCARE: [
            'hospital', 'nhs', 'health', 'clinic', 'medical', 'patient',
            'appointment', 'treatment', 'surgery', 'ward', 'velindre',
            'health board', 'ambulance', 'a&e'
        ],
        EventCategory.INFRASTRUCTURE: [
            'power', 'electric', 'water', 'gas', 'utility', 'outage',
            'blackout', 'supply', 'network', 'internet', 'phone'
        ]
    }

    # Impact modifiers by category
    CATEGORY_IMPACT = {
        EventCategory.WEATHER: {'noshow': 0.15, 'duration': 15, 'radius': 30},
        EventCategory.TRAFFIC: {'noshow': 0.10, 'duration': 20, 'radius': 15},
        EventCategory.TRANSPORT: {'noshow': 0.20, 'duration': 25, 'radius': 50},
        EventCategory.EMERGENCY: {'noshow': 0.30, 'duration': 30, 'radius': 10},
        EventCategory.PLANNED: {'noshow': 0.05, 'duration': 10, 'radius': 5},
        EventCategory.SOCIAL: {'noshow': 0.08, 'duration': 15, 'radius': 10},
        EventCategory.HEALTHCARE: {'noshow': 0.05, 'duration': 5, 'radius': 5},
        EventCategory.INFRASTRUCTURE: {'noshow': 0.15, 'duration': 20, 'radius': 20},
        EventCategory.OTHER: {'noshow': 0.05, 'duration': 5, 'radius': 10}
    }

    # Severity multipliers for impact calculation.
    # Multi-word emergency phrases are listed before single words so they
    # are matched first (Python dict iteration order preserves insertion
    # order, and we take the *max* across all matches anyway, so the
    # ordering is defensive rather than load-bearing).
    #
    # "major incident", "evacuation" and "incident declared" are NHS-
    # standard terminology for the highest emergency level and therefore
    # map to the CRITICAL band (multiplier ≥ 1.8).
    SEVERITY_KEYWORDS = {
        'major incident': 2.0,
        'mass casualty': 2.0,
        'incident declared': 1.9,
        'evacuation': 1.9,
        'critical': 2.0,
        'severe': 1.8,
        'major': 1.6,
        'serious': 1.5,
        'significant': 1.3,
        'moderate': 1.0,
        'minor': 0.6,
        'slight': 0.4
    }

    def __init__(self):
        """Initialize event classifier"""
        logger.info("Event classifier initialized")

    def classify(self, text: str, severity_hint: float = None) -> ClassificationResult:
        """
        Classify an event and determine its impact.

        Args:
            text: Event description text
            severity_hint: Optional severity score (0-1) from external source

        Returns:
            ClassificationResult object
        """
        text_lower = text.lower()

        # Find matching categories
        category_scores = {}
        all_keywords = []

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                category_scores[category] = len(matches)
                all_keywords.extend(matches)

        # Determine primary and secondary categories
        if not category_scores:
            primary = EventCategory.OTHER
            secondary = []
            confidence = 0.3
        else:
            sorted_categories = sorted(
                category_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            primary = sorted_categories[0][0]
            secondary = [cat for cat, _ in sorted_categories[1:3]]
            confidence = min(0.95, 0.5 + len(all_keywords) * 0.1)

        # Determine impact level
        impact_level, severity_multiplier = self._calculate_impact_level(
            text_lower, severity_hint
        )

        # Calculate specific impacts
        base_impact = self.CATEGORY_IMPACT[primary]
        noshow_impact = base_impact['noshow'] * severity_multiplier
        duration_impact = int(base_impact['duration'] * severity_multiplier)
        radius = base_impact['radius'] * severity_multiplier

        return ClassificationResult(
            text=text,
            primary_category=primary,
            secondary_categories=secondary,
            impact_level=impact_level,
            confidence=round(confidence, 2),
            keywords_matched=list(set(all_keywords)),
            noshow_impact=round(noshow_impact, 3),
            duration_impact=duration_impact,
            affected_radius_km=round(radius, 1)
        )

    def _calculate_impact_level(self, text: str,
                                severity_hint: float = None) -> Tuple[ImpactLevel, float]:
        """
        Calculate impact level from text and hints.

        Args:
            text: Lowercase text
            severity_hint: Optional external severity score

        Returns:
            Tuple of (ImpactLevel, severity_multiplier)
        """
        # Find severity keywords
        max_multiplier = 1.0
        for keyword, multiplier in self.SEVERITY_KEYWORDS.items():
            if keyword in text:
                max_multiplier = max(max_multiplier, multiplier)

        # Use severity hint if provided
        if severity_hint is not None:
            hint_multiplier = 0.5 + severity_hint * 1.5
            max_multiplier = max(max_multiplier, hint_multiplier)

        # Check for closure/cancellation indicators
        if any(word in text for word in ['closed', 'cancelled', 'suspended', 'blocked']):
            max_multiplier = max(max_multiplier, 1.5)

        # Determine impact level
        if max_multiplier >= 1.8:
            level = ImpactLevel.CRITICAL
        elif max_multiplier >= 1.4:
            level = ImpactLevel.HIGH
        elif max_multiplier >= 1.0:
            level = ImpactLevel.MEDIUM
        elif max_multiplier >= 0.6:
            level = ImpactLevel.LOW
        else:
            level = ImpactLevel.NONE

        return level, max_multiplier

    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple events.

        Args:
            texts: List of event descriptions

        Returns:
            List of ClassificationResult objects
        """
        return [self.classify(text) for text in texts]

    def get_category_distribution(self, results: List[ClassificationResult]) -> Dict:
        """
        Get distribution of categories from classification results.

        Args:
            results: List of ClassificationResult objects

        Returns:
            Dict with category counts and percentages
        """
        if not results:
            return {}

        counts = {}
        for result in results:
            cat = result.primary_category.value
            counts[cat] = counts.get(cat, 0) + 1

        total = len(results)
        return {
            cat: {
                'count': count,
                'percentage': round(count / total * 100, 1)
            }
            for cat, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        }

    def get_aggregate_impact(self, results: List[ClassificationResult]) -> Dict:
        """
        Get aggregate impact from multiple events.

        Args:
            results: List of ClassificationResult objects

        Returns:
            Dict with aggregate impact metrics
        """
        if not results:
            return {
                'total_noshow_impact': 0.0,
                'max_duration_impact': 0,
                'max_impact_level': 'none',
                'event_count': 0
            }

        # Calculate combined no-show impact (not simply additive)
        # Use formula: combined = 1 - (1-p1)(1-p2)...
        combined_noshow = 1.0
        for result in results:
            combined_noshow *= (1 - result.noshow_impact)
        total_noshow = 1 - combined_noshow

        max_duration = max(r.duration_impact for r in results)
        max_level = max(r.impact_level.value for r in results)

        level_names = {0: 'none', 1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}

        return {
            'total_noshow_impact': round(total_noshow, 3),
            'max_duration_impact': max_duration,
            'avg_duration_impact': round(sum(r.duration_impact for r in results) / len(results)),
            'max_impact_level': level_names.get(max_level, 'unknown'),
            'event_count': len(results),
            'high_impact_count': sum(1 for r in results if r.impact_level.value >= 3)
        }

    def is_scheduling_relevant(self, result: ClassificationResult) -> bool:
        """
        Determine if an event is relevant to scheduling.

        Args:
            result: ClassificationResult object

        Returns:
            True if event should affect scheduling
        """
        # Relevant if impact level is at least LOW
        if result.impact_level.value >= ImpactLevel.LOW.value:
            return True

        # Or if it's a healthcare category
        if result.primary_category == EventCategory.HEALTHCARE:
            return True

        # Or high confidence with some impact
        if result.confidence >= 0.7 and result.noshow_impact >= 0.05:
            return True

        return False


# Example usage
if __name__ == "__main__":
    classifier = EventClassifier()

    # Test events
    test_events = [
        "Severe weather warning: Heavy snow expected across South Wales. Do not travel unless essential.",
        "M4 closed westbound between J32 and J33 due to serious accident. Major delays expected.",
        "Train services suspended between Cardiff and Newport due to engineering works.",
        "Major rugby match at Principality Stadium this Saturday. Expect road closures.",
        "Minor roadworks on A470 near Heath Hospital. Allow extra 5 minutes.",
        "Emergency services responding to major incident in city centre. Avoid area."
    ]

    print("Event Classification Results:")
    print("=" * 70)

    results = []
    for event in test_events:
        result = classifier.classify(event)
        results.append(result)

        print(f"\nEvent: {event[:60]}...")
        print(f"  Category: {result.primary_category.value}")
        print(f"  Impact Level: {result.impact_level.name}")
        print(f"  No-show Impact: +{result.noshow_impact:.1%}")
        print(f"  Duration Impact: +{result.duration_impact} min")
        print(f"  Confidence: {result.confidence}")
        print(f"  Keywords: {result.keywords_matched[:5]}")

    # Aggregate impact
    print("\n" + "=" * 70)
    aggregate = classifier.get_aggregate_impact(results)
    print(f"\nAggregate Impact ({aggregate['event_count']} events):")
    print(f"  Combined No-show Impact: +{aggregate['total_noshow_impact']:.1%}")
    print(f"  Max Duration Impact: +{aggregate['max_duration_impact']} min")
    print(f"  Max Impact Level: {aggregate['max_impact_level']}")
    print(f"  High Impact Events: {aggregate['high_impact_count']}")

    # Category distribution
    print("\nCategory Distribution:")
    for cat, data in classifier.get_category_distribution(results).items():
        print(f"  {cat}: {data['count']} ({data['percentage']}%)")
