"""
Severity Scorer
===============

Combines multiple NLP components to produce a final severity score.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    SEVERITY_WEIGHTS,
    get_logger
)

from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .event_classifier import EventClassifier, ClassificationResult, ImpactLevel
from .entity_extractor import EntityExtractor, ExtractedEntities

logger = get_logger(__name__)


@dataclass
class SeverityScore:
    """Complete severity assessment"""
    text: str
    overall_severity: float  # 0-1 scale
    severity_level: str  # none, low, medium, high, critical
    confidence: float

    # Component scores
    sentiment_score: float
    classification_score: float
    entity_score: float

    # Impact predictions
    noshow_adjustment: float  # Expected increase in no-show rate
    duration_adjustment: int  # Expected delay in minutes
    affected_postcodes: List[str]

    # Detailed results
    sentiment: SentimentResult
    classification: ClassificationResult
    entities: ExtractedEntities

    # Recommendations
    recommended_action: str
    operating_mode_suggestion: str


class SeverityScorer:
    """
    Combines sentiment, classification, and entity extraction
    to produce a comprehensive severity score.

    Uses weighted combination of multiple signals to determine
    overall impact on scheduling.
    """

    # Severity level thresholds
    SEVERITY_THRESHOLDS = {
        'critical': 0.8,
        'high': 0.6,
        'medium': 0.4,
        'low': 0.2,
        'none': 0.0
    }

    # Operating mode recommendations
    MODE_THRESHOLDS = {
        'emergency': 0.8,
        'crisis': 0.6,
        'elevated': 0.35,
        'normal': 0.0
    }

    def __init__(self):
        """Initialize severity scorer with component analyzers"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.event_classifier = EventClassifier()
        self.entity_extractor = EntityExtractor()

        logger.info("Severity scorer initialized")

    def score(self, text: str, context: Dict = None) -> SeverityScore:
        """
        Calculate comprehensive severity score for text.

        Args:
            text: Event description text
            context: Optional context dict with additional signals

        Returns:
            SeverityScore object
        """
        context = context or {}

        # Run component analyses
        sentiment = self.sentiment_analyzer.analyze(text)
        classification = self.event_classifier.classify(text)
        entities = self.entity_extractor.extract(text)

        # Calculate component scores
        sentiment_score = self._calculate_sentiment_score(sentiment)
        classification_score = self._calculate_classification_score(classification)
        entity_score = self._calculate_entity_score(entities)

        # Weighted combination
        weights = SEVERITY_WEIGHTS
        overall = (
            sentiment_score * weights.get('sentiment', 0.3) +
            classification_score * weights.get('classification', 0.5) +
            entity_score * weights.get('entity', 0.2)
        )

        # Apply context modifiers
        overall = self._apply_context_modifiers(overall, context)

        # Clamp to 0-1
        overall = max(0.0, min(1.0, overall))

        # Determine severity level
        severity_level = self._get_severity_level(overall)

        # Calculate confidence
        confidence = self._calculate_confidence(
            sentiment, classification, entities
        )

        # Calculate impact predictions
        noshow_adj = self._calculate_noshow_adjustment(
            classification, entities, overall
        )
        duration_adj = self._calculate_duration_adjustment(
            classification, entities, overall
        )

        # Get affected postcodes
        affected_postcodes = self.entity_extractor.get_affected_postcodes(entities)

        # Generate recommendations
        recommended_action = self._get_recommended_action(
            severity_level, classification, affected_postcodes
        )
        mode_suggestion = self._get_mode_suggestion(overall)

        return SeverityScore(
            text=text,
            overall_severity=round(overall, 3),
            severity_level=severity_level,
            confidence=round(confidence, 2),
            sentiment_score=round(sentiment_score, 3),
            classification_score=round(classification_score, 3),
            entity_score=round(entity_score, 3),
            noshow_adjustment=round(noshow_adj, 3),
            duration_adjustment=duration_adj,
            affected_postcodes=affected_postcodes,
            sentiment=sentiment,
            classification=classification,
            entities=entities,
            recommended_action=recommended_action,
            operating_mode_suggestion=mode_suggestion
        )

    def _calculate_sentiment_score(self, sentiment: SentimentResult) -> float:
        """Convert sentiment to severity score"""
        # Negative sentiment and urgency contribute to severity
        negative_component = max(0, -sentiment.compound) * 0.6
        urgency_component = sentiment.urgency_score * 0.4

        return negative_component + urgency_component

    def _calculate_classification_score(self, classification: ClassificationResult) -> float:
        """Convert classification to severity score"""
        # Impact level is primary driver
        impact_score = classification.impact_level.value / 4.0  # Normalize to 0-1

        # Adjust for confidence
        adjusted = impact_score * (0.5 + classification.confidence * 0.5)

        return adjusted

    def _calculate_entity_score(self, entities: ExtractedEntities) -> float:
        """Calculate severity contribution from entities"""
        score = 0.0

        # Multiple locations increase severity
        if len(entities.locations) > 2:
            score += 0.2
        elif len(entities.locations) > 0:
            score += 0.1

        # Multiple roads affected
        if len(entities.roads) > 1:
            score += 0.2
        elif len(entities.roads) > 0:
            score += 0.1

        # Postcodes affected
        if len(entities.postcodes) > 3:
            score += 0.2
        elif len(entities.postcodes) > 0:
            score += 0.1

        # Duration indicators
        duration = self.entity_extractor.estimate_duration_minutes(entities)
        if duration:
            if duration > 120:
                score += 0.3
            elif duration > 60:
                score += 0.2
            elif duration > 30:
                score += 0.1

        return min(1.0, score)

    def _apply_context_modifiers(self, score: float, context: Dict) -> float:
        """Apply context-specific modifiers to score"""
        # Time of day modifier
        if context.get('is_peak_time'):
            score *= 1.2

        # Existing events modifier
        existing_events = context.get('existing_event_count', 0)
        if existing_events > 2:
            score *= 1.1

        # Weather condition modifier
        if context.get('bad_weather'):
            score *= 1.15

        # External severity hint
        if 'external_severity' in context:
            external = context['external_severity']
            score = score * 0.7 + external * 0.3

        return score

    def _get_severity_level(self, score: float) -> str:
        """Get severity level label from score"""
        for level, threshold in self.SEVERITY_THRESHOLDS.items():
            if score >= threshold:
                return level
        return 'none'

    def _calculate_confidence(self, sentiment: SentimentResult,
                             classification: ClassificationResult,
                             entities: ExtractedEntities) -> float:
        """Calculate overall confidence in assessment"""
        # Start with classification confidence
        confidence = classification.confidence

        # Boost if sentiment and classification agree
        if sentiment.urgency_score > 0.5 and classification.impact_level.value >= 3:
            confidence += 0.1

        # Boost if entities provide corroboration
        if entities.locations or entities.roads or entities.postcodes:
            confidence += 0.1

        # Reduce if text is very short
        if len(sentiment.text) < 50:
            confidence -= 0.1

        return max(0.1, min(0.95, confidence))

    def _calculate_noshow_adjustment(self, classification: ClassificationResult,
                                     entities: ExtractedEntities,
                                     overall: float) -> float:
        """Calculate expected no-show rate adjustment"""
        base = classification.noshow_impact

        # Scale by overall severity
        adjusted = base * (0.5 + overall * 0.5)

        # Boost if many postcodes affected
        affected = len(self.entity_extractor.get_affected_postcodes(entities))
        if affected > 5:
            adjusted *= 1.2

        return min(0.5, adjusted)  # Cap at 50% increase

    def _calculate_duration_adjustment(self, classification: ClassificationResult,
                                       entities: ExtractedEntities,
                                       overall: float) -> int:
        """Calculate expected duration adjustment in minutes"""
        base = classification.duration_impact

        # Use extracted duration if available
        extracted = self.entity_extractor.estimate_duration_minutes(entities)
        if extracted:
            base = max(base, extracted // 2)  # Use half of extracted duration

        # Scale by overall severity
        adjusted = int(base * (0.5 + overall * 0.5))

        return min(60, adjusted)  # Cap at 60 minutes

    def _get_recommended_action(self, severity_level: str,
                                classification: ClassificationResult,
                                affected_postcodes: List[str]) -> str:
        """Get recommended action based on assessment"""
        if severity_level == 'critical':
            return "Immediate review required. Consider activating emergency protocols. Contact affected patients."
        elif severity_level == 'high':
            return f"Review schedules for affected areas ({', '.join(affected_postcodes[:5])}). Send patient notifications. Add time buffers."
        elif severity_level == 'medium':
            return "Monitor situation. Consider adding 10-15 minute buffers for affected routes."
        elif severity_level == 'low':
            return "No immediate action required. Continue monitoring."
        else:
            return "No action required."

    def _get_mode_suggestion(self, overall: float) -> str:
        """Suggest operating mode based on severity"""
        for mode, threshold in self.MODE_THRESHOLDS.items():
            if overall >= threshold:
                return mode
        return 'normal'

    def score_batch(self, texts: List[str]) -> List[SeverityScore]:
        """
        Score multiple texts.

        Args:
            texts: List of event descriptions

        Returns:
            List of SeverityScore objects
        """
        return [self.score(text) for text in texts]

    def get_aggregate_severity(self, scores: List[SeverityScore]) -> Dict:
        """
        Get aggregate severity from multiple scores.

        Args:
            scores: List of SeverityScore objects

        Returns:
            Dict with aggregate metrics
        """
        if not scores:
            return {
                'max_severity': 0.0,
                'avg_severity': 0.0,
                'total_noshow_impact': 0.0,
                'max_duration_impact': 0,
                'mode_suggestion': 'normal',
                'event_count': 0
            }

        max_severity = max(s.overall_severity for s in scores)
        avg_severity = sum(s.overall_severity for s in scores) / len(scores)

        # Combine no-show impacts
        combined_noshow = 1.0
        for s in scores:
            combined_noshow *= (1 - s.noshow_adjustment)
        total_noshow = 1 - combined_noshow

        max_duration = max(s.duration_adjustment for s in scores)

        # All affected postcodes
        all_postcodes = set()
        for s in scores:
            all_postcodes.update(s.affected_postcodes)

        return {
            'max_severity': round(max_severity, 3),
            'avg_severity': round(avg_severity, 3),
            'total_noshow_impact': round(total_noshow, 3),
            'max_duration_impact': max_duration,
            'mode_suggestion': self._get_mode_suggestion(max_severity),
            'affected_postcodes': list(all_postcodes),
            'event_count': len(scores),
            'critical_count': sum(1 for s in scores if s.severity_level == 'critical'),
            'high_count': sum(1 for s in scores if s.severity_level == 'high')
        }


# Example usage
if __name__ == "__main__":
    scorer = SeverityScorer()

    # Test events
    test_events = [
        "M4 closed in both directions due to serious accident near Cardiff. Emergency services on scene. Expect severe delays for several hours.",
        "Weather warning: Heavy snow expected across South Wales tonight. Transport disruption likely.",
        "Minor delays on A470 due to roadworks near Whitchurch. Allow extra 10 minutes.",
        "Train services between Cardiff and Newport suspended until further notice due to signalling failure.",
        "Major incident declared in Cardiff city centre. Avoid the area. Road closures in place.",
    ]

    print("Severity Scoring Results:")
    print("=" * 70)

    scores = []
    for event in test_events:
        result = scorer.score(event)
        scores.append(result)

        print(f"\nEvent: {event[:60]}...")
        print(f"  Overall Severity: {result.overall_severity} ({result.severity_level})")
        print(f"  Confidence: {result.confidence}")
        print(f"  No-show Adjustment: +{result.noshow_adjustment:.1%}")
        print(f"  Duration Adjustment: +{result.duration_adjustment} min")
        print(f"  Affected Postcodes: {result.affected_postcodes[:5]}")
        print(f"  Mode Suggestion: {result.operating_mode_suggestion}")
        print(f"  Action: {result.recommended_action[:60]}...")

    # Aggregate analysis
    print("\n" + "=" * 70)
    aggregate = scorer.get_aggregate_severity(scores)
    print(f"\nAggregate Analysis ({aggregate['event_count']} events):")
    print(f"  Max Severity: {aggregate['max_severity']}")
    print(f"  Average Severity: {aggregate['avg_severity']}")
    print(f"  Combined No-show Impact: +{aggregate['total_noshow_impact']:.1%}")
    print(f"  Max Duration Impact: +{aggregate['max_duration_impact']} min")
    print(f"  Suggested Mode: {aggregate['mode_suggestion']}")
    print(f"  Critical Events: {aggregate['critical_count']}")
    print(f"  High Severity Events: {aggregate['high_count']}")
