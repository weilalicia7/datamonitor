"""
Sentiment Analyzer
==================

Analyzes sentiment and urgency in text using VADER and custom rules.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from pathlib import Path

# VADER sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    positive: float
    negative: float
    neutral: float
    compound: float
    urgency_score: float
    sentiment_label: str
    urgency_label: str


class SentimentAnalyzer:
    """
    Analyzes sentiment and urgency in text.

    Uses VADER for general sentiment and custom rules
    for urgency detection specific to scheduling context.
    """

    # Urgency indicators
    URGENCY_KEYWORDS = {
        'critical': 1.0,
        'emergency': 1.0,
        'urgent': 0.9,
        'immediate': 0.9,
        'severe': 0.8,
        'major': 0.7,
        'serious': 0.7,
        'warning': 0.6,
        'alert': 0.6,
        'significant': 0.5,
        'important': 0.4,
        'moderate': 0.3,
        'minor': 0.2,
        'slight': 0.1
    }

    # Negative impact phrases
    NEGATIVE_PHRASES = [
        'road closed', 'closure', 'cancelled', 'suspended',
        'delays expected', 'disruption', 'incident',
        'accident', 'flooding', 'evacuation', 'avoid area',
        'do not travel', 'danger', 'hazard', 'blocked'
    ]

    # Positive/resolution phrases
    POSITIVE_PHRASES = [
        'reopened', 'cleared', 'resolved', 'normal service',
        'back to normal', 'lifted', 'ended', 'improved',
        'easing', 'recovery', 'restored'
    ]

    def __init__(self):
        """Initialize sentiment analyzer"""
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        else:
            self.vader = None
            logger.warning("VADER not available, using rule-based analysis only")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment and urgency in text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult object
        """
        text_lower = text.lower()

        # Get VADER scores if available
        if self.vader:
            scores = self.vader.polarity_scores(text)
            positive = scores['pos']
            negative = scores['neg']
            neutral = scores['neu']
            compound = scores['compound']
        else:
            # Fallback rule-based sentiment
            positive, negative, neutral, compound = self._rule_based_sentiment(text_lower)

        # Calculate urgency score
        urgency_score = self._calculate_urgency(text_lower)

        # Adjust sentiment based on context
        compound, urgency_score = self._adjust_for_context(
            text_lower, compound, urgency_score
        )

        # Determine labels
        sentiment_label = self._get_sentiment_label(compound)
        urgency_label = self._get_urgency_label(urgency_score)

        return SentimentResult(
            text=text,
            positive=round(positive, 3),
            negative=round(negative, 3),
            neutral=round(neutral, 3),
            compound=round(compound, 3),
            urgency_score=round(urgency_score, 3),
            sentiment_label=sentiment_label,
            urgency_label=urgency_label
        )

    def _rule_based_sentiment(self, text: str) -> Tuple[float, float, float, float]:
        """
        Calculate sentiment using rules when VADER unavailable.

        Args:
            text: Lowercase text

        Returns:
            Tuple of (positive, negative, neutral, compound)
        """
        positive_count = sum(1 for phrase in self.POSITIVE_PHRASES if phrase in text)
        negative_count = sum(1 for phrase in self.NEGATIVE_PHRASES if phrase in text)

        total = positive_count + negative_count + 1  # +1 to avoid division by zero

        positive = positive_count / total * 0.5
        negative = negative_count / total * 0.5
        neutral = 1 - positive - negative

        # Compound score
        if negative_count > positive_count:
            compound = -0.5 * (negative_count - positive_count) / total
        elif positive_count > negative_count:
            compound = 0.5 * (positive_count - negative_count) / total
        else:
            compound = 0.0

        return positive, negative, neutral, compound

    def _calculate_urgency(self, text: str) -> float:
        """
        Calculate urgency score from text.

        Args:
            text: Lowercase text

        Returns:
            Urgency score (0-1)
        """
        max_urgency = 0.0
        urgency_count = 0

        for keyword, score in self.URGENCY_KEYWORDS.items():
            if keyword in text:
                max_urgency = max(max_urgency, score)
                urgency_count += 1

        # Boost urgency if multiple keywords found
        if urgency_count > 2:
            max_urgency = min(1.0, max_urgency + 0.1)

        # Check for time-sensitive indicators
        time_patterns = [
            r'\b(now|immediately|asap)\b',
            r'\b(today|tonight|this morning|this afternoon)\b',
            r'\b(next \d+ hours?|within \d+ hours?)\b'
        ]

        for pattern in time_patterns:
            if re.search(pattern, text):
                max_urgency = min(1.0, max_urgency + 0.1)
                break

        return max_urgency

    def _adjust_for_context(self, text: str, compound: float,
                           urgency: float) -> Tuple[float, float]:
        """
        Adjust scores based on scheduling context.

        Args:
            text: Lowercase text
            compound: Initial compound sentiment
            urgency: Initial urgency score

        Returns:
            Adjusted (compound, urgency) tuple
        """
        # Negative phrases should lower compound score
        for phrase in self.NEGATIVE_PHRASES:
            if phrase in text:
                compound = min(compound, compound - 0.2)
                urgency = max(urgency, 0.5)

        # Positive phrases should raise compound score
        for phrase in self.POSITIVE_PHRASES:
            if phrase in text:
                compound = max(compound, compound + 0.2)
                urgency = max(0, urgency - 0.2)

        # Clamp values
        compound = max(-1.0, min(1.0, compound))
        urgency = max(0.0, min(1.0, urgency))

        return compound, urgency

    def _get_sentiment_label(self, compound: float) -> str:
        """Get sentiment label from compound score"""
        if compound >= 0.5:
            return "very_positive"
        elif compound >= 0.1:
            return "positive"
        elif compound <= -0.5:
            return "very_negative"
        elif compound <= -0.1:
            return "negative"
        else:
            return "neutral"

    def _get_urgency_label(self, urgency: float) -> str:
        """Get urgency label from urgency score"""
        if urgency >= 0.8:
            return "critical"
        elif urgency >= 0.6:
            return "high"
        elif urgency >= 0.4:
            return "medium"
        elif urgency >= 0.2:
            return "low"
        else:
            return "none"

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects
        """
        return [self.analyze(text) for text in texts]

    def get_overall_sentiment(self, texts: List[str]) -> Dict:
        """
        Get aggregate sentiment across multiple texts.

        Args:
            texts: List of texts

        Returns:
            Dict with aggregate scores
        """
        if not texts:
            return {
                'avg_compound': 0.0,
                'avg_urgency': 0.0,
                'sentiment': 'neutral',
                'urgency': 'none'
            }

        results = self.analyze_batch(texts)

        avg_compound = sum(r.compound for r in results) / len(results)
        avg_urgency = sum(r.urgency_score for r in results) / len(results)
        max_urgency = max(r.urgency_score for r in results)

        return {
            'avg_compound': round(avg_compound, 3),
            'avg_urgency': round(avg_urgency, 3),
            'max_urgency': round(max_urgency, 3),
            'sentiment': self._get_sentiment_label(avg_compound),
            'urgency': self._get_urgency_label(max_urgency),
            'count': len(results)
        }


# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Test texts
    test_texts = [
        "M4 closed due to serious accident. Delays expected for several hours.",
        "Severe weather warning issued. Do not travel unless absolutely necessary.",
        "Road reopened following earlier incident. Traffic returning to normal.",
        "Minor delays on A470 due to roadworks. Allow extra time.",
        "Emergency services responding to major incident in Cardiff city centre."
    ]

    print("Sentiment Analysis Results:")
    print("=" * 60)

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text[:50]}...")
        print(f"  Sentiment: {result.sentiment_label} (compound: {result.compound})")
        print(f"  Urgency: {result.urgency_label} (score: {result.urgency_score})")

    # Aggregate analysis
    print("\n" + "=" * 60)
    overall = analyzer.get_overall_sentiment(test_texts)
    print(f"\nOverall Analysis:")
    print(f"  Average Compound: {overall['avg_compound']}")
    print(f"  Max Urgency: {overall['max_urgency']}")
    print(f"  Overall Sentiment: {overall['sentiment']}")
    print(f"  Overall Urgency: {overall['urgency']}")
