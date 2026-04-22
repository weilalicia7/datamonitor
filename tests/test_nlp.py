"""
Tests for the NLP module.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.sentiment_analyzer import SentimentAnalyzer
from nlp.event_classifier import EventClassifier, EventCategory, ImpactLevel
from nlp.entity_extractor import EntityExtractor
from nlp.severity_scorer import SeverityScorer


class TestSentimentAnalyzer(unittest.TestCase):
    """Tests for SentimentAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer)

    def test_negative_sentiment(self):
        """Test detection of negative sentiment"""
        text = "Road closed due to serious accident. Major delays expected."
        result = self.analyzer.analyze(text)

        self.assertLess(result.compound, 0)
        self.assertGreater(result.urgency_score, 0.3)

    def test_positive_sentiment(self):
        """Test detection of positive sentiment"""
        text = "Road reopened. Traffic returning to normal. All clear."
        result = self.analyzer.analyze(text)

        self.assertGreater(result.compound, -0.3)

    def test_urgency_detection(self):
        """Test urgency keyword detection"""
        urgent_text = "Emergency services responding to critical incident."
        result = self.analyzer.analyze(urgent_text)

        self.assertGreater(result.urgency_score, 0.5)
        self.assertIn(result.urgency_label, ['critical', 'high'])


class TestEventClassifier(unittest.TestCase):
    """Tests for EventClassifier class"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = EventClassifier()

    def test_weather_classification(self):
        """Test weather event classification"""
        text = "Heavy snow forecast across South Wales. Weather warning issued."
        result = self.classifier.classify(text)

        self.assertEqual(result.primary_category, EventCategory.WEATHER)

    def test_traffic_classification(self):
        """Test traffic event classification"""
        text = "M4 closed westbound due to accident. Delays of 45 minutes."
        result = self.classifier.classify(text)

        self.assertEqual(result.primary_category, EventCategory.TRAFFIC)

    def test_emergency_classification(self):
        """Test emergency event classification"""
        text = "Major incident declared. Evacuation in progress. Emergency services on scene."
        result = self.classifier.classify(text)

        self.assertEqual(result.primary_category, EventCategory.EMERGENCY)
        self.assertEqual(result.impact_level, ImpactLevel.CRITICAL)

    def test_impact_calculation(self):
        """Test impact level calculation"""
        severe_text = "Critical road closure. All traffic stopped."
        result = self.classifier.classify(severe_text)

        self.assertGreater(result.noshow_impact, 0.05)
        self.assertGreater(result.duration_impact, 0)


class TestEntityExtractor(unittest.TestCase):
    """Tests for EntityExtractor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.extractor = EntityExtractor()

    def test_road_extraction(self):
        """Test road name extraction"""
        text = "Delays on M4 and A470 due to roadworks."
        entities = self.extractor.extract(text)

        self.assertIn('M4', entities.roads)
        self.assertIn('A470', entities.roads)

    def test_location_extraction(self):
        """Test location extraction"""
        text = "Incident in Cardiff city centre affecting Whitchurch road."
        entities = self.extractor.extract(text)

        self.assertIn('Cardiff', entities.locations)
        self.assertIn('Whitchurch', entities.locations)

    def test_postcode_extraction(self):
        """Test postcode extraction"""
        text = "Flooding reported in CF14 and NP20 areas."
        entities = self.extractor.extract(text)

        self.assertIn('CF14', entities.postcodes)
        self.assertIn('NP20', entities.postcodes)

    def test_time_extraction(self):
        """Test time extraction"""
        text = "Road expected to reopen at 18:00. Delays until 15:30."
        entities = self.extractor.extract(text)

        self.assertGreater(len(entities.times), 0)


class TestSeverityScorer(unittest.TestCase):
    """Tests for SeverityScorer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.scorer = SeverityScorer()

    def test_low_severity(self):
        """Test low severity scoring"""
        text = "Minor delays expected on local roads."
        result = self.scorer.score(text)

        self.assertLess(result.overall_severity, 0.4)
        self.assertEqual(result.severity_level, 'low')

    def test_high_severity(self):
        """Test high severity scoring"""
        text = "M4 completely closed. Major incident. Emergency services attending. Do not travel."
        result = self.scorer.score(text)

        self.assertGreater(result.overall_severity, 0.5)
        self.assertIn(result.severity_level, ['high', 'critical'])

    def test_mode_suggestion(self):
        """Test operating mode suggestion"""
        critical_text = "Multiple critical incidents. Emergency evacuation. All roads closed."
        result = self.scorer.score(critical_text)

        self.assertIn(result.operating_mode_suggestion, ['crisis', 'emergency'])


class TestIntegration(unittest.TestCase):
    """Integration tests for NLP pipeline"""

    def test_full_pipeline(self):
        """Test complete NLP analysis pipeline"""
        scorer = SeverityScorer()

        events = [
            "M4 closed due to serious accident near Cardiff.",
            "Heavy snow expected overnight. Travel disruption likely.",
            "Train services suspended between Cardiff and Newport."
        ]

        results = scorer.score_batch(events)
        aggregate = scorer.get_aggregate_severity(results)

        self.assertEqual(len(results), 3)
        self.assertIn('max_severity', aggregate)
        self.assertIn('mode_suggestion', aggregate)


if __name__ == '__main__':
    unittest.main()
