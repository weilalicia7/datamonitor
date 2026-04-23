"""
Tests for ml/event_impact_model.py — Wave 3.2 (T3 coverage).

Covers:
- SentimentAnalyzer keyword detection + bounds
- EventImpactModel.create_event auto-detects severity & sentiment
- predict_impact with no events returns baseline
- predict_impact with severe events raises no-show rate
- EventImpactPrediction dataclass shape + recommendations
"""

from __future__ import annotations

import pytest

from ml.event_impact_model import (
    Event,
    EventImpactModel,
    EventImpactPrediction,
    EventSeverity,
    EventType,
    SentimentAnalyzer,
    estimate_event_severity,
)


# --------------------------------------------------------------------------- #
# SentimentAnalyzer
# --------------------------------------------------------------------------- #


class TestSentimentAnalyzer:
    def test_negative_keywords_detected(self):
        analyzer = SentimentAnalyzer()
        sentiment, keywords = analyzer.analyze(
            "Emergency road closure after major accident on the motorway"
        )
        assert sentiment < 0
        # Should match at least one negative keyword.
        assert any(k in {"emergency", "closure", "accident"} for k in keywords)
        assert -1.0 <= sentiment <= 1.0

    def test_relevance_in_bounds(self):
        analyzer = SentimentAnalyzer()
        r = analyzer.estimate_relevance(
            "Hospital appointment disruption for NHS patients", EventType.PUBLIC_HEALTH
        )
        assert 0.0 <= r <= 1.0


# --------------------------------------------------------------------------- #
# EventImpactModel
# --------------------------------------------------------------------------- #


class TestEventImpactModel:
    def test_create_event_autodetects_severity(self):
        model = EventImpactModel()
        ev = model.create_event(
            title="Severe Storm",
            description="Flood warning and road closures across region",
            event_type=EventType.WEATHER,
        )
        assert isinstance(ev, Event)
        assert ev.event_type == EventType.WEATHER
        assert ev.sentiment_score < 0
        # Severity must be a valid EventSeverity enum.
        assert isinstance(ev.severity, EventSeverity)

    def test_predict_impact_with_no_events(self):
        model = EventImpactModel()
        pred = model.predict_impact([])
        assert isinstance(pred, EventImpactPrediction)
        assert pred.absolute_increase == 0.0
        assert pred.predicted_noshow_rate == model.baseline_noshow_rate
        assert pred.recommendations == []
        assert pred.contributing_events == []

    def test_predict_impact_severe_event_increases_rate(self):
        model = EventImpactModel()
        ev = model.create_event(
            title="Pandemic outbreak",
            description="Serious outbreak with lockdown warning",
            event_type=EventType.PUBLIC_HEALTH,
            severity=EventSeverity.SEVERE,
        )
        pred = model.predict_impact([ev])
        # With a severe public-health event, predicted rate exceeds baseline.
        assert pred.predicted_noshow_rate > model.baseline_noshow_rate
        assert pred.absolute_increase > 0.0
        assert pred.contributing_events
        # Confidence interval should bracket predicted rate in reasonable range.
        lo, hi = pred.confidence_interval
        assert 0.0 <= lo <= pred.predicted_noshow_rate <= hi <= 1.0

    def test_estimate_event_severity_negative_text(self):
        out = estimate_event_severity("Major road closure emergency with storm warning")
        assert out["sentiment"] < 0
        assert out["is_negative"] is True
        assert "severity" in out
