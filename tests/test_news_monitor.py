"""
Tests for monitoring/news_monitor.py — production-readiness Wave 3.5.

feedparser.parse is mocked on every test so no real HTTP happens. The
cache file is redirected to a tmp_path to avoid contaminating production
data_cache/news_cache.json.

Structure:
- TestRelevanceScoring    — keyword tier -> (score, keywords_found)
- TestFetchFeedMocked     — happy path: parse stub -> NewsItem list
- TestFetchAllWithCache   — force=True bypasses valid cache
- TestFeedErrorHandling   — broken feed (exception) returns empty list
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.news_monitor import NewsItem, NewsMonitor  # noqa: E402


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point DATA_CACHE_DIR at tmp_path and reset feed list to a single entry."""
    monkeypatch.setattr("monitoring.news_monitor.DATA_CACHE_DIR", tmp_path)
    # A single deterministic feed avoids fan-out to BBC/Wales Online/Gov.uk.
    monkeypatch.setattr(
        "monitoring.news_monitor.NEWS_RSS_FEEDS",
        [{"name": "MockFeed", "url": "https://example.invalid/rss", "priority": 1}],
    )
    yield


def _fake_feed(entries: list[dict], bozo: bool = False):
    """Build a feedparser-like object with .entries and .bozo."""
    feed = types.SimpleNamespace()
    feed.entries = entries
    feed.bozo = bozo
    feed.bozo_exception = None if not bozo else Exception("bad feed")
    return feed


# --------------------------------------------------------------------------- #
# 1. Relevance scoring
# --------------------------------------------------------------------------- #


class TestRelevanceScoring:
    def test_high_keyword_scores_1(self):
        mon = NewsMonitor()
        score, kw = mon._calculate_relevance(
            "Major incident at NHS Wales hospital",
            "Emergency services respond to incident",
        )
        assert score == 1.0
        # Should pick up multiple tier-1 keywords
        assert any(k in kw for k in ("nhs wales", "hospital", "major incident"))

    def test_exclude_keyword_zero_score(self):
        mon = NewsMonitor()
        score, kw = mon._calculate_relevance(
            "Celebrity review podcast",
            "Entertainment news",
        )
        assert score == 0.0
        assert kw == []


# --------------------------------------------------------------------------- #
# 2. _fetch_feed (feedparser.parse stubbed)
# --------------------------------------------------------------------------- #


class TestFetchFeedMocked:
    def test_parses_entries_into_news_items(self):
        entries = [
            {
                "id": "e1",
                "title": "Flooding in Cardiff city centre",
                "description": "<p>Major flooding reported near NHS Wales sites</p>",
                "link": "https://example.invalid/article/1",
                "published": datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000"),
            }
        ]
        mon = NewsMonitor()
        with patch(
            "monitoring.news_monitor.feedparser.parse",
            return_value=_fake_feed(entries),
        ) as parse:
            items = mon._fetch_feed({"name": "MockFeed", "url": "https://x"})
        parse.assert_called_once()
        assert len(items) == 1
        it = items[0]
        assert isinstance(it, NewsItem)
        assert "flooding" in it.description.lower()
        # HTML tags stripped
        assert "<p>" not in it.description
        assert it.is_relevant is True
        assert it.relevance_score >= 0.7


# --------------------------------------------------------------------------- #
# 3. fetch_all end-to-end with mocked feedparser
# --------------------------------------------------------------------------- #


class TestFetchAllWithCache:
    def test_fetch_all_returns_sorted_items_and_writes_cache(self, tmp_path):
        now = datetime.now()
        entries = [
            {
                "id": "e-old",
                "title": "Accident on M4 near Newport",
                "description": "Traffic delays in Newport",
                "link": "https://example.invalid/m4",
                "published": (now - timedelta(hours=3))
                    .strftime("%a, %d %b %Y %H:%M:%S +0000"),
            },
            {
                "id": "e-new",
                "title": "NHS Wales issues weather warning",
                "description": "Storm expected across south wales",
                "link": "https://example.invalid/warn",
                "published": now.strftime("%a, %d %b %Y %H:%M:%S +0000"),
            },
        ]
        mon = NewsMonitor()
        with patch(
            "monitoring.news_monitor.feedparser.parse",
            return_value=_fake_feed(entries),
        ):
            items = mon.fetch_all(force=True)

        assert len(items) == 2
        # Sorted newest first
        assert items[0].published >= items[1].published
        # Cache file written
        assert mon.cache_file.exists()


# --------------------------------------------------------------------------- #
# 4. Error handling — feedparser raising inside _fetch_feed
# --------------------------------------------------------------------------- #


class TestFeedErrorHandling:
    def test_exception_in_parse_yields_empty_list(self):
        mon = NewsMonitor()
        with patch(
            "monitoring.news_monitor.feedparser.parse",
            side_effect=RuntimeError("500 Internal Server Error"),
        ):
            items = mon._fetch_feed({"name": "X", "url": "https://x"})
        assert items == []

    def test_malformed_empty_feed_returns_no_items(self):
        mon = NewsMonitor()
        with patch(
            "monitoring.news_monitor.feedparser.parse",
            return_value=_fake_feed(entries=[], bozo=True),
        ):
            items = mon._fetch_feed({"name": "X", "url": "https://x"})
        assert items == []
