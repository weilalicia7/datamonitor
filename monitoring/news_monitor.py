"""
News Monitor
============

Monitors local news feeds for events that may affect scheduling.
"""

import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import re
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    NEWS_RSS_FEEDS,
    DATA_CACHE_DIR,
    NEWS_UPDATE_INTERVAL,
    get_logger
)

logger = get_logger(__name__)


@dataclass
class NewsItem:
    """Represents a news item from RSS feed"""
    item_id: str
    title: str
    description: str
    link: str
    published: datetime
    source: str
    is_relevant: bool = False
    relevance_score: float = 0.0
    keywords_found: List[str] = None

    def __post_init__(self):
        if self.keywords_found is None:
            self.keywords_found = []


class NewsMonitor:
    """
    Monitors news feeds for relevant events.

    Uses RSS feeds from BBC Wales, Wales Online, and Gov.uk
    to detect events that may affect patient scheduling.
    """

    # Keywords that indicate relevance to scheduling
    RELEVANCE_KEYWORDS = {
        # High relevance (score: 1.0)
        'high': [
            'nhs wales', 'velindre', 'hospital', 'health board',
            'road closure', 'road closed', 'm4 closed', 'flooding',
            'emergency services', 'major incident', 'evacuation'
        ],
        # Medium relevance (score: 0.7)
        'medium': [
            'cardiff', 'newport', 'swansea', 'traffic', 'accident',
            'weather warning', 'storm', 'snow', 'ice', 'delays',
            'transport', 'bus', 'train', 'strike', 'protest'
        ],
        # Low relevance (score: 0.4)
        'low': [
            'wales', 'south wales', 'roadworks', 'event', 'concert',
            'rugby', 'football', 'stadium', 'principality'
        ]
    }

    # Keywords to exclude (not relevant)
    EXCLUDE_KEYWORDS = [
        'sport results', 'celebrity', 'entertainment', 'review',
        'opinion', 'letters', 'tv', 'radio', 'podcast'
    ]

    def __init__(self):
        """Initialize news monitor"""
        self.cache_file = DATA_CACHE_DIR / "news_cache.json"
        self._cache = self._load_cache()
        self.feeds = NEWS_RSS_FEEDS

    def _load_cache(self) -> Dict:
        """Load cached news data"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert dicts back to NewsItem objects
                    items = []
                    for item_dict in data.get('items', []):
                        try:
                            published = item_dict.get('published')
                            if isinstance(published, str):
                                published = datetime.fromisoformat(published)
                            news_item = NewsItem(
                                item_id=item_dict.get('item_id', ''),
                                title=item_dict.get('title', ''),
                                description=item_dict.get('description', ''),
                                link=item_dict.get('link', ''),
                                published=published,
                                source=item_dict.get('source', ''),
                                is_relevant=item_dict.get('is_relevant', False),
                                relevance_score=item_dict.get('relevance_score', 0.0),
                                keywords_found=item_dict.get('keywords_found', [])
                            )
                            items.append(news_item)
                        except Exception:
                            continue
                    data['items'] = items
                    return data
            except Exception as e:
                logger.warning(f"Failed to load news cache: {e}")
        return {'items': [], 'last_updated': None}

    def _save_cache(self) -> None:
        """Save news data to cache"""
        try:
            cache_data = {
                'items': [
                    {
                        'item_id': item.item_id,
                        'title': item.title,
                        'description': item.description,
                        'link': item.link,
                        'published': item.published.isoformat(),
                        'source': item.source,
                        'is_relevant': item.is_relevant,
                        'relevance_score': item.relevance_score,
                        'keywords_found': item.keywords_found
                    }
                    for item in self._cache.get('items', [])
                    if isinstance(item, NewsItem)
                ],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save news cache: {e}")

    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats from RSS feeds"""
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Default to now if parsing fails
        logger.warning(f"Could not parse date: {date_str}")
        return datetime.now()

    def _calculate_relevance(self, title: str, description: str) -> tuple:
        """
        Calculate relevance score and find keywords.

        Args:
            title: News item title
            description: News item description

        Returns:
            Tuple of (score, keywords_found)
        """
        text = f"{title} {description}".lower()

        # Check for exclusions first
        for keyword in self.EXCLUDE_KEYWORDS:
            if keyword in text:
                return 0.0, []

        keywords_found = []
        max_score = 0.0

        # Check high relevance keywords
        for keyword in self.RELEVANCE_KEYWORDS['high']:
            if keyword in text:
                keywords_found.append(keyword)
                max_score = max(max_score, 1.0)

        # Check medium relevance keywords
        for keyword in self.RELEVANCE_KEYWORDS['medium']:
            if keyword in text:
                keywords_found.append(keyword)
                max_score = max(max_score, 0.7)

        # Check low relevance keywords
        for keyword in self.RELEVANCE_KEYWORDS['low']:
            if keyword in text:
                keywords_found.append(keyword)
                max_score = max(max_score, 0.4)

        # Boost score if multiple keywords found
        if len(keywords_found) > 2:
            max_score = min(1.0, max_score + 0.1)

        return max_score, keywords_found

    def _fetch_feed(self, feed_config: Dict) -> List[NewsItem]:
        """
        Fetch and parse a single RSS feed.

        Args:
            feed_config: Feed configuration dict

        Returns:
            List of NewsItem objects
        """
        items = []

        try:
            feed = feedparser.parse(feed_config['url'])

            if feed.bozo and feed.bozo_exception:
                logger.warning(f"Feed error for {feed_config['name']}: {feed.bozo_exception}")

            for entry in feed.entries[:20]:  # Limit to 20 most recent
                # Generate unique ID
                item_id = entry.get('id', entry.get('link', str(hash(entry.get('title', '')))))

                # Parse published date
                pub_date = self._parse_date(
                    entry.get('published', entry.get('updated', datetime.now().isoformat()))
                )

                # Skip old items (more than 24 hours)
                if datetime.now() - pub_date.replace(tzinfo=None) > timedelta(hours=24):
                    continue

                # Get description
                description = entry.get('description', entry.get('summary', ''))
                # Clean HTML tags
                description = re.sub(r'<[^>]+>', '', description)[:500]

                # Calculate relevance
                score, keywords = self._calculate_relevance(
                    entry.get('title', ''),
                    description
                )

                item = NewsItem(
                    item_id=item_id,
                    title=entry.get('title', 'No title'),
                    description=description,
                    link=entry.get('link', ''),
                    published=pub_date.replace(tzinfo=None) if pub_date.tzinfo else pub_date,
                    source=feed_config['name'],
                    is_relevant=score >= 0.4,
                    relevance_score=score,
                    keywords_found=keywords
                )
                items.append(item)

        except Exception as e:
            logger.error(f"Error fetching feed {feed_config['name']}: {e}")

        return items

    def fetch_all(self, force: bool = False) -> List[NewsItem]:
        """
        Fetch news from all configured feeds.

        Args:
            force: Force refresh even if cache is valid

        Returns:
            List of NewsItem objects
        """
        # Check cache
        if not force and self._cache.get('last_updated'):
            try:
                last_updated = datetime.fromisoformat(self._cache['last_updated'])
                age = (datetime.now() - last_updated).total_seconds()
                if age < NEWS_UPDATE_INTERVAL:
                    logger.debug("Using cached news data")
                    return self._cache.get('items', [])
            except:
                pass

        all_items = []
        for feed_config in self.feeds:
            items = self._fetch_feed(feed_config)
            all_items.extend(items)
            logger.debug(f"Fetched {len(items)} items from {feed_config['name']}")

        # Sort by published date (newest first)
        all_items.sort(key=lambda x: x.published, reverse=True)

        # Update cache
        self._cache['items'] = all_items
        self._cache['last_updated'] = datetime.now().isoformat()
        self._save_cache()

        logger.info(f"Fetched {len(all_items)} news items total")
        return all_items

    def get_relevant_items(self, min_score: float = 0.4) -> List[NewsItem]:
        """
        Get only relevant news items.

        Args:
            min_score: Minimum relevance score (0-1)

        Returns:
            List of relevant NewsItem objects
        """
        all_items = self.fetch_all()
        return [
            item for item in all_items
            if item.relevance_score >= min_score
        ]

    def get_headlines(self, limit: int = 10) -> List[str]:
        """
        Get relevant headlines for display.

        Args:
            limit: Maximum number of headlines

        Returns:
            List of headline strings
        """
        relevant = self.get_relevant_items()
        return [item.title for item in relevant[:limit]]

    def search(self, query: str) -> List[NewsItem]:
        """
        Search news items by keyword.

        Args:
            query: Search query

        Returns:
            List of matching NewsItem objects
        """
        all_items = self.fetch_all()
        query_lower = query.lower()

        return [
            item for item in all_items
            if query_lower in item.title.lower()
            or query_lower in item.description.lower()
        ]

    def get_by_source(self, source: str) -> List[NewsItem]:
        """
        Get news items from a specific source.

        Args:
            source: Source name (e.g., 'BBC Wales')

        Returns:
            List of NewsItem objects
        """
        all_items = self.fetch_all()
        return [item for item in all_items if item.source == source]

    def get_emergency_alerts(self) -> List[NewsItem]:
        """
        Get high-priority emergency alerts.

        Returns:
            List of emergency-related NewsItem objects
        """
        emergency_keywords = [
            'major incident', 'emergency', 'evacuation', 'closure',
            'severe weather', 'flooding', 'accident', 'road closed'
        ]

        all_items = self.fetch_all()
        alerts = []

        for item in all_items:
            text = f"{item.title} {item.description}".lower()
            for keyword in emergency_keywords:
                if keyword in text:
                    alerts.append(item)
                    break

        return alerts


# Example usage
if __name__ == "__main__":
    monitor = NewsMonitor()

    # Fetch all news
    print("Fetching news...")
    items = monitor.fetch_all()
    print(f"Found {len(items)} items")

    # Get relevant items
    relevant = monitor.get_relevant_items()
    print(f"\nRelevant items: {len(relevant)}")
    for item in relevant[:5]:
        print(f"  [{item.source}] {item.title}")
        print(f"    Score: {item.relevance_score}, Keywords: {item.keywords_found}")

    # Get emergency alerts
    alerts = monitor.get_emergency_alerts()
    if alerts:
        print(f"\nEmergency alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert.title}")
