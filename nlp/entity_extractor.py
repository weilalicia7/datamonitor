"""
Entity Extractor
================

Extracts relevant entities from text (locations, times, roads, etc.)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import POSTCODE_COORDINATES, get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedEntities:
    """Collection of extracted entities"""
    text: str
    locations: List[str] = field(default_factory=list)
    postcodes: List[str] = field(default_factory=list)
    roads: List[str] = field(default_factory=list)
    times: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    durations: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    numbers: List[Tuple[str, float]] = field(default_factory=list)


class EntityExtractor:
    """
    Extracts scheduling-relevant entities from text.

    Uses regex patterns and keyword matching to identify
    locations, times, roads, and other relevant entities.
    """

    # South Wales locations
    LOCATIONS = [
        'cardiff', 'newport', 'swansea', 'barry', 'penarth',
        'pontypridd', 'caerphilly', 'bridgend', 'cwmbran',
        'merthyr', 'aberdare', 'rhondda', 'llantrisant',
        'whitchurch', 'heath', 'canton', 'roath', 'splott',
        'grangetown', 'ely', 'fairwater', 'llandaff'
    ]

    # Known organizations
    ORGANIZATIONS = [
        'velindre', 'nhs wales', 'health board', 'uhw',
        'university hospital', 'police', 'fire service',
        'ambulance', 'welsh government', 'transport for wales',
        'arriva', 'stagecoach', 'cardiff council', 'bbc wales'
    ]

    # Road patterns
    ROAD_PATTERNS = [
        r'\bM4\b', r'\bM5\b', r'\bM48\b', r'\bM49\b',
        r'\bA4[0-9]{1,3}\b', r'\bA48[0-9]?\b', r'\bA470\b',
        r'\bA4232\b', r'\bA469\b', r'\bA4119\b',
        r'\bB[0-9]{4}\b'
    ]

    # Postcode pattern (UK format)
    POSTCODE_PATTERN = r'\b([A-Z]{1,2}[0-9]{1,2})\s*([0-9][A-Z]{2})?\b'

    # Time patterns
    TIME_PATTERNS = [
        r'\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b',  # 24-hour
        r'\b([01]?[0-9])\.([0-5][0-9])\s*(am|pm)\b',  # 12-hour with dot
        r'\b([01]?[0-9]):([0-5][0-9])\s*(am|pm)\b',  # 12-hour with colon
        r'\b([01]?[0-9])\s*(am|pm)\b',  # Hour only
    ]

    # Date patterns
    DATE_PATTERNS = [
        r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b',  # DD/MM/YYYY
        r'\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})?\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(today|tomorrow|yesterday)\b',
        r'\b(this|next)\s+(week|weekend|month)\b',
    ]

    # Duration patterns
    DURATION_PATTERNS = [
        r'\b(\d+)\s*hours?\b',
        r'\b(\d+)\s*minutes?\b',
        r'\b(\d+)\s*mins?\b',
        r'\b(\d+)\s*days?\b',
        r'\b(several|few)\s+(hours?|minutes?|days?)\b',
        r'\buntil\s+(\d{1,2}:\d{2}|\d{1,2}\s*(am|pm))\b',
    ]

    def __init__(self):
        """Initialize entity extractor"""
        # Compile regex patterns
        self._road_regex = [re.compile(p, re.IGNORECASE) for p in self.ROAD_PATTERNS]
        self._postcode_regex = re.compile(self.POSTCODE_PATTERN, re.IGNORECASE)
        self._time_regexes = [re.compile(p, re.IGNORECASE) for p in self.TIME_PATTERNS]
        self._date_regexes = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self._duration_regexes = [re.compile(p, re.IGNORECASE) for p in self.DURATION_PATTERNS]

        logger.info("Entity extractor initialized")

    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            ExtractedEntities object
        """
        text_lower = text.lower()

        return ExtractedEntities(
            text=text,
            locations=self._extract_locations(text_lower),
            postcodes=self._extract_postcodes(text),
            roads=self._extract_roads(text),
            times=self._extract_times(text),
            dates=self._extract_dates(text_lower),
            durations=self._extract_durations(text_lower),
            organizations=self._extract_organizations(text_lower),
            numbers=self._extract_numbers(text)
        )

    def _extract_locations(self, text: str) -> List[str]:
        """Extract location names"""
        found = []
        for location in self.LOCATIONS:
            if location in text:
                found.append(location.title())
        return list(set(found))

    def _extract_postcodes(self, text: str) -> List[str]:
        """Extract UK postcodes"""
        matches = self._postcode_regex.findall(text)
        postcodes = []
        for match in matches:
            postcode = match[0].upper()
            # Validate against known postcodes
            if postcode in POSTCODE_COORDINATES or postcode[:2] in ['CF', 'NP', 'SA']:
                postcodes.append(postcode)
        return list(set(postcodes))

    def _extract_roads(self, text: str) -> List[str]:
        """Extract road names"""
        roads = []
        for pattern in self._road_regex:
            matches = pattern.findall(text)
            roads.extend(matches)
        return list(set([r.upper() for r in roads]))

    def _extract_times(self, text: str) -> List[str]:
        """Extract time references"""
        times = []
        for pattern in self._time_regexes:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    times.append(':'.join([str(m) for m in match if m]))
                else:
                    times.append(match)
        return list(set(times))

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date references"""
        dates = []
        for pattern in self._date_regexes:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    dates.append(' '.join([str(m) for m in match if m]))
                else:
                    dates.append(match)
        return list(set(dates))

    def _extract_durations(self, text: str) -> List[str]:
        """Extract duration references"""
        durations = []
        for pattern in self._duration_regexes:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    durations.append(' '.join([str(m) for m in match if m]))
                else:
                    durations.append(match)
        return list(set(durations))

    def _extract_organizations(self, text: str) -> List[str]:
        """Extract organization names"""
        found = []
        for org in self.ORGANIZATIONS:
            if org in text:
                found.append(org.title())
        return list(set(found))

    def _extract_numbers(self, text: str) -> List[Tuple[str, float]]:
        """Extract numbers with context"""
        numbers = []

        # Percentage pattern
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        for match in pct_matches:
            numbers.append(('percentage', float(match)))

        # Temperature pattern
        temp_matches = re.findall(r'(-?\d+(?:\.\d+)?)\s*°?[cCfF]', text)
        for match in temp_matches:
            numbers.append(('temperature', float(match)))

        # Speed pattern (mph, km/h)
        speed_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(mph|km/?h)', text, re.IGNORECASE)
        for match in speed_matches:
            numbers.append(('speed', float(match[0])))

        # General numbers with units
        general_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|km|miles?)', text, re.IGNORECASE)
        for match in general_matches:
            unit = match[1].lower()
            if 'min' in unit:
                numbers.append(('minutes', float(match[0])))
            elif 'hour' in unit:
                numbers.append(('hours', float(match[0])))
            elif 'km' in unit:
                numbers.append(('kilometers', float(match[0])))
            elif 'mile' in unit:
                numbers.append(('miles', float(match[0])))

        return numbers

    def get_affected_postcodes(self, entities: ExtractedEntities) -> List[str]:
        """
        Get all postcodes that might be affected based on extracted entities.

        Args:
            entities: ExtractedEntities object

        Returns:
            List of postcode districts
        """
        affected = set(entities.postcodes)

        # Map locations to postcodes
        location_postcode_map = {
            'cardiff': ['CF10', 'CF11', 'CF14', 'CF15', 'CF23', 'CF24'],
            'whitchurch': ['CF14'],
            'heath': ['CF14', 'CF23'],
            'newport': ['NP10', 'NP19', 'NP20'],
            'swansea': ['SA1', 'SA2', 'SA3'],
            'barry': ['CF62', 'CF63'],
            'penarth': ['CF64'],
            'pontypridd': ['CF37', 'CF38'],
            'caerphilly': ['CF83'],
            'bridgend': ['CF31', 'CF32'],
            'cwmbran': ['NP44'],
            'merthyr': ['CF47', 'CF48'],
        }

        for location in entities.locations:
            postcodes = location_postcode_map.get(location.lower(), [])
            affected.update(postcodes)

        # Map roads to affected postcodes
        road_postcode_map = {
            'M4': ['CF10', 'CF3', 'CF14', 'CF15', 'CF5', 'NP10', 'NP19', 'NP20'],
            'A470': ['CF14', 'CF15', 'CF37', 'CF48'],
            'A48': ['CF3', 'CF23', 'CF24', 'NP10'],
            'A4232': ['CF10', 'CF11', 'CF5'],
            'A4119': ['CF37', 'CF38', 'CF72'],
        }

        for road in entities.roads:
            postcodes = road_postcode_map.get(road.upper(), [])
            affected.update(postcodes)

        return list(affected)

    def estimate_duration_minutes(self, entities: ExtractedEntities) -> Optional[int]:
        """
        Estimate event duration in minutes from extracted entities.

        Args:
            entities: ExtractedEntities object

        Returns:
            Estimated duration in minutes or None
        """
        for num_type, value in entities.numbers:
            if num_type == 'minutes':
                return int(value)
            elif num_type == 'hours':
                return int(value * 60)

        # Parse duration strings
        for duration in entities.durations:
            duration_lower = duration.lower()
            if 'several hours' in duration_lower:
                return 180  # 3 hours
            elif 'few hours' in duration_lower:
                return 120  # 2 hours
            elif 'several minutes' in duration_lower:
                return 30
            elif 'few minutes' in duration_lower:
                return 15

        return None

    def extract_batch(self, texts: List[str]) -> List[ExtractedEntities]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of ExtractedEntities objects
        """
        return [self.extract(text) for text in texts]


# Example usage
if __name__ == "__main__":
    extractor = EntityExtractor()

    # Test texts
    test_texts = [
        "M4 closed westbound between J32 and J33 near Cardiff. Delays of 45 minutes expected until 18:00.",
        "Heavy snow forecast for CF14 and CF15 areas. Temperature dropping to -5°C tonight.",
        "Velindre Hospital reporting normal operations. NHS Wales issued guidance for patients.",
        "Major rugby match at Principality Stadium in Cardiff city centre on Saturday 15 March.",
        "Traffic delays on A470 near Pontypridd due to accident. Avoid the area for several hours."
    ]

    print("Entity Extraction Results:")
    print("=" * 70)

    for text in test_texts:
        entities = extractor.extract(text)

        print(f"\nText: {text[:60]}...")
        if entities.locations:
            print(f"  Locations: {entities.locations}")
        if entities.postcodes:
            print(f"  Postcodes: {entities.postcodes}")
        if entities.roads:
            print(f"  Roads: {entities.roads}")
        if entities.times:
            print(f"  Times: {entities.times}")
        if entities.dates:
            print(f"  Dates: {entities.dates}")
        if entities.durations:
            print(f"  Durations: {entities.durations}")
        if entities.organizations:
            print(f"  Organizations: {entities.organizations}")
        if entities.numbers:
            print(f"  Numbers: {entities.numbers}")

        affected = extractor.get_affected_postcodes(entities)
        if affected:
            print(f"  Affected Postcodes: {affected}")

        duration = extractor.estimate_duration_minutes(entities)
        if duration:
            print(f"  Estimated Duration: {duration} minutes")
