"""
Postcode Data
=============

Postcode lookup and geographic utilities.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    POSTCODE_COORDINATES,
    DEFAULT_SITES,
    get_logger
)

logger = get_logger(__name__)


@dataclass
class PostcodeInfo:
    """Information about a postcode district"""
    district: str
    latitude: float
    longitude: float
    area_name: str
    region: str


@dataclass
class TravelInfo:
    """Travel information between two points"""
    from_postcode: str
    to_site: str
    distance_km: float
    estimated_minutes: int
    route_type: str  # local, regional, motorway


class PostcodeService:
    """
    Provides postcode lookup and geographic calculations.

    Uses embedded coordinate data for South Wales postcodes.
    """

    # Area names for postcode prefixes
    AREA_NAMES = {
        'CF10': 'Cardiff City Centre',
        'CF11': 'Canton/Riverside',
        'CF14': 'Whitchurch/Heath',
        'CF15': 'Radyr/Creigiau',
        'CF23': 'Pentwyn/Llanedeyrn',
        'CF24': 'Roath/Cathays',
        'CF3': 'Rumney/Llanrumney',
        'CF5': 'Ely/Caerau',
        'CF62': 'Barry',
        'CF63': 'Barry',
        'CF64': 'Penarth',
        'CF37': 'Pontypridd',
        'CF38': 'Church Village',
        'CF47': 'Merthyr Tydfil',
        'CF72': 'Llantrisant',
        'CF83': 'Caerphilly',
        'NP10': 'Newport East',
        'NP19': 'Newport South',
        'NP20': 'Newport City',
        'NP44': 'Cwmbran',
        'SA1': 'Swansea City',
        'SA2': 'Sketty/Uplands',
        'SA3': 'Gower',
    }

    # Regions
    REGIONS = {
        'CF': 'Cardiff & Vale',
        'NP': 'Gwent',
        'SA': 'Swansea Bay'
    }

    # Average speeds by route type (km/h)
    ROUTE_SPEEDS = {
        'local': 30,
        'regional': 50,
        'motorway': 80
    }

    def __init__(self):
        """Initialize postcode service"""
        self.coordinates = POSTCODE_COORDINATES
        self.sites = {s['code']: s for s in DEFAULT_SITES}
        logger.info(f"Postcode service initialized with {len(self.coordinates)} districts")

    def lookup(self, postcode: str) -> Optional[PostcodeInfo]:
        """
        Look up postcode information.

        Args:
            postcode: Postcode or district (e.g., 'CF14' or 'CF14 4XW')

        Returns:
            PostcodeInfo object or None if not found
        """
        # Extract district
        district = self._extract_district(postcode)

        if district not in self.coordinates:
            # Try broader match
            for known_district in self.coordinates:
                if district.startswith(known_district[:3]):
                    district = known_district
                    break
            else:
                return None

        coords = self.coordinates[district]
        prefix = district[:2]

        return PostcodeInfo(
            district=district,
            latitude=coords['lat'],
            longitude=coords['lon'],
            area_name=self.AREA_NAMES.get(district, f"{district} area"),
            region=self.REGIONS.get(prefix, 'Wales')
        )

    def _extract_district(self, postcode: str) -> str:
        """Extract district from full postcode"""
        postcode = postcode.upper().strip()

        # If already a district (e.g., CF14)
        if len(postcode) <= 4:
            return postcode

        # Extract outward code (first part)
        parts = postcode.split()
        if parts:
            return parts[0]

        # Handle no space format
        import re
        match = re.match(r'^([A-Z]{1,2}[0-9]{1,2})', postcode)
        if match:
            return match.group(1)

        return postcode[:4]

    def calculate_distance(self, postcode1: str, postcode2: str) -> Optional[float]:
        """
        Calculate distance between two postcodes in km.

        Args:
            postcode1: First postcode
            postcode2: Second postcode

        Returns:
            Distance in kilometers or None if postcodes not found
        """
        info1 = self.lookup(postcode1)
        info2 = self.lookup(postcode2)

        if not info1 or not info2:
            return None

        return self._haversine(
            info1.latitude, info1.longitude,
            info2.latitude, info2.longitude
        )

    def calculate_travel_to_site(self, postcode: str,
                                  site_code: str) -> Optional[TravelInfo]:
        """
        Calculate travel information from postcode to site.

        Args:
            postcode: Patient postcode
            site_code: Site code (e.g., 'WC')

        Returns:
            TravelInfo object or None
        """
        patient_info = self.lookup(postcode)
        site = self.sites.get(site_code)

        if not patient_info or not site:
            return None

        # Calculate distance
        distance = self._haversine(
            patient_info.latitude, patient_info.longitude,
            site['lat'], site['lon']
        )

        # Determine route type
        route_type = self._determine_route_type(
            patient_info.district, distance
        )

        # Estimate travel time
        speed = self.ROUTE_SPEEDS[route_type]
        minutes = int(distance / speed * 60)

        # Add buffer for parking, walking, etc.
        minutes += 10

        return TravelInfo(
            from_postcode=patient_info.district,
            to_site=site_code,
            distance_km=round(distance, 1),
            estimated_minutes=minutes,
            route_type=route_type
        )

    def _haversine(self, lat1: float, lon1: float,
                   lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points"""
        R = 6371  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def _determine_route_type(self, district: str, distance: float) -> str:
        """Determine likely route type based on origin and distance"""
        # Short distances are local
        if distance < 10:
            return 'local'

        # Long distances likely use motorway
        if distance > 30:
            return 'motorway'

        # Check if postcode is near M4
        m4_districts = ['CF3', 'CF10', 'CF11', 'CF5', 'NP10', 'NP19', 'NP20']
        if any(district.startswith(d) for d in m4_districts):
            return 'motorway'

        return 'regional'

    def get_nearby_postcodes(self, postcode: str,
                             radius_km: float = 10) -> List[str]:
        """
        Get postcodes within radius of given postcode.

        Args:
            postcode: Centre postcode
            radius_km: Search radius in km

        Returns:
            List of nearby postcode districts
        """
        centre = self.lookup(postcode)
        if not centre:
            return []

        nearby = []
        for district, coords in self.coordinates.items():
            distance = self._haversine(
                centre.latitude, centre.longitude,
                coords['lat'], coords['lon']
            )
            if distance <= radius_km and district != centre.district:
                nearby.append(district)

        return sorted(nearby)

    def get_closest_site(self, postcode: str) -> Optional[Dict]:
        """
        Find closest site to a postcode.

        Args:
            postcode: Patient postcode

        Returns:
            Site dict with distance info
        """
        patient_info = self.lookup(postcode)
        if not patient_info:
            return None

        closest = None
        min_distance = float('inf')

        for site_code, site in self.sites.items():
            distance = self._haversine(
                patient_info.latitude, patient_info.longitude,
                site['lat'], site['lon']
            )
            if distance < min_distance:
                min_distance = distance
                closest = {
                    **site,
                    'distance_km': round(distance, 1)
                }

        return closest

    def get_all_districts(self) -> List[PostcodeInfo]:
        """Get info for all known postcode districts"""
        return [
            self.lookup(district)
            for district in sorted(self.coordinates.keys())
        ]

    def get_districts_by_region(self) -> Dict[str, List[str]]:
        """Get districts grouped by region"""
        result = {}

        for district in self.coordinates.keys():
            prefix = district[:2]
            region = self.REGIONS.get(prefix, 'Other')

            if region not in result:
                result[region] = []
            result[region].append(district)

        # Sort each region's districts
        for region in result:
            result[region] = sorted(result[region])

        return result

    def validate_postcode(self, postcode: str) -> Tuple[bool, str]:
        """
        Validate a postcode and return status message.

        Args:
            postcode: Postcode to validate

        Returns:
            Tuple of (is_valid, message)
        """
        info = self.lookup(postcode)

        if info:
            return True, f"Valid: {info.area_name} ({info.region})"
        else:
            # Check if format is correct
            import re
            pattern = r'^[A-Z]{1,2}[0-9]{1,2}[A-Z]?\s*[0-9][A-Z]{2}$'
            if re.match(pattern, postcode.upper()):
                return True, "Valid format but area not in service region"
            else:
                return False, "Invalid postcode format"


# Example usage
if __name__ == "__main__":
    service = PostcodeService()

    # Test lookup
    print("Postcode Lookup:")
    print("=" * 50)

    test_postcodes = ['CF14 4XW', 'NP20', 'SA1 1AA', 'XX99']
    for pc in test_postcodes:
        info = service.lookup(pc)
        if info:
            print(f"{pc} -> {info.area_name} ({info.region})")
        else:
            print(f"{pc} -> Not found")

    # Test travel calculation
    print("\nTravel to Whitchurch:")
    print("=" * 50)

    test_from = ['CF14', 'NP20', 'CF37', 'SA1']
    for pc in test_from:
        travel = service.calculate_travel_to_site(pc, 'WC')
        if travel:
            print(f"{pc} -> {travel.distance_km} km, ~{travel.estimated_minutes} min ({travel.route_type})")

    # Test nearby postcodes
    print("\nPostcodes within 10km of CF14:")
    print("=" * 50)
    nearby = service.get_nearby_postcodes('CF14', 10)
    print(f"{', '.join(nearby)}")

    # Test closest site
    print("\nClosest site to NP20:")
    print("=" * 50)
    closest = service.get_closest_site('NP20')
    if closest:
        print(f"{closest['name']} - {closest['distance_km']} km")

    # Districts by region
    print("\nDistricts by Region:")
    print("=" * 50)
    by_region = service.get_districts_by_region()
    for region, districts in by_region.items():
        print(f"{region}: {', '.join(districts[:5])}...")
