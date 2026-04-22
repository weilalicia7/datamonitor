"""
Maps Module
===========

Creates geographic visualizations using Folium.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

# Folium (optional import)
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    folium = None
    FOLIUM_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    POSTCODE_COORDINATES,
    DEFAULT_SITES,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    get_logger
)

logger = get_logger(__name__)


class MapGenerator:
    """
    Generates geographic map visualizations.

    Uses Folium for interactive maps that work
    with Streamlit via st.components.
    """

    # Marker colors by type
    SITE_COLORS = {
        'main': 'blue',
        'satellite': 'green',
        'community': 'orange'
    }

    SEVERITY_COLORS = {
        'low': 'green',
        'medium': 'orange',
        'high': 'red'
    }

    def __init__(self):
        """Initialize map generator"""
        if not FOLIUM_AVAILABLE:
            logger.warning("Folium not available. Maps will not be generated.")

        self.default_center = (DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
        self.default_zoom = 10

    def create_site_map(self, sites: List[Dict] = None,
                         show_postcodes: bool = False) -> Optional[Any]:
        """
        Create map showing treatment sites.

        Args:
            sites: List of site dicts (uses default if None)
            show_postcodes: Whether to show postcode markers

        Returns:
            Folium Map or None
        """
        if not FOLIUM_AVAILABLE:
            return None

        sites = sites or DEFAULT_SITES

        # Create base map
        m = folium.Map(
            location=self.default_center,
            zoom_start=self.default_zoom,
            tiles='CartoDB positron'
        )

        # Add site markers
        for site in sites:
            icon_color = self.SITE_COLORS.get(site.get('type', 'main'), 'blue')

            folium.Marker(
                location=[site['lat'], site['lon']],
                popup=folium.Popup(
                    f"<b>{site['name']}</b><br>"
                    f"Code: {site['code']}<br>"
                    f"Chairs: {site['chairs']}<br>"
                    f"Recliners: {site.get('recliners', 0)}",
                    max_width=200
                ),
                tooltip=site['name'],
                icon=folium.Icon(color=icon_color, icon='plus', prefix='fa')
            ).add_to(m)

        # Add postcode markers if requested
        if show_postcodes:
            for postcode, coords in POSTCODE_COORDINATES.items():
                folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=5,
                    color='gray',
                    fill=True,
                    fillColor='lightgray',
                    fillOpacity=0.6,
                    popup=postcode,
                    tooltip=postcode
                ).add_to(m)

        return m

    def create_patient_distribution_map(self, patient_postcodes: List[str],
                                         sites: List[Dict] = None) -> Optional[Any]:
        """
        Create heatmap of patient locations.

        Args:
            patient_postcodes: List of patient postcode districts
            sites: List of site dicts

        Returns:
            Folium Map or None
        """
        if not FOLIUM_AVAILABLE:
            return None

        sites = sites or DEFAULT_SITES

        # Create base map
        m = folium.Map(
            location=self.default_center,
            zoom_start=self.default_zoom,
            tiles='CartoDB positron'
        )

        # Count patients per postcode
        postcode_counts = {}
        for pc in patient_postcodes:
            district = pc[:4] if len(pc) > 4 else pc
            postcode_counts[district] = postcode_counts.get(district, 0) + 1

        # Create heatmap data
        heat_data = []
        for postcode, count in postcode_counts.items():
            if postcode in POSTCODE_COORDINATES:
                coords = POSTCODE_COORDINATES[postcode]
                for _ in range(count):
                    heat_data.append([coords['lat'], coords['lon']])

        # Add heatmap layer
        if heat_data:
            HeatMap(heat_data, radius=15, blur=10).add_to(m)

        # Add site markers
        for site in sites:
            folium.Marker(
                location=[site['lat'], site['lon']],
                popup=site['name'],
                icon=folium.Icon(color='blue', icon='hospital-o', prefix='fa')
            ).add_to(m)

        return m

    def create_event_map(self, events: List[Dict],
                          sites: List[Dict] = None) -> Optional[Any]:
        """
        Create map showing active events.

        Args:
            events: List of event dicts with location info
            sites: List of site dicts

        Returns:
            Folium Map or None
        """
        if not FOLIUM_AVAILABLE:
            return None

        sites = sites or DEFAULT_SITES

        # Create base map
        m = folium.Map(
            location=self.default_center,
            zoom_start=self.default_zoom,
            tiles='CartoDB positron'
        )

        # Add site markers
        for site in sites:
            folium.Marker(
                location=[site['lat'], site['lon']],
                popup=site['name'],
                icon=folium.Icon(color='blue', icon='hospital-o', prefix='fa')
            ).add_to(m)

        # Add event markers
        for event in events:
            # Determine location from affected postcodes
            postcodes = event.get('affected_postcodes', [])
            if not postcodes:
                continue

            # Use first affected postcode as location
            postcode = postcodes[0]
            if postcode not in POSTCODE_COORDINATES:
                continue

            coords = POSTCODE_COORDINATES[postcode]
            severity = event.get('severity', 0.5)

            # Determine color
            if severity >= 0.7:
                color = 'red'
            elif severity >= 0.4:
                color = 'orange'
            else:
                color = 'green'

            # Create marker
            folium.CircleMarker(
                location=[coords['lat'], coords['lon']],
                radius=10 + severity * 20,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.5,
                popup=folium.Popup(
                    f"<b>{event.get('title', 'Event')}</b><br>"
                    f"Severity: {severity:.0%}<br>"
                    f"Type: {event.get('event_type', 'Unknown')}<br>"
                    f"Affected: {', '.join(postcodes[:3])}",
                    max_width=250
                ),
                tooltip=event.get('title', 'Event')
            ).add_to(m)

            # Add affected area circle
            folium.Circle(
                location=[coords['lat'], coords['lon']],
                radius=event.get('affected_radius_km', 10) * 1000,  # Convert to meters
                color=color,
                fill=False,
                weight=2,
                dash_array='5, 5'
            ).add_to(m)

        return m

    def create_traffic_map(self, incidents: List,
                            routes: List[Dict] = None) -> Optional[Any]:
        """
        Create map showing traffic incidents.

        Args:
            incidents: List of TrafficIncident objects
            routes: List of route dicts to highlight

        Returns:
            Folium Map or None
        """
        if not FOLIUM_AVAILABLE:
            return None

        # Create base map
        m = folium.Map(
            location=self.default_center,
            zoom_start=self.default_zoom,
            tiles='CartoDB positron'
        )

        # Add site markers
        for site in DEFAULT_SITES:
            folium.Marker(
                location=[site['lat'], site['lon']],
                popup=site['name'],
                icon=folium.Icon(color='blue', icon='hospital-o', prefix='fa')
            ).add_to(m)

        # Add incident markers
        for incident in incidents:
            if incident.latitude == 0 or incident.longitude == 0:
                continue

            # Determine color based on severity
            if incident.severity >= 0.7:
                color = 'red'
            elif incident.severity >= 0.4:
                color = 'orange'
            else:
                color = 'yellow'

            folium.Marker(
                location=[incident.latitude, incident.longitude],
                popup=folium.Popup(
                    f"<b>{incident.incident_type}</b><br>"
                    f"{incident.description}<br>"
                    f"Road: {incident.road}<br>"
                    f"Delay: {incident.delay_minutes} min",
                    max_width=250
                ),
                tooltip=f"{incident.road}: {incident.incident_type}",
                icon=folium.Icon(color=color, icon='warning', prefix='fa')
            ).add_to(m)

        return m

    def create_coverage_map(self, sites: List[Dict] = None,
                            coverage_radius_km: float = 20) -> Optional[Any]:
        """
        Create map showing site coverage areas.

        Args:
            sites: List of site dicts
            coverage_radius_km: Coverage radius in km

        Returns:
            Folium Map or None
        """
        if not FOLIUM_AVAILABLE:
            return None

        sites = sites or DEFAULT_SITES

        # Create base map
        m = folium.Map(
            location=self.default_center,
            zoom_start=9,
            tiles='CartoDB positron'
        )

        colors = ['blue', 'green', 'purple', 'orange', 'red']

        for i, site in enumerate(sites):
            color = colors[i % len(colors)]

            # Add site marker
            folium.Marker(
                location=[site['lat'], site['lon']],
                popup=site['name'],
                icon=folium.Icon(color=color, icon='hospital-o', prefix='fa')
            ).add_to(m)

            # Add coverage circle
            folium.Circle(
                location=[site['lat'], site['lon']],
                radius=coverage_radius_km * 1000,
                color=color,
                fill=True,
                fillOpacity=0.1,
                weight=2,
                popup=f"{site['name']} coverage area"
            ).add_to(m)

        return m

    def map_to_html(self, m: Any) -> str:
        """
        Convert Folium map to HTML string.

        Args:
            m: Folium Map object

        Returns:
            HTML string
        """
        if not FOLIUM_AVAILABLE or m is None:
            return "<p>Map not available</p>"

        return m._repr_html_()

    def save_map(self, m: Any, filepath: str):
        """
        Save map to HTML file.

        Args:
            m: Folium Map object
            filepath: Output file path
        """
        if not FOLIUM_AVAILABLE or m is None:
            logger.warning("Cannot save map: Folium not available")
            return

        m.save(filepath)
        logger.info(f"Map saved to {filepath}")


# Example usage
if __name__ == "__main__":
    if FOLIUM_AVAILABLE:
        generator = MapGenerator()

        # Create site map
        print("Creating site map...")
        site_map = generator.create_site_map(show_postcodes=True)
        generator.save_map(site_map, "site_map.html")
        print("Site map saved to site_map.html")

        # Create patient distribution map
        print("\nCreating patient distribution map...")
        sample_postcodes = [
            'CF14', 'CF14', 'CF14', 'CF15', 'CF23', 'CF23',
            'NP20', 'NP20', 'CF37', 'CF10', 'CF24', 'SA1'
        ]
        patient_map = generator.create_patient_distribution_map(sample_postcodes)
        generator.save_map(patient_map, "patient_map.html")
        print("Patient map saved to patient_map.html")

        # Create event map
        print("\nCreating event map...")
        sample_events = [
            {
                'title': 'M4 Closure',
                'severity': 0.8,
                'event_type': 'traffic',
                'affected_postcodes': ['CF3', 'CF10', 'CF14'],
                'affected_radius_km': 15
            },
            {
                'title': 'Weather Warning',
                'severity': 0.5,
                'event_type': 'weather',
                'affected_postcodes': ['CF14', 'CF15', 'CF37'],
                'affected_radius_km': 25
            }
        ]
        event_map = generator.create_event_map(sample_events)
        generator.save_map(event_map, "event_map.html")
        print("Event map saved to event_map.html")

        # Create coverage map
        print("\nCreating coverage map...")
        coverage_map = generator.create_coverage_map(coverage_radius_km=25)
        generator.save_map(coverage_map, "coverage_map.html")
        print("Coverage map saved to coverage_map.html")

    else:
        print("Folium not available. Install with: pip install folium")
