"""
Tests for visualization/charts.py + maps.py — Wave 3.7.1/3.7.3 (T3 coverage).

Dashboard tests (visualization/dashboard.py) live in test_dashboard.py
since they need Streamlit-context mocking.

These tests assert that each chart/map factory:
- Returns the right wrapper type (Plotly Figure / Folium Map)
- Returns None / safe defaults on empty input
- Doesn't raise on representative synthetic data
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go

from visualization.charts import ChartGenerator
from visualization.maps import MapGenerator


# --------------------------------------------------------------------------- #
# Synthetic appointment helper
# --------------------------------------------------------------------------- #


def _appt(pid: str = "P001", *, chair="WC-C01", priority=2,
          start: datetime = None, duration: int = 60):
    start = start or datetime(2026, 4, 22, 10, 0)
    end = start + timedelta(minutes=duration)
    return SimpleNamespace(
        patient_id=pid,
        chair_id=chair,
        start_time=start,
        end_time=end,
        priority=priority,
        duration=duration,
    )


# --------------------------------------------------------------------------- #
# ChartGenerator
# --------------------------------------------------------------------------- #


class TestChartGenerator:
    def test_timeline_returns_plotly_figure(self):
        cg = ChartGenerator()
        appts = [_appt(f"P{i:03d}", chair=f"WC-C0{i+1}",
                       start=datetime(2026, 4, 22, 9 + i, 0))
                 for i in range(3)]
        fig = cg.create_schedule_timeline(appts, date=datetime(2026, 4, 22))
        assert isinstance(fig, go.Figure)

    def test_timeline_empty_returns_none(self):
        cg = ChartGenerator()
        assert cg.create_schedule_timeline([]) is None

    def test_priority_pie(self):
        cg = ChartGenerator()
        appts = [_appt(priority=1), _appt(priority=2), _appt(priority=2),
                 _appt(priority=3)]
        fig = cg.create_priority_pie_chart(appts)
        assert isinstance(fig, go.Figure)

    def test_utilization_chart(self):
        cg = ChartGenerator()
        # Method expects {date, utilization} time-series rows.
        data = [
            {"date": "2026-04-20", "utilization": 0.81},
            {"date": "2026-04-21", "utilization": 0.85},
            {"date": "2026-04-22", "utilization": 0.79},
        ]
        fig = cg.create_utilization_chart(data)
        assert isinstance(fig, go.Figure)


# --------------------------------------------------------------------------- #
# MapGenerator
# --------------------------------------------------------------------------- #


class TestMapGenerator:
    def test_site_map_returns_folium_object(self):
        folium = pytest.importorskip("folium")
        mg = MapGenerator()
        m = mg.create_site_map()
        # Folium returns a folium.folium.Map; just check it has _to_html_str-ish.
        assert m is not None
        assert hasattr(m, "_repr_html_") or hasattr(m, "save")

    def test_event_map_handles_empty(self):
        folium = pytest.importorskip("folium")
        mg = MapGenerator()
        m = mg.create_event_map(events=[])
        assert m is not None

    def test_map_to_html(self):
        folium = pytest.importorskip("folium")
        mg = MapGenerator()
        m = mg.create_site_map()
        html = mg.map_to_html(m)
        assert isinstance(html, str)
        assert "<html" in html.lower() or "leaflet" in html.lower()
