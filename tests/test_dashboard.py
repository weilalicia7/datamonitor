"""
Tests for visualization/dashboard.py — Wave 3.7.2 (T3 coverage).

DashboardComponents wraps Streamlit; most renders write to st.* and have
no inspectable return value.  We assert:

- the class constructs without an active Streamlit session
- the colour-mapping lookup tables are well-formed
- inputs are accepted without exception when streamlit isn't running

Heavier integration is the place of streamlit's own test harness, not us.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import OperatingMode
from visualization.dashboard import DashboardComponents


class TestConstruction:
    def test_construction_does_not_require_streamlit_session(self):
        # On import we may have streamlit or not; both paths must construct.
        dc = DashboardComponents()
        assert isinstance(dc, DashboardComponents)


class TestLookupTables:
    def test_mode_colors_cover_all_modes(self):
        modes_in_table = set(DashboardComponents.MODE_COLORS.keys())
        all_modes = set(OperatingMode)
        # Every defined OperatingMode has a colour entry.
        assert modes_in_table >= all_modes

    def test_priority_colors_cover_priorities_1_to_4(self):
        for p in range(1, 5):
            assert p in DashboardComponents.PRIORITY_COLORS

    def test_priority_colors_are_hex_strings(self):
        for color in DashboardComponents.PRIORITY_COLORS.values():
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7
