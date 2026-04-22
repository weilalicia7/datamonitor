"""
Dashboard Components
====================

Streamlit dashboard components for the scheduling system.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Streamlit (optional import)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OperatingMode,
    get_logger
)

logger = get_logger(__name__)


class DashboardComponents:
    """
    Provides reusable Streamlit dashboard components.

    Components include:
    - Status cards
    - Metric displays
    - Alert panels
    - Schedule views
    """

    # Mode colors
    MODE_COLORS = {
        OperatingMode.NORMAL: '#28a745',      # Green
        OperatingMode.ELEVATED: '#ffc107',    # Yellow
        OperatingMode.CRISIS: '#fd7e14',      # Orange
        OperatingMode.EMERGENCY: '#dc3545'    # Red
    }

    # Priority colors
    PRIORITY_COLORS = {
        1: '#dc3545',  # Red - Critical
        2: '#fd7e14',  # Orange - High
        3: '#ffc107',  # Yellow - Medium
        4: '#28a745'   # Green - Low
    }

    def __init__(self):
        """Initialize dashboard components"""
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit not available")

    def render_status_header(self, mode: OperatingMode,
                             last_updated: datetime = None):
        """
        Render system status header.

        Args:
            mode: Current operating mode
            last_updated: Last data update time
        """
        if not STREAMLIT_AVAILABLE:
            return

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown(f"""
            <div style="
                padding: 10px 20px;
                background-color: {self.MODE_COLORS[mode]};
                color: white;
                border-radius: 5px;
                font-weight: bold;
            ">
                Operating Mode: {mode.value.upper()}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if last_updated:
                st.markdown(f"Last updated: {last_updated.strftime('%H:%M:%S')}")

        with col3:
            if st.button("🔄 Refresh"):
                st.rerun()

    def render_metrics_row(self, metrics: Dict[str, Dict]):
        """
        Render a row of metric cards.

        Args:
            metrics: Dict of {label: {value, delta, delta_color}}
        """
        if not STREAMLIT_AVAILABLE:
            return

        cols = st.columns(len(metrics))

        for col, (label, data) in zip(cols, metrics.items()):
            with col:
                st.metric(
                    label=label,
                    value=data.get('value', 0),
                    delta=data.get('delta'),
                    delta_color=data.get('delta_color', 'normal')
                )

    def render_alert_panel(self, alerts: List[Dict]):
        """
        Render alert panel.

        Args:
            alerts: List of alert dicts
        """
        if not STREAMLIT_AVAILABLE:
            return

        if not alerts:
            st.info("No active alerts")
            return

        for alert in alerts:
            priority = alert.get('priority', 'medium')
            title = alert.get('title', 'Alert')
            message = alert.get('message', '')

            if priority == 'critical':
                st.error(f"🚨 **{title}**\n\n{message}")
            elif priority == 'high':
                st.warning(f"⚠️ **{title}**\n\n{message}")
            else:
                st.info(f"ℹ️ **{title}**\n\n{message}")

    def render_schedule_table(self, appointments: List,
                              show_actions: bool = True):
        """
        Render schedule as interactive table.

        Args:
            appointments: List of appointment objects
            show_actions: Whether to show action buttons
        """
        if not STREAMLIT_AVAILABLE:
            return

        if not appointments:
            st.info("No appointments scheduled")
            return

        # Convert to DataFrame
        data = []
        for apt in appointments:
            data.append({
                'Time': apt.start_time.strftime('%H:%M'),
                'Patient': apt.patient_id,
                'Chair': apt.chair_id,
                'Duration': f"{apt.duration} min",
                'Priority': f"P{apt.priority}",
                'Site': apt.site_code
            })

        df = pd.DataFrame(data)

        # Style based on priority
        def highlight_priority(row):
            priority = int(row['Priority'][1])
            color = self.PRIORITY_COLORS.get(priority, '#ffffff')
            return [f'background-color: {color}20' for _ in row]

        styled = df.style.apply(highlight_priority, axis=1)
        st.dataframe(styled, use_container_width=True)

    def render_utilization_gauge(self, utilization: float,
                                  target: float = 0.85):
        """
        Render utilization gauge.

        Args:
            utilization: Current utilization (0-1)
            target: Target utilization (0-1)
        """
        if not STREAMLIT_AVAILABLE:
            return

        pct = utilization * 100
        target_pct = target * 100

        # Determine color
        if pct >= target_pct:
            color = '#28a745'
        elif pct >= target_pct * 0.8:
            color = '#ffc107'
        else:
            color = '#dc3545'

        st.markdown(f"""
        <div style="
            background-color: #e9ecef;
            border-radius: 10px;
            height: 30px;
            width: 100%;
            position: relative;
        ">
            <div style="
                background-color: {color};
                width: {min(pct, 100)}%;
                height: 100%;
                border-radius: 10px;
            "></div>
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-weight: bold;
            ">
                {pct:.1f}% (Target: {target_pct:.0f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_event_timeline(self, events: List[Dict]):
        """
        Render event timeline.

        Args:
            events: List of event dicts with start_time, title, severity
        """
        if not STREAMLIT_AVAILABLE:
            return

        if not events:
            st.info("No active events")
            return

        for event in sorted(events, key=lambda e: e.get('severity', 0), reverse=True):
            severity = event.get('severity', 0.5)
            title = event.get('title', 'Event')

            if severity >= 0.7:
                icon = "🔴"
            elif severity >= 0.4:
                icon = "🟡"
            else:
                icon = "🟢"

            with st.expander(f"{icon} {title}"):
                st.write(event.get('description', ''))
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Severity: {severity:.0%}")
                with col2:
                    st.write(f"Type: {event.get('event_type', 'Unknown')}")

    def render_patient_search(self, patients: pd.DataFrame) -> Optional[str]:
        """
        Render patient search widget.

        Args:
            patients: Patient DataFrame

        Returns:
            Selected patient ID or None
        """
        if not STREAMLIT_AVAILABLE:
            return None

        search = st.text_input("Search Patient", placeholder="Enter patient ID or name")

        if search and len(search) >= 2:
            matches = patients[
                patients['patient_id'].str.contains(search, case=False, na=False) |
                patients.get('name', pd.Series()).str.contains(search, case=False, na=False)
            ]

            if not matches.empty:
                selected = st.selectbox(
                    "Select patient",
                    matches['patient_id'].tolist()
                )
                return selected

        return None

    def render_date_selector(self, default_date: datetime = None) -> datetime:
        """
        Render date selector widget.

        Args:
            default_date: Default selected date

        Returns:
            Selected datetime
        """
        if not STREAMLIT_AVAILABLE:
            return default_date or datetime.now()

        default_date = default_date or datetime.now()

        date = st.date_input(
            "Select Date",
            value=default_date.date(),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=90)
        )

        return datetime.combine(date, datetime.min.time())

    def render_site_selector(self, sites: List[Dict],
                             allow_multiple: bool = False) -> List[str]:
        """
        Render site selector widget.

        Args:
            sites: List of site dicts
            allow_multiple: Allow multiple selection

        Returns:
            List of selected site codes
        """
        if not STREAMLIT_AVAILABLE:
            return []

        site_options = {s['name']: s['code'] for s in sites}

        if allow_multiple:
            selected = st.multiselect(
                "Select Sites",
                options=list(site_options.keys()),
                default=list(site_options.keys())
            )
        else:
            selected = [st.selectbox(
                "Select Site",
                options=list(site_options.keys())
            )]

        return [site_options[s] for s in selected]

    def render_sidebar_filters(self, date_range: bool = True,
                                priority_filter: bool = True,
                                site_filter: bool = True) -> Dict:
        """
        Render sidebar filter widgets.

        Args:
            date_range: Include date range filter
            priority_filter: Include priority filter
            site_filter: Include site filter

        Returns:
            Dict of filter values
        """
        if not STREAMLIT_AVAILABLE:
            return {}

        filters = {}

        with st.sidebar:
            st.header("Filters")

            if date_range:
                col1, col2 = st.columns(2)
                with col1:
                    filters['start_date'] = st.date_input("From", datetime.now())
                with col2:
                    filters['end_date'] = st.date_input("To", datetime.now() + timedelta(days=7))

            if priority_filter:
                filters['priorities'] = st.multiselect(
                    "Priority",
                    options=['P1', 'P2', 'P3', 'P4'],
                    default=['P1', 'P2', 'P3', 'P4']
                )

            if site_filter:
                from config import DEFAULT_SITES
                site_names = [s['name'] for s in DEFAULT_SITES]
                filters['sites'] = st.multiselect(
                    "Sites",
                    options=site_names,
                    default=site_names
                )

        return filters

    def render_quick_actions(self) -> Optional[str]:
        """
        Render quick action buttons.

        Returns:
            Selected action or None
        """
        if not STREAMLIT_AVAILABLE:
            return None

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("➕ Add Patient"):
                return "add_patient"

        with col2:
            if st.button("🔄 Re-optimize"):
                return "optimize"

        with col3:
            if st.button("📊 Reports"):
                return "reports"

        with col4:
            if st.button("⚙️ Settings"):
                return "settings"

        return None


# Example usage (runs as standalone demo)
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        st.set_page_config(
            page_title="SACT Scheduler - Demo",
            page_icon="🏥",
            layout="wide"
        )

        components = DashboardComponents()

        st.title("Dashboard Components Demo")

        # Status header
        st.subheader("Status Header")
        components.render_status_header(OperatingMode.ELEVATED, datetime.now())

        # Metrics row
        st.subheader("Metrics Row")
        components.render_metrics_row({
            'Scheduled': {'value': 45, 'delta': 5},
            'Utilization': {'value': '82%', 'delta': '3%'},
            'No-shows': {'value': 2, 'delta': -1, 'delta_color': 'inverse'},
            'Active Events': {'value': 3}
        })

        # Alert panel
        st.subheader("Alert Panel")
        components.render_alert_panel([
            {
                'priority': 'high',
                'title': 'Weather Warning',
                'message': 'Heavy rain expected this afternoon'
            },
            {
                'priority': 'medium',
                'title': 'Traffic Update',
                'message': 'Delays on M4 westbound'
            }
        ])

        # Utilization gauge
        st.subheader("Utilization Gauge")
        components.render_utilization_gauge(0.82)

    else:
        print("Streamlit not available. Install with: pip install streamlit")
