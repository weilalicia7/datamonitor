"""
Charts Module
=============

Creates charts and visualizations using Plotly.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Plotly (optional import)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    """
    Generates charts and visualizations.

    Uses Plotly for interactive charts that work
    well with Streamlit and web interfaces.
    """

    # Color schemes
    PRIORITY_COLORS = {
        'P1': '#dc3545',
        'P2': '#fd7e14',
        'P3': '#ffc107',
        'P4': '#28a745'
    }

    SITE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def __init__(self):
        """Initialize chart generator"""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Charts will not be generated.")

    def create_schedule_timeline(self, appointments: List,
                                  date: datetime = None) -> Optional[go.Figure]:
        """
        Create Gantt-style schedule timeline.

        Args:
            appointments: List of appointment objects
            date: Date for the schedule

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or not appointments:
            return None

        data = []
        for apt in appointments:
            data.append({
                'Chair': apt.chair_id,
                'Patient': apt.patient_id,
                'Start': apt.start_time,
                'End': apt.end_time,
                'Priority': f'P{apt.priority}',
                'Duration': apt.duration
            })

        df = pd.DataFrame(data)

        fig = px.timeline(
            df,
            x_start='Start',
            x_end='End',
            y='Chair',
            color='Priority',
            color_discrete_map=self.PRIORITY_COLORS,
            hover_data=['Patient', 'Duration'],
            title=f"Schedule Timeline - {date.strftime('%Y-%m-%d') if date else 'Today'}"
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Chair",
            height=400,
            showlegend=True
        )

        # Add current time line
        now = datetime.now()
        if date and date.date() == now.date():
            fig.add_vline(x=now, line_dash="dash", line_color="red")

        return fig

    def create_utilization_chart(self, utilization_data: List[Dict]) -> Optional[go.Figure]:
        """
        Create utilization over time chart.

        Args:
            utilization_data: List of {date, utilization} dicts

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or not utilization_data:
            return None

        df = pd.DataFrame(utilization_data)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['utilization'] * 100,
            mode='lines+markers',
            name='Utilization',
            line=dict(color='#1f77b4', width=2)
        ))

        # Add target line
        fig.add_hline(
            y=85,
            line_dash="dash",
            line_color="green",
            annotation_text="Target (85%)"
        )

        fig.update_layout(
            title="Chair Utilization Over Time",
            xaxis_title="Date",
            yaxis_title="Utilization (%)",
            yaxis_range=[0, 100],
            height=300
        )

        return fig

    def create_priority_pie_chart(self, appointments: List) -> Optional[go.Figure]:
        """
        Create pie chart of appointments by priority.

        Args:
            appointments: List of appointment objects

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or not appointments:
            return None

        priority_counts = {}
        for apt in appointments:
            key = f'P{apt.priority}'
            priority_counts[key] = priority_counts.get(key, 0) + 1

        fig = go.Figure(data=[go.Pie(
            labels=list(priority_counts.keys()),
            values=list(priority_counts.values()),
            hole=0.4,
            marker_colors=[self.PRIORITY_COLORS.get(p, '#cccccc')
                          for p in priority_counts.keys()]
        )])

        fig.update_layout(
            title="Appointments by Priority",
            height=300
        )

        return fig

    def create_site_comparison_chart(self, site_data: Dict[str, Dict]) -> Optional[go.Figure]:
        """
        Create bar chart comparing site metrics.

        Args:
            site_data: Dict of {site_code: {appointments, utilization, etc.}}

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or not site_data:
            return None

        sites = list(site_data.keys())
        appointments = [site_data[s].get('appointments', 0) for s in sites]
        utilization = [site_data[s].get('utilization', 0) * 100 for s in sites]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Appointments', 'Utilization (%)']
        )

        fig.add_trace(
            go.Bar(x=sites, y=appointments, marker_color='#1f77b4', name='Appointments'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=sites, y=utilization, marker_color='#2ca02c', name='Utilization'),
            row=1, col=2
        )

        fig.update_layout(
            title="Site Comparison",
            showlegend=False,
            height=300
        )

        return fig

    def create_noshow_trend_chart(self, historical_data: List[Dict]) -> Optional[go.Figure]:
        """
        Create no-show trend chart.

        Args:
            historical_data: List of {date, noshow_rate, predicted_rate} dicts

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or not historical_data:
            return None

        df = pd.DataFrame(historical_data)

        fig = go.Figure()

        # Actual no-show rate
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['noshow_rate'] * 100,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#dc3545', width=2)
        ))

        # Predicted rate (if available)
        if 'predicted_rate' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['predicted_rate'] * 100,
                mode='lines',
                name='Predicted',
                line=dict(color='#1f77b4', width=2, dash='dash')
            ))

        fig.update_layout(
            title="No-Show Rate Trend",
            xaxis_title="Date",
            yaxis_title="No-Show Rate (%)",
            height=300,
            legend=dict(x=0, y=1)
        )

        return fig

    def create_weather_severity_chart(self, hourly_data: List[Dict]) -> Optional[go.Figure]:
        """
        Create weather severity forecast chart.

        Args:
            hourly_data: List of {hour, severity, description} dicts

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or not hourly_data:
            return None

        df = pd.DataFrame(hourly_data)

        # Color based on severity
        colors = ['#28a745' if s < 0.3 else '#ffc107' if s < 0.6 else '#dc3545'
                 for s in df['severity']]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df['hour'],
            y=df['severity'] * 100,
            marker_color=colors,
            hovertext=df.get('description', ''),
            name='Severity'
        ))

        fig.update_layout(
            title="Weather Severity Forecast",
            xaxis_title="Hour",
            yaxis_title="Severity (%)",
            yaxis_range=[0, 100],
            height=250
        )

        return fig

    def create_event_impact_chart(self, events: List[Dict]) -> Optional[go.Figure]:
        """
        Create event impact visualization.

        Args:
            events: List of event dicts with severity and impact info

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or not events:
            return None

        df = pd.DataFrame(events)

        fig = px.scatter(
            df,
            x='noshow_impact',
            y='duration_impact',
            size='severity',
            color='category',
            hover_name='title',
            title="Event Impact Analysis",
            labels={
                'noshow_impact': 'No-Show Impact (%)',
                'duration_impact': 'Duration Impact (min)'
            }
        )

        fig.update_layout(height=350)

        return fig

    def create_capacity_heatmap(self, capacity_data: pd.DataFrame) -> Optional[go.Figure]:
        """
        Create capacity heatmap by hour and day.

        Args:
            capacity_data: DataFrame with day, hour, utilization columns

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or capacity_data.empty:
            return None

        # Pivot data
        pivot = capacity_data.pivot(index='hour', columns='day', values='utilization')

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,
            x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
            y=[f'{h:02d}:00' for h in pivot.index],
            colorscale='RdYlGn_r',
            zmin=0,
            zmax=100,
            colorbar_title="Utilization %"
        ))

        fig.update_layout(
            title="Capacity Heatmap",
            xaxis_title="Day of Week",
            yaxis_title="Hour",
            height=400
        )

        return fig

    def create_patient_flow_chart(self, flow_data: List[Dict]) -> Optional[go.Figure]:
        """
        Create patient flow visualization.

        Args:
            flow_data: List of {time, arrivals, departures} dicts

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE or not flow_data:
            return None

        df = pd.DataFrame(flow_data)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['arrivals'],
            mode='lines',
            name='Arrivals',
            fill='tozeroy',
            line=dict(color='#28a745')
        ))

        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['departures'],
            mode='lines',
            name='Departures',
            fill='tozeroy',
            line=dict(color='#dc3545')
        ))

        fig.update_layout(
            title="Patient Flow",
            xaxis_title="Time",
            yaxis_title="Patients",
            height=300,
            legend=dict(x=0, y=1)
        )

        return fig


# Example usage
if __name__ == "__main__":
    if PLOTLY_AVAILABLE:
        generator = ChartGenerator()

        # Create sample data
        from datetime import datetime, timedelta

        # Sample appointments for timeline
        class MockAppointment:
            def __init__(self, patient_id, chair_id, start, duration, priority):
                self.patient_id = patient_id
                self.chair_id = chair_id
                self.start_time = start
                self.duration = duration
                self.end_time = start + timedelta(minutes=duration)
                self.priority = priority

        today = datetime.now().replace(hour=0, minute=0, second=0)
        appointments = [
            MockAppointment('P001', 'WC-C01', today.replace(hour=9), 120, 1),
            MockAppointment('P002', 'WC-C01', today.replace(hour=12), 90, 2),
            MockAppointment('P003', 'WC-C02', today.replace(hour=10), 180, 2),
            MockAppointment('P004', 'WC-C02', today.replace(hour=14), 60, 3),
            MockAppointment('P005', 'WC-C03', today.replace(hour=9), 240, 1),
        ]

        # Generate charts
        timeline = generator.create_schedule_timeline(appointments, today)
        priority_chart = generator.create_priority_pie_chart(appointments)

        # Sample utilization data
        util_data = [
            {'date': today - timedelta(days=i), 'utilization': 0.75 + (i % 3) * 0.05}
            for i in range(7)
        ]
        util_chart = generator.create_utilization_chart(util_data)

        # Display
        print("Charts generated successfully!")
        print(f"Timeline: {type(timeline)}")
        print(f"Priority: {type(priority_chart)}")
        print(f"Utilization: {type(util_chart)}")

        # Optionally show in browser
        # timeline.show()
    else:
        print("Plotly not available. Install with: pip install plotly")
