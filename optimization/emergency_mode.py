"""
Emergency Mode Handler
======================

Handles scheduling during crisis and emergency modes.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OperatingMode,
    MODE_THRESHOLDS,
    PRIORITY_DEFINITIONS,
    get_logger
)
from .optimizer import Patient, Chair, ScheduledAppointment, ScheduleOptimizer

logger = get_logger(__name__)


@dataclass
class ModeAction:
    """Action to take based on operating mode"""
    action_type: str
    description: str
    affected_patients: List[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class EmergencySchedule:
    """Emergency schedule result"""
    mode: OperatingMode
    appointments: List[ScheduledAppointment]
    rescheduled: List[str]  # Patient IDs that need rescheduling
    cancelled: List[str]    # Patient IDs that are cancelled
    notifications: List[Dict]  # Notifications to send
    community_sites_activated: bool
    actions_taken: List[ModeAction]


class EmergencyModeHandler:
    """
    Handles scheduling during different operating modes.

    Implements escalating response strategies:
    - NORMAL: Standard scheduling
    - ELEVATED: Add buffers, send notifications
    - CRISIS: Prioritize P1-P3, reschedule P4
    - EMERGENCY: Critical patients only, activate community sites
    """

    # Mode-specific settings
    MODE_SETTINGS = {
        OperatingMode.NORMAL: {
            'buffer_multiplier': 1.0,
            'min_priority': 4,  # All priorities
            'notify_patients': False,
            'extend_hours': False,
            'community_sites': False
        },
        OperatingMode.ELEVATED: {
            'buffer_multiplier': 1.15,  # 15% extra time
            'min_priority': 4,
            'notify_patients': True,
            'extend_hours': False,
            'community_sites': False
        },
        OperatingMode.CRISIS: {
            'buffer_multiplier': 1.25,
            'min_priority': 3,  # P1-P3 only
            'notify_patients': True,
            'extend_hours': True,
            'community_sites': False
        },
        OperatingMode.EMERGENCY: {
            'buffer_multiplier': 1.5,
            'min_priority': 2,  # P1-P2 only (critical)
            'notify_patients': True,
            'extend_hours': True,
            'community_sites': True
        }
    }

    # Community backup sites
    COMMUNITY_SITES = [
        {'code': 'COMM1', 'name': 'Community Centre 1', 'chairs': 5, 'lat': 51.48, 'lon': -3.18},
        {'code': 'COMM2', 'name': 'Community Centre 2', 'chairs': 3, 'lat': 51.58, 'lon': -3.22},
    ]

    def __init__(self, optimizer: ScheduleOptimizer = None):
        """
        Initialize emergency mode handler.

        Args:
            optimizer: ScheduleOptimizer instance
        """
        self.optimizer = optimizer or ScheduleOptimizer()
        self.current_mode = OperatingMode.NORMAL
        self._mode_history: List[Tuple[datetime, OperatingMode]] = []

        logger.info("Emergency mode handler initialized")

    def set_mode(self, mode: OperatingMode, reason: str = ""):
        """
        Set operating mode.

        Args:
            mode: New operating mode
            reason: Reason for mode change
        """
        if mode != self.current_mode:
            old_mode = self.current_mode
            self.current_mode = mode
            self._mode_history.append((datetime.now(), mode))

            logger.warning(
                f"Operating mode changed: {old_mode.value} -> {mode.value}"
                f"{f' ({reason})' if reason else ''}"
            )

    def determine_mode(self, severity_score: float) -> OperatingMode:
        """
        Determine operating mode based on severity score.

        Args:
            severity_score: Combined event severity (0-1)

        Returns:
            Recommended OperatingMode
        """
        for mode, threshold in sorted(
            MODE_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if severity_score >= threshold:
                return mode

        return OperatingMode.NORMAL

    def create_emergency_schedule(self, patients: List[Patient],
                                  existing_schedule: List[ScheduledAppointment],
                                  date: datetime = None) -> EmergencySchedule:
        """
        Create schedule appropriate for current mode.

        Args:
            patients: Patients to schedule
            existing_schedule: Current appointments
            date: Date for scheduling

        Returns:
            EmergencySchedule object
        """
        date = date or datetime.now()
        settings = self.MODE_SETTINGS[self.current_mode]
        actions = []

        # Filter patients by priority
        min_priority = settings['min_priority']
        eligible_patients = [p for p in patients if p.priority <= min_priority]
        deferred_patients = [p for p in patients if p.priority > min_priority]

        if deferred_patients:
            actions.append(ModeAction(
                action_type='defer',
                description=f"Deferring {len(deferred_patients)} low-priority patients",
                affected_patients=[p.patient_id for p in deferred_patients],
                priority=3
            ))

        # Adjust durations with buffer
        buffer_mult = settings['buffer_multiplier']
        for patient in eligible_patients:
            patient.expected_duration = int(patient.expected_duration * buffer_mult)

        # Add community site chairs if emergency
        chairs = list(self.optimizer.chairs)
        if settings['community_sites']:
            community_chairs = self._create_community_chairs(date)
            chairs.extend(community_chairs)
            self.optimizer.set_chairs(chairs)

            actions.append(ModeAction(
                action_type='activate_sites',
                description=f"Activated {len(self.COMMUNITY_SITES)} community sites",
                priority=1
            ))

        # Identify appointments to reschedule
        rescheduled = []
        cancelled = []

        for apt in existing_schedule:
            # Find the patient's priority
            patient = next((p for p in patients if p.patient_id == apt.patient_id), None)
            if patient and patient.priority > min_priority:
                rescheduled.append(apt.patient_id)

        if rescheduled:
            actions.append(ModeAction(
                action_type='reschedule',
                description=f"Rescheduling {len(rescheduled)} appointments",
                affected_patients=rescheduled,
                priority=2
            ))

        # Run optimization
        result = self.optimizer.optimize(eligible_patients, date)

        # Generate notifications
        notifications = self._generate_notifications(
            settings, rescheduled, deferred_patients, date
        )

        # Create emergency schedule result
        return EmergencySchedule(
            mode=self.current_mode,
            appointments=result.appointments,
            rescheduled=rescheduled,
            cancelled=[p.patient_id for p in deferred_patients],
            notifications=notifications,
            community_sites_activated=settings['community_sites'],
            actions_taken=actions
        )

    def _create_community_chairs(self, date: datetime) -> List[Chair]:
        """Create chairs for community backup sites"""
        from config import OPERATING_HOURS
        start_hour, end_hour = OPERATING_HOURS

        chairs = []
        for site in self.COMMUNITY_SITES:
            for i in range(site['chairs']):
                chairs.append(Chair(
                    chair_id=f"{site['code']}-C{i+1:02d}",
                    site_code=site['code'],
                    is_recliner=False,
                    available_from=date.replace(hour=start_hour, minute=0),
                    available_until=date.replace(hour=end_hour, minute=0)
                ))

        return chairs

    def _generate_notifications(self, settings: Dict,
                                rescheduled: List[str],
                                deferred: List[Patient],
                                date: datetime) -> List[Dict]:
        """Generate notifications for affected patients"""
        notifications = []

        if not settings['notify_patients']:
            return notifications

        # Rescheduling notifications
        for patient_id in rescheduled:
            notifications.append({
                'type': 'reschedule',
                'patient_id': patient_id,
                'message': f"Your appointment on {date.strftime('%d/%m/%Y')} needs to be rescheduled due to operational constraints. Our team will contact you shortly.",
                'priority': 'high',
                'channel': ['sms', 'email']
            })

        # Deferral notifications
        for patient in deferred:
            notifications.append({
                'type': 'defer',
                'patient_id': patient.patient_id,
                'message': f"Due to current conditions, we need to reschedule your appointment. We will contact you to arrange a new time.",
                'priority': 'medium',
                'channel': ['sms', 'email']
            })

        # General advisory
        if self.current_mode in [OperatingMode.CRISIS, OperatingMode.EMERGENCY]:
            notifications.append({
                'type': 'advisory',
                'patient_id': 'all',
                'message': f"We are currently operating under {self.current_mode.value} conditions. Please check travel conditions before leaving and allow extra time.",
                'priority': 'high',
                'channel': ['sms', 'email', 'website']
            })

        return notifications

    def get_mode_recommendations(self, severity_score: float,
                                 affected_postcodes: List[str],
                                 event_count: int) -> Dict:
        """
        Get recommendations for current situation.

        Args:
            severity_score: Overall severity
            affected_postcodes: Postcodes affected by events
            event_count: Number of active events

        Returns:
            Dict with recommendations
        """
        recommended_mode = self.determine_mode(severity_score)
        settings = self.MODE_SETTINGS[recommended_mode]

        recommendations = {
            'recommended_mode': recommended_mode.value,
            'current_mode': self.current_mode.value,
            'mode_change_needed': recommended_mode != self.current_mode,
            'severity_score': round(severity_score, 2),
            'affected_postcodes': affected_postcodes,
            'active_events': event_count,
            'actions': []
        }

        # Generate recommendations
        if recommended_mode == OperatingMode.NORMAL:
            recommendations['summary'] = "Continue normal operations"

        elif recommended_mode == OperatingMode.ELEVATED:
            recommendations['summary'] = "Elevated conditions - add buffers and notify patients"
            recommendations['actions'] = [
                "Add 15% time buffer to all appointments",
                "Send travel advisory to patients in affected areas",
                "Monitor conditions closely"
            ]

        elif recommended_mode == OperatingMode.CRISIS:
            recommendations['summary'] = "Crisis mode - prioritize critical patients"
            recommendations['actions'] = [
                "Restrict scheduling to P1-P3 patients only",
                "Reschedule P4 appointments to next available date",
                "Contact all patients with updated information",
                "Add 25% time buffer to appointments",
                "Consider extending operating hours"
            ]

        elif recommended_mode == OperatingMode.EMERGENCY:
            recommendations['summary'] = "Emergency mode - critical care only"
            recommendations['actions'] = [
                "Only schedule P1-P2 (critical) patients",
                "Reschedule all other appointments",
                "Activate community backup sites",
                "Extend operating hours if safe",
                "Coordinate with emergency services",
                "Issue public advisory"
            ]

        # Postcode-specific recommendations
        if affected_postcodes:
            recommendations['postcode_actions'] = [
                f"Contact patients in {', '.join(affected_postcodes[:5])} about potential delays",
                "Offer rescheduling to patients in severely affected areas"
            ]

        return recommendations

    def get_mode_status(self) -> Dict:
        """Get current mode status"""
        settings = self.MODE_SETTINGS[self.current_mode]

        return {
            'current_mode': self.current_mode.value,
            'settings': {
                'buffer_multiplier': settings['buffer_multiplier'],
                'min_priority_level': settings['min_priority'],
                'patient_notifications': settings['notify_patients'],
                'extended_hours': settings['extend_hours'],
                'community_sites': settings['community_sites']
            },
            'mode_history': [
                {'time': t.isoformat(), 'mode': m.value}
                for t, m in self._mode_history[-10:]
            ],
            'priority_levels_accepted': [
                f"P{i}" for i in range(1, settings['min_priority'] + 1)
            ]
        }


# Example usage
if __name__ == "__main__":
    handler = EmergencyModeHandler()

    # Test mode determination
    test_severities = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("Mode Determination Test:")
    print("=" * 50)
    for severity in test_severities:
        mode = handler.determine_mode(severity)
        print(f"  Severity {severity:.1f} -> {mode.value}")

    # Test recommendations
    print("\nRecommendations for Crisis Scenario:")
    print("=" * 50)

    recommendations = handler.get_mode_recommendations(
        severity_score=0.65,
        affected_postcodes=['CF14', 'CF15', 'CF23'],
        event_count=3
    )

    print(f"Recommended Mode: {recommendations['recommended_mode']}")
    print(f"Summary: {recommendations['summary']}")
    print("\nActions:")
    for action in recommendations['actions']:
        print(f"  - {action}")

    # Test emergency schedule creation
    print("\n" + "=" * 50)
    print("Emergency Schedule Test:")

    handler.set_mode(OperatingMode.CRISIS, "Multiple traffic incidents")

    patients = [
        Patient('P001', priority=1, protocol='Urgent', expected_duration=60,
               postcode='CF14', earliest_time=datetime.now(),
               latest_time=datetime.now() + timedelta(hours=8)),
        Patient('P002', priority=2, protocol='Standard', expected_duration=90,
               postcode='CF15', earliest_time=datetime.now(),
               latest_time=datetime.now() + timedelta(hours=8)),
        Patient('P003', priority=4, protocol='Routine', expected_duration=60,
               postcode='CF23', earliest_time=datetime.now(),
               latest_time=datetime.now() + timedelta(hours=8)),
    ]

    result = handler.create_emergency_schedule(patients, [])

    print(f"\nMode: {result.mode.value}")
    print(f"Scheduled: {len(result.appointments)}")
    print(f"Rescheduled: {result.rescheduled}")
    print(f"Cancelled/Deferred: {result.cancelled}")
    print(f"Community Sites: {result.community_sites_activated}")
    print(f"\nActions Taken:")
    for action in result.actions_taken:
        print(f"  [{action.action_type}] {action.description}")
    print(f"\nNotifications: {len(result.notifications)}")
