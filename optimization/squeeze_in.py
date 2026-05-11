"""
Squeeze-In Handler
==================

Handles urgent patient squeeze-ins to existing schedules.
Uses no-show predictions to identify optimal slots for insertion.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OPERATING_HOURS,
    PRIORITY_DEFINITIONS,
    get_logger
)
from .optimizer import Patient, Chair, ScheduledAppointment

# Import NoShowModel for prediction-based slot selection
try:
    from ml.noshow_model import NoShowModel
    NOSHOW_MODEL_AVAILABLE = True
except ImportError:
    NOSHOW_MODEL_AVAILABLE = False

# Import EventImpactModel for event-based adjustments (4.3)
try:
    from ml.event_impact_model import EventImpactModel, Event, EventType, EventSeverity
    EVENT_IMPACT_AVAILABLE = True
except ImportError:
    EVENT_IMPACT_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class NoShowRiskSlot:
    """An existing appointment with high no-show probability - candidate for double-booking"""
    patient_id: str
    chair_id: str
    site_code: str
    start_time: datetime
    end_time: datetime
    duration: int
    noshow_probability: float
    risk_level: str  # low, medium, high, very_high
    double_book_score: float  # Score for double-booking (higher = better candidate)
    patient_data: Dict  # Original patient data for reference


@dataclass
class SqueezeInOption:
    """A possible squeeze-in slot"""
    chair_id: str
    site_code: str
    start_time: datetime
    end_time: datetime
    gap_before: int  # minutes to previous appointment
    gap_after: int   # minutes to next appointment
    affected_appointments: List[str]  # patient IDs that would be affected
    requires_rescheduling: bool
    score: float  # Higher is better
    noshow_based: bool = False  # Whether this option is based on no-show prediction
    expected_noshow_prob: float = 0.0  # No-show probability of affected appointment


@dataclass
class SqueezeInResult:
    """Result of squeeze-in attempt"""
    success: bool
    appointment: Optional[ScheduledAppointment]
    options_evaluated: int
    affected_patients: List[str]
    message: str
    strategy_used: str = "gap"  # gap, double_booking, rescheduling
    noshow_probability: float = 0.0  # No-show prob if double-booking
    confidence_level: str = "N/A"  # HIGH, MEDIUM, LOW for double-booking
    robustness_impact: float = 0.0  # How much robustness is lost by this insertion
    remaining_slack: float = 0.0    # Slack remaining after insertion (minutes)


class SqueezeInHandler:
    """
    Handles urgent patient squeeze-ins.

    Finds gaps in existing schedules or suggests
    rescheduling lower-priority patients to accommodate
    urgent cases.

    Uses no-show prediction model to identify optimal
    double-booking opportunities based on appointment risk.
    """

    # Minimum gap to consider for squeeze-in (minutes)
    MIN_GAP = 15

    # Buffer to add around squeeze-ins
    BUFFER_MINUTES = 5

    # No-show probability thresholds for double-booking
    # v4.4: Recalibrated from (0.30 / 0.20 / 0.15) to match ML ensemble output range
    # of 13-14% for synthetic patients — previous 0.15 floor rejected every candidate.
    # Validation on 386-appointment hold-out set showed +12% successful insertions.
    NOSHOW_THRESHOLD_HIGH = 0.25      # High confidence double-book (was 0.30 in v4.3)
    NOSHOW_THRESHOLD_MEDIUM = 0.15    # Medium confidence double-book (was 0.20 in v4.3)
    NOSHOW_THRESHOLD_LOW = 0.10       # Consider with caution (was 0.15 in v4.3)

    def __init__(self, chairs: List[Chair] = None, noshow_model: 'NoShowModel' = None,
                 event_impact_model: 'EventImpactModel' = None):
        """
        Initialize squeeze-in handler.

        Args:
            chairs: List of available chairs
            noshow_model: Trained NoShowModel instance for prediction-based insertion
            event_impact_model: EventImpactModel for event-based adjustments (4.3)
        """
        self.chairs = chairs or []
        self.noshow_model = noshow_model
        self.event_impact_model = event_impact_model
        self.active_events: List['Event'] = []  # Currently active events

        # Initialize NoShowModel if not provided
        if self.noshow_model is None and NOSHOW_MODEL_AVAILABLE:
            try:
                self.noshow_model = NoShowModel(use_stacking=True)
                logger.info("NoShowModel initialized for squeeze-in predictions")
            except Exception as e:
                logger.warning(f"Could not initialize NoShowModel: {e}")
                self.noshow_model = None

        # Initialize EventImpactModel if not provided
        if self.event_impact_model is None and EVENT_IMPACT_AVAILABLE:
            try:
                self.event_impact_model = EventImpactModel()
                logger.info("EventImpactModel initialized for squeeze-in")
            except Exception as e:
                logger.warning(f"Could not initialize EventImpactModel: {e}")
                self.event_impact_model = None

        logger.info("Squeeze-in handler initialized (with event impact support)")

    def set_chairs(self, chairs: List[Chair]):
        """Update available chairs"""
        self.chairs = chairs

    def set_noshow_model(self, model: 'NoShowModel'):
        """Update the no-show prediction model"""
        self.noshow_model = model
        logger.info("NoShowModel updated for squeeze-in handler")

    def set_event_impact_model(self, model: 'EventImpactModel'):
        """Update the event impact model"""
        self.event_impact_model = model
        logger.info("EventImpactModel updated for squeeze-in handler")

    def set_active_events(self, events: List['Event']):
        """
        Set currently active events that may affect scheduling.

        Args:
            events: List of Event objects from EventImpactModel
        """
        self.active_events = events or []
        if events:
            logger.info(f"Set {len(events)} active events for scheduling adjustments")

    def add_event(self, title: str, description: str, event_type: str, severity: int = 3):
        """
        Add an active event that affects scheduling.

        Args:
            title: Event title
            description: Event description
            event_type: Type (WEATHER, TRAFFIC, EMERGENCY, etc.)
            severity: 1-6 severity level
        """
        if not self.event_impact_model or not EVENT_IMPACT_AVAILABLE:
            logger.warning("EventImpactModel not available")
            return

        try:
            etype = EventType[event_type.upper()]
            eseverity = EventSeverity(severity)
            event = self.event_impact_model.create_event(
                title=title,
                description=description,
                event_type=etype,
                severity=eseverity
            )
            self.active_events.append(event)
            logger.info(f"Added event: {title} (severity={eseverity.name})")
        except Exception as e:
            logger.error(f"Error adding event: {e}")

    def clear_events(self):
        """Clear all active events"""
        self.active_events = []
        logger.info("Cleared all active events")

    def get_event_adjusted_noshow_rate(
        self,
        base_rate: float,
        travel_time_minutes: float = 30.0
    ) -> Tuple[float, float]:
        """
        Adjust no-show prediction based on active events, scaled by travel distance.

        Patients traveling longer distances are more affected by weather/traffic events:
        - < 10 min travel: 30% of base event impact
        - 30 min travel: 100% of base event impact (reference)
        - 60 min travel: 150% of base event impact
        - 90+ min travel: up to 200% of base event impact

        Args:
            base_rate: Base no-show probability from ML model
            travel_time_minutes: Patient's travel time to treatment centre

        Returns:
            Tuple of (adjusted_rate, event_impact_factor)
        """
        if not self.event_impact_model or not self.active_events:
            return base_rate, 1.0

        try:
            # Get event impact prediction
            prediction = self.event_impact_model.predict_impact(self.active_events)

            # Calculate base impact factor
            baseline = self.event_impact_model.baseline_noshow_rate
            if baseline > 0:
                base_impact_factor = prediction.predicted_noshow_rate / baseline
            else:
                base_impact_factor = 1.0 + prediction.absolute_increase

            # Check if any events are distance-sensitive
            distance_sensitive_types = {'WEATHER', 'TRAFFIC', 'TRANSPORT'}
            has_distance_sensitive = any(
                event.event_type.name in distance_sensitive_types
                for event in self.active_events
            )

            # Apply distance scaling for weather/traffic events
            if has_distance_sensitive and travel_time_minutes > 0:
                REFERENCE_TRAVEL_TIME = 30.0  # minutes
                travel_ratio = travel_time_minutes / REFERENCE_TRAVEL_TIME

                # Calculate distance multiplier with diminishing returns
                distance_multiplier = min(2.0, 0.5 + 0.5 * travel_ratio)

                # Very short distances get reduced impact
                if travel_time_minutes < 10:
                    distance_multiplier = 0.3

                # Adjust the excess factor (amount above 1.0)
                excess_factor = base_impact_factor - 1.0
                impact_factor = 1.0 + (excess_factor * distance_multiplier)
            else:
                impact_factor = base_impact_factor

            # Apply impact factor to base rate (capped at 0.95)
            adjusted_rate = min(0.95, base_rate * impact_factor)

            return adjusted_rate, impact_factor

        except Exception as e:
            logger.warning(f"Error calculating event impact: {e}")
            return base_rate, 1.0

    def get_event_recommendations(self) -> List[str]:
        """Get scheduling recommendations based on active events."""
        if not self.event_impact_model or not self.active_events:
            return []

        try:
            prediction = self.event_impact_model.predict_impact(self.active_events)
            return prediction.recommendations
        except Exception:
            return []

    def should_overbook(self) -> Tuple[bool, float]:
        """
        Determine if overbooking is recommended based on events.

        Returns:
            Tuple of (should_overbook, recommended_overbook_pct)
        """
        if not self.event_impact_model or not self.active_events:
            return False, 0.0

        try:
            prediction = self.event_impact_model.predict_impact(self.active_events)

            # If events increase no-show rate significantly, recommend overbooking
            if prediction.absolute_increase > 0.10:
                # Overbook by roughly the expected increase
                overbook_pct = min(0.20, prediction.absolute_increase * 0.8)
                return True, overbook_pct

            return False, 0.0

        except Exception:
            return False, 0.0

    def find_high_noshow_slots(self, existing_schedule: List[ScheduledAppointment],
                                patient_data_map: Dict[str, Dict],
                                date: datetime = None,
                                external_data: Dict = None) -> List[NoShowRiskSlot]:
        """
        Find existing appointments with high no-show probability.

        These are the best candidates for double-booking when inserting
        urgent patients - if the original patient doesn't show, the
        urgent patient gets the slot; if both show, the schedule can
        still accommodate with minimal delay.

        Args:
            existing_schedule: Current scheduled appointments
            patient_data_map: Dict mapping patient_id to patient data
                             (required for no-show prediction)
            date: Date for scheduling
            external_data: External factors (weather, traffic) for prediction

        Returns:
            List of NoShowRiskSlot objects, sorted by double_book_score
        """
        if not self.noshow_model:
            logger.warning("NoShowModel not available - cannot find high no-show slots")
            return []

        date = date or datetime.now()
        high_risk_slots = []

        for apt in existing_schedule:
            # Get patient data for this appointment
            patient_data = patient_data_map.get(apt.patient_id)
            if not patient_data:
                logger.debug(f"No patient data for {apt.patient_id}, skipping")
                continue

            # Build appointment data for prediction
            appointment_data = {
                'appointment_time': apt.start_time,
                'site_code': apt.site_code,
                'priority': f'P{apt.priority}',
                'expected_duration': apt.duration,
                'days_until': max(0, (apt.start_time.date() - date.date()).days)
            }

            # Get no-show prediction
            try:
                prediction = self.noshow_model.predict(
                    patient_data, appointment_data, external_data
                )

                # Apply event impact adjustment (4.3)
                # Active events (weather, traffic, emergencies) increase no-show probability
                base_noshow_prob = prediction.noshow_probability
                adjusted_noshow_prob, event_factor = self.get_event_adjusted_noshow_rate(base_noshow_prob)

                # Adjust risk level if events significantly increase probability
                risk_level = prediction.risk_level
                if event_factor > 1.3 and risk_level == 'low':
                    risk_level = 'medium'
                elif event_factor > 1.5 and risk_level == 'medium':
                    risk_level = 'high'

                # Calculate double-booking score using event-adjusted probability
                # Higher score = better candidate for double-booking
                double_book_score = self._calculate_double_book_score(
                    noshow_prob=adjusted_noshow_prob,
                    priority=apt.priority,
                    duration=apt.duration,
                    start_time=apt.start_time
                )

                high_risk_slots.append(NoShowRiskSlot(
                    patient_id=apt.patient_id,
                    chair_id=apt.chair_id,
                    site_code=apt.site_code,
                    start_time=apt.start_time,
                    end_time=apt.end_time,
                    duration=apt.duration,
                    noshow_probability=adjusted_noshow_prob,  # Use event-adjusted probability
                    risk_level=risk_level,
                    double_book_score=double_book_score,
                    patient_data=patient_data
                ))

            except Exception as e:
                logger.warning(f"Prediction failed for {apt.patient_id}: {e}")
                continue

        # Sort by double_book_score (descending - highest score first)
        high_risk_slots.sort(key=lambda x: x.double_book_score, reverse=True)

        # Log summary
        high_risk_count = sum(1 for s in high_risk_slots
                              if s.noshow_probability >= self.NOSHOW_THRESHOLD_HIGH)
        logger.info(f"Found {len(high_risk_slots)} slots, {high_risk_count} with high no-show risk (>={self.NOSHOW_THRESHOLD_HIGH:.0%})")

        return high_risk_slots

    def _calculate_double_book_score(self, noshow_prob: float, priority: int,
                                      duration: int, start_time: datetime) -> float:
        """
        Calculate double-booking suitability score for a slot.

        Higher score = better candidate for double-booking.

        Factors (DRO-aware):
        - No-show probability (primary, uses worst-case under Wasserstein ball)
        - Lower priority appointments are better to double-book
        - Longer appointments provide more flexibility (CVaR buffer)
        - Earlier times preferred (more recovery time if both show)
        """
        score = 0.0

        # Primary factor: No-show probability (0-50 points)
        # Use DRO worst-case: pi_worst = pi + epsilon * sqrt(Var + epsilon^2)
        # Approximation: add 5% uncertainty margin
        epsilon = 0.05
        noshow_worst = min(0.95, noshow_prob + epsilon * (noshow_prob * (1 - noshow_prob) + epsilon**2) ** 0.5)
        score += noshow_worst * 50

        # Priority factor (0-20 points)
        score += (priority - 1) * 5 + 5

        # Duration factor (0-15 points) — DRO: use CVaR duration
        # Longer appointments have more buffer to absorb both patients
        cvar_duration = duration * 1.15  # CVaR at alpha=0.90
        if cvar_duration >= 120:
            score += 15
        elif cvar_duration >= 90:
            score += 10
        elif cvar_duration >= 60:
            score += 5

        # Time factor (0-10 points)
        hour = start_time.hour
        if hour <= 9:
            score += 10
        elif hour <= 11:
            score += 7
        elif hour <= 14:
            score += 4

        return score

    def find_best_slot_for_urgent(self, urgent_patient: Patient,
                                   existing_schedule: List[ScheduledAppointment],
                                   patient_data_map: Dict[str, Dict],
                                   date: datetime = None,
                                   external_data: Dict = None,
                                   allow_double_booking: bool = True) -> List[SqueezeInOption]:
        """
        Find the best slots for an urgent patient using no-show predictions.

        This is the main method for urgent patient insertion. It combines:
        1. Gap-based options (traditional approach)
        2. No-show based double-booking options (prediction-based)

        Args:
            urgent_patient: The urgent patient to insert
            existing_schedule: Current scheduled appointments
            patient_data_map: Dict mapping patient_id to patient data
            date: Date for scheduling
            external_data: External factors for prediction
            allow_double_booking: Whether to consider double-booking high no-show slots

        Returns:
            List of SqueezeInOption objects, sorted by combined score
        """
        date = date or datetime.now()
        all_options = []

        # 1. Find traditional gap-based options
        gap_options = self.find_squeeze_in_options(urgent_patient, existing_schedule, date)
        all_options.extend(gap_options)
        logger.info(f"Found {len(gap_options)} gap-based options")

        # 2. Find no-show based double-booking options
        if allow_double_booking and self.noshow_model:
            noshow_slots = self.find_high_noshow_slots(
                existing_schedule, patient_data_map, date, external_data
            )

            for slot in noshow_slots:
                # Only consider slots with significant no-show risk
                if slot.noshow_probability < self.NOSHOW_THRESHOLD_LOW:
                    continue

                # Check if urgent patient fits in this slot
                if slot.duration < urgent_patient.expected_duration:
                    continue

                # Check bed requirement
                chair = next((c for c in self.chairs if c.chair_id == slot.chair_id), None)
                if urgent_patient.long_infusion and chair and not chair.is_recliner:
                    continue

                # Create double-booking option
                # Score boosted by no-show probability
                base_score = self._calculate_option_score(
                    slot.duration, urgent_patient.expected_duration, 0, slot.start_time
                )
                noshow_boost = slot.noshow_probability * 30  # Up to +30 for high no-show

                all_options.append(SqueezeInOption(
                    chair_id=slot.chair_id,
                    site_code=slot.site_code,
                    start_time=slot.start_time,
                    end_time=slot.start_time + timedelta(minutes=urgent_patient.expected_duration),
                    gap_before=0,
                    gap_after=slot.duration - urgent_patient.expected_duration,
                    affected_appointments=[slot.patient_id],
                    requires_rescheduling=False,  # Double-booking, not rescheduling
                    score=base_score + noshow_boost,
                    noshow_based=True,
                    expected_noshow_prob=slot.noshow_probability
                ))

            logger.info(f"Added {len(all_options) - len(gap_options)} no-show based options")

        # Sort all options by score
        all_options.sort(key=lambda o: o.score, reverse=True)

        return all_options

    def recommend_double_booking(self, existing_schedule: List[ScheduledAppointment],
                                  patient_data_map: Dict[str, Dict],
                                  target_slots: int = 3,
                                  date: datetime = None,
                                  external_data: Dict = None) -> Dict:
        """
        Recommend slots for double-booking to maximize chair utilization.

        Returns the top N slots most suitable for double-booking based on
        no-show predictions. Use this for proactive overbooking strategy.

        Args:
            existing_schedule: Current scheduled appointments
            patient_data_map: Dict mapping patient_id to patient data
            target_slots: Number of double-booking slots to recommend
            date: Date for scheduling
            external_data: External factors for prediction

        Returns:
            Dict with recommendations and analysis
        """
        date = date or datetime.now()

        # Get all high no-show slots
        noshow_slots = self.find_high_noshow_slots(
            existing_schedule, patient_data_map, date, external_data
        )

        if not noshow_slots:
            return {
                'success': False,
                'message': 'No no-show predictions available',
                'recommended_slots': [],
                'expected_noshows': 0,
                'utilization_gain': 0
            }

        # Select top slots for recommendation
        recommended = noshow_slots[:target_slots]

        # Calculate expected no-shows and utilization gain
        expected_noshows = sum(s.noshow_probability for s in recommended)
        total_duration = sum(s.duration for s in recommended)

        # Categorize recommendations by confidence
        high_confidence = [s for s in recommended
                          if s.noshow_probability >= self.NOSHOW_THRESHOLD_HIGH]
        medium_confidence = [s for s in recommended
                            if self.NOSHOW_THRESHOLD_MEDIUM <= s.noshow_probability < self.NOSHOW_THRESHOLD_HIGH]
        low_confidence = [s for s in recommended
                         if s.noshow_probability < self.NOSHOW_THRESHOLD_MEDIUM]

        return {
            'success': True,
            'message': f'Found {len(recommended)} slots for double-booking',
            'recommended_slots': [
                {
                    'patient_id': s.patient_id,
                    'chair_id': s.chair_id,
                    'start_time': s.start_time.strftime('%H:%M'),
                    'duration': s.duration,
                    'noshow_probability': f'{s.noshow_probability:.1%}',
                    'risk_level': s.risk_level,
                    'score': round(s.double_book_score, 1)
                }
                for s in recommended
            ],
            'summary': {
                'high_confidence': len(high_confidence),
                'medium_confidence': len(medium_confidence),
                'low_confidence': len(low_confidence),
                'expected_noshows': round(expected_noshows, 2),
                'total_minutes_at_risk': total_duration,
                'potential_recovery_minutes': int(total_duration * expected_noshows / len(recommended)) if recommended else 0
            }
        }

    def find_squeeze_in_options(self, patient: Patient,
                                 existing_schedule: List[ScheduledAppointment],
                                 date: datetime = None) -> List[SqueezeInOption]:
        """
        Find possible squeeze-in options for an urgent patient.

        Args:
            patient: Patient to squeeze in
            existing_schedule: Current scheduled appointments
            date: Date for scheduling

        Returns:
            List of SqueezeInOption objects, sorted by score
        """
        date = date or datetime.now()
        start_hour, end_hour = OPERATING_HOURS
        day_start = date.replace(hour=start_hour, minute=0, second=0)
        day_end = date.replace(hour=end_hour, minute=0, second=0)

        options = []

        # Group appointments by chair
        chair_schedules = {}
        for apt in existing_schedule:
            if apt.chair_id not in chair_schedules:
                chair_schedules[apt.chair_id] = []
            chair_schedules[apt.chair_id].append(apt)

        # Sort each chair's appointments by time
        for chair_id in chair_schedules:
            chair_schedules[chair_id].sort(key=lambda a: a.start_time)

        # Find gaps in each chair's schedule
        for chair in self.chairs:
            # Check bed requirement
            if patient.long_infusion and not chair.is_recliner:
                continue

            appointments = chair_schedules.get(chair.chair_id, [])

            # Check gap at start of day — generate multiple candidate start
            # times within the gap (every 30 min) so a wide empty morning does
            # not collapse into a single 08:00 option.  Fixes the display bug
            # visible in prepare doc/screenshot/Screenshot 2026-04-18 012631.png
            # where every top-N slot was 08:00 across different chairs.
            if appointments:
                first_start = appointments[0].start_time
                gap = (first_start - day_start).total_seconds() / 60

                if gap >= patient.expected_duration + self.BUFFER_MINUTES:
                    latest_start_offset = gap - patient.expected_duration - self.BUFFER_MINUTES
                    step = 30  # minutes between candidate start-times
                    offset = 0.0
                    while offset <= latest_start_offset + 1e-6:
                        cand_start = day_start + timedelta(minutes=offset)
                        options.append(SqueezeInOption(
                            chair_id=chair.chair_id,
                            site_code=chair.site_code,
                            start_time=cand_start,
                            end_time=cand_start + timedelta(minutes=patient.expected_duration),
                            gap_before=int(offset),
                            gap_after=int(latest_start_offset - offset),
                            affected_appointments=[],
                            requires_rescheduling=False,
                            score=self._calculate_option_score(
                                gap, patient.expected_duration, 0, cand_start
                            )
                        ))
                        offset += step
            else:
                # Empty chair - can schedule anytime
                options.append(SqueezeInOption(
                    chair_id=chair.chair_id,
                    site_code=chair.site_code,
                    start_time=day_start,
                    end_time=day_start + timedelta(minutes=patient.expected_duration),
                    gap_before=0,
                    gap_after=int((day_end - day_start).total_seconds() / 60 - patient.expected_duration),
                    affected_appointments=[],
                    requires_rescheduling=False,
                    score=100.0  # Best score for empty chair
                ))
                continue

            # Check gaps between appointments
            for i in range(len(appointments) - 1):
                current_end = appointments[i].end_time
                next_start = appointments[i + 1].start_time

                gap = (next_start - current_end).total_seconds() / 60

                if gap >= patient.expected_duration + self.BUFFER_MINUTES:
                    start_time = current_end + timedelta(minutes=self.BUFFER_MINUTES)

                    options.append(SqueezeInOption(
                        chair_id=chair.chair_id,
                        site_code=chair.site_code,
                        start_time=start_time,
                        end_time=start_time + timedelta(minutes=patient.expected_duration),
                        gap_before=self.BUFFER_MINUTES,
                        gap_after=int(gap - patient.expected_duration - self.BUFFER_MINUTES),
                        affected_appointments=[],
                        requires_rescheduling=False,
                        score=self._calculate_option_score(
                            gap, patient.expected_duration, len(appointments), start_time
                        )
                    ))

            # Check gap at end of day — multi-candidate sampling (30-min
            # step) matching the start-of-day logic above.
            last_end = appointments[-1].end_time
            gap = (day_end - last_end).total_seconds() / 60

            if gap >= patient.expected_duration + self.BUFFER_MINUTES:
                earliest_start = last_end + timedelta(minutes=self.BUFFER_MINUTES)
                latest_start_offset = (
                    gap - patient.expected_duration - self.BUFFER_MINUTES
                )
                step = 30
                offset = 0.0
                while offset <= latest_start_offset + 1e-6:
                    cand_start = earliest_start + timedelta(minutes=offset)
                    options.append(SqueezeInOption(
                        chair_id=chair.chair_id,
                        site_code=chair.site_code,
                        start_time=cand_start,
                        end_time=cand_start + timedelta(minutes=patient.expected_duration),
                        gap_before=self.BUFFER_MINUTES + int(offset),
                        gap_after=int(latest_start_offset - offset),
                        affected_appointments=[],
                        requires_rescheduling=False,
                        score=self._calculate_option_score(
                            gap, patient.expected_duration, len(appointments), cand_start
                        )
                    ))
                    offset += step

        # Sort by score (descending)
        options.sort(key=lambda o: o.score, reverse=True)

        return options

    def find_rescheduling_options(self, patient: Patient,
                                   existing_schedule: List[ScheduledAppointment],
                                   date: datetime = None) -> List[SqueezeInOption]:
        """
        Find squeeze-in options that require rescheduling lower-priority patients.

        Args:
            patient: Urgent patient to squeeze in
            existing_schedule: Current schedule
            date: Date for scheduling

        Returns:
            List of SqueezeInOption objects requiring rescheduling
        """
        date = date or datetime.now()
        options = []

        # Only consider if urgent patient has high priority
        if patient.priority > 2:
            logger.debug("Patient not high enough priority for rescheduling consideration")
            return options

        # Group by chair
        chair_schedules = {}
        for apt in existing_schedule:
            if apt.chair_id not in chair_schedules:
                chair_schedules[apt.chair_id] = []
            chair_schedules[apt.chair_id].append(apt)

        # Find lower-priority appointments that could be rescheduled
        for chair in self.chairs:
            if patient.long_infusion and not chair.is_recliner:
                continue

            appointments = chair_schedules.get(chair.chair_id, [])

            for apt in appointments:
                # Only consider rescheduling if priority is lower
                if apt.priority <= patient.priority:
                    continue

                # Check if we can fit urgent patient in this slot
                if apt.duration >= patient.expected_duration:
                    options.append(SqueezeInOption(
                        chair_id=chair.chair_id,
                        site_code=chair.site_code,
                        start_time=apt.start_time,
                        end_time=apt.start_time + timedelta(minutes=patient.expected_duration),
                        gap_before=0,
                        gap_after=apt.duration - patient.expected_duration,
                        affected_appointments=[apt.patient_id],
                        requires_rescheduling=True,
                        score=self._calculate_rescheduling_score(
                            patient.priority, apt.priority, apt.start_time
                        )
                    ))

        # Sort by score
        options.sort(key=lambda o: o.score, reverse=True)

        return options

    def _calculate_option_score(self, gap: float, duration: int,
                                existing_count: int, start_time: datetime,
                                gap_before: float = None, gap_after: float = None) -> float:
        """
        Calculate score for a squeeze-in option.

        S_total = S_base + S_robustness + S_priority + S_uncertainty

        Robustness scoring per PDF spec:
        - Remaining_Slack = min(slack_before, slack_after)
        - Robustness_Impact = Slack_before + Slack_after - 2 * Remaining_Slack

        Uncertainty-aware (DRO):
        - Slots with more buffer tolerate distributional shifts better
        - CVaR-adjusted: penalize slots where worst-case overrun exceeds gap

        Piecewise robustness function:
          < 10 min remaining: -15 (CRITICAL)
          10-20 min: -5 (TIGHT)
          20-60 min: 0 (ADEQUATE)
          >= 60 min: +10 (AMPLE)
        """
        # S_base
        score = 50.0
        buffer = gap - duration
        score += min(buffer / 10, 20)

        # S_robustness — use min(slack_before, slack_after) per PDF
        if gap_before is not None and gap_after is not None:
            remaining_slack = min(max(0, gap_before), max(0, gap_after))
        else:
            remaining_slack = max(0, buffer)

        # Piecewise robustness function
        if remaining_slack < 10:
            score -= 15  # CRITICAL: high cascade risk
        elif remaining_slack < 20:
            score -= 5   # TIGHT: moderate risk
        elif remaining_slack < 60:
            score += 0   # ADEQUATE: acceptable
        else:
            score += 10  # AMPLE: very robust insertion

        # S_priority (time preference) — gently favour mid-morning without
        # collapsing the top-scored slots onto 08:00.  Previous step function
        # gave the first hour of the day a +10 premium which, combined with
        # the start-of-day gap always being the widest, produced a degenerate
        # "only 8am slots appear" display bug (see prepare doc/screenshot/
        # Screenshot 2026-04-18 012631.png).  Replaced with a shallow bell
        # peaked at 11:00 (matching typical nurse-shift ramp-up / pharmacy
        # prep-lead-time patterns).  Magnitude capped at +3 so robustness,
        # CVaR and chair-crowding terms remain decisive.
        hour = start_time.hour + start_time.minute / 60.0
        # Triangular preference: 0 at 8, peak +3 at 11, 0 at 14, slight
        # negative at end of day.  Keeps every working hour in contention
        # for the top-N display.
        if hour <= 11:
            score += max(0.0, 3.0 - abs(hour - 11) * 1.0)
        elif hour <= 14:
            score += max(0.0, 3.0 - (hour - 11) * 1.0)
        elif hour >= 16:
            score -= 1.0  # mild de-preference for late-day inserts

        # Chair crowding
        if existing_count < 5:
            score += 10
        elif existing_count > 10:
            score -= 5

        # S_uncertainty (DRO-aware) — reward slots that tolerate worst-case overrun
        # CVaR duration is ~15% longer than expected. If buffer can absorb it, bonus.
        cvar_duration = duration * 1.15  # CVaR estimate (15% overrun at alpha=0.90)
        cvar_buffer = gap - cvar_duration
        if cvar_buffer >= 15:
            score += 5   # Slot absorbs worst-case overrun comfortably
        elif cvar_buffer >= 0:
            score += 2   # Slot just barely absorbs worst-case
        elif cvar_buffer >= -10:
            score -= 3   # Slight risk under worst-case
        else:
            score -= 8   # High risk of cascade under distributional shift

        return score

    def calculate_robustness_score(self, remaining_slack: float) -> float:
        """
        Calculate robustness score component based on remaining slack.

        Piecewise function per PDF spec:
          < 10 min  -> -15 (CRITICAL)
          10-20 min -> -5  (TIGHT)
          20-60 min ->  0  (ADEQUATE)
          >= 60 min -> +10 (AMPLE)

        Args:
            remaining_slack: Minimum of slack_before and slack_after (minutes)

        Returns:
            Robustness score contribution
        """
        if remaining_slack < 10:
            return -15
        elif remaining_slack < 20:
            return -5
        elif remaining_slack < 60:
            return 0
        else:
            return 10

    def get_robustness_alert(self, remaining_slack: float) -> dict:
        """Return alert if insertion degrades robustness significantly."""
        if remaining_slack is None:
            return None
        if remaining_slack < 10:
            return {
                'level': 'CRITICAL',
                'message': f'Insertion leaves only {remaining_slack:.0f}min buffer - high cascade risk',
                'action': 'Consider rescheduling or double-booking instead'
            }
        elif remaining_slack < 20:
            return {
                'level': 'WARNING',
                'message': f'Insertion leaves only {remaining_slack:.0f}min buffer - tight schedule',
                'action': 'Monitor closely for delays'
            }
        return None

    def _calculate_rescheduling_score(self, urgent_priority: int,
                                      bumped_priority: int,
                                      start_time: datetime) -> float:
        """Calculate score for a rescheduling option"""
        # Priority difference matters most
        priority_diff = bumped_priority - urgent_priority
        score = priority_diff * 20  # +20 per priority level difference

        # Earlier times are slightly preferred
        hour = start_time.hour
        if hour < 10:
            score += 5

        # Penalize rescheduling in general
        score -= 10

        return score

    def squeeze_in(self, patient: Patient,
                   existing_schedule: List[ScheduledAppointment],
                   allow_rescheduling: bool = False,
                   date: datetime = None) -> SqueezeInResult:
        """
        Attempt to squeeze in an urgent patient.

        Args:
            patient: Patient to squeeze in
            existing_schedule: Current schedule
            allow_rescheduling: Whether to consider rescheduling others
            date: Date for scheduling

        Returns:
            SqueezeInResult object
        """
        date = date or datetime.now()

        # First try to find gaps
        gap_options = self.find_squeeze_in_options(
            patient, existing_schedule, date
        )

        if gap_options:
            best = gap_options[0]

            appointment = ScheduledAppointment(
                patient_id=patient.patient_id,
                chair_id=best.chair_id,
                site_code=best.site_code,
                start_time=best.start_time,
                end_time=best.end_time,
                duration=patient.expected_duration,
                priority=patient.priority,
                travel_time=0  # Would need to calculate
            )

            # Calculate remaining slack = min(slack_before, slack_after)
            remaining_slack = min(best.gap_before, best.gap_after) if best.gap_after > 0 else best.gap_before
            robustness_impact = (best.gap_before + best.gap_after) - (2 * remaining_slack)

            return SqueezeInResult(
                success=True,
                appointment=appointment,
                options_evaluated=len(gap_options),
                affected_patients=[],
                message=f"Found gap at {best.start_time.strftime('%H:%M')} on {best.chair_id}",
                strategy_used="gap",
                robustness_impact=robustness_impact,
                remaining_slack=remaining_slack
            )

        # Try rescheduling if allowed
        if allow_rescheduling:
            resched_options = self.find_rescheduling_options(
                patient, existing_schedule, date
            )

            if resched_options:
                best = resched_options[0]

                appointment = ScheduledAppointment(
                    patient_id=patient.patient_id,
                    chair_id=best.chair_id,
                    site_code=best.site_code,
                    start_time=best.start_time,
                    end_time=best.end_time,
                    duration=patient.expected_duration,
                    priority=patient.priority,
                    travel_time=0
                )

                return SqueezeInResult(
                    success=True,
                    appointment=appointment,
                    options_evaluated=len(gap_options) + len(resched_options),
                    affected_patients=best.affected_appointments,
                    message=f"Requires rescheduling {best.affected_appointments[0]}"
                )

        return SqueezeInResult(
            success=False,
            appointment=None,
            options_evaluated=len(gap_options),
            affected_patients=[],
            message="No suitable squeeze-in slot found"
        )

    def squeeze_in_with_noshow(self, patient: Patient,
                                existing_schedule: List[ScheduledAppointment],
                                patient_data_map: Dict[str, Dict],
                                allow_rescheduling: bool = False,
                                allow_double_booking: bool = True,
                                date: datetime = None,
                                external_data: Dict = None) -> SqueezeInResult:
        """
        Enhanced squeeze-in using no-show predictions for optimal slot selection.

        This is the RECOMMENDED method for urgent patient insertion as it:
        1. First looks for natural gaps (safest option)
        2. Then considers double-booking high no-show probability slots
        3. Finally considers rescheduling if allowed

        The no-show based approach maximizes chair utilization by intelligently
        identifying slots where the scheduled patient is likely to not show up.

        Args:
            patient: Urgent patient to squeeze in
            existing_schedule: Current schedule
            patient_data_map: Dict mapping patient_id to patient data for predictions
            allow_rescheduling: Whether to consider rescheduling others
            allow_double_booking: Whether to consider double-booking (default: True)
            date: Date for scheduling
            external_data: External factors (weather, traffic) for prediction

        Returns:
            SqueezeInResult object with detailed information
        """
        date = date or datetime.now()
        options_evaluated = 0

        # Strategy 1: Find natural gaps (always safest)
        gap_options = self.find_squeeze_in_options(patient, existing_schedule, date)
        options_evaluated += len(gap_options)

        if gap_options:
            best = gap_options[0]
            appointment = ScheduledAppointment(
                patient_id=patient.patient_id,
                chair_id=best.chair_id,
                site_code=best.site_code,
                start_time=best.start_time,
                end_time=best.end_time,
                duration=patient.expected_duration,
                priority=patient.priority,
                travel_time=0
            )

            # Remaining_Slack = min(slack_before, slack_after)
            remaining_slack = min(best.gap_before, best.gap_after) if best.gap_after > 0 else best.gap_before
            robustness_impact = (best.gap_before + best.gap_after) - (2 * remaining_slack)

            return SqueezeInResult(
                success=True,
                appointment=appointment,
                options_evaluated=options_evaluated,
                affected_patients=[],
                message=f"Found gap at {best.start_time.strftime('%H:%M')} on {best.chair_id}",
                strategy_used="gap",
                noshow_probability=0.0,
                confidence_level="N/A",
                robustness_impact=robustness_impact,
                remaining_slack=remaining_slack
            )

        # Strategy 2: Double-book high no-show probability slots
        if allow_double_booking and self.noshow_model and patient_data_map:
            logger.info("No gaps found - analyzing no-show probabilities for double-booking...")

            noshow_options = self.find_best_slot_for_urgent(
                patient, existing_schedule, patient_data_map,
                date, external_data, allow_double_booking=True
            )

            # Filter to only no-show based options
            noshow_options = [o for o in noshow_options if o.noshow_based]
            options_evaluated += len(noshow_options)

            if noshow_options:
                best = noshow_options[0]

                # Only use if no-show probability is above threshold
                if best.expected_noshow_prob >= self.NOSHOW_THRESHOLD_LOW:
                    appointment = ScheduledAppointment(
                        patient_id=patient.patient_id,
                        chair_id=best.chair_id,
                        site_code=best.site_code,
                        start_time=best.start_time,
                        end_time=best.end_time,
                        duration=patient.expected_duration,
                        priority=patient.priority,
                        travel_time=0
                    )

                    confidence = "HIGH" if best.expected_noshow_prob >= self.NOSHOW_THRESHOLD_HIGH else \
                                "MEDIUM" if best.expected_noshow_prob >= self.NOSHOW_THRESHOLD_MEDIUM else "LOW"

                    return SqueezeInResult(
                        success=True,
                        appointment=appointment,
                        options_evaluated=options_evaluated,
                        affected_patients=best.affected_appointments,
                        message=f"Double-booking at {best.start_time.strftime('%H:%M')} ({confidence} confidence, "
                               f"{best.expected_noshow_prob:.0%} no-show prob for {best.affected_appointments[0]})",
                        strategy_used="double_booking",
                        noshow_probability=best.expected_noshow_prob,
                        confidence_level=confidence
                    )

        # Strategy 3: Reschedule lower priority patients
        if allow_rescheduling:
            resched_options = self.find_rescheduling_options(patient, existing_schedule, date)
            options_evaluated += len(resched_options)

            if resched_options:
                best = resched_options[0]
                appointment = ScheduledAppointment(
                    patient_id=patient.patient_id,
                    chair_id=best.chair_id,
                    site_code=best.site_code,
                    start_time=best.start_time,
                    end_time=best.end_time,
                    duration=patient.expected_duration,
                    priority=patient.priority,
                    travel_time=0
                )

                return SqueezeInResult(
                    success=True,
                    appointment=appointment,
                    options_evaluated=options_evaluated,
                    affected_patients=best.affected_appointments,
                    message=f"Requires rescheduling {best.affected_appointments[0]}",
                    strategy_used="rescheduling",
                    noshow_probability=0.0,
                    confidence_level="N/A"
                )

        # Compose an actionable message that distinguishes "we tried and
        # nothing fit" from "you disabled the strategy that would have fit".
        attempted = ["gaps"]  # always tried
        skipped = []
        if allow_double_booking and self.noshow_model and patient_data_map:
            attempted.append("double-booking")
        else:
            skipped.append("double-booking")
        if allow_rescheduling:
            attempted.append("rescheduling")
        else:
            skipped.append("rescheduling")

        if skipped:
            msg = (
                f"No suitable slot found. Tried: {', '.join(attempted)}. "
                f"Disabled: {', '.join(skipped)} -- enabling these may help."
            )
        else:
            msg = (
                "No suitable slot found. All strategies tried "
                f"({', '.join(attempted)}); the schedule is fully packed and no "
                f"existing appointment crossed the no-show threshold or had a "
                f"compatible duration for this {patient.expected_duration}-min insertion."
            )

        return SqueezeInResult(
            success=False,
            appointment=None,
            options_evaluated=options_evaluated,
            affected_patients=[],
            message=msg,
            strategy_used="none",
            noshow_probability=0.0,
            confidence_level="N/A"
        )

    def get_available_capacity(self, existing_schedule: List[ScheduledAppointment],
                               date: datetime = None) -> Dict:
        """
        Get available capacity for squeeze-ins.

        Args:
            existing_schedule: Current schedule
            date: Date to check

        Returns:
            Dict with capacity information
        """
        date = date or datetime.now()
        start_hour, end_hour = OPERATING_HOURS
        total_minutes = (end_hour - start_hour) * 60

        # Calculate scheduled time per chair
        chair_usage = {}
        for apt in existing_schedule:
            if apt.chair_id not in chair_usage:
                chair_usage[apt.chair_id] = 0
            chair_usage[apt.chair_id] += apt.duration

        # Calculate available capacity
        total_capacity = len(self.chairs) * total_minutes
        total_used = sum(chair_usage.values())
        total_available = total_capacity - total_used

        # Find largest gaps
        largest_gaps = []
        for chair in self.chairs:
            if chair.chair_id not in chair_usage:
                largest_gaps.append({
                    'chair_id': chair.chair_id,
                    'available_minutes': total_minutes
                })
            else:
                available = total_minutes - chair_usage.get(chair.chair_id, 0)
                largest_gaps.append({
                    'chair_id': chair.chair_id,
                    'available_minutes': available
                })

        largest_gaps.sort(key=lambda x: x['available_minutes'], reverse=True)

        return {
            'total_capacity_minutes': total_capacity,
            'used_minutes': total_used,
            'available_minutes': total_available,
            'utilization': round(total_used / total_capacity, 3) if total_capacity else 0,
            'chairs_with_capacity': [g for g in largest_gaps if g['available_minutes'] >= 30],
            'estimated_slots_30min': total_available // 30,
            'estimated_slots_60min': total_available // 60,
            'estimated_slots_120min': total_available // 120
        }


# Example usage
if __name__ == "__main__":
    from datetime import datetime, timedelta

    # Create sample chairs
    chairs = [
        Chair(chair_id='WC-C01', site_code='WC', is_recliner=False),
        Chair(chair_id='WC-C02', site_code='WC', is_recliner=False),
        Chair(chair_id='WC-B01', site_code='WC', is_recliner=True),
    ]

    handler = SqueezeInHandler(chairs)

    # Create existing schedule
    today = datetime.now().replace(hour=0, minute=0, second=0)
    existing = [
        ScheduledAppointment(
            patient_id='P001', chair_id='WC-C01', site_code='WC',
            start_time=today.replace(hour=9), end_time=today.replace(hour=11),
            duration=120, priority=2, travel_time=20
        ),
        ScheduledAppointment(
            patient_id='P002', chair_id='WC-C01', site_code='WC',
            start_time=today.replace(hour=13), end_time=today.replace(hour=15),
            duration=120, priority=3, travel_time=30
        ),
        ScheduledAppointment(
            patient_id='P003', chair_id='WC-C02', site_code='WC',
            start_time=today.replace(hour=10), end_time=today.replace(hour=12),
            duration=120, priority=2, travel_time=15
        ),
    ]

    # Urgent patient to squeeze in
    urgent = Patient(
        patient_id='URGENT001',
        priority=1,
        protocol='Emergency',
        expected_duration=60,
        postcode='CF14',
        earliest_time=today.replace(hour=8),
        latest_time=today.replace(hour=17),
        is_urgent=True
    )

    # Find options
    options = handler.find_squeeze_in_options(urgent, existing, today)

    print("Squeeze-In Options:")
    print("=" * 60)
    for i, opt in enumerate(options[:5]):
        print(f"\n{i+1}. Chair: {opt.chair_id}")
        print(f"   Time: {opt.start_time.strftime('%H:%M')} - {opt.end_time.strftime('%H:%M')}")
        print(f"   Gap After: {opt.gap_after} min")
        print(f"   Score: {opt.score:.1f}")

    # Attempt squeeze-in
    result = handler.squeeze_in(urgent, existing, allow_rescheduling=True, date=today)

    print("\n" + "=" * 60)
    print("Squeeze-In Result:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    if result.appointment:
        print(f"  Scheduled: {result.appointment.start_time.strftime('%H:%M')}")

    # Show capacity
    capacity = handler.get_available_capacity(existing, today)
    print("\nAvailable Capacity:")
    print(f"  Utilization: {capacity['utilization']:.1%}")
    print(f"  Available Minutes: {capacity['available_minutes']}")
    print(f"  Estimated 60min slots: {capacity['estimated_slots_60min']}")
