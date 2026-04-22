"""
Scheduling Constraints
======================

Defines and manages scheduling constraints.
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OPERATING_HOURS,
    PRIORITY_DEFINITIONS,
    get_logger
)

logger = get_logger(__name__)


class ConstraintType(Enum):
    """Types of scheduling constraints"""
    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Prefer to satisfy, but can violate


class ConstraintCategory(Enum):
    """Categories of constraints"""
    TEMPORAL = "temporal"  # Time-based constraints
    RESOURCE = "resource"  # Chair/bed availability
    PATIENT = "patient"    # Patient-specific
    CLINICAL = "clinical"  # Clinical requirements
    OPERATIONAL = "operational"  # Business rules


@dataclass
class Constraint:
    """A scheduling constraint"""
    name: str
    constraint_type: ConstraintType
    category: ConstraintCategory
    description: str
    check_function: Callable = None
    penalty_weight: float = 1.0  # For soft constraints
    is_active: bool = True


@dataclass
class ConstraintViolation:
    """Record of a constraint violation"""
    constraint_name: str
    constraint_type: ConstraintType
    patient_id: str
    details: str
    penalty: float


class ConstraintManager:
    """
    Manages scheduling constraints.

    Provides constraint checking, violation tracking,
    and penalty calculation for soft constraints.
    """

    def __init__(self):
        """Initialize constraint manager"""
        self.constraints: Dict[str, Constraint] = {}
        self._register_default_constraints()
        logger.info("Constraint manager initialized")

    def _register_default_constraints(self):
        """Register default scheduling constraints"""
        # Hard temporal constraints
        self.add_constraint(Constraint(
            name="operating_hours",
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.TEMPORAL,
            description="Appointments must be within operating hours",
            check_function=self._check_operating_hours
        ))

        self.add_constraint(Constraint(
            name="no_overlap",
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.RESOURCE,
            description="Appointments cannot overlap on the same chair",
            check_function=self._check_no_overlap
        ))

        self.add_constraint(Constraint(
            name="bed_requirement",
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.CLINICAL,
            description="Patients requiring beds must be assigned to beds",
            check_function=self._check_bed_requirement
        ))

        # Hard clinical constraints
        self.add_constraint(Constraint(
            name="minimum_duration",
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.CLINICAL,
            description="Appointments must meet minimum protocol duration",
            check_function=self._check_minimum_duration
        ))

        # Soft constraints
        self.add_constraint(Constraint(
            name="priority_order",
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.CLINICAL,
            description="Higher priority patients should be scheduled earlier",
            penalty_weight=10.0,
            check_function=self._check_priority_order
        ))

        self.add_constraint(Constraint(
            name="travel_time",
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.PATIENT,
            description="Minimize patient travel time",
            penalty_weight=5.0,
            check_function=self._check_travel_time
        ))

        self.add_constraint(Constraint(
            name="preferred_site",
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.PATIENT,
            description="Assign patients to preferred site when possible",
            penalty_weight=3.0,
            check_function=self._check_preferred_site
        ))

        self.add_constraint(Constraint(
            name="workload_balance",
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.OPERATIONAL,
            description="Balance workload across sites",
            penalty_weight=2.0,
            check_function=self._check_workload_balance
        ))

        self.add_constraint(Constraint(
            name="buffer_time",
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.OPERATIONAL,
            description="Maintain buffer between appointments",
            penalty_weight=1.0,
            check_function=self._check_buffer_time
        ))

    def add_constraint(self, constraint: Constraint):
        """Add a constraint"""
        self.constraints[constraint.name] = constraint
        logger.debug(f"Added constraint: {constraint.name}")

    def remove_constraint(self, name: str):
        """Remove a constraint"""
        if name in self.constraints:
            del self.constraints[name]
            logger.debug(f"Removed constraint: {name}")

    def enable_constraint(self, name: str):
        """Enable a constraint"""
        if name in self.constraints:
            self.constraints[name].is_active = True

    def disable_constraint(self, name: str):
        """Disable a constraint"""
        if name in self.constraints:
            self.constraints[name].is_active = False

    def check_constraints(self, appointment: Dict,
                         context: Dict) -> List[ConstraintViolation]:
        """
        Check all constraints for an appointment.

        Args:
            appointment: Appointment details
            context: Context including other appointments, chairs, etc.

        Returns:
            List of ConstraintViolation objects
        """
        violations = []

        for constraint in self.constraints.values():
            if not constraint.is_active:
                continue

            if constraint.check_function is None:
                continue

            violation = constraint.check_function(appointment, context)
            if violation:
                violations.append(ConstraintViolation(
                    constraint_name=constraint.name,
                    constraint_type=constraint.constraint_type,
                    patient_id=appointment.get('patient_id', 'unknown'),
                    details=violation,
                    penalty=constraint.penalty_weight if constraint.constraint_type == ConstraintType.SOFT else float('inf')
                ))

        return violations

    def has_hard_violations(self, violations: List[ConstraintViolation]) -> bool:
        """Check if any violations are hard constraints"""
        return any(v.constraint_type == ConstraintType.HARD for v in violations)

    def calculate_penalty(self, violations: List[ConstraintViolation]) -> float:
        """Calculate total penalty from soft constraint violations"""
        return sum(
            v.penalty for v in violations
            if v.constraint_type == ConstraintType.SOFT
        )

    # Constraint check functions
    def _check_operating_hours(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check operating hours constraint"""
        start = appointment.get('start_time')
        end = appointment.get('end_time')

        if not start or not end:
            return None

        start_hour, end_hour = OPERATING_HOURS

        if start.hour < start_hour:
            return f"Start time {start.strftime('%H:%M')} is before opening ({start_hour}:00)"

        if end.hour > end_hour or (end.hour == end_hour and end.minute > 0):
            return f"End time {end.strftime('%H:%M')} is after closing ({end_hour}:00)"

        return None

    def _check_no_overlap(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check no overlap constraint"""
        chair_id = appointment.get('chair_id')
        start = appointment.get('start_time')
        end = appointment.get('end_time')
        patient_id = appointment.get('patient_id')

        existing_appointments = context.get('appointments', [])

        for existing in existing_appointments:
            if existing.get('patient_id') == patient_id:
                continue

            if existing.get('chair_id') != chair_id:
                continue

            ex_start = existing.get('start_time')
            ex_end = existing.get('end_time')

            # Check overlap
            if start < ex_end and end > ex_start:
                return f"Overlaps with {existing.get('patient_id')} on chair {chair_id}"

        return None

    def _check_bed_requirement(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check bed requirement constraint"""
        long_infusion = appointment.get('long_infusion', False)
        if not long_infusion:
            return None

        chair_id = appointment.get('chair_id')
        chairs = context.get('chairs', {})

        chair = chairs.get(chair_id)
        if chair and not chair.get('is_recliner', False):
            return f"Patient requires bed but assigned to chair {chair_id}"

        return None

    def _check_minimum_duration(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check minimum duration constraint"""
        duration = appointment.get('duration', 0)
        protocol = appointment.get('protocol', '')

        # Minimum 15 minutes for any appointment
        if duration < 15:
            return f"Duration {duration}min is below minimum (15min)"

        return None

    def _check_priority_order(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check priority order constraint"""
        priority = appointment.get('priority', 2)
        start = appointment.get('start_time')

        existing_appointments = context.get('appointments', [])

        # Check if lower priority patients are scheduled earlier
        for existing in existing_appointments:
            ex_priority = existing.get('priority', 2)
            ex_start = existing.get('start_time')

            if ex_priority > priority and ex_start < start:
                # Lower priority patient scheduled earlier
                return f"P{priority} scheduled after P{ex_priority}"

        return None

    def _check_travel_time(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check travel time constraint"""
        travel_time = appointment.get('travel_time', 0)

        if travel_time > 60:
            return f"Travel time {travel_time}min exceeds preferred maximum (60min)"

        return None

    def _check_preferred_site(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check preferred site constraint"""
        preferred = appointment.get('preferred_site')
        assigned = appointment.get('site_code')

        if preferred and assigned and preferred != assigned:
            return f"Assigned to {assigned} instead of preferred {preferred}"

        return None

    def _check_workload_balance(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check workload balance constraint"""
        site_code = appointment.get('site_code')
        existing_appointments = context.get('appointments', [])

        # Count appointments per site
        site_counts = {}
        for apt in existing_appointments:
            site = apt.get('site_code')
            site_counts[site] = site_counts.get(site, 0) + 1

        # Add current appointment
        site_counts[site_code] = site_counts.get(site_code, 0) + 1

        if site_counts:
            max_count = max(site_counts.values())
            min_count = min(site_counts.values())

            if max_count - min_count > 10:
                return f"Site {site_code} has {site_counts[site_code]} appointments (imbalanced)"

        return None

    def _check_buffer_time(self, appointment: Dict, context: Dict) -> Optional[str]:
        """Check buffer time constraint"""
        chair_id = appointment.get('chair_id')
        start = appointment.get('start_time')
        end = appointment.get('end_time')

        existing_appointments = context.get('appointments', [])
        buffer_minutes = context.get('buffer_minutes', 5)

        for existing in existing_appointments:
            if existing.get('chair_id') != chair_id:
                continue

            ex_end = existing.get('end_time')
            ex_start = existing.get('start_time')

            # Check buffer before
            if ex_end and start:
                gap = (start - ex_end).total_seconds() / 60
                if 0 < gap < buffer_minutes:
                    return f"Only {gap:.0f}min gap after previous appointment (need {buffer_minutes}min)"

            # Check buffer after
            if ex_start and end:
                gap = (ex_start - end).total_seconds() / 60
                if 0 < gap < buffer_minutes:
                    return f"Only {gap:.0f}min gap before next appointment (need {buffer_minutes}min)"

        return None

    def get_constraint_summary(self) -> Dict:
        """Get summary of all constraints"""
        summary = {
            'total': len(self.constraints),
            'active': sum(1 for c in self.constraints.values() if c.is_active),
            'hard': sum(1 for c in self.constraints.values()
                       if c.constraint_type == ConstraintType.HARD),
            'soft': sum(1 for c in self.constraints.values()
                       if c.constraint_type == ConstraintType.SOFT),
            'by_category': {}
        }

        for constraint in self.constraints.values():
            cat = constraint.category.value
            if cat not in summary['by_category']:
                summary['by_category'][cat] = []
            summary['by_category'][cat].append({
                'name': constraint.name,
                'type': constraint.constraint_type.value,
                'active': constraint.is_active
            })

        return summary


# Example usage
if __name__ == "__main__":
    manager = ConstraintManager()

    # Sample appointment
    appointment = {
        'patient_id': 'P001',
        'chair_id': 'WC-C01',
        'site_code': 'WC',
        'start_time': datetime.now().replace(hour=9, minute=0),
        'end_time': datetime.now().replace(hour=12, minute=0),
        'duration': 180,
        'priority': 2,
        'travel_time': 25,
        'long_infusion': False
    }

    context = {
        'appointments': [],
        'chairs': {'WC-C01': {'is_recliner': False}},
        'buffer_minutes': 5
    }

    # Check constraints
    violations = manager.check_constraints(appointment, context)

    print("Constraint Check Results:")
    print("=" * 50)

    if violations:
        print(f"Found {len(violations)} violations:")
        for v in violations:
            print(f"  [{v.constraint_type.value}] {v.constraint_name}: {v.details}")
        print(f"\nTotal Penalty: {manager.calculate_penalty(violations)}")
    else:
        print("No violations found!")

    print("\nConstraint Summary:")
    summary = manager.get_constraint_summary()
    print(f"Total: {summary['total']} ({summary['active']} active)")
    print(f"Hard: {summary['hard']}, Soft: {summary['soft']}")
