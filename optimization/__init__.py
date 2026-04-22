"""
SACT Scheduling System - Optimization Module
============================================

OR-Tools based scheduling optimization with ML enhancement.
"""

from .optimizer import ScheduleOptimizer, Patient, Chair, ScheduledAppointment, OptimizationResult
from .constraints import ConstraintManager, Constraint, ConstraintViolation
from .squeeze_in import SqueezeInHandler, SqueezeInOption, SqueezeInResult
from .emergency_mode import EmergencyModeHandler, EmergencySchedule
from .column_generation import ColumnGenerator, Column, CGResult

__all__ = [
    'ScheduleOptimizer',
    'Patient',
    'Chair',
    'ScheduledAppointment',
    'OptimizationResult',
    'ConstraintManager',
    'Constraint',
    'ConstraintViolation',
    'SqueezeInHandler',
    'SqueezeInOption',
    'SqueezeInResult',
    'EmergencyModeHandler',
    'EmergencySchedule',
    'ColumnGenerator',
    'Column',
    'CGResult',
]
