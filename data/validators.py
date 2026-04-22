"""
Data Validators
===============

Validates input data for the scheduling system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    POSTCODE_COORDINATES,
    PRIORITY_DEFINITIONS,
    get_logger
)

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


class DataValidator:
    """
    Validates data quality and integrity.

    Checks:
    - Required fields present
    - Data types correct
    - Values in valid ranges
    - Referential integrity
    """

    # UK postcode pattern (simplified)
    POSTCODE_PATTERN = r'^[A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9][A-Z]{2}$'

    # Valid priority values
    VALID_PRIORITIES = [1, 2, 3, 4]

    # Valid protocols (common ones)
    VALID_PROTOCOLS = [
        'R-CHOP', 'CHOP', 'ABVD', 'FEC', 'FOLFOX', 'FOLFIRI',
        'PACLITAXEL', 'CARBOPLATIN', 'RITUXIMAB', 'HERCEPTIN',
        'PEMBROLIZUMAB', 'NIVOLUMAB', 'DOCETAXEL', 'GEMCITABINE'
    ]

    def __init__(self):
        """Initialize validator"""
        self._postcode_regex = re.compile(self.POSTCODE_PATTERN, re.IGNORECASE)
        logger.debug("Data validator initialized")

    def validate_patients(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate patient data.

        Args:
            df: Patient DataFrame

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        stats = {
            'total_records': len(df),
            'valid_records': 0,
            'invalid_postcodes': 0,
            'invalid_priorities': 0
        }

        # Check required columns
        required = ['patient_id', 'postcode', 'priority']
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")
            return ValidationResult(is_valid=False, errors=errors)

        valid_count = 0

        for idx, row in df.iterrows():
            row_valid = True

            # Validate patient_id
            if pd.isna(row['patient_id']) or str(row['patient_id']).strip() == '':
                errors.append(f"Row {idx}: Empty patient_id")
                row_valid = False

            # Validate postcode
            postcode = str(row['postcode']).strip().upper()
            if not self._validate_postcode(postcode):
                warnings.append(f"Row {idx}: Invalid postcode format '{postcode}'")
                stats['invalid_postcodes'] += 1

            # Check if postcode is in known areas
            district = postcode.split()[0] if ' ' in postcode else postcode[:4]
            if district not in POSTCODE_COORDINATES:
                warnings.append(f"Row {idx}: Unknown postcode district '{district}'")

            # Validate priority
            priority = row['priority']
            if priority not in self.VALID_PRIORITIES:
                errors.append(f"Row {idx}: Invalid priority '{priority}'")
                stats['invalid_priorities'] += 1
                row_valid = False

            if row_valid:
                valid_count += 1

        stats['valid_records'] = valid_count

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    def validate_appointments(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate appointment data.

        Args:
            df: Appointment DataFrame

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        stats = {
            'total_records': len(df),
            'valid_records': 0,
            'unknown_protocols': 0,
            'invalid_durations': 0
        }

        # Check required columns
        required = ['patient_id', 'protocol', 'duration']
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")
            return ValidationResult(is_valid=False, errors=errors)

        valid_count = 0

        for idx, row in df.iterrows():
            row_valid = True

            # Validate patient_id
            if pd.isna(row['patient_id']) or str(row['patient_id']).strip() == '':
                errors.append(f"Row {idx}: Empty patient_id")
                row_valid = False

            # Validate protocol
            protocol = str(row['protocol']).strip().upper()
            if protocol not in self.VALID_PROTOCOLS:
                warnings.append(f"Row {idx}: Unknown protocol '{protocol}'")
                stats['unknown_protocols'] += 1

            # Validate duration
            duration = row['duration']
            if pd.isna(duration):
                errors.append(f"Row {idx}: Missing duration")
                row_valid = False
            elif isinstance(duration, (int, float)):
                if duration < 15:
                    errors.append(f"Row {idx}: Duration {duration} too short (min 15 min)")
                    stats['invalid_durations'] += 1
                    row_valid = False
                elif duration > 480:
                    warnings.append(f"Row {idx}: Duration {duration} unusually long (>8 hours)")

            # Validate date if present
            if 'requested_date' in df.columns:
                date = row.get('requested_date')
                if pd.notna(date):
                    try:
                        if isinstance(date, str):
                            pd.to_datetime(date)
                    except:
                        warnings.append(f"Row {idx}: Invalid date format '{date}'")

            if row_valid:
                valid_count += 1

        stats['valid_records'] = valid_count

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    def _validate_postcode(self, postcode: str) -> bool:
        """Check if postcode matches UK format"""
        return bool(self._postcode_regex.match(postcode))

    def validate_schedule(self, appointments: List,
                         chairs: List) -> ValidationResult:
        """
        Validate a generated schedule.

        Args:
            appointments: List of scheduled appointments
            chairs: List of available chairs

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        stats = {
            'total_appointments': len(appointments),
            'overlapping': 0,
            'outside_hours': 0,
            'overbooked_chairs': 0
        }

        # Check for overlaps per chair
        chair_schedules = {}
        for apt in appointments:
            chair_id = apt.chair_id
            if chair_id not in chair_schedules:
                chair_schedules[chair_id] = []
            chair_schedules[chair_id].append(apt)

        for chair_id, chair_apts in chair_schedules.items():
            # Sort by start time
            sorted_apts = sorted(chair_apts, key=lambda a: a.start_time)

            for i in range(len(sorted_apts) - 1):
                current = sorted_apts[i]
                next_apt = sorted_apts[i + 1]

                if current.end_time > next_apt.start_time:
                    errors.append(
                        f"Overlap on {chair_id}: {current.patient_id} "
                        f"({current.end_time.strftime('%H:%M')}) overlaps with "
                        f"{next_apt.patient_id} ({next_apt.start_time.strftime('%H:%M')})"
                    )
                    stats['overlapping'] += 1

        # Check operating hours
        from config import OPERATING_HOURS
        start_hour, end_hour = OPERATING_HOURS

        for apt in appointments:
            if apt.start_time.hour < start_hour:
                errors.append(
                    f"{apt.patient_id}: Starts before opening "
                    f"({apt.start_time.strftime('%H:%M')})"
                )
                stats['outside_hours'] += 1

            if apt.end_time.hour > end_hour or (apt.end_time.hour == end_hour and apt.end_time.minute > 0):
                warnings.append(
                    f"{apt.patient_id}: Ends after closing "
                    f"({apt.end_time.strftime('%H:%M')})"
                )
                stats['outside_hours'] += 1

        # Check chair existence
        valid_chair_ids = {c.chair_id for c in chairs} if chairs else set()
        for apt in appointments:
            if valid_chair_ids and apt.chair_id not in valid_chair_ids:
                errors.append(f"{apt.patient_id}: Assigned to unknown chair {apt.chair_id}")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    def validate_excel_file(self, file_path: str) -> ValidationResult:
        """
        Validate an Excel file before processing.

        Args:
            file_path: Path to Excel file

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        stats = {}

        path = Path(file_path)

        # Check file exists
        if not path.exists():
            errors.append(f"File not found: {file_path}")
            return ValidationResult(is_valid=False, errors=errors)

        # Check file extension
        if path.suffix.lower() not in ['.xlsx', '.xls', '.csv']:
            errors.append(f"Unsupported file type: {path.suffix}")
            return ValidationResult(is_valid=False, errors=errors)

        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        stats['file_size_mb'] = round(size_mb, 2)

        if size_mb > 50:
            errors.append(f"File too large: {size_mb:.1f} MB (max 50 MB)")
            return ValidationResult(is_valid=False, errors=errors)
        elif size_mb > 10:
            warnings.append(f"Large file: {size_mb:.1f} MB may be slow to process")

        # Try to read file
        try:
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path, nrows=5)
            else:
                df = pd.read_excel(path, nrows=5)

            stats['columns'] = list(df.columns)
            stats['preview_rows'] = len(df)

        except Exception as e:
            errors.append(f"Cannot read file: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors)

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )


# Example usage
if __name__ == "__main__":
    validator = DataValidator()

    # Test patient validation
    patient_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', '', 'P004'],
        'postcode': ['CF14 4XW', 'INVALID', 'NP20 1AB', 'CF23 5PQ'],
        'priority': [1, 2, 5, 3]  # 5 is invalid
    })

    print("Patient Validation:")
    print("=" * 50)
    result = validator.validate_patients(patient_data)
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Stats: {result.stats}")

    # Test appointment validation
    appointment_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'protocol': ['R-CHOP', 'UNKNOWN-PROTOCOL', 'FEC'],
        'duration': [180, 5, 90]  # 5 is too short
    })

    print("\nAppointment Validation:")
    print("=" * 50)
    result = validator.validate_appointments(appointment_data)
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Stats: {result.stats}")
