"""
Data Processor
==============

Handles data loading, transformation, and preparation.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    DATA_CACHE_DIR,
    get_logger
)
from .validators import DataValidator

logger = get_logger(__name__)


@dataclass
class ProcessedData:
    """Container for processed data"""
    patients: pd.DataFrame
    appointments: pd.DataFrame
    sites: pd.DataFrame
    protocols: pd.DataFrame
    metadata: Dict


class DataProcessor:
    """
    Processes Excel/CSV data for the scheduling system.

    Handles:
    - Loading patient lists from Excel
    - Parsing appointment requests
    - Validating data integrity
    - Transforming to standard format
    """

    # Expected columns for each data type
    PATIENT_COLUMNS = {
        'required': ['patient_id', 'postcode', 'priority'],
        'optional': ['name', 'phone', 'email', 'notes', 'long_infusion',
                    'preferred_site', 'previous_appointments', 'no_shows']
    }

    APPOINTMENT_COLUMNS = {
        'required': ['patient_id', 'protocol', 'duration'],
        'optional': ['requested_date', 'earliest_time', 'latest_time',
                    'site_preference', 'notes', 'cycle_number']
    }

    def __init__(self):
        """Initialize data processor"""
        self.validator = DataValidator()
        self._data_cache = {}
        logger.info("Data processor initialized")

    def load_excel(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """
        Load data from Excel file.

        Args:
            file_path: Path to Excel file
            sheet_name: Specific sheet to load (optional)

        Returns:
            DataFrame with loaded data
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if sheet_name:
                df = pd.read_excel(path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(path)

            # Clean column names
            df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

            logger.info(f"Loaded {len(df)} rows from {path.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            df = pd.read_csv(path)

            # Clean column names
            df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

            logger.info(f"Loaded {len(df)} rows from {path.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def process_patient_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process patient list data.

        Args:
            df: Raw patient DataFrame

        Returns:
            Processed patient DataFrame
        """
        # Validate required columns
        missing = set(self.PATIENT_COLUMNS['required']) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Create processed DataFrame
        processed = df.copy()

        # Standardize patient_id
        processed['patient_id'] = processed['patient_id'].astype(str).str.strip()

        # Standardize postcode (uppercase, first part only)
        processed['postcode'] = (
            processed['postcode']
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        processed['postcode_district'] = processed['postcode'].str.split().str[0]

        # Standardize priority
        processed['priority'] = self._parse_priority(processed['priority'])

        # Add defaults for optional columns
        if 'long_infusion' not in processed.columns:
            processed['long_infusion'] = False
        else:
            processed['long_infusion'] = processed['long_infusion'].fillna(False).astype(bool)

        if 'preferred_site' not in processed.columns:
            processed['preferred_site'] = None

        if 'previous_appointments' not in processed.columns:
            processed['previous_appointments'] = 0

        if 'no_shows' not in processed.columns:
            processed['no_shows'] = 0

        # Validate data
        validation_results = self.validator.validate_patients(processed)
        if validation_results['errors']:
            logger.warning(f"Validation errors: {validation_results['errors']}")

        logger.info(f"Processed {len(processed)} patients")
        return processed

    def process_appointment_requests(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process appointment request data.

        Args:
            df: Raw appointment request DataFrame

        Returns:
            Processed appointment DataFrame
        """
        # Validate required columns
        missing = set(self.APPOINTMENT_COLUMNS['required']) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        processed = df.copy()

        # Standardize patient_id
        processed['patient_id'] = processed['patient_id'].astype(str).str.strip()

        # Standardize protocol
        processed['protocol'] = processed['protocol'].astype(str).str.strip().str.upper()

        # Parse duration
        processed['duration'] = self._parse_duration(processed['duration'])

        # Parse dates
        if 'requested_date' in processed.columns:
            processed['requested_date'] = pd.to_datetime(
                processed['requested_date'],
                errors='coerce'
            )

        # Parse times
        if 'earliest_time' in processed.columns:
            processed['earliest_time'] = self._parse_time(processed['earliest_time'])
        else:
            processed['earliest_time'] = '08:00'

        if 'latest_time' in processed.columns:
            processed['latest_time'] = self._parse_time(processed['latest_time'])
        else:
            processed['latest_time'] = '17:00'

        # Add cycle number
        if 'cycle_number' not in processed.columns:
            processed['cycle_number'] = 1

        # Validate
        validation_results = self.validator.validate_appointments(processed)
        if validation_results['errors']:
            logger.warning(f"Validation errors: {validation_results['errors']}")

        logger.info(f"Processed {len(processed)} appointment requests")
        return processed

    def _parse_priority(self, priority_col: pd.Series) -> pd.Series:
        """Parse priority column to integers 1-4"""
        def parse_single(val):
            if pd.isna(val):
                return 2  # Default to P2

            val_str = str(val).upper().strip()

            # Handle P1, P2, etc.
            if val_str.startswith('P'):
                try:
                    return int(val_str[1])
                except:
                    pass

            # Handle numeric
            try:
                num = int(float(val_str))
                return max(1, min(4, num))
            except:
                pass

            # Handle text
            priority_map = {
                'URGENT': 1, 'CRITICAL': 1,
                'HIGH': 2,
                'MEDIUM': 3, 'STANDARD': 3,
                'LOW': 4, 'ROUTINE': 4
            }
            return priority_map.get(val_str, 2)

        return priority_col.apply(parse_single)

    def _parse_duration(self, duration_col: pd.Series) -> pd.Series:
        """Parse duration column to minutes"""
        def parse_single(val):
            if pd.isna(val):
                return 60  # Default 60 minutes

            val_str = str(val).lower().strip()

            # Try numeric (assume minutes)
            try:
                return int(float(val_str))
            except:
                pass

            # Handle "Xh Ym" or "X hours"
            import re
            hours_match = re.search(r'(\d+)\s*h', val_str)
            mins_match = re.search(r'(\d+)\s*m', val_str)

            total = 0
            if hours_match:
                total += int(hours_match.group(1)) * 60
            if mins_match:
                total += int(mins_match.group(1))

            return total if total > 0 else 60

        return duration_col.apply(parse_single)

    def _parse_time(self, time_col: pd.Series) -> pd.Series:
        """Parse time column to HH:MM format"""
        def parse_single(val):
            if pd.isna(val):
                return '08:00'

            val_str = str(val).strip()

            # Handle datetime objects
            if hasattr(val, 'strftime'):
                return val.strftime('%H:%M')

            # Handle HH:MM format
            import re
            match = re.match(r'(\d{1,2}):?(\d{2})?', val_str)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2)) if match.group(2) else 0
                return f"{hour:02d}:{minute:02d}"

            return '08:00'

        return time_col.apply(parse_single)

    def merge_patient_appointments(self, patients: pd.DataFrame,
                                    appointments: pd.DataFrame) -> pd.DataFrame:
        """
        Merge patient and appointment data.

        Args:
            patients: Patient DataFrame
            appointments: Appointment DataFrame

        Returns:
            Merged DataFrame
        """
        merged = appointments.merge(
            patients,
            on='patient_id',
            how='left',
            suffixes=('', '_patient')
        )

        # Fill missing patient data with defaults
        merged['postcode'] = merged['postcode'].fillna('CF14')
        merged['priority'] = merged['priority'].fillna(2).astype(int)

        logger.info(f"Merged data: {len(merged)} records")
        return merged

    def load_and_process(self, patient_file: str = None,
                         appointment_file: str = None) -> ProcessedData:
        """
        Load and process all data files.

        Args:
            patient_file: Path to patient data file
            appointment_file: Path to appointment data file

        Returns:
            ProcessedData object
        """
        # Load patient data
        if patient_file and Path(patient_file).exists():
            if patient_file.endswith('.csv'):
                patients_raw = self.load_csv(patient_file)
            else:
                patients_raw = self.load_excel(patient_file)
            patients = self.process_patient_list(patients_raw)
        else:
            patients = pd.DataFrame(columns=['patient_id', 'postcode', 'priority'])

        # Load appointment data
        if appointment_file and Path(appointment_file).exists():
            if appointment_file.endswith('.csv'):
                appointments_raw = self.load_csv(appointment_file)
            else:
                appointments_raw = self.load_excel(appointment_file)
            appointments = self.process_appointment_requests(appointments_raw)
        else:
            appointments = pd.DataFrame(columns=['patient_id', 'protocol', 'duration'])

        # Load site configuration (from config or file)
        from config import DEFAULT_SITES
        sites = pd.DataFrame(DEFAULT_SITES)

        # Load protocol data
        protocols = self._load_protocols()

        # Metadata
        metadata = {
            'processed_at': datetime.now().isoformat(),
            'patient_count': len(patients),
            'appointment_count': len(appointments),
            'site_count': len(sites)
        }

        return ProcessedData(
            patients=patients,
            appointments=appointments,
            sites=sites,
            protocols=protocols,
            metadata=metadata
        )

    def _load_protocols(self) -> pd.DataFrame:
        """Load treatment protocol data"""
        from ml.duration_model import DurationModel

        # Get protocol durations from model
        durations = DurationModel.PROTOCOL_DURATIONS

        protocols = pd.DataFrame([
            {'protocol': name, 'default_duration': duration}
            for name, duration in durations.items()
            if name != 'default'
        ])

        return protocols

    def export_schedule(self, appointments: List, output_path: str,
                        format: str = 'excel'):
        """
        Export schedule to file.

        Args:
            appointments: List of scheduled appointments
            output_path: Output file path
            format: 'excel' or 'csv'
        """
        # Convert to DataFrame
        data = []
        for apt in appointments:
            data.append({
                'patient_id': apt.patient_id,
                'chair_id': apt.chair_id,
                'site': apt.site_code,
                'date': apt.start_time.strftime('%Y-%m-%d'),
                'start_time': apt.start_time.strftime('%H:%M'),
                'end_time': apt.end_time.strftime('%H:%M'),
                'duration_min': apt.duration,
                'priority': f'P{apt.priority}'
            })

        df = pd.DataFrame(data)

        # Export
        output_path = Path(output_path)
        if format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(df)} appointments to {output_path}")

    def cache_data(self, key: str, data: Any):
        """Cache data in memory"""
        self._data_cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def get_cached(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """Get cached data if not expired"""
        if key not in self._data_cache:
            return None

        cached = self._data_cache[key]
        age = (datetime.now() - cached['timestamp']).total_seconds()

        if age > max_age_seconds:
            del self._data_cache[key]
            return None

        return cached['data']


# Example usage
if __name__ == "__main__":
    processor = DataProcessor()

    # Create sample data
    patient_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'postcode': ['CF14 4XW', 'NP20 1AB', 'CF23 5PQ'],
        'priority': ['P1', 'P2', 'P3'],
        'long_infusion': [False, True, False]
    })

    appointment_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'protocol': ['R-CHOP', 'FEC', 'Paclitaxel'],
        'duration': ['3h', '90', '2h 30m'],
        'requested_date': ['2024-01-15', '2024-01-15', '2024-01-16']
    })

    # Process data
    patients = processor.process_patient_list(patient_data)
    appointments = processor.process_appointment_requests(appointment_data)

    print("Processed Patients:")
    print(patients[['patient_id', 'postcode_district', 'priority', 'long_infusion']])

    print("\nProcessed Appointments:")
    print(appointments[['patient_id', 'protocol', 'duration', 'requested_date']])

    # Merge
    merged = processor.merge_patient_appointments(patients, appointments)
    print("\nMerged Data:")
    print(merged[['patient_id', 'protocol', 'duration', 'priority', 'postcode_district']])
