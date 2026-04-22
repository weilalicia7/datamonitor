"""
Feature Engineering
====================

Creates features for ML models from patient and scheduling data.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    POSTCODE_COORDINATES,
    DEFAULT_SITES,
    get_logger
)

# SACT v4.0 adapter — normalises both real and synthetic data to identical ML features
try:
    from data.sact_v4_schema import SACTv4DataAdapter, SACT_V4_SCHEMA
    _SACT_V4_FIELDS = set(SACT_V4_SCHEMA.keys())
    _sact_adapter = SACTv4DataAdapter()
    _SACT_ADAPTER_AVAILABLE = True
except ImportError:
    _SACT_ADAPTER_AVAILABLE = False
    _SACT_V4_FIELDS = set()

logger = get_logger(__name__)


@dataclass
class PatientFeatures:
    """Features for a single patient"""
    patient_id: str
    features: Dict[str, float]
    feature_names: List[str]


class FeatureEngineer:
    """
    Creates features for no-show and duration prediction models.

    Features include:
    - Patient history features
    - Temporal features (time of day, day of week, seasonality)
    - Geographic features (distance, travel time)
    - Treatment features (protocol, priority)
    - External features (weather, traffic, events)
    """

    # Time-based feature definitions
    TIME_SLOTS = {
        'early_morning': (7, 9),
        'mid_morning': (9, 11),
        'late_morning': (11, 13),
        'early_afternoon': (13, 15),
        'late_afternoon': (15, 17),
        'evening': (17, 19)
    }

    # Treatment complexity factors
    TREATMENT_COMPLEXITY = {
        'simple': 1.0,
        'standard': 1.5,
        'complex': 2.0,
        'very_complex': 3.0
    }

    def __init__(self):
        """Initialize feature engineer"""
        self._site_coords = {
            site['code']: (site['lat'], site['lon'])
            for site in DEFAULT_SITES
        }
        logger.info("Feature engineer initialized")

    def create_patient_features(self, patient_data: Dict,
                                appointment_data: Dict,
                                external_data: Dict = None) -> PatientFeatures:
        """
        Create features for a single patient appointment.

        Args:
            patient_data: Patient demographics and history
            appointment_data: Appointment details
            external_data: Weather, traffic, event data

        Returns:
            PatientFeatures object
        """
        features = {}
        external_data = external_data or {}

        # Patient history features
        features.update(self._create_history_features(patient_data))

        # Temporal features
        features.update(self._create_temporal_features(appointment_data))

        # Geographic features
        features.update(self._create_geographic_features(
            patient_data, appointment_data
        ))

        # Treatment features
        features.update(self._create_treatment_features(appointment_data))

        # External features
        features.update(self._create_external_features(
            appointment_data, external_data
        ))

        return PatientFeatures(
            patient_id=patient_data.get('patient_id', 'unknown'),
            features=features,
            feature_names=list(features.keys())
        )

    def _create_history_features(self, patient_data: Dict) -> Dict[str, float]:
        """Create features from patient history - includes all 22 PDF required fields"""
        features = {}

        # Previous no-shows (PDF field 20)
        total_appointments = patient_data.get('total_appointments',
                            patient_data.get('Historical_Total_Appointments', 0))
        no_shows = patient_data.get('no_shows',
                   patient_data.get('Previous_NoShows',
                   patient_data.get('Historical_NoShow_Count', 0)))

        features['prev_noshow_rate'] = (
            no_shows / max(1, total_appointments)
        )
        features['prev_noshow_count'] = float(no_shows)  # PDF field 20
        features['total_appointments'] = float(total_appointments)
        features['is_new_patient'] = float(total_appointments == 0)

        # Previous cancellations (PDF field 21)
        cancellations = patient_data.get('cancellations',
                       patient_data.get('Previous_Cancellations', 0))
        features['prev_cancel_rate'] = (
            cancellations / max(1, total_appointments)
        )
        features['prev_cancel_count'] = float(cancellations)  # PDF field 21

        # Late arrivals
        late_arrivals = patient_data.get('late_arrivals', 0)
        features['prev_late_rate'] = (
            late_arrivals / max(1, total_appointments)
        )

        # Days since last appointment
        last_visit = patient_data.get('last_visit_date')
        if last_visit:
            if isinstance(last_visit, str):
                last_visit = datetime.fromisoformat(last_visit)
            days_since = (datetime.now() - last_visit).days
            features['days_since_last_visit'] = float(days_since)
        else:
            features['days_since_last_visit'] = 365.0  # Default for new patients

        # Treatment cycle (PDF field 5)
        features['treatment_cycle_number'] = float(
            patient_data.get('treatment_cycle',
            patient_data.get('Cycle_Number', 1))
        )
        features['is_first_cycle'] = float(features['treatment_cycle_number'] == 1)

        # Age Band features (PDF field 14: <40, 40-60, 60-75, >75)
        age = patient_data.get('age', patient_data.get('Age', 50))
        age_band = patient_data.get('age_band', patient_data.get('Age_Band', ''))
        # Derive age_band from age if not provided
        if not age_band:
            if age < 40:
                age_band = '<40'
            elif age < 60:
                age_band = '40-60'
            elif age < 75:
                age_band = '60-75'
            else:
                age_band = '>75'
        features['age'] = float(age)
        features['age_band'] = age_band  # String for rule-based model
        features['age_band_under40'] = float(age < 40 or age_band == '<40')
        features['age_band_40_60'] = float((40 <= age < 60) or age_band == '40-60')
        features['age_band_60_75'] = float((60 <= age < 75) or age_band == '60-75')
        features['age_band_over75'] = float(age >= 75 or age_band == '>75')

        # Has Comorbidities (PDF field 15)
        has_comorbidities = patient_data.get('has_comorbidities',
                           patient_data.get('Has_Comorbidities', False))
        features['has_comorbidities'] = float(has_comorbidities)

        # IV Access Difficulty (PDF field 16)
        iv_difficulty = patient_data.get('iv_access_difficulty',
                       patient_data.get('IV_Access_Difficulty', False))
        features['iv_access_difficulty'] = float(iv_difficulty)

        # Requires 1:1 Nursing (PDF field 17)
        requires_1to1 = patient_data.get('requires_1to1_nursing',
                       patient_data.get('Requires_1to1_Nursing', False))
        features['requires_1to1_nursing'] = float(requires_1to1)

        # Contact Preference (PDF field 22: SMS/Phone/Email/Post)
        contact_pref = patient_data.get('contact_preference',
                      patient_data.get('Contact_Preference', 'Phone'))
        features['contact_pref_sms'] = float(contact_pref == 'SMS')
        features['contact_pref_phone'] = float(contact_pref == 'Phone')
        features['contact_pref_email'] = float(contact_pref == 'Email')
        features['contact_pref_post'] = float(contact_pref == 'Post')

        return features

    def _create_temporal_features(self, appointment_data: Dict) -> Dict[str, float]:
        """Create time-based features - includes PDF fields 2, 12, 18"""
        features = {}

        # Parse appointment date/time (PDF field 2)
        apt_time = appointment_data.get('appointment_time',
                   appointment_data.get('Date'))
        if isinstance(apt_time, str):
            try:
                apt_time = datetime.fromisoformat(apt_time)
            except:
                apt_time = datetime.now()
        elif apt_time is None:
            apt_time = datetime.now()

        # Hour of day
        features['hour'] = float(apt_time.hour if hasattr(apt_time, 'hour') else 10)

        # Time slot one-hot encoding
        for slot_name, (start, end) in self.TIME_SLOTS.items():
            hour = apt_time.hour if hasattr(apt_time, 'hour') else 10
            features[f'slot_{slot_name}'] = float(start <= hour < end)

        # Day of week (PDF field 12: Mon-Fri)
        day_of_week = appointment_data.get('Day_Of_Week_Num',
                      appointment_data.get('day_of_week', apt_time.weekday()))
        if isinstance(day_of_week, str):
            day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
            day_of_week = day_map.get(day_of_week, 0)
        features['day_of_week'] = float(day_of_week)

        # Day of week one-hot (PDF field 12)
        for i, day in enumerate(['mon', 'tue', 'wed', 'thu', 'fri']):
            features[f'is_{day}'] = float(day_of_week == i)

        # Is weekend
        features['is_weekend'] = float(day_of_week >= 5)

        # Month
        month = apt_time.month if hasattr(apt_time, 'month') else 1
        features['month'] = float(month)

        # Season (affects no-show rates)
        features['is_winter'] = float(month in [12, 1, 2])
        features['is_spring'] = float(month in [3, 4, 5])
        features['is_summer'] = float(month in [6, 7, 8])
        features['is_autumn'] = float(month in [9, 10, 11])

        # Appointment Booked Date and lead time (PDF field 18)
        booked_date = appointment_data.get('Appointment_Booked_Date',
                     appointment_data.get('booked_date'))
        days_until = appointment_data.get('days_until',
                    appointment_data.get('Days_Booked_In_Advance', 0))

        if booked_date and not days_until:
            if isinstance(booked_date, str):
                try:
                    booked_date = datetime.fromisoformat(booked_date)
                    days_until = (apt_time - booked_date).days
                except:
                    days_until = 7

        features['days_until_appointment'] = float(days_until)
        features['booked_same_day'] = float(days_until <= 1)
        features['booked_same_week'] = float(days_until <= 7)
        features['booked_2_weeks'] = float(7 < days_until <= 14)
        features['booked_long_advance'] = float(days_until > 30)

        # Booking lead time has strong correlation with no-shows
        features['booking_lead_days'] = float(min(days_until, 60))  # Cap at 60 days

        return features

    def _create_geographic_features(self, patient_data: Dict,
                                    appointment_data: Dict) -> Dict[str, float]:
        """Create location-based features - includes PDF field 10 (Travel Time)"""
        features = {}

        # Get patient postcode
        postcode = patient_data.get('postcode',
                  patient_data.get('Postcode_District', ''))[:4]
        patient_coords = POSTCODE_COORDINATES.get(postcode)

        # Get site coordinates (PDF field 3)
        site_code = appointment_data.get('site_code',
                   appointment_data.get('Site_Code', 'WC'))
        site_coords = self._site_coords.get(site_code)

        # Travel Time to Site (PDF field 10)
        travel_time = patient_data.get('Travel_Time_Min',
                     patient_data.get('travel_time_min', 0))

        if patient_coords and site_coords:
            # Calculate distance
            distance = self._haversine_distance(
                patient_coords['lat'], patient_coords['lon'],
                site_coords[0], site_coords[1]
            )
            features['distance_km'] = distance
            features['travel_distance_km'] = patient_data.get('Travel_Distance_KM', distance)

            # Distance categories
            features['is_local'] = float(distance < 10)
            features['is_medium_distance'] = float(10 <= distance < 30)
            features['is_long_distance'] = float(distance >= 30)
            features['is_very_long_distance'] = float(distance >= 50)

            # Use provided travel time or estimate (PDF field 10)
            if travel_time > 0:
                features['travel_time_min'] = float(travel_time)
            else:
                features['travel_time_min'] = distance / 40 * 60  # Estimate at 40 km/h

            features['est_travel_time_min'] = features['travel_time_min']
        else:
            # Use provided travel time or defaults
            features['distance_km'] = patient_data.get('Travel_Distance_KM', 15.0)
            features['travel_distance_km'] = features['distance_km']
            features['is_local'] = float(features['distance_km'] < 10)
            features['is_medium_distance'] = float(10 <= features['distance_km'] < 30)
            features['is_long_distance'] = float(features['distance_km'] >= 30)
            features['is_very_long_distance'] = float(features['distance_km'] >= 50)
            features['travel_time_min'] = float(travel_time) if travel_time > 0 else 25.0
            features['est_travel_time_min'] = features['travel_time_min']

        # Travel time categories (strong predictor of no-shows)
        features['travel_under_15min'] = float(features['travel_time_min'] < 15)
        features['travel_15_30min'] = float(15 <= features['travel_time_min'] < 30)
        features['travel_30_60min'] = float(30 <= features['travel_time_min'] < 60)
        features['travel_over_60min'] = float(features['travel_time_min'] >= 60)

        # Postcode area features
        if postcode:
            features['is_cardiff'] = float(postcode.startswith('CF'))
            features['is_newport'] = float(postcode.startswith('NP'))
            features['is_swansea'] = float(postcode.startswith('SA'))
            features['is_valleys'] = float(postcode in ['CF37', 'CF38', 'CF81', 'CF82', 'CF83', 'NP44'])
        else:
            features['is_cardiff'] = 0.0
            features['is_newport'] = 0.0
            features['is_swansea'] = 0.0
            features['is_valleys'] = 0.0

        return features

    def _create_treatment_features(self, appointment_data: Dict) -> Dict[str, float]:
        """Create treatment-related features - includes all 22 PDF required fields"""
        features = {}

        # Priority level P1-P4 (PDF field 13)
        priority = appointment_data.get('priority',
                   appointment_data.get('Priority', 'P2'))
        priority_num = int(priority[1]) if priority and len(priority) >= 2 else 2
        features['priority_level'] = float(priority_num)

        # Priority one-hot (P1=urgent, P2=high, P3=medium, P4=low)
        for i in range(1, 5):
            features[f'is_priority_{i}'] = float(priority_num == i)

        # Treatment type and complexity
        treatment_type = appointment_data.get('treatment_type', 'standard')
        complexity = self.TREATMENT_COMPLEXITY.get(treatment_type.lower(), 1.5)
        features['treatment_complexity'] = complexity

        # Planned/Expected duration (PDF field 7)
        expected_duration = appointment_data.get('expected_duration',
                           appointment_data.get('Planned_Duration',
                           appointment_data.get('Scheduled_Duration', 60)))
        features['expected_duration_min'] = float(expected_duration)
        features['is_short_treatment'] = float(expected_duration < 45)
        features['is_long_treatment'] = float(expected_duration > 120)
        features['is_very_long_treatment'] = float(expected_duration > 180)

        # Regimen Code (PDF field 4)
        regimen = appointment_data.get('protocol',
                  appointment_data.get('Regimen_Code', 'standard'))
        # Common regimen type features
        features['is_immunotherapy'] = float(
            any(r in str(regimen).lower() for r in ['pembrolizumab', 'nivolumab', 'ipilimumab'])
        )
        features['is_chemo_combo'] = float(
            any(r in str(regimen).lower() for r in ['folfox', 'folfiri', 'r-chop', 'capox'])
        )

        # Cycle Number (PDF field 5)
        cycle_num = appointment_data.get('cycle_number',
                   appointment_data.get('Cycle_Number', 1))
        features['cycle_number'] = float(cycle_num)
        features['is_first_cycle'] = float(cycle_num == 1)
        features['is_early_cycle'] = float(cycle_num <= 3)

        # Treatment Day (PDF field 6)
        treatment_day = appointment_data.get('treatment_day',
                       appointment_data.get('Treatment_Day', 'Day 1'))
        if isinstance(treatment_day, str):
            day_num = int(''.join(filter(str.isdigit, treatment_day)) or '1')
        else:
            day_num = int(treatment_day)
        features['treatment_day_num'] = float(day_num)
        features['is_day_one'] = float(day_num == 1)

        # Chair Number (PDF field 9) - used for resource tracking
        chair_number = appointment_data.get('chair_number',
                      appointment_data.get('Chair_Number', 1))
        features['chair_number'] = float(chair_number)

        # Chair/Bed requirements
        long_infusion = appointment_data.get('long_infusion',
                      appointment_data.get('Long_Infusion', False))
        features['long_infusion'] = float(long_infusion)

        # Site Code features (PDF field 3)
        site_code = appointment_data.get('site_code',
                   appointment_data.get('Site_Code', 'WC'))
        features['is_main_site'] = float(site_code == 'WC')
        features['is_outreach_site'] = float(site_code in ['PCH', 'RGH', 'POW', 'CWM'])

        return features

    def _create_external_features(self, appointment_data: Dict,
                                  external_data: Dict) -> Dict[str, float]:
        """Create features from external factors"""
        features = {}

        # Weather features
        weather = external_data.get('weather', {})
        features['weather_severity'] = float(weather.get('severity', 0.0))
        features['precipitation_prob'] = float(
            weather.get('precipitation_probability', 0) / 100
        )
        features['is_bad_weather'] = float(
            weather.get('severity', 0) > 0.3
        )

        # Traffic features
        traffic = external_data.get('traffic', {})
        features['traffic_severity'] = float(traffic.get('severity', 0.0))
        features['expected_delay_min'] = float(traffic.get('delay_minutes', 0))
        features['is_rush_hour'] = float(traffic.get('is_rush_hour', False))

        # Event features
        events = external_data.get('events', {})
        features['event_severity'] = float(events.get('severity', 0.0))
        features['num_active_events'] = float(events.get('count', 0))

        # Combined external score
        features['combined_external_severity'] = (
            features['weather_severity'] * 0.3 +
            features['traffic_severity'] * 0.4 +
            features['event_severity'] * 0.3
        )

        return features

    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km"""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def create_features_dataframe(self, patients: List[Dict],
                                  appointments: List[Dict],
                                  external_data: Dict = None) -> pd.DataFrame:
        """
        Create features DataFrame for multiple patient appointments.

        Args:
            patients: List of patient data dicts
            appointments: List of appointment data dicts
            external_data: Shared external data

        Returns:
            DataFrame with features
        """
        all_features = []

        for patient, appointment in zip(patients, appointments):
            pf = self.create_patient_features(
                patient, appointment, external_data
            )
            row = {'patient_id': pf.patient_id}
            row.update(pf.features)
            all_features.append(row)

        df = pd.DataFrame(all_features)
        logger.info(f"Created features for {len(df)} appointments")

        return df

    def prepare_dataframe_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a DataFrame for ML training/inference.

        Detects whether the DataFrame contains SACT v4.0 fields (real data)
        or standard synthetic fields, then routes through SACTv4DataAdapter
        to produce identical derived features from both sources.

        Args:
            df: Raw appointments DataFrame (real SACT CSV or synthetic)

        Returns:
            DataFrame with all derived ML features added (BSA, age, lead time,
            priority, travel, ICD-10 group, no-show history features)
        """
        if not _SACT_ADAPTER_AVAILABLE:
            logger.warning("SACTv4DataAdapter not available — skipping SACT v4 normalisation")
            return df

        # Detect SACT v4 content: check for at least 3 SACT v4 native fields
        sact_overlap = _SACT_V4_FIELDS.intersection(set(df.columns))
        if len(sact_overlap) >= 3:
            logger.info(
                f"SACT v4 fields detected ({len(sact_overlap)} columns) — "
                "routing through SACTv4DataAdapter"
            )
            df = _sact_adapter.adapt(df)
        else:
            logger.debug(
                f"Standard synthetic data detected ({len(df.columns)} columns) — "
                "skipping SACT v4 adapter"
            )

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Create dummy data to get feature names
        dummy_patient = {'patient_id': 'dummy', 'postcode': 'CF14'}
        dummy_appointment = {'appointment_time': datetime.now()}
        pf = self.create_patient_features(dummy_patient, dummy_appointment)
        return pf.feature_names


# Example usage
if __name__ == "__main__":
    engineer = FeatureEngineer()

    # Sample patient data
    patient = {
        'patient_id': 'P001',
        'postcode': 'CF14 4XW',
        'total_appointments': 12,
        'no_shows': 1,
        'cancellations': 2,
        'late_arrivals': 3,
        'last_visit_date': datetime.now() - timedelta(days=21),
        'treatment_cycle': 3
    }

    # Sample appointment data
    appointment = {
        'appointment_time': datetime.now() + timedelta(days=7, hours=10),
        'site_code': 'WC',
        'priority': 'P2',
        'treatment_type': 'standard',
        'expected_duration': 90,
        'protocol': 'CHOP',
        'days_until': 7
    }

    # Sample external data
    external = {
        'weather': {
            'severity': 0.3,
            'precipitation_probability': 60
        },
        'traffic': {
            'severity': 0.2,
            'delay_minutes': 10,
            'is_rush_hour': True
        },
        'events': {
            'severity': 0.1,
            'count': 1
        }
    }

    # Create features
    features = engineer.create_patient_features(patient, appointment, external)

    print("Patient Features:")
    print("=" * 50)
    for name, value in sorted(features.features.items()):
        print(f"  {name}: {value:.3f}")

    print(f"\nTotal features: {len(features.feature_names)}")
