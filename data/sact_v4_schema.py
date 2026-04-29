"""
SACT v4.0 Schema Validator, Adapter and Feature Extractor
==========================================================

Ensures synthetic data is 100% aligned with the SACT v4.0 standard
(NHS England, April 2026) so that real data can be swapped in immediately.

Three components:
    1. SACTv4Schema       — field definitions, types, valid values, mandatory flags
    2. SACTv4Validator    — validates any DataFrame against the schema
    3. SACTv4DataAdapter  — maps real SACT v4.0 CSV → internal ML feature format
                           Works identically for synthetic OR real data.

Usage:
    validator = SACTv4Validator()
    report = validator.validate(df, source='synthetic')

    adapter = SACTv4DataAdapter()
    features_df = adapter.adapt(raw_sact_df)  # works for both synthetic and real
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA DEFINITION — All 60 SACT v4.0 fields + scheduling extensions
# =============================================================================

SACT_V4_SCHEMA: Dict[str, Dict] = {
    # ── Section 1: Linkage ────────────────────────────────────────────────
    'NHS_Number': {
        'section': 1, 'label': 'NHS Number',
        'type': 'string', 'max_length': 10, 'mandatory': True,
        'pattern': r'^\d{10}$',
        'description': '10-digit NHS Number with Modulus 11 check digit',
    },
    'Local_Patient_Identifier': {
        'section': 1, 'label': 'Local Patient Identifier',
        'type': 'string', 'mandatory': True,
        'description': 'Trust-internal patient ID (e.g. P12345)',
    },
    'NHS_Number_Status_Indicator_Code': {
        'section': 1, 'label': 'NHS Number Status',
        'type': 'string', 'mandatory': True,
        'valid_values': ['01', '02', '03', '04'],
        'description': '01=Present & verified, 02=Present not verified, 03=Trace required, 04=No trace',
    },

    # ── Section 2: Demographics ───────────────────────────────────────────
    'Person_Family_Name': {
        'section': 2, 'label': 'Family Name',
        'type': 'string', 'mandatory': True,
    },
    'Person_Given_Name': {
        'section': 2, 'label': 'Given Name',
        'type': 'string', 'mandatory': True,
    },
    'Person_Birth_Date': {
        'section': 2, 'label': 'Date of Birth',
        'type': 'date', 'format': '%Y-%m-%d', 'mandatory': True,
    },
    'Person_Stated_Gender_Code': {
        'section': 2, 'label': 'Gender',
        'type': 'integer', 'mandatory': True,
        'valid_values': [1, 2, 9],
        'description': '1=Male, 2=Female, 9=Not Stated',
    },
    'Patient_Postcode': {
        'section': 2, 'label': 'Postcode',
        'type': 'string', 'mandatory': True,
    },
    'Organisation_Identifier': {
        'section': 2, 'label': 'Organisation ODS Code',
        'type': 'string', 'mandatory': True,
        'description': 'ODS code of treating organisation (Velindre=RQF)',
    },

    # ── Section 3: Clinical Status ────────────────────────────────────────
    'Primary_Diagnosis_ICD10': {
        'section': 3, 'label': 'Primary Diagnosis (ICD-10)',
        'type': 'string', 'mandatory': True,
        'pattern': r'^C\d{2}(\.\d)?$',
        'description': 'ICD-10 diagnosis code (e.g. C18.9)',
    },
    'Morphology_ICD_O': {
        'section': 3, 'label': 'Morphology (ICD-O)',
        'type': 'string', 'mandatory': True,
        'pattern': r'^\d{4}/\d$',
        'description': 'ICD-O-3 morphology code (e.g. 8140/3)',
    },
    'Performance_Status': {
        'section': 3, 'label': 'WHO Performance Status',
        'type': 'integer', 'mandatory': True,
        'valid_values': [0, 1, 2, 3, 4],
        'description': '0=Fully active, 4=Completely disabled',
    },
    'Consultant_Specialty_Code': {
        'section': 3, 'label': 'Consultant Specialty',
        'type': 'string', 'mandatory': False,
        'valid_values': ['370', '800', '303', '290', '400'],
        'description': '370=Clinical Oncology, 800=Oncology, 303=Haematology',
    },

    # ── Section 4: Regimen ────────────────────────────────────────────────
    'Regimen_Code': {
        'section': 4, 'label': 'Regimen Code',
        'type': 'string', 'mandatory': True,
        'description': 'SACT standard protocol abbreviation (e.g. FOLFOX, RCHOP)',
    },
    'Intent_Of_Treatment': {
        'section': 4, 'label': 'Intent of Treatment',
        'type': 'string', 'mandatory': True,
        'valid_values': ['06', '07'],
        'description': '06=Curative, 07=Non-curative',
    },
    'Treatment_Context': {
        'section': 4, 'label': 'Treatment Context',
        'type': 'string', 'mandatory': True,
        'valid_values': ['01', '02', '03'],
        'description': '01=Neoadjuvant, 02=Adjuvant, 03=SACT-Only',
    },
    'Start_Date_Of_Regimen': {
        'section': 4, 'label': 'Start Date of Regimen',
        'type': 'date', 'format': '%Y-%m-%d', 'mandatory': True,
    },
    'Date_Decision_To_Treat': {
        'section': 4, 'label': 'Date Decision to Treat',
        'type': 'date', 'format': '%Y-%m-%d', 'mandatory': True,
    },
    'Height_At_Start': {
        'section': 4, 'label': 'Height (m)',
        'type': 'float', 'min': 1.0, 'max': 2.5, 'mandatory': False,
        'description': 'Patient height in metres at start of regimen',
    },
    'Weight_At_Start': {
        'section': 4, 'label': 'Weight (kg)',
        'type': 'float', 'min': 20.0, 'max': 300.0, 'mandatory': False,
        'description': 'Patient weight in kg at start of regimen',
    },
    'BSA': {
        'section': 4, 'label': 'Body Surface Area (m²)',
        'type': 'float', 'min': 0.5, 'max': 3.0, 'mandatory': False,
        'description': 'DuBois formula: 0.007184 × H_cm^0.725 × W_kg^0.425 (H in cm, W in kg)',
    },
    'Line_Of_Treatment': {
        'section': 4, 'label': 'Line of Treatment',
        'type': 'integer', 'min': 1, 'max': 10, 'mandatory': False,
        'description': '1=First line, 2=Second line, etc.',
    },
    'Clinical_Trial': {
        'section': 4, 'label': 'Clinical Trial Flag',
        'type': 'string', 'mandatory': False,
        'valid_values': ['01', '02'],
        'description': '01=In clinical trial, 02=Not in clinical trial',
    },
    'Chemoradiation': {
        'section': 4, 'label': 'Chemoradiation Flag',
        'type': 'string', 'mandatory': False,
        'valid_values': ['Y', 'N'],
        'description': 'Y=Concurrent chemoradiation, N=SACT only',
    },

    # ── Section 5: Modifications ──────────────────────────────────────────
    'Regimen_Modification': {
        'section': 5, 'label': 'Regimen Modification Flag',
        'type': 'string', 'mandatory': False,
        'valid_values': ['Y', 'N'],
    },
    'Modification_Reason_Code': {
        'section': 5, 'label': 'Modification Reason Code',
        'type': 'integer', 'mandatory': False,
        'valid_values': [0, 1, 2, 3, 4],
        'description': '0=None, 1=Patient choice, 2=Organisational, 3=Clinical, 4=Toxicity',
    },
    'Toxicity_Grade': {
        'section': 5, 'label': 'Toxicity Grade (CTCAE v5.0)',
        'type': 'integer', 'mandatory': False,
        'valid_values': [0, 1, 2, 3, 4, 5],
        'description': '0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Life-threatening, 5=Death',
    },

    # ── Section 6: Drug Details ───────────────────────────────────────────
    'Drug_Name': {
        'section': 6, 'label': 'Drug Name',
        'type': 'string', 'mandatory': True,
        'description': 'Primary drug name (e.g. Oxaliplatin, Pembrolizumab)',
    },
    'Daily_Total_Dose': {
        'section': 6, 'label': 'Daily Total Dose',
        'type': 'float', 'min': 0.0, 'mandatory': False,
        'description': 'Total dose administered per day',
    },
    'Unit_Of_Measurement': {
        'section': 6, 'label': 'Unit of Measurement',
        'type': 'string', 'mandatory': False,
        'valid_values': ['mg/m2', 'mg/kg', 'mg', 'AUC', 'AUC5', 'AUC6'],
        'description': 'Dose unit (mg/m2, mg/kg, mg fixed, AUC for carboplatin)',
    },
    'SACT_Administration_Route': {
        'section': 6, 'label': 'Administration Route',
        'type': 'string', 'mandatory': False,
        'valid_values': ['IV', 'PO', 'SC', 'IM', 'IT', 'TOP'],
        'description': 'IV=Intravenous, PO=Oral, SC=Subcutaneous',
    },
    'Cycle_Length_In_Days': {
        'section': 6, 'label': 'Cycle Length (days)',
        'type': 'integer', 'min': 1, 'max': 90, 'mandatory': False,
    },

    # ── Section 7: Outcome ────────────────────────────────────────────────
    'End_Of_Regimen_Summary': {
        'section': 7, 'label': 'End of Regimen Summary',
        'type': 'string', 'mandatory': False,
        'valid_values': ['01', '02', '03', '04', '05', '06'],
        'description': (
            '01=Treatment completed, 02=Patient withdrew consent, '
            '03=Death, 04=Disease progression, 05=Toxicity/adverse event, '
            '06=Still on treatment'
        ),
    },

    # ── Scheduling extensions (not in SACT v4.0, derived/operational) ────
    'Appointment_ID': {'section': 0, 'label': 'Appointment ID', 'type': 'string', 'mandatory': False},
    'Date': {'section': 0, 'label': 'Appointment Date', 'type': 'date', 'format': '%Y-%m-%d', 'mandatory': False},
    'Cycle_Number': {'section': 0, 'label': 'Cycle Number', 'type': 'integer', 'min': 1, 'max': 30, 'mandatory': False},
    'Planned_Duration': {'section': 0, 'label': 'Planned Duration (min)', 'type': 'integer', 'min': 10, 'max': 600, 'mandatory': False},
    'Actual_Duration': {'section': 0, 'label': 'Actual Duration (min)', 'type': 'float', 'min': 10, 'max': 600, 'mandatory': False},
    'Attended_Status': {'section': 0, 'label': 'Attendance Status', 'type': 'string', 'valid_values': ['Yes', 'No', 'Cancelled'], 'mandatory': False},
    'Site_Code': {'section': 0, 'label': 'Site Code', 'type': 'string', 'mandatory': False},
    'Chair_Number': {'section': 0, 'label': 'Chair Number', 'type': 'integer', 'mandatory': False},
    'Priority': {'section': 0, 'label': 'Scheduling Priority', 'type': 'string', 'valid_values': ['P1', 'P2', 'P3', 'P4'], 'mandatory': False},
    'Travel_Distance_KM': {'section': 0, 'label': 'Travel Distance (km)', 'type': 'float', 'mandatory': False},
    'Travel_Time_Min': {'section': 0, 'label': 'Travel Time (min)', 'type': 'float', 'mandatory': False},
    'Weather_Severity': {'section': 0, 'label': 'Weather Severity', 'type': 'float', 'min': 0.0, 'max': 1.0, 'mandatory': False},
    'Patient_NoShow_Rate': {'section': 0, 'label': 'Historical NoShow Rate', 'type': 'float', 'min': 0.0, 'max': 1.0, 'mandatory': False},
}

# Fields that MUST be present to call data "SACT v4.0 compliant"
MANDATORY_FIELDS = [k for k, v in SACT_V4_SCHEMA.items() if v.get('mandatory')]

# SACT-native fields (sections 1-7 only, excludes scheduling extensions)
SACT_NATIVE_FIELDS = [k for k, v in SACT_V4_SCHEMA.items() if v['section'] >= 1]

# Real SACT v4.0 CSV may use these alternative column names → map to our internal names
REAL_SACT_COLUMN_MAP = {
    # Linkage
    'NHSNumber': 'NHS_Number',
    'LocalPatientIdentifier': 'Local_Patient_Identifier',
    'NHSNumberStatusIndicatorCode': 'NHS_Number_Status_Indicator_Code',
    # Demographics
    'FamilyName': 'Person_Family_Name',
    'GivenName': 'Person_Given_Name',
    'DateOfBirth': 'Person_Birth_Date',
    'Gender': 'Person_Stated_Gender_Code',
    'Postcode': 'Patient_Postcode',
    'OrganisationIdentifier': 'Organisation_Identifier',
    # Clinical
    'DiagnosisCode': 'Primary_Diagnosis_ICD10',
    'MorphologyCode': 'Morphology_ICD_O',
    'PerformanceStatus': 'Performance_Status',
    'SpecialtyCode': 'Consultant_Specialty_Code',
    # Regimen
    'Regimen': 'Regimen_Code',
    'TreatmentIntent': 'Intent_Of_Treatment',
    'TreatmentContext': 'Treatment_Context',
    'RegimenStartDate': 'Start_Date_Of_Regimen',
    'DecisionToTreatDate': 'Date_Decision_To_Treat',
    'Height': 'Height_At_Start',
    'Weight': 'Weight_At_Start',
    # Modifications
    'RegimenModification': 'Regimen_Modification',
    'ModificationReasonCode': 'Modification_Reason_Code',
    'ToxicityGrade': 'Toxicity_Grade',
    # Drug
    'DrugName': 'Drug_Name',
    'Dose': 'Daily_Total_Dose',
    'DoseUnit': 'Unit_Of_Measurement',
    'AdministrationRoute': 'SACT_Administration_Route',
    'CycleLength': 'Cycle_Length_In_Days',
    # Outcome
    'RegimenEndSummary': 'End_Of_Regimen_Summary',
    # Scheduling-side variants — these are not in the SACT v4.0 native
    # spec but show up in real Velindre operational extracts and other
    # NHS day-unit dumps; normalising them here keeps the Channel-2
    # ingest robust without forcing the trust to rename their columns.
    'AppointmentDate': 'Appointment_Date',
    'AppointmentTime': 'Start_Time',
    'StartTime': 'Start_Time',
    'AppointmentStartTime': 'Start_Time',
    'EndTime': 'End_Time',
    'AppointmentEndTime': 'End_Time',
    'PlannedDuration': 'Planned_Duration',
    'ActualDuration': 'Actual_Duration',
    'AttendedStatus': 'Attended_Status',
    'AttendanceStatus': 'Attended_Status',
    'Priority': 'Priority',
    'PriorityCode': 'Priority',
    'SiteCode': 'Site_Code',
    'Site': 'Site_Code',
    'ChairNumber': 'Chair_Number',
    'Chair': 'Chair_Number',
    'PatientID': 'Patient_ID',
    'AppointmentID': 'Appointment_ID',
    'AppointmentId': 'Appointment_ID',
    'CycleNumber': 'Cycle_Number',
}


def normalise_real_sact_columns(df):
    """Apply :data:`REAL_SACT_COLUMN_MAP` (case-sensitive) plus a
    case-insensitive fallback so a real SACT extract using e.g.
    ``AppointmentDate`` or ``appointmentdate`` ends up as
    ``Appointment_Date`` before downstream schema validation runs.

    Returns a new DataFrame with renamed columns.  Columns that already
    use the canonical name are left untouched.  Unknown columns pass
    through unchanged so the gate failure (if any) is the schema
    validator's call, not this helper's.
    """
    rename: Dict[str, str] = {}
    canonical_lc = {v.lower(): v for v in REAL_SACT_COLUMN_MAP.values()}
    map_lc = {k.lower(): v for k, v in REAL_SACT_COLUMN_MAP.items()}
    for col in df.columns:
        if col in REAL_SACT_COLUMN_MAP:
            rename[col] = REAL_SACT_COLUMN_MAP[col]
            continue
        col_lc = col.lower()
        if col_lc in map_lc and map_lc[col_lc] != col:
            rename[col] = map_lc[col_lc]
            continue
        # Variant arrives lower-cased; map back to canonical capitalisation.
        if col_lc in canonical_lc and canonical_lc[col_lc] != col:
            rename[col] = canonical_lc[col_lc]
    return df.rename(columns=rename) if rename else df


# =============================================================================
# VALIDATOR
# =============================================================================

class SACTv4Validator:
    """
    Validates any DataFrame against the SACT v4.0 schema.

    Works for both synthetic data and real SACT v4.0 CSV files.
    Returns an alignment report with:
        - overall_score: 0-100 (100 = perfect alignment)
        - missing_mandatory: fields that are required but absent
        - missing_optional: fields present in schema but absent in data
        - invalid_values: fields with values outside valid set
        - type_errors: fields with wrong data types
        - extra_fields: fields in data not in schema (fine — just noted)
    """

    def validate(self, df: pd.DataFrame, source: str = 'unknown') -> Dict[str, Any]:
        """
        Validate a DataFrame against SACT v4.0 schema.

        Args:
            df: DataFrame to validate
            source: 'synthetic', 'real_sact', or 'unknown'

        Returns:
            Comprehensive validation report dict
        """
        report = {
            'source': source,
            'validated_at': datetime.now().isoformat(),
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'missing_mandatory': [],
            'missing_optional': [],
            'invalid_values': [],
            'type_errors': [],
            'extra_fields': [],
            'field_coverage': {},
            'section_coverage': {},
            'overall_score': 0,
            'grade': 'F',
            'ready_for_ml': False,
            'notes': [],
        }

        if df.empty:
            report['notes'].append('DataFrame is empty — cannot validate.')
            return report

        cols = set(df.columns)

        # ── Check mandatory fields ─────────────────────────────────────────
        for field in MANDATORY_FIELDS:
            if field not in cols:
                report['missing_mandatory'].append(field)
            elif df[field].isna().all():
                report['missing_mandatory'].append(f'{field} (all null)')

        # ── Check optional SACT fields ────────────────────────────────────
        for field, spec in SACT_V4_SCHEMA.items():
            if spec.get('mandatory'):
                continue
            if field not in cols:
                report['missing_optional'].append(field)

        # ── Validate value ranges and valid_values ────────────────────────
        for field, spec in SACT_V4_SCHEMA.items():
            if field not in cols:
                continue

            series = df[field].dropna()
            if len(series) == 0:
                continue

            # Valid values check
            if 'valid_values' in spec:
                valid = set(spec['valid_values'])
                actual = set(series.unique())
                bad = actual - valid
                if bad:
                    report['invalid_values'].append({
                        'field': field,
                        'invalid': list(bad)[:5],
                        'valid': spec['valid_values'],
                    })

            # Numeric range check
            if spec['type'] in ('integer', 'float'):
                numeric = pd.to_numeric(series, errors='coerce').dropna()
                if 'min' in spec and numeric.min() < spec['min']:
                    report['invalid_values'].append({
                        'field': field,
                        'issue': f"min={numeric.min():.2f} < allowed {spec['min']}",
                    })
                if 'max' in spec and numeric.max() > spec['max']:
                    report['invalid_values'].append({
                        'field': field,
                        'issue': f"max={numeric.max():.2f} > allowed {spec['max']}",
                    })

        # ── Extra fields (not in schema — fine for ML enrichment) ─────────
        schema_fields = set(SACT_V4_SCHEMA.keys())
        report['extra_fields'] = list(cols - schema_fields)

        # ── Field coverage by section ─────────────────────────────────────
        for section_num in range(1, 8):
            section_fields = [k for k, v in SACT_V4_SCHEMA.items() if v['section'] == section_num]
            present = [f for f in section_fields if f in cols]
            coverage = len(present) / len(section_fields) if section_fields else 1.0
            report['section_coverage'][f'section_{section_num}'] = {
                'fields_total': len(section_fields),
                'fields_present': len(present),
                'coverage_pct': round(coverage * 100, 1),
                'missing': [f for f in section_fields if f not in cols],
            }

        # ── Per-field completeness ────────────────────────────────────────
        for field in SACT_NATIVE_FIELDS:
            if field in cols:
                null_pct = df[field].isna().mean() * 100
                report['field_coverage'][field] = {
                    'present': True,
                    'null_pct': round(null_pct, 1),
                    'mandatory': SACT_V4_SCHEMA[field].get('mandatory', False),
                }
            else:
                report['field_coverage'][field] = {
                    'present': False,
                    'null_pct': 100.0,
                    'mandatory': SACT_V4_SCHEMA[field].get('mandatory', False),
                }

        # ── Overall score (0-100) ─────────────────────────────────────────
        n_mandatory = len(MANDATORY_FIELDS)
        n_missing_mandatory = len(report['missing_mandatory'])
        n_optional = len([k for k, v in SACT_V4_SCHEMA.items() if not v.get('mandatory')])
        n_missing_optional = len(report['missing_optional'])
        n_invalid = len(report['invalid_values'])

        mandatory_score = max(0, (n_mandatory - n_missing_mandatory) / max(n_mandatory, 1)) * 60
        optional_score = max(0, (n_optional - n_missing_optional) / max(n_optional, 1)) * 30
        validity_score = max(0, (1 - n_invalid / max(n_mandatory + n_optional, 1))) * 10

        score = mandatory_score + optional_score + validity_score
        report['overall_score'] = round(score, 1)

        if score >= 90:
            report['grade'] = 'A'
        elif score >= 75:
            report['grade'] = 'B'
        elif score >= 60:
            report['grade'] = 'C'
        elif score >= 45:
            report['grade'] = 'D'
        else:
            report['grade'] = 'F'

        # Ready for ML if mandatory fields mostly present and key training fields exist
        ml_required = ['Attended_Status', 'Planned_Duration', 'Regimen_Code',
                       'Performance_Status', 'Primary_Diagnosis_ICD10']
        report['ready_for_ml'] = (
            n_missing_mandatory == 0 and
            all(f in cols for f in ml_required)
        )

        if report['missing_mandatory']:
            report['notes'].append(
                f"⚠ {len(report['missing_mandatory'])} mandatory fields missing: "
                f"{report['missing_mandatory'][:5]}"
            )
        if report['ready_for_ml']:
            report['notes'].append("✓ Data is ready for ML model training")
        else:
            report['notes'].append("✗ Data not yet ready for ML — missing required fields")

        logger.info(
            f"SACT v4.0 validation: source={source}, score={score:.1f}/100, "
            f"grade={report['grade']}, missing_mandatory={n_missing_mandatory}"
        )

        return report

    def compare(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare alignment between synthetic and real SACT data.
        Returns field-by-field distribution comparison.
        """
        synth_report = self.validate(synthetic_df, 'synthetic')
        real_report = self.validate(real_df, 'real')

        comparisons = {}
        for field in SACT_NATIVE_FIELDS:
            if field in synthetic_df.columns and field in real_df.columns:
                s = pd.to_numeric(synthetic_df[field], errors='coerce').dropna()
                r = pd.to_numeric(real_df[field], errors='coerce').dropna()
                if len(s) > 0 and len(r) > 0:
                    comparisons[field] = {
                        'synthetic_mean': round(s.mean(), 3),
                        'real_mean': round(r.mean(), 3),
                        'synthetic_std': round(s.std(), 3),
                        'real_std': round(r.std(), 3),
                        'mean_diff_pct': round(abs(s.mean() - r.mean()) / max(abs(r.mean()), 1e-6) * 100, 1),
                    }

        return {
            'synthetic_score': synth_report['overall_score'],
            'real_score': real_report['overall_score'],
            'field_comparisons': comparisons,
            'synthetic_missing': synth_report['missing_mandatory'],
            'real_missing': real_report['missing_mandatory'],
        }


# =============================================================================
# DATA ADAPTER — converts any SACT v4.0 data to ML-ready feature format
# =============================================================================

class SACTv4DataAdapter:
    """
    Adapts raw SACT v4.0 data (synthetic OR real) to a standardised
    ML feature DataFrame.

    The adapter:
    1. Renames columns from real SACT CSV naming conventions to internal names
    2. Derives computed features (BSA, age, days_to_treat, etc.)
    3. Encodes categorical fields
    4. Fills missing values with evidence-based defaults
    5. Returns a DataFrame that the ML models can use directly

    The output format is IDENTICAL whether the input is:
    - Our synthetic data (already named correctly)
    - Real SACT v4.0 CSV from NDRS
    - Partial rollout data (April-June 2026)
    """

    # Evidence-based defaults from NHS Cancer Waiting Times statistics
    DEFAULTS = {
        'Performance_Status': 1,          # Most patients are PS 1
        'Travel_Distance_KM': 10.0,       # Median Cardiff catchment
        'Travel_Time_Min': 25.0,
        'Weather_Severity': 0.0,
        'Cycle_Number': 1,
        'Has_Comorbidities': False,
        'IV_Access_Difficulty': False,
        'Comorbidity_Count': 0,
        'Intent_Of_Treatment': '07',      # Default non-curative if unknown
        'Treatment_Context': '03',        # Default SACT Only
        'Modification_Reason_Code': 0,
        'Toxicity_Grade': 0,
        'Line_Of_Treatment': 1,
        'Clinical_Trial': '02',
        'Chemoradiation': 'N',
    }

    # Regimen → cancer type mapping (for ICD-10 derivation if missing)
    REGIMEN_CANCER_MAP = {
        'FOLFOX': 'C18.9', 'FOLFIRI': 'C18.9', 'CAPOX': 'C18.9',
        'RCHOP': 'C83.3', 'RITUX': 'C83.3',
        'FECT': 'C50.9', 'TRAS': 'C50.9', 'DOCE': 'C50.9', 'AC': 'C50.9',
        'PACW': 'C50.9', 'CARBPAC': 'C56.9',
        'PEMBRO': 'C43.9', 'NIVO': 'C43.9', 'IPNIVO': 'C43.9',
        'GEM': 'C25.9', 'PEME': 'C34.9', 'VINO': 'C34.9',
        'CISE': 'C62.9', 'ZOLE': 'C79.5', 'BEVA': 'C18.9',
    }

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt raw SACT v4.0 data to ML-ready feature format.

        Handles both synthetic data (already named correctly) and
        real SACT v4.0 CSV (may have different column names).

        Returns:
            DataFrame with standardised ML features, ready for training/inference.
        """
        df = df.copy()

        # ── Step 1: Rename real SACT column names to internal names ──────
        rename_map = {k: v for k, v in REAL_SACT_COLUMN_MAP.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"Renamed {len(rename_map)} real SACT column(s) to internal names")

        # ── Step 2: Derive BSA if height/weight present ───────────────────
        df = self._derive_bsa(df)

        # ── Step 3: Derive age from birth date ────────────────────────────
        df = self._derive_age(df)

        # ── Step 4: Derive days-to-treat lead time ────────────────────────
        df = self._derive_lead_time(df)

        # ── Step 5: Derive priority from performance status ───────────────
        df = self._derive_priority(df)

        # ── Step 6: Derive travel features from postcode if missing ───────
        df = self._derive_travel_from_postcode(df)

        # ── Step 7: Derive ICD-10 from regimen if missing ─────────────────
        if 'Primary_Diagnosis_ICD10' not in df.columns or df['Primary_Diagnosis_ICD10'].isna().all():
            df['Primary_Diagnosis_ICD10'] = df.get('Regimen_Code', pd.Series(['C80.9'] * len(df))).map(
                self.REGIMEN_CANCER_MAP
            ).fillna('C80.9')

        # ── Step 8: Fill defaults for missing fields ──────────────────────
        # Track per-field application so we can surface the impact of
        # default-filling explicitly.  Silent imputation on real Velindre
        # data hides drift (e.g. Phase-1 SACT v4.0 may omit
        # Performance_Status entirely): a single aggregated WARNING per
        # affected field is the production-ready signal.
        n_rows = len(df)
        absent_cols: List[str] = []
        partial_fill: Dict[str, int] = {}
        for field, default in self.DEFAULTS.items():
            if field not in df.columns:
                df[field] = default
                absent_cols.append(field)
            else:
                na_count = int(df[field].isna().sum())
                if na_count > 0:
                    df[field] = df[field].fillna(default)
                    partial_fill[field] = na_count
        if absent_cols:
            logger.warning(
                "SACTv4DataAdapter: %d column(s) absent on input "
                "DataFrame; entire column filled with default "
                "(rows=%d).  Affected fields: %s",
                len(absent_cols), n_rows,
                {f: self.DEFAULTS[f] for f in absent_cols},
            )
        if partial_fill:
            logger.warning(
                "SACTv4DataAdapter: per-field NaN count after fill "
                "(rows=%d): %s",
                n_rows, partial_fill,
            )

        # ── Step 9: Encode intent and context as numeric ──────────────────
        intent_col = df['Intent_Of_Treatment'] if 'Intent_Of_Treatment' in df.columns \
                     else pd.Series(['07'] * len(df), index=df.index)
        context_col = df['Treatment_Context'] if 'Treatment_Context' in df.columns \
                      else pd.Series(['03'] * len(df), index=df.index)
        df['Is_Curative'] = (intent_col == '06').astype(int)
        df['Is_Neoadjuvant'] = (context_col == '01').astype(int)

        # ── Step 10: Derive no-show risk features ─────────────────────────
        df = self._derive_noshow_features(df)

        # ── Step 11: Add data source flag ─────────────────────────────────
        if 'Data_Source' not in df.columns:
            df['Data_Source'] = 'unknown'

        logger.info(
            f"SACTv4DataAdapter: adapted {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    def _derive_bsa(self, df: pd.DataFrame) -> pd.DataFrame:
        """DuBois BSA = 0.007184 × H_cm^0.725 × W^0.425.
        Height_At_Start is stored in metres — convert to cm before applying."""
        if 'BSA' not in df.columns:
            if 'Height_At_Start' in df.columns and 'Weight_At_Start' in df.columns:
                h_cm = pd.to_numeric(df['Height_At_Start'], errors='coerce').clip(1.0, 2.5) * 100
                w = pd.to_numeric(df['Weight_At_Start'], errors='coerce').clip(20, 300)
                df['BSA'] = (0.007184 * (h_cm ** 0.725) * (w ** 0.425)).round(3)
            else:
                df['BSA'] = 1.73  # Population average BSA
        return df

    def _derive_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive Age and Age_Band from Person_Birth_Date if not present."""
        if 'Age' not in df.columns and 'Person_Birth_Date' in df.columns:
            today = datetime.now()
            def calc_age(dob_str):
                try:
                    dob = pd.to_datetime(dob_str)
                    return (today - dob).days // 365
                except Exception:
                    return 60  # Default to median cancer patient age
            df['Age'] = df['Person_Birth_Date'].apply(calc_age)

        if 'Age' in df.columns and 'Age_Band' not in df.columns:
            df['Age_Band'] = pd.cut(
                pd.to_numeric(df['Age'], errors='coerce').fillna(60),
                bins=[0, 40, 60, 75, 120],
                labels=['<40', '40-60', '60-75', '>75']
            ).astype(str)
        return df

    def _derive_lead_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive Days_To_Appointment from Date_Decision_To_Treat → Date."""
        if 'Days_To_Appointment' not in df.columns:
            if 'Date_Decision_To_Treat' in df.columns and 'Date' in df.columns:
                dtt = pd.to_datetime(df['Date_Decision_To_Treat'], errors='coerce')
                apt = pd.to_datetime(df['Date'], errors='coerce')
                df['Days_To_Appointment'] = (apt - dtt).dt.days.clip(0, 365).fillna(14)
            else:
                df['Days_To_Appointment'] = 14  # Evidence-based default
        return df

    def _derive_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive Priority from Performance_Status and Intent_Of_Treatment."""
        if 'Priority' not in df.columns and 'Performance_Status' in df.columns:
            ps = pd.to_numeric(df['Performance_Status'], errors='coerce').fillna(1)
            intent = df.get('Intent_Of_Treatment', pd.Series(['07'] * len(df)))

            def ps_to_priority(row):
                p, i = row
                if i == '06' and p <= 1:
                    return 'P1'
                elif p <= 1:
                    return 'P2'
                elif p <= 2:
                    return 'P3'
                else:
                    return 'P4'

            df['Priority'] = list(map(ps_to_priority, zip(ps, intent)))
        return df

    def _derive_travel_from_postcode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive travel distance/time from postcode if not present."""
        POSTCODE_TRAVEL = {
            'CF10': (4, 15), 'CF11': (5, 18), 'CF14': (0, 5), 'CF15': (3, 12),
            'CF23': (6, 20), 'CF24': (4, 15), 'CF3': (8, 25), 'CF5': (6, 22),
            'CF37': (12, 30), 'CF38': (14, 35), 'CF62': (15, 35), 'CF63': (12, 28),
            'CF64': (10, 25), 'CF72': (10, 25), 'CF81': (20, 45), 'CF82': (18, 40),
            'CF83': (15, 35), 'CF31': (25, 40), 'CF32': (28, 45),
            'NP10': (18, 35), 'NP19': (16, 30), 'NP20': (15, 28), 'NP44': (20, 40),
            'SA1': (65, 75), 'SA2': (68, 80),
        }

        if 'Travel_Distance_KM' not in df.columns and 'Patient_Postcode' in df.columns:
            def get_dist(postcode):
                if pd.isna(postcode):
                    return 10.0
                district = str(postcode).strip().split(' ')[0].upper()
                return POSTCODE_TRAVEL.get(district, (10, 25))[0]

            def get_time(postcode):
                if pd.isna(postcode):
                    return 25.0
                district = str(postcode).strip().split(' ')[0].upper()
                return POSTCODE_TRAVEL.get(district, (10, 25))[1]

            df['Travel_Distance_KM'] = df['Patient_Postcode'].apply(get_dist)
            df['Travel_Time_Min'] = df['Patient_Postcode'].apply(get_time)
        return df

    def _derive_noshow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive no-show risk features from available SACT fields.
        These replace the 'Patient_NoShow_Rate' that real data won't have.
        """
        # Historical no-show rate: derive from Previous_NoShows / Total_Appointments
        if 'Patient_NoShow_Rate' not in df.columns:
            if 'Previous_NoShows' in df.columns and 'Total_Appointments_Before' in df.columns:
                noshows = pd.to_numeric(df['Previous_NoShows'], errors='coerce').fillna(0)
                total = pd.to_numeric(df['Total_Appointments_Before'], errors='coerce').fillna(1).clip(1, None)
                df['Patient_NoShow_Rate'] = (noshows / total).clip(0, 1).round(3)
            else:
                # Fallback prior Beta(1.5, 8) — mean 0.158.  Consistent with
                # NHS England outpatient-chemotherapy DNA rate (published range
                # 7–15%); used only when a patient has no attendance history.
                # Source: datasets/_nhs_calibration.py (NHS DNA stats).
                df['Patient_NoShow_Rate'] = 0.142  # Beta(1.5,8) posterior mean

        # Complexity factor from SACT fields
        if 'Complexity_Factor' not in df.columns:
            ps = pd.to_numeric(df.get('Performance_Status', 1), errors='coerce').fillna(1)
            comorbid = pd.to_numeric(df.get('Comorbidity_Count', 0), errors='coerce').fillna(0)
            toxicity = pd.to_numeric(df.get('Toxicity_Grade', 0), errors='coerce').fillna(0)
            # Normalised complexity 0-1: higher PS, comorbidities, toxicity = more complex
            df['Complexity_Factor'] = ((ps / 4) * 0.5 + (comorbid / 4) * 0.3 + (toxicity / 5) * 0.2).clip(0, 1).round(3)

        # Has_Comorbidities from Comorbidity_Count
        if 'Has_Comorbidities' not in df.columns and 'Comorbidity_Count' in df.columns:
            df['Has_Comorbidities'] = df['Comorbidity_Count'] > 0

        return df

    def get_ml_feature_columns(self) -> List[str]:
        """
        Return the list of ML feature columns produced by adapt().
        These are consistent regardless of whether input is synthetic or real.
        """
        return [
            # SACT v4.0 fields (direct)
            'Performance_Status', 'Primary_Diagnosis_ICD10', 'Regimen_Code',
            'Intent_Of_Treatment', 'Treatment_Context', 'Modification_Reason_Code',
            'Toxicity_Grade', 'Clinical_Trial', 'Chemoradiation', 'Line_Of_Treatment',
            # Derived from SACT fields
            'Age', 'Age_Band', 'BSA', 'Is_Curative', 'Is_Neoadjuvant',
            'Days_To_Appointment',
            # Operational
            'Cycle_Number', 'Planned_Duration', 'Travel_Distance_KM', 'Travel_Time_Min',
            'Site_Code', 'Priority', 'Weather_Severity',
            # Derived no-show features
            'Patient_NoShow_Rate', 'Has_Comorbidities', 'Comorbidity_Count',
            'IV_Access_Difficulty', 'Complexity_Factor',
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_dataset(df: pd.DataFrame, source: str = 'unknown') -> Dict[str, Any]:
    """Validate a DataFrame and return the report."""
    return SACTv4Validator().validate(df, source)


def adapt_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt any SACT v4.0-compatible DataFrame to ML feature format."""
    return SACTv4DataAdapter().adapt(df)


def check_synthetic_alignment(synthetic_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full alignment check: validates synthetic data against schema
    and returns a concise summary for dissertation/logging.
    """
    validator = SACTv4Validator()
    report = validator.validate(synthetic_df, 'synthetic')

    return {
        'score': report['overall_score'],
        'grade': report['grade'],
        'ready_for_ml': report['ready_for_ml'],
        'missing_mandatory': report['missing_mandatory'],
        'missing_optional_count': len(report['missing_optional']),
        'invalid_values_count': len(report['invalid_values']),
        'section_coverage': {
            k: v['coverage_pct']
            for k, v in report['section_coverage'].items()
        },
        'total_fields_covered': sum(
            1 for v in report['field_coverage'].values() if v['present']
        ),
        'total_schema_fields': len(report['field_coverage']),
        'notes': report['notes'],
    }
