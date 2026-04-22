# Real Hospital Data Directory

## For Velindre Cancer Centre Production Data

Place real patient and appointment data here. The system will automatically detect and use it instead of synthetic data.

## How to Activate

1. Place your Excel/CSV files in this directory
2. Restart the Flask server (`python flask_app.py`)
3. The system will retrain all 12 ML models on real data

## Required Files (SACT v4.0 Field Names)

### patients.xlsx

| Column | SACT v4.0 Name | Type | Required | Example |
|--------|---------------|------|----------|---------|
| NHS_Number | NHS_NUMBER | String(10) | Mandatory | 4567890123 |
| Local_Patient_Identifier | LOCAL_PATIENT_IDENTIFIER | String | Mandatory | P12345 |
| Person_Given_Name | PERSON_GIVEN_NAME | String | Required | John |
| Person_Family_Name | PERSON_FAMILY_NAME | String | Required | Davies |
| Person_Birth_Date | PERSON_BIRTH_DATE | Date | Required | 1960-05-15 |
| Person_Stated_Gender_Code | PERSON_STATED_GENDER_CODE | Integer | Required | 1=M, 2=F, 9=NS |
| Patient_Postcode | POSTCODE_OF_USUAL_ADDRESS | String | Required | CF14 4XW |
| Organisation_Identifier | ORGANISATION_IDENTIFIER | String | Required | RQF |
| Primary_Diagnosis_ICD10 | PRIMARY_DIAGNOSIS_(ICD) | String | Required | C50.9 |
| Performance_Status | PERFORMANCE_STATUS_ADULT | Integer | Required | 0-4 (WHO) |
| Regimen_Code | REGIMEN | String | Mandatory | FOLFOX |
| Intent_Of_Treatment | INTENT_OF_TREATMENT | String(2) | Required | 06=Curative, 07=Non-curative |
| Cycle_Number | - | Integer | Required | 1-20 |
| Priority | - | String | Required | P1-P4 |
| Height_At_Start | HEIGHT_AT_START_OF_REGIMEN | Float | Required | 1.72 (metres) |
| Weight_At_Start | WEIGHT_AT_START_OF_REGIMEN | Float | Required | 78.5 (kg) |

Also accepted (optional but helpful for ML):
- Travel_Distance_KM, Travel_Time_Min
- Previous_NoShows, Previous_Cancellations
- Has_Comorbidities, IV_Access_Difficulty
- Contact_Preference (SMS/Phone/Email/Post)

### historical_appointments.xlsx

This is the most important file for ML training. Each row = one past appointment with outcome.

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| Patient_ID | String | Yes | Links to patients |
| Date | Date (YYYY-MM-DD) | Yes | Appointment date |
| Time | String (HH:MM) | Yes | Start time |
| Attended_Status | Categorical | Yes | **Yes / No / Cancelled** — this is the ML target |
| Actual_Duration | Integer | Yes | Actual minutes (for duration model) |
| Planned_Duration | Integer | Yes | Scheduled minutes |
| Regimen_Code | String | Yes | FOLFOX, RCHOP, etc. |
| Site_Code | String | Yes | WC, NP, BGD, CWM, SA |
| Cycle_Number | Integer | Yes | Treatment cycle |
| Weather_Severity | Float | Optional | 0.0-1.0 |
| Cancellation_Reason | String | Optional | Patient/Weather/Medical |

**Minimum 500 records recommended for ML training. More is better.**

### appointments.xlsx

Upcoming scheduled appointments.

| Column | Type | Required |
|--------|------|----------|
| Appointment_ID | String | Yes |
| Patient_ID | String | Yes |
| Date | Date | Yes |
| Start_Time | String (HH:MM) | Yes |
| End_Time | String (HH:MM) | Yes |
| Duration_Minutes | Integer | Yes |
| Site_Code | String | Yes |
| Chair_Number | Integer | Yes |
| Regimen_Code | String | Yes |
| Priority | String | Yes |

## Column Name Flexibility

The system supports both SACT v4.0 names and common alternatives:

| Accepted Names | Maps To |
|---------------|---------|
| Patient_ID, patient_id | Patient identifier |
| Priority, priority | P1-P4 |
| Regimen_Code, protocol | Treatment protocol |
| Postcode_District, postcode | Patient location |
| Previous_NoShows, no_shows | Historical no-shows |

## Data Security

**This directory may contain patient identifiable information (PII).**

Ensure compliance with:
- NHS Data Security and Protection Toolkit (Level 2+)
- UK GDPR (Article 9 — special category health data)
- Caldicott Principles
- Velindre data sharing agreement

**DO NOT commit real patient data to version control.**

The `.gitignore` already excludes `datasets/real_data/*.xlsx` and `datasets/real_data/*.csv`.
