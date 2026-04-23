"""
Tests for datasets/generate_sample_data.py — production-readiness Wave 3.6.

This module contains pure data generators backed by random.Random and
np.random.  Tests seed both RNGs for determinism, keep row counts small
(<=20 patients) so each test completes well under 10s, and assert
returned DataFrame column names / row counts / value ranges.

Structure:
- TestHelpers                 — get_age_band, generate_nhs_number, generate_patient_id
- TestGenerateRegimens        — regimens.xlsx schema
- TestGenerateSites           — sites.xlsx schema
- TestGeneratePatients        — patient DataFrame contract (size + key columns)
- TestHistoricalAppointments  — appointments derived from patients
- TestStaffAndMetrics         — generate_staff + generate_historical_metrics
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import generate_sample_data as gsd  # noqa: E402


@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed both RNGs for deterministic output per test."""
    random.seed(42)
    np.random.seed(42)
    yield


# --------------------------------------------------------------------------- #
# 1. Small pure helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_age_band_boundaries(self):
        assert gsd.get_age_band(25) == "<40"
        assert gsd.get_age_band(39) == "<40"
        assert gsd.get_age_band(40) == "40-60"
        assert gsd.get_age_band(59) == "40-60"
        assert gsd.get_age_band(60) == "60-75"
        assert gsd.get_age_band(74) == "60-75"
        assert gsd.get_age_band(80) == ">75"

    def test_nhs_number_is_10_digits_and_mod11_valid(self):
        # 10 numbers is enough to exercise the retry loop (check==10 case)
        for _ in range(10):
            n = gsd.generate_nhs_number()
            assert len(n) == 10 and n.isdigit()
            # First digit must be 4-7 (England range per source doc)
            assert n[0] in "4567"
            digits = [int(c) for c in n]
            total = sum(d * w for d, w in zip(digits[:9], [10, 9, 8, 7, 6, 5, 4, 3, 2]))
            rem = total % 11
            check = 0 if (11 - rem) == 11 else 11 - rem
            assert check == digits[9], f"mod-11 fail on {n}"

    def test_patient_id_has_expected_format(self):
        pid = gsd.generate_patient_id()
        assert pid.startswith("P")
        assert pid[1:].isdigit()
        assert len(pid) == 6  # P + 5 digits


# --------------------------------------------------------------------------- #
# 2. Regimen schema
# --------------------------------------------------------------------------- #


class TestGenerateRegimens:
    def test_regimens_dataframe_schema(self):
        df = gsd.generate_regimens()
        assert isinstance(df, pd.DataFrame)
        # One row per entry in REGIMENS
        assert len(df) == len(gsd.REGIMENS)
        expected_cols = {
            "Regimen_Code",
            "Regimen_Name",
            "Cycle_Length_Days",
            "Treatment_Days",
            "Duration_C1",
            "Duration_C2",
            "Duration_C3_Plus",
            "Nursing_Ratio",
            "Pharmacy_Lead_Time",
            "Long_Infusion",
            "Cancer_Type",
        }
        assert expected_cols.issubset(df.columns)
        # Cycle length is always a positive integer
        assert (df["Cycle_Length_Days"] > 0).all()


# --------------------------------------------------------------------------- #
# 3. Site schema
# --------------------------------------------------------------------------- #


class TestGenerateSites:
    def test_sites_dataframe_shape(self):
        df = gsd.generate_sites()
        assert len(df) == len(gsd.SITES)
        assert set(df.columns) >= {
            "Site_Code", "Site_Name", "Chairs", "Recliners",
            "Operating_Hours", "Nurses_AM", "Nurses_PM",
            "Latitude", "Longitude",
        }
        # Chairs & nurses must be non-negative
        assert (df["Chairs"] >= 0).all()
        assert (df["Nurses_AM"] >= 0).all()
        # WC is the main site and must be present
        assert "WC" in df["Site_Code"].tolist()


# --------------------------------------------------------------------------- #
# 4. Patient generation
# --------------------------------------------------------------------------- #


class TestGeneratePatients:
    def test_generates_correct_count_with_valid_fields(self):
        df = gsd.generate_patients(n_patients=15)
        assert len(df) == 15
        # SACT v4.0 mandatory linkage fields
        required_cols = {
            "NHS_Number",
            "Local_Patient_Identifier",
            "Patient_ID",
            "Person_Family_Name",
            "Person_Given_Name",
            "Regimen_Code",
            "Postcode_District",
            "Age",
            "Age_Band",
            "Priority",
            "Historical_NoShow_Rate",
        }
        assert required_cols.issubset(df.columns), (
            f"missing cols: {required_cols - set(df.columns)}"
        )
        # Value-range assertions
        assert df["Age"].between(25, 85).all()
        assert df["Historical_NoShow_Rate"].between(0, 1).all()
        assert df["Priority"].isin({"P1", "P2", "P3", "P4"}).all()
        assert df["Regimen_Code"].isin(gsd.REGIMENS).all()
        assert df["Postcode_District"].isin(gsd.POSTCODES).all()
        # NHS numbers are all 10-digit
        assert df["NHS_Number"].astype(str).str.len().eq(10).all()


# --------------------------------------------------------------------------- #
# 5. Historical appointments derived from patients
# --------------------------------------------------------------------------- #


class TestHistoricalAppointments:
    def test_historical_rows_follow_patient_regimens(self):
        patients = gsd.generate_patients(n_patients=10)
        hist = gsd.generate_historical_appointments(patients, n_months=2)
        assert isinstance(hist, pd.DataFrame)
        # Each appointment references a real patient
        assert set(hist["Patient_ID"]).issubset(set(patients["Patient_ID"]))
        # Key columns present
        assert {
            "Appointment_ID", "Patient_ID", "Date", "Site_Code",
            "Regimen_Code", "Cycle_Number", "Planned_Duration",
            "Attended_Status", "Showed_Up",
        }.issubset(hist.columns)
        # Planned duration must be positive integer
        assert (hist["Planned_Duration"] > 0).all()
        # Attended_Status in the documented vocabulary
        assert hist["Attended_Status"].isin({"Yes", "No", "Cancelled"}).all()
        # Dates sorted (generator calls .sort_values('Date'))
        assert list(hist["Date"]) == sorted(hist["Date"])


# --------------------------------------------------------------------------- #
# 6. Staff + metrics
# --------------------------------------------------------------------------- #


class TestStaffAndMetrics:
    def test_staff_roster_totals_expected_count(self):
        df = gsd.generate_staff()
        # Sum of all counts in the roles tuple
        expected_total = 20 + 6 + 2 + 4 + 6 + 8 + 4 + 10 + 5 + 3
        assert len(df) == expected_total
        # Staff_ID is unique
        assert df["Staff_ID"].is_unique
        # Site codes belong to SITES
        assert df["Site_Code"].isin({s["code"] for s in gsd.SITES}).all()

    def test_metrics_generator_shape(self):
        df = gsd.generate_historical_metrics(n_days=14)
        # Skips weekends: ~10 weekdays in 14 consecutive days
        expected_min = 5 * len(gsd.SITES)
        expected_max = 14 * len(gsd.SITES)
        assert expected_min <= len(df) <= expected_max
        assert (df["Chair_Utilization"] <= 1.0).all()
        assert (df["Total_Appointments"] > 0).all()
