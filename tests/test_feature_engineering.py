"""
Tests for ml/feature_engineering.py — Wave 3.1.2 (T3 coverage).

Covers:
- single-patient feature creation (PatientFeatures dataclass shape)
- history features (rates, counts, new-patient flag)
- temporal features (day-of-week, time-slot one-hot)
- geographic distance to site
- treatment + external features
- batch DataFrame builder
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from ml.feature_engineering import FeatureEngineer, PatientFeatures


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _patient(pid="P001", *, history=10, no_shows=2, cancellations=1,
             postcode="CF14"):
    return {
        "patient_id": pid,
        "total_appointments": history,
        "no_shows": no_shows,
        "cancellations": cancellations,
        "late_arrivals": 0,
        "last_visit_date": (datetime.now() - timedelta(days=15)).isoformat(),
        "age": 65,
        "Age_Band": "60-75",
        "postcode": postcode,
        "lat": 51.48,
        "lon": -3.18,
    }


def _appointment(when=None, *, priority="P2", protocol="FOLFOX"):
    when = when or (datetime.now() + timedelta(days=7)).replace(hour=10, minute=0)
    # FeatureEngineer reads `appointment_time` (or `Date`); set both for
    # parity with both code paths it supports.
    return {
        "appointment_time": when.isoformat(),
        "Date": when.isoformat(),
        "site_code": "WC",
        "duration_minutes": 60,
        "priority": priority,
        "protocol": protocol,
        "Days_To_Appointment": 7,
    }


# --------------------------------------------------------------------------- #
# create_patient_features
# --------------------------------------------------------------------------- #


class TestCreatePatientFeatures:
    def test_returns_patient_features(self):
        fe = FeatureEngineer()
        out = fe.create_patient_features(_patient(), _appointment())
        assert isinstance(out, PatientFeatures)
        assert out.patient_id == "P001"
        assert isinstance(out.features, dict)
        assert len(out.feature_names) == len(out.features)

    def test_history_rate_correct(self):
        fe = FeatureEngineer()
        out = fe.create_patient_features(
            _patient(history=10, no_shows=2), _appointment(),
        )
        # 2/10 = 0.2
        assert out.features["prev_noshow_rate"] == pytest.approx(0.2)
        assert out.features["prev_noshow_count"] == pytest.approx(2.0)

    def test_new_patient_flag(self):
        fe = FeatureEngineer()
        out = fe.create_patient_features(
            _patient(history=0, no_shows=0), _appointment(),
        )
        assert out.features["is_new_patient"] == 1.0


# --------------------------------------------------------------------------- #
# Temporal features
# --------------------------------------------------------------------------- #


class TestTemporalFeatures:
    def test_day_of_week_flags(self):
        fe = FeatureEngineer()
        # 2026-04-22 is a Wednesday — pin a known date.
        wed = datetime(2026, 4, 22, 10, 0)
        out = fe.create_patient_features(_patient(), _appointment(when=wed))
        # Should set is_wed (or equivalent) — at minimum one of mon..fri must be 1.
        flags = {k: v for k, v in out.features.items() if k.startswith("is_")}
        weekday_flags = [k for k in flags if k in {"is_mon", "is_tue", "is_wed",
                                                   "is_thu", "is_fri"}]
        # Exactly one weekday flag set to 1
        active = [k for k in weekday_flags if flags[k] == 1.0]
        assert len(active) == 1

    def test_time_slot_one_hot(self):
        fe = FeatureEngineer()
        morning = datetime(2026, 4, 22, 10, 0)   # mid_morning bucket
        out = fe.create_patient_features(
            _patient(), _appointment(when=morning),
        )
        # Slot keys are `slot_<name>` from FeatureEngineer.TIME_SLOTS;
        # exactly one bucket should be 1.0 for hour=10.
        slot_keys = [k for k in out.features if k.startswith("slot_")]
        active = [k for k in slot_keys if out.features[k] == 1.0]
        assert active == ["slot_mid_morning"]


# --------------------------------------------------------------------------- #
# Geographic features
# --------------------------------------------------------------------------- #


class TestGeographicFeatures:
    def test_distance_zero_at_site(self):
        fe = FeatureEngineer()
        # Patient at the WC site coords → distance ≈ 0
        from config import DEFAULT_SITES
        wc = next(s for s in DEFAULT_SITES if s["code"] == "WC")
        p = _patient()
        p["lat"] = wc["lat"]
        p["lon"] = wc["lon"]
        out = fe.create_patient_features(p, _appointment())
        assert out.features["distance_km"] < 0.5

    def test_long_distance_flag(self):
        fe = FeatureEngineer()
        # SA1 (Swansea) is in POSTCODE_COORDINATES and ~60 km from WC,
        # which trips both is_medium_distance + is_long_distance bands.
        p = _patient(postcode="SA1")
        out = fe.create_patient_features(p, _appointment())
        assert out.features["distance_km"] > 30
        assert out.features["is_long_distance"] == 1.0


# --------------------------------------------------------------------------- #
# Treatment + external
# --------------------------------------------------------------------------- #


class TestTreatmentAndExternal:
    def test_priority_decoded(self):
        fe = FeatureEngineer()
        out_p1 = fe.create_patient_features(_patient(), _appointment(priority="P1"))
        out_p4 = fe.create_patient_features(_patient(), _appointment(priority="P4"))
        assert out_p1.features["priority_level"] == 1
        assert out_p4.features["priority_level"] == 4

    def test_external_severity_aggregates(self):
        fe = FeatureEngineer()
        # External data is nested by source — feature_engineer pulls
        # severity from external["weather"]["severity"] etc.
        ext = {
            "weather": {"severity": 0.7},
            "traffic": {"severity": 0.3},
        }
        out = fe.create_patient_features(_patient(), _appointment(), ext)
        assert out.features["weather_severity"] == pytest.approx(0.7)
        assert out.features["traffic_severity"] == pytest.approx(0.3)
        # Combined score blends with documented weights (0.3/0.4/0.3).
        assert out.features["combined_external_severity"] == pytest.approx(
            0.7 * 0.3 + 0.3 * 0.4 + 0.0
        )


# --------------------------------------------------------------------------- #
# Batch DataFrame builder
# --------------------------------------------------------------------------- #


class TestBatchBuilder:
    def test_create_features_dataframe(self):
        fe = FeatureEngineer()
        patients = [_patient(f"P{i:03d}") for i in range(4)]
        appts = [_appointment() for _ in range(4)]
        df = fe.create_features_dataframe(patients, appts)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "prev_noshow_rate" in df.columns
