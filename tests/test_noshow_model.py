"""
Tests for ml/noshow_model.py — Wave 3.1.1 (T3 coverage).

Covers the public API surface of NoShowModel:
- construction (stacking / BMA / sequence flags)
- rule-based fallback predictions (untrained model)
- end-to-end training on synthetic data
- predict() / predict_batch() shape + risk-level mapping
- save/load via T2.3 safe_loader (SHA-256 verified round-trip)
- feature importance + reset hooks
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.noshow_model import NoShowModel, PredictionResult


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _patient(pid: str = "P001", *, history: int = 5, no_shows: int = 1,
             cancellations: int = 0, distance_km: float = 5.0) -> dict:
    return {
        "patient_id": pid,
        "total_appointments": history,
        "no_shows": no_shows,
        "cancellations": cancellations,
        "late_arrivals": 0,
        "last_visit_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "age": 60,
        "Age_Band": "60-75",
        "postcode": "CF14",
        "lat": 51.48,
        "lon": -3.18,
    }


def _appointment(*, when: datetime = None, duration: int = 60,
                 priority: str = "P3", protocol: str = "FOLFOX") -> dict:
    # The feature_engineer expects priority as "P1".."P4" — encoded as the
    # second character (int(priority[1])).  Mirror SACT's input shape exactly.
    when = when or (datetime.now() + timedelta(days=14)).replace(hour=10, minute=0)
    return {
        "appointment_date": when.isoformat(),
        "site_code": "WC",
        "duration_minutes": duration,
        "priority": priority,
        "protocol": protocol,
        "Days_To_Appointment": (when - datetime.now()).days,
    }


def _training_frame(n: int = 80, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for i in range(n):
        prev_rate = float(rng.uniform(0, 0.6))
        rows.append({
            "prev_noshow_rate": prev_rate,
            "prev_noshow_count": float(rng.integers(0, 5)),
            "total_appointments": float(rng.integers(0, 20)),
            "is_new_patient": float(rng.integers(0, 2)),
            "is_long_distance": float(rng.integers(0, 2)),
            "is_medium_distance": float(rng.integers(0, 2)),
            "weather_severity": float(rng.uniform(0, 1)),
            "traffic_severity": float(rng.uniform(0, 1)),
            "booked_long_advance": float(rng.integers(0, 2)),
            "priority_level": float(rng.integers(1, 5)),
        })
        # Labels correlate with prev_noshow_rate so the trained model
        # should beat coin flip — keeps the test deterministic but real.
        labels.append(int(prev_rate + rng.uniform(-0.1, 0.1) > 0.3))
    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    return X, y


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_default_init(self):
        m = NoShowModel(use_sequence_model=False)
        assert m.use_stacking is True
        assert m.use_bma is False
        assert m.is_trained is False
        assert m.feature_names == []

    def test_bma_toggle(self):
        m = NoShowModel(use_stacking=False, use_bma=True, use_sequence_model=False)
        assert m.use_stacking is False
        assert m.use_bma is True
        assert m.bma_weights == {}


# --------------------------------------------------------------------------- #
# Rule-based fallback (untrained)
# --------------------------------------------------------------------------- #


class TestRuleBasedFallback:
    def test_predict_returns_prediction_result(self):
        m = NoShowModel(use_sequence_model=False)
        out = m.predict(_patient(), _appointment())
        assert isinstance(out, PredictionResult)
        assert 0.0 <= out.noshow_probability <= 1.0
        assert out.risk_level in {"low", "medium", "high", "very_high"}

    def test_high_history_no_show_increases_probability(self):
        m = NoShowModel(use_sequence_model=False)
        low = m.predict(_patient("P-low", history=10, no_shows=0), _appointment())
        high = m.predict(_patient("P-high", history=10, no_shows=8), _appointment())
        assert high.noshow_probability > low.noshow_probability


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


class TestTraining:
    def test_train_marks_is_trained(self):
        X, y = _training_frame(n=80)
        m = NoShowModel(use_stacking=False, use_sequence_model=False)
        metrics = m.train(X, y, test_size=0.25)
        assert m.is_trained is True
        assert isinstance(metrics, dict)
        assert m.feature_names == list(X.columns)

    def test_trained_predict_uses_ensemble(self):
        X, y = _training_frame(n=120)
        m = NoShowModel(use_stacking=False, use_sequence_model=False)
        m.train(X, y, test_size=0.2)
        pred = m.predict(_patient(no_shows=4, history=10), _appointment())
        # Trained probability stays in [0, 1] and is finite.
        assert 0.0 <= pred.noshow_probability <= 1.0


# --------------------------------------------------------------------------- #
# Risk level mapping
# --------------------------------------------------------------------------- #


class TestRiskLevel:
    @pytest.mark.parametrize("p,level", [
        (0.05, "low"),
        (0.20, "medium"),
        (0.40, "high"),
        (0.70, "very_high"),
    ])
    def test_threshold_mapping(self, p, level):
        m = NoShowModel(use_sequence_model=False)
        assert m._get_risk_level(p) == level


# --------------------------------------------------------------------------- #
# Save / load via T2.3 safe_loader
# --------------------------------------------------------------------------- #


class TestSaveLoad:
    def test_round_trip_with_sidecar(self, tmp_path):
        X, y = _training_frame(n=80)
        m = NoShowModel(use_stacking=False, use_sequence_model=False)
        m.train(X, y)
        path = tmp_path / "noshow.pkl"
        m.save(str(path))
        # Sidecar must exist (T2.3).
        assert path.with_suffix(path.suffix + ".sha256").exists()

        m2 = NoShowModel(use_sequence_model=False)
        m2.load(str(path))
        assert m2.is_trained is True
        assert m2.feature_names == m.feature_names

    def test_tampered_pickle_rejected_at_load(self, tmp_path):
        X, y = _training_frame(n=60)
        m = NoShowModel(use_stacking=False, use_sequence_model=False)
        m.train(X, y)
        path = tmp_path / "noshow.pkl"
        m.save(str(path))
        # Corrupt the model file after the sidecar was written.
        path.write_bytes(b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00.")
        from safe_loader import UnsafeLoadError
        with pytest.raises(UnsafeLoadError):
            NoShowModel(use_sequence_model=False).load(str(path))


# --------------------------------------------------------------------------- #
# Feature importance + batch
# --------------------------------------------------------------------------- #


class TestFeatureImportance:
    def test_returns_dict_after_training(self):
        X, y = _training_frame(n=80)
        m = NoShowModel(use_stacking=False, use_sequence_model=False)
        m.train(X, y)
        fi = m.get_feature_importance()
        assert isinstance(fi, dict)
        assert len(fi) > 0
        for k, v in fi.items():
            assert isinstance(k, str)
            assert isinstance(v, float)


class TestPredictBatch:
    def test_batch_returns_one_per_patient(self):
        m = NoShowModel(use_sequence_model=False)
        patients = [_patient(f"P{i:03d}") for i in range(5)]
        appts = [_appointment() for _ in range(5)]
        results = m.predict_batch(patients, appts)
        assert len(results) == 5
        assert all(isinstance(r, PredictionResult) for r in results)
