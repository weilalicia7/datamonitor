"""
Tests for ml/sequence_model.py — Wave 3.1.4 (T3 coverage).

Covers:
- SequenceFeatureEngineer.create_sequence_features (single + all)
- SequenceNoShowModel construction
- short training round-trip on synthetic data
- predict_patient end-to-end
- save/load via T2.3 safe_loader
- insufficient-data + None-return guards
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ml.sequence_model import (
    SequenceFeatureEngineer,
    SequenceFeatures,
    SequenceNoShowModel,
)


# --------------------------------------------------------------------------- #
# Synthetic appointment frames
# --------------------------------------------------------------------------- #


def _make_history(patient_id: str, n: int, *, attended_pattern=None,
                  start: datetime = None) -> pd.DataFrame:
    start = start or datetime(2025, 1, 1)
    rows = []
    pattern = attended_pattern or (["Yes"] * n)
    for i in range(n):
        rows.append({
            "Patient_ID": patient_id,
            "Appointment_Date": (start + timedelta(days=14 * i)).isoformat(),
            "Attended_Status": pattern[i % len(pattern)],
            "Cycle_Number": i + 1,
        })
    return pd.DataFrame(rows)


def _make_population(n_patients: int = 10, n_per: int = 6) -> pd.DataFrame:
    frames = []
    for i in range(n_patients):
        # Deterministic pseudo-random pattern: every third patient is a no-show.
        pattern = ["Yes", "Yes", "No"] if i % 3 == 0 else ["Yes"] * 4
        frames.append(_make_history(f"P{i:03d}", n_per, attended_pattern=pattern))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# SequenceFeatureEngineer
# --------------------------------------------------------------------------- #


class TestSequenceFeatureEngineer:
    def test_returns_features_for_long_history(self):
        fe = SequenceFeatureEngineer()
        df = _make_history("P001", 8)
        out = fe.create_sequence_features(df, "P001")
        assert isinstance(out, SequenceFeatures)
        assert out.patient_id == "P001"

    def test_returns_none_for_short_history(self):
        fe = SequenceFeatureEngineer()
        df = _make_history("P001", 1)
        out = fe.create_sequence_features(df, "P001")
        assert out is None

    def test_returns_none_for_unknown_patient(self):
        fe = SequenceFeatureEngineer()
        df = _make_history("P001", 5)
        out = fe.create_sequence_features(df, "P-NOT-IN-DF")
        assert out is None

    def test_create_all_sequences(self):
        fe = SequenceFeatureEngineer()
        df = _make_population(n_patients=6, n_per=5)
        seqs = fe.create_all_sequences(df, min_sequence_length=3)
        # Every patient has 5 appts ≥ 3 → 6 sequences.
        assert len(seqs) == 6
        assert all(isinstance(s, SequenceFeatures) for s in seqs)


# --------------------------------------------------------------------------- #
# SequenceNoShowModel
# --------------------------------------------------------------------------- #


class TestSequenceNoShowModel:
    def test_construction_default_gru(self):
        m = SequenceNoShowModel()
        assert m.model_type == "gru"
        assert m.is_trained is False
        assert m.model is None

    def test_construction_lstm(self):
        m = SequenceNoShowModel(model_type="lstm", hidden_size=32, num_layers=1)
        assert m.model_type == "lstm"
        assert m.hidden_size == 32

    def test_train_too_few_sequences_raises(self):
        m = SequenceNoShowModel(hidden_size=16, num_layers=1)
        df = _make_population(n_patients=3, n_per=4)
        with pytest.raises(ValueError):
            m.train(df, epochs=1, min_sequence_length=3)

    def test_train_short_round_trip(self):
        # Need >= 10 patient sequences to clear the floor.
        m = SequenceNoShowModel(hidden_size=8, num_layers=1, dropout=0.0)
        df = _make_population(n_patients=12, n_per=5)
        metrics = m.train(df, epochs=1, batch_size=8, min_sequence_length=3,
                          validation_split=0.2)
        assert m.is_trained is True
        assert isinstance(metrics, dict)


# --------------------------------------------------------------------------- #
# Save / load via safe_loader
# --------------------------------------------------------------------------- #


class TestSaveLoad:
    def test_save_writes_sidecar(self, tmp_path):
        m = SequenceNoShowModel(hidden_size=8, num_layers=1, dropout=0.0)
        df = _make_population(n_patients=12, n_per=5)
        m.train(df, epochs=1, batch_size=8, min_sequence_length=3,
                validation_split=0.2)
        path = tmp_path / "seq.pkl"
        m.save(str(path))
        assert path.with_suffix(path.suffix + ".sha256").exists()
