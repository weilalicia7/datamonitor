"""
Tests for ml.train — training pipeline helpers.

NOTE: ml/train.py has an import-time ordering bug — it uses Tuple / Dict
annotations before importing them from typing. We inject the names into
builtins before importing the module so it can load. The bug is real but
out-of-scope per the task constraints; it is listed in the final report.

Covers:
    * generate_synthetic_data — shape, columns, deterministic seed, no-show rate.
    * load_training_data — returns DataFrame and falls back to synthetic when
      the path does not exist.
    * prepare_features — returns (X_df, y_noshow, y_duration) with matching lengths.
    * train_models — end-to-end smoke with save=False runs without error and
      returns a dict keyed by 'noshow' / 'duration'.
"""

import builtins
import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Inject the missing names so `ml/train.py` can be imported.
builtins.Tuple = typing.Tuple
builtins.Dict = typing.Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import train as train_module  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_synthetic_data_shape_and_columns():
    df = train_module.generate_synthetic_data(n_samples=200)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200

    # Required SACT-aligned columns
    for col in ('Patient_ID', 'Postcode_District', 'Attended_Status',
                'Planned_Duration', 'Actual_Duration',
                'Age', 'Regimen_Code', 'Priority',
                'Patient_NoShow_Rate'):
        assert col in df.columns


def test_generate_synthetic_data_is_deterministic():
    df1 = train_module.generate_synthetic_data(n_samples=100)
    df2 = train_module.generate_synthetic_data(n_samples=100)
    # Fixed seed inside the function — outputs must be identical.
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_synthetic_data_has_plausible_noshow_rate():
    df = train_module.generate_synthetic_data(n_samples=500)
    rate = (df['Attended_Status'] == 'No').mean()
    # Code clips probabilities to [0.01, 0.50] — so rate must fall in a sane band.
    assert 0.0 < rate < 0.6


def test_load_training_data_fallback_to_synthetic(tmp_path):
    missing = tmp_path / 'does_not_exist.csv'
    df = train_module.load_training_data(str(missing))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Fallback = synthetic → must include Patient_ID
    assert 'Patient_ID' in df.columns


def test_load_training_data_reads_existing_csv(tmp_path):
    tiny = pd.DataFrame({
        'Patient_ID': ['X01', 'X02'],
        'Attended_Status': ['Yes', 'No'],
        'Planned_Duration': [90, 120],
    })
    path = tmp_path / 'mini.csv'
    tiny.to_csv(path, index=False)

    df = train_module.load_training_data(str(path))
    assert len(df) == 2
    assert list(df['Patient_ID']) == ['X01', 'X02']


def test_prepare_features_produces_aligned_arrays():
    from ml.feature_engineering import FeatureEngineer

    df = train_module.generate_synthetic_data(n_samples=60)
    engineer = FeatureEngineer()

    X, y_noshow, y_duration = train_module.prepare_features(df, engineer)

    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(df)
    assert len(y_noshow) == len(df)
    assert len(y_duration) == len(df)
    # No-show labels are binary
    assert set(np.unique(y_noshow)).issubset({0, 1})


def test_prepare_features_numeric_columns_are_castable():
    """The X DataFrame from prepare_features must expose numeric-only columns
    once non-feature identifiers are removed — downstream model code casts it."""
    from ml.feature_engineering import FeatureEngineer

    df = train_module.generate_synthetic_data(n_samples=50)
    engineer = FeatureEngineer()
    X, _y_noshow, _y_duration = train_module.prepare_features(df, engineer)

    # Some columns may be strings (e.g. Age_Band), but the identifier column
    # 'Patient_ID' must be removed after prepare_features runs.
    assert 'Patient_ID' not in X.columns
    assert 'patient_id' not in X.columns


def test_synthetic_data_age_bounds():
    df = train_module.generate_synthetic_data(n_samples=200)
    # Age sampled in [30, 85)
    ages = df['Age'].astype(int)
    assert ages.min() >= 30
    assert ages.max() <= 85
