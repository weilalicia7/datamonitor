"""
Tests for ml.auto_recalibration — ModelRecalibrator, RecalibrationResult.

Covers:
    * Default construction: versions dir created, empty log.
    * Update-level decision logic for each of the documented sources.
    * Level 0 real-time refresh with a stub event_impact_model.
    * Status reporting and history persistence.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.auto_recalibration import ModelRecalibrator, RecalibrationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class StubEventModel:
    """Minimal event-impact model stub — records update_coefficients calls."""

    def __init__(self):
        self.updated_with = None

    def update_coefficients(self, data):
        self.updated_with = data


@pytest.fixture()
def tmp_versions_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_recalibrator_default_construction(tmp_versions_dir):
    rec = ModelRecalibrator(models={}, versions_dir=tmp_versions_dir)

    assert rec.recalibration_log == []
    assert rec.current_version.startswith('v')
    assert Path(tmp_versions_dir).exists()
    assert rec.baselines == {}


def test_determine_update_level_routing(tmp_versions_dir):
    rec = ModelRecalibrator(versions_dir=tmp_versions_dir)

    # Real-time sources always Level 0
    assert rec.determine_update_level('weather', drift_score=0.0) == 0
    assert rec.determine_update_level('traffic', drift_score=0.5) == 0

    # SACT v4 routing
    assert rec.determine_update_level(
        'sact_v4_patient_data', drift_score=0.0,
        sact_v4_quality='complete') == 3
    assert rec.determine_update_level(
        'sact_v4_patient_data', drift_score=0.0,
        sact_v4_quality='preliminary') == 2
    assert rec.determine_update_level(
        'sact_v4_patient_data', drift_score=0.0,
        sact_v4_quality=None) == 1  # Unknown quality → baseline

    # Drift-based routing for other sources
    assert rec.determine_update_level('ons_open_data', drift_score=0.30) == 3
    assert rec.determine_update_level('ons_open_data', drift_score=0.20) == 2
    assert rec.determine_update_level('ons_open_data', drift_score=0.10) == 1
    assert rec.determine_update_level('ons_open_data', drift_score=0.01) == 0


def test_level0_parameter_refresh_invokes_stub(tmp_versions_dir):
    stub = StubEventModel()
    rec = ModelRecalibrator(
        models={'event_impact_model': stub},
        versions_dir=tmp_versions_dir,
    )
    # Tiny synthetic data frame — the stub just stores it.
    df = pd.DataFrame({'weather_severity': [0.1, 0.5, 0.7]})

    result = rec.execute_recalibration(level=0, source='weather', data=df)

    assert isinstance(result, RecalibrationResult)
    assert result.success
    assert result.level == 0
    assert 'event_impact_model' in result.models_updated
    assert stub.updated_with is df
    # Log is appended to.
    assert len(rec.recalibration_log) == 1


def test_get_status_reports_history_and_levels(tmp_versions_dir):
    rec = ModelRecalibrator(models={}, versions_dir=tmp_versions_dir)
    rec.set_baselines({'noshow_auc': 0.82})
    # Run one Level 0 with no models — still logs a success.
    rec.execute_recalibration(level=0, source='weather', data=None)

    status = rec.get_status()

    assert status['baselines'] == {'noshow_auc': 0.82}
    assert status['total_recalibrations'] == 1
    assert status['n_models'] == 0
    # All four levels described
    assert set(status['recalibration_levels']) == {0, 1, 2, 3}
    # Version directory is counted
    assert status['versions_on_disk'] == 0  # nothing snapshotted yet
