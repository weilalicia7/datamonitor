"""
Tests for datasets/_nhs_calibration.py — production-readiness Wave 3.6.

`_nhs_calibration.py` reads a real NHS CWT CSV from
`datasets/nhs_open_data/cancer_waiting_times/` and derives:
  * internal_cancer_weights (probability distribution, sums to ~1)
  * drug_62d_within_fraction (0-1, from the 62D 'Within/Total' columns)
  * drug_62d_total_volume (integer patient count)

NOTE ON SCOPE: the task brief mentioned an `_anonymise()` postcode +
NHS-number SHA-256 utility.  No such function exists in the module —
it only contains calibration logic — so tests cover the real public
surface (load() / summary() / CalibrationBundle) instead.

Structure:
- TestFallbackPath       — missing CWT dir -> source == 'fallback_prior'
- TestLoadRealCwt        — calibration bundle from bundled CSVs
- TestInternalWeightsSum — weights form a valid probability distribution
- TestSummaryText        — summary() prints key fields without crashing
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets._nhs_calibration as nhs_calib  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cache():
    """Calibration is cached at module level — clear before and after each test."""
    nhs_calib._cache = None
    yield
    nhs_calib._cache = None


# --------------------------------------------------------------------------- #
# 1. Fallback path — tmp dir has no CWT files
# --------------------------------------------------------------------------- #


class TestFallbackPath:
    def test_missing_cwt_dir_returns_fallback_prior(self, tmp_path, monkeypatch):
        # Point the module at a directory with no CSVs
        empty = tmp_path / "empty_cwt"
        empty.mkdir()
        monkeypatch.setattr(nhs_calib, "_CWT_DIR", empty)
        bundle = nhs_calib.load(force_reload=True)
        assert bundle.source == "fallback_prior"
        # Default prior values still present
        assert bundle.drug_62d_within_fraction == pytest.approx(0.64)
        assert bundle.noshow_beta_alpha == 1.5
        assert bundle.noshow_beta_beta == 8.0
        # No weights derived when there's no CWT input
        assert bundle.internal_cancer_weights == {}


# --------------------------------------------------------------------------- #
# 2. Happy path using the bundled CWT snapshots
# --------------------------------------------------------------------------- #


class TestLoadRealCwt:
    def test_loads_nhs_open_data_from_bundled_csvs(self):
        bundle = nhs_calib.load(force_reload=True)
        # The repo ships several CSVs in datasets/nhs_open_data/... so the
        # happy path must succeed.
        assert bundle.source == "nhs_open_data"
        assert bundle.cwt_file is not None
        assert bundle.cwt_file.endswith(".csv")
        # Volume is derived from the 62D 'Total' column -> strictly positive
        assert bundle.drug_62d_total_volume > 0
        # Within-fraction is a probability
        assert 0.0 < bundle.drug_62d_within_fraction <= 1.0

    def test_cached_load_returns_same_object(self):
        first = nhs_calib.load(force_reload=True)
        second = nhs_calib.load()  # no force_reload -> cache hit
        assert second is first


# --------------------------------------------------------------------------- #
# 3. Internal cancer weights form a proper probability distribution
# --------------------------------------------------------------------------- #


class TestInternalWeightsSum:
    def test_weights_sum_to_one_and_keys_are_internal(self):
        bundle = nhs_calib.load(force_reload=True)
        if bundle.source != "nhs_open_data":
            pytest.skip("CWT CSVs unavailable in this environment")
        weights = bundle.internal_cancer_weights
        assert weights, "Expected at least one derived cancer-type weight"
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        # Every key must be one of the internal keys declared in CWT_TO_INTERNAL
        valid = {k for vals in nhs_calib.CWT_TO_INTERNAL.values() for k in vals}
        assert set(weights).issubset(valid)


# --------------------------------------------------------------------------- #
# 4. summary() returns a readable text blob that names the key fields
# --------------------------------------------------------------------------- #


class TestSummaryText:
    def test_summary_contains_key_fields(self):
        s = nhs_calib.summary()
        assert "NHS calibration source" in s
        assert "62-day drug-regimen compliance" in s
        assert "No-show Beta" in s
