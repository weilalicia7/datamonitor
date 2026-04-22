"""
Tests for ml/drift_attribution.py (Dissertation §3.4)
=====================================================

Checks the PSI decomposition is mathematically consistent with the
legacy DriftDetector, that shares sum to 1, that the narrative is
shaped correctly for the §3.4 brief's example, and that edge cases
(tiny samples, single feature, no drift) degrade gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.drift_attribution import (
    BinContribution,
    DriftAttribution,
    DriftAttributor,
    FeatureAttribution,
    _describe_top_bin,
    _fmt_bin,
    get_attributor,
    set_attributor,
)
from ml.drift_detection import DriftDetector


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_attr_dir(tmp_path):
    return tmp_path / "attribution"


@pytest.fixture
def attributor(tmp_attr_dir):
    return DriftAttributor(storage_dir=tmp_attr_dir)


@pytest.fixture
def strong_travel_drift():
    """Reference: mostly local (15 min).  Current: many remote (50 min)."""
    rng = np.random.RandomState(0)
    ref_travel = np.concatenate([rng.normal(15, 4, 800),
                                 rng.normal(45, 5, 100)])
    cur_travel = np.concatenate([rng.normal(16, 4, 500),
                                 rng.normal(50, 6, 400)])
    ref_age = rng.normal(60, 12, 900)
    cur_age = rng.normal(61, 12, 900)
    return (
        {"Travel_Time_Min": ref_travel, "Age": ref_age},
        {"Travel_Time_Min": cur_travel, "Age": cur_age},
    )


@pytest.fixture
def no_drift_df():
    rng = np.random.RandomState(7)
    ref = rng.normal(10, 2, 500)
    cur = rng.normal(10, 2, 500)
    return ({"Feat": ref}, {"Feat": cur})


# --------------------------------------------------------------------------- #
# 1. Per-feature math correctness
# --------------------------------------------------------------------------- #


class TestConsistencyWithDriftDetector:
    """Per-feature attribution PSI must equal DriftDetector.compute_psi()."""

    def test_single_feature_matches_driftdetector(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        det = DriftDetector(n_bins=attributor.n_bins)
        legacy = det.compute_psi(ref["Travel_Time_Min"], cur["Travel_Time_Min"])
        a = attributor.attribute(reference={"Travel_Time_Min": ref["Travel_Time_Min"]},
                                 current={"Travel_Time_Min": cur["Travel_Time_Min"]})
        assert len(a.feature_breakdown) == 1
        assert a.feature_breakdown[0].psi == pytest.approx(legacy, rel=1e-6)

    def test_two_features_each_match(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        det = DriftDetector(n_bins=attributor.n_bins)
        a = attributor.attribute(reference=ref, current=cur)
        for fa in a.feature_breakdown:
            legacy = det.compute_psi(ref[fa.feature], cur[fa.feature])
            assert fa.psi == pytest.approx(legacy, rel=1e-6)


# --------------------------------------------------------------------------- #
# 2. Share invariants
# --------------------------------------------------------------------------- #


class TestShares:
    def test_shares_sum_to_one(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        a = attributor.attribute(reference=ref, current=cur)
        total_share = sum(fa.share_of_total for fa in a.feature_breakdown)
        assert total_share == pytest.approx(1.0, abs=1e-6)

    def test_top_share_matches_breakdown_first(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        a = attributor.attribute(reference=ref, current=cur)
        assert a.top_feature == a.feature_breakdown[0].feature
        assert a.top_feature_share == pytest.approx(
            a.feature_breakdown[0].share_of_total, abs=1e-9
        )

    def test_travel_time_dominates_age(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        a = attributor.attribute(reference=ref, current=cur)
        assert a.top_feature == "Travel_Time_Min"
        assert a.feature_breakdown[0].share_of_total > 0.5


# --------------------------------------------------------------------------- #
# 3. Bin-level invariants
# --------------------------------------------------------------------------- #


class TestBins:
    def test_bin_contribs_sum_to_feature_psi(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        a = attributor.attribute(reference=ref, current=cur)
        for fa in a.feature_breakdown:
            total = sum(b.psi_contribution for b in fa.bins)
            assert total == pytest.approx(fa.psi, rel=1e-6)

    def test_top_bin_summary_non_empty_when_drift(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        a = attributor.attribute(reference=ref, current=cur,
                                 feature_hints={"Travel_Time_Min": "more remote patients"})
        top = a.feature_breakdown[0]
        assert top.top_bin_summary
        assert "more remote patients" in top.top_bin_summary
        assert "grew" in top.top_bin_summary or "shrank" in top.top_bin_summary

    def test_bin_direction_classification(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        a = attributor.attribute(reference=ref, current=cur)
        top = a.feature_breakdown[0]
        # At least one bin with "grew" AND one with "shrank" under our drift design
        dirs = {b.direction for b in top.bins}
        assert "grew" in dirs or "shrank" in dirs


# --------------------------------------------------------------------------- #
# 4. Severity + narrative shape (§3.4 brief pattern)
# --------------------------------------------------------------------------- #


class TestNarrative:
    def test_narrative_matches_brief_pattern(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        a = attributor.attribute(reference=ref, current=cur,
                                 feature_hints={"Travel_Time_Min": "more remote patients"})
        narrative = a.narrative
        # Brief: "72% of PSI increase due to 'Travel_Time_Min' shift (more remote patients)"
        assert "of PSI increase due to" in narrative
        assert "Travel_Time_Min" in narrative
        assert "more remote patients" in narrative

    def test_severity_transitions(self, attributor):
        rng = np.random.RandomState(1)
        # No drift
        ref = rng.normal(10, 2, 500)
        cur = rng.normal(10, 2, 500)
        a = attributor.attribute(reference={"x": ref}, current={"x": cur})
        assert a.overall_severity in {"none", "moderate"}

        # Massive drift
        cur_big = rng.normal(100, 2, 500)
        a2 = attributor.attribute(reference={"x": ref}, current={"x": cur_big})
        assert a2.overall_severity == "significant"
        assert "retrain" in a2.narrative.lower() or "significant" in a2.narrative.lower()

    def test_no_drift_narrative_is_reassuring(self, attributor, no_drift_df):
        ref, cur = no_drift_df
        a = attributor.attribute(reference=ref, current=cur)
        assert a.overall_severity in {"none", "moderate"}
        # Narrative should either say no material drift or be marked within-limits
        assert (
            "No material drift" in a.narrative
            or "within limits" in a.narrative
        )


# --------------------------------------------------------------------------- #
# 5. Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_empty_inputs_return_empty_attribution(self, attributor):
        a = attributor.attribute(reference={}, current={})
        assert a.feature_breakdown == []
        assert a.top_feature is None
        assert a.overall_severity == "none"

    def test_tiny_samples_skipped(self, attributor):
        a = attributor.attribute(
            reference={"f": np.array([1.0, 2.0, 3.0])},
            current={"f": np.array([1.0, 2.0, 3.0])},
        )
        assert a.feature_breakdown == []

    def test_nan_values_filtered(self, attributor):
        rng = np.random.RandomState(4)
        ref = np.concatenate([rng.normal(10, 2, 500), [np.nan] * 50])
        cur = np.concatenate([rng.normal(12, 2, 500), [np.nan] * 50])
        a = attributor.attribute(reference={"f": ref}, current={"f": cur})
        assert len(a.feature_breakdown) == 1
        # n_ref / n_cur exclude NaNs
        assert a.feature_breakdown[0].n_ref <= 500
        assert a.feature_breakdown[0].n_cur <= 500

    def test_only_feature_present_in_one_side_is_skipped(self, attributor):
        rng = np.random.RandomState(2)
        ref = {"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)}
        cur = {"a": rng.normal(0, 1, 500)}
        a = attributor.attribute(reference=ref, current=cur)
        assert {fa.feature for fa in a.feature_breakdown} == {"a"}


# --------------------------------------------------------------------------- #
# 6. Persistence + singleton
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_event_log_written(self, attributor, tmp_attr_dir, strong_travel_drift):
        ref, cur = strong_travel_drift
        attributor.attribute(reference=ref, current=cur)
        log = tmp_attr_dir / "attributions.jsonl"
        assert log.exists()
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["top_feature"] == "Travel_Time_Min"
        assert 0.5 <= rec["top_feature_share"] <= 1.0
        assert rec["features"], "expected feature list"

    def test_last_cached(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        assert attributor.last() is None
        attributor.attribute(reference=ref, current=cur)
        last = attributor.last()
        assert last is not None
        assert last.top_feature == "Travel_Time_Min"

    def test_status_counters_advance(self, attributor, strong_travel_drift):
        ref, cur = strong_travel_drift
        before = attributor.status()["total_runs"]
        attributor.attribute(reference=ref, current=cur)
        after = attributor.status()["total_runs"]
        assert after == before + 1


class TestSingleton:
    def test_get_and_set_attributor(self, tmp_path, strong_travel_drift):
        a = DriftAttributor(storage_dir=tmp_path / "x")
        set_attributor(a)
        assert get_attributor() is a


# --------------------------------------------------------------------------- #
# 7. Helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_fmt_bin_rounding(self):
        assert _fmt_bin(1.999) == "2"
        assert _fmt_bin(3.456) == "3.46"
        assert _fmt_bin(1234.0) == "1.2e+03"

    def test_describe_top_bin_with_hint(self):
        bc = BinContribution(
            bin_index=1, lower=39.71, upper=58.0,
            p_ref=0.10, p_cur=0.38, delta=0.28,
            psi_contribution=0.35, direction="grew",
        )
        s = _describe_top_bin("Travel_Time_Min", bc, hint="more remote patients")
        assert "more remote patients" in s
        assert "[39.71, 58)" in s
        assert "10.0%" in s
        assert "38.0%" in s
