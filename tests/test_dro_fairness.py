"""
Tests for ml/dro_fairness.py (Dissertation §4.1)
================================================

Verifies the Wasserstein-1 DRO upper bound on demographic parity:
* worst_case_gap >= empirical_gap
* epsilon=0 collapses bound to empirical
* smaller group inflates the bound
* identical group rates pass
* biased rates fail
* certificate persistence
* invariants on the output dataclass
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.dro_fairness import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    DROFairnessCertificate,
    DROFairnessCertifier,
    MIN_GROUP_SIZE,
    PairCertificate,
    _group_by,
    _z_score,
    get_certifier,
    set_certifier,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_cert_dir(tmp_path):
    return tmp_path / "dro_fairness"


@pytest.fixture
def certifier(tmp_cert_dir):
    return DROFairnessCertifier(storage_dir=tmp_cert_dir)


@pytest.fixture
def balanced_cohort():
    """50 young / 50 old, identical 80% schedule rate => gap=0."""
    patients = [
        {"Patient_ID": f"P{i:03d}", "Age_Band": ("Young" if i < 50 else "Old")}
        for i in range(100)
    ]
    scheduled = {f"P{i:03d}" for i in range(40)} | {f"P{i:03d}" for i in range(50, 90)}
    return patients, scheduled


@pytest.fixture
def biased_cohort():
    """Young: 45/50 = 0.9; Old: 30/50 = 0.6 => gap=0.30."""
    patients = [
        {"Patient_ID": f"P{i:03d}", "Age_Band": ("Young" if i < 50 else "Old")}
        for i in range(100)
    ]
    scheduled = {f"P{i:03d}" for i in range(45)} | {f"P{i:03d}" for i in range(50, 80)}
    return patients, scheduled


@pytest.fixture
def minority_cohort():
    """Very small minority: 5 young / 95 old."""
    patients = [
        {"Patient_ID": f"P{i:03d}", "Age_Band": ("Young" if i < 5 else "Old")}
        for i in range(100)
    ]
    # 2/5 young scheduled, 80/95 old scheduled => gap = |0.4 - 0.842| = 0.442
    scheduled = {f"P{i:03d}" for i in range(2)} | {f"P{i:03d}" for i in range(5, 85)}
    return patients, scheduled


# --------------------------------------------------------------------------- #
# 1. Upper-bound invariant: DRO worst-case >= empirical
# --------------------------------------------------------------------------- #


class TestBoundInvariants:
    def test_worst_case_gte_empirical(self, certifier, biased_cohort):
        p, s = biased_cohort
        cert = certifier.certify(p, s, group_column="Age_Band",
                                 epsilon=DEFAULT_EPSILON, delta=DEFAULT_DELTA)
        for pc in cert.pair_certificates:
            assert pc.worst_case_gap + 1e-9 >= pc.empirical_gap

    def test_epsilon_zero_collapses_to_empirical(self, certifier, biased_cohort):
        p, s = biased_cohort
        cert = certifier.certify(p, s, group_column="Age_Band",
                                 epsilon=0.0, delta=0.15)
        for pc in cert.pair_certificates:
            assert pc.worst_case_gap == pytest.approx(pc.empirical_gap, abs=1e-9)

    def test_larger_epsilon_inflates_bound(self, certifier, biased_cohort):
        p, s = biased_cohort
        c1 = certifier.certify(p, s, group_column="Age_Band",
                               epsilon=0.01, delta=0.15)
        c2 = certifier.certify(p, s, group_column="Age_Band",
                               epsilon=0.05, delta=0.15)
        assert c2.worst_pair_gap > c1.worst_pair_gap

    def test_minority_group_inflates_bound(self, certifier, biased_cohort, minority_cohort):
        # Balanced 50/50 biased vs. 5/95 biased with same empirical gap
        # The minority case should have a strictly larger worst_case bound
        # because the 1/π_a + 1/π_b term is bigger for a tiny group.
        p1, s1 = biased_cohort
        p2, s2 = minority_cohort
        c1 = certifier.certify(p1, s1, group_column="Age_Band", epsilon=0.03)
        c2 = certifier.certify(p2, s2, group_column="Age_Band", epsilon=0.03)
        # Inflation above empirical gap is strictly larger for minority
        inf1 = c1.pair_certificates[0].worst_case_gap - c1.pair_certificates[0].empirical_gap
        inf2 = c2.pair_certificates[0].worst_case_gap - c2.pair_certificates[0].empirical_gap
        assert inf2 > inf1


# --------------------------------------------------------------------------- #
# 2. Pass / fail semantics
# --------------------------------------------------------------------------- #


class TestPassFail:
    def test_balanced_cohort_is_certified(self, certifier, balanced_cohort):
        p, s = balanced_cohort
        cert = certifier.certify(p, s, group_column="Age_Band",
                                 epsilon=DEFAULT_EPSILON, delta=DEFAULT_DELTA)
        assert cert.overall_certified is True
        assert cert.worst_pair_gap < DEFAULT_DELTA

    def test_biased_cohort_fails(self, certifier, biased_cohort):
        p, s = biased_cohort
        cert = certifier.certify(p, s, group_column="Age_Band",
                                 epsilon=DEFAULT_EPSILON, delta=DEFAULT_DELTA)
        assert cert.overall_certified is False
        assert any("FAIL" in cert.narrative for _ in [1])  # narrative present
        assert cert.worst_pair is not None

    def test_tight_budget_flips_to_fail(self, certifier, balanced_cohort):
        p, s = balanced_cohort
        cert = certifier.certify(p, s, group_column="Age_Band",
                                 epsilon=0.05, delta=0.001)
        assert cert.overall_certified is False


# --------------------------------------------------------------------------- #
# 3. Conservative (SE-inflated) bound
# --------------------------------------------------------------------------- #


class TestConservativeBound:
    def test_conservative_upper_gte_worst_case(self, certifier, biased_cohort):
        p, s = biased_cohort
        cert = certifier.certify(p, s, group_column="Age_Band",
                                 epsilon=DEFAULT_EPSILON, delta=DEFAULT_DELTA)
        for pc in cert.pair_certificates:
            assert pc.se_adjusted_upper + 1e-9 >= pc.worst_case_gap

    def test_conservative_becomes_relevant_at_small_n(self, certifier):
        # 10 vs 10 patients, 5 vs 3 scheduled => gap = 0.2
        patients = [
            {"Patient_ID": f"P{i}", "Age_Band": ("A" if i < 10 else "B")}
            for i in range(20)
        ]
        scheduled = {f"P{i}" for i in range(5)} | {f"P{i}" for i in range(10, 13)}
        cert = certifier.certify(patients, scheduled, group_column="Age_Band",
                                 epsilon=0.02, delta=0.50)
        # With only 10 per group, SE-adjusted should be meaningfully larger
        pc = cert.pair_certificates[0]
        assert pc.se_adjusted_upper > pc.worst_case_gap + 0.05


# --------------------------------------------------------------------------- #
# 4. Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_empty_patient_list(self, certifier):
        cert = certifier.certify([], set(), group_column="Age_Band")
        assert cert.pair_certificates == []
        assert cert.overall_certified is True  # vacuously true

    def test_single_group_certifies_vacuously(self, certifier):
        patients = [{"Patient_ID": f"P{i}", "Age_Band": "Young"} for i in range(20)]
        scheduled = {f"P{i}" for i in range(15)}
        cert = certifier.certify(patients, scheduled, group_column="Age_Band")
        assert cert.pair_certificates == []
        assert cert.overall_certified is True
        assert cert.n_groups == 1

    def test_group_below_min_size_filtered(self, certifier):
        patients = (
            [{"Patient_ID": f"Y{i}", "Age_Band": "Young"} for i in range(2)]  # < MIN_GROUP_SIZE
            + [{"Patient_ID": f"O{i}", "Age_Band": "Old"} for i in range(20)]
        )
        scheduled = {"Y0", "Y1"} | {f"O{i}" for i in range(10)}
        cert = certifier.certify(patients, scheduled, group_column="Age_Band")
        # Young excluded => only Old left => no pair => vacuous certification
        assert cert.pair_certificates == []
        assert cert.overall_certified is True

    def test_missing_group_column_becomes_unknown(self, certifier):
        patients = [{"Patient_ID": f"P{i}"} for i in range(30)]  # no Age_Band
        scheduled = {f"P{i}" for i in range(20)}
        cert = certifier.certify(patients, scheduled, group_column="Age_Band")
        # All funneled to "unknown" => single group => vacuous certification
        assert cert.n_groups == 1


# --------------------------------------------------------------------------- #
# 5. Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_certificate_log_written(self, certifier, tmp_cert_dir, biased_cohort):
        p, s = biased_cohort
        certifier.certify(p, s, group_column="Age_Band")
        log = tmp_cert_dir / "certificates.jsonl"
        assert log.exists()
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["group_column"] == "Age_Band"
        assert rec["overall_certified"] is False
        assert rec["pairs"], "pairs list should not be empty"

    def test_last_cached_after_run(self, certifier, biased_cohort):
        assert certifier.last() is None
        p, s = biased_cohort
        certifier.certify(p, s, group_column="Age_Band")
        last = certifier.last()
        assert last is not None
        assert last.overall_certified is False

    def test_status_counters_advance(self, certifier, biased_cohort):
        before = certifier.status()["total_runs"]
        p, s = biased_cohort
        certifier.certify(p, s, group_column="Age_Band")
        after = certifier.status()["total_runs"]
        assert after == before + 1

    def test_update_config_round_trip(self, certifier):
        cfg = certifier.update_config(epsilon=0.07, delta=0.12,
                                      confidence=0.99,
                                      enforce_as_hard_constraint=True)
        assert cfg["epsilon"] == 0.07
        assert cfg["delta"] == 0.12
        assert cfg["confidence"] == 0.99
        assert cfg["enforce_as_hard_constraint"] is True


# --------------------------------------------------------------------------- #
# 6. Narrative shape
# --------------------------------------------------------------------------- #


class TestNarrative:
    def test_pass_narrative_contains_pass_keyword(self, certifier, balanced_cohort):
        p, s = balanced_cohort
        cert = certifier.certify(p, s, group_column="Age_Band",
                                 epsilon=0.02, delta=0.50)
        assert "PASS" in cert.narrative

    def test_fail_narrative_contains_worst_pair(self, certifier, biased_cohort):
        p, s = biased_cohort
        cert = certifier.certify(p, s, group_column="Age_Band",
                                 epsilon=0.02, delta=0.15)
        assert "FAIL" in cert.narrative
        assert cert.worst_pair is not None
        assert cert.worst_pair[0] in cert.narrative or cert.worst_pair[1] in cert.narrative


# --------------------------------------------------------------------------- #
# 7. Helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_group_by_counts(self):
        patients = [{"Patient_ID": "A", "G": "x"},
                    {"Patient_ID": "B", "G": "x"},
                    {"Patient_ID": "C", "G": "y"}]
        scheduled = {"A", "C"}
        g = _group_by(patients, scheduled, "G")
        assert g["x"]["total"] == 2 and g["x"]["scheduled"] == 1
        assert g["y"]["total"] == 1 and g["y"]["scheduled"] == 1

    def test_z_score_lookup(self):
        assert _z_score(0.95) == pytest.approx(1.96, abs=0.01)
        assert _z_score(0.90) == pytest.approx(1.6449, abs=0.01)
        assert _z_score(0.99) == pytest.approx(2.5758, abs=0.01)

    def test_singleton_get_set(self, tmp_path):
        c = DROFairnessCertifier(storage_dir=tmp_path / "c")
        set_certifier(c)
        assert get_certifier() is c


# --------------------------------------------------------------------------- #
# 8. JSON round-trip of certificate
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_to_dict_is_json_safe(self, certifier, biased_cohort):
        p, s = biased_cohort
        cert = certifier.certify(p, s, group_column="Age_Band")
        d = cert.to_dict()
        dumped = json.dumps(d)
        back = json.loads(dumped)
        assert back["overall_certified"] is False
        assert isinstance(back["worst_pair"], list)
        assert "pair_certificates" in back
