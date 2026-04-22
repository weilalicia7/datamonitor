"""
Tests for ml/individual_fairness.py (Dissertation §4.2)
======================================================

Verify the Dwork-style Lipschitz certificate:
* Constant outcomes on similar pairs → strictly Lipschitz
* Clearly unfair pair → detected with the expected ordering
* Larger L makes more pairs pass (monotonicity)
* Larger tau brings more pairs into scope (monotonicity)
* Feature normalisation works (min-max to [0, 1])
* NearestNeighbors radius query returns the right pairs
* Persistence + invariants on the output dataclass
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.individual_fairness import (
    DEFAULT_FEATURES,
    DEFAULT_L,
    DEFAULT_TAU,
    FeatureNormalizer,
    LipschitzFairnessCertificate,
    LipschitzFairnessCertifier,
    ViolatingPair,
    _coerce_float,
    _similar_pairs,
    get_certifier,
    set_certifier,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_cert_dir(tmp_path):
    return tmp_path / "individual_fairness"


@pytest.fixture
def certifier(tmp_cert_dir):
    return LipschitzFairnessCertifier(storage_dir=tmp_cert_dir)


@pytest.fixture
def clustered_patients():
    """100 patients clustered around 3 phenotypes."""
    rng = np.random.RandomState(42)
    out = []
    for i in range(100):
        cluster = i % 3
        base = (
            [65, 2, 60, 10, 0.10] if cluster == 0
            else [45, 1, 90, 30, 0.25] if cluster == 1
            else [28, 3, 45, 5, 0.05]
        )
        out.append({
            'Patient_ID': f'P{i:03d}',
            'age': base[0] + rng.normal(0, 2),
            'priority': base[1],
            'expected_duration': base[2] + rng.normal(0, 5),
            'distance_km': max(0.0, base[3] + rng.normal(0, 1)),
            'no_show_rate': max(0.0, min(1.0, base[4] + rng.normal(0, 0.02))),
        })
    return out


@pytest.fixture
def identical_outcomes(clustered_patients):
    return {p['Patient_ID']: 1.0 for p in clustered_patients}


@pytest.fixture
def biased_outcomes(clustered_patients):
    # P000..P069 scheduled, P070..P099 not — creates many violations across clusters
    return {
        p['Patient_ID']: (1.0 if i < 70 else 0.0)
        for i, p in enumerate(clustered_patients)
    }


# --------------------------------------------------------------------------- #
# 1. Identity: identical outcomes on similar pairs ⇒ strictly Lipschitz
# --------------------------------------------------------------------------- #


class TestStrictlyLipschitz:
    def test_all_scheduled_passes(self, certifier, clustered_patients, identical_outcomes):
        cert = certifier.certify(clustered_patients, identical_outcomes)
        assert cert.strictly_lipschitz is True
        assert cert.n_violations == 0
        assert cert.certified is True
        assert cert.worst_excess == 0.0

    def test_narrative_mentions_strict(self, certifier, clustered_patients, identical_outcomes):
        cert = certifier.certify(clustered_patients, identical_outcomes)
        assert "STRICTLY-LIPSCHITZ" in cert.narrative


# --------------------------------------------------------------------------- #
# 2. Clearly-unfair pair is detected
# --------------------------------------------------------------------------- #


class TestUnfairDetection:
    def test_biased_cohort_fails(self, certifier, clustered_patients, biased_outcomes):
        cert = certifier.certify(clustered_patients, biased_outcomes)
        assert cert.certified is False
        assert cert.n_violations > 0
        assert cert.worst_excess > 0.0
        assert cert.top_violations, "expected at least one ViolatingPair"

    def test_worst_violation_has_positive_excess(self, certifier, clustered_patients, biased_outcomes):
        cert = certifier.certify(clustered_patients, biased_outcomes)
        top = cert.top_violations[0]
        assert top.excess > 0.0
        assert top.outcome_gap > top.lipschitz_bound

    def test_top_violations_sorted_desc(self, certifier, clustered_patients, biased_outcomes):
        cert = certifier.certify(clustered_patients, biased_outcomes)
        excesses = [v.excess for v in cert.top_violations]
        assert excesses == sorted(excesses, reverse=True)


# --------------------------------------------------------------------------- #
# 3. Monotonicity of L and tau
# --------------------------------------------------------------------------- #


class TestMonotonicity:
    def test_larger_L_reduces_violations(self, certifier, clustered_patients, biased_outcomes):
        c1 = certifier.certify(clustered_patients, biased_outcomes, L=0.5)
        c2 = certifier.certify(clustered_patients, biased_outcomes, L=5.0)
        # With a bigger L the bound L*d is bigger, so fewer pairs exceed it
        assert c2.n_violations <= c1.n_violations

    def test_larger_tau_widens_pool(self, certifier, clustered_patients, biased_outcomes):
        c1 = certifier.certify(clustered_patients, biased_outcomes, tau=0.10)
        c2 = certifier.certify(clustered_patients, biased_outcomes, tau=0.30)
        # Larger radius must see at least as many similar pairs
        assert c2.n_similar_pairs >= c1.n_similar_pairs

    def test_infinite_L_certifies_everything(self, certifier, clustered_patients, biased_outcomes):
        cert = certifier.certify(clustered_patients, biased_outcomes, L=1e6)
        assert cert.n_violations == 0
        assert cert.certified is True


# --------------------------------------------------------------------------- #
# 4. Feature normaliser
# --------------------------------------------------------------------------- #


class TestFeatureNormalizer:
    def test_min_max_range(self, clustered_patients):
        norm = FeatureNormalizer(DEFAULT_FEATURES).fit(clustered_patients)
        X = norm.transform(clustered_patients)
        assert X.min() >= 0.0 - 1e-9
        assert X.max() <= 1.0 + 1e-9

    def test_no_nans(self, clustered_patients):
        norm = FeatureNormalizer(DEFAULT_FEATURES).fit(clustered_patients)
        X = norm.transform(clustered_patients)
        assert not np.isnan(X).any()

    def test_missing_features_filled_with_mean(self):
        patients = [
            {'Patient_ID': 'A', 'age': 65, 'priority': 2,
             'expected_duration': 60, 'distance_km': 10.0, 'no_show_rate': 0.1},
            {'Patient_ID': 'B'},  # no features at all → NaN → mean after transform
        ]
        norm = FeatureNormalizer(DEFAULT_FEATURES).fit(patients)
        X = norm.transform(patients)
        assert not np.isnan(X).any()


# --------------------------------------------------------------------------- #
# 5. Nearest-neighbour pair search
# --------------------------------------------------------------------------- #


class TestSimilarPairs:
    def test_radius_query_returns_close_pairs(self):
        X = np.array([[0.0, 0.0], [0.01, 0.01], [1.0, 1.0]])
        pairs = _similar_pairs(X, radius=0.05)
        # 0 and 1 should be close enough, 2 is far
        assert (0, 1) in {(p[0], p[1]) for p in pairs}
        assert not any(p[0] == 2 or p[1] == 2 for p in pairs)

    def test_zero_radius_yields_no_pairs(self):
        X = np.random.RandomState(0).rand(30, 4)
        pairs = _similar_pairs(X, radius=0.0)
        assert pairs == [] or all(p[2] == 0.0 for p in pairs)


# --------------------------------------------------------------------------- #
# 6. Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_two_patients_vacuous_pass(self, certifier):
        patients = [
            {'Patient_ID': 'A', 'age': 40, 'priority': 2,
             'expected_duration': 60, 'distance_km': 10, 'no_show_rate': 0.1},
            {'Patient_ID': 'B', 'age': 70, 'priority': 3,
             'expected_duration': 60, 'distance_km': 10, 'no_show_rate': 0.1},
        ]
        cert = certifier.certify(patients, {'A': 1.0, 'B': 0.0})
        # 1 pair < MIN_PAIRS = 3 → vacuous pass
        assert cert.certified is True
        assert cert.n_similar_pairs < 3

    def test_outcomes_missing_patients_dropped(self, certifier, clustered_patients):
        outcomes = {p['Patient_ID']: 1.0 for p in clustered_patients[:10]}
        cert = certifier.certify(clustered_patients, outcomes)
        assert cert.n_patients == 10


# --------------------------------------------------------------------------- #
# 7. Persistence + singleton
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_log_written(self, certifier, tmp_cert_dir, clustered_patients, biased_outcomes):
        certifier.certify(clustered_patients, biased_outcomes)
        log = tmp_cert_dir / "certificates.jsonl"
        assert log.exists()
        lines = log.read_text().strip().splitlines()
        rec = json.loads(lines[-1])
        assert rec["L"] == certifier.L
        assert rec["n_violations"] >= 0

    def test_last_cached(self, certifier, clustered_patients, biased_outcomes):
        assert certifier.last() is None
        certifier.certify(clustered_patients, biased_outcomes)
        assert certifier.last() is not None

    def test_status_increments(self, certifier, clustered_patients, biased_outcomes):
        before = certifier.status()["total_runs"]
        certifier.certify(clustered_patients, biased_outcomes)
        after = certifier.status()["total_runs"]
        assert after == before + 1

    def test_update_config_round_trip(self, certifier):
        cfg = certifier.update_config(L=2.5, tau=0.2, violation_budget=0.02,
                                      enforce_as_hard_constraint=True)
        assert cfg["L"] == 2.5
        assert cfg["tau"] == 0.2
        assert cfg["violation_budget"] == 0.02
        assert cfg["enforce_as_hard_constraint"] is True


class TestSingleton:
    def test_get_set(self, tmp_path):
        c = LipschitzFairnessCertifier(storage_dir=tmp_path / "x")
        set_certifier(c)
        assert get_certifier() is c


# --------------------------------------------------------------------------- #
# 8. Lazy-constraint emitter
# --------------------------------------------------------------------------- #


class TestLazyConstraints:
    def test_iter_violating_pairs_returns_tuples(self, certifier, clustered_patients, biased_outcomes):
        triples = certifier.iter_violating_pairs(clustered_patients, biased_outcomes)
        assert isinstance(triples, list)
        assert triples  # expect at least one
        for a, b, excess in triples:
            assert isinstance(a, str) and isinstance(b, str)
            assert excess > 0

    def test_iter_empty_when_certified(self, certifier, clustered_patients, identical_outcomes):
        triples = certifier.iter_violating_pairs(clustered_patients, identical_outcomes)
        assert triples == []


# --------------------------------------------------------------------------- #
# 9. Coercion helper
# --------------------------------------------------------------------------- #


class TestCoerceFloat:
    def test_numeric_pass_through(self):
        assert _coerce_float(3.14) == 3.14
        assert _coerce_float(7) == 7.0

    def test_priority_code(self):
        assert _coerce_float("P1") == 1.0
        assert _coerce_float("P4") == 4.0

    def test_none_becomes_nan(self):
        import math
        assert math.isnan(_coerce_float(None))

    def test_numeric_string(self):
        assert _coerce_float("2.5") == 2.5


# --------------------------------------------------------------------------- #
# 10. JSON round-trip
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_to_dict_is_json_safe(self, certifier, clustered_patients, biased_outcomes):
        cert = certifier.certify(clustered_patients, biased_outcomes)
        dumped = json.dumps(cert.to_dict())
        back = json.loads(dumped)
        assert back["certified"] is False
        assert isinstance(back["top_violations"], list)
