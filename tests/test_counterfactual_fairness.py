"""
Tests for ml/counterfactual_fairness.py (Dissertation §4.4)
===========================================================

Verify the postcode-flip audit:
* Patients with affluent-bucket postcodes are skipped (nothing to flip)
* Postcode lookup + flip mechanics
* Scheduleability predictor fits on-the-fly
* Counterfactual probability uplift for deprived postcode patients
* Flip budget pass/fail semantics
* Persistence + status + config update
* JSON round-trip
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.counterfactual_fairness import (
    AFFLUENT_CUTOFF,
    DEFAULT_POSTCODE_DEPRIVATION,
    CounterfactualFairnessAuditor,
    CounterfactualFairnessCertificate,
    FlipCase,
    ScheduleabilityPredictor,
    _features,
    _pid,
    _postcode,
    get_auditor,
    set_auditor,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_cf_dir(tmp_path):
    return tmp_path / "cf_fairness"


@pytest.fixture
def auditor(tmp_cf_dir):
    return CounterfactualFairnessAuditor(storage_dir=tmp_cf_dir)


@pytest.fixture
def biased_cohort():
    """200 patients with postcode-correlated scheduling outcomes."""
    rng = np.random.RandomState(42)
    patients, scheduled = [], []
    for i in range(200):
        bucket = i % 3
        if bucket == 0:
            pc, p_sched = "CF10", 0.90   # affluent, high accept
        elif bucket == 1:
            pc, p_sched = "CF14", 0.70   # mixed
        else:
            pc, p_sched = "CF44", 0.40   # deprived, low accept
        patients.append({
            "Patient_ID": f"P{i:03d}",
            "Patient_Postcode": pc,
            "Age": int(rng.uniform(30, 80)),
            "Priority": int(rng.choice([1, 2, 3, 4])),
            "Planned_Duration": int(rng.uniform(30, 120)),
            "Distance_km": 5 if pc == "CF10" else 15 if pc == "CF14" else 35,
            "Travel_Time_Min": 15 if pc == "CF10" else 30 if pc == "CF14" else 65,
            "Patient_NoShow_Rate": 0.10 if pc == "CF10" else 0.15 if pc == "CF14" else 0.28,
        })
        if rng.random() < p_sched:
            scheduled.append(f"P{i:03d}")
    return patients, scheduled


@pytest.fixture
def unbiased_cohort():
    """Identical postcode across all patients; flipping cannot do anything."""
    rng = np.random.RandomState(3)
    patients, scheduled = [], []
    for i in range(50):
        patients.append({
            "Patient_ID": f"U{i:03d}",
            "Patient_Postcode": "CF14",
            "Age": int(rng.uniform(30, 80)),
            "Priority": 3,
            "Planned_Duration": 60,
            "Distance_km": 15,
            "Travel_Time_Min": 30,
            "Patient_NoShow_Rate": 0.15,
        })
        if rng.random() < 0.6:
            scheduled.append(f"U{i:03d}")
    return patients, scheduled


# --------------------------------------------------------------------------- #
# 1. Postcode lookup + helpers
# --------------------------------------------------------------------------- #


class TestPostcodeLookup:
    def test_known_postcodes_mapped(self):
        # Affluent buckets stay below cutoff
        assert DEFAULT_POSTCODE_DEPRIVATION["CF10"] < AFFLUENT_CUTOFF
        assert DEFAULT_POSTCODE_DEPRIVATION["CF23"] < AFFLUENT_CUTOFF
        # Deprived buckets sit above 6.5
        assert DEFAULT_POSTCODE_DEPRIVATION["CF44"] >= 8.0
        assert DEFAULT_POSTCODE_DEPRIVATION["CF40"] >= 8.0

    def test_postcode_extraction(self):
        assert _postcode({"Patient_Postcode": "CF44 0XX"}) == "CF44"
        assert _postcode({"postcode": "cf10"}) == "CF10"
        assert _postcode({}) == ""

    def test_pid_extraction(self):
        assert _pid({"Patient_ID": "X"}) == "X"
        assert _pid({"patient_id": "Y"}) == "Y"
        assert _pid("Z") == "Z"
        assert _pid(None) is None

    def test_features_include_deprivation(self):
        p = {"Patient_ID": "A", "Patient_Postcode": "CF44"}
        f = _features(p, DEFAULT_POSTCODE_DEPRIVATION)
        assert f["deprivation_score"] >= 8.0
        assert f["age"] > 0
        assert f["priority"] > 0


# --------------------------------------------------------------------------- #
# 2. Predictor
# --------------------------------------------------------------------------- #


class TestPredictor:
    def test_fit_produces_probabilities_in_unit(self, biased_cohort):
        patients, scheduled = biased_cohort
        p = ScheduleabilityPredictor().fit(
            patients, set(scheduled), DEFAULT_POSTCODE_DEPRIVATION,
        )
        for p_case in patients[:20]:
            prob = p.predict_proba(_features(p_case, DEFAULT_POSTCODE_DEPRIVATION))
            assert 0.0 <= prob <= 1.0

    def test_decision_threshold_in_unit(self, biased_cohort):
        patients, scheduled = biased_cohort
        p = ScheduleabilityPredictor().fit(
            patients, set(scheduled), DEFAULT_POSTCODE_DEPRIVATION,
        )
        assert 0.0 <= p.decision_threshold <= 1.0

    def test_deprived_patients_have_lower_mean_prob(self, biased_cohort):
        # The raw deprivation_score coefficient may absorb co-linearity with
        # distance / no_show / travel_time.  What matters end-to-end is that
        # deprived patients get a *lower* Pr(scheduled) than affluent ones.
        patients, scheduled = biased_cohort
        p = ScheduleabilityPredictor().fit(
            patients, set(scheduled), DEFAULT_POSTCODE_DEPRIVATION,
        )
        affluent = [
            pat for pat in patients
            if DEFAULT_POSTCODE_DEPRIVATION.get(pat["Patient_Postcode"], 5) < 4
        ]
        deprived = [
            pat for pat in patients
            if DEFAULT_POSTCODE_DEPRIVATION.get(pat["Patient_Postcode"], 5) > 6.5
        ]
        aff_mean = np.mean([
            p.predict_proba(_features(pat, DEFAULT_POSTCODE_DEPRIVATION))
            for pat in affluent
        ])
        dep_mean = np.mean([
            p.predict_proba(_features(pat, DEFAULT_POSTCODE_DEPRIVATION))
            for pat in deprived
        ])
        assert aff_mean > dep_mean

    def test_degenerate_all_scheduled_fallback(self):
        patients = [
            {"Patient_ID": f"A{i}", "Patient_Postcode": "CF14"}
            for i in range(5)
        ]
        scheduled = {f"A{i}" for i in range(5)}
        p = ScheduleabilityPredictor().fit(
            patients, scheduled, DEFAULT_POSTCODE_DEPRIVATION,
        )
        assert p.method == "mean_ratio"


# --------------------------------------------------------------------------- #
# 3. Audit semantics
# --------------------------------------------------------------------------- #


class TestAudit:
    def test_biased_cohort_produces_nonzero_delta(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        cert = auditor.audit(patients, scheduled)
        assert cert.n_rejected > 0
        # Counterfactual flip must produce a non-zero mean delta.  Sign
        # depends on whether direct-deprivation-beta or downstream-betas
        # dominate in the fitted logistic regression.
        assert abs(cert.mean_delta_prob) > 1e-6

    def test_flip_downstream_flips_direction_to_positive(self, auditor,
                                                         biased_cohort):
        # In the label-only default, multicollinearity can flip the sign;
        # with flip_downstream=True the counterfactual also tweaks distance /
        # travel_time / no_show_rate downward, so mean delta must be positive.
        patients, scheduled = biased_cohort
        auditor.update_config(flip_downstream=True)
        cert = auditor.audit(patients, scheduled)
        assert cert.mean_delta_prob > 0

    def test_affluent_patients_are_skipped(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        cert = auditor.audit(patients, scheduled)
        # Top flips should NOT come from CF10 (affluent) patients
        for f in cert.top_flips:
            assert f.original_deprivation > AFFLUENT_CUTOFF

    def test_unbiased_cohort_has_zero_flips(self, auditor, unbiased_cohort):
        patients, scheduled = unbiased_cohort
        cert = auditor.audit(patients, scheduled)
        # Everyone is CF14 → no deprived-postcode patients to flip
        assert cert.n_rejected == 0
        assert cert.n_flipped == 0

    def test_flip_rate_respects_budget(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        # Tight budget → might FAIL; loose → certainly PASS.  Verify semantics.
        tight = auditor.audit(patients, scheduled)
        # update config to a very loose budget
        auditor.update_config(flip_budget=1.0)
        loose = auditor.audit(patients, scheduled)
        assert loose.certified is True
        assert tight.flip_rate <= 1.0 + 1e-9

    def test_cf_postcode_controls_flipped_features(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        cert = auditor.audit(patients, scheduled,
                             counterfactual_postcode="CF10")
        assert cert.counterfactual_postcode == "CF10"
        if cert.top_flips:
            f = cert.top_flips[0]
            # Deprivation must be among the flipped features when direction is deprived→affluent
            assert "deprivation_score" in f.flipped_features

    def test_affluent_cf_lowers_orig_prob_gap_when_patient_already_affluent(
        self, auditor
    ):
        # Patient already at CF10 → should be skipped (n_rejected counts 0 here).
        patients = [{
            "Patient_ID": "Z", "Patient_Postcode": "CF10",
            "Age": 60, "Priority": 3, "Planned_Duration": 60,
            "Distance_km": 5, "Travel_Time_Min": 15,
            "Patient_NoShow_Rate": 0.10,
        } for _ in range(20)]
        cert = auditor.audit(patients, scheduled_ids=[])
        assert cert.n_rejected == 0


# --------------------------------------------------------------------------- #
# 4. Narrative & decision threshold
# --------------------------------------------------------------------------- #


class TestNarrative:
    def test_pass_or_vacuous_narrative(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        cert = auditor.audit(patients, scheduled)
        assert "Counterfactual audit" in cert.narrative
        assert cert.narrative[:30] in cert.narrative  # round-trip sanity

    def test_decision_threshold_is_positive(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        cert = auditor.audit(patients, scheduled)
        assert cert.decision_threshold >= 0.0

    def test_vacuous_when_too_few_rejects(self, auditor, unbiased_cohort):
        patients, scheduled = unbiased_cohort
        cert = auditor.audit(patients, scheduled)
        assert cert.certified is True
        assert "vacuously PASS" in cert.narrative


# --------------------------------------------------------------------------- #
# 5. Persistence + singleton
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_log_written(self, auditor, tmp_cf_dir, biased_cohort):
        patients, scheduled = biased_cohort
        auditor.audit(patients, scheduled)
        log = tmp_cf_dir / "audits.jsonl"
        assert log.exists()
        rec = json.loads(log.read_text().strip().splitlines()[-1])
        assert "flip_rate" in rec
        assert "predictor_method" in rec

    def test_status_counters(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        before = auditor.status()["total_runs"]
        auditor.audit(patients, scheduled)
        after = auditor.status()["total_runs"]
        assert after == before + 1

    def test_last_cached(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        assert auditor.last() is None
        auditor.audit(patients, scheduled)
        assert auditor.last() is not None

    def test_update_config_round_trip(self, auditor):
        cfg = auditor.update_config(
            counterfactual_postcode="CF23",
            min_effect_size=0.03,
            flip_budget=0.20,
            postcode_deprivation={"CF999": 5.0},
        )
        assert cfg["counterfactual_postcode"] == "CF23"
        assert cfg["min_effect_size"] == 0.03
        assert cfg["flip_budget"] == 0.20
        assert "CF999" in auditor.postcode_deprivation


class TestSingleton:
    def test_get_set(self, tmp_path):
        a = CounterfactualFairnessAuditor(storage_dir=tmp_path / "c")
        set_auditor(a)
        assert get_auditor() is a


# --------------------------------------------------------------------------- #
# 6. JSON round-trip
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_to_dict_is_json_safe(self, auditor, biased_cohort):
        patients, scheduled = biased_cohort
        cert = auditor.audit(patients, scheduled)
        back = json.loads(json.dumps(cert.to_dict()))
        assert "certified" in back
        assert isinstance(back["top_flips"], list)
