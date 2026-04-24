"""
Tests for ml/causal_model.py + ml/causal_validation.py — Wave 3.2 (T3 coverage).

Covers:
- CausalDAG construction + parents/children/descendants + d-separation sanity
- SchedulingCausalModel builds scheduling DAG and supports causal_effect
- do-operator returns InterventionDistribution with callable probability()
- counterfactual returns CounterfactualResult with explanation
- compute_causal_effect returns CausalEffect dataclass
- IV 2SLS estimation on synthetic data
- DML on synthetic data with binary treatment (sklearn fallback ok)
- CausalValidator run_all_tests on synthetic data returns dict with pass/fail
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.causal_model import (
    CausalDAG,
    CausalEffect,
    CounterfactualResult,
    DMLResult,
    DoubleMachineLearning,
    InterventionDistribution,
    IVEstimationResult,
    InstrumentalVariablesEstimator,
    SchedulingCausalModel,
    compute_intervention_effect,
)
from ml.causal_validation import CausalValidator


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _iv_data(n=120, seed=0):
    """Weather -> Traffic -> No-Show (valid IV with reasonable F-stat)."""
    rng = np.random.default_rng(seed)
    weather = rng.uniform(0, 1, n)
    # Traffic delay strongly driven by weather.
    traffic = 10 + 30 * weather + rng.normal(0, 3, n)
    # No-show driven by traffic.
    logits = -2 + 0.05 * traffic + rng.normal(0, 0.3, n)
    noshow = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    return pd.DataFrame({
        "weather_severity": weather,
        "traffic_delay": traffic,
        "no_show": noshow,
        "age": rng.integers(20, 90, n),
    })


def _dml_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Binary treatment depends on covariates.
    t = (0.3 * x1 - 0.2 * x2 + rng.normal(0, 0.5, n) > 0).astype(int)
    # Outcome: 0.5*T + 0.4*x1 + 0.2*x2 + noise
    y = 0.5 * t + 0.4 * x1 + 0.2 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t, "outcome": y})


def _validation_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    weather_severity = rng.uniform(0, 1, n)
    attended = ["Yes"] * n
    for i in range(n):
        p_noshow = 0.1 + 0.3 * weather_severity[i]
        if rng.random() < p_noshow:
            attended[i] = "No"
    return pd.DataFrame({
        "Weather_Severity": weather_severity,
        "Attended_Status": attended,
        "Day_Of_Week_Num": rng.integers(0, 5, n),
        "Traffic_Delay_Minutes": 10 + 30 * weather_severity + rng.normal(0, 5, n),
        "Chair_Number": rng.integers(1, 20, n),
        "Appointment_Hour": rng.integers(8, 17, n),
        "Patient_ID": [f"P{i:04d}" for i in range(n)],
        "Actual_Duration": rng.normal(120, 30, n),
    })


# --------------------------------------------------------------------------- #
# CausalDAG
# --------------------------------------------------------------------------- #


class TestCausalDAG:
    def test_add_nodes_and_edges(self):
        dag = CausalDAG()
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        assert dag.get_parents("B") == ["A"]
        assert dag.get_children("B") == ["C"]
        assert dag.get_ancestors("C") == {"A", "B"}
        assert dag.get_descendants("A") == {"B", "C"}

    def test_find_backdoor_adjustment_set(self):
        dag = CausalDAG()
        # Classic confounding structure: Z -> X, Z -> Y, X -> Y
        dag.add_edge("Z", "X")
        dag.add_edge("Z", "Y")
        dag.add_edge("X", "Y")
        adj = dag.find_backdoor_adjustment_set("X", "Y")
        assert adj is not None
        # Z must be in adjustment set to block backdoor.
        assert "Z" in adj


# --------------------------------------------------------------------------- #
# SchedulingCausalModel + do-calculus
# --------------------------------------------------------------------------- #


class TestSchedulingCausalModel:
    def test_causal_effect_returns_dataclass(self):
        model = SchedulingCausalModel()
        model._use_default_probabilities()
        effect = model.compute_causal_effect(
            treatment="reminder",
            outcome="no_show",
            treatment_value="phone",
            control_value="none",
        )
        assert isinstance(effect, CausalEffect)
        assert 0.0 <= effect.causal_effect <= 1.0
        assert 0.0 <= effect.baseline_effect <= 1.0
        # ATE is well-defined finite number.
        assert np.isfinite(effect.ate)
        assert effect.identification_formula

    def test_do_returns_intervention_distribution(self):
        model = SchedulingCausalModel()
        model._use_default_probabilities()
        dist = model.do("appointment_time", "early")
        assert isinstance(dist, InterventionDistribution)
        p = dist.probability("no_show", outcome_value=1)
        assert 0.0 <= p <= 1.0

    def test_counterfactual_returns_dataclass(self):
        model = SchedulingCausalModel()
        model._use_default_probabilities()
        result = model.counterfactual(
            observation={"weather": "severe", "patient_history": "poor", "no_show": 1},
            intervention={"reminder": "phone"},
            outcome="no_show",
        )
        assert isinstance(result, CounterfactualResult)
        assert 0.0 <= result.probability <= 1.0
        assert result.explanation


# --------------------------------------------------------------------------- #
# Instrumental Variables
# --------------------------------------------------------------------------- #


class TestIV:
    def test_iv_estimator_fit_produces_result(self):
        data = _iv_data(n=200, seed=1)
        estimator = InstrumentalVariablesEstimator()
        result = estimator.fit(
            data,
            instrument="weather_severity",
            treatment="traffic_delay",
            outcome="no_show",
            covariates=["age"],
        )
        assert isinstance(result, IVEstimationResult)
        assert result.n_observations == 200
        # F-stat is non-negative.
        assert result.first_stage_f_stat >= 0.0
        lo, hi = result.confidence_interval
        assert lo <= result.causal_effect <= hi


# --------------------------------------------------------------------------- #
# Double Machine Learning
# --------------------------------------------------------------------------- #


class TestDML:
    def test_dml_fit_returns_result(self):
        data = _dml_data(n=100, seed=2)
        dml = DoubleMachineLearning(n_folds=3, random_state=0)
        result = dml.fit(
            data,
            treatment="treatment",
            outcome="outcome",
            covariates=["x1", "x2"],
        )
        assert isinstance(result, DMLResult)
        assert result.n_folds == 3
        assert result.n_observations == 100
        # Standard error + CI should be present.
        assert result.standard_error >= 0
        lo, hi = result.confidence_interval
        assert lo <= result.treatment_effect <= hi


# --------------------------------------------------------------------------- #
# Convenience + Validator
# --------------------------------------------------------------------------- #


class TestConvenience:
    def test_compute_intervention_effect_default(self):
        effect = compute_intervention_effect("reminder", "sms", "no_show")
        assert isinstance(effect, CausalEffect)
        assert effect.treatment == "reminder"


class TestCausalValidator:
    def test_run_all_tests_returns_structured_dict(self):
        data = _validation_data(n=300, seed=5)
        validator = CausalValidator(tolerance=0.1)
        out = validator.run_all_tests(data, causal_model=None)
        assert "total_tests" in out
        assert "passed" in out
        assert "failed" in out
        assert "tests" in out
        assert isinstance(out["tests"], list)
        assert out["total_tests"] == len(out["tests"])
        assert "interpretation" in out


class TestCausalValidationScope:
    """
    Regression for §4.6.1 external-review finding: the dissertation
    claimed "All seven tests pass ... confirms the weather-no-show
    ATE = 0.084 is robust" as if this were real-world validation.
    Because the synthetic data the validator runs against were
    generated from the same DAG the tests verify, passing the tests
    confirms implementation correctness — NOT real-world causal
    identification in the Velindre population.

    Lock this scope at analysis time so the prose cannot silently
    upgrade from "synthetic DAG recovery" to "real-world validated"
    without a real observational cohort being wired in.
    """

    def test_validation_runs_on_synthetic_data(self):
        """
        Sanity: the validator runs against a synthetic frame built
        by the dissertation's own generator, NOT a real Velindre
        cohort.  The tracer is the column coming from the synthetic
        DAG sampler (Weather_Severity + Traffic_Delay_Minutes + the
        DAG's other observable nodes); real Velindre data would
        carry postcode / regimen / staff metadata instead.
        """
        data = _validation_data(n=100, seed=0)
        assert isinstance(data, pd.DataFrame)
        # Synthetic-DAG observables that must all be present
        for col in ("Weather_Severity", "Traffic_Delay_Minutes",
                    "Appointment_Hour"):
            assert col in data.columns, (
                f"missing DAG observable column {col!r} — fixture drift"
            )
        # Real-Velindre-only columns that MUST NOT be present in
        # the synthetic validation frame.  If any appear, someone
        # has swapped in a real cohort and the §4.6.1 prose needs
        # to flip from "synthetic DAG recovery" to a real-cohort
        # claim.
        velindre_only = ("Patient_Postcode", "Regimen_Code", "Site_Code",
                         "Chair_Type", "Priority_Int")
        leaked = [c for c in velindre_only if c in data.columns]
        assert leaked == [], (
            f"Validation fixture includes real-Velindre columns {leaked}; "
            "if you are intentionally swapping in real observational data, "
            "also update dissertation §4.6.1's \\causalValidationScope "
            "narrative + the R macro emitter in dissertation_analysis.R "
            "Section 21."
        )

    def test_passing_validator_does_not_imply_realworld_causality(self):
        """
        This is a semantic-contract test — it doesn't check behaviour,
        it asserts a documentation invariant that the author must keep
        honest.  Parsed by the dissertation regression suite: if the
        R script's \\causalValidationScope macro ever flips from the
        "synthetic DAG recovery" family to "real observational",
        this test must be updated in the same commit to describe why.
        """
        SCOPE_LABEL = "synthetic DAG recovery (implementation-verification only)"
        assert SCOPE_LABEL.startswith("synthetic"), (
            "If you intentionally upgraded the causal validation to "
            "a real observational cohort, flip this constant, update "
            "dissertation §4.6.1's \\causalValidationScope prose, and "
            "add the real-cohort fixture to test_causal_model.py."
        )
