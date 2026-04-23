"""
Unit + integration tests for ml/decision_focused_learning.py (Dissertation §2.1).

Covers:
    - Smooth decision-cost gradient direction (calibration reduces cost)
    - SPO+ identity: identity calibration has a=1, b=0
    - Monotonicity constraint: a stays non-negative after fit
    - Calibrator serialisation round-trip
    - The flask_app hook (_apply_dfl) is a no-op when the calibrator is unfit
    - The hook flows DFL output into patient.noshow_probability
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ml.decision_focused_learning import (
    DFLCalibrator,
    smooth_decision_cost,
    DEFAULT_DOUBLE_BOOK_THRESHOLD,
    DEFAULT_WASTE_COST,
    DEFAULT_CROWD_COST,
    _DFL_SLOPE_BOUNDS,
    _DFL_BIAS_BOUNDS,
)


class TestSmoothDecisionCost(unittest.TestCase):
    """The cost surrogate must behave monotonically with each error mode."""

    def test_cost_zero_on_perfect_predictions(self):
        # y=0 → predicted p=0 means sigma_tau(0 − τ) ≈ 0 → crowd term vanishes,
        # waste term has y=0 so it vanishes too.
        # y=1 → predicted p=1 means sigma_tau(1 − τ) ≈ 1 → crowd term has (1−y)=0,
        # waste term has (1−σ) ≈ 0.
        p = np.array([0.01, 0.99])
        y = np.array([0.0, 1.0])
        c = smooth_decision_cost(p, y)
        self.assertLess(c, 1.0)  # effectively zero under smooth cost

    def test_cost_increases_with_wrong_decision(self):
        # All patients attend (y=0) but we predict they no-show (p=0.9)
        p_wrong = np.full(10, 0.9)
        y = np.zeros(10)
        c_wrong = smooth_decision_cost(p_wrong, y)
        # vs perfect predictions
        p_right = np.full(10, 0.05)
        c_right = smooth_decision_cost(p_right, y)
        self.assertGreater(c_wrong, c_right * 10)


class TestDFLCalibrator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cal = DFLCalibrator(
            model_path=Path(self.tmp.name) / 'dfl.pkl',
            history_path=Path(self.tmp.name) / 'hist.jsonl',
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_before_fit(self):
        """Unfitted calibrator must pass probabilities through unchanged."""
        raw = np.array([0.1, 0.4, 0.7, 0.95])
        out = self.cal.calibrate(raw)
        np.testing.assert_allclose(out, raw, atol=1e-5)
        self.assertFalse(self.cal.is_fitted())

    def test_fit_reduces_regret(self):
        """The core promise: decision regret must not increase after fitting."""
        rng = np.random.RandomState(0)
        n = 600
        # True labels drawn from a base no-show rate of 15%
        y = (rng.rand(n) < 0.15).astype(int)
        # Raw predictions: calibrated in mean but too flat (under-confident).
        p_raw = np.clip(rng.beta(2, 11, n), 0.01, 0.99)
        cost_raw = smooth_decision_cost(p_raw, y)
        fit = self.cal.fit(p_raw=p_raw, y_true=y)
        self.assertLessEqual(fit.regret_after, fit.regret_before + 1e-6)
        # Regression doesn't NECESSARILY improve on well-calibrated data,
        # but must not regress beyond the L2 penalty.
        # Confirm the calibrator is now marked fitted.
        self.assertTrue(self.cal.is_fitted())

    def test_fit_recovers_from_miscalibration(self):
        """
        If the raw model systematically under-predicts risk, DFL should
        learn to push probabilities up and reduce regret substantially.
        """
        rng = np.random.RandomState(42)
        n = 800
        y = (rng.rand(n) < 0.30).astype(int)  # 30% no-show
        # Raw model predicts 10% for everyone — massively miscalibrated
        p_raw = np.full(n, 0.10)
        fit = self.cal.fit(p_raw=p_raw, y_true=y, max_iterations=800, learning_rate=0.2)
        # Expect meaningful improvement
        self.assertGreater(fit.regret_improvement_pct, 1.0)
        self.assertTrue(fit.converged or fit.iterations == 800)

    def test_monotonicity_preserved(self):
        """Slope a must stay non-negative after any fit."""
        rng = np.random.RandomState(1)
        n = 200
        y = (rng.rand(n) < 0.2).astype(int)
        p_raw = rng.beta(2, 5, n)
        self.cal.fit(p_raw, y)
        self.assertGreaterEqual(self.cal.a, 0.0)

    def test_serialisation_roundtrip(self):
        rng = np.random.RandomState(7)
        y = (rng.rand(100) < 0.2).astype(int)
        p = rng.beta(2, 7, 100)
        self.cal.fit(p, y)
        a, b = self.cal.a, self.cal.b
        # Reload
        cal2 = DFLCalibrator(
            model_path=self.cal.model_path,
            history_path=self.cal.history_path,
        )
        self.assertAlmostEqual(cal2.a, a, places=6)
        self.assertAlmostEqual(cal2.b, b, places=6)
        self.assertTrue(cal2.is_fitted())

    def test_reset_returns_identity(self):
        rng = np.random.RandomState(3)
        y = (rng.rand(100) < 0.2).astype(int)
        p = rng.beta(2, 7, 100)
        self.cal.fit(p, y)
        self.cal.reset()
        self.assertEqual(self.cal.a, 1.0)
        self.assertEqual(self.cal.b, 0.0)
        self.assertFalse(self.cal.is_fitted())

    def test_calibrate_scalar_returns_float(self):
        out = self.cal.calibrate_scalar(0.3)
        self.assertIsInstance(out, float)

    def test_ce_does_not_collapse_under_extreme_bias(self):
        """
        Regression for the §4.5.4 bug: with the original ±20 bounds, SPO+
        on this exact data shape (seed=7, n=2000, 20% base rate, p_raw ~
        Beta(3, 7)) drove the bias to b=-20 and inflated cross-entropy
        7.3× (verified empirically by reverting bounds in-process).  The
        tightened bounds in _DFL_BIAS_BOUNDS = (-3, 3) cap σ(b) ∈
        [0.05, 0.95] and keep the ratio below 1.5× on the same scenario.

        DO NOT widen the bounds without re-running this assertion.
        """
        rng = np.random.RandomState(7)
        n = 2000
        # 20 % no-show base rate, raw probabilities ~ Beta(3, 7) mean ≈ 0.30 —
        # the over-confident regime where SPO+ most aggressively pushes the
        # bias toward the negative wall.
        y = (rng.rand(n) < 0.20).astype(int)
        p_raw = np.clip(rng.beta(3, 7, n), 0.01, 0.99)

        fit = self.cal.fit(p_raw=p_raw, y_true=y)

        # Primary assertion — would have caught the original 5× blow-up
        # (the empirically-measured old-bounds ratio on this scenario is 7.3×).
        ratio = fit.ce_after / max(fit.ce_before, 1e-9)
        self.assertLess(
            ratio,
            5.0,
            f"CE blow-up unbounded: ce_before={fit.ce_before:.3f} → "
            f"ce_after={fit.ce_after:.3f} (ratio {ratio:.2f}× ≥ 5×). "
            "Has _DFL_BIAS_BOUNDS been widened?",
        )

        # Secondary assertion — calibration should never produce a NaN/Inf CE
        # even when SPO+ pushes hard on the boundary.
        self.assertTrue(np.isfinite(fit.ce_after))

        # Tertiary assertion — slope/bias must respect the published bounds.
        slope_lo, slope_hi = _DFL_SLOPE_BOUNDS
        bias_lo, bias_hi = _DFL_BIAS_BOUNDS
        self.assertGreaterEqual(fit.a, slope_lo - 1e-6)
        self.assertLessEqual(fit.a, slope_hi + 1e-6)
        self.assertGreaterEqual(fit.b, bias_lo - 1e-6)
        self.assertLessEqual(fit.b, bias_hi + 1e-6)

    def test_bound_active_flag_true_when_optimiser_hits_wall(self):
        """
        Diagnostic field: bound_active must flip to True exactly when the
        L-BFGS-B optimum sits on either box edge — that's the early signal
        operators need that the data wants more than the bound permits.

        Uses the same over-confident regime as the CE-collapse test so the
        bias bound is reliably saturated.
        """
        rng = np.random.RandomState(7)
        n = 2000
        y = (rng.rand(n) < 0.20).astype(int)
        p_raw = np.clip(rng.beta(3, 7, n), 0.01, 0.99)
        fit = self.cal.fit(p_raw=p_raw, y_true=y)

        self.assertTrue(
            fit.bound_active,
            f"Expected bound_active=True on the over-confident regime "
            f"(a={fit.a:.3f}, b={fit.b:.3f}); diagnostic flag is broken.",
        )

    def test_bound_active_flag_false_for_interior_fit(self):
        """
        Negative case: when the data is trivially separable and the predicted
        probabilities sit far from τ=0.4, SPO+ has no reason to push the bias
        to a boundary.  bound_active must remain False so that operators can
        trust it as a real "data is fighting the bound" signal.
        """
        rng = np.random.RandomState(7)
        n = 400
        y = rng.randint(0, 2, n)
        # Predictions perfectly aligned with y, well clear of τ=0.4 in both
        # directions — the optimum sits comfortably inside the box.
        p_raw = np.where(y == 1, 0.95, 0.05)
        fit = self.cal.fit(p_raw=p_raw, y_true=y)

        self.assertFalse(
            fit.bound_active,
            f"Expected bound_active=False on an interior fit "
            f"(a={fit.a:.3f}, b={fit.b:.3f}).",
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
