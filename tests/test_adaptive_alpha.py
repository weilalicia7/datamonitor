"""
Unit + integration tests for ml/adaptive_alpha.py (Dissertation §2.2).

Covers:
    - α(·) is clamped to [α_floor, α_ceil] under extreme inputs
    - α is monotone non-decreasing in P_noshow (β_noshow ≥ 0)
    - α is monotone non-decreasing in occupancy (β_occupancy ≥ 0)
    - β = 0 recovers the legacy fixed-α baseline (back-compat)
    - Persistence round-trip via AdaptiveAlphaPolicy.save / load
    - ConformalPredictor.predict() accepts per-row α and produces
      monotonically wider intervals for larger α (validity direction)
    - Disabled policy bypasses the adaptive path entirely
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ml.adaptive_alpha import (
    AdaptiveAlphaPolicy,
    DEFAULT_ALPHA_BASE,
    DEFAULT_ALPHA_FLOOR,
    DEFAULT_ALPHA_CEIL,
)
from ml.conformal_prediction import ConformalPredictor


class TestAdaptiveAlphaPolicy(unittest.TestCase):

    def test_clamp_at_floor_when_inputs_minimal(self):
        pol = AdaptiveAlphaPolicy(alpha_base=0.0, beta_noshow=0.0,
                                  beta_occupancy=0.0,
                                  alpha_floor=0.05, alpha_ceil=0.20)
        # alpha_base below floor is clamped up to floor
        self.assertAlmostEqual(pol.compute(0.0, 0.0), 0.05)

    def test_clamp_at_ceil_under_extreme_inputs(self):
        pol = AdaptiveAlphaPolicy(alpha_base=0.10, beta_noshow=1.0,
                                  beta_occupancy=1.0,
                                  alpha_floor=0.01, alpha_ceil=0.20)
        # With P_noshow=1 and occupancy=1, raw = 0.10+1+1 = 2.10 → ceil
        self.assertAlmostEqual(pol.compute(1.0, 1.0), 0.20)

    def test_monotone_in_noshow_probability(self):
        pol = AdaptiveAlphaPolicy()
        a_low = pol.compute(0.05, 0.5)
        a_mid = pol.compute(0.20, 0.5)
        a_high = pol.compute(0.50, 0.5)
        self.assertLess(a_low, a_high)
        self.assertLessEqual(a_low, a_mid)
        self.assertLessEqual(a_mid, a_high)

    def test_monotone_in_occupancy(self):
        pol = AdaptiveAlphaPolicy()
        a0 = pol.compute(0.15, 0.0)
        a50 = pol.compute(0.15, 0.5)
        a100 = pol.compute(0.15, 1.0)
        self.assertLessEqual(a0, a50)
        self.assertLessEqual(a50, a100)

    def test_backcompat_default_recovers_baseline(self):
        """With β_1 = β_2 = 0 the policy is a constant α_base — the legacy behaviour."""
        pol = AdaptiveAlphaPolicy(alpha_base=0.10, beta_noshow=0.0, beta_occupancy=0.0)
        self.assertAlmostEqual(pol.compute(0.0, 0.0), 0.10)
        self.assertAlmostEqual(pol.compute(0.5, 0.5), 0.10)
        self.assertAlmostEqual(pol.compute(1.0, 1.0), 0.10)

    def test_disabled_policy_returns_base(self):
        pol = AdaptiveAlphaPolicy(alpha_base=0.10, beta_noshow=0.5,
                                  beta_occupancy=0.5, enabled=False)
        # Any input → alpha_base
        self.assertAlmostEqual(pol.compute(0.9, 0.9), 0.10)

    def test_inputs_clamped_before_formula(self):
        """P_noshow and occupancy are clamped to [0, 1] defensively."""
        pol = AdaptiveAlphaPolicy()
        # pathological inputs from upstream should not push α outside bounds
        self.assertLessEqual(pol.compute(10.0, 10.0), pol.alpha_ceil)
        self.assertGreaterEqual(pol.compute(-5.0, -5.0), pol.alpha_floor)

    def test_batch_equivalent_to_scalar(self):
        pol = AdaptiveAlphaPolicy()
        ps = [0.05, 0.15, 0.40]
        batch = pol.compute_batch(ps, occupancy=0.6)
        scalar = [pol.compute(p, 0.6) for p in ps]
        np.testing.assert_allclose(batch, scalar, rtol=1e-9)

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'adaptive_alpha.json'
            pol = AdaptiveAlphaPolicy(
                alpha_base=0.12, beta_noshow=0.22, beta_occupancy=0.11,
                alpha_floor=0.02, alpha_ceil=0.18, enabled=True,
            )
            pol.save(path)
            self.assertTrue(path.exists())
            reloaded = AdaptiveAlphaPolicy.load(path)
            self.assertAlmostEqual(reloaded.alpha_base, 0.12)
            self.assertAlmostEqual(reloaded.beta_noshow, 0.22)
            self.assertAlmostEqual(reloaded.beta_occupancy, 0.11)
            self.assertAlmostEqual(reloaded.alpha_floor, 0.02)
            self.assertAlmostEqual(reloaded.alpha_ceil, 0.18)
            self.assertTrue(reloaded.enabled)


class TestAdaptiveAlphaInConformalPredictor(unittest.TestCase):
    """
    End-to-end: the ConformalPredictor must (a) accept alpha_adaptive,
    (b) reproduce legacy behaviour when alpha_adaptive is None, (c)
    widen intervals for larger α values.
    """

    def setUp(self):
        rng = np.random.RandomState(0)
        n = 400
        X = rng.randn(n, 3)
        y = X.sum(axis=1) + rng.randn(n) * 0.5
        self.pred = ConformalPredictor(alpha=0.10)
        self.pred.fit(X, y)
        self.X_test = rng.randn(20, 3)

    def test_backcompat_no_alpha_adaptive(self):
        out = self.pred.predict(self.X_test)
        widths = [p.interval_width for p in out]
        self.assertTrue(all(w > 0 for w in widths))
        # All intervals share the same quantile
        self.assertTrue(np.allclose(widths, widths[0], rtol=1e-9))

    def test_wider_alpha_smaller_interval(self):
        """Larger α ⇒ lower coverage ⇒ narrower interval (not wider!)."""
        out_tight = self.pred.predict(self.X_test, alpha_adaptive=[0.01] * 20)
        out_loose = self.pred.predict(self.X_test, alpha_adaptive=[0.20] * 20)
        w_tight = out_tight[0].interval_width
        w_loose = out_loose[0].interval_width
        self.assertGreater(w_tight, w_loose)

    def test_per_row_alpha_produces_monotone_widths(self):
        alphas = [0.01, 0.05, 0.10, 0.15, 0.20] * 4  # length 20
        out = self.pred.predict(self.X_test, alpha_adaptive=alphas)
        # Patients with the same features but different α should still
        # order interval widths monotonically.  Since our test features
        # vary too, check the relationship only for the mean-of-same-α
        # subgroups.
        by_alpha = {}
        for a, p in zip(alphas, out):
            by_alpha.setdefault(a, []).append(p.interval_width)
        mean_width_by_alpha = [(a, float(np.mean(ws)))
                               for a, ws in sorted(by_alpha.items())]
        widths_sorted = [w for _, w in mean_width_by_alpha]
        # α ascending → width descending (wider CI at lower α)
        self.assertEqual(widths_sorted, sorted(widths_sorted, reverse=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
