"""
Unit tests for ml/rct_randomization.py (Dissertation §2.4).

Covers:
  * Deterministic hash-based assignment — same inputs → same arm.
  * Trial-rate control — over a large batch the empirical rate sits
    within a 99 % binomial tolerance of the configured rate.
  * Arm balance — over the trial subset, the 4 arms are close to
    uniform (χ² test tolerance).
  * ATE recovery on synthetic outcomes: seeded with a known truth,
    the Wald CI covers it.
  * `apply_rct_prior` precision-weighted shrinkage:
        – recovers pure DML when σ_rct → ∞
        – recovers pure RCT when σ_rct → 0
        – w_rct in [0, 1]
  * Under-powered flag fires when an arm has fewer than min_n_per_arm
    observations.
  * Assignment + outcome JSONL round-trip reloads cleanly.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ml.rct_randomization import (
    TrialArm,
    TrialAssigner,
    ATEResult,
    apply_rct_prior,
    DEFAULT_TRIAL_RATE,
    DEFAULT_MIN_N_PER_ARM,
)


class TestHashDeterministicAssignment(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.assigner = TrialAssigner(
            assignments_path=tmp / 'a.jsonl',
            outcomes_path=tmp / 'o.jsonl',
            history_path=tmp / 'h.jsonl',
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_inputs_give_same_arm(self):
        a1 = self.assigner.assign('APT100', 'P001')
        a2 = self.assigner.assign('APT100', 'P001')
        a3 = self.assigner.assign('APT100', 'P001')
        self.assertEqual(a1, a2)
        self.assertEqual(a2, a3)

    def test_different_seeds_change_assignment(self):
        a1 = self.assigner.assign('APT100', 'P001')
        alt = TrialAssigner(
            seed='different-seed',
            assignments_path=Path(self.tmp.name) / 'a2.jsonl',
            outcomes_path=Path(self.tmp.name) / 'o2.jsonl',
            history_path=Path(self.tmp.name) / 'h2.jsonl',
        )
        # Over a large batch, different seeds produce different allocations;
        # for a single row the result may coincide — check divergence in
        # aggregate rather than asserting on one row.
        ids = [('APT%04d' % i, 'P%04d' % (i // 3)) for i in range(500)]
        match = sum(
            1 for appt, pid in ids
            if self.assigner.assign(appt, pid) == alt.assign(appt, pid)
        )
        self.assertLess(match, 500)  # some divergence expected


class TestTrialRate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.assigner = TrialAssigner(
            trial_rate=0.05,
            assignments_path=tmp / 'a.jsonl',
            outcomes_path=tmp / 'o.jsonl',
            history_path=tmp / 'h.jsonl',
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_rate_within_binomial_tolerance(self):
        n = 5000
        flagged = sum(
            1 for i in range(n)
            if self.assigner.assign(f'APT{i:05d}', f'P{i % 500:04d}') is not None
        )
        p = flagged / n
        # ±3 SE binomial band at p=0.05, n=5000 → ~0.0308 – 0.0692
        self.assertGreater(p, 0.03)
        self.assertLess(p, 0.07)

    def test_arm_balance(self):
        """Among flagged bookings the 4 arms are roughly balanced."""
        counts = {a.value: 0 for a in TrialArm}
        for i in range(20000):
            arm = self.assigner.assign(f'APT{i:05d}', f'P{i % 2000:04d}')
            if arm is not None:
                counts[arm.value] += 1
        total = sum(counts.values())
        # Under perfect uniformity each arm gets 25 % of the trial.
        # Allow each arm 18 – 32 % (well inside χ² tolerance at this N).
        for a, n in counts.items():
            share = n / max(1, total)
            self.assertGreater(share, 0.18, f'arm {a} under-represented: {share:.3f}')
            self.assertLess(share, 0.32, f'arm {a} over-represented: {share:.3f}')


class TestATEComputation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.assigner = TrialAssigner(
            min_n_per_arm=10,
            assignments_path=tmp / 'a.jsonl',
            outcomes_path=tmp / 'o.jsonl',
            history_path=tmp / 'h.jsonl',
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_ate_recovers_synthetic_truth(self):
        """
        Seed 300 outcomes each for CONTROL (attended rate 0.70) and
        SMS_24H (attended rate 0.85); the Wald 95% CI must cover the
        true ATE of 0.15.
        """
        rng = np.random.RandomState(42)
        for i in range(300):
            self.assigner.record_outcome(
                f'APT-C{i}', f'P-C{i}', TrialArm.CONTROL, bool(rng.rand() < 0.70),
            )
            self.assigner.record_outcome(
                f'APT-T{i}', f'P-T{i}', TrialArm.SMS_24H, bool(rng.rand() < 0.85),
            )
        ate = self.assigner.compute_ate(TrialArm.SMS_24H)
        self.assertFalse(ate.under_powered)
        self.assertGreaterEqual(ate.ate, 0.08)  # expected ~0.15
        self.assertLessEqual(ate.ate, 0.22)
        self.assertLessEqual(ate.ci_low, 0.15)
        self.assertGreaterEqual(ate.ci_high, 0.15)

    def test_under_powered_flag(self):
        self.assigner.record_outcome('APT-1', 'P-1', TrialArm.CONTROL, True)
        self.assigner.record_outcome('APT-2', 'P-2', TrialArm.SMS_24H, True)
        ate = self.assigner.compute_ate(TrialArm.SMS_24H)
        self.assertTrue(ate.under_powered)


class TestPersistenceRoundtrip(unittest.TestCase):

    def test_assignment_and_outcome_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            a1 = TrialAssigner(
                assignments_path=tmp_p / 'a.jsonl',
                outcomes_path=tmp_p / 'o.jsonl',
                history_path=tmp_p / 'h.jsonl',
            )
            a1.log_assignment('APT1', 'P1', TrialArm.CONTROL)
            a1.record_outcome('APT1', 'P1', TrialArm.CONTROL, True)
            a1.record_outcome('APT2', 'P2', TrialArm.SMS_24H, False)

            a2 = TrialAssigner(
                assignments_path=tmp_p / 'a.jsonl',
                outcomes_path=tmp_p / 'o.jsonl',
                history_path=tmp_p / 'h.jsonl',
            )
            counts = a2.arm_counts()
            self.assertEqual(counts['control']['total'], 1)
            self.assertEqual(counts['control']['attended'], 1)
            self.assertEqual(counts['sms_24h']['total'], 1)
            self.assertEqual(counts['sms_24h']['noshow'], 1)


class TestRCTPriorShrinkage(unittest.TestCase):

    def test_infinite_rct_se_yields_dml(self):
        out = apply_rct_prior(dml_ate=0.20, dml_se=0.05,
                              rct_ate=0.50, rct_se=1e6)
        self.assertAlmostEqual(out['posterior_ate'], 0.20, places=3)
        self.assertLess(out['w_rct'], 1e-6)

    def test_tiny_rct_se_yields_rct(self):
        out = apply_rct_prior(dml_ate=0.20, dml_se=0.05,
                              rct_ate=0.50, rct_se=1e-4)
        self.assertAlmostEqual(out['posterior_ate'], 0.50, places=3)
        self.assertGreater(out['w_rct'], 0.999)

    def test_w_rct_in_unit_interval(self):
        for dml_se in (0.01, 0.05, 0.10):
            for rct_se in (0.01, 0.05, 0.10):
                out = apply_rct_prior(0.2, dml_se, 0.3, rct_se)
                self.assertGreaterEqual(out['w_rct'], 0.0)
                self.assertLessEqual(out['w_rct'], 1.0)

    def test_posterior_se_is_tighter(self):
        """Posterior SE must be ≤ both DML SE and RCT SE."""
        out = apply_rct_prior(0.2, 0.05, 0.3, 0.06)
        self.assertLessEqual(out['posterior_se'], 0.05 + 1e-9)
        self.assertLessEqual(out['posterior_se'], 0.06 + 1e-9)


class TestConfigPersistence(unittest.TestCase):

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            a1 = TrialAssigner(
                trial_rate=0.10, min_n_per_arm=25,
                assignments_path=tmp_p / 'a.jsonl',
                outcomes_path=tmp_p / 'o.jsonl',
                history_path=tmp_p / 'h.jsonl',
            )
            a1.save_config(tmp_p / 'cfg.json')
            a2 = TrialAssigner.load_config(tmp_p / 'cfg.json')
            self.assertAlmostEqual(a2.trial_rate, 0.10)
            self.assertEqual(a2.min_n_per_arm, 25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
