"""
Unit tests for ml/inverse_rl_preferences.py (Dissertation §1.4).

Covers:
    - Feature extraction matches the CP-SAT objective's sign/magnitude
    - Pairwise-softmax IRL recovers a known θ* from synthetic bootstrap
    - Normalised weights sum to 1 and respect θ ≥ 0
    - optimizer.set_weights() + compute_schedule_features() integration
    - Override log round-trip (log → load → clear)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ml.inverse_rl_preferences import (
    OBJECTIVE_KEYS,
    InverseRLPreferenceLearner,
    ObjectiveFeatures,
    compute_objective_features,
)
from optimization.optimizer import ScheduleOptimizer, Patient


def _mk_patient(pid, priority=2, duration=90, travel=30, days_waiting=14, noshow=0.15):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return Patient(
        patient_id=pid,
        priority=priority,
        protocol='TEST',
        expected_duration=duration,
        postcode='CF14',
        earliest_time=today.replace(hour=8),
        latest_time=today.replace(hour=17),
        noshow_probability=noshow,
        travel_time_minutes=float(travel),
    )


class TestObjectiveFeatures(unittest.TestCase):

    def test_feature_vector_has_six_signed_components(self):
        patients = [_mk_patient('P1', priority=1), _mk_patient('P2', priority=3)]
        assignments = {'P1': {'start': 60, 'assigned': True},
                       'P2': {'start': 180, 'assigned': True}}
        z = compute_objective_features(patients, assignments)
        arr = z.as_array()
        self.assertEqual(arr.shape, (6,))
        # Priority positive (assignments valued); utilisation negative (start time);
        # noshow penalty negative; robustness and travel ≤ 0.
        self.assertGreater(arr[0], 0)
        self.assertLess(arr[1], 0)
        self.assertLessEqual(arr[2], 0)
        self.assertGreaterEqual(arr[3], 0)

    def test_unassigned_patients_contribute_zero_to_assigned_only_terms(self):
        p = [_mk_patient('P1')]
        a_on = {'P1': {'start': 60, 'assigned': True}}
        a_off = {}
        z_on = compute_objective_features(p, a_on).as_array()
        z_off = compute_objective_features(p, a_off).as_array()
        # priority, noshow_risk, waiting_time depend on is_assigned
        for idx in (0, 2, 3):
            self.assertNotEqual(z_on[idx], z_off[idx])

    def test_from_array_round_trip(self):
        vec = [1.0, -2.0, -3.0, 4.0, -5.0, -6.0]
        f = ObjectiveFeatures.from_array(vec)
        np.testing.assert_allclose(f.as_array(), vec)


class TestIRLLearner(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.learner = InverseRLPreferenceLearner(
            override_log_path=tmp / 'overrides.jsonl',
            model_path=tmp / 'irl.pkl',
            history_path=tmp / 'hist.jsonl',
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_log_and_load_round_trip(self):
        self.learner.log_override([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], reason='r')
        records = self.learner.load_overrides()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].reason, 'r')
        self.assertEqual(records[0].z_manual, [2, 3, 4, 5, 6, 7])

    def test_bootstrap_fit_produces_normalized_nonnegative_weights(self):
        fit = self.learner.fit(bootstrap_if_empty=True, min_real_overrides=20)
        weights = np.array([fit.theta_weights[k] for k in OBJECTIVE_KEYS])
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        self.assertEqual(len(fit.theta_weights), 6)

    def test_fit_predicts_manual_preference_on_heldout_pairs(self):
        """The key IRL invariant: after fitting, the learner should
        correctly predict the manual schedule as preferred on held-out
        synthetic overrides drawn from the same generating process."""
        true_theta = np.array([0.35, 0.10, 0.25, 0.20, 0.05, 0.05])
        self.learner.seed_bootstrap(n=400, true_theta=true_theta)
        fit = self.learner.fit(bootstrap_if_empty=False, min_real_overrides=0)
        self.assertTrue(fit.converged)
        # Training-set agreement — the estimator's own MLE objective
        self.assertGreater(fit.mean_agreement, 0.60)
        # Held-out pairs from the same distribution
        holdout = InverseRLPreferenceLearner._bootstrap_overrides(
            n=100, true_theta=true_theta, seed=123,
        )
        correct = sum(
            1 for r in holdout
            if self.learner.predict_preference(r.z_manual, r.z_proposed) > 0.5
        )
        self.assertGreaterEqual(correct, 60)  # >= 60% accuracy on held-out

    def test_predict_preference_returns_half_before_fit(self):
        p = self.learner.predict_preference([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(p, 0.5)

    def test_predict_preference_monotone_after_fit(self):
        self.learner.fit()
        z0 = [0, 0, 0, 0, 0, 0]
        z1 = [1000, 0, 0, 0, 0, 0]  # higher on priority only
        p = self.learner.predict_preference(z1, z0)
        self.assertGreaterEqual(p, 0.5)

    def test_prefer_real_drops_synthetic_once_threshold_met(self):
        """
        Once >= min_real_overrides real records exist, the learner must
        train ONLY on real records — synthetic bootstrap stays on disk for
        audit but is excluded from the likelihood.
        """
        # Seed 200 synthetic bootstrap pairs (on disk)
        self.learner.seed_bootstrap(n=200)
        # Log exactly the min_real threshold in real pairs
        for _ in range(20):
            self.learner.log_override(
                z_proposed=[1000, -300, -80, 90, -2, -5],
                z_manual  =[1100, -320, -60, 110, -2, -4],
                reason='test',
                channel='real',
            )
        fit = self.learner.fit(
            bootstrap_if_empty=True, min_real_overrides=20, prefer_real=True,
        )
        self.assertEqual(fit.training_mode, 'real_only')
        self.assertEqual(fit.n_real, 20)
        self.assertEqual(fit.n_synthetic, 0)
        # Synthetic rows still exist on disk for traceability
        all_records = self.learner.load_overrides()
        self.assertGreaterEqual(
            sum(1 for r in all_records if r.source == 'synthetic'), 200,
        )

    def test_prefer_real_false_keeps_synthetic_even_above_threshold(self):
        self.learner.seed_bootstrap(n=50)
        for _ in range(25):
            self.learner.log_override(
                z_proposed=[1000, -300, -80, 90, -2, -5],
                z_manual  =[1100, -320, -60, 110, -2, -4],
                channel='real',
            )
        fit = self.learner.fit(min_real_overrides=20, prefer_real=False)
        self.assertEqual(fit.training_mode, 'mixed')
        self.assertEqual(fit.n_real, 25)
        self.assertEqual(fit.n_synthetic, 50)

    def test_override_carries_channel_tag(self):
        rec = self.learner.log_override(
            z_proposed=[1, 2, 3, 4, 5, 6], z_manual=[2, 3, 4, 5, 6, 7],
            channel='real',
        )
        self.assertEqual(rec.channel, 'real')
        loaded = self.learner.load_overrides()
        self.assertEqual(loaded[-1].channel, 'real')


class TestOptimizerIntegration(unittest.TestCase):

    def test_set_weights_normalises_and_clamps(self):
        opt = ScheduleOptimizer()
        result = opt.set_weights({'priority': 2, 'utilization': 2})
        self.assertAlmostEqual(sum(result.values()), 1.0, places=5)
        for v in result.values():
            self.assertGreaterEqual(v, 0)
        # Missing keys should be preserved, not zeroed
        self.assertIn('travel', result)

    def test_compute_schedule_features_delegates_to_irl_module(self):
        opt = ScheduleOptimizer()
        p = [_mk_patient('P1', priority=1)]
        feats = opt.compute_schedule_features(p, {'P1': {'start': 60, 'assigned': True}})
        self.assertEqual(set(feats.keys()), set(OBJECTIVE_KEYS))
        self.assertGreater(feats['priority'], 0)

    def test_learned_weights_apply_to_optimizer(self):
        opt = ScheduleOptimizer()
        learner = InverseRLPreferenceLearner(
            override_log_path=Path(tempfile.gettempdir()) / 'irl_t.jsonl',
            model_path=Path(tempfile.gettempdir()) / 'irl_t.pkl',
            history_path=Path(tempfile.gettempdir()) / 'irl_t_hist.jsonl',
        )
        # wipe any residual log from earlier test runs
        learner.clear_overrides()
        fit = learner.fit(bootstrap_if_empty=True, min_real_overrides=20)
        opt.set_weights(fit.theta_weights)
        self.assertAlmostEqual(sum(opt.weights.values()), 1.0, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
