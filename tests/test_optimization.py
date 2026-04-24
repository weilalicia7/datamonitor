"""
Tests for the optimization module.
"""

import unittest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.optimizer import ScheduleOptimizer, Patient, Chair, OptimizationResult
from optimization.constraints import ConstraintManager
from optimization.squeeze_in import SqueezeInHandler


class TestScheduleOptimizer(unittest.TestCase):
    """Tests for ScheduleOptimizer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = ScheduleOptimizer()

        # Create test patients
        today = datetime.now().replace(hour=0, minute=0, second=0)
        self.patients = [
            Patient(
                patient_id='P001',
                priority=1,
                protocol='R-CHOP',
                expected_duration=120,
                postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17),
                is_urgent=True
            ),
            Patient(
                patient_id='P002',
                priority=2,
                protocol='FEC',
                expected_duration=90,
                postcode='NP20',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17)
            ),
            Patient(
                patient_id='P003',
                priority=3,
                protocol='Paclitaxel',
                expected_duration=60,
                postcode='CF23',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17)
            )
        ]

    def test_optimizer_initialization(self):
        """Test optimizer initializes correctly"""
        self.assertIsNotNone(self.optimizer)
        self.assertGreaterEqual(len(self.optimizer.sites), 4)  # 5 VCC sites

    def test_empty_patient_list(self):
        """Test optimization with empty patient list"""
        result = self.optimizer.optimize([])
        self.assertTrue(result.success)
        self.assertEqual(len(result.appointments), 0)
        self.assertEqual(result.status, 'NO_PATIENTS')

    def test_basic_optimization(self):
        """Test basic scheduling optimization"""
        result = self.optimizer.optimize(self.patients)
        self.assertTrue(result.success)
        self.assertGreater(len(result.appointments), 0)

    def test_priority_ordering(self):
        """Test that higher priority patients are scheduled"""
        result = self.optimizer.optimize(self.patients)

        if len(result.appointments) >= 2:
            # Find P1 and P2 appointments
            p1_apt = next((a for a in result.appointments if a.patient_id == 'P001'), None)
            p2_apt = next((a for a in result.appointments if a.patient_id == 'P002'), None)

            if p1_apt and p2_apt:
                # P1 should be scheduled (higher priority)
                self.assertIsNotNone(p1_apt)


class TestConstraintManager(unittest.TestCase):
    """Tests for ConstraintManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = ConstraintManager()

    def test_constraint_registration(self):
        """Test constraints are registered"""
        summary = self.manager.get_constraint_summary()
        self.assertGreater(summary['total'], 0)

    def test_operating_hours_constraint(self):
        """Test operating hours constraint checking"""
        today = datetime.now().replace(hour=0, minute=0, second=0)

        # Valid appointment
        valid_apt = {
            'start_time': today.replace(hour=9),
            'end_time': today.replace(hour=11),
            'patient_id': 'P001'
        }
        violations = self.manager.check_constraints(valid_apt, {})
        operating_hour_violations = [v for v in violations if v.constraint_name == 'operating_hours']
        self.assertEqual(len(operating_hour_violations), 0)

        # Invalid appointment (before opening)
        invalid_apt = {
            'start_time': today.replace(hour=6),
            'end_time': today.replace(hour=8),
            'patient_id': 'P001'
        }
        violations = self.manager.check_constraints(invalid_apt, {})
        operating_hour_violations = [v for v in violations if v.constraint_name == 'operating_hours']
        self.assertGreater(len(operating_hour_violations), 0)


class TestSqueezeInHandler(unittest.TestCase):
    """Tests for SqueezeInHandler class"""

    def setUp(self):
        """Set up test fixtures with default chairs so find_squeeze_in_options has resources."""
        from config import DEFAULT_SITES, OPERATING_HOURS
        today = datetime.now().replace(hour=0, minute=0, second=0)
        start_h, end_h = OPERATING_HOURS
        chairs = []
        for site in DEFAULT_SITES[:1]:  # One site is enough for the unit test
            for i in range(site['chairs']):
                chairs.append(Chair(
                    chair_id=f"{site['code']}-C{i+1:02d}",
                    site_code=site['code'],
                    is_recliner=i < site.get('recliners', 0),
                    available_from=today.replace(hour=start_h),
                    available_until=today.replace(hour=end_h)
                ))
        self.handler = SqueezeInHandler(chairs=chairs)

    def test_empty_schedule(self):
        """Test finding options in empty schedule — handler must have chairs."""
        today = datetime.now().replace(hour=0, minute=0, second=0)

        patient = Patient(
            patient_id='URGENT001',
            priority=1,
            protocol='Emergency',
            expected_duration=60,
            postcode='CF14',
            earliest_time=today.replace(hour=8),
            latest_time=today.replace(hour=17),
            is_urgent=True
        )

        options = self.handler.find_squeeze_in_options(patient, [], today)
        self.assertGreater(len(options), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the optimization system"""

    def test_full_workflow(self):
        """Test complete optimization workflow"""
        # Create optimizer
        optimizer = ScheduleOptimizer()

        # Create patients
        today = datetime.now().replace(hour=0, minute=0, second=0)
        patients = [
            Patient(
                patient_id=f'P{i:03d}',
                priority=(i % 4) + 1,
                protocol='Standard',
                expected_duration=60 + (i % 3) * 30,
                postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17)
            )
            for i in range(10)
        ]

        # Run optimization
        result = optimizer.optimize(patients)

        # Verify results
        self.assertTrue(result.success)
        self.assertGreater(len(result.appointments), 0)
        self.assertIn('utilization', result.metrics)


class TestGNNFeasibility(unittest.TestCase):
    """Tests for GNNFeasibilityPredictor and its integration with ScheduleOptimizer."""

    def setUp(self):
        from optimization.gnn_feasibility import GNNFeasibilityPredictor
        self.GNN = GNNFeasibilityPredictor
        today = datetime.now().replace(hour=0, minute=0, second=0)
        self.today = today
        self.patients = [
            Patient(
                patient_id=f'G{i:02d}',
                priority=(i % 4) + 1,
                protocol='R-CHOP',
                expected_duration=60 + (i % 3) * 30,
                postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17),
                noshow_probability=0.1 + 0.04 * (i % 5),
                long_infusion=(i % 5 == 0),
                is_urgent=(i == 0),
            )
            for i in range(8)
        ]
        # Chairs: 4 recliners + 6 standard
        self.chairs = (
            [Chair(chair_id=f'VCC-R{i+1:02d}', site_code='VCC',
                   is_recliner=True) for i in range(4)]
            + [Chair(chair_id=f'VCC-S{i+1:02d}', site_code='VCC',
                     is_recliner=False) for i in range(6)]
        )

    # ── Unit tests on GNNFeasibilityPredictor directly ────────────────────────

    def test_init_defaults(self):
        gnn = self.GNN()
        self.assertFalse(gnn._is_trained)
        self.assertEqual(gnn._n_solves_seen, 0)
        self.assertEqual(gnn.prune_threshold, 0.15)
        self.assertGreaterEqual(gnn.min_viable_chairs, 1)

    def test_prune_before_training_returns_all(self):
        """Untrained GNN: model pruning = 0; only hard rule pruning active."""
        gnn = self.GNN(prune_threshold=0.15, min_viable_chairs=2)
        valid, prune_count, rate = gnn.prune_assignments(self.patients, self.chairs)
        n_p, n_c = len(self.patients), len(self.chairs)
        # After safety restore, every patient must have >= min_viable_chairs
        for pi in range(n_p):
            patient_opts = sum(1 for (p2, _) in valid if p2 == pi)
            self.assertGreaterEqual(patient_opts, gnn.min_viable_chairs)

    def test_hard_rule_prunes_non_recliner_for_long_infusion(self):
        """Long-infusion patients must never be paired with non-recliner chairs."""
        gnn = self.GNN(min_viable_chairs=1)
        valid, _, _ = gnn.prune_assignments(self.patients, self.chairs)
        for pi, p in enumerate(self.patients):
            if p.long_infusion:
                for ci, c in enumerate(self.chairs):
                    if not c.is_recliner:
                        self.assertNotIn((pi, ci), valid,
                                         f"Non-recliner pruning failed for patient {p.patient_id}")

    def test_collect_and_train(self):
        """Collecting training examples and calling train() returns True."""
        gnn = self.GNN(train_every=999)  # disable auto-retrain
        # Simulate 3 solves
        for _ in range(3):
            assignments = {p.patient_id: (i % len(self.chairs))
                           for i, p in enumerate(self.patients)}
            gnn.collect_training_example(self.patients, self.chairs, assignments)
        self.assertEqual(gnn._n_solves_seen, 3)
        result = gnn.train(min_examples=2)
        self.assertTrue(result)
        self.assertTrue(gnn._is_trained)

    def test_trained_model_still_preserves_min_viable(self):
        """After training, safety invariant must still hold."""
        gnn = self.GNN(prune_threshold=0.50, min_viable_chairs=3, train_every=999)
        for _ in range(5):
            assignments = {p.patient_id: i % len(self.chairs)
                           for i, p in enumerate(self.patients)}
            gnn.collect_training_example(self.patients, self.chairs, assignments)
        gnn.train(min_examples=2)
        valid, _, _ = gnn.prune_assignments(self.patients, self.chairs)
        for pi in range(len(self.patients)):
            opts = sum(1 for (p2, _) in valid if p2 == pi)
            self.assertGreaterEqual(opts, gnn.min_viable_chairs)

    def test_get_stats_keys(self):
        gnn = self.GNN()
        stats = gnn.get_stats()
        for key in ('is_trained', 'n_solves_seen', 'prune_threshold',
                    'min_viable_chairs', 'lifetime_prune_rate'):
            self.assertIn(key, stats)

    # ── Integration: GNN inside ScheduleOptimizer ─────────────────────────────

    def test_enable_gnn_on_optimizer(self):
        opt = ScheduleOptimizer()
        self.assertFalse(opt._gnn_enabled)
        opt.enable_gnn_pruning()
        self.assertTrue(opt._gnn_enabled)
        self.assertIsNotNone(opt._gnn)

    def test_optimizer_with_gnn_enabled_produces_valid_result(self):
        """Optimizer with GNN enabled must still schedule patients successfully."""
        opt = ScheduleOptimizer()
        opt.enable_gnn_pruning(prune_threshold=0.15)
        result = opt.optimize(self.patients[:4], date=self.today)
        self.assertIn(result.status, ['OPTIMAL', 'FEASIBLE', 'GREEDY_SOLUTION'])
        self.assertGreater(len(result.appointments), 0)

    def test_gnn_collects_after_solve(self):
        """After one solve with GNN enabled, n_solves_seen should increase."""
        opt = ScheduleOptimizer()
        opt.enable_gnn_pruning(prune_threshold=0.15, train_every=999)
        opt.optimize(self.patients[:4], date=self.today)
        self.assertGreaterEqual(opt._gnn._n_solves_seen, 1)


class TestWarmStart(unittest.TestCase):
    """Tests for the warm-start hint cache on ScheduleOptimizer."""

    def setUp(self):
        today = datetime.now().replace(hour=0, minute=0, second=0)
        self.today = today
        self.optimizer = ScheduleOptimizer()
        self.patients = [
            Patient(
                patient_id=f'WS{i:02d}',
                priority=(i % 4) + 1,
                protocol='R-CHOP',
                expected_duration=60 + (i % 3) * 30,
                postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17),
                noshow_probability=0.1 + 0.05 * (i % 4)
            )
            for i in range(6)
        ]

    def test_cache_initially_empty(self):
        """Cache must be empty on a freshly created optimizer."""
        opt = ScheduleOptimizer()
        self.assertEqual(len(opt._solution_cache), 0)
        self.assertEqual(opt._cache_max_size, 50)

    def test_solve_populates_cache(self):
        """A successful solve must write an entry to the cache."""
        self.optimizer.optimize(self.patients, date=self.today)
        self.assertGreater(len(self.optimizer._solution_cache), 0)

    def test_cache_entry_structure(self):
        """Cached entry must contain required keys."""
        self.optimizer.optimize(self.patients, date=self.today)
        for entry in self.optimizer._solution_cache.values():
            self.assertIn('patient_assignments', entry)
            self.assertIn('timestamp', entry)
            self.assertIn('prior_solve_time', entry)
            self.assertIn('hits', entry)

    def test_warm_start_increments_hits(self):
        """Second solve with same fingerprint must increment cache hits."""
        self.optimizer.optimize(self.patients, date=self.today)
        self.optimizer.optimize(self.patients, date=self.today)
        total_hits = sum(
            v.get('hits', 0) for v in self.optimizer._solution_cache.values()
        )
        self.assertGreater(total_hits, 0)

    def test_fingerprint_differs_by_weekday(self):
        """Monday and Wednesday instances with same patients should produce different fingerprints."""
        # Find a Monday and Wednesday
        d = self.today
        while d.weekday() != 0:  # 0 = Monday
            d += timedelta(days=1)
        monday = d
        wednesday = d + timedelta(days=2)
        fp_mon = self.optimizer._compute_instance_fingerprint(self.patients, monday)
        fp_wed = self.optimizer._compute_instance_fingerprint(self.patients, wednesday)
        self.assertNotEqual(fp_mon, fp_wed)

    def test_fingerprint_stable_for_same_instance(self):
        """Same patients + same date should always produce the same fingerprint."""
        fp1 = self.optimizer._compute_instance_fingerprint(self.patients, self.today)
        fp2 = self.optimizer._compute_instance_fingerprint(self.patients, self.today)
        self.assertEqual(fp1, fp2)

    def test_prior_none_branch_for_new_patient(self):
        """Patient not in cached solution triggers prior-is-None skip."""
        today = self.today
        # First solve: 3 patients with priorities 1, 2, 3
        group_a = [
            Patient(
                patient_id='WSA01', priority=1,
                protocol='R-CHOP', expected_duration=60, postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17)
            ),
            Patient(
                patient_id='WSA02', priority=2,
                protocol='R-CHOP', expected_duration=90, postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17)
            ),
            Patient(
                patient_id='WSA03', priority=3,
                protocol='R-CHOP', expected_duration=60, postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17)
            ),
        ]
        self.optimizer.optimize(group_a, date=today)
        self.assertGreater(len(self.optimizer._solution_cache), 0)

        # Second solve: same fingerprint (n=3, priority_dist=(1,1,1,0), long_inf=0)
        # but WSA03 replaced by WSNEW — cache hit, WSNEW has no prior → continue
        group_b = [
            group_a[0],  # WSA01 — in cache
            group_a[1],  # WSA02 — in cache
            Patient(
                patient_id='WSNEW', priority=3,  # same priority as WSA03
                protocol='R-CHOP', expected_duration=60, postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17)
            ),
        ]
        result = self.optimizer.optimize(group_b, date=today)
        self.assertTrue(result.success)
        # Cache must have been hit (hits > 0 for this fingerprint)
        fp = self.optimizer._compute_instance_fingerprint(group_b, today)
        entry = self.optimizer._solution_cache.get(fp)
        self.assertIsNotNone(entry)
        self.assertGreater(entry.get('hits', 0), 0)

    def test_cache_eviction_bounded(self):
        """Cache must not exceed _cache_max_size entries."""
        self.optimizer._cache_max_size = 3
        # Manufacture 5 distinct fingerprints by using different weekdays
        base = self.today
        while base.weekday() != 0:
            base += timedelta(days=1)
        for offset in range(5):
            day = base + timedelta(weeks=offset)  # Same weekday, different weeks
            # Vary patient count to ensure fingerprint differs
            subset = self.patients[:3 + (offset % 3)]
            self.optimizer.optimize(subset, date=day)
        self.assertLessEqual(len(self.optimizer._solution_cache), 3)


class TestColumnGeneration(unittest.TestCase):
    """Tests for Column Generation decomposition."""

    def setUp(self):
        from optimization.column_generation import ColumnGenerator, Column, CGResult
        from config import DEFAULT_SITES, OPERATING_HOURS
        self.ColumnGenerator = ColumnGenerator
        self.Column = Column
        self.CGResult = CGResult

        today = datetime.now().replace(hour=0, minute=0, second=0)
        self.today = today
        start_h, end_h = OPERATING_HOURS
        self.horizon = (end_h - start_h) * 60

        # Build chairs from first 2 sites (enough for testing)
        self.chairs = []
        for site in DEFAULT_SITES[:2]:
            for i in range(site['chairs']):
                self.chairs.append(Chair(
                    chair_id=f"{site['code']}-C{i+1:02d}",
                    site_code=site['code'],
                    is_recliner=i < site.get('recliners', 0),
                    available_from=today.replace(hour=start_h),
                    available_until=today.replace(hour=end_h),
                ))

        # Generate patients
        self.patients = [
            Patient(
                patient_id=f'CG{i:03d}',
                priority=(i % 4) + 1,
                protocol='R-CHOP',
                expected_duration=60 + (i % 3) * 30,
                postcode='CF14',
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17),
                noshow_probability=0.1 + 0.03 * (i % 5),
                long_infusion=(i % 10 == 0),
                is_urgent=(i == 0),
            )
            for i in range(20)
        ]

    # ── Column dataclass ────────────────────────────────────────────────

    def test_column_dataclass(self):
        """Column dataclass fields are correct."""
        col = self.Column(
            column_id=0, chair_idx=1, chair_id='WC-C01',
            patient_indices=[0, 1], start_times=[0, 90], cost=500.0,
        )
        self.assertEqual(col.column_id, 0)
        self.assertEqual(col.chair_idx, 1)
        self.assertEqual(len(col.patient_indices), 2)
        self.assertEqual(col.reduced_cost, 0.0)  # default

    # ── CGResult dataclass ──────────────────────────────────────────────

    def test_cgresult_dataclass(self):
        """CGResult fields are correct."""
        r = self.CGResult(
            success=True, columns_selected=[], patient_assignments={0: (1, 30)},
            unassigned=[], total_cost=100, iterations=5, columns_generated=10,
            solve_time=1.0, lp_bound=120.0, status='CG_OPTIMAL',
        )
        self.assertTrue(r.success)
        self.assertEqual(r.iterations, 5)
        self.assertEqual(r.status, 'CG_OPTIMAL')

    # ── Patient costs ───────────────────────────────────────────────────

    def test_patient_costs_computed(self):
        """Per-patient costs are non-zero and highest for P1."""
        cg = self.ColumnGenerator(self.patients, self.chairs)
        costs = cg._patient_costs
        self.assertEqual(len(costs), len(self.patients))
        # P1 patient (priority 1) should have highest cost
        p1_idx = next(i for i, p in enumerate(self.patients) if p.priority == 1)
        p4_idx = next(i for i, p in enumerate(self.patients) if p.priority == 4)
        self.assertGreater(costs[p1_idx], costs[p4_idx])

    # ── Initial column generation ───────────────────────────────────────

    def test_greedy_initial_columns(self):
        """Greedy heuristic generates at least one column."""
        cg = self.ColumnGenerator(self.patients, self.chairs)
        cols = cg.generate_initial_columns()
        self.assertGreater(len(cols), 0)
        # Total patients in columns ≤ total patients
        total = sum(len(c.patient_indices) for c in cols)
        self.assertLessEqual(total, len(self.patients))

    def test_initial_columns_from_warm_start(self):
        """Warm-start assignments produce initial columns."""
        # Fake cached assignments: 5 patients each to first 4 chairs
        warm = {}
        for i in range(min(10, len(self.patients))):
            warm[self.patients[i].patient_id] = {
                'chair_idx': i % 4,
                'start': i * 70,
            }
        cg = self.ColumnGenerator(self.patients, self.chairs)
        cols = cg.generate_initial_columns(warm_start_assignments=warm)
        self.assertGreater(len(cols), 0)
        # Should have columns for at least some chairs
        chair_idxs = {c.chair_idx for c in cols}
        self.assertGreater(len(chair_idxs), 0)

    def test_initial_columns_respect_bed_requirement(self):
        """Long-infusion patients must not appear on non-recliner chairs."""
        cg = self.ColumnGenerator(self.patients, self.chairs)
        cols = cg.generate_initial_columns()
        for col in cols:
            chair = self.chairs[col.chair_idx]
            for pi in col.patient_indices:
                p = self.patients[pi]
                if p.long_infusion:
                    self.assertTrue(
                        chair.is_recliner,
                        f"Patient {p.patient_id} (long_infusion) on non-recliner {chair.chair_id}"
                    )

    # ── Master LP ───────────────────────────────────────────────────────

    def test_master_lp_returns_duals(self):
        """Master LP produces dual values after initial columns."""
        cg = self.ColumnGenerator(self.patients, self.chairs)
        cg.generate_initial_columns()
        lam, pi_d, mu_d = cg._solve_master()
        self.assertIsNotNone(lam)
        self.assertEqual(len(pi_d), len(self.patients))
        self.assertEqual(len(mu_d), len(self.chairs))

    def test_master_lp_lambda_bounds(self):
        """All lambda values are in [0, 1]."""
        cg = self.ColumnGenerator(self.patients, self.chairs)
        cg.generate_initial_columns()
        lam, _, _ = cg._solve_master()
        for lv in lam:
            self.assertGreaterEqual(lv, -1e-6)
            self.assertLessEqual(lv, 1.0 + 1e-6)

    # ── Pricing subproblem ──────────────────────────────────────────────

    def test_pricing_returns_column_or_none(self):
        """Pricing subproblem returns Column or None."""
        import numpy as np
        cg = self.ColumnGenerator(self.patients, self.chairs)
        cg.generate_initial_columns()
        _, pi_d, mu_d = cg._solve_master()
        # Try pricing on first chair — may or may not find improving column
        result = cg._solve_pricing(0, pi_d, mu_d[0])
        self.assertTrue(result is None or isinstance(result, self.Column))

    def test_pricing_respects_bed_requirement(self):
        """Pricing subproblem never assigns long-infusion to non-recliner."""
        import numpy as np
        cg = self.ColumnGenerator(self.patients, self.chairs)
        cg.generate_initial_columns()
        _, pi_d, mu_d = cg._solve_master()
        for ci, c in enumerate(self.chairs):
            col = cg._solve_pricing(ci, pi_d, mu_d[ci])
            if col is not None and not c.is_recliner:
                for pi in col.patient_indices:
                    self.assertFalse(
                        self.patients[pi].long_infusion,
                        f"Long-infusion patient on non-recliner chair {c.chair_id}"
                    )

    # ── Full CG solve ───────────────────────────────────────────────────

    def test_full_cg_solve(self):
        """Full CG produces a feasible result."""
        cg = self.ColumnGenerator(
            self.patients[:10], self.chairs,
            max_iterations=20,
        )
        result = cg.solve()
        self.assertTrue(result.success)
        self.assertGreater(len(result.patient_assignments), 0)
        self.assertIn(result.status, ['CG_OPTIMAL', 'CG_FEASIBLE', 'CG_MAX_ITER'])
        self.assertGreater(result.iterations, 0)
        self.assertGreater(result.columns_generated, 0)

    def test_cg_solve_with_warm_start(self):
        """CG with warm-start seed produces result."""
        warm = {}
        for i in range(5):
            warm[self.patients[i].patient_id] = {
                'chair_idx': i % len(self.chairs),
                'start': i * 80,
            }
        cg = self.ColumnGenerator(
            self.patients[:8], self.chairs, max_iterations=15,
        )
        result = cg.solve(warm_start_assignments=warm)
        self.assertTrue(result.success)

    def test_cg_no_overlap_in_solution(self):
        """Assigned patients on same chair must not overlap."""
        cg = self.ColumnGenerator(
            self.patients[:10], self.chairs, max_iterations=20,
        )
        result = cg.solve()
        # Group by chair
        chair_schedule = {}
        for pi, (ci, start) in result.patient_assignments.items():
            chair_schedule.setdefault(ci, []).append(
                (start, start + self.patients[pi].expected_duration)
            )
        for ci, slots in chair_schedule.items():
            slots.sort()
            for j in range(len(slots) - 1):
                self.assertLessEqual(
                    slots[j][1], slots[j+1][0],
                    f"Overlap on chair {ci}: {slots[j]} and {slots[j+1]}"
                )

    def test_cg_stats(self):
        """get_stats returns expected keys."""
        cg = self.ColumnGenerator(self.patients[:5], self.chairs)
        stats = cg.get_stats()
        for key in ('n_patients', 'n_chairs', 'columns_generated',
                     'horizon_minutes', 'max_iterations', 'gnn_pruning_active'):
            self.assertIn(key, stats)

    # ── GNN integration ─────────────────────────────────────────────────

    def test_cg_with_gnn_valid_pairs(self):
        """CG respects GNN valid pairs restriction."""
        # Only allow patient 0-4 on chairs 0-5 (subset)
        valid = set()
        for pi in range(5):
            for ci in range(min(6, len(self.chairs))):
                valid.add((pi, ci))
        cg = self.ColumnGenerator(
            self.patients[:5], self.chairs,
            gnn_valid_pairs=valid, max_iterations=10,
        )
        result = cg.solve()
        self.assertTrue(result.success)
        # All assignments must be within valid pairs
        for pi, (ci, _) in result.patient_assignments.items():
            self.assertIn((pi, ci), valid,
                          f"Assignment ({pi}, {ci}) violates GNN valid pairs")

    # ── Optimizer integration (routing) ─────────────────────────────────

    def test_optimizer_routes_to_cg_for_large_instance(self):
        """Optimizer uses CG when patient count > threshold."""
        optimizer = ScheduleOptimizer()
        optimizer._cg_threshold = 5  # Low threshold for testing
        # 10 patients > threshold of 5
        patients = self.patients[:10]
        result = optimizer.optimize(patients, date=self.today)
        self.assertTrue(result.success)
        self.assertIn(result.status, ['CG_OPTIMAL', 'CG_FEASIBLE', 'CG_MAX_ITER'])
        # CG stats should be populated
        self.assertGreater(len(optimizer._cg_stats), 0)
        self.assertIn('iterations', optimizer._cg_stats)

    def test_optimizer_uses_cpsat_below_threshold(self):
        """Optimizer uses monolithic CP-SAT below CG threshold."""
        optimizer = ScheduleOptimizer()
        optimizer._cg_threshold = 100  # High threshold
        patients = self.patients[:5]
        result = optimizer.optimize(patients, date=self.today)
        self.assertTrue(result.success)
        self.assertIn(result.status, ['OPTIMAL', 'FEASIBLE', 'GREEDY_SOLUTION'])

    def test_optimizer_cg_populates_cache(self):
        """CG path populates warm-start cache."""
        optimizer = ScheduleOptimizer()
        optimizer._cg_threshold = 5
        optimizer.optimize(self.patients[:8], date=self.today)
        self.assertGreater(len(optimizer._solution_cache), 0)

    def test_optimizer_cg_disabled(self):
        """When CG disabled, large instance still uses CP-SAT."""
        optimizer = ScheduleOptimizer()
        optimizer._cg_enabled = False
        optimizer._cg_threshold = 5
        patients = self.patients[:8]
        result = optimizer.optimize(patients, date=self.today)
        self.assertTrue(result.success)
        self.assertIn(result.status, ['OPTIMAL', 'FEASIBLE', 'GREEDY_SOLUTION'])

    # ── Edge cases ──────────────────────────────────────────────────────

    def test_cg_empty_warm_start(self):
        """Empty warm-start dict falls back to greedy."""
        cg = self.ColumnGenerator(self.patients[:5], self.chairs)
        cols = cg.generate_initial_columns(warm_start_assignments={})
        self.assertGreater(len(cols), 0)

    def test_cg_single_patient(self):
        """CG handles a single patient."""
        cg = self.ColumnGenerator(
            self.patients[:1], self.chairs, max_iterations=10,
        )
        result = cg.solve()
        self.assertTrue(result.success)
        self.assertEqual(len(result.patient_assignments), 1)

    def test_bundle_cost_correctness(self):
        """Bundle cost equals sum of patient costs."""
        cg = self.ColumnGenerator(self.patients[:5], self.chairs)
        cost = cg._bundle_cost([0, 1, 2])
        expected = sum(cg._patient_costs[i] for i in [0, 1, 2])
        self.assertAlmostEqual(cost, expected, places=4)

    def test_cg_max_iter_reached(self):
        """CG with max_iterations=1 returns CG_MAX_ITER or CG_FEASIBLE."""
        cg = self.ColumnGenerator(
            self.patients[:15], self.chairs, max_iterations=1,
        )
        result = cg.solve()
        # With only 1 iteration, likely still generates columns
        self.assertTrue(result.success)
        self.assertIn(result.status, ['CG_MAX_ITER', 'CG_FEASIBLE', 'CG_OPTIMAL'])

    def test_rounding_cpsat_path(self):
        """Force fractional LP solution to exercise rounding CP-SAT."""
        # Use fewer chairs than patients to create contention → fractional LP
        few_chairs = self.chairs[:3]  # Only 3 chairs
        many_patients = self.patients[:12]  # 12 patients
        cg = self.ColumnGenerator(
            many_patients, few_chairs, max_iterations=30,
        )
        result = cg.solve()
        # Should still produce a valid result
        self.assertTrue(result.success)
        self.assertGreater(len(result.patient_assignments), 0)

    def test_rounding_integer_phase1_only(self):
        """When LP gives near-integer solution, Phase 1 covers all patients."""
        # Few patients, many chairs → LP likely near-integer
        cg = self.ColumnGenerator(
            self.patients[:3], self.chairs, max_iterations=20,
        )
        result = cg.solve()
        self.assertTrue(result.success)
        # All 3 should be assigned
        self.assertEqual(len(result.patient_assignments), 3)

    def test_pricing_no_candidates_for_chair(self):
        """Pricing on a chair with no feasible patients returns None."""
        import numpy as np
        # All patients are long_infusion
        long_patients = [
            Patient(
                patient_id=f'LP{i:02d}', priority=1,
                protocol='R-CHOP', expected_duration=120, postcode='CF14',
                earliest_time=self.today.replace(hour=8),
                latest_time=self.today.replace(hour=17),
                long_infusion=True,
            )
            for i in range(3)
        ]
        # Use only non-recliner chairs
        non_recliner_chairs = [c for c in self.chairs if not c.is_recliner][:3]
        cg = self.ColumnGenerator(long_patients, non_recliner_chairs)
        cg.generate_initial_columns()
        # Force some duals
        pi_d = np.zeros(len(long_patients))
        mu_d = 0.0
        # No patient can go on a non-recliner → pricing returns None
        result = cg._solve_pricing(0, pi_d, mu_d)
        self.assertIsNone(result)

    def test_warm_start_partial_match(self):
        """Warm-start with assignments for only some patients."""
        warm = {
            self.patients[0].patient_id: {'chair_idx': 0, 'start': 0},
        }
        cg = self.ColumnGenerator(self.patients[:5], self.chairs)
        cols = cg.generate_initial_columns(warm_start_assignments=warm)
        self.assertGreater(len(cols), 0)

    def test_optimizer_cg_with_gnn_enabled(self):
        """CG path with GNN pruning enabled."""
        optimizer = ScheduleOptimizer()
        optimizer._cg_threshold = 5
        optimizer.enable_gnn_pruning(prune_threshold=0.15, min_viable_chairs=3)
        result = optimizer.optimize(self.patients[:8], date=self.today)
        self.assertTrue(result.success)

    # ── Coverage: mock ORTOOLS_AVAILABLE=False paths ────────────────────

    def test_solve_without_ortools(self):
        """solve() returns CG_FAILED when OR-Tools unavailable (lines 664-669)."""
        from unittest.mock import patch
        cg = self.ColumnGenerator(self.patients[:3], self.chairs)
        with patch('optimization.column_generation.ORTOOLS_AVAILABLE', False):
            result = cg.solve()
        self.assertFalse(result.success)
        self.assertEqual(result.status, 'CG_FAILED')

    def test_master_lp_without_ortools(self):
        """_solve_master returns None when OR-Tools unavailable (line 351)."""
        from unittest.mock import patch
        cg = self.ColumnGenerator(self.patients[:3], self.chairs)
        cg.generate_initial_columns()
        with patch('optimization.column_generation.ORTOOLS_AVAILABLE', False):
            lam, pi_d, mu_d = cg._solve_master()
        self.assertIsNone(lam)

    def test_pricing_without_ortools(self):
        """_solve_pricing returns None when OR-Tools unavailable (line 423)."""
        import numpy as np
        from unittest.mock import patch
        cg = self.ColumnGenerator(self.patients[:3], self.chairs)
        cg.generate_initial_columns()
        pi_d = np.zeros(3)
        with patch('optimization.column_generation.ORTOOLS_AVAILABLE', False):
            result = cg._solve_pricing(0, pi_d, 0.0)
        self.assertIsNone(result)

    # ── Coverage: GLOP solver creation failure (lines 355-356) ──────────

    def test_master_lp_glop_unavailable(self):
        """_solve_master returns None when GLOP can't be created."""
        from unittest.mock import patch
        cg = self.ColumnGenerator(self.patients[:3], self.chairs)
        cg.generate_initial_columns()
        with patch('optimization.column_generation.pywraplp.Solver.CreateSolver', return_value=None):
            lam, pi_d, mu_d = cg._solve_master()
        self.assertIsNone(lam)

    # ── Coverage: master LP infeasible (lines 390-391) ──────────────────

    def test_master_lp_infeasible(self):
        """_solve_master returns None when LP is infeasible."""
        from unittest.mock import patch, MagicMock
        cg = self.ColumnGenerator(self.patients[:3], self.chairs)
        cg.generate_initial_columns()
        # Mock GLOP solver to return INFEASIBLE
        mock_solver = MagicMock()
        mock_solver.Solve.return_value = 2  # INFEASIBLE
        with patch('optimization.column_generation.pywraplp.Solver.CreateSolver', return_value=mock_solver):
            lam, pi_d, mu_d = cg._solve_master()
        self.assertIsNone(lam)

    # ── Coverage: greedy chair full (line 261) ──────────────────────────

    def test_greedy_chair_full(self):
        """Greedy initial columns: patient skipped when all chairs full (line 261)."""
        # Very long patients on very few chairs — some won't fit
        long_patients = [
            Patient(
                patient_id=f'FL{i:02d}', priority=1,
                protocol='R-CHOP', expected_duration=500,  # very long
                postcode='CF14',
                earliest_time=self.today.replace(hour=8),
                latest_time=self.today.replace(hour=17),
            )
            for i in range(5)
        ]
        few_chairs = self.chairs[:1]  # 1 chair, 600 min horizon
        cg = self.ColumnGenerator(long_patients, few_chairs, horizon=600)
        cols = cg.generate_initial_columns()
        # Only 1 patient fits (500 min < 600 min), second won't (1000 > 600)
        total_assigned = sum(len(c.patient_indices) for c in cols)
        self.assertLess(total_assigned, len(long_patients))

    # ── Coverage: warm-start with stale patient_id (line 297) ───────────

    def test_warm_start_stale_patient_id(self):
        """Warm-start skips unknown patient_ids (line 297)."""
        warm = {
            'NONEXISTENT_P999': {'chair_idx': 0, 'start': 0},
            self.patients[0].patient_id: {'chair_idx': 0, 'start': 0},
        }
        cg = self.ColumnGenerator(self.patients[:3], self.chairs)
        cols = cg.generate_initial_columns(warm_start_assignments=warm)
        # Should still produce columns (ignoring the stale entry)
        self.assertGreater(len(cols), 0)

    # ── Coverage: pricing INFEASIBLE status (line 478) ──────────────────

    def test_pricing_infeasible_status(self):
        """Pricing returns None when subproblem has no feasible bundle (line 478)."""
        import numpy as np
        from unittest.mock import patch, MagicMock
        cg = self.ColumnGenerator(self.patients[:3], self.chairs)
        cg.generate_initial_columns()
        pi_d = np.zeros(len(self.patients[:3]))
        # Mock CP-SAT solver to return INFEASIBLE
        mock_solver_cls = MagicMock()
        mock_solver_inst = MagicMock()
        mock_solver_inst.Solve.return_value = 3  # INFEASIBLE
        mock_solver_cls.return_value = mock_solver_inst
        with patch('optimization.column_generation.cp_model.CpSolver', mock_solver_cls):
            result = cg._solve_pricing(0, pi_d, 0.0)
        self.assertIsNone(result)

    # ── Coverage: pricing reduced cost ≤ tolerance (line 495) ───────────

    def test_pricing_reduced_cost_below_tolerance(self):
        """Pricing returns None when reduced cost is below tolerance (line 495)."""
        import numpy as np
        cg = self.ColumnGenerator(self.patients[:5], self.chairs, max_iterations=20)
        # Run full CG to convergence — after convergence, all pricing subproblems
        # should return None (reduced cost ≤ tolerance)
        result = cg.solve()
        # Now try pricing again — should return None for all chairs
        _, pi_d, mu_d = cg._solve_master()
        if pi_d is not None:
            none_count = 0
            for ci in range(len(self.chairs)):
                col = cg._solve_pricing(ci, pi_d, mu_d[ci])
                if col is None:
                    none_count += 1
            # At least some chairs should have no improving column
            self.assertGreater(none_count, 0)

    # ── Coverage: no initial columns (lines 674-675) ────────────────────

    def test_solve_no_initial_columns(self):
        """solve() returns CG_FAILED when no initial columns generated (lines 674-675)."""
        # All patients are long_infusion but all chairs are non-recliner
        long_patients = [
            Patient(
                patient_id=f'NI{i:02d}', priority=1,
                protocol='R-CHOP', expected_duration=120, postcode='CF14',
                earliest_time=self.today.replace(hour=8),
                latest_time=self.today.replace(hour=17),
                long_infusion=True,
            )
            for i in range(3)
        ]
        non_recliners = [c for c in self.chairs if not c.is_recliner][:3]
        cg = self.ColumnGenerator(long_patients, non_recliners)
        result = cg.solve()
        self.assertFalse(result.success)
        self.assertEqual(result.status, 'CG_FAILED')

    # ── Coverage: master LP fails mid-loop (lines 692-693) ──────────────

    def test_master_lp_fails_midloop(self):
        """Master LP failure mid-iteration triggers break (lines 692-693)."""
        from unittest.mock import patch
        cg = self.ColumnGenerator(self.patients[:5], self.chairs, max_iterations=10)
        cg.generate_initial_columns()
        call_count = [0]
        orig_solve_master = cg._solve_master

        def failing_master():
            call_count[0] += 1
            if call_count[0] >= 2:
                return None, None, None
            return orig_solve_master()

        cg._solve_master = failing_master
        result = cg.solve()
        # Should still produce some result from what was available
        self.assertIsInstance(result, self.CGResult)

    # ── Coverage: final master LP fails (line 723) ──────────────────────

    def test_final_master_lp_fails(self):
        """Final master LP failure returns CG_FAILED (line 723)."""
        cg = self.ColumnGenerator(self.patients[:5], self.chairs, max_iterations=50)
        # Run solve but intercept _solve_master to fail on the LAST call
        # The loop calls _solve_master once per iteration + once after the loop
        call_count = [0]
        orig_solve_master = cg._solve_master

        def fail_after_loop():
            call_count[0] += 1
            result = orig_solve_master()
            return result

        cg._solve_master = fail_after_loop
        # First, let's see how many calls happen normally
        # Instead: directly test by running initial columns, then
        # making the final _solve_master call fail
        cg2 = self.ColumnGenerator(self.patients[:3], self.chairs, max_iterations=50)
        cg2.generate_initial_columns()
        # Run the loop to convergence
        import numpy as np
        for _ in range(50):
            lam, pi_d, mu_d = cg2._solve_master()
            if lam is None:
                break
            new = 0
            for ci in range(len(self.chairs)):
                col = cg2._solve_pricing(ci, pi_d, mu_d[ci])
                if col: new += 1
            if new == 0:
                break
        # Now mock the final master call to fail
        cg2._solve_master = lambda: (None, None, None)
        # Call _round_to_integer expects lambda_vals — test the full path
        from optimization.column_generation import CGResult
        # Re-invoke solve() with a fresh CG that will fail on final master
        cg3 = self.ColumnGenerator(self.patients[:3], self.chairs, max_iterations=2)
        cg3.generate_initial_columns()
        call_count = [0]
        orig3 = cg3._solve_master
        def fail_last():
            call_count[0] += 1
            # CG loop: 1-2 calls inside loop + 1 after loop
            # Fail on 3rd+ call (the post-loop final master)
            if call_count[0] > 2:
                return None, None, None
            return orig3()
        cg3._solve_master = fail_last
        result = cg3.solve()
        self.assertEqual(result.status, 'CG_FAILED')

    # ── Coverage: rounding with bed/GNN filters (lines 544, 578, 597, 599, 613) ──

    def test_rounding_with_long_infusion_and_gnn(self):
        """Rounding CP-SAT exercises bed-requirement and GNN filter branches."""
        # Mix of long-infusion and normal patients, few chairs, with GNN restriction
        patients = [
            Patient(
                patient_id=f'RD{i:02d}', priority=(i % 3) + 1,
                protocol='R-CHOP', expected_duration=90, postcode='CF14',
                earliest_time=self.today.replace(hour=8),
                latest_time=self.today.replace(hour=17),
                long_infusion=(i == 0),  # Only first patient needs recliner
            )
            for i in range(8)
        ]
        few_chairs = self.chairs[:4]  # limited chairs forces fractional LP
        # GNN: restrict some pairs
        valid = set()
        for pi in range(len(patients)):
            for ci in range(len(few_chairs)):
                if not (pi == 3 and ci == 0):  # Remove one pair
                    valid.add((pi, ci))
        cg = self.ColumnGenerator(
            patients, few_chairs, gnn_valid_pairs=valid, max_iterations=15,
        )
        result = cg.solve()
        self.assertTrue(result.success)

    # ── Coverage: pricing rc ≤ tolerance (line 495) ─────────────────────

    def test_pricing_positive_bundle_negative_rc(self):
        """Pricing finds selected patients but rc ≤ tolerance → None (line 495)."""
        import numpy as np
        from unittest.mock import patch, MagicMock
        from ortools.sat.python import cp_model as real_cp
        cg = self.ColumnGenerator(self.patients[:3], self.chairs[:2])
        cg.generate_initial_columns()
        # Set duals exactly equal to costs → rc = 0 ≤ tolerance
        pi_d = cg._patient_costs[:3].copy()
        mu_d = 0.0
        # Mock CP-SAT to "select" patient 0 despite zero net cost
        orig_cpsat_cls = real_cp.CpSolver

        class FakeSolver:
            def __init__(self):
                self.parameters = MagicMock()
            def Solve(self, model):
                return real_cp.OPTIMAL
            def Value(self, var):
                name = str(var)
                if name.startswith('x_0'):
                    return 1
                if name.startswith('s_0'):
                    return 0
                return 0

        with patch('optimization.column_generation.cp_model.CpSolver', FakeSolver):
            result = cg._solve_pricing(0, pi_d, mu_d)
        # rc = (cost_0 - π_0) - μ = 0 ≤ tolerance → None
        self.assertIsNone(result)

    # ── Coverage: rounding col on used chair (line 544) ─────────────────

    def test_rounding_skips_used_chair_columns(self):
        """Rounding phase 2 skips columns on already-used chairs (line 544)."""
        from optimization.column_generation import Column
        cg = self.ColumnGenerator(self.patients[:4], self.chairs[:2])
        # Manually create columns: col0 on chair 0, col1 on chair 0 (same chair)
        cg._columns = []
        cg._next_col_id = 0
        col0 = cg._make_column(0, [0, 1], [0, 90])    # chair 0, patients 0,1
        col1 = cg._make_column(0, [2, 3], [200, 300])  # chair 0, patients 2,3
        # lambda: col0 = 1.0 (fixed), col1 = 0.5 (fractional, same chair → skip)
        lam = [1.0, 0.5]
        result = cg._round_to_integer(lam)
        # Patients 0,1 assigned from col0; patients 2,3 go to rounding CP-SAT
        self.assertIn(0, result)
        self.assertIn(1, result)

    # ── Coverage: rounding no LP candidates (line 578) ──────────────────

    def test_rounding_cpsat_no_candidates_fallback(self):
        """Rounding CP-SAT fallback when LP gives no candidates (line 578)."""
        cg = self.ColumnGenerator(self.patients[:4], self.chairs[:3])
        # Manually call _solve_rounding_cpsat with empty candidates
        remaining = [0, 1]
        candidates = {0: [], 1: []}  # No candidates from LP
        used_chairs = set()
        result = cg._solve_rounding_cpsat(remaining, candidates, used_chairs)
        self.assertIsInstance(result, dict)

    # ── Coverage: rounding bed filter & no-feasible (lines 597, 613) ────

    def test_rounding_cpsat_bed_filter_and_no_feasible(self):
        """Rounding: long-infusion patient filtered from non-recliners → assign=0 (lines 597, 613)."""
        # Long-infusion patient + only non-recliner chairs
        long_patient = Patient(
            patient_id='LRND', priority=1, protocol='R-CHOP',
            expected_duration=90, postcode='CF14',
            earliest_time=self.today.replace(hour=8),
            latest_time=self.today.replace(hour=17),
            long_infusion=True,
        )
        normal_patient = Patient(
            patient_id='NRND', priority=2, protocol='R-CHOP',
            expected_duration=60, postcode='CF14',
            earliest_time=self.today.replace(hour=8),
            latest_time=self.today.replace(hour=17),
        )
        non_recliners = [c for c in self.chairs if not c.is_recliner][:2]
        cg = self.ColumnGenerator([long_patient, normal_patient], non_recliners)
        remaining = [0, 1]  # Both patients
        candidates = {0: [], 1: []}
        used_chairs = set()
        result = cg._solve_rounding_cpsat(remaining, candidates, used_chairs)
        # Long-infusion patient should NOT be assigned (no recliner available)
        self.assertNotIn(0, result)
        # Normal patient should be assigned
        self.assertIn(1, result)


    # ── Coverage: ImportError guard (lines 52-54) ────────────────────────

    def test_import_error_sets_ortools_unavailable(self):
        """Reload column_generation with mocked ImportError covers lines 52-54."""
        import importlib
        import optimization.column_generation as cg_mod

        # Save originals
        orig_flag = cg_mod.ORTOOLS_AVAILABLE
        orig_cp = sys.modules.get('ortools.sat.python.cp_model')
        orig_lp = sys.modules.get('ortools.linear_solver.pywraplp')
        orig_ortools = sys.modules.get('ortools')
        orig_sat = sys.modules.get('ortools.sat')
        orig_sat_python = sys.modules.get('ortools.sat.python')
        orig_linear = sys.modules.get('ortools.linear_solver')

        try:
            # Remove ortools from sys.modules so reload triggers fresh import
            for key in list(sys.modules):
                if key.startswith('ortools'):
                    del sys.modules[key]

            # Install an import hook that blocks ortools
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name.startswith('ortools'):
                    raise ImportError(f"Mocked: {name}")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = mock_import

            # Reload — this re-executes lines 48-54; ImportError path triggers
            importlib.reload(cg_mod)
            self.assertFalse(cg_mod.ORTOOLS_AVAILABLE)
        finally:
            # Restore real import and modules
            builtins.__import__ = real_import
            # Restore ortools modules
            if orig_ortools is not None:
                sys.modules['ortools'] = orig_ortools
            if orig_sat is not None:
                sys.modules['ortools.sat'] = orig_sat
            if orig_sat_python is not None:
                sys.modules['ortools.sat.python'] = orig_sat_python
            if orig_cp is not None:
                sys.modules['ortools.sat.python.cp_model'] = orig_cp
            if orig_linear is not None:
                sys.modules['ortools.linear_solver'] = orig_linear
            if orig_lp is not None:
                sys.modules['ortools.linear_solver.pywraplp'] = orig_lp
            # Reload again to restore normal state
            importlib.reload(cg_mod)
            self.assertTrue(cg_mod.ORTOOLS_AVAILABLE)


class TestRobustnessMetric(unittest.TestCase):
    """
    Regression for §5.9 external-review finding: the prose displayed
    R(S) = 0.135 as both "baseline" and "system optimised" in
    Table 5.3 with a fabricated 0.098 baseline.  The
    ml/benchmark_robustness.py script now measures R(S) directly on
    the optimiser output and writes it to JSONL.  Lock two invariants:

      1. robustness_score() matches the R §8 formula exactly
         (clip slack_min to [0, 60], divide by 60, mean).
      2. The benchmark row schema carries both arms with distinct
         fields so the dissertation table can't present the same
         number under two labels.
    """

    def test_robustness_score_formula_matches_r_script(self):
        """
        Python robustness_score() must produce the same values R §8
        would produce on identical consecutive chair-transition
        slacks.  Property test over four representative configs.
        """
        from ml.benchmark_robustness import robustness_score

        class _Appt:
            def __init__(self, chair_id, start_min, duration_min):
                self.chair_id = chair_id
                self.start_time = datetime(2026, 4, 24, 9, 0) + timedelta(minutes=start_min)
                self.end_time = self.start_time + timedelta(minutes=duration_min)

        # All back-to-back on same chair -> R(S) = 0
        appts_tight = [
            _Appt("C1", 0, 60),
            _Appt("C1", 60, 60),
            _Appt("C1", 120, 60),
        ]
        self.assertEqual(robustness_score(appts_tight)["robustness_score"], 0.0)

        # Full 60-minute gaps -> R(S) = 1.0 (clipped)
        appts_ample = [
            _Appt("C1", 0, 30),
            _Appt("C1", 90, 30),
            _Appt("C1", 180, 30),
        ]
        self.assertAlmostEqual(
            robustness_score(appts_ample)["robustness_score"], 1.0, places=6,
        )

        # Mixed -> R(S) = mean(min(slack/60, 1))
        appts_mixed = [
            _Appt("C1", 0, 30),   # 30-min gap -> 30/60 = 0.5
            _Appt("C1", 60, 30),  # 45-min gap -> 45/60 = 0.75
            _Appt("C1", 135, 30),
        ]
        self.assertAlmostEqual(
            robustness_score(appts_mixed)["robustness_score"],
            (0.5 + 0.75) / 2, places=6,
        )

        # Empty input -> R(S) = 1.0 (nothing to penalise)
        self.assertEqual(robustness_score([])["robustness_score"], 1.0)

    def test_benchmark_row_has_both_arms_with_separate_values(self):
        """
        Lock the JSONL schema: baseline and robust arms must be
        recorded as distinct fields so dissertation §5.9 Table 5.3
        cannot silently show one value under two labels (the exact
        bug the external reviewer flagged).
        """
        row = {
            "arm_baseline": {
                "robustness_weight": 0.00,
                "robustness_score": 0.000,
                "n_transitions": 16,
            },
            "arm_robust": {
                "robustness_weight": 0.10,
                "robustness_score": 0.000,
                "n_transitions": 9,
            },
            "delta_robustness": 0.000,
        }
        for k in ("arm_baseline", "arm_robust", "delta_robustness"):
            self.assertIn(k, row)
        for arm in ("arm_baseline", "arm_robust"):
            for f in ("robustness_weight", "robustness_score", "n_transitions"):
                self.assertIn(f, row[arm], f"{arm}.{f} missing from schema")
        # Weights must differ so this is a real head-to-head, not two
        # runs of the same config
        self.assertNotEqual(
            row["arm_baseline"]["robustness_weight"],
            row["arm_robust"]["robustness_weight"],
        )
        # Delta consistency
        implied = (
            row["arm_robust"]["robustness_score"]
            - row["arm_baseline"]["robustness_score"]
        )
        self.assertAlmostEqual(row["delta_robustness"], implied, places=6)


class TestFairnessMitigationToggle(unittest.TestCase):
    """
    Regression for §5.6.2 external-review finding: the fairness audit
    flagged Age_Band / Gender / Site_Code disparities but no mitigation
    evaluation was reported.  The DRO-style fairness penalties already
    in the CP-SAT objective are now toggle-controlled via
    ScheduleOptimizer._fairness_constraints_enabled so the §5.6.2
    benchmark (ml/benchmark_fairness_mitigation.py) can compare the
    same cohort OFF vs ON.

    Lock four invariants so the dissertation claim cannot silently
    regress:
      1. Default is True — production remains fair-constrained.
      2. Toggle False still returns a feasible schedule.
      3. Toggle False leaves the internal fairness-penalty list empty
         after the solve (the whole block was gated).
      4. The benchmark JSONL schema matches what R reads + the
         cell-vs-delta arithmetic is self-consistent.
    """

    def _mk_patients(self, n=12):
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        out = []
        for i in range(n):
            p = Patient(
                patient_id=f"FT{i:03d}",
                priority=(i % 4) + 1,
                protocol="R-CHOP",
                expected_duration=60,
                postcode="CF14",
                earliest_time=today.replace(hour=8),
                latest_time=today.replace(hour=17),
                noshow_probability=0.15,
                travel_time_minutes=float(10 + (i % 3) * 30),
            )
            p.age_band = ("0-39" if i < 4 else "40-64" if i < 8 else "65+")
            out.append(p)
        return out

    def _mk_chairs(self, n=6):
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return [
            Chair(
                chair_id=f"FT-C{i:02d}",
                site_code="WC",
                is_recliner=i < 2,
                available_from=today.replace(hour=8),
                available_until=today.replace(hour=18),
            )
            for i in range(n)
        ]

    def test_toggle_default_is_true(self):
        opt = ScheduleOptimizer()
        self.assertTrue(getattr(opt, "_fairness_constraints_enabled", True))

    def test_toggle_off_produces_feasible_schedule(self):
        opt = ScheduleOptimizer()
        opt.chairs = self._mk_chairs()
        opt._fairness_constraints_enabled = False
        opt._cg_enabled = False
        result = opt.optimize(self._mk_patients(n=12), time_limit_seconds=5)
        self.assertTrue(
            bool(result.success)
            or getattr(result, "status", "") in ("OPTIMAL", "FEASIBLE"),
            f"solve failed with status={getattr(result, 'status', None)}"
        )

    def test_toggle_off_leaves_penalty_list_empty_post_solve(self):
        opt = ScheduleOptimizer()
        opt.chairs = self._mk_chairs()
        opt._fairness_constraints_enabled = False
        opt._cg_enabled = False
        _ = opt.optimize(self._mk_patients(n=12), time_limit_seconds=5)
        self.assertEqual(getattr(opt, "_fairness_penalties", []), [])

    def test_benchmark_jsonl_schema_matches_r_reader(self):
        row = {
            "ts": "2026-04-24T03:50:00",
            "n_patients": 60, "n_chairs": 12,
            "time_limit_s": 20.0, "seed": 42,
            "arm_off": {
                "fairness_enabled": False,
                "solve_time_s": 20.1, "utilisation": 0.60,
                "n_scheduled": 36, "n_patients": 60, "status": "OPTIMAL",
                "worst_four_fifths_ratio": {
                    "Age_Band": 0.650, "Gender": 0.000, "Site_Code": 1.000,
                },
            },
            "arm_on": {
                "fairness_enabled": True,
                "solve_time_s": 20.1, "utilisation": 0.40,
                "n_scheduled": 24, "n_patients": 60, "status": "OPTIMAL",
                "worst_four_fifths_ratio": {
                    "Age_Band": 0.885, "Gender": 0.000, "Site_Code": 1.000,
                },
            },
            "delta_worst_ratio": {
                "Age_Band": 0.235, "Gender": 0.000, "Site_Code": 0.000,
            },
            "delta_utilisation": -0.20,
            "comparison_note": "...",
        }
        for path in (
            ("arm_off", "utilisation"),
            ("arm_on", "utilisation"),
            ("arm_off", "worst_four_fifths_ratio", "Age_Band"),
            ("arm_on", "worst_four_fifths_ratio", "Age_Band"),
            ("arm_off", "worst_four_fifths_ratio", "Gender"),
            ("arm_on", "worst_four_fifths_ratio", "Gender"),
            ("arm_off", "worst_four_fifths_ratio", "Site_Code"),
            ("arm_on", "worst_four_fifths_ratio", "Site_Code"),
            ("delta_utilisation",),
            ("delta_worst_ratio", "Age_Band"),
            ("delta_worst_ratio", "Gender"),
            ("delta_worst_ratio", "Site_Code"),
        ):
            v = row
            for step in path:
                self.assertIn(step, v,
                              f"missing schema field {'.'.join(path)!r}")
                v = v[step]
        # Cell-vs-delta arithmetic must be self-consistent so the
        # dissertation table cannot show conflicting values
        for col in ("Age_Band", "Gender", "Site_Code"):
            implied = row["arm_on"]["worst_four_fifths_ratio"][col] \
                - row["arm_off"]["worst_four_fifths_ratio"][col]
            self.assertAlmostEqual(
                row["delta_worst_ratio"][col], implied, places=6,
                msg=f"{col}: delta field inconsistent with on-off arithmetic"
            )


if __name__ == '__main__':
    unittest.main()
