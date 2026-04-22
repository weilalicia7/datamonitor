"""
Unit + integration tests for ml/feature_store.py (Dissertation §3.1).

Covers:
  * Entity + FeatureView registration
  * Batch materialisation from a synthetic historical DataFrame
  * Online serving latency (≤100 ms budget for 200 patients, 4 views)
  * Streaming push — patient's features recompute within the same call
  * Point-in-time correctness: training-time `as_of(T)` never returns
    a feature derived from an event after T
  * TTL window: 30-day stats only count events within 30 days
  * Schema version + materialisation log persist across restarts
  * Round-trip save/load keeps the online dict intact
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from ml.feature_store import (
    Entity, FeatureView, FeatureStore, SCHEMA_VERSION,
)


def _mk_events(n_patients: int = 50, appts_per_patient: int = 12,
               noshow_rate_by_patient=None, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic DataFrame shaped like historical_appointments.xlsx."""
    rng = np.random.RandomState(seed)
    base = datetime(2026, 1, 1)
    rows = []
    for p in range(n_patients):
        pid = f"P{p:04d}"
        ns_rate = (noshow_rate_by_patient or {}).get(pid, float(rng.beta(1.5, 8)))
        for j in range(appts_per_patient):
            dt = base + timedelta(days=7 * j + int(rng.randint(0, 3)))
            attended_flag = (rng.rand() > ns_rate)
            rows.append({
                'Patient_ID': pid,
                'Appointment_ID': f'APT_{p:04d}_{j:02d}',
                'Date': dt.isoformat(),
                'Attended_Status': 'Yes' if attended_flag else ('No' if rng.rand() < 0.8 else 'Cancelled'),
                'Planned_Duration': 90,
                'Actual_Duration': float(60 + rng.randn() * 15) if attended_flag else None,
                'Cycle_Number': j + 1,
                'Regimen_Modification_Flag': int(rng.rand() < 0.05),
            })
    return pd.DataFrame(rows)


class TestRegistrationAndMaterialisation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeatureStore(storage_dir=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_views_registered(self):
        views = {v['name'] for v in self.store.status()['views']}
        self.assertEqual(
            views,
            {'patient_30d_stats', 'patient_90d_stats', 'patient_cycle_ctx', 'patient_trend'},
        )

    def test_schema_version_surfaced(self):
        self.assertEqual(self.store.status()['schema_version'], SCHEMA_VERSION)

    def test_materialise_populates_online_store(self):
        df = _mk_events(n_patients=20, appts_per_patient=10)
        mat = self.store.materialize(df)
        self.assertEqual(mat.n_entities, 20)
        self.assertGreaterEqual(mat.n_events_scanned, 200)
        row = self.store.get_online_features(['P0000']).get('P0000') or {}
        self.assertIn('patient_30d_stats__noshow_rate_30d', row)
        self.assertIn('patient_cycle_ctx__current_cycle', row)
        self.assertIn('patient_trend__attended_streak', row)


class TestOnlineServingLatency(unittest.TestCase):
    """Enforce the §3.1 performance guarantee: <100 ms for 200 patients."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeatureStore(storage_dir=Path(self.tmp.name))
        df = _mk_events(n_patients=200, appts_per_patient=6)
        self.store.materialize(df)

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_serving_under_100ms(self):
        pids = [f'P{p:04d}' for p in range(200)]
        t0 = time.perf_counter()
        out = self.store.get_online_features(pids, log_latency=False)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(len(out), 200)
        self.assertLess(elapsed, 100.0,
                        f"serving 200 patients took {elapsed:.2f} ms, budget 100 ms")


class TestStreamingPush(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeatureStore(storage_dir=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_push_event_recomputes_features(self):
        now = datetime.utcnow()
        # 3 prior attended visits
        for j in range(3):
            self.store.push_event({
                'Patient_ID': 'S01',
                'Appointment_ID': f'A{j}',
                'Date': (now - timedelta(days=21 - j * 7)).isoformat(),
                'Attended_Status': 'Yes',
                'Actual_Duration': 90.0,
                'Cycle_Number': j + 1,
            })
        # now push a no-show — 30d noshow rate should jump from 0 to 0.25
        self.store.push_event({
            'Patient_ID': 'S01',
            'Appointment_ID': 'A3',
            'Date': now.isoformat(),
            'Attended_Status': 'No',
            'Actual_Duration': None,
            'Cycle_Number': 4,
        })
        row = self.store.get_online_features(['S01']).get('S01') or {}
        self.assertAlmostEqual(
            row.get('patient_30d_stats__noshow_rate_30d'), 0.25, places=4,
        )
        self.assertEqual(row.get('patient_30d_stats__appointment_count_30d'), 4)

    def test_push_event_latency_small(self):
        t0 = time.perf_counter()
        for i in range(10):
            self.store.push_event({
                'Patient_ID': f'SX{i}',
                'Appointment_ID': f'Z{i}',
                'Date': datetime.utcnow().isoformat(),
                'Attended_Status': 'Yes',
                'Actual_Duration': 60.0,
                'Cycle_Number': 1,
            })
        elapsed = time.perf_counter() - t0
        # 10 pushes well under 1s even on CPU-only CI
        self.assertLess(elapsed, 1.0)


class TestPointInTimeCorrectness(unittest.TestCase):
    """
    Train-time `as_of(T)` must NEVER include events dated after T.
    This is the validity contract for the feature store.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeatureStore(storage_dir=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_as_of_excludes_future_events(self):
        base = datetime(2026, 3, 1)
        # 3 attended visits in March, then a no-show on 2026-04-10
        for j, d in enumerate([base, base + timedelta(days=7),
                               base + timedelta(days=14)]):
            self.store.push_event({
                'Patient_ID': 'PIT', 'Appointment_ID': f'A{j}',
                'Date': d.isoformat(), 'Attended_Status': 'Yes',
                'Actual_Duration': 80.0, 'Cycle_Number': j + 1,
            })
        self.store.push_event({
            'Patient_ID': 'PIT', 'Appointment_ID': 'A3',
            'Date': datetime(2026, 4, 10).isoformat(),
            'Attended_Status': 'No', 'Actual_Duration': None, 'Cycle_Number': 4,
        })
        # As-of 2026-04-01 the no-show was in the future → excluded
        feat_mar = self.store.as_of('PIT', datetime(2026, 4, 1))
        # As-of 2026-05-01 the no-show is in the past → included
        feat_may = self.store.as_of('PIT', datetime(2026, 5, 1))
        self.assertEqual(feat_mar['patient_30d_stats__noshow_rate_30d'], 0.0)
        self.assertGreater(feat_may['patient_30d_stats__noshow_rate_30d'], 0.0)


class TestTTLWindow(unittest.TestCase):

    def test_30d_rolling_only_counts_last_30_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(storage_dir=Path(tmp))
            now = datetime.utcnow()
            # one old event (60 days ago) + three recent ones
            for j, days_ago in enumerate([60, 10, 5, 2]):
                store.push_event({
                    'Patient_ID': 'T1', 'Appointment_ID': f'A{j}',
                    'Date': (now - timedelta(days=days_ago)).isoformat(),
                    'Attended_Status': 'Yes', 'Actual_Duration': 90.0,
                    'Cycle_Number': j + 1,
                })
            row = store.get_online_features(['T1']).get('T1') or {}
            # 30d count = 3 (not 4).  90d count = 4.
            self.assertEqual(row['patient_30d_stats__appointment_count_30d'], 3)
            self.assertEqual(row['patient_90d_stats__appointment_count_90d'], 4)


class TestPersistence(unittest.TestCase):

    def test_roundtrip_online_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1 = FeatureStore(storage_dir=Path(tmp))
            s1.materialize(_mk_events(n_patients=10, appts_per_patient=4))
            row1 = s1.get_online_features(['P0001']).get('P0001') or {}
            self.assertIn('patient_30d_stats__noshow_rate_30d', row1)
            # new instance, same dir → online state is restored
            s2 = FeatureStore(storage_dir=Path(tmp))
            row2 = s2.get_online_features(['P0001']).get('P0001') or {}
            self.assertEqual(
                row1['patient_30d_stats__noshow_rate_30d'],
                row2['patient_30d_stats__noshow_rate_30d'],
            )

    def test_schema_file_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1 = FeatureStore(storage_dir=Path(tmp))
            s1.materialize(_mk_events(n_patients=5, appts_per_patient=3))
            schema_path = Path(tmp) / 'schema.json'
            self.assertTrue(schema_path.exists())
            payload = json.loads(schema_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['schema_version'], SCHEMA_VERSION)
            self.assertIn('patient_30d_stats', payload['views'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
