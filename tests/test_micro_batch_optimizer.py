"""
Unit tests for ml/micro_batch_optimizer.py (Dissertation §3.2).

Covers:
  * Routing — urgent insert → fast path, non-urgent → queued, full
    batch → slow path, unknown type → rejected.
  * Threshold — slow path fires at exactly `change_threshold`
    queued changes and not before.
  * Time-based trigger — slow path fires when the elapsed window
    exceeds `slow_path_interval_s` even if the queue is below
    threshold.
  * Fast-path budget — a mocked fast-path callable that sleeps
    ≤ 10 ms returns within the 50 ms budget.
  * Failed fast path falls through to the queue and doesn't corrupt
    the counters.
  * Slow-path failure re-queues every consumed change (no data loss).
  * Latency log file is append-only JSONL with the required keys.
  * Config update validates positive integers and persists.
  * Background RL thread starts and stops cleanly (no hang).
  * Status dataclass reports the expected state invariants.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.micro_batch_optimizer import (
    MicroBatchOptimizer,
    DEFAULT_CHANGE_THRESHOLD,
)


class TestRouting(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fast_calls = []
        self.slow_calls = []
        self.mb = MicroBatchOptimizer(
            fast_path_fn=lambda p: (self.fast_calls.append(p)
                                     or {'success': True, 'chair_id': 'WC-C01'}),
            slow_path_fn=lambda pending: (self.slow_calls.append(pending)
                                          or {'success': True, 'n': len(pending)}),
            rl_tick_fn=None,
            change_threshold=3,
            slow_path_interval_s=9999,
            storage_dir=Path(self.tmp.name),
        )

    def tearDown(self):
        self.mb.stop_background_rl()
        self.tmp.cleanup()

    def test_urgent_insert_takes_fast_path(self):
        out = self.mb.submit_change('insert', {'patient_id': 'P1', 'is_urgent': True})
        self.assertEqual(out.path, 'fast')
        self.assertTrue(out.success)
        self.assertEqual(len(self.fast_calls), 1)
        self.assertEqual(len(self.slow_calls), 0)

    def test_non_urgent_insert_is_queued(self):
        out = self.mb.submit_change('insert', {'patient_id': 'P1', 'is_urgent': False})
        self.assertEqual(out.path, 'queued')
        self.assertEqual(len(self.fast_calls), 0)
        self.assertEqual(self.mb.status().queue_size, 1)

    def test_unknown_change_type_rejected(self):
        out = self.mb.submit_change('teleport', {})
        self.assertEqual(out.path, 'rejected')
        self.assertFalse(out.success)


class TestThreshold(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.slow_calls = []
        self.mb = MicroBatchOptimizer(
            fast_path_fn=None,          # force everything into queue
            slow_path_fn=lambda pending: (self.slow_calls.append(pending)
                                          or {'success': True}),
            rl_tick_fn=None,
            change_threshold=3,
            slow_path_interval_s=9999,
            storage_dir=Path(self.tmp.name),
        )

    def tearDown(self):
        self.mb.stop_background_rl()
        self.tmp.cleanup()

    def test_slow_path_fires_at_threshold(self):
        a = self.mb.submit_change('cancel', {'appointment_id': 'A1'})
        b = self.mb.submit_change('reschedule', {'appointment_id': 'A2'})
        self.assertEqual(a.path, 'queued')
        self.assertEqual(b.path, 'queued')
        self.assertEqual(len(self.slow_calls), 0)
        c = self.mb.submit_change('cancel', {'appointment_id': 'A3'})
        self.assertEqual(c.path, 'slow')
        self.assertEqual(len(self.slow_calls), 1)
        self.assertEqual(len(self.slow_calls[0]), 3)
        # queue is drained after a successful slow path
        self.assertEqual(self.mb.status().queue_size, 0)

    def test_time_based_trigger(self):
        """After a slow-path run, elapsed-time trigger must fire."""
        # Force a first slow path so last_full_reopt_ts is set
        for i in range(3):
            self.mb.submit_change('cancel', {'appointment_id': f'X{i}'})
        self.assertEqual(self.mb.status().total_slow_path, 1)
        # Tighten the interval to 0 and submit one change
        self.mb.update_config(slow_path_interval_s=1)
        # Manually backdate the last_full_reopt_ts so the next submit fires
        self.mb.last_full_reopt_ts = datetime.utcnow() - timedelta(seconds=10)
        out = self.mb.submit_change('cancel', {'appointment_id': 'Z'})
        self.assertEqual(out.path, 'slow')


class TestFastPathBudget(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mb = MicroBatchOptimizer(
            fast_path_fn=lambda p: (time.sleep(0.005)
                                     or {'success': True}),   # 5 ms fake work
            slow_path_fn=lambda pending: {'success': True},
            rl_tick_fn=None,
            fast_path_budget_ms=50.0,
            storage_dir=Path(self.tmp.name),
        )

    def tearDown(self):
        self.mb.stop_background_rl()
        self.tmp.cleanup()

    def test_fast_path_under_budget(self):
        out = self.mb.submit_change('insert', {'patient_id': 'P1', 'is_urgent': True})
        self.assertEqual(out.path, 'fast')
        self.assertLess(out.latency_ms, self.mb.fast_path_budget_ms,
                        f"fast path took {out.latency_ms:.2f} ms")


class TestFailureHandling(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_fast_path_raise_falls_back_to_queue(self):
        def _boom(_payload):
            raise RuntimeError("boom")
        mb = MicroBatchOptimizer(
            fast_path_fn=_boom,
            slow_path_fn=lambda p: {'success': True},
            rl_tick_fn=None,
            change_threshold=99,
            slow_path_interval_s=9999,
            storage_dir=Path(self.tmp.name),
        )
        out = mb.submit_change('insert', {'patient_id': 'P1', 'is_urgent': True})
        # fast-path exception fell through → change ends up queued
        self.assertEqual(out.path, 'queued')
        self.assertEqual(mb.status().queue_size, 1)
        self.assertEqual(mb.status().total_fast_path, 0)
        mb.stop_background_rl()

    def test_slow_path_exception_requeues(self):
        def _slow_fail(pending):
            raise RuntimeError("solver blew up")
        mb = MicroBatchOptimizer(
            fast_path_fn=None, slow_path_fn=_slow_fail, rl_tick_fn=None,
            change_threshold=1, slow_path_interval_s=9999,
            storage_dir=Path(self.tmp.name),
        )
        out = mb.submit_change('cancel', {'id': 'A1'})
        self.assertEqual(out.path, 'slow')
        self.assertFalse(out.success)
        self.assertEqual(mb.status().queue_size, 1)   # re-queued
        mb.stop_background_rl()


class TestLatencyLog(unittest.TestCase):

    def test_log_file_is_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            mb = MicroBatchOptimizer(
                fast_path_fn=lambda p: {'success': True},
                slow_path_fn=lambda p: {'success': True},
                rl_tick_fn=None,
                change_threshold=2,
                slow_path_interval_s=9999,
                storage_dir=Path(tmp),
            )
            mb.submit_change('insert', {'patient_id': 'P1', 'is_urgent': True})
            mb.submit_change('cancel', {'id': 'A1'})
            mb.submit_change('cancel', {'id': 'A2'})    # triggers slow
            mb.stop_background_rl()
            lines = [l for l in mb.latency_path.read_text(encoding='utf-8').splitlines() if l]
            self.assertGreaterEqual(len(lines), 3)
            for line in lines:
                row = json.loads(line)
                for key in ('ts', 'path', 'change_type', 'latency_ms', 'success'):
                    self.assertIn(key, row)
                self.assertIn(row['path'], ('fast', 'slow', 'queued'))


class TestConfig(unittest.TestCase):

    def test_update_config_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            mb = MicroBatchOptimizer(storage_dir=Path(tmp))
            cfg = mb.update_config(change_threshold=7, slow_path_interval_s=120,
                                   fast_path_budget_ms=40.0, rl_tick_s=15)
            self.assertEqual(cfg['change_threshold'], 7)
            self.assertEqual(cfg['slow_path_interval_s'], 120)
            self.assertEqual(cfg['fast_path_budget_ms'], 40.0)
            self.assertEqual(cfg['rl_tick_s'], 15)
            with self.assertRaises(ValueError):
                mb.update_config(change_threshold=0)
            with self.assertRaises(ValueError):
                mb.update_config(slow_path_interval_s=0)
            # persisted
            self.assertTrue(mb.config_path.exists())
            mb.stop_background_rl()


class TestBackgroundRL(unittest.TestCase):

    def test_start_stop_is_clean(self):
        ticks = []
        with tempfile.TemporaryDirectory() as tmp:
            mb = MicroBatchOptimizer(
                rl_tick_fn=lambda: (ticks.append(1) or {'ok': True}),
                rl_tick_s=1,   # short so the test is fast
                storage_dir=Path(tmp),
            )
            mb.start_background_rl()
            time.sleep(1.5)
            mb.stop_background_rl(timeout=2.0)
            self.assertGreaterEqual(len(ticks), 1)
            # thread should be stopped
            self.assertFalse(mb._rl_thread.is_alive())

    def test_no_rl_fn_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            mb = MicroBatchOptimizer(rl_tick_fn=None, storage_dir=Path(tmp))
            mb.start_background_rl()         # should log and no-op
            self.assertIsNone(mb._rl_thread)
            mb.stop_background_rl()


class TestStatus(unittest.TestCase):

    def test_status_reflects_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            mb = MicroBatchOptimizer(
                fast_path_fn=lambda p: {'success': True},
                slow_path_fn=lambda p: {'success': True},
                rl_tick_fn=None,
                change_threshold=2, slow_path_interval_s=9999,
                storage_dir=Path(tmp),
            )
            mb.submit_change('insert', {'patient_id': 'P1', 'is_urgent': True})
            mb.submit_change('cancel', {'id': 'A1'})
            mb.submit_change('cancel', {'id': 'A2'})   # fires slow
            s = mb.status()
            self.assertEqual(s.total_fast_path, 1)
            self.assertEqual(s.total_slow_path, 1)
            self.assertEqual(s.total_queued, 2)
            self.assertEqual(s.queue_size, 0)
            self.assertIsNotNone(s.last_full_reopt_ts)
            mb.stop_background_rl()


if __name__ == '__main__':
    unittest.main(verbosity=2)
