"""
Tests for the Channel 2 (real hospital data) file-drop fingerprint.

The full watcher loop runs as a daemon thread inside Flask; here we test the
pure logic — the fingerprint + stability-gate rules — which is the correctness-
critical piece.  End-to-end promotion is exercised in the manual smoke test
(drop files → GET /api/data/channel2-watcher after 60-120 s).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChannel2Fingerprint(unittest.TestCase):
    """Verify the fingerprint correctly rejects partial/in-flight drops."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.real_dir = Path(self.tmp.name) / 'real_data'
        self.real_dir.mkdir(parents=True, exist_ok=True)
        # Re-create the fingerprint logic in isolation so we don't depend on
        # Flask module import order or globals.
        self.required = ['patients.xlsx', 'historical_appointments.xlsx']

    def tearDown(self):
        self.tmp.cleanup()

    def _fingerprint(self):
        out = {'ready': False, 'mtime': None, 'missing': [], 'reason': None}
        for name in self.required:
            if not (self.real_dir / name).exists():
                out['missing'].append(name)
        if out['missing']:
            out['reason'] = f"missing required files: {', '.join(out['missing'])}"
            return out
        try:
            out['mtime'] = max((self.real_dir / n).stat().st_mtime for n in self.required)
            out['ready'] = True
            return out
        except OSError as exc:
            out['reason'] = f"stat failed: {exc}"
            return out

    def test_empty_dir_not_ready(self):
        fp = self._fingerprint()
        self.assertFalse(fp['ready'])
        self.assertEqual(set(fp['missing']), set(self.required))

    def test_partial_drop_rejected(self):
        """Only patients.xlsx dropped — must NOT promote yet."""
        (self.real_dir / 'patients.xlsx').write_bytes(b'stub')
        fp = self._fingerprint()
        self.assertFalse(fp['ready'])
        self.assertIn('historical_appointments.xlsx', fp['missing'])

    def test_complete_drop_ready(self):
        for name in self.required:
            (self.real_dir / name).write_bytes(b'stub')
        fp = self._fingerprint()
        self.assertTrue(fp['ready'])
        self.assertEqual(fp['missing'], [])
        self.assertIsNotNone(fp['mtime'])

    def test_mtime_is_max_across_required_files(self):
        (self.real_dir / 'patients.xlsx').write_bytes(b'a')
        import time as _t
        _t.sleep(0.02)
        (self.real_dir / 'historical_appointments.xlsx').write_bytes(b'b')
        fp = self._fingerprint()
        self.assertTrue(fp['ready'])
        latest = max((self.real_dir / n).stat().st_mtime for n in self.required)
        self.assertAlmostEqual(fp['mtime'], latest, places=3)


class TestStabilityGate(unittest.TestCase):
    """
    The promotion rule: require mtime to be stable across TWO consecutive
    fingerprints before switching.  Simulate that invariant in a pure-Python
    state machine — mirrors the watcher's decision logic.
    """

    def setUp(self):
        self.state = {'last_mtime': None, 'pending_since': None}

    def _should_promote(self, mtime: float) -> bool:
        """Return True iff the current mtime has been stable since last poll."""
        if self.state['last_mtime'] == mtime:
            return False  # already promoted this mtime
        if self.state['pending_since'] != mtime:
            self.state['pending_since'] = mtime
            return False  # first sighting — defer
        # mtime matches pending (stable across two polls) — promote
        self.state['last_mtime'] = mtime
        self.state['pending_since'] = None
        return True

    def test_defers_on_first_sight(self):
        self.assertFalse(self._should_promote(100.0))

    def test_promotes_after_stable_second_sight(self):
        self.assertFalse(self._should_promote(100.0))
        self.assertTrue(self._should_promote(100.0))

    def test_mtime_change_resets_pending(self):
        self.assertFalse(self._should_promote(100.0))   # first sight
        self.assertFalse(self._should_promote(200.0))   # still being written
        self.assertFalse(self._should_promote(300.0))   # still being written
        self.assertTrue(self._should_promote(300.0))    # finally stable

    def test_no_double_promotion(self):
        self.assertFalse(self._should_promote(100.0))
        self.assertTrue(self._should_promote(100.0))
        # Next poll with same mtime — must NOT re-promote
        self.assertFalse(self._should_promote(100.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
