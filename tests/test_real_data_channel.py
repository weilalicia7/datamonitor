"""
Tests for ml/real_data_channel.py (Dissertation §4.7 regression).

Lock four invariants so the real-data channel claim cannot drift:

  1. Dormant state: with only the README.md placeholder in
     datasets/real_data/, detect_channel() returns "synthetic" and
     the reason string explicitly names the missing file.
  2. Schema validator rejects a frame missing a required SACT v4.0
     column; passes on a frame containing all required columns.
  3. Anonymisation check flags 10-digit all-numeric Patient_IDs
     (NHS-Number pattern) as a warning.
  4. End-to-end: when a synthesised "real-shaped" file is placed in
     datasets/real_data/, detect_channel(strict=False) returns
     "real" and load_real_cohort() returns the DataFrame.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ml import real_data_channel as rdc


class TestRealDataChannel(unittest.TestCase):

    def test_dormant_state_returns_synthetic(self):
        """In the repo's committed state, datasets/real_data/ contains
        only README.md; detect_channel() must stay on synthetic and
        explain why in the reason string."""
        status = rdc.detect_channel(strict=True)
        self.assertEqual(status.channel, rdc.CHANNEL_SYNTHETIC)
        self.assertFalse(status.appts_file_present)
        self.assertIn(
            "historical_appointments.xlsx",
            status.reason,
            f"reason should name the missing file, got {status.reason!r}"
        )

    def test_schema_validator_rejects_missing_required_column(self):
        df = pd.DataFrame({
            "Patient_ID": ["P1", "P2"],
            "Planned_Duration": [60, 90],
            # Missing many required columns
        })
        ok, errors = rdc.validate_schema(df)
        self.assertFalse(ok)
        self.assertGreater(len(errors), 0)
        # At least one error must name Appointment_Date
        self.assertTrue(
            any("Appointment_Date" in e or "Date" in e for e in errors),
            f"expected a date-column error, got {errors}"
        )

    def test_schema_validator_accepts_full_frame(self):
        df = pd.DataFrame({
            "Patient_ID": ["H1", "H2"],
            "Appointment_Date": ["2026-04-22", "2026-04-23"],
            "Planned_Duration": [60, 90],
            "Actual_Duration": [65, 95],
            "Attended_Status": ["Yes", "No"],
            "Priority": ["P2", "P3"],
            "Regimen_Code": ["R-CHOP", "FOLFOX"],
            "Site_Code": ["WC", "WC"],
            "Chair_Number": [1, 2],
        })
        ok, errors = rdc.validate_schema(df)
        self.assertTrue(ok, f"schema should accept this frame; errors={errors}")

    def test_anonymisation_check_flags_raw_nhs_numbers(self):
        df = pd.DataFrame({
            "Patient_ID": ["4567890123", "4567890124", "4567890125"],
        })
        _ok, warnings = rdc.anonymisation_check(df)
        self.assertTrue(
            any("10-digit" in w or "NHS Number" in w for w in warnings),
            f"expected an NHS-Number warning, got {warnings}"
        )

    def test_anonymisation_check_passes_hashed_ids(self):
        df = pd.DataFrame({
            "Patient_ID": ["P0001", "P0002", "P0003"],
        })
        _ok, warnings = rdc.anonymisation_check(df)
        self.assertFalse(
            any("NHS Number" in w or "10-digit" in w for w in warnings),
            f"hashed IDs should not trigger an NHS-Number warning: {warnings}"
        )

    def test_end_to_end_activation_on_synthesised_real_frame(self):
        """
        Materialise a minimal 'real-shaped' file into
        datasets/real_data/, confirm the detector flips to 'real',
        then clean up so the rest of the test suite (and the
        dissertation build) stays in the dormant state.
        """
        real_dir = rdc.REAL_DATA_DIR
        target = rdc.REAL_APPTS_FILE
        created_dir = not real_dir.exists()
        real_dir.mkdir(parents=True, exist_ok=True)

        # Synthesise a 50-row frame that passes the schema but NOT
        # the strict 500-row minimum — use detect_channel(strict=False)
        # for the flip-verification so we don't need to actually
        # materialise 500 rows.
        df = pd.DataFrame({
            "Patient_ID": [f"H{i:04d}" for i in range(50)],
            "Appointment_Date": ["2026-04-22"] * 50,
            "Planned_Duration": [90] * 50,
            "Actual_Duration": [95] * 50,
            "Attended_Status": ["Yes"] * 40 + ["No"] * 10,
            "Priority": ["P2"] * 50,
            "Regimen_Code": ["R-CHOP"] * 50,
            "Site_Code": ["WC"] * 50,
            "Chair_Number": list(range(50)),
        })

        had_existing = target.exists()
        backup_path = None
        if had_existing:
            backup_path = target.with_suffix(".xlsx.bak")
            shutil.move(str(target), str(backup_path))
        try:
            df.to_excel(target, index=False)
            # strict=False so the 500-row minimum doesn't block the flip
            status = rdc.detect_channel(strict=False)
            self.assertEqual(
                status.channel, rdc.CHANNEL_REAL,
                f"expected real-channel activation, got {status.channel} "
                f"({status.reason})"
            )
            self.assertTrue(status.appts_file_present)
            self.assertEqual(status.n_rows_detected, 50)
        finally:
            # Remove the synthesised file, restore backup if any, and
            # only remove the dir if we created it.
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            if backup_path is not None and backup_path.exists():
                shutil.move(str(backup_path), str(target))
            if created_dir:
                try:
                    real_dir.rmdir()
                except OSError:
                    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
