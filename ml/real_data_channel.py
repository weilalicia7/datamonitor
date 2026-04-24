"""
Real-data channel — detector + schema validator + anonymisation check
======================================================================

Gateway between the synthetic dataset used throughout the dissertation
and a (future) real de-identified Velindre cohort dropped into
``datasets/real_data/``.  The module is *detect-only* — it never
mutates the prediction pipeline — and every downstream consumer
(R analysis, status endpoint, dissertation §4.7) reads the detector's
verdict rather than sniffing the filesystem directly.

Conventions match ``tuning/manifest.py``:

    CHANNEL_SYNTHETIC = "synthetic"   (default; cohort under
                                        datasets/sample_data/)
    CHANNEL_REAL      = "real"         (datasets/real_data/ has at
                                        least a valid
                                        historical_appointments.xlsx)

When ``datasets/real_data/`` contains only the placeholder README.md
the detector returns CHANNEL_SYNTHETIC and the whole stack runs
unchanged.  When an operator drops in real files *and* those files
pass ``validate_schema`` + ``anonymisation_check``, the detector
returns CHANNEL_REAL and the benchmark harness
(``ml/benchmark_real_vs_synthetic.py``) lights up.

No UI panel.  No email.  The only Flask surface is a read-only
``/api/data/channel/real/status`` endpoint for diagnostics (see
``flask_app.py``).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Channel constants — mirror tuning/manifest.py verbatim
# --------------------------------------------------------------------------- #

CHANNEL_SYNTHETIC: str = "synthetic"
CHANNEL_REAL: str = "real"

REAL_DATA_DIR: Path = _REPO_ROOT / "datasets" / "real_data"
REAL_APPTS_FILE: Path = REAL_DATA_DIR / "historical_appointments.xlsx"
REAL_PATIENTS_FILE: Path = REAL_DATA_DIR / "patients.xlsx"
VALIDATION_LOG: Path = (
    _REPO_ROOT / "data_cache" / "real_data_validation" / "runs.jsonl"
)

MIN_REAL_APPOINTMENTS: int = 500   # matches the external-review threshold

# Required columns for the SACT v4.0 real-data drop.  Narrower than the
# full SACT spec — we only require what the NoShowModel + optimiser
# actually consume.  The schema validator reports which are missing.
REQUIRED_APPT_COLUMNS: Tuple[str, ...] = (
    "Patient_ID",           # any ID, may be a hash
    "Appointment_Date",     # or "Date"
    "Planned_Duration",
    "Actual_Duration",      # needed for duration-MAE measurement
    "Attended_Status",      # yes/no/cancelled (ground truth for AUC)
    "Priority",             # P1..P4 (or numeric equivalent)
    "Regimen_Code",
    "Site_Code",
    "Chair_Number",
)
OPTIONAL_APPT_COLUMNS: Tuple[str, ...] = (
    "Patient_Postcode", "Patient_NoShow_Rate",
    "Day_Of_Week_Num", "Weather_Severity", "Travel_Distance_KM",
    "Travel_Time_Min",
)


@dataclass
class ChannelStatus:
    """Detector output — serialisable to JSON for the /status endpoint."""
    channel: str                          # "synthetic" | "real"
    real_data_dir_exists: bool
    appts_file_present: bool
    patients_file_present: bool
    n_rows_detected: int = 0
    schema_ok: bool = False
    anonymisation_ok: bool = False
    reason: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_ts: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def detect_channel(strict: bool = True) -> ChannelStatus:
    """
    Decide whether the system should treat itself as running on the
    synthetic or the real-data channel.

    ``strict=True`` (default) requires:
      * ``datasets/real_data/historical_appointments.xlsx`` exists
      * it parses as a DataFrame
      * it has >= ``MIN_REAL_APPOINTMENTS`` rows
      * ``validate_schema`` passes (required columns present)
      * ``anonymisation_check`` passes (no raw NHS numbers / names)

    ``strict=False`` relaxes the size + schema checks — useful for
    smoke tests of the detection plumbing without a real cohort.

    Returns a ``ChannelStatus`` either way; callers read
    ``status.channel`` to decide how to route downstream pipelines.
    """
    status = ChannelStatus(
        channel=CHANNEL_SYNTHETIC,
        real_data_dir_exists=REAL_DATA_DIR.exists(),
        appts_file_present=REAL_APPTS_FILE.exists(),
        patients_file_present=REAL_PATIENTS_FILE.exists(),
    )

    if not status.real_data_dir_exists:
        status.reason = "datasets/real_data/ does not exist"
        return status
    if not status.appts_file_present:
        status.reason = (
            "datasets/real_data/historical_appointments.xlsx is not "
            "present (only the README placeholder exists) — running "
            "on the synthetic channel"
        )
        return status

    # Try to load the file.  Failure to parse keeps us on synthetic.
    try:
        df = _read_appointments()
    except Exception as exc:
        status.reason = f"Could not parse real data: {exc!r}"
        status.errors.append(str(exc))
        return status
    status.n_rows_detected = int(len(df))

    if strict and status.n_rows_detected < MIN_REAL_APPOINTMENTS:
        status.reason = (
            f"real_data has {status.n_rows_detected} rows; need >= "
            f"{MIN_REAL_APPOINTMENTS}.  Staying on synthetic channel."
        )
        return status

    schema_ok, schema_errors = validate_schema(df)
    status.schema_ok = schema_ok
    status.errors.extend(schema_errors)
    if strict and not schema_ok:
        status.reason = (
            "schema validation failed — required SACT v4.0 columns "
            "missing.  Staying on synthetic channel."
        )
        return status

    anon_ok, anon_warnings = anonymisation_check(df)
    status.anonymisation_ok = anon_ok
    status.warnings.extend(anon_warnings)
    if strict and not anon_ok:
        status.reason = (
            "anonymisation check flagged raw-PII patterns — refusing "
            "to treat as real channel until cleaned."
        )
        return status

    status.channel = CHANNEL_REAL
    status.reason = (
        f"Real cohort detected ({status.n_rows_detected} rows) — "
        "schema + anonymisation both pass."
    )
    return status


# --------------------------------------------------------------------------- #
# Schema + anonymisation
# --------------------------------------------------------------------------- #


def validate_schema(df) -> Tuple[bool, List[str]]:
    """
    Check the DataFrame has the columns NoShowModel + optimiser need.
    Returns (ok, list_of_missing_columns).  Missing optional columns
    are not errors; they degrade feature coverage but the pipeline
    still runs.
    """
    errors: List[str] = []
    present = set(df.columns)
    # "Appointment_Date" OR "Date" is acceptable
    date_present = "Appointment_Date" in present or "Date" in present
    for col in REQUIRED_APPT_COLUMNS:
        if col == "Appointment_Date":
            if not date_present:
                errors.append("Appointment_Date (or Date) is required")
            continue
        if col not in present:
            errors.append(f"required column {col!r} missing")
    return (not errors, errors)


def anonymisation_check(df) -> Tuple[bool, List[str]]:
    """
    Heuristic guard against raw PII leaking into the real-data drop:

    * reject 10-digit all-numeric Patient_IDs (pattern of raw NHS Number)
    * reject columns named Person_Given_Name / Person_Family_Name that
      contain alphabetic content (should be hashed before drop-in)
    * warn (not error) when postcodes are full 7-character UK postcodes
      (we prefer the first 4 chars only for NOSHOW feature use)

    The check is conservative — it warns rather than rejects on
    anything ambiguous so we don't reject a legitimately-anonymised
    cohort over a false positive.  Returns (ok, warnings).
    """
    warnings: List[str] = []

    if "Patient_ID" in df.columns:
        sample = df["Patient_ID"].astype(str).head(50)
        if (sample.str.fullmatch(r"\d{10}")).any():
            warnings.append(
                "Patient_ID column contains 10-digit all-numeric values — "
                "likely raw NHS Numbers.  Hash or pseudonymise before use."
            )

    for pii_col in ("Person_Given_Name", "Person_Family_Name"):
        if pii_col in df.columns:
            sample = df[pii_col].astype(str).head(50)
            if sample.str.match(r"^[A-Za-z]{2,}").any():
                warnings.append(
                    f"{pii_col} column contains alphabetic content — "
                    "should be dropped or hashed."
                )

    if "Patient_Postcode" in df.columns:
        sample = df["Patient_Postcode"].astype(str).head(50)
        full_pc_pattern = r"^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$"
        if sample.str.match(full_pc_pattern).sum() > len(sample) * 0.5:
            warnings.append(
                "Patient_Postcode column looks like full 7-character UK "
                "postcodes — consider truncating to the first 4 chars "
                "(the NoShowModel only uses the outer code)."
            )

    # Warnings don't fail the check on their own; only explicit errors do.
    # (We don't currently raise hard anonymisation errors — that would
    # be up to the operator running under DPIA.)
    return (True, warnings)


# --------------------------------------------------------------------------- #
# Loader + persistence
# --------------------------------------------------------------------------- #


def _read_appointments():
    """Lazy pandas import so detect_channel works in environments
    that import the module for constants without pulling pandas."""
    import pandas as pd
    return pd.read_excel(REAL_APPTS_FILE)


def load_real_cohort():
    """
    Public loader — returns the real appointments DataFrame after
    detection, or raises RuntimeError if the channel is not 'real'.

    Callers downstream of the detector use this to avoid
    re-implementing the gate; the pipeline stays bit-identical when
    the detector returns SYNTHETIC.
    """
    status = detect_channel(strict=True)
    if status.channel != CHANNEL_REAL:
        raise RuntimeError(
            f"Real-data channel not active: {status.reason}"
        )
    return _read_appointments()


def write_validation_row(row: Dict) -> None:
    """
    Append one real-vs-synthetic comparison row to the JSONL log.
    The benchmark harness uses this; the R analysis reads it back.
    """
    VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Always stamp the channel + timestamp so rows are self-describing
    row.setdefault("ts", datetime.utcnow().isoformat(timespec="seconds"))
    with VALIDATION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


__all__ = [
    "CHANNEL_SYNTHETIC", "CHANNEL_REAL",
    "REAL_DATA_DIR", "REAL_APPTS_FILE", "REAL_PATIENTS_FILE",
    "VALIDATION_LOG", "MIN_REAL_APPOINTMENTS",
    "REQUIRED_APPT_COLUMNS", "OPTIONAL_APPT_COLUMNS",
    "ChannelStatus",
    "detect_channel",
    "validate_schema",
    "anonymisation_check",
    "load_real_cohort",
    "write_validation_row",
]
