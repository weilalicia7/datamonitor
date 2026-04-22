"""
SACT Version Adapter Layer (Dissertation §5.4)
==============================================

The NHS England Systemic Anti-Cancer Therapy (SACT) dataset is
versioned.  The current live schema is v4.0 (May 2025) with 82
fields; NHS Digital has signalled v4.1 for ~2028 with additional
molecular-marker / biomarker-panel fields and a gender-code
rename.  Our production pipeline must not break the day that
arrives.  §5.4 asks us to design the adapter layer now:

    class SACTVersionAdapter(ABC):
        @abstractmethod
        def to_internal(self, raw_df) -> pd.DataFrame: ...

    class SACTv4Adapter(SACTVersionAdapter): ...
    class SACTv4_1Adapter(SACTVersionAdapter): ...  # placeholder

This module implements that contract:

* :class:`SACTVersionAdapter` — abstract base.
* :class:`SACTv4Adapter` — maps the authoritative v4.0 schema into
  the internal canonical schema the rest of this project uses
  (``Patient_ID``, ``Date``, ``Age_Band``, …).  This is the
  adapter currently active in production.
* :class:`SACTv4_1Adapter` — placeholder with the best publicly
  available guess at v4.1 field deltas, derived from NHS Digital's
  DAPB / ISB consultations.  All new-in-v4.1 fields are carried
  forward to the canonical schema under neutral names so downstream
  ML does not have to change.
* ``ADAPTERS`` — version-string registry consumed by
  :func:`adapt` + :func:`auto_detect_version`.  Adding a new
  adapter is a one-line registration.
* Every invocation writes one row to
  ``data_cache/sact_adapter/adapter_events.jsonl`` --- the §33
  dissertation R script reads it for schema-coverage analysis.

Design contract
---------------
* **Canonical schema is the fixed point**.  Downstream ML is
  wired to the canonical columns; new SACT versions only add
  mappings.
* **Forward-compatibility is additive**, not destructive.  A v4.1
  adapter may populate *new* canonical columns (e.g.
  ``Molecular_Marker_Status``) but never removes the v4.0 ones.
* **Auto-detection is safe**.  ``auto_detect_version`` picks the
  *highest* version whose signature fields all appear in the
  incoming DataFrame; an unknown schema falls back to ``v4.0``
  and a WARNING is logged.
* **Every invocation is audited**.  Logs include field-coverage,
  version picked, and the rows-in / rows-out counts.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CACHE_DIR, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults / storage
# ---------------------------------------------------------------------------
SACT_DIR: Path = DATA_CACHE_DIR / "sact_adapter"
SACT_LOG: Path = SACT_DIR / "adapter_events.jsonl"


# ---------------------------------------------------------------------------
# Canonical internal schema — the fixed point every adapter targets
# ---------------------------------------------------------------------------
#
# Adding a new canonical column is a breaking change for downstream ML;
# do it only when a v5.x schema arrives and the project has been
# notified.  Canonical columns should be stable, plain-Python-typed, and
# self-describing.
CANONICAL_COLUMNS: Tuple[str, ...] = (
    # Identity
    "Patient_ID",
    "NHS_Number_Hashed",
    # Event
    "Date",
    "Appointment_Hour",
    "Attended_Status",
    "Regimen_Code",
    "Cycle_Number",
    # Demographics
    "Age",
    "Age_Band",
    "Gender_Code",
    "Ethnic_Category",
    "Patient_Postcode",
    # Clinical context
    "Primary_Diagnosis_Code",
    "Performance_Status",
    # Resourcing
    "Planned_Duration",
    "Actual_Duration",
    "Site_Code",
    "Chair_Id",
    # External
    "Weather_Severity",
    # v4.1 forward-compatible columns (may be empty on v4.0 input)
    "Molecular_Marker_Status",
    "Biomarker_Panel_Code",
    "Subcutaneous_Administration_Flag",
    "Comorbidity_Count",
    "Commissioning_Organisation_Code",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdapterEvent:
    ts: str
    version: str
    rows_in: int
    rows_out: int
    columns_seen: List[str]
    columns_mapped: List[str]
    columns_missing: List[str]
    columns_new_in_version: List[str]
    auto_detected: bool


@dataclass
class VersionSpec:
    """Human-readable description of one SACT version."""
    version: str
    published: str                        # "May 2025", "~2028", etc.
    expected_fields: Tuple[str, ...]      # fields whose presence identifies the version
    new_since_prior: Tuple[str, ...]      # fields new vs the previous version
    renamed_from_prior: Tuple[Tuple[str, str], ...]  # [(old, new), ...]
    canonical_map: Dict[str, str]         # raw_name -> canonical_name


# ---------------------------------------------------------------------------
# Version specifications
# ---------------------------------------------------------------------------


SPEC_V4_0 = VersionSpec(
    version="v4.0",
    published="May 2025",
    expected_fields=(
        "Patient_ID", "Regimen_Code", "Cycle_Number",
        "Person_Stated_Gender_Code", "Ethnic_Category_Code",
    ),
    new_since_prior=(),
    renamed_from_prior=(),
    canonical_map={
        # The production system already exports canonical-style names
        # when it writes historical_appointments.xlsx / patients.xlsx,
        # so most mappings are identity.  We only rewrite the few
        # fields NHS Digital uses differently.
        "Patient_ID": "Patient_ID",
        "NHS_Number_Hashed": "NHS_Number_Hashed",
        "Date": "Date",
        "Appointment_Date": "Date",
        "Appointment_Hour": "Appointment_Hour",
        "Attended_Status": "Attended_Status",
        "Regimen_Code": "Regimen_Code",
        "Cycle_Number": "Cycle_Number",
        "Age": "Age",
        "Age_Band": "Age_Band",
        "Person_Stated_Gender_Code": "Gender_Code",
        "Gender_Code": "Gender_Code",
        "Ethnic_Category_Code": "Ethnic_Category",
        "Ethnic_Category": "Ethnic_Category",
        "Patient_Postcode": "Patient_Postcode",
        "Primary_Diagnosis_Code": "Primary_Diagnosis_Code",
        "ICD10_Code": "Primary_Diagnosis_Code",
        "Performance_Status": "Performance_Status",
        "WHO_Status": "Performance_Status",
        "Planned_Duration": "Planned_Duration",
        "Actual_Duration": "Actual_Duration",
        "Duration_Actual": "Actual_Duration",
        "Site_Code": "Site_Code",
        "Provider_Code": "Site_Code",
        "Chair_Id": "Chair_Id",
        "Chair_ID": "Chair_Id",
        "Weather_Severity": "Weather_Severity",
    },
)

SPEC_V4_1 = VersionSpec(
    version="v4.1",
    published="~2028 (projected)",
    expected_fields=(
        # v4.1 signature fields — presence of any of these strongly hints
        # we're on v4.1.  In the DAPB consultation drafts these are the
        # additions with the highest likelihood of landing first.
        "Molecular_Marker_Status",
        "Biomarker_Panel_Code",
        "Gender_Identity_Code",              # replaces Person_Stated_Gender_Code
        "Commissioning_Organisation_Code",   # replaces Provider_Code
    ),
    new_since_prior=(
        "Molecular_Marker_Status",
        "Biomarker_Panel_Code",
        "Subcutaneous_Administration_Flag",
        "Comorbidity_Count",
        "Commissioning_Organisation_Code",
    ),
    renamed_from_prior=(
        ("Person_Stated_Gender_Code", "Gender_Identity_Code"),
        ("Ethnic_Category_Code", "Ethnic_Category"),
        ("Provider_Code", "Commissioning_Organisation_Code"),
    ),
    canonical_map={
        # Carry the v4.0 mappings forward
        **SPEC_V4_0.canonical_map,
        # v4.1-specific
        "Gender_Identity_Code": "Gender_Code",
        "Commissioning_Organisation_Code": "Commissioning_Organisation_Code",
        "Provider_Code": "Commissioning_Organisation_Code",
        "Molecular_Marker_Status": "Molecular_Marker_Status",
        "Biomarker_Panel_Code": "Biomarker_Panel_Code",
        "Subcutaneous_Administration_Flag": "Subcutaneous_Administration_Flag",
        "Comorbidity_Count": "Comorbidity_Count",
    },
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SACTVersionAdapter(ABC):
    """Map one SACT raw schema into the internal canonical schema."""

    #: Must be overridden by subclasses to populate the registry.
    VERSION: str = ""

    #: Published reference string for the spec.
    PUBLISHED: str = ""

    @abstractmethod
    def to_internal(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with columns drawn from
        :data:`CANONICAL_COLUMNS`.  Unknown-to-this-version canonical
        columns must be filled with sensible defaults (NaN for numeric,
        empty string for categorical) so downstream ML can always
        assume the full schema.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Default helpers shared by concrete adapters
    # ------------------------------------------------------------------ #

    def _apply_canonical_map(
        self, raw_df: pd.DataFrame, canonical_map: Dict[str, str]
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=raw_df.index)
        # Project each canonical column from whatever raw column maps to it.
        # Last-seen wins when two raw columns map to the same canonical.
        for raw_col, canonical_col in canonical_map.items():
            if raw_col in raw_df.columns:
                out[canonical_col] = raw_df[raw_col].values
        # Fill in canonical columns that weren't mapped at all
        for col in CANONICAL_COLUMNS:
            if col not in out.columns:
                out[col] = self._default_value(col)
        return out[list(CANONICAL_COLUMNS)]

    @staticmethod
    def _default_value(col: str) -> Any:
        """Sensible empty value for a canonical column."""
        if col in {"Age", "Cycle_Number", "Appointment_Hour",
                   "Planned_Duration", "Actual_Duration",
                   "Comorbidity_Count"}:
            return pd.NA
        if col in {"Weather_Severity"}:
            return 0.0
        return ""


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class SACTv4Adapter(SACTVersionAdapter):
    """Adapter for the current NHS SACT v4.0 schema (82 fields)."""

    VERSION = "v4.0"
    PUBLISHED = SPEC_V4_0.published

    def to_internal(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df is None or len(raw_df) == 0:
            return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
        return self._apply_canonical_map(raw_df, SPEC_V4_0.canonical_map)


class SACTv4_1Adapter(SACTVersionAdapter):
    """
    Adapter for the projected NHS SACT v4.1 schema (~2028).

    This is a \\emph{placeholder} per the \\S5.4 brief: it captures the
    mapping deltas that NHS Digital has signalled in DAPB / ISB
    consultations (molecular markers, biomarker panels, subcutaneous
    administration flag, comorbidity count, gender identity rename,
    commissioner-code rename) so the production pipeline can drop in
    immediately once the real schema is released.  Field semantics will
    be finalised on publication; the adapter's \\emph{structure} --- a
    mapping from raw columns to canonical columns --- is stable.
    """

    VERSION = "v4.1"
    PUBLISHED = SPEC_V4_1.published

    def to_internal(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df is None or len(raw_df) == 0:
            return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
        out = self._apply_canonical_map(raw_df, SPEC_V4_1.canonical_map)
        # Sanity normalisation for SC-admin flag: coerce truthy strings
        if "Subcutaneous_Administration_Flag" in out.columns:
            out["Subcutaneous_Administration_Flag"] = (
                out["Subcutaneous_Administration_Flag"]
                .apply(_coerce_bool_flag)
            )
        return out


# ---------------------------------------------------------------------------
# Registry + auto-detection
# ---------------------------------------------------------------------------


ADAPTERS: Dict[str, SACTVersionAdapter] = {
    "v4.0": SACTv4Adapter(),
    "v4.1": SACTv4_1Adapter(),
}
SPECS: Dict[str, VersionSpec] = {
    "v4.0": SPEC_V4_0,
    "v4.1": SPEC_V4_1,
}


def auto_detect_version(raw_df: pd.DataFrame) -> str:
    """Pick the highest version whose signature fields all appear in ``raw_df``.

    Falls back to ``v4.0`` (the current production schema) and logs a
    WARNING if no version matches strictly.
    """
    if raw_df is None or len(raw_df) == 0:
        return "v4.0"
    cols = set(raw_df.columns)
    # Prefer the latest version whose expected-fields subset is fully present.
    for v in sorted(SPECS.keys(), reverse=True):
        expected = set(SPECS[v].expected_fields)
        if expected and expected.issubset(cols):
            return v
    # No strict match — log a soft warning and default to v4.0.
    logger.warning(
        "SACT auto-detect: no registered version has all expected fields "
        "in the input DataFrame; falling back to v4.0.  Present columns "
        "(first 20): %s",
        sorted(cols)[:20],
    )
    return "v4.0"


def adapt(
    raw_df: pd.DataFrame,
    version: Optional[str] = None,
    force_auto_detect: bool = False,
) -> Tuple[pd.DataFrame, AdapterEvent]:
    """Convert ``raw_df`` to the canonical schema.

    Args
    ----
    raw_df:
        The raw SACT-formatted DataFrame (e.g. the output of
        ``pandas.read_excel`` on ``historical_appointments.xlsx``).
    version:
        Explicit version label (e.g. ``"v4.0"``, ``"v4.1"``).  When
        provided, overrides auto-detect.
    force_auto_detect:
        If True, ignore ``version`` and always call
        :func:`auto_detect_version`.
    """
    auto = version is None or force_auto_detect
    v = auto_detect_version(raw_df) if auto else str(version)
    if v not in ADAPTERS:
        logger.warning(
            "SACT version '%s' has no adapter registered; defaulting to v4.0", v
        )
        v = "v4.0"
    adapter = ADAPTERS[v]
    out = adapter.to_internal(raw_df)

    cols_seen = list(raw_df.columns) if raw_df is not None else []
    cols_mapped = [c for c in cols_seen if c in SPECS[v].canonical_map]
    cols_missing = [
        c for c in SPECS[v].expected_fields
        if c not in (raw_df.columns if raw_df is not None else [])
    ]
    event = AdapterEvent(
        ts=datetime.utcnow().isoformat(timespec="seconds"),
        version=v,
        rows_in=int(len(raw_df)) if raw_df is not None else 0,
        rows_out=int(len(out)),
        columns_seen=cols_seen,
        columns_mapped=cols_mapped,
        columns_missing=cols_missing,
        columns_new_in_version=list(SPECS[v].new_since_prior),
        auto_detected=bool(auto),
    )
    _append_event(event)
    return out, event


# ---------------------------------------------------------------------------
# Status + diagnostics wrapper
# ---------------------------------------------------------------------------


class SACTAdapterPipeline:
    """Flask-friendly wrapper bundling detect, adapt, and status."""

    def __init__(self, storage_dir: Path = SACT_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.storage_dir / "adapter_events.jsonl"
        self._lock = threading.Lock()
        self._last_event: Optional[AdapterEvent] = None
        self._n_runs: int = 0

    def detect(self, raw_df: pd.DataFrame) -> str:
        return auto_detect_version(raw_df)

    def adapt(
        self,
        raw_df: pd.DataFrame,
        version: Optional[str] = None,
        force_auto_detect: bool = False,
    ) -> Tuple[pd.DataFrame, AdapterEvent]:
        out, event = adapt(
            raw_df, version=version, force_auto_detect=force_auto_detect,
        )
        # Also persist to the pipeline's configured log so test fixtures
        # with custom storage_dir see the write.  Safe to duplicate —
        # module-level SACT_LOG is the global default; per-pipeline
        # logs let us sandbox tests and per-tenant deployments.
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": event.ts,
                "version": event.version,
                "rows_in": event.rows_in,
                "rows_out": event.rows_out,
                "columns_seen_count": len(event.columns_seen),
                "columns_mapped_count": len(event.columns_mapped),
                "columns_missing": event.columns_missing,
                "columns_new_in_version": event.columns_new_in_version,
                "auto_detected": event.auto_detected,
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"SACT pipeline log write failed: {exc}")
        with self._lock:
            self._last_event = event
            self._n_runs += 1
        return out, event

    def status(self) -> Dict[str, Any]:
        last = self._last_event
        return {
            "registered_versions": sorted(ADAPTERS.keys()),
            "canonical_columns": list(CANONICAL_COLUMNS),
            "total_runs": self._n_runs,
            "log_path": str(self.log_path),
            "last_version": last.version if last else None,
            "last_rows_in": last.rows_in if last else None,
            "last_rows_out": last.rows_out if last else None,
            "last_columns_mapped": len(last.columns_mapped) if last else 0,
            "last_auto_detected": last.auto_detected if last else None,
            "last_ts": last.ts if last else None,
        }

    def version_info(self) -> List[Dict[str, Any]]:
        """Human-readable diff between registered versions."""
        return [
            {
                "version": spec.version,
                "published": spec.published,
                "expected_fields": list(spec.expected_fields),
                "new_since_prior": list(spec.new_since_prior),
                "renamed_from_prior": [
                    {"from": a, "to": b}
                    for a, b in spec.renamed_from_prior
                ],
                "canonical_field_count": len(set(spec.canonical_map.values())),
            }
            for _, spec in sorted(SPECS.items())
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_event(event: AdapterEvent) -> None:
    try:
        SACT_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": event.ts,
            "version": event.version,
            "rows_in": event.rows_in,
            "rows_out": event.rows_out,
            "columns_seen_count": len(event.columns_seen),
            "columns_mapped_count": len(event.columns_mapped),
            "columns_missing": event.columns_missing,
            "columns_new_in_version": event.columns_new_in_version,
            "auto_detected": event.auto_detected,
        }
        with open(SACT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"SACT adapter log write failed: {exc}")


def _coerce_bool_flag(v: Any) -> Any:
    """Normalise Y/N, True/False, 1/0 → Python bool; pass NaN through."""
    if v is None:
        return pd.NA
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v != v:   # NaN
            return pd.NA
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return pd.NA
    if s in {"y", "yes", "true", "1"}:
        return True
    if s in {"n", "no", "false", "0"}:
        return False
    return pd.NA


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors §3.1–§5.3 style)
# ---------------------------------------------------------------------------


_GLOBAL: Optional[SACTAdapterPipeline] = None


def get_pipeline() -> SACTAdapterPipeline:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = SACTAdapterPipeline()
    return _GLOBAL


def set_pipeline(p: SACTAdapterPipeline) -> None:
    global _GLOBAL
    _GLOBAL = p


def register_adapter(version: str, adapter: SACTVersionAdapter,
                     spec: VersionSpec) -> None:
    """Register a third-party adapter at runtime (e.g. for v5.0)."""
    ADAPTERS[version] = adapter
    SPECS[version] = spec
    logger.info(f"SACT adapter registered: {version} ({spec.published})")
