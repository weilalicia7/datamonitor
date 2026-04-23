"""
Tests for ml/sact_version_adapter.py (Dissertation §5.4)
========================================================

Verifies the SACT version-adapter layer: ABC contract, v4.0
passthrough, v4.1 placeholder mapping, auto-detection, registry,
idempotent roundtrip, persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.sact_version_adapter import (
    ADAPTERS,
    CANONICAL_COLUMNS,
    SPECS,
    SPEC_V4_0,
    SPEC_V4_1,
    AdapterEvent,
    SACTAdapterPipeline,
    SACTv4Adapter,
    SACTv4_1Adapter,
    SACTVersionAdapter,
    VersionSpec,
    _coerce_bool_flag,
    adapt,
    auto_detect_version,
    get_pipeline,
    register_adapter,
    set_pipeline,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def df_v40():
    return pd.DataFrame({
        "Patient_ID": ["P001", "P002", "P003"],
        "Regimen_Code": ["XLFOX", "DOCE", "GEM"],
        "Cycle_Number": [1, 2, 3],
        "Person_Stated_Gender_Code": ["M", "F", "F"],
        "Ethnic_Category_Code": ["A", "B", "A"],
        "Date": pd.to_datetime(["2026-04-22", "2026-04-22", "2026-04-23"]),
        "Appointment_Hour": [9, 11, 14],
        "Provider_Code": ["7A2", "7A2", "7A1"],
        "Weather_Severity": [1.0, 2.0, 0.5],
        "Age_Band": ["<40", "40-60", ">75"],
    })


@pytest.fixture
def df_v41(df_v40):
    df = df_v40.copy()
    df["Molecular_Marker_Status"] = ["BRCA1+", "TP53-", "BRAF-"]
    df["Biomarker_Panel_Code"] = ["P001", "P002", "P001"]
    df["Subcutaneous_Administration_Flag"] = ["Y", "N", "Y"]
    df["Comorbidity_Count"] = [1, 3, 0]
    # Rename renamed-fields in-place
    df = df.rename(columns={
        "Person_Stated_Gender_Code": "Gender_Identity_Code",
        "Ethnic_Category_Code": "Ethnic_Category",
        "Provider_Code": "Commissioning_Organisation_Code",
    })
    return df


@pytest.fixture
def tmp_pipeline(tmp_path):
    return SACTAdapterPipeline(storage_dir=tmp_path / "sact")


# --------------------------------------------------------------------------- #
# 1. ABC
# --------------------------------------------------------------------------- #


class TestABC:
    def test_abc_not_instantiable(self):
        with pytest.raises(TypeError):
            SACTVersionAdapter()

    def test_v4_adapter_is_concrete(self):
        SACTv4Adapter()
        SACTv4_1Adapter()


# --------------------------------------------------------------------------- #
# 2. v4.0 passthrough
# --------------------------------------------------------------------------- #


class TestV40Adapter:
    def test_v40_output_matches_canonical_schema(self, df_v40):
        adapter = SACTv4Adapter()
        out = adapter.to_internal(df_v40)
        assert list(out.columns) == list(CANONICAL_COLUMNS)
        assert len(out) == len(df_v40)

    def test_v40_maps_gender_correctly(self, df_v40):
        out = SACTv4Adapter().to_internal(df_v40)
        assert list(out["Gender_Code"]) == ["M", "F", "F"]

    def test_v40_maps_provider_to_site(self, df_v40):
        out = SACTv4Adapter().to_internal(df_v40)
        assert list(out["Site_Code"]) == ["7A2", "7A2", "7A1"]

    def test_v40_new_v41_fields_default_empty(self, df_v40):
        out = SACTv4Adapter().to_internal(df_v40)
        # Molecular_Marker_Status etc. are still in the canonical schema,
        # just empty (string-typed default)
        assert all(out["Molecular_Marker_Status"] == "")
        assert all(out["Biomarker_Panel_Code"] == "")

    def test_v40_empty_input_returns_empty_output(self):
        out = SACTv4Adapter().to_internal(pd.DataFrame())
        assert list(out.columns) == list(CANONICAL_COLUMNS)
        assert len(out) == 0


# --------------------------------------------------------------------------- #
# 3. v4.1 placeholder
# --------------------------------------------------------------------------- #


class TestV41Adapter:
    def test_v41_maps_new_fields(self, df_v41):
        out = SACTv4_1Adapter().to_internal(df_v41)
        assert list(out["Molecular_Marker_Status"]) == ["BRCA1+", "TP53-", "BRAF-"]
        assert list(out["Biomarker_Panel_Code"]) == ["P001", "P002", "P001"]
        assert list(out["Comorbidity_Count"]) == [1, 3, 0]

    def test_v41_gender_identity_renamed(self, df_v41):
        out = SACTv4_1Adapter().to_internal(df_v41)
        assert list(out["Gender_Code"]) == ["M", "F", "F"]

    def test_v41_commissioning_org_renamed(self, df_v41):
        out = SACTv4_1Adapter().to_internal(df_v41)
        # New canonical column for v4.1
        assert list(out["Commissioning_Organisation_Code"]) == ["7A2", "7A2", "7A1"]

    def test_v41_sc_flag_coerced(self, df_v41):
        out = SACTv4_1Adapter().to_internal(df_v41)
        assert list(out["Subcutaneous_Administration_Flag"]) == [True, False, True]


# --------------------------------------------------------------------------- #
# 4. Auto-detection
# --------------------------------------------------------------------------- #


class TestAutoDetect:
    def test_detect_v40(self, df_v40):
        assert auto_detect_version(df_v40) == "v4.0"

    def test_detect_v41(self, df_v41):
        assert auto_detect_version(df_v41) == "v4.1"

    def test_prefers_higher_version(self):
        """When signature fields of multiple versions are present,
        auto-detect picks the highest version."""
        df = pd.DataFrame({
            "Patient_ID": ["P1"],
            "Regimen_Code": ["X"],
            "Cycle_Number": [1],
            "Person_Stated_Gender_Code": ["M"],
            "Ethnic_Category_Code": ["A"],
            "Molecular_Marker_Status": ["BRCA"],
            "Biomarker_Panel_Code": ["P1"],
            "Gender_Identity_Code": ["M"],
            "Commissioning_Organisation_Code": ["7A2"],
        })
        assert auto_detect_version(df) == "v4.1"

    def test_fallback_to_v40_on_unknown(self):
        df = pd.DataFrame({"random_field": [1, 2, 3]})
        assert auto_detect_version(df) == "v4.0"

    def test_empty_df_returns_v40(self):
        assert auto_detect_version(pd.DataFrame()) == "v4.0"


# --------------------------------------------------------------------------- #
# 5. adapt() end-to-end
# --------------------------------------------------------------------------- #


class TestAdaptEndToEnd:
    def test_auto_detect_by_default(self, df_v41):
        out, event = adapt(df_v41)
        assert event.version == "v4.1"
        assert event.auto_detected is True
        assert event.rows_out == 3

    def test_explicit_version_overrides_autodetect(self, df_v41):
        # Force v4.0 on a v4.1 DataFrame
        out, event = adapt(df_v41, version="v4.0")
        assert event.version == "v4.0"
        assert event.auto_detected is False
        # v4.1 new fields should NOT be mapped under v4.0 rules
        assert all(out["Molecular_Marker_Status"] == "")

    def test_unknown_version_falls_back(self, df_v40):
        out, event = adapt(df_v40, version="v9.9")
        assert event.version == "v4.0"   # fell back

    def test_idempotent_on_canonical_input(self, df_v40):
        """Adapting twice should give the same canonical schema columns."""
        out1, _ = adapt(df_v40)
        out2, _ = adapt(out1, version="v4.0")
        assert list(out1.columns) == list(out2.columns)


# --------------------------------------------------------------------------- #
# 6. Registry + runtime registration
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_both_versions_registered(self):
        assert "v4.0" in ADAPTERS
        assert "v4.1" in ADAPTERS

    def test_specs_in_sync(self):
        assert set(ADAPTERS.keys()) == set(SPECS.keys())

    def test_register_new_adapter_runtime(self):
        class SACTv5Stub(SACTVersionAdapter):
            VERSION = "v5.0"
            def to_internal(self, raw_df):
                return SACTv4Adapter().to_internal(raw_df)

        new_spec = VersionSpec(
            version="v5.0", published="~2032 (projected)",
            expected_fields=("Genomic_Panel_Uuid",),
            new_since_prior=("Genomic_Panel_Uuid",),
            renamed_from_prior=(),
            canonical_map=dict(SPEC_V4_1.canonical_map),
        )
        register_adapter("v5.0", SACTv5Stub(), new_spec)
        assert "v5.0" in ADAPTERS
        # Cleanup
        del ADAPTERS["v5.0"]
        del SPECS["v5.0"]


# --------------------------------------------------------------------------- #
# 7. Pipeline (flask wrapper)
# --------------------------------------------------------------------------- #


class TestPipeline:
    def test_status_before_any_adapt(self, tmp_pipeline):
        s = tmp_pipeline.status()
        assert "v4.0" in s["registered_versions"]
        assert "v4.1" in s["registered_versions"]
        assert s["total_runs"] == 0

    def test_status_after_adapt(self, tmp_pipeline, df_v41):
        tmp_pipeline.adapt(df_v41)
        s = tmp_pipeline.status()
        assert s["total_runs"] == 1
        assert s["last_version"] == "v4.1"
        assert s["last_rows_in"] == 3

    def test_version_info(self, tmp_pipeline):
        info = tmp_pipeline.version_info()
        versions = {v["version"] for v in info}
        assert versions == {"v4.0", "v4.1"}
        v41 = next(v for v in info if v["version"] == "v4.1")
        assert "Molecular_Marker_Status" in v41["new_since_prior"]
        # renamed_from_prior should contain the gender rename
        renames = {(r["from"], r["to"]) for r in v41["renamed_from_prior"]}
        assert ("Person_Stated_Gender_Code", "Gender_Identity_Code") in renames


class TestSingleton:
    def test_get_set(self, tmp_path):
        p = SACTAdapterPipeline(storage_dir=tmp_path / "s")
        set_pipeline(p)
        assert get_pipeline() is p


# --------------------------------------------------------------------------- #
# 8. Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_event_log_written(self, tmp_pipeline, df_v40, tmp_path):
        tmp_pipeline.adapt(df_v40)
        log = tmp_path / "sact" / "adapter_events.jsonl"
        assert log.exists()
        rec = json.loads(log.read_text().strip().splitlines()[-1])
        assert rec["version"] == "v4.0"
        assert rec["rows_in"] == 3
        assert rec["rows_out"] == 3


# --------------------------------------------------------------------------- #
# 9. Coerce helper
# --------------------------------------------------------------------------- #


class TestCoerceBool:
    def test_truthy(self):
        for v in ["Y", "yes", "True", 1, True, "1"]:
            assert _coerce_bool_flag(v) is True

    def test_falsy(self):
        for v in ["N", "no", "False", 0, False, "0"]:
            assert _coerce_bool_flag(v) is False

    def test_missing(self):
        for v in [None, "", "nan", "NULL"]:
            assert pd.isna(_coerce_bool_flag(v))


# --------------------------------------------------------------------------- #
# 10. Edge cases (Wave 3.8)
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Error-path / degenerate-input coverage per Wave 3.8.  The adapter
    must never crash on real-world messy input; it should fall back
    defensively to v4.0 and always return the canonical schema."""

    def test_unknown_version_string_falls_back_to_v40(self, df_v40):
        """Passing version='v9.9' (unregistered) must not raise — the
        adapter logs a warning and returns the v4.0 mapping."""
        out, event = adapt(df_v40, version="v9.9")
        assert event.version == "v4.0"
        # Output still conforms to the canonical schema
        assert list(out.columns) == list(CANONICAL_COLUMNS)
        assert len(out) == len(df_v40)

    def test_malformed_empty_dataframe_returns_empty_canonical(self):
        """Empty DataFrame → empty canonical output, version v4.0,
        no rows in or out, no exceptions."""
        empty = pd.DataFrame()
        out, event = adapt(empty)
        assert len(out) == 0
        assert list(out.columns) == list(CANONICAL_COLUMNS)
        assert event.rows_in == 0
        assert event.rows_out == 0
        assert event.version == "v4.0"

    def test_missing_required_columns_still_produces_canonical(self):
        """Random-junk columns → falls back to v4.0 and records the
        missing required fields in event.columns_missing."""
        df = pd.DataFrame({"totally_unrelated_col": [1, 2, 3]})
        out, event = adapt(df)
        assert list(out.columns) == list(CANONICAL_COLUMNS)
        assert len(out) == 3
        # Required v4.0 fields are reported missing
        assert set(event.columns_missing) >= {
            "Patient_ID", "Regimen_Code", "Cycle_Number",
        }

    def test_mixed_v40_v41_columns_picks_v41(self):
        """When both v4.0 and v4.1 signature columns are present,
        auto-detect picks the higher (v4.1) version and applies its
        rename rules consistently."""
        mixed = pd.DataFrame({
            "Patient_ID": ["P1"],
            "Regimen_Code": ["X"],
            "Cycle_Number": [1],
            # Both naming conventions present
            "Person_Stated_Gender_Code": ["M"],
            "Gender_Identity_Code": ["F"],
            "Ethnic_Category_Code": ["A"],
            "Ethnic_Category": ["A"],
            "Molecular_Marker_Status": ["BRCA1+"],
            "Biomarker_Panel_Code": ["P001"],
            "Commissioning_Organisation_Code": ["7A2"],
        })
        assert auto_detect_version(mixed) == "v4.1"
        out, event = adapt(mixed)
        assert event.version == "v4.1"
        # v4.1 canonical map reads Gender_Identity_Code → Gender_Code,
        # so the output must reflect the v4.1 field, not the v4.0 duplicate.
        assert list(out["Gender_Code"]) == ["F"]
        assert list(out["Molecular_Marker_Status"]) == ["BRCA1+"]

    def test_idempotency_on_canonical_input(self, df_v40):
        """Adapting an already-canonical DataFrame twice must yield the
        same column set both times (round-trip stability)."""
        out1, _ = adapt(df_v40)
        out2, _ = adapt(out1, version="v4.0")
        out3, _ = adapt(out2, version="v4.0")
        assert list(out1.columns) == list(out2.columns) == list(out3.columns)
        # Row count preserved across repeated adapts
        assert len(out1) == len(out2) == len(out3) == len(df_v40)
